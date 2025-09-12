import random
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pickle as pkl
from Bio import Align
from Bio.Align import substitution_matrices
from tqdm import tqdm
from joblib import Parallel, delayed
from pathlib import Path
from research_template import (
    get_path,
    check_files_exist,
    setup_dataset_directories,
)
import research_template as rt
import yaml

# TODO: 加强任务难度

# region d/l feature
# 将SMILES字符串转换为图数据


# --- FIXED VERSION of canonicalize_smiles ---
def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, canonical=True)
    else:
        print(f"Warning: Invalid SMILES string found and will be ignored: {smiles}")
        return None  # Return None for invalid SMILES


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILE string")

    # 提取原子特征
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(
            [
                atom.GetAtomicNum(),  # 原子序数
                atom.GetDegree(),  # 成键数
                atom.GetTotalNumHs(),  # 附着氢原子数量
                atom.GetFormalCharge(),  # 价态
                int(atom.GetIsAromatic()),  # 是否为芳环
            ]
        )

    atom_features = torch.tensor(atom_features, dtype=torch.float)

    # 构建邻接矩阵
    edges = []
    for bond in mol.GetBonds():
        edges.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        edges.append((bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()))  # 无向图

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # 转换为PyG数据格式
    x = atom_features
    data = Data(x=x, edge_index=edge_index)
    return data


class MoleculeGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


# 特征提取
def extract_molecule_features(smiles, model, device):
    graph_data = smiles_to_graph(smiles)
    # 在PyG的GCNConv中, x的形状需要是 [num_nodes, in_channels]
    x, edge_index = graph_data.x.to(device), graph_data.edge_index.to(device)
    node_embeddings = model(x, edge_index)  # 直接调用模型
    molecule_embedding = node_embeddings.mean(dim=0)
    return molecule_embedding


# endregion

# region p feature
# 氨基酸编码
aa_index = {
    "A": 0,
    "R": 1,
    "N": 2,
    "D": 3,
    "C": 4,
    "E": 5,
    "Q": 6,
    "G": 7,
    "H": 8,
    "I": 9,
    "L": 10,
    "K": 11,
    "M": 12,
    "F": 13,
    "P": 14,
    "S": 15,
    "T": 16,
    "W": 17,
    "Y": 18,
    "V": 19,
    "X": 20,
    "U": 21,
}


# 将氨基酸序列转换为图数据
def aa_sequence_to_graph(sequence):
    # 提取氨基酸特征
    node_features = torch.tensor(
        [aa_index[aa] for aa in sequence], dtype=torch.float
    ).unsqueeze(1)

    # 构建邻接矩阵
    edges = []
    for i in range(len(sequence) - 1):
        edges.append((i, i + 1))
        edges.append((i + 1, i))  # 无向图

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # 转换为PyG数据格式
    x = node_features
    data = Data(x=x, edge_index=edge_index)
    return data


# 定义GCN模型
class ProteinGCN(torch.nn.Module):
    def __init__(self):
        super(ProteinGCN, self).__init__()
        self.conv1 = GCNConv(1, 64)
        self.conv2 = GCNConv(64, 128)  # 20为氨基酸种类数

    def forward(self, data):
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# 特征提取
def extract_protein_features(sequence: str, model, device):
    graph_data = aa_sequence_to_graph(sequence)
    node_embeddings = model(graph_data.to(device))
    # 使用平均池化作为读出函数
    protein_embedding = node_embeddings.mean(dim=0)
    return protein_embedding


# endregion


def calculate_drug_similarity(drug_list):
    # 将SMILES转换为RDKit分子对象
    molecules = [Chem.MolFromSmiles(smiles) for smiles in drug_list]
    fpgen = GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = [fpgen.GetFingerprint(mol) for mol in molecules]

    num_molecules = len(molecules)
    similarity_matrix = np.zeros((num_molecules, num_molecules))

    # 2. 将外层循环用 tqdm 包裹起来
    # desc 参数为进度条提供了一个清晰的描述
    for i in tqdm(
        range(num_molecules), desc="Calculating Drug Similarities", leave=False
    ):
        # 由于矩阵是对称的，我们可以只计算上三角部分来优化性能
        for j in range(i, num_molecules):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                # 计算Tanimoto相似性
                similarity = AllChem.DataStructs.TanimotoSimilarity(
                    fingerprints[i], fingerprints[j]
                )
                # 因为矩阵是对称的，所以同时填充 (i, j) 和 (j, i)
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity

    return similarity_matrix


# region p sim
def align_pair(seq1: str, seq2: str, aligner_config: dict) -> float:
    # 在每个并行进程中重新创建一个 Aligner 对象
    try:
        aligner = Align.PairwiseAligner(**aligner_config)
        return aligner.score(seq1.upper(), seq2.upper())
    except Exception as e:
        print(f"Sequences: {seq1}\n{seq2}")  ##测试为什么用新的align算法会失败
        print(f"Error aligning sequences: {e}")
        return 0.0


def get_custom_blosum62_with_U():
    """
    加载标准的BLOSUM62矩阵,并增加对'U'(硒半胱氨酸)的支持。
    'U'的打分规则将完全模仿'C'(半胱氨酸)。
    """
    # 加载标准的BLOSUM62矩阵
    blosum62 = substitution_matrices.load("BLOSUM62")
    # 将其转换为Python字典以便修改
    custom_matrix_dict = dict(blosum62)
    # 获取所有标准氨基酸的字母 (包括 B, Z, X)
    old_alphabet = blosum62.alphabet
    # 为'U'增加打分规则
    for char in old_alphabet:
        # U-X 的得分 = C-X 的得分

        score = custom_matrix_dict.get(("C", char), custom_matrix_dict.get((char, "C")))
        if score is not None:
            custom_matrix_dict[("U", char)] = score
            custom_matrix_dict[(char, "U")] = score
    # U-U 的得分 = C-C 的得分
    custom_matrix_dict[("U", "U")] = custom_matrix_dict[("C", "C")]
    return substitution_matrices.Array(data=custom_matrix_dict)


def calculate_protein_similarity(sequence_list, cpus: int):
    # 定义 Aligner 的配置字典，以便传递给并行任务s
    aligner_config = {
        "mode": "local",
        "substitution_matrix": get_custom_blosum62_with_U(),
        "open_gap_score": -10,
        "extend_gap_score": -0.5,
    }
    num_sequences = len(sequence_list)
    print("--> Pre-calculating self-alignment scores for normalization...")
    self_scores = Parallel(n_jobs=cpus)(
        delayed(align_pair)(seq, seq, aligner_config)
        for seq in tqdm(
            sequence_list, desc="Self-Alignment Pre-computation", leave=False
        )
    )
    # 加上一个很小的数，防止未来出现除以零的错误 (例如空序列导致score为0)
    self_scores = np.array(self_scores, dtype=np.float32) + 1e-8

    # =========================================================================
    # 3. [两两比对阶段] 生成所有唯一的比对任务并并行计算原始分数
    # =========================================================================
    tasks = []
    for i in range(num_sequences):
        for j in range(i, num_sequences):
            # 我们只计算上三角部分，包括对角线
            tasks.append((i, j))

    print("--> Calculating pairwise raw alignment scores...")
    raw_pairwise_scores = Parallel(n_jobs=cpus)(
        delayed(align_pair)(sequence_list[i], sequence_list[j], aligner_config)
        for i, j in tqdm(tasks, desc="Pairwise Alignment", leave=False)
    )

    # =========================================================================
    # 4. [归一化与填充阶段] 使用预计算的分数进行归一化并构建矩阵
    # =========================================================================
    print("--> Populating similarity matrix with normalized scores...")
    similarity_matrix = np.zeros((num_sequences, num_sequences), dtype=np.float32)

    for (i, j), raw_score in zip(
        tqdm(
            tasks,
            desc="Normalizing and Filling Matrix",
            leave=False,
        ),
        raw_pairwise_scores,
    ):
        if i == j:
            similarity_matrix[i, j] = 1.0
        else:
            # 计算归一化的分母
            denominator = np.sqrt(self_scores[i] * self_scores[j])

            # 应用归一化公式
            normalized_score = raw_score / denominator

            # 使用 np.clip 确保分数严格落在 [0, 1] 区间，处理浮点数精度问题
            final_score = np.clip(normalized_score, 0, 1)

            # 因为矩阵是对称的，所以同时填充 (i, j) 和 (j, i)
            similarity_matrix[i, j] = final_score
            similarity_matrix[j, i] = final_score

    return similarity_matrix


# endregion


# region init
def load_config(config_path="config.yaml"):
    """Loads the YAML config file."""
    # We assume the script is run from the `src` directory, so config is `../config.yaml`
    # A more robust way is to determine the project root, but this works for now.
    project_root = Path(__file__).parent.parent
    with open(project_root / config_path, "r") as f:
        return yaml.safe_load(f)


config = load_config()
data_config = config.get("data", {})
params_config = config.get("params", {})
runtime_config = config.get("runtime", {})
device = runtime_config["gpu"]
primary_dataset = data_config["primary_dataset"]
restart_flag = runtime_config["force_restart"]
# set random seed
random.seed(runtime_config["seed"])
np.random.seed(runtime_config["seed"])
torch.manual_seed(runtime_config["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed(runtime_config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# make dir
setup_dataset_directories(config)
full_df = pd.read_csv(get_path(config, f"{primary_dataset}.raw.dti_interactions"))
# endregion

# region list
# ===================================================================
# --- STAGE 1: Load, Merge, and Index Entities ---
# ===================================================================

# 1a. 加载基础数据集的实体
print("--- [Stage 1a] Loading base entities from full_df.csv ---")
checkpoint_files_dict = {
    "drug": f"{primary_dataset}.processed.indexes.drug",
    "ligand": f"{primary_dataset}.processed.indexes.ligand",
    "protein": f"{primary_dataset}.processed.indexes.protein",
    "nodes": f"{primary_dataset}.processed.nodes_metadata",
}
if not check_files_exist(config, *checkpoint_files_dict.values()) or restart_flag:
    # 使用 .unique().tolist() 更高效
    unique_smiles = full_df["SMILES"].dropna().unique()
    base_drugs_list = [
        canonicalize_smiles(s)
        for s in tqdm(unique_smiles, desc="Canonicalizing Drug SMILES", leave=False)
    ]
    base_drugs_list = [s for s in base_drugs_list if s is not None]
    base_proteins_list = full_df["Protein"].unique().tolist()

    # 1b. 根据开关，加载GtoPdb的“扩展包”实体
    extra_ligands_list = []
    extra_proteins_list = []
    if data_config["use_gtopdb"]:
        print(
            "\n--- [Stage 1b] GtoPdb integration ENABLED. Loading extra entities... ---"
        )
        try:
            gtopdb_ligands_df = pd.read_csv(
                get_path(config, "gtopdb.processed.ligands"),
                header=None,
                names=["CID", "SMILES"],
            )
            extra_ligands_list = gtopdb_ligands_df["SMILES"].unique().tolist()
            print(f"-> Found {len(extra_ligands_list)} unique ligands from GtoPdb.")

            gtopdb_proteins_df = pd.read_csv(
                get_path(config, "gtopdb.processed.proteins"),
                header=None,
                names=["UniProt", "Sequence"],
            )
            extra_proteins_list = gtopdb_proteins_df["Sequence"].unique().tolist()
            print(f"-> Found {len(extra_proteins_list)} unique proteins from GtoPdb.")

        except FileNotFoundError:
            print(
                "Warning: Processed GtoPdb files not found. Continuing without GtoPdb data."
            )
    else:
        print("\n--- [Stage 1b] GtoPdb integration DISABLED. ---")
    print(
        f"DEBUG_1A: Initial unique drugs = {len(set(base_drugs_list))}, proteins = {len(set(base_proteins_list))}"
    )
    if data_config["use_gtopdb"]:
        print(
            f"DEBUG_1B: Initial unique extra ligands = {len(set(extra_ligands_list))}, extra proteins = {len(set(extra_proteins_list))}"
        )
    # endregion

    # region index
    # In data_proc.py, replace the entire "region index"

    # region index
    print("\n--- [Stage 2] Creating and saving index files... ---")

    # 1. 首先，获取所有 drug 和 ligand 的唯一SMILES集合
    unique_drug_smiles = set(base_drugs_list)
    unique_ligand_smiles = set(extra_ligands_list)

    # 2. [核心修复] 对所有分子进行统一规划和去重
    # 我们定义一个优先级规则：如果一个分子同时是drug和ligand，我们视其为drug。
    # 首先确定纯粹的ligand（只在ligand列表里出现）
    pure_ligand_smiles = unique_ligand_smiles - unique_drug_smiles

    # 所有的drug SMILES（包括那些也可能是ligand的）
    all_drug_smiles = unique_drug_smiles

    # 3. 分别对两组SMILES列表进行排序，以保证处理顺序的确定性
    sorted_unique_drugs = sorted(list(all_drug_smiles))
    sorted_unique_ligands = sorted(list(pure_ligand_smiles))
    # 2. 初始化索引字典和节点元数据列表
    drug2index = {}
    ligand2index = {}
    node_data = []
    current_id = 0

    for smile in sorted_unique_drugs:
        drug2index[smile] = current_id
        node_data.append({"node_id": current_id, "node_type": "drug"})
        current_id += 1

    for smile in sorted_unique_ligands:
        ligand2index[smile] = current_id
        node_data.append({"node_id": current_id, "node_type": "ligand"})
        current_id += 1

    # --- 统一处理所有蛋白质 (这部分逻辑不变) ---
    final_proteins_list = sorted(list(set(base_proteins_list + extra_proteins_list)))
    protein_start_index = current_id  # ID从最后一个小分子之后开始
    prot2index = {
        seq: i + protein_start_index for i, seq in enumerate(final_proteins_list)
    }
    for seq, idx in prot2index.items():
        node_data.append({"node_id": idx, "node_type": "protein"})
    print(
        f"DEBUG_2: Total indexed entities = drug({len(drug2index)}) + ligand({len(ligand2index)}) + protein({len(prot2index)}) = {len(drug2index) + len(ligand2index) + len(prot2index)}"
    )
    print(f"DEBUG_2: Total rows in node_data list = {len(node_data)}")
    # endregion index

    # --- 保存所有文件 ---
    pkl.dump(drug2index, open(get_path(config, checkpoint_files_dict["drug"]), "wb"))
    pkl.dump(
        ligand2index, open(get_path(config, checkpoint_files_dict["ligand"]), "wb")
    )
    pkl.dump(prot2index, open(get_path(config, checkpoint_files_dict["protein"]), "wb"))
    # 6. 保存节点元数据文件
    AllNode_df = pd.DataFrame(node_data)  # node_data已经包含了所有类型
    AllNode_df.to_csv(
        get_path(config, checkpoint_files_dict["nodes"]), index=False, header=True
    )
    print("-> Index and metadata files saved successfully.")

else:
    print("\n--- [Stage 2] Loading indices and metadata from cache... ---")
    # Reuse the list defined above. No more duplication.
    drug_idx_path = get_path(config, checkpoint_files_dict["drug"])
    ligand_idx_path = get_path(config, checkpoint_files_dict["ligand"])
    prot_idx_path = get_path(config, checkpoint_files_dict["protein"])

    drug2index = pkl.load(open(drug_idx_path, "rb"))
    ligand2index = pkl.load(open(ligand_idx_path, "rb"))
    prot2index = pkl.load(open(prot_idx_path, "rb"))

# --- 为后续步骤准备最终的、完整的实体列表 ---
# 这些列表现在是从索引字典的键中动态生成的，而不是作为中间变量传来传去
final_smiles_list = sorted(list(drug2index.keys())) + sorted(list(ligand2index.keys()))
final_proteins_list = sorted(list(prot2index.keys()))
dl2index = {**drug2index, **ligand2index}  # 统一的小分子索引字典
print(f"DEBUG_3: Length of final_smiles_list = {len(final_smiles_list)}")
print(f"DEBUG_3: Length of final_proteins_list = {len(final_proteins_list)}")
print(
    f"DEBUG_3: Total nodes based on final_lists = {len(final_smiles_list) + len(final_proteins_list)}"
)
# endregion

# ===================================================================
# --- STAGE 3: Generate Features, Similarity Matrices, and Edges ---
# ===================================================================

# region features&sim

checkpoint_files_dict = {
    "molecule_similarity_matrix": f"{primary_dataset}.processed.similarity_matrices.molecule",
    "protein_similarity_matrix": f"{primary_dataset}.processed.similarity_matrices.protein",
    "node_features": f"{primary_dataset}.processed.node_features",
}
if not check_files_exist(config, *checkpoint_files_dict.values()) or restart_flag:
    print("\n--- [Stage 3] Generating features and similarity matrices... ---")
    print("--> Initializing feature extraction models...")
    # Using your GCNLayer, but a standard GCNConv would be better.
    # .eval() is crucial: it disables dropout and other training-specific layers.
    molecule_feature_extractor = MoleculeGCN(5, 128).to(device).eval()
    protein_feature_extractor = ProteinGCN().to(device).eval()

    drug_embeddings = []
    # Use the new function signature in the loop
    for d in tqdm(final_smiles_list, desc="Extracting Molecule Features", leave=False):
        embedding = extract_molecule_features(d, molecule_feature_extractor, device)
        drug_embeddings.append(embedding.cpu().detach().numpy())

    protein_embeddings = []
    for p in tqdm(final_proteins_list, desc="Extracting Protein Features", leave=False):
        embedding = extract_protein_features(p, protein_feature_extractor, device)
        protein_embeddings.append(embedding.cpu().detach().numpy())

    # prot_similarity_matrix = cosine_similarity(protein_embeddings)
    dl_similarity_matrix = calculate_drug_similarity(final_smiles_list)
    # notice:用U与C一样的方式计算相似度
    prot_similarity_matrix = calculate_protein_similarity(
        final_proteins_list, runtime_config["cpus"]
    )

    pkl.dump(
        dl_similarity_matrix,
        open(
            get_path(config, checkpoint_files_dict["molecule_similarity_matrix"]),
            "wb",
        ),
    )
    pkl.dump(
        prot_similarity_matrix,
        open(
            get_path(config, checkpoint_files_dict["protein_similarity_matrix"]),
            "wb",
        ),
    )
    features_df = pd.concat(
        [pd.DataFrame(drug_embeddings), pd.DataFrame(protein_embeddings)], axis=0
    )
    features_df.to_csv(
        get_path(config, checkpoint_files_dict["node_features"]),
        index=False,
        header=False,
    )
    print(f"DEBUG_4: Shape of final features_df = {features_df.shape}")
    # --- The most critical assertion ---
    num_nodes_from_stage2 = len(drug2index) + len(ligand2index) + len(prot2index)
    assert features_df.shape[0] == num_nodes_from_stage2, (
        f"FATAL: Feature matrix length ({features_df.shape[0]}) does not match indexed node count ({num_nodes_from_stage2})!"
    )
else:
    # 如果文件已存在，则加载它们以供后续步骤使用
    print("\n--- [Stage 3] Loading features and similarity matrices from cache... ---")
    dl_similarity_matrix = pkl.load(
        open(
            get_path(config, checkpoint_files_dict["molecule_similarity_matrix"]),
            "rb",
        )
    )
    prot_similarity_matrix = pkl.load(
        open(
            get_path(config, checkpoint_files_dict["protein_similarity_matrix"]),
            "rb",
        )
    )
    features_df = pd.read_csv(
        get_path(config, checkpoint_files_dict["node_features"]), header=None
    )


# ===================================================================
# --- STAGE 4: Generate Labeled Edges and Full Heterogeneous Graph ---
# ===================================================================

# region base edges
# 统一的检查点，检查最终的两个核心产物
checkpoint_files_dict = {
    "typed_edge": f"{primary_dataset}.processed.typed_edge_list",
    "link_labels": f"{primary_dataset}.processed.link_prediction_labels",
}
if not check_files_exist(config, *checkpoint_files_dict.values()) or restart_flag:
    print(
        "\n--- [Stage 4] Generating labeled edges for training and full typed graph... ---"
    )

    # --- 4a. 准备链接预测任务的正负样本 ---
    print("\n-> Generating positive and negative samples for link prediction...")

    # [NEW] Use a set to store NORMALIZED positive pairs to automatically handle duplicates and symmetry from the source file.
    positive_pairs_normalized_set = set()

    # 从基础数据集 (DrugBank) 中提取 D-P 正样本
    print("--> Scanning positive interactions from DrugBank...")
    for _, row in tqdm(
        full_df[full_df["Y"] == 1].iterrows(),
        total=full_df[full_df["Y"] == 1].shape[0],
        desc="DrugBank Pairs",
        leave=False,
    ):
        smiles = row["SMILES"]
        protein = row["Protein"]
        if smiles in dl2index and protein in prot2index:
            d_idx = dl2index[smiles]
            p_idx = prot2index[protein]
            # [CORE FIX] Normalize the pair by sorting the IDs before adding to the set
            normalized_pair = tuple(sorted((d_idx, p_idx)))
            positive_pairs_normalized_set.add(normalized_pair)

    # (可选) 从GtoPdb中提取 L-P 正样本
    if data_config["use_gtopdb"]:
        print("--> Scanning positive interactions from GtoPdb...")
        gtopdb_edges_df = pd.read_csv(
            rt.get_path(config, "gtopdb.processed.interactions"),
            header=None,
            names=["Sequence", "SMILES", "Affinity"],
        )
        for _, row in tqdm(
            gtopdb_edges_df.iterrows(),
            total=len(gtopdb_edges_df),
            desc="GtoPdb Pairs",
            leave=False,
        ):
            smiles = row["SMILES"]
            sequence = row["Sequence"]
            if smiles in dl2index and sequence in prot2index:
                l_idx = dl2index[smiles]
                p_idx = prot2index[sequence]
                # [CORE FIX] Normalize the pair here as well
                normalized_pair = tuple(sorted((l_idx, p_idx)))
                positive_pairs_normalized_set.add(normalized_pair)

    # Convert the clean, unique, normalized set back to a list for further processing
    positive_pairs = list(positive_pairs_normalized_set)
    print(
        f"-> Found {len(positive_pairs)} unique, normalized positive pairs after cleaning."
    )

    # --- 进行随机负采样 ---
    # The logic here is now safer because the positive set is clean.
    print("--> Performing negative sampling...")
    negative_pairs = []
    all_molecule_ids = list(dl2index.values())
    all_protein_ids = list(prot2index.values())

    # The set of positive pairs is already normalized, so we can use it directly for checking
    # positive_set_for_neg_sampling = {tuple(sorted(p)) for p in positive_pairs} # This is redundant now

    with tqdm(total=len(positive_pairs), desc="Negative Sampling", leave=False) as pbar:
        while len(negative_pairs) < len(positive_pairs):
            dl_idx = random.choice(all_molecule_ids)
            p_idx = random.choice(all_protein_ids)

            # Normalize the candidate pair before checking for existence
            normalized_candidate = tuple(sorted((dl_idx, p_idx)))

            if normalized_candidate not in positive_pairs_normalized_set:
                # We store the original (potentially directional) pair for the negative set
                negative_pairs.append((dl_idx, p_idx))
                pbar.update(1)

    print(
        f"-> Generated {len(positive_pairs)} positive and {len(negative_pairs)} negative samples."
    )

    # --- 保存统一的、带标签的边文件 ---
    # Note: For positive pairs, we are now saving the normalized (min_id, max_id) version.
    # This ensures consistency.
    pos_df = pd.DataFrame(positive_pairs, columns=["source", "target"])
    pos_df["label"] = 1
    neg_df = pd.DataFrame(negative_pairs, columns=["source", "target"])
    neg_df["label"] = 0

    dl_p_edges_df = pd.concat([pos_df, neg_df], ignore_index=True)
    dl_p_edges_df.to_csv(
        rt.get_path(
            config, checkpoint_files_dict["link_labels"]
        ),  # Assuming graph_files_dict is defined
        index=False,
        header=True,
    )
    # end region

    # region sim edges
    # --- 4b. 构建完整的、带类型的异构图边列表 ---
    print("\n-> Assembling full heterogeneous graph with typed edges...")
    typed_edges_list = []

    # 1. 添加所有正样本 D-P 和 L-P 边
    for d_idx, p_idx in positive_pairs:
        # 通过ID范围判断源节点是drug还是ligand
        if d_idx < len(drug2index):
            typed_edges_list.append([d_idx, p_idx, "drug_protein_interaction"])
        else:
            typed_edges_list.append([d_idx, p_idx, "ligand_protein_interaction"])

    # 2. 添加 P-P 相似性边
    print("\n-> Processing Protein-Protein similarity edges...")
    prot_start_index = len(drug2index) + len(ligand2index)

    # [MODIFIED] Get threshold and sampling config
    p_sim_threshold = params_config["protein_similarity_threshold"]
    max_pp_edges_to_sample = params_config.get(
        "max_pp_edges", -1
    )  # Use .get for safety

    # Step 1: Find ALL potential edges above the threshold
    p_rows, p_cols = np.where(prot_similarity_matrix > p_sim_threshold)
    potential_edges = []
    for i, j in zip(p_rows, p_cols):
        if i < j:
            # We store the score as well, in case we want to sample top-K in the future
            score = prot_similarity_matrix[i, j]
            potential_edges.append(((i, j), score))

    print(
        f"-> Found {len(potential_edges)} potential P-P edges with similarity > {p_sim_threshold}."
    )

    # Step 2: [NEW] Apply sampling if configured
    final_pp_edges_indices = [
        edge for edge, score in potential_edges
    ]  # Default to all edges

    # Check if max_pp_edges_to_sample is a valid positive number
    if max_pp_edges_to_sample and max_pp_edges_to_sample > 0:
        if len(potential_edges) > max_pp_edges_to_sample:
            print(
                f"-> Sampling {max_pp_edges_to_sample} edges from the potential pool..."
            )
            # Simple random sampling
            sampled_indices = random.sample(
                range(len(potential_edges)), max_pp_edges_to_sample
            )
            final_pp_edges_indices = [potential_edges[i][0] for i in sampled_indices]
        else:
            print(
                "-> Number of potential edges is less than or equal to the sampling limit. Using all."
            )

    # Step 3: Add the final list of edges to the graph
    pp_count = 0
    for i, j in final_pp_edges_indices:
        typed_edges_list.append(
            [
                i + prot_start_index,
                j + prot_start_index,
                "protein_protein_similarity",
            ]
        )
        pp_count += 1
    print(f"-> Added {pp_count} Protein-Protein edges to the graph.")

    # 3. 添加 D-D / L-L / D-L 相似性边
    mol_rows, mol_cols = np.where(
        dl_similarity_matrix > params_config["molecule_similarity_threshold"]
    )
    dd_count, ll_count, dl_count = 0, 0, 0
    # 我们需要一个快速的方法来查找ID的类型
    id_to_type_map = {idx: "drug" for idx in drug2index.values()}
    id_to_type_map.update({idx: "ligand" for idx in ligand2index.values()})

    for i, j in zip(mol_rows, mol_cols):
        if i < j:
            type1 = id_to_type_map[i]
            type2 = id_to_type_map[j]
            if type1 == "drug" and type2 == "drug":
                edge_type = "drug_drug_similarity"
                dd_count += 1
            elif type1 == "ligand" and type2 == "ligand":
                edge_type = "ligand_ligand_similarity"
                ll_count += 1
            else:  # one is drug, one is ligand
                edge_type = "drug_ligand_similarity"
                dl_count += 1
            typed_edges_list.append([i, j, edge_type])

    print(
        f"-> Added {dd_count} Drug-Drug, {ll_count} Ligand-Ligand, and {dl_count} Drug-Ligand edges."
    )

    # 保存统一的、带类型的总边文件
    typed_edges_df = pd.DataFrame(
        typed_edges_list, columns=["source", "target", "edge_type"]
    )
    typed_edges_df.to_csv(
        get_path(config, checkpoint_files_dict["typed_edge"]),
        index=False,
        header=True,
    )
    print(f"-> Total edges in heterogeneous graph: {len(typed_edges_df)}")

else:
    print("\n--- [Stage 4] Final graph files already exist. Skipping. ---")

# endregion (final_graph_construction)

print("\nData processing pipeline finished successfully!")
