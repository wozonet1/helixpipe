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
from research_template import get_path, check_files_exist
import research_template as rt
from omegaconf import DictConfig
from pathlib import Path


# region d/l feature
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

    if mol.GetNumBonds() > 0:
        edges = []
        for bond in mol.GetBonds():
            edges.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            edges.append((bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()))  # Undirected

        # This creates a [num_bonds * 2, 2] tensor, then transposes to [2, num_bonds * 2]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        # If there are no bonds, create an empty edge_index of the correct shape [2, 0]
        edge_index = torch.empty((2, 0), dtype=torch.long)

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
class ProteinGCN(
    torch.nn.Module,
):
    def __init__(self):
        super(ProteinGCN, self).__init__()
        self.conv1 = GCNConv(1, 64)
        self.conv2 = GCNConv(64, 128)  # 20为氨基酸种类数

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
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


def _generate_and_save_labeled_set(
    positive_pairs: list,
    positive_pairs_normalized_set: set,
    dl2index: dict,
    prot2index: dict,
    config: DictConfig,
    output_path: Path,
):
    print(
        f"    -> Generating labeled set with {len(positive_pairs)} positive pairs for: {output_path.name}..."
    )

    negative_pairs = []
    all_molecule_ids = list(dl2index.values())
    all_protein_ids = list(prot2index.values())

    # Get the strategy from config
    sampling_strategy = config.params.negative_sampling_strategy

    with tqdm(
        total=len(positive_pairs),
        desc=f"Negative Sampling ({sampling_strategy})",
        leave=False,  # Set leave to False for cleaner logging inside a loop
    ) as pbar:
        if sampling_strategy == "popular":
            # --- Popularity-Biased Strategy ---
            mol_degrees = {}
            prot_degrees = {}
            # [MODIFICATION] Popularity is calculated based on ALL positive pairs for a stable distribution
            for (
                u,
                v,
            ) in positive_pairs_normalized_set:  # TODO: 目前使用全局popular负采样
                mol_degrees[u] = mol_degrees.get(u, 0) + 1
                prot_degrees[v] = prot_degrees.get(v, 0) + 1

            mol_weights = [mol_degrees.get(mol_id, 1) for mol_id in all_molecule_ids]
            prot_weights = [prot_degrees.get(prot_id, 1) for prot_id in all_protein_ids]

            while len(negative_pairs) < len(positive_pairs):
                dl_idx = random.choices(all_molecule_ids, weights=mol_weights, k=1)[0]
                p_idx = random.choices(all_protein_ids, weights=prot_weights, k=1)[0]
                normalized_candidate = tuple(sorted((dl_idx, p_idx)))

                # Check against the set of ALL positives to prevent generating a known interaction
                if normalized_candidate not in positive_pairs_normalized_set:
                    negative_pairs.append((dl_idx, p_idx))
                    pbar.update(1)

        elif sampling_strategy == "uniform":
            # --- Uniform Random Strategy ---
            while len(negative_pairs) < len(positive_pairs):
                dl_idx = random.choice(all_molecule_ids)
                p_idx = random.choice(all_protein_ids)
                normalized_candidate = tuple(sorted((dl_idx, p_idx)))

                if normalized_candidate not in positive_pairs_normalized_set:
                    negative_pairs.append((dl_idx, p_idx))
                    pbar.update(1)
        else:
            raise ValueError(
                f"Unknown negative_sampling_strategy: '{sampling_strategy}'"
            )

    print(f"        -> Generated {len(negative_pairs)} negative samples.")

    # --- [THIS PART IS ALSO MOVED FROM YOUR ORIGINAL CODE] ---
    # Save the unified, labeled edge file
    pos_df = pd.DataFrame(positive_pairs, columns=["source", "target"])
    pos_df["label"] = 1
    neg_df = pd.DataFrame(negative_pairs, columns=["source", "target"])
    neg_df["label"] = 0
    labeled_df = (
        pd.concat([pos_df, neg_df], ignore_index=True)
        .sample(frac=1)
        .reset_index(drop=True)
    )

    rt.ensure_path_exists(output_path)
    labeled_df.to_csv(output_path, index=False, header=True)


def process_data(config: DictConfig):
    # region init
    data_config = config["data"]
    params_config = config["params"]
    runtime_config = config["runtime"]
    device = runtime_config["gpu"]
    restart_flag = runtime_config.get("force_restart", False)

    full_df = pd.read_csv(get_path(config, "raw.dti_interactions"))
    # endregion

    # region list
    # ===================================================================
    # --- STAGE 1: Load, Merge, and Index Entities ---
    # ===================================================================

    # 1a. 加载基础数据集的实体
    print("--- [Stage 1a] Loading base entities from full_df.csv ---")
    checkpoint_files_dict = {
        "drug": "processed.indexes.drug",
        "ligand": "processed.indexes.ligand",
        "protein": "processed.indexes.protein",
        "nodes": "processed.nodes_metadata",
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
                print(
                    f"-> Found {len(extra_proteins_list)} unique proteins from GtoPdb."
                )

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
        final_proteins_list = sorted(
            list(set(base_proteins_list + extra_proteins_list))
        )
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
        pkl.dump(
            drug2index, open(get_path(config, checkpoint_files_dict["drug"]), "wb")
        )
        pkl.dump(
            ligand2index, open(get_path(config, checkpoint_files_dict["ligand"]), "wb")
        )
        pkl.dump(
            prot2index, open(get_path(config, checkpoint_files_dict["protein"]), "wb")
        )
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
    final_smiles_list = sorted(list(drug2index.keys())) + sorted(
        list(ligand2index.keys())
    )
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
        "molecule_similarity_matrix": "processed.similarity_matrices.molecule",
        "protein_similarity_matrix": "processed.similarity_matrices.protein",
        "node_features": "processed.node_features",
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
        for d in tqdm(
            final_smiles_list, desc="Extracting Molecule Features", leave=False
        ):
            embedding = extract_molecule_features(d, molecule_feature_extractor, device)
            drug_embeddings.append(embedding.cpu().detach().numpy())

        protein_embeddings = []
        for p in tqdm(
            final_proteins_list, desc="Extracting Protein Features", leave=False
        ):
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
        print(
            "\n--- [Stage 3] Loading features and similarity matrices from cache... ---"
        )
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

    # region positive edges

    # ===================================================================
    # --- STAGE 4: Generate Labeled Edges and Full Heterogeneous Graph ---
    # ===================================================================
    eval_config = config.training.evaluation
    split_mode = eval_config.mode
    seed = config.runtime.seed
    test_fraction = config.training.evaluation.test_fraction
    labels_template_key = "processed.link_prediction_labels_template"
    graph_template_key = "processed.typed_edge_list_template"
    train_labels_path = rt.get_path(
        config, labels_template_key, split_suffix=eval_config.train_file_suffix
    )
    test_labels_path = rt.get_path(
        config, labels_template_key, split_suffix=eval_config.test_file_suffix
    )
    print(f"--> Preparing and splitting positive pairs with mode: '{split_mode}'...")
    print(
        "--> Labeled train/test files not found or restart forced. Generating new splits..."
    )
    positive_pairs_normalized_set = set()

    # Filter the DataFrame for positive labels first
    positive_interactions_df = full_df[full_df["Y"] == 1].copy()

    # Pre-compute a mapping from raw to canonical SMILES to avoid re-computation
    raw_smiles_in_pos = positive_interactions_df["SMILES"].dropna().unique()
    canonical_map = {s: canonicalize_smiles(s) for s in raw_smiles_in_pos}
    positive_interactions_df["canonical_smiles"] = positive_interactions_df[
        "SMILES"
    ].map(canonical_map)

    for _, row in tqdm(
        positive_interactions_df.iterrows(),
        total=len(positive_interactions_df),
        desc="Processing Interaction Pairs",
    ):
        canonical_s, protein = row["canonical_smiles"], row["Protein"]
        if pd.notna(canonical_s) and canonical_s in dl2index and protein in prot2index:
            d_idx, p_idx = dl2index[canonical_s], prot2index[protein]
            normalized_pair = tuple(sorted((d_idx, p_idx)))
            positive_pairs_normalized_set.add(normalized_pair)

    # Scan interactions from GtoPdb if enabled
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
        ):
            smiles, sequence = row["SMILES"], row["Sequence"]
            if smiles in dl2index and sequence in prot2index:
                l_idx, p_idx = dl2index[smiles], prot2index[sequence]
                normalized_pair = tuple(sorted((l_idx, p_idx)))
                positive_pairs_normalized_set.add(normalized_pair)

    positive_pairs = list(positive_pairs_normalized_set)
    print(
        f"-> Found {len(positive_pairs)} unique, normalized positive pairs after cleaning."
    )
    from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

    num_folds = eval_config.k_folds
    # --- [新增逻辑] 根据 k_folds 的值选择不同的执行路径 ---
    if num_folds > 1:
        # --- 路径 A: K-Fold Cross-Validation (k > 1) ---
        print(f"--- [SETUP] Initializing {num_folds}-Fold Cross-Validation ---")
        # 1. 确定要进行K-Fold分割的对象 (实体ID 或 交互边)
        # -------------------------------------------------------------------
        entities_to_split = None
        if split_mode in ["drug", "protein"]:
            print(
                f"--> Preparing entities for COLD-START split on: {split_mode.upper()}"
            )
            if split_mode == "drug":
                entities_to_split = sorted(
                    list(set([p[0] for p in positive_pairs if p[0] < len(drug2index)]))
                )
            else:  # protein
                entities_to_split = sorted(list(set([p[1] for p in positive_pairs])))
            print(f"    - Found {len(entities_to_split)} unique entities to split.")

            # 对于实体级别的冷启动，我们通常使用标准的KFold
            kf = KFold(n_splits=eval_config.k_folds, shuffle=True, random_state=seed)
            split_iterator = kf.split(entities_to_split)

        elif split_mode == "random":
            print("--> Preparing edges for WARM-START split...")
            entities_to_split = positive_pairs
            # 对于warm-start，我们可以直接分割边。为了保持标签比例，使用StratifiedKFold更好
            # (需要一个虚拟的y标签来进行分层，这里我们用0和1交替)
            dummy_y = [i % 2 for i in range(len(entities_to_split))]
            skf = StratifiedKFold(
                n_splits=eval_config.k_folds, shuffle=True, random_state=seed
            )
            split_iterator = skf.split(entities_to_split, dummy_y)

        else:
            raise ValueError(f"Unknown split_mode '{split_mode}' in config.")
    else:
        print(
            f"--- [SETUP] Initializing Single Validation Split (k=1) with test_fraction={test_fraction} ---"
        )
        # 在这种模式下，我们不使用 KFold，split_iterator 将是一个只包含一次分割的列表
        split_iterator = [None]  # 创建一个只循环一次的迭代器
        entities_to_split = (
            positive_pairs  # 对于单次分割，我们总是基于所有边来创建训练/测试集
        )

    # 2. 遍历 K-Fold 的每一次分割，生成并保存对应的数据
    # -------------------------------------------------------------------
    # [关键] 这个循环将取代之前的所有分割逻辑
    for fold_idx, split_result in enumerate(split_iterator, 1):
        print(
            f"\n--- [K-FOLD] Generating data for Fold {fold_idx}/{eval_config.k_folds} ---"
        )

        train_positive_pairs, test_positive_pairs = [], []

        if num_folds > 1:
            # K-Fold 逻辑 (与之前相同)
            train_indices, test_indices = split_result
            if split_mode in ["drug", "protein"]:
                test_entity_ids = set([entities_to_split[i] for i in test_indices])
                entity_idx = 0 if split_mode == "drug" else 1
                train_positive_pairs = [
                    p for p in positive_pairs if p[entity_idx] not in test_entity_ids
                ]
                test_positive_pairs = [
                    p for p in positive_pairs if p[entity_idx] in test_entity_ids
                ]
            else:  # random
                train_positive_pairs = [entities_to_split[i] for i in train_indices]
                test_positive_pairs = [entities_to_split[i] for i in test_indices]
        else:
            # Single Split 逻辑 (使用 train_test_split)
            if split_mode in ["drug", "protein"]:
                # 冷启动的单次分割
                entity_list = sorted(
                    list(
                        set(
                            [
                                p[0] if split_mode == "drug" else p[1]
                                for p in positive_pairs
                            ]
                        )
                    )
                )
                train_entities, test_entities = train_test_split(
                    entity_list,
                    test_size=config.training.evaluation.test_fraction,
                    random_state=seed,
                )
                test_entity_ids = set(test_entities)
                entity_idx = 0 if split_mode == "drug" else 1
                train_positive_pairs = [
                    p for p in positive_pairs if p[entity_idx] not in test_entity_ids
                ]
                test_positive_pairs = [
                    p for p in positive_pairs if p[entity_idx] in test_entity_ids
                ]
            else:  # random
                # 热启动的单次分割
                train_positive_pairs, test_positive_pairs = train_test_split(
                    positive_pairs,
                    test_size=test_fraction,
                    random_state=seed,
                    stratify=[p[1] for p in positive_pairs],
                )

        print(
            f"    -> Split complete: {len(train_positive_pairs)} TRAIN positives, {len(test_positive_pairs)} TEST positives."
        )

        # 3. 为当前Fold生成并保存带标签的边集文件 (文件名包含fold_idx)
        # -------------------------------------------------------------------
        train_labels_path = rt.get_path(
            config,
            labels_template_key,
            split_suffix=f"_fold{fold_idx}{eval_config.train_file_suffix}",
        )
        _generate_and_save_labeled_set(
            positive_pairs=train_positive_pairs,
            positive_pairs_normalized_set=positive_pairs_normalized_set,
            dl2index=dl2index,
            prot2index=prot2index,
            config=config,
            output_path=train_labels_path,
        )

        test_labels_path = rt.get_path(
            config,
            labels_template_key,
            split_suffix=f"_fold{fold_idx}{eval_config.test_file_suffix}",
        )
        _generate_and_save_labeled_set(
            positive_pairs=test_positive_pairs,
            positive_pairs_normalized_set=positive_pairs_normalized_set,
            dl2index=dl2index,
            prot2index=prot2index,
            config=config,
            output_path=test_labels_path,
        )
        print(f"    -> Saved labeled edges for Fold {fold_idx} successfully.")
    # endregion

    # region sim edges
    num_folds = eval_config.k_folds
    for fold_idx in range(1, num_folds + 1):
        # In the future, we would re-split data for each fold here.
        # For now, we just use the single split we've already created.

        print(f"\n--- Processing Graph for Fold {fold_idx}/{num_folds} ---")
        graph_output_path = rt.get_path(
            config, graph_template_key, split_suffix=str(fold_idx)
        )

        relations_config = config.relations.flags
        typed_edges_list = []
        # coldsplit
        train_node_ids = set()
        if split_mode == "drug":
            train_drug_ids = set([p[0] for p in train_positive_pairs])
            train_node_ids = (
                set(prot2index.values())
                .union(set(ligand2index.values()))
                .union(train_drug_ids)
            )
        elif split_mode == "protein":
            train_protein_ids = set([p[1] for p in train_positive_pairs])
            train_node_ids = set(dl2index.values()).union(train_protein_ids)
        else:  # random
            train_node_ids = set(dl2index.values()).union(set(prot2index.values()))

        print(
            f"--> Constructing training graph with a total of {len(train_node_ids)} allowed nodes."
        )

        # --- Add Interaction Edges based on switches ---
        dp_added, lp_added = 0, 0
        # The `positive_pairs` list contains all unique (Molecule_ID, Protein_ID) tuples.
        for u, v in train_positive_pairs:
            # We need to re-check the type of the molecule node `u`.
            is_drug = u < len(drug2index)
            if is_drug and relations_config.get("dp_interaction", True):
                typed_edges_list.append([u, v, "drug_protein_interaction"])
                dp_added += 1
            elif not is_drug and relations_config.get("lp_interaction", True):
                typed_edges_list.append([u, v, "ligand_protein_interaction"])
                lp_added += 1
        print(
            f"--> Added {dp_added} DP and {lp_added} LP interactions based on config."
        )

        # --- Conditionally add PP similarity edges ---
        if relations_config.get("pp_similarity", True):
            print("--> Adding Protein-Protein similarity edges...")
            prot_start_index = len(drug2index) + len(ligand2index)
            p_sim_threshold = params_config["protein_similarity_threshold"]
            max_pp_edges_to_sample = params_config.get("max_pp_edges", -1)

            p_rows, p_cols = np.where(prot_similarity_matrix > p_sim_threshold)
            potential_edges = [(i, j) for i, j in zip(p_rows, p_cols) if i < j]

            print(
                f"    - Found {len(potential_edges)} potential P-P edges with similarity > {p_sim_threshold}."
            )

            final_pp_edges_indices = potential_edges
            if max_pp_edges_to_sample and 0 < max_pp_edges_to_sample < len(
                potential_edges
            ):
                print(
                    f"    - Sampling {max_pp_edges_to_sample} edges from the potential pool..."
                )
                sampled_indices = random.sample(
                    range(len(potential_edges)), max_pp_edges_to_sample
                )
                final_pp_edges_indices = [potential_edges[i] for i in sampled_indices]

            for i, j in final_pp_edges_indices:
                # Convert local protein indices to global IDs
                global_id_i = i + prot_start_index
                global_id_j = j + prot_start_index

                # [CRITICAL FILTER] Only add the edge if BOTH nodes are in the training set
                if global_id_i in train_node_ids and global_id_j in train_node_ids:
                    typed_edges_list.append(
                        [global_id_i, global_id_j, "protein_protein_similarity"]
                    )

            print(f"    - Added {len(final_pp_edges_indices)} P-P edges to the graph.")

        # --- Conditionally add Molecule similarity edges ---
        dd_added, dl_added, ll_added = 0, 0, 0
        if any(
            [
                relations_config.get(k)
                for k in ["dd_similarity", "dl_similarity", "ll_similarity"]
            ]
        ):
            print("--> Adding Molecule similarity edges...")
            mol_rows, mol_cols = np.where(
                dl_similarity_matrix > params_config["molecule_similarity_threshold"]
            )
            id_to_type_map = {idx: "drug" for idx in drug2index.values()}
            id_to_type_map.update({idx: "ligand" for idx in ligand2index.values()})

            for i, j in zip(mol_rows, mol_cols):
                if i < j and i in train_node_ids and j in train_node_ids:
                    type1, type2 = id_to_type_map[i], id_to_type_map[j]

                    if (
                        type1 == "drug"
                        and type2 == "drug"
                        and relations_config.get("dd_similarity", True)
                    ):
                        typed_edges_list.append([i, j, "drug_drug_similarity"])
                        dd_added += 1
                    elif (
                        type1 == "ligand"
                        and type2 == "ligand"
                        and relations_config.get("ll_similarity", True)
                    ):
                        typed_edges_list.append([i, j, "ligand_ligand_similarity"])
                        ll_added += 1
                    elif type1 != type2 and relations_config.get("dl_similarity", True):
                        u, v = (i, j) if type1 == "drug" else (j, i)
                        typed_edges_list.append([u, v, "drug_ligand_similarity"])
                        dl_added += 1
            print(
                f"    - Added {dd_added} D-D, {dl_added} D-L, and {ll_added} L-L edges based on config."
            )
        # endregion
        # --- Finalize and Save using the dynamic path ---
        print(f"\n--> Saving final graph structure to: {graph_output_path}")
        rt.ensure_path_exists(graph_output_path)

        typed_edges_df = pd.DataFrame(
            typed_edges_list, columns=["source", "target", "edge_type"]
        )
        typed_edges_df.to_csv(graph_output_path, index=False, header=True)

        print(f"-> Total edges in the final heterogeneous graph: {len(typed_edges_df)}")

    # This is the final print of the script
    print("\nData processing pipeline finished successfully!")
