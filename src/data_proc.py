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
from utils import get_data_filepath, check_files_exist
import yaml


# region d/l feature
# 将SMILES字符串转换为图数据
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


# 定义GCN层
class GCNLayer(torch.nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = torch.nn.Linear(in_feats * 2, out_feats)

    def forward(self, x, edge_index):
        # 聚合邻居节点特征
        agg = torch.zeros_like(x)
        if edge_index.numel() == 0:
            print("ZERO")
            return torch.zeros([20, 128]).to(device)
        agg[edge_index[0]] = x[edge_index[1]]
        out = self.linear(torch.cat((x, agg), dim=-1))
        return out


# 特征提取
def extract_features(smiles):
    graph_data = smiles_to_graph(smiles)
    gcn_layer = GCNLayer(in_feats=5, out_feats=128).to(device)  # 5个原子特征到128维嵌入
    node_embeddings = gcn_layer(
        graph_data.x.to(device), graph_data.edge_index.to(device)
    )
    # 使用平均池化作为读出函数
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
def extract_aa_features(sequence):
    graph_data = aa_sequence_to_graph(sequence)
    model = ProteinGCN().to(device)
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
    for i in tqdm(range(num_molecules), desc="Calculating Drug Similarities"):
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


def calculate_protein_similarity(sequence_list):
    # 定义 Aligner 的配置字典，以便传递给并行任务s
    aligner_config = {
        "mode": "local",
        "substitution_matrix": get_custom_blosum62_with_U(),
        "open_gap_score": -10,
        "extend_gap_score": -0.5,
    }
    num_sequences = len(sequence_list)
    similarity_matrix = np.zeros((num_sequences, num_sequences))
    tasks = []

    for i in range(num_sequences):
        for j in range(i, num_sequences):
            tasks.append((i, j))

    # 4. 使用 Parallel 和 delayed 来执行并行计算
    results = Parallel(n_jobs=runtime_config["cpus"])(
        delayed(align_pair)(sequence_list[i], sequence_list[j], aligner_config)
        for i, j in tqdm(tasks, desc="Calculating Similarities (Multi-CPU)")
    )

    # 5. 将计算结果填充回矩阵
    similarity_matrix = np.zeros((num_sequences, num_sequences))
    for (i, j), score in zip(tasks, results):
        if i == j:
            similarity_matrix[i, j] = 1.0  # 理论上应该是序列长度，但这里按原逻辑设为1
        else:
            similarity_matrix[i, j] = score
            similarity_matrix[j, i] = score

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
# set random seed
random.seed(runtime_config["seed"])
np.random.seed(runtime_config["seed"])
torch.manual_seed(runtime_config["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed(runtime_config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# get paths
current_path = Path(__file__)
datapath = current_path.parent.parent / f"data/{data_config['dataset_name']}"
foldername = "gtopdb" if data_config["use_gtopdb"] else "baseline"
gtopdb_processed_dir = datapath.parent / "gtopdb/processed"
# create dirs if not exist
if not (datapath / foldername).exists():
    (datapath / foldername).mkdir(parents=True, exist_ok=True)
    (datapath / foldername / "indexes").mkdir(parents=True, exist_ok=True)

full_df = pd.read_csv(datapath / "full.csv")
# endregion

# region list
# ===================================================================
# --- STAGE 1: Load, Merge, and Index Entities ---
# ===================================================================

# 1a. 加载基础数据集的实体
print("--- [Stage 1a] Loading base entities from full_df.csv ---")
required_index_files = [
    "indexes.drug",
    "indexes.ligand",
    "indexes.protein",
    "nodes_metadata",
]
if not check_files_exist(config, *required_index_files):
    # 使用 .unique().tolist() 更高效
    base_drugs_list = full_df["SMILES"].unique().tolist()
    base_proteins_list = full_df["Protein"].unique().tolist()
    print(
        f"-> Found {len(base_drugs_list)} unique drugs and {len(base_proteins_list)} unique proteins in base dataset."
    )

    # 1b. 根据开关，加载GtoPdb的“扩展包”实体
    extra_ligands_list = []
    extra_proteins_list = []
    if data_config["use_gtopdb"]:
        print(
            "\n--- [Stage 1b] GtoPdb integration ENABLED. Loading extra entities... ---"
        )
        try:
            gtopdb_ligands_df = pd.read_csv(
                gtopdb_processed_dir / "gtopdb_ligands.csv",
                header=None,
                names=["CID", "SMILES"],
            )
            extra_ligands_list = gtopdb_ligands_df["SMILES"].unique().tolist()
            print(f"-> Found {len(extra_ligands_list)} unique ligands from GtoPdb.")

            gtopdb_proteins_df = pd.read_csv(
                gtopdb_processed_dir / "gtopdb_proteins.csv",
                header=None,
                names=["UniProt", "Sequence"],
            )
            extra_proteins_list = gtopdb_proteins_df["Sequence"].unique().tolist()
            print(f"-> Found {len(extra_proteins_list)} unique proteins from GtoPdb.")

        except FileNotFoundError:
            print(
                "Warning: Processed GtoPdb files not found. Continuing without GtoPdb data."
            )
            data_config["use_gtopdb"] = False  # 将开关关掉，确保后续逻辑正确
    else:
        print("\n--- [Stage 1b] GtoPdb integration DISABLED. ---")
    # endregion

    # region index
    print("\n--- [Stage 2] Creating and saving index files... ---")
    ligands_map = {smile: "ligand" for smile in extra_ligands_list}
    drugs_map = {smile: "drug" for smile in base_drugs_list}
    all_molecules_map = {**ligands_map, **drugs_map}

    # 2. 直接遍历这个map，一次性创建索引和节点元数据
    drug2index = {}
    ligand2index = {}
    node_data = []

    # 我们需要一个确定的顺序，所以先对SMILES排序
    sorted_smiles = sorted(all_molecules_map.keys())

    # 为所有小分子分配连续的ID (0 to N_mol-1)
    for idx, smile in enumerate(sorted_smiles):
        node_type = all_molecules_map[smile]
        node_data.append({"node_id": idx, "node_type": node_type})
        if node_type == "drug":
            drug2index[smile] = idx
        else:  # node_type == 'ligand'
            ligand2index[smile] = idx

    print(f"-> Indexed {len(drug2index)} drugs and {len(ligand2index)} ligands.")

    # --- 统一处理所有蛋白质 ---

    # 3. 对蛋白质进行去重和排序
    # --- 蛋白质处理 ---
    final_proteins_list = sorted(list(set(base_proteins_list + extra_proteins_list)))
    print(
        f"-> Total unique proteins: {len(final_proteins_list)}, \
        base proteins: {len(base_proteins_list)}, \
        added {len(final_proteins_list) - len(base_proteins_list)} from GtoPdb."
    )

    # 4. 为所有蛋白质分配连续的、偏移后的ID
    protein_start_index = len(node_data)  # ID从最后一个小分子之后开始
    prot2index = {
        seq: i + protein_start_index for i, seq in enumerate(final_proteins_list)
    }

    for seq, idx in prot2index.items():
        node_data.append({"node_id": idx, "node_type": "protein"})

    print(f"-> Indexed {len(prot2index)} total proteins.")

    # --- 保存所有文件 ---

    # 5. 保存索引文件
    pkl.dump(
        drug2index,
        open(get_data_filepath(config, "indexes.drug"), "wb"),
    )
    pkl.dump(
        ligand2index,
        open(get_data_filepath(config, "indexes.ligand"), "wb"),
    )
    pkl.dump(
        prot2index,
        open(get_data_filepath(config, "indexes.protein"), "wb"),
    )

    # 6. 保存节点元数据文件
    AllNode_df = pd.DataFrame(node_data)  # node_data已经包含了所有类型
    AllNode_df.to_csv(
        get_data_filepath(config, "nodes_metadata"),
        index=False,
        header=True,
    )
    print("-> Index and metadata files saved successfully.")

else:
    print("\n--- [Stage 2] Loading indices and metadata from cache... ---")
    index_files = {
        "drug": "indexes.drug",
        "ligand": "indexes.ligand",
        "prot": "indexes.protein",
    }
    # 使用字典推导式一次性加载
    indices = {
        key: pkl.load(get_data_filepath(config, path).open("rb"))
        for key, path in index_files.items()
    }
    drug2index, ligand2index, prot2index = (
        indices["drug"],
        indices["ligand"],
        indices["prot"],
    )
# --- 为后续步骤准备最终的、完整的实体列表 ---
# 这些列表现在是从索引字典的键中动态生成的，而不是作为中间变量传来传去
final_smiles_list = sorted(list(drug2index.keys()) + list(ligand2index.keys()))
final_proteins_list = sorted(list(prot2index.keys()))
dl2index = {**drug2index, **ligand2index}  # 统一的小分子索引字典
# endregion

# ===================================================================
# --- STAGE 3: Generate Features, Similarity Matrices, and Edges ---
# ===================================================================

# region features&sim
checkpoint_files = [
    "molecule_similarity_matrix",
    "protein_similarity_matrix",
    "node_features",
]
if not check_files_exist(config, *checkpoint_files):
    print("\n--- [Stage 3] Generating features and similarity matrices... ---")
    drug_embeddings = []
    for i, d in enumerate(final_smiles_list):
        drug_embeddings.append(extract_features(d).cpu().detach().numpy())

    protein_embeddings = []
    for i, p in enumerate(final_proteins_list):
        protein_embeddings.append(extract_aa_features(p).cpu().detach().numpy())

    # prot_similarity_matrix = cosine_similarity(protein_embeddings)
    dl_similarity_matrix = calculate_drug_similarity(final_smiles_list)
    # notice:用U与C一样的方式计算相似度
    prot_similarity_matrix = calculate_protein_similarity(final_proteins_list)

    pkl.dump(
        dl_similarity_matrix,
        open(get_data_filepath(config, "molecule_similarity_matrix"), "wb"),
    )
    pkl.dump(
        prot_similarity_matrix,
        open(get_data_filepath(config, "protein_similarity_matrix"), "wb"),
    )
    features_df = pd.concat(
        [pd.DataFrame(drug_embeddings), pd.DataFrame(protein_embeddings)], axis=0
    )
    features_df.to_csv(
        get_data_filepath(config, "node_features"),
        index=False,
        header=False,
    )
else:
    # 如果文件已存在，则加载它们以供后续步骤使用
    print("\n--- [Stage 3] Loading features and similarity matrices from cache... ---")
    features_df = pd.read_csv(get_data_filepath(config, "node_features"), header=None)
    dl_similarity_matrix = pkl.load(
        open(get_data_filepath(config, "molecule_similarity_matrix"), "rb")
    )
    prot_similarity_matrix = pkl.load(
        open(get_data_filepath(config, "protein_similarity_matrix"), "rb")
    )


# ===================================================================
# --- STAGE 4: Generate Labeled Edges and Full Heterogeneous Graph ---
# ===================================================================

# region base edges
# 统一的检查点，检查最终的两个核心产物
checkpoint_files = ["typed_edge_list", "link_prediction_labels"]
if not check_files_exist(config, *checkpoint_files):
    print(
        "\n--- [Stage 4] Generating labeled edges for training and full typed graph... ---"
    )

    # --- 4a. 准备链接预测任务的正负样本 ---
    print("\n-> Generating positive and negative samples for link prediction...")
    positive_pairs = []

    # 从基础数据集 (DrugBank) 中提取 D-P 正样本
    for _, row in full_df[full_df["Y"] == 1].iterrows():
        if row["SMILES"] in dl2index and row["Protein"] in prot2index:
            d_idx = dl2index[row["SMILES"]]
            p_idx = prot2index[row["Protein"]]
            positive_pairs.append((d_idx, p_idx))

    # (可选) 从GtoPdb中提取 L-P 正样本
    if data_config["use_gtopdb"]:
        gtopdb_edges_df = pd.read_csv(
            gtopdb_processed_dir / "gtopdb_p-l_edges.csv",
            header=None,
            names=["Sequence", "SMILES", "Affinity"],
        )
        for _, row in gtopdb_edges_df.iterrows():
            if row["SMILES"] in dl2index and row["Sequence"] in prot2index:
                l_idx = dl2index[row["SMILES"]]
                p_idx = prot2index[row["Sequence"]]
                positive_pairs.append((l_idx, p_idx))

    # 进行随机负采样
    negative_pairs = []
    all_molecule_ids = list(dl2index.values())
    all_protein_ids = list(prot2index.values())
    positive_set = set(positive_pairs)  # 使用集合以提高查找效率

    with tqdm(total=len(positive_pairs), desc="Negative Sampling") as pbar:
        while len(negative_pairs) < len(positive_pairs):
            dl_idx = random.choice(all_molecule_ids)
            p_idx = random.choice(all_protein_ids)
            if (dl_idx, p_idx) not in positive_set:
                negative_pairs.append((dl_idx, p_idx))
                pbar.update(1)

    print(
        f"-> Generated {len(positive_pairs)} positive and {len(negative_pairs)} negative samples."
    )

    # 保存统一的、带标签的边文件
    pos_df = pd.DataFrame(positive_pairs, columns=["source", "target"])
    pos_df["label"] = 1
    neg_df = pd.DataFrame(negative_pairs, columns=["source", "target"])
    neg_df["label"] = 0
    dl_p_edges_df = pd.concat([pos_df, neg_df], ignore_index=True)
    dl_p_edges_df.to_csv(
        get_data_filepath(config, "link_prediction_labels"),
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
    mol_rows, mol_cols = np.where(dl_similarity_matrix > 0.988)
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
        get_data_filepath(config, "typed_edge_list"),
        index=False,
        header=True,
    )
    print(f"-> Total edges in heterogeneous graph: {len(typed_edges_df)}")

else:
    print("\n--- [Stage 4] Final graph files already exist. Skipping. ---")

# endregion (final_graph_construction)

print("\nData processing pipeline finished successfully!")
