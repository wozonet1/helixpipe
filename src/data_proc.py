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


def process_data(config: DictConfig):
    # region init
    data_config = config["data"]
    params_config = config["params"]
    runtime_config = config["runtime"]
    device = runtime_config["gpu"]
    primary_dataset = data_config["primary_dataset"]
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
    # In src/data_proc.py

    # ===================================================================
    # --- STAGE 4: Generate Labeled Edges and Full Heterogeneous Graph ---
    # ===================================================================
    graph_files_dict = {
        "typed_edges_template": "processed.typed_edge_list_template",
        "link_labels": "processed.link_prediction_labels",
    }

    # The check for the main graph file now dynamically resolves the hashed filename.
    # We check 'link_labels' separately as it's generated only once.
    # Note: The restart_flag will override these checks.
    typed_edges_path = rt.get_path(config, graph_files_dict["typed_edges_template"])
    link_labels_path = rt.get_path(config, graph_files_dict["link_labels"])

    if not typed_edges_path.exists() or not link_labels_path.exists() or restart_flag:
        print("\n--- [Stage 4] Generating labeled edges and/or full typed graph... ---")

        # --- 4a. 准备链接预测任务的正负样本 ---
        # This part runs only if the link_labels file is missing or restart is forced.
        if not link_labels_path.exists() or restart_flag:
            print("--> Generating positive and negative samples for link prediction...")

            # Use a set to store NORMALIZED positive pairs to handle duplicates/symmetry.
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
                if (
                    pd.notna(canonical_s)
                    and canonical_s in dl2index
                    and protein in prot2index
                ):
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
            # endregion

            # region negative edges
            negative_pairs = []
            all_molecule_ids = list(dl2index.values())
            all_protein_ids = list(prot2index.values())

            # Get the strategy from config, default to 'uniform' for backward compatibility
            sampling_strategy = params_config.get(
                "negative_sampling_strategy", "uniform"
            )
            print(
                f"--> Performing negative sampling with strategy: '{sampling_strategy}'..."
            )

            with tqdm(
                total=len(positive_pairs),
                desc=f"Negative Sampling ({sampling_strategy})",
            ) as pbar:
                if sampling_strategy == "popular":
                    # --- Popularity-Biased Strategy ---
                    # a. Calculate node degrees (popularity) from positive pairs
                    mol_degrees = {}
                    prot_degrees = {}
                    for u, v in positive_pairs:
                        # Assuming u is molecule and v is protein after normalization
                        mol_degrees[u] = mol_degrees.get(u, 0) + 1
                        prot_degrees[v] = prot_degrees.get(v, 0) + 1

                    # b. Create weight lists that correspond to the ID lists
                    # Use a small default weight (e.g., 1) for nodes that have 0 degree in the positive set
                    mol_weights = [
                        mol_degrees.get(mol_id, 1) for mol_id in all_molecule_ids
                    ]
                    prot_weights = [
                        prot_degrees.get(prot_id, 1) for prot_id in all_protein_ids
                    ]

                    while len(negative_pairs) < len(positive_pairs):
                        # c. Use random.choices for weighted sampling
                        dl_idx = random.choices(
                            all_molecule_ids, weights=mol_weights, k=1
                        )[0]
                        p_idx = random.choices(
                            all_protein_ids, weights=prot_weights, k=1
                        )[0]

                        normalized_candidate = tuple(sorted((dl_idx, p_idx)))
                        if normalized_candidate not in positive_pairs_normalized_set:
                            negative_pairs.append((dl_idx, p_idx))
                            pbar.update(1)

                elif sampling_strategy == "uniform":
                    # --- Uniform Random Strategy (Original Logic) ---
                    while len(negative_pairs) < len(positive_pairs):
                        dl_idx = random.choice(all_molecule_ids)
                        p_idx = random.choice(all_protein_ids)

                        normalized_candidate = tuple(sorted((dl_idx, p_idx)))
                        if normalized_candidate not in positive_pairs_normalized_set:
                            negative_pairs.append((dl_idx, p_idx))
                            pbar.update(1)
                else:
                    raise ValueError(
                        f"Unknown negative_sampling_strategy: '{sampling_strategy}'. "
                        f"Choose from 'uniform' or 'popular'."
                    )

            print(
                f"-> Generated {len(positive_pairs)} positive and {len(negative_pairs)} negative samples."
            )

            # Save the unified, labeled edge file
            pos_df = pd.DataFrame(positive_pairs, columns=["source", "target"])
            pos_df["label"] = 1
            neg_df = pd.DataFrame(negative_pairs, columns=["source", "target"])
            neg_df["label"] = 0
            dl_p_edges_df = pd.concat([pos_df, neg_df], ignore_index=True)

            rt.ensure_path_exists(link_labels_path)
            dl_p_edges_df.to_csv(link_labels_path, index=False, header=True)
            print(f"-> Saved link prediction labels to {link_labels_path}")
        else:
            print(
                "\n--> Link prediction labels file already exists. Loading positive pairs from cache."
            )
            labeled_edges_df = pd.read_csv(link_labels_path)
            positive_pairs_df = labeled_edges_df[labeled_edges_df["label"] == 1]
            positive_pairs = list(
                zip(positive_pairs_df["source"], positive_pairs_df["target"])
            )
        # endregion

        # region sim edges
        # --- 4b. 构建完整的、带类型的异构图边列表 ---
        print(
            "\n-> Assembling full heterogeneous graph based on `include_relations` config..."
        )

        relations_config = config.relations.flags
        typed_edges_list = []

        # --- Add Interaction Edges based on switches ---
        dp_added, lp_added = 0, 0
        # The `positive_pairs` list contains all unique (Molecule_ID, Protein_ID) tuples.
        for u, v in positive_pairs:
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
                typed_edges_list.append(
                    [
                        i + prot_start_index,
                        j + prot_start_index,
                        "protein_protein_similarity",
                    ]
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
                if i < j:
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
        print(f"\n--> Saving final graph structure to: {typed_edges_path}")
        rt.ensure_path_exists(typed_edges_path)

        typed_edges_df = pd.DataFrame(
            typed_edges_list, columns=["source", "target", "edge_type"]
        )
        typed_edges_df.to_csv(typed_edges_path, index=False, header=True)

        print(f"-> Total edges in the final heterogeneous graph: {len(typed_edges_df)}")

    else:
        relations_suffix = rt.get_relations_suffix(config)

        # 2. Get the specific filename that was found and caused the skip
        # (This makes the message even more informative)
        typed_edges_key = graph_files_dict["typed_edges_template"]
        found_path = rt.get_path(config, typed_edges_key)

        # 3. Print a clear, informative, and accurate message
        print(
            "\n--- [Stage 4] Skipping graph generation: All necessary files found. ---"
        )
        print(f"    - Configuration Suffix: '{relations_suffix}'")
        print(f"    - Found existing graph file: {found_path.name}")

    # This is the final print of the script
    print("\nData processing pipeline finished successfully!")


# # region Hydra Entry
# @hydra.main(config_path="../conf", config_name="config", version_base=None)
# def hydra_entry_point(cfg: DictConfig):
#     """
#     This function serves as the Hydra entry point to run data processing.
#     """

#     # Convert OmegaConf to a plain dict for maximum compatibility
#     config_dict = OmegaConf.to_container(cfg, resolve=True)

#     # Perform global setup
#     rt.set_seeds(config_dict["runtime"]["seed"])
#     rt.setup_dataset_directories(config_dict)

#     # Execute the main logic
#     process_data(config_dict)


# if __name__ == "__main__":
#     hydra_entry_point()
# # end region
