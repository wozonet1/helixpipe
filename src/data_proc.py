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
import argparse
from pathlib import Path
from utils import DPLdata_filepath

# region gtopdb data process


# def integrate_gtopdb_data(
#     data_dir, smiles_list, protein_sequences, drug2index, prot2index
# ):
#     """
#     将预处理好的GtoPdb数据整合进现有的节点和边列表中。
#     这个函数是幂等的：如果整合已经完成，它会从缓存加载。
#     主义data_dir应该是当前数据集的目录,而不是processed_gtopdb目录。
#     """
#     print("\n--- [Integration Stage] Starting GtoPdb Data Integration ---")

#     # 定义缓存文件的路径
#     integrated_data_cache_path = data_dir / "integration_cache.pkl"

#     # 检查点逻辑
#     if integrated_data_cache_path.exists():
#         print("-> Found existing integration cache. Loading...")
#         with open(integrated_data_cache_path, "rb") as f:
#             cached_data = pkl.load(f)
#         print("--- [Integration Stage] Complete (from cache). ---")
#         return (
#             cached_data["smiles_list"],
#             cached_data["protein_sequences"],
#             cached_data["drug2index"],
#             cached_data["prot2index"],
#             cached_data["extra_pl_edges"],
#         )

#     print("-> No cache found. Performing first-time integration...")

#     # --- 1. 加载GtoPdb处理后的文件 ---
#     gtopdb_processed_dir = data_dir.parent / "gtopdb/processed_gtopdb"
#     try:
#         gtopdb_ligands = pd.read_csv(
#             gtopdb_processed_dir / "gtopdb_ligands.csv",
#             header=None,
#             names=["PubChem CID", "SMILES"],
#         )
#         gtopdb_edges = pd.read_csv(
#             gtopdb_processed_dir / "gtopdb_p-l_edges.csv",
#             header=None,
#             names=["UniProt ID", "PubChem CID", "Affinity"],
#         )
#     except FileNotFoundError:
#         print(
#             "Warning: Processed GtoPdb files not found. Skipping integration."
#             "Please run gtopdb_proc.py first if you want to include this data."
#         )
#         # 如果文件不存在，直接返回原始数据，不进行任何操作
#         return smiles_list, protein_sequences, drug2index, prot2index, []

#     # --- 2. 整合新的配体节点 (Ligands) ---
#     original_drug_count = len(drug2index)
#     new_smiles_added = 0

#     for _, row in gtopdb_ligands.iterrows():
#         smiles = row["SMILES"]
#         # 只有当这个SMILES是全新的，才把它加入
#         if smiles not in drug2index:
#             smiles_list.append(smiles)
#             drug2index[smiles] = len(drug2index)  # 分配新的全局唯一ID
#             new_smiles_added += 1

#     print(f"-> Added {new_smiles_added} new unique ligand nodes from GtoPdb.")

#     # --- 3. 整合新的蛋白质-配体边 (P-L Edges) ---
#     extra_pl_edges = []
#     edges_added = 0

#     # 创建一个SMILES到ID的反向映射，以方便查找
#     cid_to_smiles = gtopdb_ligands.set_index("PubChem CID")["SMILES"]

#     for _, row in gtopdb_edges.iterrows():
#         uniprot_id = row["UniProt ID"]
#         pubchem_cid = row["PubChem CID"]

#         # 查找对应的SMILES
#         smiles = cid_to_smiles.get(pubchem_cid)

#         # 只有当这条边的两个端点都存在于我们的索引中时，才添加它
#         if uniprot_id in prot2index and smiles in drug2index:
#             prot_idx = prot2index[uniprot_id]
#             drug_idx = drug2index[smiles]
#             # 您也可以在这里保留亲和力值 row['Affinity']，如果后续需要的话
#             extra_pl_edges.append((prot_idx, drug_idx))
#             edges_added += 1

#     print(f"-> Added {edges_added} new Protein-Ligand edges from GtoPdb.")

#     # --- 4. 保存缓存，以便下次直接加载 ---
#     cache_data = {
#         "smiles_list": smiles_list,
#         "protein_sequences": protein_sequences,
#         "drug2index": drug2index,
#         "prot2index": prot2index,
#         "extra_pl_edges": extra_pl_edges,
#     }
#     with open(integrated_data_cache_path, "wb") as f:
#         pkl.dump(cache_data, f)

#     print(f"-> Integration results saved to cache: {integrated_data_cache_path}")
#     print("--- [Integration Stage] Complete. ---")

#     return smiles_list, protein_sequences, drug2index, prot2index, extra_pl_edges


# endregion

# region d feature


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


def calculate_drug_similarity(smiles_list):
    # 将SMILES转换为RDKit分子对象
    molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
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
    results = Parallel(n_jobs=args.cpus)(
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

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="DrugBank", help="type of dataset.")
parser.add_argument("--cpus", type=int, default=80, help="Number of cpus to use.")
parser.add_argument("--seed", type=int, default=514, help="Random seed.")
parser.add_argument("--gpu", type=str, default="cuda:1", help="which gpu to use.")
parser.add_argument(
    "--gtopdb", type=bool, default=False, help="Whether to integrate GtoPdb data."
)
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
device = args.gpu

current_path = Path(__file__)
datapath = current_path.parent.parent / f"data/{args.dataset}"
foldername = "gtopdb" if args.gtopdb else "baseline"

if not (datapath / foldername).exists():
    (datapath / foldername).mkdir(parents=True, exist_ok=True)


def check_file_exists(*file_names: str):
    for filename in file_names:
        if not DPLdata_filepath(datapath, filename, foldername).exists():
            return False
    print(f"All {file_names} in  {foldername} exist!")
    return True


full = pd.read_csv(datapath / "full.csv")
smiles_list = []
protein_sequences = []

for i, row in full.iterrows():
    if row["SMILES"] not in smiles_list:
        smiles_list.append(row["SMILES"])
    if row["Protein"] not in protein_sequences:
        protein_sequences.append(row["Protein"])

if args.gtopdb:
    extra_p_sequences = []
    extra_p = pd.read_csv(
        datapath.parent / "gtopdb/processed/gtopdb_p-l_edges.csv",
        header=None,
        low_memory=False,
    )
    for i, row in extra_p.iterrows():
        extra_p_sequences.append(row[0])
    print(f"Extra proteins from GtoPdb: {len(extra_p_sequences)}")
    base_p_num = len(protein_sequences)
    protein_sequences += extra_p_sequences
    protein_sequences = list(set(protein_sequences))
    final_p_num = len(protein_sequences)
    print(
        f"Added {final_p_num - base_p_num} unique proteins, \
        Total proteins after GtoPdb integration: {final_p_num}"
    )

# endregion


# region index
if not check_file_exists("drug2index.pkl", "prot2index.pkl"):
    prot2index = {}
    drug2index = {}
    for i, d in enumerate(smiles_list):
        if d not in drug2index:
            drug2index[d] = len(drug2index)
    for i, p in enumerate(protein_sequences):
        if p not in prot2index:
            prot2index[p] = len(drug2index) + len(prot2index)

    pkl.dump(
        drug2index, open(DPLdata_filepath(datapath, "drug2index.pkl", foldername), "wb")
    )
    pkl.dump(
        prot2index, open(DPLdata_filepath(datapath, "prot2index.pkl", foldername), "wb")
    )
# TODO: add ligand edge
# endregion

# region base edge&nodes
if not check_file_exists(
    "DrPrNum_DrPr.csv",
    "AllNegative_DrPr.csv",
    "Allnode_DrPr.csv",  # TODO: add typed nodes and edges
    "num.pkl",
):
    positive_pair_d = []
    positive_pair_p = []
    positive_pair = []
    for i, row in full.iterrows():
        if int(row["Y"]) == 1:
            positive_pair_d.append(drug2index[row["SMILES"]])
            positive_pair_p.append(prot2index[row["Protein"]])
            positive_pair.append(
                (drug2index[row["SMILES"]], prot2index[row["Protein"]])
            )

    negative_pair_d = []
    negative_pair_p = []
    ind_d = list(drug2index.values())
    ind_p = list(prot2index.values())
    for _ in range(len(positive_pair)):
        i_d = random.choice(ind_d)
        i_p = random.choice(ind_p)
        if (i_d, i_p) not in positive_pair:
            negative_pair_d.append(i_d)
            negative_pair_p.append(i_p)

    Allnode = [i for i in range(len(drug2index) + len(prot2index))]
    Allnode_df = pd.DataFrame({"node": Allnode})

    print(f"pos pair: {len(positive_pair)}, neg pair: {len(negative_pair_d)}")

    DrPrNum_Drpr = pd.DataFrame({"pair_d": positive_pair_d, "pair_p": positive_pair_p})
    DrPrNum_Drpr.to_csv(
        DPLdata_filepath(datapath, "DrPrNum_DrPr.csv", foldername),
        index=False,
        header=False,
    )
    AllNegative_DrPr = pd.DataFrame(
        {"pair_d": negative_pair_d, "pair_p": negative_pair_p}
    )
    AllNegative_DrPr.to_csv(
        DPLdata_filepath(datapath, "AllNegative_DrPr.csv", foldername),
        index=False,
        header=False,
    )
    Allnode_df.to_csv(
        DPLdata_filepath(datapath, "Allnode_DrPr.csv", foldername),
        index=False,
        header=False,
    )

    num = {"drug_num": len(drug2index), "prot_num": len(prot2index)}
    pkl.dump(num, open(DPLdata_filepath(datapath, "num.pkl", foldername), "wb"))

# endregion

# region features&sim
if not check_file_exists(
    "drug_similarity_matrix.pkl",
    "prot_similarity_matrix.pkl",
    "AllNodeAttribute_DrPr.csv",
):
    drug_embeddings = []
    for i, d in enumerate(smiles_list):
        drug_embeddings.append(extract_features(d).cpu().detach().numpy())

    protein_embeddings = []
    for i, p in enumerate(protein_sequences):
        protein_embeddings.append(extract_aa_features(p).cpu().detach().numpy())

    # prot_similarity_matrix = cosine_similarity(protein_embeddings)
    drug_similarity_matrix = calculate_drug_similarity(smiles_list)
    # notice:用U与C一样的方式计算相似度
    prot_similarity_matrix = calculate_protein_similarity(protein_sequences)

    pkl.dump(
        drug_similarity_matrix,
        open(
            DPLdata_filepath(datapath, "drug_similarity_matrix.pkl", foldername), "wb"
        ),
    )
    pkl.dump(
        prot_similarity_matrix,
        open(
            DPLdata_filepath(datapath, "prot_similarity_matrix.pkl", foldername), "wb"
        ),
    )
    AllNodeAttribute_DrPr = pd.concat(
        [pd.DataFrame(drug_embeddings), pd.DataFrame(protein_embeddings)], axis=0
    )
    AllNodeAttribute_DrPr.to_csv(
        DPLdata_filepath(datapath, "AllNodeAttribute_DrPr.csv", foldername),
        index=False,
        header=False,
    )

# region sim edges
if not check_file_exists(
    "prot_edge.csv",
    "drug_edge.csv",
    "drug_prot_edge.csv",
):
    prot_edge1 = []
    prot_edge2 = []
    drug_edge1 = []
    drug_edge2 = []

    for i, row in pd.read_csv(
        DPLdata_filepath(datapath, "DrPrNum_DrPr.csv", foldername), header=None
    ).iterrows():
        prot_edge1.append(int(row[0]))
        prot_edge2.append(int(row[1]))
        drug_edge1.append(int(row[0]))
        drug_edge2.append(int(row[1]))

    print(f"{args.dataset} positive edges: {len(prot_edge1)}")

    pp = 0
    p_e1 = []
    p_e2 = []
    for i in range(len(prot_similarity_matrix)):
        for j in range(i + 1, len(prot_similarity_matrix)):
            if prot_similarity_matrix[i][j] > 0.985:
                p_e1.append(i)
                p_e2.append(j)
                pp += 1

    random_array = np.random.randint(0, len(p_e1) - 1, size=min(500, len(p_e1) - 1))
    for ind in random_array:
        prot_edge1.append(p_e1[ind])
        prot_edge2.append(p_e2[ind])

    print("Prot-Prot edges: ", pp)
    df = pd.DataFrame({"0": prot_edge1, "1": prot_edge2})
    df.to_csv(
        DPLdata_filepath(datapath, "prot_edge.csv", foldername),
        index=False,
        header=False,
    )
    print("P edges: ", len(prot_edge1))

    dd = 0
    d_e1 = []
    d_e2 = []
    for i in range(len(drug_similarity_matrix)):
        for j in range(i + 1, len(drug_similarity_matrix)):
            if drug_similarity_matrix[i][j] > 0.988:
                d_e1.append(i)
                d_e2.append(j)
                dd += 1

    random_array = np.random.randint(0, len(d_e1) - 1, size=min(500, len(d_e1) - 1))
    for ind in random_array:
        prot_edge1.append(d_e1[ind])
        prot_edge2.append(d_e2[ind])
        drug_edge1.append(d_e1[ind])
        drug_edge2.append(d_e2[ind])

    print("Drug-Drug edges: ", dd)
    print("Total edges: ", len(prot_edge1))
    print("D edges: ", len(drug_edge1))

    df = pd.DataFrame({"0": drug_edge1, "1": drug_edge2})
    df.to_csv(
        DPLdata_filepath(datapath, "drug_edge.csv", foldername),
        index=False,
        header=False,
    )

    df = pd.DataFrame({"0": prot_edge1, "1": prot_edge2})
    df.to_csv(
        DPLdata_filepath(datapath, "drug_prot_edge.csv", foldername),
        index=False,
        header=False,
    )
# endregion

# region ligands
