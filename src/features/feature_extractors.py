from rdkit import Chem
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


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
