import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv
from torch_geometric.data import HeteroData


class RGCNLinkPredictor(torch.nn.Module):
    """
    一个端到端的异构图链接预测模型。

    它集成了：
    1. 一个基于 SAGEConv 的异构图编码器 (Encoder)。
    2. 一个用于 DTI 预测的链接解码器 (Decoder)。

    该模型直接与 train.py 中的 HeteroData 对象兼容。
    """

    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
        metadata: tuple,
        target_edge_type: tuple = ("drug", "drug_protein_interaction", "protein"),
    ):
        """
        初始化端到端模型。

        Args:
            hidden_channels (int): GNN隐藏层的维度.
            out_channels (int): 最终节点嵌入的维度 (Decoder也将使用此维度).
            num_layers (int): GNN的层数.
            dropout (float): Dropout 比例.
            metadata (tuple): 异构图的元数据 (node_types, edge_types)，由 train.py 提供.
            target_edge_type (tuple): 需要预测的链接类型.
        """
        super().__init__()

        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.target_edge_type = target_edge_type

        node_types, edge_types = metadata

        # --- 1. 构建异构 GNN 编码器 ---
        self.convs = torch.nn.ModuleList()

        # 第一层：将输入特征映射到隐藏层维度
        # HeteroConv 为每种边类型创建一个 SAGEConv 实例
        conv = HeteroConv(
            {
                edge_type: SAGEConv((-1, -1), hidden_channels)
                for edge_type in edge_types
            },
            aggr="sum",
        )
        self.convs.append(conv)

        # 中间层：隐藏层到隐藏层
        for _ in range(num_layers - 2):
            conv = HeteroConv(
                {
                    edge_type: SAGEConv((-1, -1), hidden_channels)
                    for edge_type in edge_types
                },
                aggr="sum",
            )
            self.convs.append(conv)

        # 最后一层：隐藏层到输出维度 (out_channels)
        conv = HeteroConv(
            {edge_type: SAGEConv((-1, -1), out_channels) for edge_type in edge_types},
            aggr="sum",
        )
        self.convs.append(conv)

        self.dropout = dropout

        # --- 2. 链接解码器 (Decoder) ---
        # 对于 'drug_protein_interaction'，我们使用一个简单的点积解码器。
        # 对于更复杂的 Het-NDLS，这里将是我们的创新点。
        # self.decoder = ... (这里不需要显式的层，直接在 decode 方法中实现)

    def forward(self, x_dict, edge_index_dict):
        """
        执行消息传递以计算节点嵌入。

        Args:
            x_dict (dict): 节点特征字典 {node_type: Tensor}.
            edge_index_dict (dict): 边索引字典 {edge_type: Tensor}.

        Returns:
            dict: 更新后的节点嵌入字典 {node_type: Tensor}.
        """
        if self.training and getattr(self, "_debug_probe_2_done", False) is False:
            print("\n" + "=" * 50)
            print(" " * 15 + "DEBUG PROBE 2: GNN ENCODER INPUT")
            print("=" * 50)
            print("Shapes of node features received by the first GNN layer:")
            for node_type, features in x_dict.items():
                print(f"  - '{node_type}': {features.shape}")
            print("=" * 50 + "\n")
            self._debug_probe_2_done = True  # 设置一个标志，确保只打印一次

        for conv in self.convs[:-1]:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            x_dict = {
                key: F.dropout(x, p=self.dropout, training=self.training)
                for key, x in x_dict.items()
            }

        # 最后一层不使用 ReLU 和 Dropout
        x_dict = self.convs[-1](x_dict, edge_index_dict)
        x_dict = {
            node_type: F.normalize(x, p=2.0, dim=-1) for node_type, x in x_dict.items()
        }
        return x_dict

    def decode(self, z_dict, edge_label_index):
        """
        为给定的边 (正样本或负样本) 计算链接存在概率的 logit。

        Args:
            z_dict (dict): 由 forward() 产生的节点嵌入.
            edge_label_index (Tensor): 需要评分的边的索引 [2, num_edges].

        Returns:
            Tensor: 每条边的预测分数 (logits).
        """
        src_type, _, dst_type = self.target_edge_type

        # 提取头节点和尾节点的嵌入
        z_src = z_dict[src_type][edge_label_index[0]]
        z_dst = z_dict[dst_type][edge_label_index[1]]

        # 使用点积作为解码器 (DistMult 的简化版)
        # 这是一个高效且常用的基线解码器
        dist_sq = ((z_src - z_dst) ** 2).sum(dim=-1)
        return -dist_sq

    def get_loss(
        self,
        hetero_data: HeteroData,
        edge_label_index: torch.Tensor,
        edge_label: torch.Tensor,
    ):
        """
        端到端的损失计算函数，直接在 train.py 中调用。

        Args:
            hetero_data (HeteroData): 包含图结构和节点特征的训练图.
            edge_label_index (Tensor): 一个 batch 的正/负样本边.
            edge_label (Tensor): 对应边的真实标签 (0 或 1).

        Returns:
            Tensor: 标量损失值.
        """
        # 1. 消息传递 (Encoder)
        # 注意: 这里只使用训练图的边 (hetero_data.edge_index_dict)
        z_dict = self.forward(hetero_data.x_dict, hetero_data.edge_index_dict)

        # 2. 链接预测 (Decoder)
        # 对 batch 中的边进行评分
        scores = self.decode(z_dict, edge_label_index)

        # 3. 计算损失
        loss = F.binary_cross_entropy_with_logits(scores, edge_label.float())

        return loss

    @torch.no_grad()
    def inference(self, hetero_data: HeteroData, edge_label_index: torch.Tensor):
        """
        推理函数，用于评估阶段。返回预测概率。
        """
        # 1. 消息传递
        z_dict = self.forward(hetero_data.x_dict, hetero_data.edge_index_dict)

        # 2. 链接预测
        scores = self.decode(z_dict, edge_label_index)

        # 3. 转换为概率 (0-1)
        probs = torch.sigmoid(scores)

        return probs
