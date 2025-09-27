# src/data_utils/loaders.py

import pandas as pd
import torch
from torch_geometric.data import HeteroData
from omegaconf import DictConfig
import torch_geometric.transforms as T

# [优化] 将 research_template 的导入也放在这里
# 这样，所有与路径管理相关的依赖都集中在了这个文件中
import research_template as rt


def create_global_to_local_maps(config: DictConfig) -> dict:
    """从nodes.csv创建全局到局部的ID映射字典。"""
    nodes_df = pd.read_csv(rt.get_path(config, "processed.nodes_metadata"))
    maps = {}
    node_type_groups = nodes_df.groupby("node_type")
    for node_type in sorted(node_type_groups.groups.keys()):
        group = node_type_groups.get_group(node_type)

        # group['node_id'].values 保证了我们是按照全局ID的顺序来创建局部ID
        maps[node_type] = {
            global_id: local_id
            for local_id, global_id in enumerate(group["node_id"].values)
        }

    return maps


def convert_df_to_local_tensors(
    df: pd.DataFrame,
    global_to_local_maps: dict,
    src_node_type: str,
    dst_node_type: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    一个辅助函数，将包含全局ID的DataFrame转换为包含局部ID的PyTorch张量。
    """
    # 1. 使用映射字典进行ID转换
    src_local = torch.tensor(
        [global_to_local_maps[src_node_type][gid] for gid in df["source"]],
        dtype=torch.long,
    )
    dst_local = torch.tensor(
        [global_to_local_maps[dst_node_type][gid] for gid in df["target"]],
        dtype=torch.long,
    )

    # 2. 准备边索引张量和标签张量
    edge_label_index = torch.stack([src_local, dst_local]).to(device)
    edge_label = torch.from_numpy(df["label"].values).to(device)
    return edge_label_index, edge_label


def load_graph_data_for_fold(
    config: DictConfig,
    fold_idx: int,
    global_to_local_maps: dict,  # <-- [关键] 接收“权威指南”
) -> HeteroData:
    """
    [最终正确版] 为指定的Fold加载图结构，并使用【外部传入】的ID映射来构建一个
    包含【局部ID】的HeteroData对象。

    这个版本能够正确处理所有节点类型（包括ligand）和所有边类型。
    """
    print(
        f"--- [Loader] Loading and building graph with LOCAL IDs for Fold {fold_idx}... ---"
    )

    # 1. 加载节点数据
    nodes_df = pd.read_csv(rt.get_path(config, "processed.nodes_metadata"))
    features_tensor = torch.from_numpy(
        pd.read_csv(rt.get_path(config, "processed.node_features"), header=None).values
    ).float()

    data = HeteroData()

    # 2. 填充节点特征
    # [优化] 使用我们传入的映射字典的键，来保证顺序一致性
    for node_type in global_to_local_maps.keys():
        # 从 nodes_df 中筛选出当前类型的所有节点
        mask = nodes_df["node_type"] == node_type
        # 按照全局ID，从总特征张量中提取出当前类型的特征
        data[node_type].x = features_tensor[nodes_df[mask]["node_id"].values]

    # 3. 加载边数据，并使用【传入的】映射转换为局部ID
    edges_path = rt.get_path(
        config, "processed.typed_edge_list_template", split_suffix=f"_fold{fold_idx}"
    )
    edges_df = pd.read_csv(edges_path)

    # [优化] 这个临时的id_to_type_str现在变得更可靠，因为它也基于nodes_df
    id_to_type_str = pd.Series(
        nodes_df.node_type.values, index=nodes_df.node_id
    ).to_dict()

    for edge_type_str, group in edges_df.groupby("edge_type"):
        sources_global = group["source"].values
        targets_global = group["target"].values

        source_type = id_to_type_str[sources_global[0]]
        target_type = id_to_type_str[targets_global[0]]
        edge_type_tuple = (source_type, edge_type_str, target_type)

        # [关键] 使用由 train.py 传入的、唯一的 global_to_local_maps
        source_local_ids = [
            global_to_local_maps[source_type][gid] for gid in sources_global
        ]
        target_local_ids = [
            global_to_local_maps[target_type][gid] for gid in targets_global
        ]

        data[edge_type_tuple].edge_index = torch.tensor(
            [source_local_ids, target_local_ids], dtype=torch.long
        )
    data = T.ToUndirected(merge=True)(data)
    print(f"--- HeteroData object for Fold {fold_idx} constructed successfully. ---")
    return data


def load_labels_for_fold(
    config: DictConfig,
    fold_idx: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    为指定的Fold加载带标签的训练和测试边集。
    """
    print(f"--- [Loader] Loading labeled edges for Fold {fold_idx}... ---")

    lp_labels_key = "processed.link_prediction_labels_template"
    train_suffix = config.training.evaluation.train_file_suffix
    test_suffix = config.training.evaluation.test_file_suffix

    train_path = rt.get_path(
        config, lp_labels_key, split_suffix=f"_fold{fold_idx}{train_suffix}"
    )
    test_path = rt.get_path(
        config, lp_labels_key, split_suffix=f"_fold{fold_idx}{test_suffix}"
    )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df
