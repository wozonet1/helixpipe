# src/data_utils/loaders.py
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from project_types import AppConfig

# [优化] 将 research_template 的导入也放在这里
# 这样，所有与路径管理相关的依赖都集中在了这个文件中
import research_template as rt


def create_global_to_local_maps(config: AppConfig) -> dict:
    """从nodes.csv创建全局到局部的ID映射字典。"""
    nodes_df = pd.read_csv(rt.get_path(config, "processed.common.nodes_metadata"))
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


def create_global_id_to_type_map(config: AppConfig) -> dict:
    """
    【新增】从nodes.csv创建一个全局ID到节点类型的反向映射字典。
    """
    nodes_df = pd.read_csv(rt.get_path(config, "processed.common.nodes_metadata"))
    # 使用pandas的高效功能，直接将两列转换为字典
    return pd.Series(nodes_df.node_type.values, index=nodes_df.node_id).to_dict()


def load_graph_structure_from_files(config: AppConfig, fold_idx: int) -> HeteroData:
    """
    【底层】加载指定fold的图结构和节点特征，组装成一个原始的HeteroData对象。
    """
    # 1. 加载节点数据
    nodes_df = pd.read_csv(rt.get_path(config, "processed.common.nodes_metadata"))
    features_array = np.load(rt.get_path(config, "processed.common.node_features"))
    features_tensor = torch.from_numpy(features_array).float()

    data = HeteroData()

    # 2. 填充节点特征 (按类型)
    for node_type, group in nodes_df.groupby("node_type"):
        node_ids = group["node_id"].values
        data[node_type].x = features_tensor[node_ids]

    # 3. 加载边数据
    edges_path = rt.get_path(
        config, "processed.specific.graph_template", prefix=f"fold_{fold_idx}"
    )
    edges_df = pd.read_csv(edges_path)

    id_to_type_str = pd.Series(
        nodes_df.node_type.values, index=nodes_df.node_id
    ).to_dict()

    # 4. 填充边索引 (使用全局ID)
    for edge_type_str, group in edges_df.groupby("edge_type"):
        sources = group["source"].values
        targets = group["target"].values
        source_type = id_to_type_str[sources[0]]
        target_type = id_to_type_str[targets[0]]

        # 注意：这里的edge_index仍然是全局ID
        edge_index_np = np.stack([sources, targets])
        data[source_type, edge_type_str, target_type].edge_index = torch.from_numpy(
            edge_index_np
        ).long()
    return data


def load_supervision_labels_for_fold(
    config: AppConfig,
    fold_idx: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    为指定的Fold加载带标签的训练和测试边集。
    """
    print(f"--- [Loader] Loading labeled edges for Fold {fold_idx}... ---")

    lp_labels_key = "processed.specific.labels_template"

    train_path = rt.get_path(
        config, lp_labels_key, prefix=f"fold_{fold_idx}", suffix="train"
    )
    test_path = rt.get_path(
        config, lp_labels_key, prefix=f"fold_{fold_idx}", suffix="test"
    )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df
