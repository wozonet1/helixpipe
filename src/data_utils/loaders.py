# src/data_utils/loaders.py

import pandas as pd
import torch
from torch_geometric.data import HeteroData
from omegaconf import DictConfig

# [优化] 将 research_template 的导入也放在这里
# 这样，所有与路径管理相关的依赖都集中在了这个文件中
import research_template as rt


def load_graph_data_for_fold(config: DictConfig, fold_idx: int) -> HeteroData:
    """
    为指定的Fold加载图结构，并构建一个PyG HeteroData对象。
    """
    print(f"--- [Loader] Loading graph data for Fold {fold_idx}... ---")

    # --- 1. 加载节点级的元数据和特征 ---
    nodes_df = pd.read_csv(rt.get_path(config, "processed.nodes_metadata"))
    features_tensor = torch.from_numpy(
        pd.read_csv(rt.get_path(config, "processed.node_features"), header=None).values
    ).float()

    data = HeteroData()

    # --- 2. 填充节点特征，并创建一个临时的全局->局部ID映射 ---
    # [优化] 这个 node_type_map 实际上只在这个函数内部被用作辅助工具
    node_type_map = {}
    for node_type in nodes_df["node_type"].unique():
        mask = nodes_df["node_type"] == node_type
        # PyG的HeteroData会自动根据添加顺序创建从0开始的局部ID
        data[node_type].x = features_tensor[nodes_df[mask]["node_id"].values]

        # 我们需要这个映射来将 typed_edges.csv 中的全局ID转换为局部ID
        node_ids_of_type = nodes_df[mask]["node_id"].values
        for local_id, global_id in enumerate(node_ids_of_type):
            node_type_map[global_id] = (node_type, local_id)

    # --- 3. 填充边的索引（使用局部ID） ---
    edges_path = rt.get_path(
        config, "processed.typed_edge_list_template", split_suffix=f"_fold{fold_idx}"
    )
    edges_df = pd.read_csv(edges_path)

    id_to_type_str = {row.node_id: row.node_type for row in nodes_df.itertuples()}

    for edge_type_str, group in edges_df.groupby("edge_type"):
        sources = group["source"].values
        targets = group["target"].values

        source_type = id_to_type_str[sources[0]]
        target_type = id_to_type_str[targets[0]]
        edge_type_tuple = (source_type, edge_type_str, target_type)

        # 使用我们创建的映射，将全局ID转换为局部ID
        source_local_ids = [node_type_map[gid][1] for gid in sources]
        target_local_ids = [node_type_map[gid][1] for gid in targets]

        data[edge_type_tuple].edge_index = torch.tensor(
            [source_local_ids, target_local_ids], dtype=torch.long
        )

    print(f"--- HeteroData object for Fold {fold_idx} constructed successfully. ---")
    return data


def load_labels_for_fold(
    config: DictConfig, fold_idx: int
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
