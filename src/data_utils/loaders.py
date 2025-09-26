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
    for node_type, group in nodes_df.groupby("node_type"):
        local_df = (
            group.reset_index(drop=True)
            .reset_index()
            .rename(columns={"index": "local_id"})
        )
        maps[node_type] = local_df.set_index("node_id")["local_id"].to_dict()
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


<<<<<<< HEAD
def load_graph_data_for_fold(config: DictConfig, fold_idx: int) -> HeteroData:
    print(f"--- [Loader] Loading and building graph for Fold {fold_idx}... ---")
=======
def load_graph_data_for_fold(
    config: DictConfig, fold_idx: int, global_to_local_maps: dict
) -> HeteroData:
    """
    [最终正确版] 为指定的Fold加载图结构，并使用【外部传入】的ID映射来构建一个
    包含【局部ID】的HeteroData对象。
    """
    print(
        f"--- [Loader] Loading and building graph with LOCAL IDs for Fold {fold_idx}... ---"
    )
>>>>>>> 5b499c6 (data_proc采用了辅助函数,以及归纳式训练,现在又是0.5了)

    # 1. 加载节点数据
    nodes_df = pd.read_csv(rt.get_path(config, "processed.nodes_metadata"))
    features_tensor = torch.from_numpy(
        pd.read_csv(rt.get_path(config, "processed.node_features"), header=None).values
    ).float()

    # --- [核心修复] 创建一个“统一”的图，而不是“分离”的图 ---

    data = HeteroData()

<<<<<<< HEAD
    # a. 将【所有】节点特征，放入一个统一的 data['node'] 存储中
    #    节点的顺序，严格按照全局ID 0, 1, 2...
    data["node"].x = features_tensor

    # b. 创建一个 node_type 张量，告诉PyG每个节点的类型
    #    这个张量与 features_tensor 中的节点一一对应
    node_type_names = list(nodes_df["node_type"].unique())
    type_name_to_id = {name: i for i, name in enumerate(node_type_names)}
    node_type_tensor = torch.tensor(
        [type_name_to_id[t] for t in nodes_df["node_type"]], dtype=torch.long
    )

    # c. 加载边数据（依然是全局ID）
=======
    # 2. 填充节点特征
    node_type_groups = nodes_df.groupby("node_type")
    for node_type in sorted(node_type_groups.groups.keys()):
        group = node_type_groups.get_group(node_type)
        data[node_type].x = features_tensor[group["node_id"].values]

    # 3. 加载边数据，并使用【传入的】映射转换为局部ID
>>>>>>> 5b499c6 (data_proc采用了辅助函数,以及归纳式训练,现在又是0.5了)
    edges_path = rt.get_path(
        config, "processed.typed_edge_list_template", split_suffix=f"_fold{fold_idx}"
    )
    edges_df = pd.read_csv(edges_path)

    # d. 填充边索引（依然是全局ID）
    id_to_type_str = {row.node_id: row.node_type for row in nodes_df.itertuples()}
    for edge_type_str, group in edges_df.groupby("edge_type"):
<<<<<<< HEAD
        sources = torch.from_numpy(group["source"].values)
        targets = torch.from_numpy(group["target"].values)

        source_type = id_to_type_str[sources[0].item()]
        target_type = id_to_type_str[targets[0].item()]
        edge_type_tuple = (source_type, edge_type_str, target_type)

        data[edge_type_tuple].edge_index = torch.stack([sources, targets], dim=0)
=======
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
>>>>>>> 5b499c6 (data_proc采用了辅助函数,以及归纳式训练,现在又是0.5了)

    # e. [关键] 使用PyG的内置转换函数，将这个“全局ID图”自动转换为我们最终需要的“局部ID图”
    #    这个函数会自动地、正确地处理所有ID映射和节点类型分割
    print(
        "    -> Converting graph from global to local node indices using PyG's transform..."
    )
    data = T.ToDevice("cpu", "x")(data)  # 确保所有特征都在CPU上
    data = T.ToHetero(
        node_type_tensor=node_type_tensor, node_type_names=node_type_names
    )(data)

    # f. 应用ToUndirected转换
    data = T.ToUndirected(merge=True)(data)

    # 4. 应用ToUndirected转换
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
