import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

from nasnet.typing import AppConfig

# 从同一包内的兄弟模块导入底层函数
from .loaders import (
    create_global_id_to_type_map,
    create_global_to_local_maps,
    load_graph_structure_from_files,
    load_supervision_labels_for_fold,
)


def prepare_e2e_data(config: AppConfig, fold_idx: int) -> tuple:
    """
    【总准备函数】为E2E工作流，执行所有的数据加载、转换和净化步骤。
    """
    print(f"\n--- [Data Prep] Starting full data preparation for Fold {fold_idx} ---")

    # 1. 调用底层加载器，获取原始数据对象
    hetero_graph_global = load_graph_structure_from_files(config, fold_idx)
    train_df, test_df = load_supervision_labels_for_fold(config, fold_idx)
    maps = create_global_to_local_maps(config)
    id_to_type_map = create_global_id_to_type_map(config)
    print("Step 1/4: Raw data loaded from disk.")

    # 2. 将图中的全局ID转换为局部ID
    hetero_graph_local = HeteroData()
    for node_type in hetero_graph_global.node_types:
        hetero_graph_local[node_type].x = hetero_graph_global[node_type].x

    for edge_type in hetero_graph_global.edge_types:
        src_type, _, dst_type = edge_type
        src_global = hetero_graph_global[edge_type].edge_index[0]
        dst_global = hetero_graph_global[edge_type].edge_index[1]

        src_local = torch.tensor(
            [maps[src_type][gid.item()] for gid in src_global], dtype=torch.long
        )
        dst_local = torch.tensor(
            [maps[dst_type][gid.item()] for gid in dst_global], dtype=torch.long
        )

        hetero_graph_local[edge_type].edge_index = torch.stack([src_local, dst_local])
    print("Step 2/4: Graph converted to use local node IDs.")

    # 3. 执行所有必要的净化和转换
    # a. Purify dtypes and move to CPU (idempotent)
    for store in hetero_graph_local.stores:
        for key, value in store.items():
            if torch.is_tensor(value):
                store[key] = value.cpu().contiguous()
    # b. Force undirectedness
    hetero_graph_local = T.ToUndirected()(hetero_graph_local)
    print("Step 3/4: Graph purified (device, memory) and made undirected.")

    hetero_graph_local = T.AddSelfLoops()(hetero_graph_local)
    print("Step 4/4: Self-loops added to all node types.")  # 更新步骤编号
    # 4. 准备监督边 (同样转换为局部ID)
    # a. 转换训练监督边
    train_src_local = torch.tensor(
        [maps[id_to_type_map[gid]][gid] for gid in train_df["source"]], dtype=torch.long
    )
    # 目标节点通常都是蛋白质，但为了健壮性，我们同样使用动态查找
    train_dst_local = torch.tensor(
        [maps[id_to_type_map[gid]][gid] for gid in train_df["target"]], dtype=torch.long
    )
    train_edge_label_index_local = torch.stack([train_src_local, train_dst_local])

    # b. 转换测试监督边
    test_src_local = torch.tensor(
        [maps[id_to_type_map[gid]][gid] for gid in test_df["source"]], dtype=torch.long
    )
    test_dst_local = torch.tensor(
        [maps[id_to_type_map[gid]][gid] for gid in test_df["target"]], dtype=torch.long
    )
    test_edge_label_index_local = torch.stack([test_src_local, test_dst_local])
    test_labels = torch.from_numpy(test_df["label"].values).float()

    print("--- [Data Prep] All data prepared and ready for training. ---")

    return (
        hetero_graph_local,
        train_edge_label_index_local,
        test_edge_label_index_local,
        test_labels,
        maps,
    )
