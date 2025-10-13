import torch
import sys

# [关键] 手动将项目根目录添加到Python路径中
sys.path.append(".")
from src.data_utils.loaders import load_graph_data_for_fold, load_labels_for_fold
from torch_geometric.loader import LinkNeighborLoader


def main_debug():
    print("--- Starting Minimal Loader Debug ---")

    # 模拟一个简化的config对象
    class SimpleConfig:
        def __init__(self):
            self.data = {
                "primary_dataset": "DrugBank",
                "use_gtopdb": False,
            }  # 补全
            self.predictor = {"params": {"num_layers": 2}}

    config_mock = SimpleConfig()

    # 1. 加载数据
    hetero_graph = load_graph_data_for_fold(config_mock, 1)
    train_df, _ = load_labels_for_fold(config_mock, 1)

    # 2. 准备监督边
    target_edge_type = ("drug", "drug_protein_interaction", "protein")
    train_pos_df = train_df[train_df["label"] == 1]
    train_edge_label_index = (
        torch.from_numpy(train_pos_df[["source", "target"]].values)
        .t()
        .contiguous()
        .long()
    )

    # 3. 实例化Loader
    try:
        print("--> Instantiating Loader...")
        loader = LinkNeighborLoader(
            data=hetero_graph,
            num_neighbors=[-1] * 2,
            edge_label_index=(target_edge_type, train_edge_label_index),
            batch_size=512,
            shuffle=True,
            neg_sampling_ratio=1.0,
            num_workers=0,
        )
        print("--> Loader instantiated SUCCESSFULLY.")
    except Exception as e:
        print(f"!!! FAILED during Loader instantiation: {e}")
        return

    # 4. 尝试取出第一个batch
    try:
        print("--> Attempting to fetch the first batch...")
        first_batch = next(iter(loader))
        print("--> First batch fetched SUCCESSFULLY!")
        print("Batch content:", first_batch)
    except Exception as e:
        print(f"!!! FAILED when fetching the first batch: {e}")


if __name__ == "__main__":
    main_debug()
