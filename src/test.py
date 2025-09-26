# analysis/inspect_data.py

import pandas as pd
import hydra
from omegaconf import DictConfig
import research_template as rt


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def inspect_data(cfg: DictConfig):
    """
    一个用于快速检查特定Fold的数据内容的调试脚本。
    """
    fold_to_inspect = 1

    print(f"--- Inspecting Data for Fold {fold_to_inspect} ---")

    # --- 1. 检查图谱拓扑文件 ---
    graph_path = rt.get_path(
        cfg,
        "processed.typed_edge_list_template",
        split_suffix=f"_fold{fold_to_inspect}",
    )
    print(f"\n--- Reading Graph Topology: {graph_path} ---")
    try:
        graph_df = pd.read_csv(graph_path)
        print("Graph Info:")
        print(graph_df.info())
        print("\nEdge Type Counts:")
        print(graph_df["edge_type"].value_counts())
        print("\nFirst 5 rows:")
        print(graph_df.head())
    except FileNotFoundError:
        print("!!! Graph file not found. Please run data processing first.")

    # --- 2. 检查训练标签文件 ---
    train_labels_path = rt.get_path(
        cfg,
        "processed.link_prediction_labels_template",
        split_suffix=f"_fold{fold_to_inspect}{cfg.training.evaluation.train_file_suffix}",
    )
    print(f"\n--- Reading Training Labels: {train_labels_path} ---")
    try:
        train_df = pd.read_csv(train_labels_path)
        print("Train Labels Info:")
        print(train_df.info())
        print("\nLabel Distribution:")
        print(train_df["label"].value_counts())
        print("\nFirst 5 rows:")
        print(train_df.head())
    except FileNotFoundError:
        print("!!! Train labels file not found.")


if __name__ == "__main__":
    inspect_data()
