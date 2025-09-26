import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

# [优化] 将 research_template 的导入放在 try-except 块中，提供更友好的错误提示
try:
    import research_template as rt
except ImportError:
    print(
        "Error: Could not import 'research_template'. Please ensure it is installed correctly."
    )
    sys.exit(1)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def visualize_split_diff(cfg: DictConfig):
    """
    一个由Hydra驱动的脚本，用于可视化冷启动分割前后图谱结构的差异。

    它通过对比“完整图谱”和“第一折的训练图谱”中各种边的数量，
    直观地展示冷启动分割对图谱信息造成的“破坏”程度，以证明评估的严格性。
    """
    print("--- Starting Split Difference Visualization ---")
    print(OmegaConf.to_yaml(cfg.analysis))
    print(f"Using Relations: {cfg.relations.name}")

    # --- 1. 解析文件路径 ---
    # [优化] 将文件路径的解析集中在一起，并在开始时就进行存在性检查
    try:
        # 训练图谱: 这是 `typed_edges` 文件，它只包含训练边
        train_graph_path = rt.get_path(
            cfg, "processed.typed_edge_list_template", split_suffix="_fold1"
        )

        # 测试集标签: 我们需要它来找回被分割出去的正样本交互边
        test_labels_path = rt.get_path(
            cfg,
            "processed.link_prediction_labels_template",
            split_suffix=f"_fold1{cfg.training.evaluation.test_file_suffix}",
        )

        print(f"Loading training graph from: {train_graph_path}")
        print(f"Loading test labels from: {test_labels_path}")

        train_graph_df = pd.read_csv(train_graph_path)
        test_labels_df = pd.read_csv(test_labels_path)

    except FileNotFoundError as e:
        print(f"\n!!! FATAL ERROR: A required file was not found: {e.filename}")
        print(
            "    This likely means `data_proc.py` has not been run successfully for the"
        )
        print(f"    '{cfg.relations.name}' relation group with k_folds >= 1.")
        sys.exit(1)

    # --- 2. 计算边类型的数量 ---

    # a) 训练图谱的边数量
    train_graph_counts = train_graph_df["edge_type"].value_counts().reset_index()
    train_graph_counts.columns = ["edge_type", "count"]
    train_graph_counts["graph_type"] = "Training Graph (Fold 1)"

    # b) "完整图谱" 的边数量
    # [优化] 我们通过一个务实的方法来构建“完整图谱”的统计数据：
    # "完整图谱" = "训练图谱" + "测试集中的正样本交互边"
    # 这在可视化上完美地展示了交互边的损失，同时避免了重新计算所有相似性边的复杂性。
    full_graph_counts = train_graph_counts.copy()
    full_graph_counts["graph_type"] = "Full Graph"

    num_test_positives = (test_labels_df["label"] == 1).sum()

    # [优化] 动态识别交互边的名称，而不是硬编码
    # 我们假设交互边是那些不包含 "similarity" 的边
    interaction_edge_types = [
        edge
        for edge in full_graph_counts["edge_type"].unique()
        if "similarity" not in edge
    ]

    if not interaction_edge_types:
        print(
            "Warning: Could not dynamically find interaction edge types. Visualization might be incomplete."
        )
    else:
        # 假设所有测试集正样本都属于第一个找到的交互类型（在DTI/DTA场景下通常是正确的）
        target_interaction_edge = interaction_edge_types[0]
        print(
            f"Adding {num_test_positives} test interactions to '{target_interaction_edge}' for Full Graph."
        )

        full_graph_counts.loc[
            full_graph_counts["edge_type"] == target_interaction_edge, "count"
        ] += num_test_positives

    # --- 3. 合并数据并绘图 ---
    plot_df = pd.concat([full_graph_counts, train_graph_counts], ignore_index=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(
        figsize=(cfg.analysis.plot_settings.width, cfg.analysis.plot_settings.height)
    )

    sns.barplot(
        data=plot_df,
        x="edge_type",
        y="count",
        hue="graph_type",
        palette=cfg.analysis.plot_settings.palette,
        ax=ax,
    )

    ax.set_yscale("log")
    ax.set_title(
        f"Cold-Start Split Effect on Graph Structure ({cfg.relations.name})\n"
        f"Dataset: {cfg.data.primary_dataset}",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xlabel("Edge Type", fontsize=12)
    ax.set_ylabel("Number of Edges (Log Scale)", fontsize=12)
    ax.tick_params(axis="x", rotation=45, labelsize=10)
    ax.legend(title="Graph Type")

    plt.tight_layout()

    # --- 4. 保存图像 ---
    output_dir = Path("split_diff")
    output_dir.mkdir(exist_ok=True)

    relations_suffix = rt.get_relations_suffix(cfg)
    save_path = (
        output_dir
        / f"{cfg.data.primary_dataset}_{relations_suffix}_split_comparison.png"
    )

    plt.savefig(save_path, dpi=300)
    print(f"\nVisualization successful! Plot saved to: {save_path}")


if __name__ == "__main__":
    visualize_split_diff()
