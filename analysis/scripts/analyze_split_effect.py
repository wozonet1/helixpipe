import pandas as pd
import hydra
from omegaconf import DictConfig
from pathlib import Path

import research_template as rt
from .plot_utils import plot_side_by_side_bar_chart

# 确保解析器被注册
rt.register_hydra_resolvers()


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def analyze_split_effect(cfg: DictConfig):
    """
    【V2 重构版】可视化冷启动分割对图结构的影响。

    比较 "Fold 1 训练图" 和 "理论上的完整图" 的边数差异。
    """
    print("\n" + "=" * 80)
    print(" " * 18 + "STARTING SPLIT EFFECT ANALYSIS (V2)")
    print("=" * 80)
    print(
        f"--> Analyzing config: data_structure={cfg.data_structure.name}, data_params={cfg.data_params.name}, relations={cfg.relations.name}, mode={cfg.training.coldstart.mode}"
    )

    # --- 1. 精确定位并加载所需文件 ---
    print("\n--- [Step 1/3] Loading graph and label files... ---")
    try:
        # 我们将分析第一折 (fold_idx=1)
        fold_idx = 1

        # a. 加载第一折的训练图文件
        train_graph_path = rt.get_path(
            cfg,
            "processed.specific.graph_template",
            prefix=f"fold_{fold_idx}",
        )
        train_graph_df = pd.read_csv(train_graph_path)

        # b. 加载第一折的测试标签文件，以找回被移除的交互
        test_labels_path = rt.get_path(
            cfg,
            "processed.specific.labels_template",
            prefix=f"fold_{fold_idx}",
            suffix="test",
        )
        test_labels_df = pd.read_csv(test_labels_path)

        print(f"--> Files loaded successfully for Fold {fold_idx}.")

    except FileNotFoundError as e:
        print(f"❌ FATAL: A required data file was not found: {e}")
        print(
            "   Please run the main data processing pipeline (`run.py`) for this configuration first."
        )
        return

    # --- 2. 计算两种图的边类型数量 ---
    print("\n--- [Step 2/3] Calculating edge counts for both graph types... ---")

    # a. "训练图 (Training Graph)" 的边数统计
    train_counts = train_graph_df["edge_type"].value_counts().reset_index()
    train_counts.columns = ["edge_type", "count"]
    train_counts["graph_type"] = "Training Graph (Fold 1)"

    # b. “理论上的完整图 (Full Graph)” 的边数统计
    #    我们通过将测试集中的正样本交互“加回”训练图的统计数据来模拟它
    full_counts = train_counts.copy()
    full_counts["graph_type"] = "Full Graph (Simulated)"

    # 统计测试集中的正样本数量
    label_col = cfg.data_structure.schema.internal.labeled_edges_output.label
    num_test_positives = (test_labels_df[label_col] == 1).sum()

    # 动态地确定交互边的类型
    # 我们的约定是交互边不包含 "similarity"
    interaction_edge_types = [
        et for et in train_counts["edge_type"].unique() if "similarity" not in et
    ]

    if num_test_positives > 0 and interaction_edge_types:
        # 在DTI场景下，测试集中的正样本通常只属于一种交互类型
        # 我们假设它们都属于被找到的第一个交互类型
        target_interaction_edge = interaction_edge_types[0]

        # 找到 'full_counts' 中对应的行并增加数量
        target_row_idx = full_counts.index[
            full_counts["edge_type"] == target_interaction_edge
        ]
        if not target_row_idx.empty:
            full_counts.loc[target_row_idx, "count"] += num_test_positives
            print(
                f"--> Simulated 'Full Graph' by adding {num_test_positives} positive test interactions to '{target_interaction_edge}'."
            )
        else:
            print(
                f"⚠️ WARNING: Could not find interaction edge type '{target_interaction_edge}' in graph stats."
            )

    # --- 3. 合并数据并生成可视化 ---
    print("\n--- [Step 3/3] Generating comparison plot... ---")

    plot_df = pd.concat([full_counts, train_counts], ignore_index=True)

    # 定义输出路径和文件名
    output_dir_name = f"{cfg.data_structure.name}-{cfg.data_params.name}"
    output_filename = (
        f"{cfg.relations.name}-{cfg.training.coldstart.mode}_split_effect.png"
    )
    output_path = Path("analysis/plots") / output_dir_name / output_filename

    plot_title = (
        f"Effect of '{cfg.training.coldstart.mode.upper()}' Cold-Start on Graph Structure\n"
        f"Config: {cfg.data_structure.name} / {cfg.data_params.name} / {cfg.relations.name}"
    )

    plot_side_by_side_bar_chart(
        df=plot_df,
        x_col="edge_type",
        y_col="count",
        hue_col="graph_type",
        title=plot_title,
        xlabel="Edge Type",
        ylabel="Number of Edges (Log Scale)",
        output_path=output_path,
        log_scale=True,
    )

    print("\n" + "=" * 80)
    print(" " * 28 + "ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    analyze_split_effect()
