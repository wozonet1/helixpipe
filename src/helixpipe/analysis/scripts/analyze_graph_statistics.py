import hydra
import matplotlib.pyplot as plt
import pandas as pd
import research_template as rt
import seaborn as sns  # 用于盒图

from helixpipe.configs import register_all_schemas
from helixpipe.typing import AppConfig
from helixpipe.utils import get_path, register_hydra_resolvers

from .plot_utils import plot_bar_chart_with_counts

register_all_schemas()
register_hydra_resolvers()


@hydra.main(
    config_path=str(rt.get_project_root() / "conf"),
    config_name="config",
    version_base=None,
)
def main(cfg: AppConfig):
    """
    对【每个Fold的图构建结果】进行统计分析，包括节点和边的数量/类型。
    回答的问题是：“模型在每个训练Fold中实际看到的图是什么样的？”
    """
    print("\n" + "=" * 80)
    print(" " * 20 + "STARTING GRAPH BUILD STATISTICS SCRIPT")
    print("=" * 80)

    try:
        # --- 步骤 1: 准备输出目录 ---
        output_dir = (
            rt.get_project_root()
            / "analysis_outputs"
            / (cfg.dataset_collection.name or "base")
            / cfg.data_params.name
            / cfg.relations.name
            / "graph_build_statistics"  # 新的专用目录
        )
        rt.ensure_path_exists(output_dir / "summary.txt")

        # --- 步骤 2: 循环遍历所有Fold，收集统计数据 ---
        all_fold_stats = []
        for fold_idx in range(1, cfg.training.k_folds + 1):
            print(f"\n--- Analyzing Fold {fold_idx} ---")
            fold_stats = {"fold": fold_idx}

            # a. 加载图文件
            graph_path_factory = get_path(cfg, "processed.specific.graph_template")
            graph_path = graph_path_factory(prefix=f"fold_{fold_idx}", suffix="train")
            if not graph_path.exists():
                print(
                    f"    - WARNING: Graph file for Fold {fold_idx} not found. Skipping."
                )
                continue
            graph_df = pd.read_csv(graph_path)

            # b. 节点统计 (参与图构建的)
            nodes_in_graph = set(graph_df["source"].unique()) | set(
                graph_df["target"].unique()
            )
            fold_stats["num_nodes"] = len(nodes_in_graph)

            # c. 边统计
            edge_counts = graph_df["edge_type"].value_counts()
            fold_stats.update(edge_counts.to_dict())
            fold_stats["total_edges"] = edge_counts.sum()

            all_fold_stats.append(fold_stats)

        if not all_fold_stats:
            raise FileNotFoundError("No valid graph files found across all folds.")

        # --- 步骤 3: 聚合数据并生成报告 ---
        stats_df = pd.DataFrame(all_fold_stats).set_index("fold").fillna(0)

        # 计算平均值和标准差
        summary_stats = stats_df.agg(["mean", "std"]).round(2)

        print("\n" + "=" * 50)
        print("Per-Fold Statistics:")
        print(stats_df.to_string())
        print("\nSummary Statistics (Mean/Std):")
        print(summary_stats.to_string())

        # 保存报告
        report_text = (
            "Per-Fold Statistics:\n"
            + stats_df.to_string()
            + "\n\nSummary Statistics (Mean/Std):\n"
            + summary_stats.to_string()
        )
        with open(output_dir / "graph_build_report.txt", "w") as f:
            f.write(report_text)

        # --- 步骤 4: 可视化 ---
        # a. 绘制平均边数量条形图 (与之前类似)
        plot_data = summary_stats.loc["mean"].drop(
            labels=["num_nodes", "total_edges"], errors="ignore"
        )
        plot_bar_chart_with_counts(
            data=plot_data.sort_values(ascending=False),
            output_path=output_dir / "average_edge_counts.png",
            title="Average Edge Type Distribution (over Folds)",
            xlabel="Edge Type",
            ylabel="Average Number of Edges",
        )

        # b. [NEW] 绘制盒图，展示每个边类型的跨Fold稳定性
        plot_df = stats_df.drop(columns=["num_nodes", "total_edges"])
        if not plot_df.empty:
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.boxplot(data=plot_df, ax=ax)
            ax.set_title("Edge Count Stability Across Folds", fontsize=16)
            ax.set_xlabel("Edge Type", fontsize=12)
            ax.set_ylabel("Number of Edges", fontsize=12)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(output_dir / "edge_counts_stability_boxplot.png", dpi=300)
            plt.close(fig)

        print(f"\n✅ GRAPH BUILD ANALYSIS COMPLETE. Outputs saved to: {output_dir}")

    except FileNotFoundError as e:
        print(
            f"❌ FATAL: A required data file was not found: {e.filename if hasattr(e, 'filename') else e}"
        )
        print(
            "   Please ensure you have successfully run the main data processing pipeline (`run.py`) with the corresponding configuration first."
        )
        return
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
