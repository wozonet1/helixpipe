from pathlib import Path

import hydra
import pandas as pd
import research_template as rt
from omegaconf import DictConfig

# 确保解析器被注册
rt.register_hydra_resolvers()


def analyze_single_graph_structure(cfg: DictConfig) -> pd.DataFrame | None:
    """
    分析【单一一组】实验配置下所有Fold的图结构，并返回其平均边数统计。
    """
    # 1. 构造一个能代表当前实验配置的唯一名称
    config_name = f"{cfg.data_params.name}\n({cfg.relations.name})"

    fold_edge_counts = []

    # 2. 循环遍历该配置下的所有Fold
    for fold_idx in range(1, cfg.training.k_folds + 1):
        try:
            # 3. 使用 get_path 精确定位图文件
            graph_path = rt.get_path(
                cfg,
                "processed.specific.graph_template",
                prefix=f"fold_{fold_idx}",
            )
            if not graph_path.exists():
                # print(f"  - Fold {fold_idx} graph not found, skipping.")
                continue

            edges_df = pd.read_csv(graph_path)
            if not edges_df.empty:
                counts = edges_df["edge_type"].value_counts().rename("count")
                fold_edge_counts.append(counts)

        except Exception as e:
            print(f"  - WARNING: Failed to process graph for Fold {fold_idx}: {e}")
            continue

    if not fold_edge_counts:
        return None

    # 4. 计算所有Fold的平均边数
    all_folds_df = pd.concat(fold_edge_counts, axis=1).fillna(0)
    mean_counts = all_folds_df.mean(axis=1).round().astype(int).reset_index()
    mean_counts.columns = ["edge_type", "count"]
    mean_counts["config_name"] = config_name

    return mean_counts


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    【V2 重构版】一个由Hydra multirun驱动的图结构比较分析脚本。
    它会为命令行中指定的每一组配置，分别计算平均边数，最后汇总成一张对比图。
    """
    print("\n" + "=" * 80)
    print(" " * 18 + "STARTING GRAPH STRUCTURE ANALYSIS (V2)")
    print("=" * 80)

    # 1. 分析当前配置下的图结构
    #    在多任务运行时，Hydra会为每个任务传入不同的cfg
    print(
        f"--> Analyzing configuration: data_params={cfg.data_params.name}, relations={cfg.relations.name}"
    )
    mean_counts_df = analyze_single_graph_structure(cfg)

    # 2. 在Hydra的多任务输出目录中，保存本次任务的分析结果
    if mean_counts_df is not None:
        output_path = (
            Path.cwd() / "mean_edge_counts.csv"
        )  # Hydra会自动为每个任务创建独立的工作目录
        mean_counts_df.to_csv(output_path, index=False)
        print(
            f"--> Saved mean edge counts for this configuration to '{output_path.name}'"
        )
    else:
        print("--> No graph files found for this configuration.")


if __name__ == "__main__":
    main()
