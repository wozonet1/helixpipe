from collections import Counter
from pathlib import Path

import community as community_louvain  # for community detection
import hydra
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import research_template as rt
import seaborn as sns

from helixpipe.configs import register_all_schemas

# 导入所有需要的项目内部模块
from helixpipe.typing import AppConfig
from helixpipe.utils import get_path, register_hydra_resolvers

# 在所有Hydra操作之前，执行全局注册
register_all_schemas()
register_hydra_resolvers()

# --- 绘图与分析的辅助函数 ---


def analyze_degree_distribution(G: nx.Graph, output_dir: Path, config: AppConfig):
    """
    分析图的度分布，并生成直方图和对数-对数图。
    """
    print("\n--- [Topology] Analyzing Node Degree Distribution...")
    if not G.nodes:
        print("    - Graph has no nodes. Skipping.")
        return

    degrees = [val for (node, val) in G.degree()]

    # 打印基本统计
    print(f"    - Total Nodes: {G.number_of_nodes()}")
    print(f"    - Max Degree: {np.max(degrees)}")
    print(f"    - Min Degree: {np.min(degrees)}")
    print(f"    - Mean Degree: {np.mean(degrees):.2f}")

    plt.style.use("seaborn-v0_8-whitegrid")

    # --- 1. 绘制度分布直方图 ---
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.histplot(degrees, bins=50, ax=ax)
    ax.set_title(
        f"Node Degree Distribution\n(Config: {config.relations.name})", fontsize=16
    )
    ax.set_xlabel("Degree", fontsize=12)
    ax.set_ylabel("Frequency (Node Count)", fontsize=12)
    ax.set_yscale("log")  # 度分布通常是长尾的，log scale更清晰
    plt.tight_layout()
    plt.savefig(output_dir / "degree_distribution_hist.png", dpi=300)
    plt.close(fig)

    # --- 2. 绘制对数-对数度分布图 (用于检查无标度特性) ---
    degree_counts = Counter(degrees)
    degree_vals, counts = zip(*sorted(degree_counts.items()))

    # 转换为概率
    total_nodes = G.number_of_nodes()
    probabilities = [c / total_nodes for c in counts]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.scatter(degree_vals, probabilities, marker="o")
    ax.set_title(
        f"Log-Log Degree Distribution (Scale-Free Check)\n(Config: {config.relations.name})",
        fontsize=16,
    )
    ax.set_xlabel("Degree (k)", fontsize=12)
    ax.set_ylabel("Probability P(k)", fontsize=12)
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(output_dir / "degree_distribution_loglog.png", dpi=300)
    plt.close(fig)

    print("    - Degree distribution plots saved.")


def analyze_connectivity_and_communities(G: nx.Graph, output_dir: Path) -> str:
    """
    分析图的连通性和社群结构，返回一份文本报告。
    """
    print("\n--- [Topology] Analyzing Connectivity and Communities...")
    if not G.nodes:
        print("    - Graph has no nodes. Skipping.")
        return "Graph has no nodes."

    report_lines = ["=" * 30, "Connectivity & Community Report", "=" * 30]

    # --- 1. 连通分量分析 ---
    is_connected = nx.is_connected(G)
    report_lines.append("\n[Connectivity]")
    report_lines.append(f"Is the graph fully connected? {is_connected}")

    if not is_connected:
        num_components = nx.number_connected_components(G)
        report_lines.append(f"Number of connected components: {num_components}")

        # 找到最大连通分量
        largest_cc = max(nx.connected_components(G), key=len)
        largest_cc_size = len(largest_cc)
        largest_cc_ratio = (largest_cc_size / G.number_of_nodes()) * 100
        report_lines.append(
            f"Size of largest connected component: {largest_cc_size} nodes ({largest_cc_ratio:.2f}% of total)"
        )

    # --- 2. 社群检测 (Louvain) ---
    print("    - Running Louvain community detection...")
    partition = community_louvain.best_partition(G)
    num_communities = len(set(partition.values()))

    report_lines.append("\n[Community Structure (Louvain)]")
    report_lines.append(f"Number of communities detected: {num_communities}")

    modularity = community_louvain.modularity(partition, G)
    report_lines.append(f"Modularity of the partition: {modularity:.4f}")

    community_sizes = Counter(partition.values())
    report_lines.append(f"Size of largest community: {max(community_sizes.values())}")
    report_lines.append(f"Size of smallest community: {min(community_sizes.values())}")

    print("    - Analysis complete.")
    return "\n".join(report_lines)


# --- 主函数 ---


@hydra.main(
    config_path=str(rt.get_project_root() / "conf"),
    config_name="config",
    version_base=None,
)
def main(cfg: AppConfig):
    """
    一个独立的、由Hydra驱动的脚本，用于加载一个已生成的图文件，
    并对其进行深入的拓扑结构分析。
    """
    print("\n" + "=" * 80)
    print(" " * 20 + "STARTING GRAPH TOPOLOGY ANALYSIS SCRIPT")
    print("=" * 80)

    try:
        # --- 步骤 1: 准备输入和输出 ---
        print("--- [Step 1/3] Loading graph data...")

        # a. 默认分析 Fold 1 的训练图
        fold_idx_to_analyze = 1
        graph_path_factory = get_path(cfg, "processed.specific.graph_template")
        graph_path = graph_path_factory(
            prefix=f"fold_{fold_idx_to_analyze}", suffix="train"
        )

        if not graph_path.exists():
            raise FileNotFoundError(
                f"Graph file for Fold {fold_idx_to_analyze} not found at '{graph_path}'"
            )

        graph_df = pd.read_csv(graph_path)
        print(f"    - Successfully loaded graph with {len(graph_df)} edges.")

        # b. 准备输出目录 (这次必须包含 relations.name)
        output_dir = (
            rt.get_project_root()
            / "analysis_outputs"
            / (cfg.dataset_collection.name or "base")
            / cfg.data_params.name
            / cfg.relations.name
            / "topology_analysis"
        )
        rt.ensure_path_exists(output_dir / "dummy.txt")

        # --- 步骤 2: 构建 NetworkX 图对象 ---
        print("\n--- [Step 2/3] Building NetworkX graph object...")
        # 我们将所有边都视为无向边来进行整体分析
        G = nx.from_pandas_edgelist(
            graph_df,
            source=cfg.data_structure.schema.internal.graph_output.source_node,
            target=cfg.data_structure.schema.internal.graph_output.target_node,
            edge_attr=cfg.data_structure.schema.internal.graph_output.edge_type,
            create_using=nx.Graph(),
        )
        print(
            f"    - Graph object created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges."
        )

        # --- 步骤 3: 执行并保存分析 ---
        print("\n--- [Step 3/3] Running analysis functions...")

        # a. 度分布分析
        analyze_degree_distribution(G, output_dir, cfg)

        # b. 连通性与社群分析
        report_text = analyze_connectivity_and_communities(G, output_dir)

        # c. 保存文本报告
        report_path = output_dir / "topology_report.txt"
        with open(report_path, "w") as f:
            f.write(report_text)
        print(f"\n    - Topology text report saved to: {report_path.name}")

        print("\n" + "=" * 80)
        print(f"✅ TOPOLOGY ANALYSIS COMPLETE. All outputs saved to: {output_dir}")
        print("=" * 80)

    except FileNotFoundError as e:
        print(f"❌ FATAL: A required data file was not found: {e}")
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
