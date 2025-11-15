from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import research_template as rt
import seaborn as sns
import umap.umap_ as umap

# 导入降维和聚类所需的库
# 您需要先安装它们: pip install scikit-learn umap-learn
from sklearn.cluster import KMeans

from helixpipe.configs import register_all_schemas

# 导入所有需要的项目内部模块
from helixpipe.typing import AppConfig
from helixpipe.utils import get_path, register_hydra_resolvers

# 在所有Hydra操作之前，执行全局注册
register_all_schemas()
register_hydra_resolvers()


def process_and_plot_embeddings(
    embeddings: np.ndarray,
    entity_type: str,
    output_dir: Path,
    config: AppConfig,
    n_samples: int = 5000,
    n_clusters: int = 15,
    seed: int = 42,
):
    """
    一个辅助函数，负责对给定的嵌入进行采样、降维、聚类和绘图。
    """
    print(
        f"\n--- [Visualizer] Processing {entity_type} embeddings ({len(embeddings)} total)..."
    )
    if len(embeddings) == 0:
        print("    - No embeddings to process. Skipping.")
        return

    # --- 1. 采样 ---
    if len(embeddings) > n_samples:
        print(f"    - Sampling {n_samples} entities for visualization...")
        rng = np.random.default_rng(seed)
        sample_indices = rng.choice(len(embeddings), size=n_samples, replace=False)
        sampled_embeddings = embeddings[sample_indices]
    else:
        sampled_embeddings = embeddings

    # --- 2. 使用 UMAP 进行降维 (通常比 t-SNE 更快，且能更好地保留全局结构) ---
    print(
        f"    - Running UMAP to reduce dimensions from {sampled_embeddings.shape[1]} to 2..."
    )
    reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=15, min_dist=0.1)
    embedding_2d_umap = reducer.fit_transform(sampled_embeddings)

    # --- 3. 使用 K-Means 进行聚类以“染色” ---
    # 我们在2D空间上进行聚类，以便颜色与视觉分组直接对应
    print(f"    - Running K-Means to find {n_clusters} clusters for coloring...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
    cluster_labels = kmeans.fit_predict(embedding_2d_umap)

    # --- 4. 绘图 ---
    plot_df = pd.DataFrame(
        {
            "x": embedding_2d_umap[:, 0],
            "y": embedding_2d_umap[:, 1],
            "cluster": cluster_labels,
        }
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 12))

    sns.scatterplot(
        data=plot_df,
        x="x",
        y="y",
        hue="cluster",
        palette="viridis",  # 使用一个色彩丰富的色板
        s=5,  # 使用小点以避免重叠
        legend=None,  # 不显示图例，因为它没有实际意义
        ax=ax,
    )

    title = (
        f"UMAP Projection of {entity_type} Embedding Space ({n_samples} samples)\n"
        f"Colors indicate K-Means clusters (k={n_clusters})"
    )
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")

    output_path = output_dir / f"{entity_type.lower()}_embedding_space_umap.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"    - UMAP visualization saved to: {output_path.name}")


project_root = rt.get_project_root()
config_path = project_root / "conf"


@hydra.main(config_path=str(config_path), config_name="config", version_base=None)
def main(cfg: AppConfig):
    """
    一个独立的、由Hydra驱动的脚本，用于对嵌入空间进行降维和可视化，
    以进行“拓扑结构”的健全性检查。
    """
    print("\n" + "=" * 80)
    print(" " * 15 + "STARTING EMBEDDING SPACE VISUALIZATION SCRIPT")
    print("=" * 80)

    try:
        # --- 步骤 1: 加载核心数据产物 ---
        print("--- [Step 1/2] Loading node features and metadata...")

        nodes_df = pd.read_csv(get_path(cfg, "processed.common.nodes_metadata"))
        features_np = np.load(get_path(cfg, "processed.common.node_features"))

        # --- 步骤 2: 分离嵌入并调用绘图函数 ---
        num_molecules = (nodes_df["node_type"] != "protein").sum()

        molecule_embeddings = features_np[:num_molecules]
        protein_embeddings = features_np[num_molecules:]

        # 准备输出目录
        output_dir = (
            rt.get_project_root()
            / "analysis_outputs"
            / (cfg.dataset_collection.name or "base")
            / cfg.data_params.name
            / "embedding_space_sanity_check"  # 创建一个专门的子目录
        )
        rt.ensure_path_exists(output_dir / "dummy.txt")

        # 为分子和蛋白质分别进行可视化
        process_and_plot_embeddings(
            embeddings=molecule_embeddings,
            entity_type="Molecule",
            output_dir=output_dir,
            config=cfg,
            seed=cfg.runtime.seed,
        )

        process_and_plot_embeddings(
            embeddings=protein_embeddings,
            entity_type="Protein",
            output_dir=output_dir,
            config=cfg,
            seed=cfg.runtime.seed,
        )

        print("\n" + "=" * 80)
        print(f"✅ VISUALIZATION COMPLETE. All plots saved to: {output_dir}")
        print("=" * 80)

    except FileNotFoundError as e:
        print(
            f"❌ FATAL: A required data file was not found: {e.filename if hasattr(e, 'filename') else e}"
        )
        print(
            "   Please ensure you have successfully run the main data processing pipeline (`run.py`) first to generate all necessary files."
        )
        return
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
