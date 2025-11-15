from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig

import helixlib as hx

# ... (您已有的绘图函数) ...


def plot_bar_chart_with_counts(
    data: pd.Series,
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    log_scale: bool = True,
):
    """
    为一个pd.Series (例如 value_counts() 的结果) 绘制一个美观的条形图。
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))

    data.plot(kind="bar", ax=ax)

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=45, ha="right")

    if log_scale:
        ax.set_yscale("log")

    plt.tight_layout()
    hx.ensure_path_exists(output_path)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"    - Bar chart saved to: {output_path.name}")


def plot_similarity_distributions(
    df: pd.DataFrame, output_dir: Path, config: DictConfig
):
    """
    为DataFrame中不同类型的相似度绘制分布直方图和统计信息。
    """
    if df.empty:
        print("    - No similarity data to plot. Skipping distribution plots.")
        return

    plt.style.use("seaborn-v0_8-whitegrid")

    for sim_type, group_df in df.groupby("type"):
        fig, ax = plt.subplots(figsize=(12, 7))

        sns.histplot(group_df["similarity"], bins=50, kde=True, ax=ax, stat="density")

        mean_val = group_df["similarity"].mean()
        median_val = group_df["similarity"].median()
        ax.axvline(mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.3f}")
        ax.axvline(
            median_val, color="green", linestyle="-.", label=f"Median: {median_val:.3f}"
        )

        title = (
            f"Distribution of '{sim_type}' (Top-{config.data_params.similarity_top_k} Candidates)\n"
            f"Dataset: {config.data_structure.name} / Params: {config.data_params.name}"
        )
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xlabel("Cosine Similarity", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.legend()

        plt.tight_layout()
        output_path = output_dir / f"{sim_type}_distribution.png"
        plt.savefig(output_path, dpi=300)
        plt.close(fig)
        print(f"    - Distribution plot saved to: {output_path.name}")


def plot_pos_neg_similarity_kde(
    df: pd.DataFrame, output_path: Path, config: DictConfig
):
    """
    在同一张图上绘制正负样本的相似度核密度估计曲线。
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))

    sns.kdeplot(
        data=df, x="similarity", hue="label", fill=True, common_norm=False, ax=ax
    )

    title = (
        f"Similarity Distribution of Positive vs. Negative Pairs\n"
        f"Dataset: {config.data_structure.name} / Params: {config.data_params.name}"
    )
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("Embedding Cosine Similarity (Drug-Protein)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(title="Pair Type")

    plt.tight_layout()
    hx.ensure_path_exists(output_path)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"    - Positive vs. Negative KDE plot saved to: {output_path.name}")


def plot_side_by_side_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    log_scale: bool = True,
):
    """
    一个通用的、用于绘制并排比较条形图的函数。
    """
    if df.empty:
        print(f"--> No data to plot for '{title}'. Skipping.")
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14, 8))

    sns.barplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=hue_col,
        ax=ax,
    )

    if log_scale:
        ax.set_yscale("log")
        # 为log scale的y轴添加次刻度线，增强可读性
        ax.minorticks_on()
        ax.grid(which="minor", axis="y", linestyle=":")

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis="x", rotation=30, ha="right", labelsize=10)
    ax.legend(title=hue_col.replace("_", " ").title())

    plt.tight_layout()
    hx.ensure_path_exists(output_path)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"--> Side-by-side bar chart saved to: {output_path}")


def plot_grouped_bar_chart(
    df: pd.DataFrame,
    index_col: str,
    columns_col: str,
    values_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    log_scale: bool = True,
):
    """
    一个更通用的、用于绘制分组条形图的函数。
    """
    if df.empty:
        print(f"--> No data to plot for '{title}'. Skipping.")
        return

    # 1. 数据透视
    plot_df = (
        df.pivot_table(index=index_col, columns=columns_col, values=values_col)
        .fillna(0)
        .astype(int)
    )

    # 2. 排序以保证一致性
    plot_df = plot_df[plot_df.max().sort_values(ascending=False).index]  # 按数量排序列
    plot_df = plot_df.sort_index(ascending=True)  # 按名称排列行

    # 3. 绘图
    sns.set_style("whitegrid")
    ax = plot_df.plot(
        kind="bar", stacked=False, figsize=(16, 9), width=0.8, logy=log_scale
    )

    # 4. 美化
    ax.set_title(title, fontsize=20, fontweight="bold", pad=20)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=12)
    plt.xticks(rotation=15, ha="right")
    ax.legend(
        title=columns_col.replace("_", " ").title(),
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout(rect=[0, 0.03, 0.85, 0.95])

    # 5. 保存
    hx.ensure_path_exists(output_path)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"--> Grouped bar chart saved to: {output_path}")


def plot_histogram(data: np.ndarray, title: str, xlabel: str, output_path: Path):
    """通用的直方图绘制函数。"""
    if data is None or len(data) == 0:
        print(f"--> No data to plot for '{title}'. Skipping.")
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))

    sns.histplot(data, bins=50, kde=True, ax=ax)  # 简化了颜色参数，使用默认主题

    mean_val = np.mean(data)
    median_val = np.median(data)
    ax.axvline(mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.3f}")
    ax.axvline(
        median_val, color="green", linestyle="-.", label=f"Median: {median_val:.3f}"
    )

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Frequency (Count of Pairs)", fontsize=12)
    ax.legend()

    plt.tight_layout()
    hx.ensure_path_exists(output_path)  # 确保输出目录存在
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"--> Distribution plot saved to: {output_path}")


# 可以在这里添加更多通用的绘图函数...
