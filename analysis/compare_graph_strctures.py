# analysis/compare_graph_structures.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import hydra
from omegaconf import DictConfig
from pathlib import Path

# Assuming your research_template is installed or in the Python path
import research_template as rt


def plot_edge_distribution_bars(ax, edges_df: pd.DataFrame, title: str):
    """
    [NEW & IMPROVED] Helper function to generate a horizontal bar chart for
    edge type distribution on a given Axes. Bar charts are often better for
    comparing categories.
    """
    if edges_df is None or edges_df.empty:
        ax.text(0.5, 0.5, "No Data Found", ha="center", va="center", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    # 1. Calculate counts for each edge type
    edge_counts = edges_df["edge_type"].value_counts().sort_values(ascending=True)
    total_edges = sum(edge_counts)

    # 2. Plot horizontal bar chart
    bars = ax.barh(
        edge_counts.index,
        edge_counts.values,
        color=sns.color_palette("viridis", len(edge_counts)),
    )

    # 3. Add labels (count and percentage) to each bar
    for bar in bars:
        width = bar.get_width()
        percentage = 100 * width / total_edges
        label = f" {width} ({percentage:.1f}%)"
        ax.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            ha="left",
            fontsize=10,
        )

    # 4. Beautify the plot
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("Number of Edges", fontsize=12)
    ax.tick_params(axis="y", labelsize=10)
    # Remove the top and right spines for a cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Set x-axis limit to give space for labels
    ax.set_xlim(right=ax.get_xlim()[1] * 1.3)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def analyze_and_compare_structures(cfg: DictConfig):
    """
    [REFACTORED] Analyzes and compares the edge type distribution of the 'baseline'
    and 'gtopdb' variants for the primary dataset defined in the config.
    """
    print("--- Starting Graph Structure Comparison Analysis ---")

    # This script will be run from the root, so we get the CWD
    output_dir = Path.cwd() / "analysis"
    output_dir.mkdir(exist_ok=True)

    # --- 1. Load data for both variants robustly ---
    variants_data = {}
    for variant_name, use_gtopdb_flag in [
        (
            "baseline",
            False,
        ),
        ("gtopdb", True),
    ]:
        print(
            f"--> Loading '{variant_name}' variant for dataset '{cfg.data.primary_dataset}'..."
        )
        try:
            # Create a clean config for each variant to avoid side-effects
            variant_config = cfg.copy()
            variant_config.data.use_gtopdb = use_gtopdb_flag

            # Get the path and load the data
            edges_path = rt.get_path(
                variant_config, "processed.typed_edge_list_template"
            )
            variants_data[variant_name] = pd.read_csv(edges_path)
            print(
                f"    - Loaded {len(variants_data[variant_name])} edges from {edges_path.name}"
            )
        except FileNotFoundError:
            print(
                f"    - WARNING: Typed edges file not found for '{variant_name}' variant. Skipping."
            )
            variants_data[variant_name] = None

    # --- 2. Create the comparison plot ---
    print("--> Generating comparison plot...")
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(
        len(variants_data),
        1,
        figsize=(10, 5 * len(variants_data)),
        squeeze=False,  # Ensures axes is always a 2D array
    )
    axes = axes.flatten()  # Flatten to 1D array for easy iteration

    fig.suptitle(
        f"Graph Structure Comparison for: {cfg.data.primary_dataset.upper()}",
        fontsize=20,
        fontweight="bold",
    )

    # Plot for each variant
    for i, (title, df) in enumerate(variants_data.items()):
        plot_edge_distribution_bars(axes[i], df, title)

    # --- 3. Finalize and Save the plot ---
    # Use a more robust layout manager
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust rect to make space for suptitle

    output_filename = (
        output_dir / f"structure_comparison_{cfg.data.primary_dataset}.png"
    )
    plt.savefig(output_filename, dpi=300)

    print("\n" + "=" * 50)
    print("      Analysis Complete!")
    print("=" * 50)
    print(f"-> Comparison plot saved to: {output_filename}")
    print("=" * 50)


if __name__ == "__main__":
    # This allows the script to be run directly from the project root directory
    analyze_and_compare_structures()
