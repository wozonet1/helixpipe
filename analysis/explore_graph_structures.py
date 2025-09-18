# analysis/explore_graph_structures_final.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import hydra
from omegaconf import DictConfig
from pathlib import Path
import re

# research_template is assumed to be installed and available


def plot_incremental_comparison(
    all_counts_df: pd.DataFrame, title: str, output_path: Path
):
    """
    [COMBINED & IMPROVED] Generates a grouped bar chart to compare edge counts
    across multiple discovered experimental configurations, using a log scale
    for clarity.
    """
    if all_counts_df.empty:
        print("WARNING: No data to plot.")
        return

    # --- 1. Prepare the data for plotting ---
    # Pivot the table to have experiment configs as the main groups (x-axis)
    # and edge types as the categories to compare.
    plot_df = (
        all_counts_df.pivot_table(
            index="config_name", columns="edge_type", values="count"
        )
        .fillna(0)
        .astype(int)
    )

    # Sort columns (edge_types) by their maximum count for a consistent legend order
    plot_df = plot_df[plot_df.max().sort_values(ascending=False).index]

    # Sort rows (configs) alphabetically for a consistent plot order
    plot_df = plot_df.sort_index(ascending=True)

    # --- 2. Create the plot ---
    sns.set_style("whitegrid")

    # Create the grouped bar plot from the transposed DataFrame
    ax = plot_df.plot(
        kind="bar",
        stacked=False,  # Grouped, not stacked
        figsize=(16, 9),
        width=0.8,
        logy=True,  # Use logarithmic scale for the y-axis
    )

    # --- 3. Beautify and Add Labels ---
    ax.set_title(title, fontsize=20, fontweight="bold", pad=20)
    ax.set_ylabel("Number of Edges (Log Scale)", fontsize=12)
    ax.set_xlabel("Experimental Configurations", fontsize=12)

    # Rotate x-axis labels for better readability if there are many configs
    if len(plot_df) > 4:
        plt.xticks(rotation=15, ha="right")
    else:
        plt.xticks(rotation=0)

    # Improve the legend
    ax.legend(
        title="Edge Types",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
    )

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Adjust layout to make space for the legend on the side
    plt.tight_layout(rect=[0, 0.03, 0.85, 0.95])

    # --- 4. Save the figure ---
    plt.savefig(output_path, dpi=300)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def explore_and_visualize_structures(cfg: DictConfig):
    """
    Automatically discovers all generated graph structures for a given dataset
    and visualizes their edge distributions using an incremental comparison plot.
    """
    print("--- Starting Automated Graph Structure Exploration (Final Version) ---")

    # Correctly use hydra's original CWD to build absolute paths
    project_root = Path(hydra.utils.get_original_cwd())
    output_dir = project_root / "analysis" / "edge_structures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Define the search path based on the LATEST config structure ---
    dataset_name = cfg.data.primary_dataset  # Using cfg.data, not cfg.dataset
    processed_dir = project_root / "data" / dataset_name / "processed"

    # --- 2. Discover files and aggregate all edge counts ---
    print(f"--> Scanning for graph files in: {processed_dir}")
    all_counts = []

    # Sort the glob results to ensure a consistent processing order
    for file_path in sorted(processed_dir.rglob("typed_edges-*.csv")):
        variant_name = file_path.parent.name
        match = re.search(r"typed_edges-(.*)\.csv", file_path.name)
        relations_suffix = match.group(1) if match else "unknown"

        # Create a cleaner name for the x-axis tick labels
        config_name = f"{variant_name.capitalize()}\n({relations_suffix})"

        try:
            edges_df = pd.read_csv(file_path)
            if not edges_df.empty:
                edge_counts = edges_df["edge_type"].value_counts().reset_index()
                edge_counts.columns = ["edge_type", "count"]
                edge_counts["config_name"] = config_name
                all_counts.append(edge_counts)
        except Exception as e:
            print(f"  - WARNING: Failed to process {file_path.name}: {e}")

    if not all_counts:
        print("!!! ERROR: No processable 'typed_edges-....csv' files found.")
        return

    all_counts_df = pd.concat(all_counts, ignore_index=True)

    print(
        f"--> Found and processed {len(all_counts_df['config_name'].unique())} graph structures."
    )

    # --- 3. Generate the final comparison plot ---
    output_filename = output_dir / f"{dataset_name}.png"
    plot_incremental_comparison(
        all_counts_df,
        f"Incremental Edge Comparison for: {dataset_name.upper()}",
        output_filename,
    )

    print("\n" + "=" * 50)
    print("      Analysis Complete!")
    print("=" * 50)
    print(f"-> Comparison plot saved to: {output_filename}")
    print("=" * 50)


if __name__ == "__main__":
    explore_and_visualize_structures()
