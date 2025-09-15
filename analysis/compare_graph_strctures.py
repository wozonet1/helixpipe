import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# --- Add project root to path to allow importing research_template ---
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root.parent / "common_utils"))
import research_template as rt


def plot_edge_distribution(ax, edges_df, title):
    """
    Helper function to generate a pie chart for edge type distribution on a given Axes.
    """
    if edges_df.empty:
        ax.text(
            0.5,
            0.5,
            "No Edges Found",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=12,
        )
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
        return

    # 1. Calculate counts and percentages for each edge type
    edge_counts = edges_df["edge_type"].value_counts()

    # 2. Define colors and explosion for small slices
    colors = plt.cm.Paired.colors
    # Explode the smaller slices to make them more visible
    # We'll "explode" any slice that is less than 5% of the total
    explode = [0.1 if (count / sum(edge_counts)) < 0.05 else 0 for count in edge_counts]

    # 3. Create a custom label formatter for the legend
    # e.g., "protein_protein_similarity (15 edges, 0.2%)"
    def legend_label_formatter(label, count, total):
        percentage = 100 * count / total
        return f"{label} ({count} edges, {percentage:.1f}%)"

    legend_labels = [
        legend_label_formatter(label, count, sum(edge_counts))
        for label, count in edge_counts.items()
    ]

    # 4. Plot the pie chart WITHOUT internal labels first
    wedges, texts = ax.pie(
        edge_counts,
        startangle=90,
        colors=colors,
        explode=explode,
        # Define wedge properties to add a border
        wedgeprops=dict(width=0.4, edgecolor="w"),
    )

    # 5. Add a clean legend to the side
    ax.legend(
        wedges,
        legend_labels,
        title="Edge Types & Distribution",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        fontsize="medium",
    )

    # 6. Set the title and ensure the pie is a circle
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.


def analyze_and_compare_structures():
    """
    Analyzes and compares the edge type distribution of the 'baseline' and 'gtopdb'
    variants for the primary dataset defined in the config.
    """
    print("--- Starting Graph Structure Comparison Analysis ---")

    config = rt.load_config()
    primary_dataset = config["data"]["primary_dataset"]

    # --- DataFrames to hold the edge data for both variants ---
    baseline_edges_df = None
    gtopdb_edges_df = None

    # --- Load Baseline Data ---
    print(f"--> Loading 'baseline' variant for dataset '{primary_dataset}'...")
    try:
        # We need to temporarily set the config flag to load the correct path
        config["data"]["use_gtopdb"] = False
        baseline_path = rt.get_path(
            config, f"{primary_dataset}.processed.typed_edge_list_template"
        )
        baseline_edges_df = pd.read_csv(baseline_path)
        print(f"    - Loaded {len(baseline_edges_df)} edges from {baseline_path.name}")
    except FileNotFoundError:
        print("    - WARNING: Baseline typed_edges file not found. Skipping.")

    # --- Load GtoPdb Data ---
    print(f"--> Loading 'gtopdb' variant for dataset '{primary_dataset}'...")
    try:
        config["data"]["use_gtopdb"] = True
        gtopdb_path = rt.get_path(
            config, f"{primary_dataset}.processed.typed_edge_list_template"
        )
        gtopdb_edges_df = pd.read_csv(gtopdb_path)
        print(f"    - Loaded {len(gtopdb_edges_df)} edges from {gtopdb_path.name}")
    except FileNotFoundError:
        print("    - WARNING: GtoPdb typed_edges file not found. Skipping.")

    # --- Create the comparison plot ---
    print("--> Generating comparison plot...")

    # Create a figure with two subplots (side-by-side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(
        f"Graph Structure Comparison for Dataset: {primary_dataset.upper()}",
        fontsize=20,
        fontweight="bold",
    )

    # Plot for the baseline variant
    if baseline_edges_df is not None:
        plot_edge_distribution(ax1, baseline_edges_df, "Baseline Variant")

    # Plot for the gtopdb variant
    if gtopdb_edges_df is not None:
        plot_edge_distribution(ax2, gtopdb_edges_df, "GtoPdb-Enhanced Variant")

    # Adjust layout to prevent labels from overlapping
    plt.tight_layout(
        rect=[0, 0, 0.85, 0.95]
    )  # Adjust rect to make space for the legend

    # Save the plot
    output_filename = (
        project_root / "analysis" / f"structure_comparison_{primary_dataset}.png"
    )
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")

    print("\n" + "=" * 50)
    print("      Analysis Complete!")
    print("=" * 50)
    print(f"-> Comparison plot saved to: {output_filename}")
    print("=" * 50)


if __name__ == "__main__":
    analyze_and_compare_structures()
