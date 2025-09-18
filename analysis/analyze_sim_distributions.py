# analysis/analyze_similarity_distributions.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import research_template as rt
import hydra
from omegaconf import DictConfig
import pickle as pkl


# --- Helper Functions for Plotting ---
def plot_histogram(data: np.ndarray, title: str, xlabel: str, output_path: Path):
    """Generic function to plot and save a distribution histogram."""
    if data is None or len(data) == 0:
        print(f"--> No data to plot for '{title}'. Skipping.")
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))

    sns.histplot(data, bins=50, kde=True, ax=ax, color="skyblue", edgecolor="black")

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
    plt.savefig(output_path, dpi=300)
    plt.close(fig)  # Close the figure to free up memory
    print(f"--> Distribution plot saved to: {output_path}")


# --- Main Analysis Logic ---
@hydra.main(config_path="../conf", config_name="config", version_base=None)
def analyze_all_similarities(cfg: DictConfig):
    """
    Analyzes and visualizes the distribution of all similarity relationships
    (D-D, P-P, L-L, D-L) for a given dataset configuration.
    """
    print("--- Starting Full Similarity Distribution Analysis ---")

    # --- 1. Setup Paths and Config ---
    project_root = Path(hydra.utils.get_original_cwd())
    dataset_name = cfg.data.primary_dataset

    output_dir = project_root / "analysis" / "similarity_distributions" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"--> Plots will be saved in: {output_dir}")

    # --- 2. Load Similarity Matrices ---
    # We will load matrices from the 'gtopdb' variant as it contains all node types.
    cfg.data.use_gtopdb = True
    try:
        print("--> Loading pre-computed similarity matrices...")
        mol_sim_matrix = pkl.load(
            open(rt.get_path(cfg, "processed.similarity_matrices.molecule"), "rb")
        )
        prot_sim_matrix = pkl.load(
            open(rt.get_path(cfg, "processed.similarity_matrices.protein"), "rb")
        )

        # Load indexes to understand the matrix structure
        drug2index = pkl.load(open(rt.get_path(cfg, "processed.indexes.drug"), "rb"))
        ligand2index = pkl.load(
            open(rt.get_path(cfg, "processed.indexes.ligand"), "rb")
        )
    except FileNotFoundError as e:
        print(f"\n!!! ERROR: Could not find necessary file: {e.filename}")
        print("    Please run `data_proc.py` with `data.use_gtopdb=true` first.")
        return

    num_drugs = len(drug2index)
    num_ligands = len(ligand2index)

    # --- 3. Extract and Plot Distributions ---
    # We extract the upper triangle of the matrices to avoid duplicates and self-similarity.

    # a) Drug-Drug Similarity (D-D)
    dd_matrix_view = mol_sim_matrix[:num_drugs, :num_drugs]
    dd_sims = dd_matrix_view[np.triu_indices_from(dd_matrix_view, k=1)]
    plot_histogram(
        dd_sims,
        f"Drug-Drug Similarity Distribution for {dataset_name}",
        "Tanimoto Similarity",
        output_dir / "dd_similarity.png",
    )

    # b) Protein-Protein Similarity (P-P)
    pp_sims = prot_sim_matrix[np.triu_indices_from(prot_sim_matrix, k=1)]
    plot_histogram(
        pp_sims,
        f"Protein-Protein Similarity Distribution for {dataset_name}",
        "Normalized Alignment Score",
        output_dir / "pp_similarity.png",
    )

    if num_ligands > 0:
        # c) Ligand-Ligand Similarity (L-L)
        ll_matrix_view = mol_sim_matrix[num_drugs:, num_drugs:]
        ll_sims = ll_matrix_view[np.triu_indices_from(ll_matrix_view, k=1)]
        plot_histogram(
            ll_sims,
            f"Ligand-Ligand Similarity Distribution for {dataset_name}",
            "Tanimoto Similarity",
            output_dir / "ll_similarity.png",
        )

        # d) Drug-Ligand Similarity (D-L)
        dl_matrix_view = mol_sim_matrix[:num_drugs, num_drugs:]
        dl_sims = dl_matrix_view.flatten()  # All pairs are relevant here
        plot_histogram(
            dl_sims,
            f"Drug-Ligand Similarity Distribution for {dataset_name}",
            "Tanimoto Similarity",
            output_dir / "dl_similarity.png",
        )

    print("\n" + "=" * 50)
    print("      Analysis Complete!")
    print("=" * 50)


if __name__ == "__main__":
    analyze_all_similarities()
