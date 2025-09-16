import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import sys
from pathlib import Path
import research_template as rt

# --- Add project root to path to allow importing research_template ---
# This makes the script runnable from the `analysis/` directory
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root.parent / "common_utils"))


def analyze_drug_ligand_similarity():
    """
    Analyzes and visualizes the Tanimoto similarity distribution
    between drugs from the primary dataset and ligands from GtoPdb.
    """
    print("--- Starting Drug-Ligand Similarity Analysis ---")

    # 1. Load configuration and paths
    config = rt.load_config()
    primary_dataset = config["data"]["primary_dataset"]
    # We need the 'gtopdb' variant of the processed data for this analysis
    config["data"]["use_gtopdb"] = True

    try:
        drug_index_path = rt.get_path(
            config, f"{primary_dataset}.processed.indexes.drug"
        )
        ligand_index_path = rt.get_path(
            config, f"{primary_dataset}.processed.indexes.ligand"
        )

        drug2index = pd.read_pickle(drug_index_path)
        ligand2index = pd.read_pickle(ligand_index_path)
    except FileNotFoundError:
        print("\n!!! ERROR: Index files for 'gtopdb' variant not found.")
        print(
            "    Please run `data_proc.py` with `use_gtopdb: true` in your config first."
        )
        return
    except Exception as e:
        print(f"\n!!! An unexpected error occurred: {e}")
        return

    drug_smiles = list(drug2index.keys())
    ligand_smiles = list(ligand2index.keys())

    if not drug_smiles or not ligand_smiles:
        print("--> No drugs or ligands found to compare. Exiting.")
        return

    print(f"--> Found {len(drug_smiles)} drugs and {len(ligand_smiles)} ligands.")

    # 2. Generate Morgan fingerprints for all molecules
    print("--> Generating Morgan fingerprints for all molecules...")
    drug_mols = [Chem.MolFromSmiles(s) for s in drug_smiles]
    ligand_mols = [Chem.MolFromSmiles(s) for s in ligand_smiles]

    # Filter out any invalid molecules
    drug_mols = [m for m in drug_mols if m is not None]
    ligand_mols = [m for m in ligand_mols if m is not None]

    fpgen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    drug_fps = [fpgen.GetFingerprint(m) for m in drug_mols]
    ligand_fps = [fpgen.GetFingerprint(m) for m in ligand_mols]

    # 3. Calculate all pairwise D-L similarities
    print(
        "--> Calculating pairwise Drug-Ligand similarities (this may take a while)..."
    )
    dl_similarities = []

    for drug_fp in tqdm(drug_fps, desc="Comparing Drugs to Ligands"):
        # DataStructs.BulkTanimotoSimilarity is highly optimized
        sims = DataStructs.BulkTanimotoSimilarity(drug_fp, ligand_fps)
        dl_similarities.extend(sims)

    dl_similarities = np.array(dl_similarities)

    # 4. Plot the distribution histogram
    print("--> Plotting the similarity distribution histogram...")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))

    sns.histplot(
        dl_similarities, bins=50, kde=True, ax=ax, color="steelblue", edgecolor="black"
    )

    # Add annotations and titles
    mean_sim = np.mean(dl_similarities)
    median_sim = np.median(dl_similarities)
    max_sim = np.max(dl_similarities)

    ax.axvline(mean_sim, color="red", linestyle="--", label=f"Mean: {mean_sim:.3f}")
    ax.axvline(
        median_sim, color="green", linestyle="-", label=f"Median: {median_sim:.3f}"
    )

    ax.set_title(
        f"Distribution of Tanimoto Similarities between {len(drug_mols)} Drugs and {len(ligand_mols)} Ligands",
        fontsize=16,
    )
    ax.set_xlabel("Tanimoto Similarity", fontsize=12)
    ax.set_ylabel("Frequency (Count of Pairs)", fontsize=12)
    ax.legend()

    # Add text box for summary statistics
    stats_text = f"Max Similarity: {max_sim:.3f}\nTotal Pairs: {len(dl_similarities):,}"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(
        0.65,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )

    # Save the plot
    output_filename = (
        project_root / "analysis" / f"dl_similarity_distribution_{primary_dataset}.png"
    )
    plt.savefig(output_filename, dpi=300)

    print("\n" + "=" * 50)
    print("      Analysis Complete!")
    print("=" * 50)
    print("-> Summary Statistics:")
    print(f"   - Maximum Similarity Found: {max_sim:.4f}")
    print(f"   - Mean Similarity: {mean_sim:.4f}")
    print(f"   - Median Similarity: {median_sim:.4f}")
    print(f"-> The distribution plot has been saved to: {output_filename}")
    print("=" * 50)


if __name__ == "__main__":
    # Before running, ensure you have matplotlib and seaborn installed
    # pip install matplotlib seaborn
    analyze_drug_ligand_similarity()
    print("--- Finished Drug-Ligand Similarity Analysis ---")
