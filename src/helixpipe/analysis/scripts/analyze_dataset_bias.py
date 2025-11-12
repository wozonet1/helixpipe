from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import research_template as rt
import seaborn as sns
from bioservices import UniProt
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors
from tqdm import tqdm

# 导入所有需要的项目内部模块
from helixpipe.configs import AppConfig, register_all_schemas
from helixpipe.utils import get_path, register_hydra_resolvers

from .plot_utils import (
    plot_bar_chart_with_counts,
)  # 假设我们在plot_utils中创建一个新的绘图函数

# 在所有Hydra操作之前，执行全局注册
register_all_schemas()
register_hydra_resolvers()

# --- 分析函数 ---


# FIXME: 调整API
def analyze_target_family_distribution(nodes_df: pd.DataFrame, output_dir: Path):
    """
    分析数据集中靶点蛋白质的家族分布。
    """
    print("\n--- [Bias Analysis] Analyzing Target Protein Family Distribution...")

    protein_nodes = nodes_df[nodes_df["node_type"] == "protein"]
    uniprot_ids = protein_nodes["authoritative_id"].tolist()

    if not uniprot_ids:
        print("    - No protein nodes found. Skipping.")
        return

    print(
        f"    - Querying UniProt for {len(uniprot_ids)} proteins... (This may take a while)"
    )

    # 使用 bioservices 连接 UniProt
    u = UniProt()
    # UniProt API 限制每次查询500个ID
    batch_size = 500
    all_results = []

    for i in tqdm(range(0, len(uniprot_ids), batch_size), desc="    - UniProt Batches"):
        batch_ids = uniprot_ids[i : i + batch_size]
        try:
            # 查询Pfam数据库的注释信息
            results = u.mapping(
                fr="UniProtKB_AC-ID", to="Pfam", query=",".join(batch_ids)
            )
            all_results.extend(results.get("results", []))
        except Exception as e:
            print(f"    - WARNING: UniProt API call failed for a batch. Error: {e}")

    if not all_results:
        print("    - Could not retrieve any Pfam family information from UniProt.")
        return

    # 解析结果
    protein_to_family = {}
    for res in all_results:
        protein_id = res["from"]
        if "to" in res and res["to"]:
            # 我们只取第一个Pfam家族作为代表
            family_name = res["to"]["name"]
            protein_to_family[protein_id] = family_name

    family_series = pd.Series(protein_to_family, name="family")

    # 统计并可视化
    family_counts = family_series.value_counts()

    # 为了图表清晰，我们只展示Top N的家族，其余归为 "Other"
    top_n = 20
    if len(family_counts) > top_n:
        top_families = family_counts.head(top_n)
        other_count = family_counts.iloc[top_n:].sum()
        top_families["Other"] = other_count
        plot_data = top_families
    else:
        plot_data = family_counts

    print("\n    - Top Protein Families:")
    print(plot_data)

    # 绘图
    plot_bar_chart_with_counts(
        data=plot_data,
        output_path=output_dir / "target_family_distribution.png",
        title="Distribution of Target Protein Families",
        xlabel="Protein Family (Pfam)",
        ylabel="Number of Proteins",
    )


def analyze_chemical_space_coverage(nodes_df: pd.DataFrame, output_dir: Path):
    """
    分析数据集中分子的化学空间覆盖度。
    """
    print("\n--- [Bias Analysis] Analyzing Chemical Space Coverage...")

    mol_nodes = nodes_df[
        (nodes_df["node_type"] == "drug") | (nodes_df["node_type"] == "ligand")
    ]
    smiles_list = mol_nodes["structure"].dropna().tolist()

    if not smiles_list:
        print("    - No molecule nodes with SMILES found. Skipping.")
        return

    print(f"    - Calculating properties for {len(smiles_list)} molecules...")

    properties = []
    for smiles in tqdm(smiles_list, desc="    - RDKit Calculations"):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            properties.append(
                {
                    "MW": Descriptors.MolWt(mol),
                    "LogP": Crippen.MolLogP(mol),
                    "TPSA": Descriptors.TPSA(mol),
                    "NumRings": Descriptors.RingCount(mol),
                }
            )

    props_df = pd.DataFrame(properties)

    print("\n    - Chemical Property Statistics:")
    print(props_df.describe())

    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        "Distribution of Molecule Physicochemical Properties",
        fontsize=18,
        fontweight="bold",
    )

    sns.histplot(props_df["MW"], bins=50, kde=True, ax=axes[0, 0]).set_title(
        "Molecular Weight (MW)"
    )
    sns.histplot(props_df["LogP"], bins=50, kde=True, ax=axes[0, 1]).set_title("LogP")
    sns.histplot(props_df["TPSA"], bins=50, kde=True, ax=axes[1, 0]).set_title(
        "Topological Polar Surface Area (TPSA)"
    )
    sns.histplot(
        props_df["NumRings"],
        bins=np.arange(props_df["NumRings"].max() + 2) - 0.5,
        ax=axes[1, 1],
    ).set_title("Number of Rings")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = output_dir / "chemical_space_coverage.png"
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"\n    - Chemical space distribution plot saved to: {output_path.name}")


# --- 主函数 ---


@hydra.main(
    config_path=str(rt.get_project_root() / "conf"),
    config_name="config",
    version_base=None,
)
def main(cfg: AppConfig):
    """
    一个独立的、由Hydra驱动的脚本，用于分析数据集的偏见与覆盖度。
    """
    print("\n" + "=" * 80)
    print(" " * 20 + "STARTING DATASET BIAS & COVERAGE ANALYSIS")
    print("=" * 80)

    try:
        # --- 步骤 1: 加载核心数据产物 ---
        print("--- [Step 1/2] Loading node metadata...")

        nodes_df = pd.read_csv(get_path(cfg, "processed.common.nodes_metadata"))
        print(f"    - Successfully loaded metadata for {len(nodes_df)} nodes.")

        # --- 步骤 2: 准备输出目录 ---
        output_dir = (
            rt.get_project_root()
            / "analysis_outputs"
            / (cfg.dataset_collection.name or "base")
            / cfg.data_params.name
            / "dataset_bias_analysis"
        )
        rt.ensure_path_exists(output_dir / "summary.txt")

        # --- 步骤 3: 执行各项分析 ---
        analyze_target_family_distribution(nodes_df, output_dir)
        analyze_chemical_space_coverage(nodes_df, output_dir)

        print("\n" + "=" * 80)
        print(f"✅ BIAS ANALYSIS COMPLETE. All outputs saved to: {output_dir}")
        print("=" * 80)

    except FileNotFoundError as e:
        print(
            f"❌ FATAL: A required data file was not found: {e.filename if hasattr(e, 'filename') else e}"
        )
        print(
            "   Please ensure you have successfully run the main data processing pipeline (`run.py`) to generate 'nodes.csv'."
        )
        return
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
