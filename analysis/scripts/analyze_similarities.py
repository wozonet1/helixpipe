import numpy as np
import pandas as pd
import pickle as pkl
from pathlib import Path
import hydra
from omegaconf import DictConfig

import research_template as rt

# 导入我们新的绘图工具
from .plot_utils import plot_histogram

# 确保解析器被注册
rt.register_hydra_resolvers()


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def analyze_similarity_distributions(cfg: DictConfig):
    """
    【V2 重构版】分析并可视化给定实验配置下的相似性分布。
    此脚本完全由传入的config驱动。
    """
    print("\n" + "=" * 80)
    print(" " * 15 + "STARTING SIMILARITY DISTRIBUTION ANALYSIS (V2)")
    print("=" * 80)

    # --- 1. 设置路径和加载数据 ---
    print("--- [Step 1/3] Loading nodes metadata and similarity matrices... ---")

    try:
        # 使用get_path和正确的file_key来加载文件
        nodes_df = pd.read_csv(
            rt.get_path(cfg, "data_structure.paths.processed.common.nodes_metadata")
        )
        mol_sim_path = rt.get_path(
            cfg, "data_structure.paths.processed.common.similarity_matrices.molecule"
        )
        prot_sim_path = rt.get_path(
            cfg, "data_structure.paths.processed.common.similarity_matrices.protein"
        )

        mol_sim_matrix = pkl.load(open(mol_sim_path, "rb"))
        prot_sim_matrix = pkl.load(open(prot_sim_path, "rb"))

    except FileNotFoundError as e:
        print(f"❌ FATAL: A required data file was not found: {e.filename}")
        print(
            "   Please ensure you have successfully run the main data processing pipeline (`run.py`)"
        )
        print(
            f"   with the corresponding configuration (e.g., `data_structure={cfg.data_structure.name}`, `data_params={cfg.data_params.name}`)."
        )
        return

    # --- 2. 从 nodes.csv 中解析节点类型和ID范围 ---
    # 这是取代旧的 ...2index.pkl 文件的关键步骤
    print("\n--- [Step 2/3] Parsing node types and ID ranges from nodes.csv... ---")

    # 筛选出不同类型的节点
    drug_nodes = nodes_df[nodes_df["node_type"] == "drug"]
    ligand_nodes = nodes_df[nodes_df["node_type"] == "ligand"]

    # 获取它们的局部索引范围 (相对于分子矩阵)
    # 假设 nodes.csv 是按 global_id 排序的，并且 drug 在 ligand 之前
    drug_indices = drug_nodes.index.to_numpy()
    # ligand的局部索引是它们的全局ID减去drug的数量
    num_drugs = len(drug_nodes)
    ligand_indices = ligand_nodes.index.to_numpy()

    print(f"--> Found {len(drug_nodes)} drugs and {len(ligand_nodes)} ligands.")

    # --- 3. 创建并准备输出目录 ---
    # 目录名将反映当前的配置
    plot_dir_name = f"{cfg.data_structure.name}-{cfg.data_params.name}"
    output_dir = Path("analysis/plots") / plot_dir_name
    print(f"--> Plots will be saved in: {output_dir}")

    # --- 4. 提取、分析并绘图 ---
    print("\n--- [Step 3/3] Extracting and plotting similarity distributions... ---")

    # a) Drug-Drug Similarity (D-D)
    if num_drugs > 1:
        # 使用 np.ix_ 来选择矩阵的子块
        dd_matrix_view = mol_sim_matrix[np.ix_(drug_indices, drug_indices)]
        dd_sims = dd_matrix_view[np.triu_indices_from(dd_matrix_view, k=1)]
        plot_histogram(
            dd_sims,
            f"Drug-Drug Similarity\n(Dataset: {cfg.data_structure.name}, Params: {cfg.data_params.name})",
            "Embedding Cosine Similarity",
            output_dir / "dd_similarity.png",
        )

    # b) Protein-Protein Similarity (P-P)
    pp_sims = prot_sim_matrix[np.triu_indices_from(prot_sim_matrix, k=1)]
    plot_histogram(
        pp_sims,
        f"Protein-Protein Similarity\n(Dataset: {cfg.data_structure.name}, Params: {cfg.data_params.name})",
        "Embedding Cosine Similarity",
        output_dir / "pp_similarity.png",
    )

    # c) Ligand-Ligand Similarity (L-L)
    if len(ligand_nodes) > 1:
        ll_matrix_view = mol_sim_matrix[np.ix_(ligand_indices, ligand_indices)]
        ll_sims = ll_matrix_view[np.triu_indices_from(ll_matrix_view, k=1)]
        plot_histogram(
            ll_sims,
            f"Ligand-Ligand Similarity\n(Dataset: {cfg.data_structure.name}, Params: {cfg.data_params.name})",
            "Embedding Cosine Similarity",
            output_dir / "ll_similarity.png",
        )

    # d) Drug-Ligand Similarity (D-L)
    if num_drugs > 0 and len(ligand_nodes) > 0:
        dl_matrix_view = mol_sim_matrix[np.ix_(drug_indices, ligand_indices)]
        dl_sims = dl_matrix_view.flatten()
        plot_histogram(
            dl_sims,
            f"Drug-Ligand Similarity\n(Dataset: {cfg.data_structure.name}, Params: {cfg.data_params.name})",
            "Embedding Cosine Similarity",
            output_dir / "dl_similarity.png",
        )

    print("\n" + "=" * 80)
    print(" " * 25 + "ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    analyze_similarity_distributions()
