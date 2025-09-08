# src/gtopdb_proc.py

import pandas as pd
import sys


def process_gtopdb_data(input_dir: str, output_dir: str):
    """
    解析Guide to PHARMACOLOGY的原始数据文件,提取内源性配体-靶点相互作用，
    并生成可用于下游异构网络构建的干净文件。

    参数:
        input_dir (Path): 存放下载的GtoPdb原始CSV文件的目录路径。
        output_dir (Path): 存放处理后输出文件的目录路径。
    """
    print("==========================================================")
    print("  Starting Guide to PHARMACOLOGY Data Processing Pipeline  ")
    print("==========================================================")

    # --- 1. 加载核心数据文件 ---
    print(f"\n[Step 1/4] Loading raw data files from: {input_dir}")
    try:
        interactions_df = pd.read_csv(
            input_dir + "/interactions.csv", low_memory=False, comment="#"
        )
        ligands_df = pd.read_csv(
            input_dir + "/ligands.csv", low_memory=False, comment="#"
        )
    except FileNotFoundError as e:
        print(f"Error: Raw CSV file not found! {e}")
        print(
            "Please ensure 'interactions.csv' and 'ligands.csv' are in the specified input directory."
        )
        sys.exit(1)  # 退出程序

    print(f"-> Loaded {len(interactions_df)} total interactions.")
    print(f"-> Loaded {len(ligands_df)} total ligands.")

    # --- 2. 筛选内源性配体相互作用 ---
    print("\n[Step 2/4] Filtering for high-quality endogenous ligand interactions...")

    # 核心筛选条件：'endogenous' 列为 True
    print(interactions_df.columns)
    print(interactions_df["Endogenous"].unique())
    endogenous_interactions = interactions_df[interactions_df["Endogenous"]].copy()
    print(
        f"-> Found {len(endogenous_interactions)} interactions involving endogenous ligands."
    )

    # 数据清洗：只保留有UniProt ID、配体ID和亲和力数值的关键记录
    required_cols = ["Target UniProt ID", "Ligand ID", "Affinity Median"]
    endogenous_interactions.dropna(subset=required_cols, inplace=True)
    print(
        f"-> After cleaning (removing entries with missing IDs or affinity), {len(endogenous_interactions)} interactions remain."
    )

    # --- 3. 提取并清洗配体信息 (获取SMILES) ---
    print("\n[Step 3/4] Extracting SMILES for the relevant endogenous ligands...")

    # 获取我们需要的配体的唯一ID列表
    relevant_ligand_ids = endogenous_interactions["Ligand ID"].unique()

    # 从大的配体表中，只筛选出我们需要的配体信息
    relevant_ligands = ligands_df[
        ligands_df["Ligand ID"].isin(relevant_ligand_ids)
    ].copy()

    # 清洗配体数据：只保留有SMILES和PubChem CID的记录
    relevant_ligands.dropna(subset=["SMILES", "PubChem CID"], inplace=True)
    relevant_ligands["PubChem CID"] = relevant_ligands["PubChem CID"].astype(
        int
    )  # 确保CID是整数
    print(f"-> Found SMILES for {len(relevant_ligands)} unique endogenous ligands.")

    # --- 4. 合并信息并保存为最终的输出文件 ---
    print("\n[Step 4/4] Merging data and saving to output files...")

    # 将相互作用数据与配体数据通过 'ligand_id' 连接起来
    final_edges = pd.merge(
        endogenous_interactions,
        relevant_ligands,
        left_on="Ligand ID",
        right_on="Ligand ID",
    )

    # a) 准备并保存新的 P-L 边文件
    # 文件格式: Target UniProt ID, ligand_pubchem_cid, Affinity Median
    output_edges = (
        final_edges[["Target UniProt ID", "PubChem CID", "Affinity Median"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    # GtoPdb的 'Target UniProt ID' 列可能包含多个以'|'分隔的ID，我们只取第一个
    output_edges["Target UniProt ID"] = (
        output_edges["Target UniProt ID"].str.split("|").str[0]
    )
    output_edges.rename(
        columns={"Target UniProt ID": 0, "PubChem CID": 1, "Affinity Median": 2},
        inplace=True,
    )

    output_edge_path = output_dir + "/gtopdb_p-l_edges.csv"
    output_edges.to_csv(output_edge_path, index=False, header=False)
    print(
        f"-> Successfully saved {len(output_edges)} new Protein-Ligand edges to: {output_edge_path}"
    )

    # b) 准备并保存新的配体信息文件
    # 文件格式: PubChem CID, SMILES
    output_ligands = (
        final_edges[["PubChem CID", "SMILES"]].drop_duplicates().reset_index(drop=True)
    )
    output_ligands.rename(columns={"PubChem CID": 0, "SMILES": 1}, inplace=True)

    output_ligand_path = output_dir + "/gtopdb_ligands.csv"
    output_ligands.to_csv(output_ligand_path, index=False, header=False)
    print(
        f"-> Successfully saved info for {len(output_ligands)} new Ligand nodes to: {output_ligand_path}"
    )

    print("\n==========================================================")
    print("  GtoPdb Processing Finished Successfully!  ")
    print("==========================================================")


if __name__ == "__main__":
    # --- 配置区 ---
    input_directory = "../data/gtopdb/raw"
    output_directory = "../data/gtopdb/processed"

    # --- 执行主函数 ---
    process_gtopdb_data(input_directory, output_directory)
