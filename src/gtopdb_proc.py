# src/gtopdb_proc.py

import pandas as pd
import sys
from tqdm import tqdm
import time
from pathlib import Path
import requests
import re


def is_valid_uniprot_accession(accession):
    """
    使用正则表达式，快速检查一个ID是否符合UniProt Accession的典型格式。
    这是一个简化版检查，但能过滤掉大部分非Accession的ID。
    """
    # 典型的UniProt Accession格式: e.g., P12345, Q9Y261, A0A024R1R8
    # 规则: 字母开头，后面跟5个或更多数字/字母
    # [OPQ][0-9][A-Z0-9]{3}[0-9] | [A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}
    # 我们用一个简化的版本
    pattern = re.compile(
        r"^[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9]$|^[OPQ][0-9][A-Z0-9]{3}[0-9]$",
        re.IGNORECASE,
    )
    return bool(pattern.match(str(accession)))


def fetch_sequences_from_uniprot(uniprot_ids):
    """
    【最终版】
    根据UniProt ID列表，使用requests库直接调用UniProt官方API，批量获取蛋白质序列。
    这是一个更现代、更健壮的实现。
    """
    valid_ids = sorted(
        [uid for uid in set(uniprot_ids) if is_valid_uniprot_accession(uid)]
    )
    invalid_ids = set(uniprot_ids) - set(valid_ids)
    if invalid_ids:
        print(
            f"Warning: Skipped {len(invalid_ids)} IDs with non-standard format. Examples: {list(invalid_ids)[:5]}"
        )

    print(f"Fetching sequences for {len(valid_ids)} valid UniProt IDs...")

    base_url = "https://rest.uniprot.org/uniprotkb/stream"
    sequences_map = {}
    chunk_size = 100

    for i in tqdm(range(0, len(valid_ids), chunk_size), desc="Querying UniProt API"):
        chunk = valid_ids[i : i + chunk_size]

        params = {
            "query": " OR ".join(f"(accession:{acc})" for acc in chunk),
            "format": "fasta",
        }

        try:
            response = requests.get(base_url, params=params)

            # 检查请求是否成功
            if response.status_code == 200:
                fasta_text = response.text

                # 解析返回的FASTA文本 (这部分逻辑和之前一样)
                for entry in fasta_text.strip().split(">"):
                    if not entry.strip():
                        continue
                    lines = entry.strip().split("\n")
                    header = lines[0]
                    seq = "".join(lines[1:])

                    try:
                        uid = header.split("|")[1]
                        sequences_map[uid] = seq
                    except IndexError:
                        print(
                            f"\nWarning: Could not parse UniProt ID from header: '{header}'"
                        )
            elif response.status_code == 400 and len(chunk) > 1:
                print(
                    f"\nWarning: Batch request failed (400 Bad Request). Switching to individual retry for {len(chunk)} IDs..."
                )
                for single_id in tqdm(chunk, desc="Retrying individually", leave=False):
                    single_params = {
                        "query": f"(accession:{single_id})",
                        "format": "fasta",
                    }
                    try:
                        single_response = requests.get(
                            base_url, params=single_params, timeout=10
                        )
                        if single_response.status_code == 200:
                            s_fasta = single_response.text
                            if s_fasta and s_fasta.startswith(">"):
                                s_lines = s_fasta.strip().split("\n")
                                s_header, s_seq = s_lines[0], "".join(s_lines[1:])
                                s_uid = s_header.split("|")[1]
                                sequences_map[s_uid] = s_seq
                        else:
                            print(
                                f"-> Failed for single ID: {single_id} (Status: {single_response.status_code})"
                            )
                    except Exception as single_e:
                        print(
                            f"-> Network/Parse error for single ID {single_id}: {single_e}"
                        )
                    time.sleep(0.2)  # 单个查询之间也稍作等待

            else:
                # 如果请求失败，打印出错误状态码
                print(
                    f"\nWarning: UniProt API request failed for chunk starting with {chunk[0]}. Status code: {response.status_code}"
                )

        except requests.exceptions.RequestException as e:
            # 捕获网络层面的错误
            print("\n--- NETWORK ERROR during UniProt fetch ---")
            print(f"Error: {e}")
            print("------------------------------------------")

        time.sleep(1)  # 保持API礼仪

    print(f"-> FINAL: Successfully fetched {len(sequences_map)} sequences.")
    return sequences_map


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
            input_dir / "interactions.csv", low_memory=False, comment="#"
        )
        ligands_df = pd.read_csv(
            input_dir / "ligands.csv", low_memory=False, comment="#"
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
    final_df = pd.merge(
        endogenous_interactions,
        relevant_ligands,
        left_on="Ligand ID",
        right_on="Ligand ID",
    )
    print("\n[NEW Step] Fetching protein sequences for all relevant targets...")

    # 1. 从合并后的数据中，收集所有需要查询序列的唯一UniProt ID
    #    先处理'|'分隔符，然后展开列表，最后取唯一值
    all_target_uniprot_ids = (
        final_df["Target UniProt ID"]
        .str.split("|")
        .explode()
        .dropna()
        .unique()
        .tolist()
    )

    # 2. 调用我们写的函数，去UniProt“补全”序列信息
    uniprot_to_sequence_map = fetch_sequences_from_uniprot(all_target_uniprot_ids)

    # 3. 将查询到的序列映射回我们的主DataFrame
    #    我们只关心主要ID（第一个ID）的序列
    main_uniprot_id = final_df["Target UniProt ID"].str.split("|").str[0]
    final_df["target_sequence"] = main_uniprot_id.map(uniprot_to_sequence_map)

    # 4. 清洗：丢弃那些因为某些原因没能查到序列的记录
    original_count = len(final_df)
    final_df.dropna(subset=["target_sequence"], inplace=True)
    print(
        f"-> Found sequences for {len(final_df)} out of {original_count} interactions. Proceeding with these."
    )

    # --- 5. 保存最终的输出文件 (现在输出的文件信息更完整了) ---
    print("\n[Final Step] Saving enriched data to output files...")

    # a) 保存新的 P-L 边文件，现在包含【序列】而不是ID
    # 文件格式: protein_sequence, ligand_smiles, affinity_median
    output_edges = final_df[["target_sequence", "SMILES", "Affinity Median"]].copy()
    output_edges.rename(
        columns={"target_sequence": 0, "SMILES": 1, "Affinity Median": 2}, inplace=True
    )

    output_edge_path = output_dir / "gtopdb_p-l_edges.csv"
    output_edges.to_csv(output_edge_path, index=False, header=False)
    print(
        f"-> Successfully saved {len(output_edges)} Protein-Ligand edges (with sequences and SMILES) to: {output_edge_path}"
    )

    # b) 保存新的配体信息文件 (可以保持不变，也可以只保存一次)
    output_ligands = final_df[["PubChem CID", "SMILES"]].drop_duplicates()
    output_ligands.rename(columns={"PubChem CID": 0, "SMILES": 1}, inplace=True)
    output_ligand_path = output_dir / "gtopdb_ligands.csv"
    output_ligands.to_csv(output_ligand_path, index=False, header=False)
    print(
        f"-> Successfully saved info for {len(output_ligands)} Ligand nodes to: {output_ligand_path}"
    )

    # c) (推荐) 另外保存一份蛋白质的 "ID-序列" 对应表，以备后用
    output_proteins = final_df[
        ["Target UniProt ID", "target_sequence"]
    ].drop_duplicates()
    output_proteins["Target UniProt ID"] = (
        output_proteins["Target UniProt ID"].str.split("|").str[0]
    )
    output_proteins.rename(
        columns={"Target UniProt ID": 0, "target_sequence": 1}, inplace=True
    )
    output_protein_path = output_dir / "gtopdb_proteins.csv"
    output_proteins.to_csv(output_protein_path, index=False, header=False)
    print(
        f"-> Successfully saved info for {len(output_proteins)} Protein nodes to: {output_protein_path}"
    )

    print("\n==========================================================")
    print("  GtoPdb Processing Finished Successfully!  ")
    print("==========================================================")


if __name__ == "__main__":
    # --- 配置区 ---
    input_directory = Path("../data/gtopdb/raw")
    output_directory = Path("../data/gtopdb/processed")

    # --- 执行主函数 ---
    process_gtopdb_data(input_directory, output_directory)
