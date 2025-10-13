from rdkit import Chem
import pubchempy as pcp
from tqdm import tqdm
import time
import pickle as pkl
from pathlib import Path
import pandas as pd
import requests
from urllib.parse import urlencode


def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, canonical=True)
    else:
        print(f"Warning: Invalid SMILES string found and will be ignored: {smiles}")
        return None  # Return None for invalid SMILES


def canonicalize_smiles_to_cid(
    smiles_list: list[str], cache_path: Path = None
) -> dict[str, int]:
    """
    【核心标准化函数】将一个SMILES字符串列表，批量转换为PubChem CID。

    该函数实现了：
    1. 缓存机制：如果提供了cache_path且文件存在，则直接从缓存加载。
    2. 批量查询：将SMILES分批次提交给PubChem API，以提高效率。
    3. 延时与重试：遵守API使用礼仪，并在遇到临时网络问题时自动重试。
    4. 错误处理：优雅地处理无法被PubChem识别的无效SMILES。

    Args:
        smiles_list (list[str]): 待转换的SMILES字符串列表。
        cache_path (Path, optional): 指向缓存文件(.pkl)的路径。
                                     如果提供，函数会优先使用缓存。 Defaults to None.

    Returns:
        dict[str, int]: 一个从原始SMILES字符串，映射到其对应PubChem CID（整数）的字典。
                        无法转换的SMILES将不会出现在字典中。
    """
    # --- 1. 检查缓存 ---
    if cache_path and cache_path.exists():
        print(f"--> [Canonicalizer] Loading cached SMILES->CID map from: {cache_path}")
        with open(cache_path, "rb") as f:
            return pkl.load(f)

    print(
        "--> [Canonicalizer] No cache found. Starting SMILES to CID conversion via PubChem API..."
    )

    # 获取唯一的、非空的SMILES列表
    unique_smiles = sorted(list(set(s for s in smiles_list if s and pd.notna(s))))

    smiles_to_cid_map = {}
    not_found_smiles = []

    # PubChem PUG-REST API对请求频率有限制，批量和延时是必须的
    batch_size = 100  # 每次查询100个

    progress_bar = tqdm(
        range(0, len(unique_smiles), batch_size),
        desc="[Canonicalizer] Querying PubChem",
    )

    for i in progress_bar:
        batch_smiles = unique_smiles[i : i + batch_size]

        # --- 2. 批量查询与重试逻辑 ---
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # pcp.get_cids() 是一个强大的批量查询函数
                # 'smiles' 指定查询类型
                # 'name' 也可以，但'smiles'更精确
                results = pcp.get_cids(
                    batch_smiles, namespace="smiles", as_dataframe=True
                )

                # 处理查询成功的结果
                for smiles, row in results.iterrows():
                    # get_cids 返回的DataFrame的index就是原始的SMILES
                    # 我们取CID列的第一个值（通常也只有一个）
                    if not pd.isna(row["CID"]):
                        smiles_to_cid_map[smiles] = int(row["CID"])

                # 记录那些在PubChem中找不到的SMILES
                found_smiles = set(results.index)
                for smiles in batch_smiles:
                    if smiles not in found_smiles:
                        not_found_smiles.append(smiles)

                break  # 成功，跳出重试循环

            except (pcp.PubChemHTTPError, ConnectionError) as e:
                print(
                    f"\n   - WARNING: PubChem API error on attempt {attempt + 1}/{max_retries}: {e}"
                )
                if attempt < max_retries - 1:
                    wait_time = 2 ** (attempt + 1)  # 指数退避：2, 4, 8秒
                    print(f"     Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(
                        f"   - FATAL: Failed to query batch after {max_retries} attempts. Skipping this batch."
                    )
                    not_found_smiles.extend(batch_smiles)  # 整个batch都算作未找到

        # 遵守API礼仪，每次批量查询后都稍作等待
        time.sleep(0.2)

    print("--> [Canonicalizer] Conversion finished.")
    print(f"    - Successfully converted: {len(smiles_to_cid_map)} SMILES")
    if not_found_smiles:
        print(f"    - Could not find CID for: {len(not_found_smiles)} SMILES")
        # (可选) 可以将not_found_smiles列表写入一个日志文件，以供后续分析
        # with open("not_found_smiles.log", "a") as f:
        #     for smiles in not_found_smiles:
        #         f.write(f"{smiles}\n")

    # --- 3. 保存到缓存 ---
    if cache_path:
        print(f"--> [Canonicalizer] Saving new SMILES->CID map to cache: {cache_path}")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pkl.dump(smiles_to_cid_map, f)

    return smiles_to_cid_map


# ------------------- 蛋白质标准化核心函数 -------------------


def canonicalize_sequences_to_uniprot(
    sequence_list: list[str], cache_path: Path = None, force_regenerate: bool = False
) -> dict[str, str]:
    """
    【生产版】将蛋白质序列列表，批量转换为权威的UniProt Accession ID。

    该函数实现了：
    1. 缓存机制：优先从本地缓存加载。
    2. 批量提交：将序列分批次提交给UniProt ID Mapping API。
    3. 轮询与结果获取：遵循UniProt API的两阶段异步查询模式。
    4. 错误处理与API礼仪。

    Args:
        sequence_list (list[str]): 待转换的蛋白质氨基酸序列列表。
        cache_path (Path, optional): 指向缓存文件(.pkl)的路径。
        force_regenerate (bool): 如果为True，将忽略现有缓存，强制重新生成。

    Returns:
        dict[str, str]: 一个从原始序列，映射到其对应UniProt ID（字符串）的字典。
    """
    # --- 1. 检查缓存 ---
    if cache_path and cache_path.exists() and not force_regenerate:
        print(
            f"--> [Canonicalizer] Loading cached Sequence->UniProt map from: {cache_path}"
        )
        with open(cache_path, "rb") as f:
            return pkl.load(f)

    if force_regenerate:
        print(
            "--> [Canonicalizer] `force_regenerate` is True. Starting Seq->UniProt conversion..."
        )
    else:
        print(
            "--> [Canonicalizer] No cache found. Starting Seq->UniProt conversion via UniProt API..."
        )

    unique_sequences = sorted(list(set(s for s in sequence_list if s and pd.notna(s))))

    seq_to_uniprot_map = {}

    # UniProt API的最佳实践，同样是分批处理
    batch_size = 100  # UniProt建议的批量大小

    progress_bar = tqdm(
        range(0, len(unique_sequences), batch_size),
        desc="[Canonicalizer] Submitting jobs to UniProt",
    )

    for i in progress_bar:
        batch_seqs = unique_sequences[i : i + batch_size]

        # --- 2. UniProt API两阶段查询 ---
        # 阶段 A: 提交查询任务，并获取一个任务ID
        job_id = _submit_uniprot_id_mapping_job(batch_seqs)

        if job_id:
            # 阶段 B: 使用任务ID，轮询API直到任务完成，然后获取结果
            results = _get_uniprot_id_mapping_results(job_id)

            # 3. 解析结果
            for result in results:
                original_sequence = result["from"]
                uniprot_id = result["to"]["primaryAccession"]
                seq_to_uniprot_map[original_sequence] = uniprot_id

        # 每次任务后等待，以示API礼仪
        time.sleep(1)

    print(f"--> [Canonicalizer] Conversion finished.")
    print(f"    - Successfully mapped: {len(seq_to_uniprot_map)} sequences")
    print(
        f"    - Could not map: {len(unique_sequences) - len(seq_to_uniprot_map)} sequences"
    )

    # --- 4. 保存到缓存 ---
    if cache_path:
        print(
            f"--> [Canonicalizer] Saving new Sequence->UniProt map to cache: {cache_path}"
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pkl.dump(seq_to_uniprot_map, f)

    return seq_to_uniprot_map


# --- UniProt API的辅助函数 ---


def _submit_uniprot_id_mapping_job(sequence_batch: list[str]) -> str | None:
    """辅助函数：向UniProt提交ID Mapping任务。"""
    params = {
        "from": "Sequence",
        "to": "UniProtKB",
        "queries": ",".join(sequence_batch),
    }
    try:
        response = requests.post("https://rest.uniprot.org/idmapping/run", data=params)
        response.raise_for_status()
        return response.json().get("jobId")
    except requests.exceptions.RequestException as e:
        print(f"\n   - WARNING: Failed to submit UniProt job. Error: {e}")
        return None


def _get_uniprot_id_mapping_results(job_id: str) -> list:
    """辅助函数：轮询并获取已完成的UniProt ID Mapping任务的结果。"""
    while True:
        try:
            status_response = requests.get(
                f"https://rest.uniprot.org/idmapping/status/{job_id}"
            )
            status_response.raise_for_status()
            status_data = status_response.json()

            if status_data.get("jobStatus") == "FINISHED":
                # 任务完成，获取最终结果
                results_response = requests.get(
                    f"https://rest.uniprot.org/idmapping/details/{job_id}"
                )
                results_response.raise_for_status()
                return results_response.json().get("results", [])
            elif status_data.get("jobStatus") in ["RUNNING", "NEW"]:
                # 任务仍在进行，等待
                time.sleep(2)
            else:
                # 任务出现错误
                print(
                    f"\n   - WARNING: UniProt job {job_id} failed with status: {status_data.get('jobStatus')}"
                )
                return []
        except requests.exceptions.RequestException as e:
            print(f"\n   - WARNING: Failed to check UniProt job status. Error: {e}")
            return []
