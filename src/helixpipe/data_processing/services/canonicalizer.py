import pickle as pkl
import time
from pathlib import Path

import pandas as pd
import pubchempy as pcp
import requests
from rdkit import Chem
from tqdm import tqdm


def canonicalize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, canonical=True)
        else:
            return None  # 如果初始解析就失败，返回None
    except Exception:
        # 捕获所有可能的意外错误
        return None


def canonicalize_smiles_to_cid(
    smiles_list: list[str],
    cache_path: Path = None,
    force_regenerate: bool = False,  # 新增 force_regenerate
) -> dict[str, int]:
    """
    【新版 - 更健壮】将一个SMILES字符串列表，转换为PubChem CID。
    采用逐个查询的方式，以最大限度地提高成功率和错误隔离。
    """
    if cache_path and cache_path.exists() and not force_regenerate:
        print(f"--> [Canonicalizer] Loading cached SMILES->CID map from: {cache_path}")
        with open(cache_path, "rb") as f:
            return pkl.load(f)

    print(
        "--> [Canonicalizer] Starting SMILES to CID conversion (robust, one-by-one mode)..."
    )

    unique_smiles = sorted(list(set(s for s in smiles_list if s and pd.notna(s))))

    smiles_to_cid_map = {}
    not_found_smiles = []

    # [核心修改] 不再使用批量API，而是逐个查询
    progress_bar = tqdm(unique_smiles, desc="[Canonicalizer] Querying PubChem")

    for smiles in progress_bar:
        max_retries = 3
        cid = None
        for attempt in range(max_retries):
            try:
                # pcp.get_compounds 返回一个Compound对象列表
                compounds = pcp.get_compounds(smiles, "smiles")
                if compounds:
                    # 我们只取第一个结果的CID
                    cid = compounds[0].cid
                    break  # 成功，跳出重试循环
            except (pcp.PubChemHTTPError, requests.exceptions.RequestException) as e:
                # 增加了对底层requests错误的捕获
                if attempt < max_retries - 1:
                    time.sleep(1 * (attempt + 1))  # 稍作等待再重试
                else:
                    # 记录多次重试后仍然失败的错误
                    print(
                        f"\n   - WARNING: API error for SMILES '{smiles}' after {max_retries} attempts: {e}"
                    )

        if cid:
            smiles_to_cid_map[smiles] = cid
        else:
            not_found_smiles.append(smiles)

        # [修改] 每次查询后都稍作延时，避免过于频繁
        time.sleep(0.05)  # 50毫秒

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
