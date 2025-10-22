import json
import pickle as pkl
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pubchempy as pcp
import requests
from rdkit import Chem
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm

session = requests.Session()
retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))


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


def is_valid_uniprot_accession(accession):
    """
    使用正则表达式,快速检查一个ID是否符合UniProt Accession的典型格式。
    这是一个简化版检查,但能过滤掉大部分非Accession的ID。
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
            response = session.get(base_url, params=params)

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
                        single_response = session.get(
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


def fetch_smiles_from_pubchem(
    cids: List[int], batch_size: int = 100, proxies: Optional[Dict[str, str]] = None
) -> Dict[int, str]:
    """
    【V8 - 最终修正版】
    使用requests GET请求，并使用正确的JSON键名(ConnectivitySMILES)来解析响应。
    """
    if not cids:
        return {}

    print(
        f"--> [PubChem Fetcher] Querying SMILES for {len(cids)} CIDs using direct GET requests..."
    )
    if proxies:
        print(f"    - Using proxies: {proxies}")

    # 【核心修正1】我们请求的属性可以是 CanonicalSMILES，但响应的键可能是不同的。
    # 为了更精确，我们可以同时请求 CanonicalSMILES 和 ConnectivitySMILES
    properties_to_fetch = "CanonicalSMILES,ConnectivitySMILES"

    cid_to_smiles_map: Dict[int, str] = {}

    for i in tqdm(range(0, len(cids), batch_size), desc="   - Fetching SMILES batches"):
        batch_cids = cids[i : i + batch_size]
        cids_str = ",".join(map(str, batch_cids))

        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cids_str}/property/{properties_to_fetch}/JSON"

        try:
            response = session.get(url, proxies=proxies, timeout=60)
            response.raise_for_status()

            data = response.json()
            results = data.get("PropertyTable", {}).get("Properties", [])

            for item in results:
                cid = item.get("CID")
                # 【核心修正2】优先使用 CanonicalSMILES，如果不存在，则回退到 ConnectivitySMILES
                smiles = item.get("CanonicalSMILES") or item.get("ConnectivitySMILES")
                if cid and smiles:
                    cid_to_smiles_map[int(cid)] = canonicalize_smiles(smiles)

        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            print(
                f"\n    - ❌ NETWORK/JSON/PROXY ERROR for batch starting with CID {batch_cids[0]}: {e}"
            )
            continue

        time.sleep(0.25)

    print(
        f"--> [PubChem Fetcher] Successfully fetched canonicalized SMILES for {len(cid_to_smiles_map)} CIDs."
    )
    return cid_to_smiles_map
