# 文件: src/nasnet/data_processing/services/structure_provider.py (全新)

import json
import re
import time
from typing import Dict, List, Optional

import requests
import research_template as rt
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry  # type: ignore
from tqdm import tqdm

from nasnet.configs import AppConfig
from nasnet.utils import get_path


# 【核心修正】将 is_valid_uniprot_accession 作为类的私有辅助方法
def _is_valid_uniprot_format(accession: str) -> bool:
    """
    使用UniProt官方的严格格式进行API请求前的语法检查。
    """
    if not isinstance(accession, str):
        return False
    # 表达式来源: UniProt官方文档，并通过本地权威列表100%验证
    pattern = re.compile(
        r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$",
        re.IGNORECASE,
    )
    return bool(pattern.match(accession))


class StructureProvider:
    """
    一个独立的、可复用的服务类，负责从权威在线数据源获取结构信息（序列/SMILES）。
    它封装了API调用、批处理、重试和错误处理的所有逻辑。
    """

    # --- 使用我们通过基准测试确定的最优Batch Size作为类常量 ---
    OPTIMAL_UNIPROT_BATCH_SIZE = 200
    OPTIMAL_PUBCHEM_BATCH_SIZE = 400

    def __init__(self, config: AppConfig, proxies: Optional[Dict[str, str]] = None):
        self.config = config
        self.proxies = proxies
        self._session = self._create_session()
        print("--- [StructureProvider] Initialized. ---")

    def _create_session(self) -> requests.Session:
        """创建一个带有自动重试机制的requests Session。"""
        session = requests.Session()
        # 定义重试策略：总共重试3次，对常见的服务器端错误进行重试
        retries = Retry(
            total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("https://", adapter)
        return session

    # --- 公共接口 ---

    def get_sequences(
        self, uniprot_ids: List[str], force_restart: bool = False
    ) -> Dict[str, str]:
        """为UniProt ID列表获取序列，使用增量缓存。"""
        return rt.run_cached_operation(
            cache_path=get_path(self.config, "cache.ids.enriched_protein_sequences"),
            calculation_func=self._fetch_sequences_from_uniprot,
            ids_to_process=uniprot_ids,
            force_restart=force_restart,
            operation_name="Enriched Protein Sequences",
            verbose=self.config.runtime.verbose,
        )

    def get_smiles(
        self, cids: List[int], force_restart: bool = False
    ) -> Dict[int, str]:
        """为PubChem CID列表获取SMILES，使用增量缓存。"""
        return rt.run_cached_operation(
            cache_path=get_path(self.config, "cache.ids.enriched_molecule_smiles"),
            calculation_func=self._fetch_smiles_from_pubchem,
            ids_to_process=cids,
            force_restart=force_restart,
            operation_name="Enriched Molecule SMILES",
            verbose=self.config.runtime.verbose,
        )

    # --- 私有实现方法 ---

    def _fetch_sequences_from_uniprot(self, ids_to_fetch: List[str]) -> Dict[str, str]:
        """【私有版】调用UniProt API，获取蛋白质序列。"""
        formatted_ids = {uid for uid in ids_to_fetch if _is_valid_uniprot_format(uid)}
        valid_ids = sorted(list(formatted_ids))
        if not valid_ids:
            return {}

        print(
            f"--> [UniProt Fetcher] Fetching sequences for {len(valid_ids)} UniProt IDs..."
        )

        sequences_map: Dict[str, str] = {}

        for i in tqdm(
            range(0, len(valid_ids), self.OPTIMAL_UNIPROT_BATCH_SIZE),
            desc="   - UniProt Batches",
        ):
            chunk = valid_ids[i : i + self.OPTIMAL_UNIPROT_BATCH_SIZE]
            params = {
                "query": " OR ".join(f"(accession:{acc})" for acc in chunk),
                "format": "fasta",
                "fields": "accession,sequence",
            }
            try:
                response = self._session.get(
                    "https://rest.uniprot.org/uniprotkb/stream",
                    params=params,
                    proxies=self.proxies,
                    timeout=60,
                )
                response.raise_for_status()

                for entry in response.text.strip().split(">"):
                    if not entry.strip():
                        continue
                    lines = entry.strip().split("\n")
                    header, seq = lines[0], "".join(lines[1:])
                    try:
                        uid = header.split("|")[1]
                        if seq:
                            sequences_map[uid] = seq
                    except (IndexError, KeyError):
                        if len(header) < 100:
                            print(
                                f"\n    - Warning: Could not parse UniProt ID from header: '{header}'"
                            )
                        continue
            except requests.exceptions.RequestException as e:
                print(f"\n    - ❌ NETWORK ERROR during UniProt fetch for a batch: {e}")

            time.sleep(0.5)

        print(f"--> [UniProt Fetcher] Fetched {len(sequences_map)} sequences.")
        return sequences_map

    def _fetch_smiles_from_pubchem(self, ids_to_fetch: List[int]) -> Dict[int, str]:
        """【私有版】调用PubChem API，获取SMILES。"""
        if not ids_to_fetch:
            return {}

        print(f"--> [PubChem Fetcher] Querying SMILES for {len(ids_to_fetch)} CIDs...")

        cid_to_smiles_map: Dict[int, str] = {}

        for i in tqdm(
            range(0, len(ids_to_fetch), self.OPTIMAL_PUBCHEM_BATCH_SIZE),
            desc="   - PubChem Batches",
        ):
            batch_cids = ids_to_fetch[i : i + self.OPTIMAL_PUBCHEM_BATCH_SIZE]
            cids_str = ",".join(map(str, batch_cids))

            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cids_str}/property/CanonicalSMILES,ConnectivitySMILES/JSON"

            try:
                response = self._session.get(url, proxies=self.proxies, timeout=60)
                response.raise_for_status()

                data = response.json()
                results = data.get("PropertyTable", {}).get("Properties", [])

                for item in results:
                    cid = item.get("CID")
                    smiles = item.get("CanonicalSMILES") or item.get(
                        "ConnectivitySMILES"
                    )
                    if cid and smiles:
                        cid_to_smiles_map[int(cid)] = (
                            smiles  # 暂不在这里canonicalize，让调用者决定
                        )

            except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
                print(f"\n    - ❌ NETWORK/JSON/PROXY ERROR for batch: {e}")
                continue

            time.sleep(0.25)

        print(
            f"--> [PubChem Fetcher] Fetched SMILES for {len(cid_to_smiles_map)} CIDs."
        )
        return cid_to_smiles_map
