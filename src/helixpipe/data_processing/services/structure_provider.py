# 文件: src/helixpipe/data_processing/services/structure_provider.py (全新)

import json
import logging
import re
import time
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry  # type: ignore
from tqdm import tqdm

import helixlib as hx
from helixpipe.typing import CID, PID, SMILES, AppConfig, ProteinSequence
from helixpipe.utils import get_path

logger = logging.getLogger(__name__)


class StructureProvider:
    """
    一个独立的、可复用的服务类，负责从权威在线数据源获取结构信息（序列/SMILES）。
    它封装了API调用、批处理、重试和错误处理的所有逻辑。
    """

    # --- 使用我们通过基准测试确定的最优Batch Size作为类常量,注意,虽然uniprot测试为200最佳,但是实际上200太长,改为100 ---
    OPTIMAL_UNIPROT_BATCH_SIZE = 100
    OPTIMAL_PUBCHEM_BATCH_SIZE = 400

    def __init__(
        self, config: AppConfig, proxies: Optional[dict[str, str]] = None
    ) -> None:
        self.config = config
        self.proxies = proxies
        self._session = self._create_session()
        logger.info("--- [StructureProvider] Initialized. ---")

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
        self, uniprot_ids: list[PID], force_restart: bool = False
    ) -> dict[PID, ProteinSequence]:
        """为UniProt ID列表获取序列，使用增量缓存。"""
        return hx.run_cached_operation(
            cache_path=get_path(self.config, "cache.ids.enriched_protein_sequences"),
            calculation_func=self._fetch_sequences_from_uniprot,
            ids_to_process=uniprot_ids,
            force_restart=force_restart,
            offline_mode=self.config.runtime.strict_offline_mode,
            operation_name="Enriched Protein Sequences",
            verbose=self.config.runtime.verbose,
        )

    def get_smiles(
        self, cids: list[CID], force_restart: bool = False
    ) -> dict[CID, SMILES]:
        """为PubChem CID列表获取SMILES，使用增量缓存。"""
        return hx.run_cached_operation(
            cache_path=get_path(self.config, "cache.ids.enriched_molecule_smiles"),
            calculation_func=self._fetch_smiles_from_pubchem,
            ids_to_process=cids,
            force_restart=force_restart,
            offline_mode=self.config.runtime.strict_offline_mode,
            operation_name="Enriched Molecule SMILES",
            verbose=self.config.runtime.verbose,
        )

    # --- 私有实现方法 ---

    # --- 【核心修改】为原有的 _fetch_sequences_from_uniprot 方法增加降级逻辑 ---
    def _fetch_sequences_from_uniprot(
        self, ids_to_fetch: list[PID]
    ) -> dict[PID, ProteinSequence]:
        """
        【V4 - 带GET降级版】调用UniProt API获取蛋白质序列。
        主要使用高效的GET stream请求，但在失败时降级为逐一查询。
        """
        # 1. 预处理：过滤掉格式不正确的ID (逻辑保持不变)
        valid_ids = sorted(
            list({uid for uid in ids_to_fetch if self._is_valid_uniprot_format(uid)})
        )
        if not valid_ids:
            return {}

        logger.info(
            f"--> [UniProt Fetcher] Fetching sequences for {len(valid_ids)} valid UniProt IDs..."
        )
        sequences_map: dict[PID, ProteinSequence] = {}

        # 2. 按批次大小进行分块 (逻辑保持不变)
        for i in tqdm(
            range(0, len(valid_ids), self.OPTIMAL_UNIPROT_BATCH_SIZE),
            desc="   - UniProt Batches",
        ):
            chunk = valid_ids[i : i + self.OPTIMAL_UNIPROT_BATCH_SIZE]

            # --- 降级策略包裹 ---
            try:
                # --- 步骤 2a: 尝试使用原来的高效GET stream请求 ---
                # (这部分是您提供的、经过验证的原始代码)
                query = " OR ".join(f"(accession:{acc})" for acc in chunk)
                params = {
                    "query": query,
                    "format": "fasta",
                    "fields": "accession,sequence",
                }

                # 【风险点】检查URL长度，如果过长，主动抛出异常以触发降级
                # UniProt官方未明确GET的URL长度限制，但一般Web服务器在2000-8000字符
                # 我们设定一个保守的阈值，例如 4000
                url = self._session.prepare_request(
                    requests.Request(
                        "GET",
                        "https://rest.uniprot.org/uniprotkb/stream",
                        params=params,
                    )
                ).url
                if url is None:
                    raise RuntimeError
                if len(url) > 4000:
                    raise ValueError(
                        "Generated GET request URL is too long, forcing fallback."
                    )

                response = self._session.get(
                    "https://rest.uniprot.org/uniprotkb/stream",
                    params=params,
                    proxies=self.proxies,
                    timeout=60,
                )
                response.raise_for_status()  # 检查 4xx/5xx 错误

                chunk_results = {}
                for entry in response.text.strip().split(">"):
                    if not entry.strip():
                        continue
                    lines = entry.strip().split("\n")
                    header, seq = lines[0], "".join(lines[1:])
                    try:
                        # 使用更健壮的正则表达式解析
                        uid_match = re.search(
                            r"([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})",
                            header,
                        )
                        if uid_match and seq:
                            chunk_results[uid_match.group(1)] = seq
                    except (IndexError, KeyError):
                        if len(header) < 100:
                            logger.warning(
                                f"\n    - Warning: Could not parse UniProt ID from header: '{header}'"
                            )
                        continue
                if (
                    self.config.runtime.verbose > 0
                ):  # 只在深度调试模式下打印详细的失败ID
                    requested_ids_set = set(chunk)
                    returned_ids_set = set(chunk_results.keys())

                    missing_ids = requested_ids_set - returned_ids_set

                    if missing_ids:
                        logger.warning(
                            f"\n    - DEBUG (UniProt): For a batch of {len(chunk)} IDs, "
                            f"server did not return sequences for {len(missing_ids)} IDs."
                        )
                        # 为了避免刷屏，只打印少量样本
                        logger.warning(
                            f"      - Sample of missing IDs: {sorted(list(missing_ids))[:10]}"
                        )
                sequences_map.update(chunk_results)

            except (requests.exceptions.RequestException, ValueError) as e:
                # --- 步骤 2b: 如果GET请求失败 (网络错误、超时、4xx/5xx、或URL过长)，则降级 ---
                logger.warning(
                    f"\n    - ⚠️ WARNING: Batch GET request failed: {e}. "
                    f"Falling back to one-by-one query for this batch ({len(chunk)} IDs)..."
                )

                chunk_results_fallback = self._fetch_batch_one_by_one(chunk)
                sequences_map.update(chunk_results_fallback)

            time.sleep(0.5)
        logger.info(
            f"--> [UniProt Fetcher] Fetched {len(sequences_map)} / {len(valid_ids)} sequences."
        )
        return sequences_map

    # --- 辅助方法: 逐一查询 (这个函数保持不变，作为降级的实现) ---
    def _fetch_batch_one_by_one(
        self, id_chunk: list[PID]
    ) -> dict[PID, ProteinSequence]:
        """作为最终的降级手段，逐个请求ID。"""
        results = {}
        for pid in tqdm(id_chunk, desc="     - Fallback (one-by-one)", leave=False):
            url = f"https://rest.uniprot.org/uniprotkb/{pid}.fasta"
            try:
                response = self._session.get(url, proxies=self.proxies, timeout=20)
                if response.status_code == 404:
                    logger.warning(
                        f"\n       - INFO: UniProt ID '{pid}' not found (404)."
                    )
                    continue
                response.raise_for_status()

                lines = response.text.strip().split("\n")
                if len(lines) > 1:
                    seq = "".join(lines[1:])
                    if seq:
                        results[pid] = seq
            except requests.exceptions.RequestException as e:
                logger.error(
                    f"\n       - ❌ ERROR: Failed to fetch sequence for single UniProt ID '{pid}': {e}"
                )

            time.sleep(0.1)
        return results

    def _fetch_smiles_from_pubchem(self, ids_to_fetch: list[CID]) -> dict[CID, SMILES]:
        """【私有版】调用PubChem API，获取SMILES。"""
        if not ids_to_fetch:
            return {}

        logger.info(
            f"--> [PubChem Fetcher] Querying SMILES for {len(ids_to_fetch)} CIDs..."
        )

        cid_to_smiles_map: dict[int, str] = {}

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
                logger.error(f"\n    - ❌ NETWORK/JSON/PROXY ERROR for batch: {e}")
                continue

            time.sleep(0.25)

        logger.info(
            f"--> [PubChem Fetcher] Fetched SMILES for {len(cid_to_smiles_map)} / {len(ids_to_fetch)} CIDs."
        )
        return cid_to_smiles_map

    # 【核心修正】将 is_valid_uniprot_accession 作为类的私有辅助方法
    def _is_valid_uniprot_format(self, accession: str) -> bool:
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
