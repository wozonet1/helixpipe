import re
import time
from typing import TYPE_CHECKING, Set

import requests
import research_template as rt
from tqdm import tqdm

from nasnet.utils import get_path

if TYPE_CHECKING:
    from omegaconf import DictConfig

# --- 私有辅助函数 ---


def _is_valid_uniprot_format(pid: str) -> bool:
    """使用正则表达式，对UniProt ID进行快速的本地格式检查。"""
    if not isinstance(pid, str):
        return False
    # 经典的UniProt Accession Number格式
    uniprot_pattern = re.compile(
        r"([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})"
    )
    return bool(uniprot_pattern.fullmatch(pid))


def _fetch_uniprot_batch_results(job_id: str, config: "DictConfig") -> dict:
    """轮询UniProt API以获取异步任务的结果。"""
    api_cfg = config.validators.uniprot
    while True:
        status_url = f"{api_cfg.api_url}/status/{job_id}"
        response = requests.get(status_url)
        response.raise_for_status()
        status_data = response.json()

        if status_data.get("jobStatus") == "FINISHED":
            results_url = f"{api_cfg.api_url}/results/{job_id}?format=json"
            response = requests.get(results_url)
            response.raise_for_status()
            return response.json()
        elif status_data.get("jobStatus") in ["ERROR", "FAILED"]:
            raise RuntimeError(f"UniProt job {job_id} failed.")

        time.sleep(2)  # 等待2秒再查询


def _validate_uniprot_ids_online(
    ids_to_query: Set[str], config: "DictConfig"
) -> Set[str]:
    """
    【私有】封装所有与UniProt API交互的逻辑，进行批量在线验证。
    返回一个通过验证的ID子集。
    """
    if not ids_to_query:
        return set()

    print(
        f"    - Performing online validation for {len(ids_to_query)} new UniProt IDs..."
    )
    api_cfg = config.validators.uniprot
    valid_human_pids = set()

    id_list = sorted(list(ids_to_query))

    progress_bar = tqdm(
        range(0, len(id_list), api_cfg.batch_size), desc="      UniProt Batches"
    )

    for i in progress_bar:
        batch_ids = id_list[i : i + api_cfg.batch_size]

        try:
            # 1. 提交批量任务
            submit_url = f"{api_cfg.api_url}/run"
            response = requests.post(
                submit_url,
                data={
                    "from": "UniProtKB_AC-ID",
                    "to": "UniProtKB",
                    "ids": ",".join(batch_ids),
                },
            )
            response.raise_for_status()
            job_id = response.json().get("jobId")

            if not job_id:
                print(
                    f"      - WARNING: Failed to submit batch starting with {batch_ids[0]}. Skipping."
                )
                continue

            # 2. 获取任务结果
            results_data = _fetch_uniprot_batch_results(job_id, config)

            # 3. 解析结果并根据物种ID过滤
            for result in results_data.get("results", []):
                # 确保to字段存在并且包含organism信息
                if (
                    "to" in result
                    and isinstance(result["to"], dict)
                    and "organism" in result["to"]
                ):
                    if result["to"]["organism"].get("taxonId") == api_cfg.taxon_id:
                        valid_human_pids.add(result["from"])

        except (requests.exceptions.RequestException, RuntimeError) as e:
            print(
                f"      - ❌ ERROR: API request failed for batch starting with {batch_ids[0]}. Error: {e}. Skipping batch."
            )

        time.sleep(1)  # API礼仪，在每个批量任务之间稍作等待

    print(
        f"    - Online validation complete. Found {len(valid_human_pids)} valid human UniProt IDs."
    )
    return valid_human_pids


# --- 公共接口函数 ---


def get_human_uniprot_whitelist(
    ids_to_check: Set[str], config: "DictConfig"
) -> Set[str]:
    """
    【公共接口】获取一个经过验证的人类UniProt ID白名单。

    此函数是幂等的、可缓存的。它会：
    1. 加载全局的白名单缓存。
    2. 对输入ID进行本地格式预过滤。
    3. 找出需要在线查询的新ID。
    4. (如果需要) 调用在线API进行验证。
    5. 将新验证的ID更新回全局缓存。
    6. 返回输入ID中所有被确认为“人类”的ID集合。

    Args:
        ids_to_check (Set[str]): 待检查的所有UniProt ID的集合。
        config (DictConfig): 全局配置对象。

    Returns:
        Set[str]: 一个子集，包含了输入ID中所有有效的、属于人类的UniProt ID。
    """
    force_restart = config.validators.get("force_restart", False)

    # 1. 获取全局缓存文件路径
    cache_path = get_path(config, "cache.uniprot_whitelist")
    rt.ensure_path_exists(cache_path)

    # 2. 加载现有白名单 (缓存)
    known_human_pids = set()
    if cache_path.exists() and not force_restart:
        with open(cache_path, "r") as f:
            known_human_pids = set(f.read().strip().splitlines())

    # 3. 对输入ID进行本地格式预过滤
    pre_filtered_ids = {pid for pid in ids_to_check if _is_valid_uniprot_format(pid)}

    # 4. 计算需要在线查询的新ID
    new_ids_to_query = pre_filtered_ids - known_human_pids

    # 5. (如果需要) 执行在线验证并更新缓存
    if new_ids_to_query:
        newly_validated_human_pids = _validate_uniprot_ids_online(
            new_ids_to_query, config
        )

        if newly_validated_human_pids:
            # a. 更新内存中的集合
            known_human_pids.update(newly_validated_human_pids)
            # b. 【核心】将【所有】已知的ID重写回缓存文件，确保文件总是最新的、无重复的
            with open(cache_path, "w") as f:
                f.write("\n".join(sorted(list(known_human_pids))))
            print(
                f"    - Global UniProt whitelist cache updated at '{cache_path.name}'."
            )
    else:
        print(
            "    - No new UniProt IDs to validate online. All are covered by local checks or cache."
        )

    # 6. 返回输入ID与最终白名单的交集
    return ids_to_check & known_human_pids


# TODO: 实际上未缓存
def get_valid_pubchem_cids(cids_to_check: Set[any], config: "DictConfig") -> Set[int]:
    """
    【公共接口】获取一个经过本地验证的PubChem CID白名单。
    目前只进行格式和类型检查。
    """
    print(f"    - Performing local validation for {len(cids_to_check)} PubChem CIDs...")

    valid_cids = set()
    for cid in cids_to_check:
        try:
            # 尝试将CID转换为整数，如果失败则跳过
            int_cid = int(cid)
            # 确保CID是正数
            if int_cid > 0:
                valid_cids.add(int_cid)
        except (ValueError, TypeError):
            continue

    print(
        f"    - Local validation complete. Found {len(valid_cids)} valid PubChem CIDs."
    )
    return valid_cids
