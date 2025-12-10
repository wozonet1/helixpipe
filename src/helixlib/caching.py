# 文件: src/research_template/caching.py (最终升级版)

import logging
import pickle as pkl
from pathlib import Path
from typing import Callable

from .path_manager import ensure_path_exists

logger = logging.getLogger(__name__)


def run_cached_operation(
    *,
    cache_path: Path,
    calculation_func: Callable[[list], dict],
    ids_to_process: list,
    force_restart: bool = False,
    offline_mode: bool = False,
    operation_name: str = "cached operation",
    verbose: int = 1,
) -> dict:
    """
    一个通用的、支持【增量更新】的模板函数。
    它只为缓存中不存在的新ID执行计算，并将结果合并回缓存。

    Args:
        cache_path (Path): 缓存文件的绝对路径。
        calculation_func (Callable): 接收一个ID列表作为输入的计算函数。
        ids_to_process (list): 本次操作需要处理的所有ID的完整列表。
        force_restart (bool): 是否强制重新计算所有ID。
        operation_name (str): 用于日志打印的操作名称。
        verbose (int): 日志详细级别。

    Returns:
        dict: 一个字典，包含所有 `ids_to_process` 对应的结果。
    """
    if not ids_to_process:
        return {}

    # 1. 加载现有缓存
    cached_data: dict = {}
    if cache_path.exists() and not force_restart:
        if verbose > 0:
            print(
                f"\n--> [Cache Hit] for '{operation_name}'. Loading from '{cache_path.name}'..."
            )
        with open(cache_path, "rb") as f:
            try:
                cached_data = pkl.load(f)
            except (pkl.UnpicklingError, EOFError):
                print(
                    f"    - ⚠️ WARNING: Cache file '{cache_path.name}' is corrupted. Treating as cache miss."
                )
                cached_data = {}
    else:
        if verbose > 0:
            print(f"\n--> [Cache Miss/Restart] for '{operation_name}'.")
    if offline_mode:
        logger.info(
            f"--> [Offline Mode] Skipping online fetch for '{operation_name}'. Using cache only."
        )

        # 即使 force_restart=True，offline_mode 的优先级也应该更高（或者互斥）
        # 这里我们只从缓存中筛选出请求的 ID
        # 对于缓存中不存在的 ID，它们将被默默丢弃（不返回）
        return {k: v for k, v in cached_data.items() if k in set(ids_to_process)}
    # 2. 计算需要增量获取的ID
    requested_ids_set: set = set(ids_to_process)
    cached_ids_set: set = set(cached_data.keys())

    ids_to_fetch = list(requested_ids_set - cached_ids_set)

    # 3. 如果有新ID，则执行计算
    if ids_to_fetch:
        if verbose > 0:
            print(
                f"--> Found {len(ids_to_fetch)} new items for '{operation_name}'. Executing calculation..."
            )

        # 【核心】只对新ID调用计算函数
        newly_fetched_data = calculation_func(ids_to_fetch)

        if newly_fetched_data:
            # 4. 合并新旧数据
            cached_data.update(newly_fetched_data)

            # 5. 将更新后的完整缓存写回磁盘
            ensure_path_exists(cache_path)
            if verbose > 0:
                print(
                    f"--> Saving updated map for '{operation_name}' back to cache ({len(cached_data)} total items)."
                )
            with open(cache_path, "wb") as f:
                pkl.dump(cached_data, f)
    else:
        if verbose > 0:
            print(
                f"--> All {len(requested_ids_set)} requested items for '{operation_name}' are already in the cache."
            )

    # 6. 从完整的缓存中筛选出本次请求的结果并返回
    return {k: v for k, v in cached_data.items() if k in requested_ids_set}
