import logging
import shutil
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Callable, Generator, Tuple, Union

import research_template as rt
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.errors import (
    ConfigKeyError,
    InterpolationResolutionError,
)
from research_template import check_paths_exist as generic_check_paths_exist
from research_template.errors import ConfigPathError

logger = logging.getLogger(__name__)


def _walk_config(
    cfg: DictConfig, prefix: str = ""
) -> Generator[Tuple[str, any], None, None]:
    """
    一个辅助函数，用于递归地遍历OmegaConf配置树，并生成所有叶子节点的
    (完整路径, 值) 对。

    Args:
        cfg (DictConfig): 要遍历的配置节点。
        prefix (str): 用于构建完整路径的当前前缀。

    Yields:
        Generator[Tuple[str, any], None, None]: 一个生成器，每次产生一个
                                                (full_key, value) 元组。
    """
    for key, value in cfg.items_ex(resolve=False):
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, (DictConfig, ListConfig)):
            # 如果值是容器，则继续递归
            yield from _walk_config(value, prefix=full_key)
        else:
            # 如果是叶子节点，则产生结果
            yield full_key, value


def setup_dataset_directories(config: DictConfig) -> None:
    """
    【V3 重构版】根据配置，创建所有必需的目录，并按需清理旧的实验产物。
    此函数现在是“声明式”的：它遍历配置中定义的所有路径，并确保它们存在。
    """
    logger.info("\n--- [Setup] Verifying and setting up dataset directories... ---")

    try:
        # --- 1. 按需清理 (更智能的清理) ---
        if config.runtime.get("force_restart", False):
            # 我们只清理与【当前实验配置】相关的【specific】文件夹。
            # 这可以避免误删其他实验（例如，不同relations）的昂贵产物。

            # 使用get_path来定位要清理的目录
            # 我们通过一个 specific 模板键来获取其父目录
            dummy_specific_key = "processed.specific.graph_template"
            dir_to_clean = get_path(config, dummy_specific_key, prefix="dummy").parent

            if dir_to_clean.exists():
                variant_name = config.data_params.name
                experiment_name = dir_to_clean.name
                logger.warning(
                    f"!!! WARNING: `force_restart` is True for variant '{variant_name}' and experiment '{experiment_name}'."
                )
                logger.into(f"    Deleting directory: {dir_to_clean}")
                shutil.rmtree(dir_to_clean)

        # --- 2. 声明式地创建所有需要的目录 ---
        logger.into("--> Ensuring all necessary directories exist...")

        # a. 遍历 `paths` 配置块中的所有叶子节点 (路径模板)
        #    OmegaConf.select_leaves 会返回一个生成器，包含 (key, value)
        paths_to_ensure = []
        paths_config_node = config.data_structure.paths
        for key, value_template in _walk_config(paths_config_node):
            full_key = f"data_structure.paths.{key}"

            try:
                # b. 调用 get_path 来解析每个路径模板
                #    对于需要格式化的模板，我们传入dummy值以获取其父目录
                if "{" in str(value_template):
                    path = get_path(config, full_key, prefix="dummy", suffix="dummy")
                else:
                    path = get_path(config, full_key)
                paths_to_ensure.append(path)
            except (KeyError, ValueError):
                # 忽略那些无法在当前上下文中解析的路径 (例如，gtopdb的专属路径在bindingdb运行时)
                # logger.into(f"    - Skipping path key '{full_key}' (cannot be resolved in current context: {e})")
                pass

        # c. 确保每个解析出的路径的父目录都存在
        created_count = 0
        unique_parent_dirs = {p.parent for p in paths_to_ensure}
        for parent_dir in sorted(list(unique_parent_dirs)):  # 排序以便于查看日志
            if not parent_dir.exists():
                parent_dir.mkdir(parents=True, exist_ok=True)
                created_count += 1

        if created_count > 0:
            logger.into(f"--> Successfully created {created_count} new directories.")
        else:
            logger.into("-> All necessary directories already exist.")

        # --- 3. (可选) 检查原始文件目录是否为空 ---
        raw_dir_path = get_path(config, "raw.dummy_file_to_get_dir").parent
        if not any(raw_dir_path.iterdir()):
            logger.warning(
                f"-> ⚠️  WARNING: The 'raw' directory is empty: {raw_dir_path}"
            )

        logger.info("--- [Setup] Directory setup complete. ---")

    except (rt.ConfigKeyError, KeyError) as e:
        # 使用我们之前设计的自定义异常来提供清晰的错误报告
        missing_key = e.full_key if isinstance(e, rt.ConfigKeyError) else str(e)
        raise rt.ConfigPathError(
            message=f"Failed during directory setup due to missing config key: {missing_key}",
            missing_key=missing_key,
            file_key="N/A (during setup)",
        ) from e


def get_path(
    cfg: DictConfig, short_key: str, **kwargs
) -> Union[Path, Callable[..., Path]]:
    """
    【V11 路径工厂最终版】一个智能的、双模式的路径获取函数。

    - **模式1 (静态路径)**: 如果解析出的路径是一个完整的路径（不含'{'占位符），
      它会直接返回一个【绝对路径的 Path 对象】。

    - **模式2 (路径工厂)**: 如果解析出的路径是一个模板（包含'{'占位符），
      它会返回一个【可调用的“路径工厂”函数】。这个工厂函数接收关键字参数来
      填充模板，并返回最终的绝对路径 Path 对象。

    Args:
        cfg (DictConfig): 完整的Hydra配置对象。
        short_key (str): 点分隔的短键，例如 "raw.authoritative_dti"。
        **kwargs: (可选) 用于立即填充模板的一部分参数。

    Returns:
        Union[Path, Callable[..., Path]]:
            - 一个Path对象（对于静态路径）。
            - 一个可调用对象（对于路径模板）。
    """
    full_key = f"data_structure.paths.{short_key}"
    try:
        resolved_path_str = OmegaConf.select(cfg, full_key)

    except InterpolationResolutionError as e:
        # 【核心修改】捕获插值错误，并传递更丰富的上下文
        raise ConfigPathError(
            message="Failed to resolve path due to an interpolation error.",
            file_key=full_key,
            failed_interpolation_key=str(e).split("'")[1]
            if "'" in str(e)
            else "unknown",  # 尝试提取出错的键
            original_exception=e,
        ) from e
    except ConfigKeyError as e:
        # 【核心修改】捕获键错误
        raise ConfigPathError(
            message="A required configuration key was not found.",
            file_key=full_key,
            original_exception=e,
        ) from e

    # --- 核心的、双模式逻辑 ---

    project_root = rt.get_project_root()

    # 1. 检查解析出的字符串是否是一个需要后续格式化的模板
    if "{" in resolved_path_str:
        # 定义一个闭包函数作为我们的“路径工厂”
        def path_factory(**factory_kwargs) -> Path:
            # 使用 format_map 和 defaultdict，可以安全地处理不完整的参数
            final_str = resolved_path_str.format_map(defaultdict(str, factory_kwargs))
            if "{" in final_str:
                # 检查格式化后是否还有未填充的占位符
                logger.warning(
                    f"Warning: Path template for key '{short_key}' may be incompletely formatted: {final_str}"
                )
            return project_root / final_str

        # 2. 如果在调用 get_path 时已经提供了 kwargs，则使用 functools.partial
        #    来“预填充”这些参数，返回一个新的、参数更少的函数。
        if kwargs:
            return partial(path_factory, **kwargs)
        else:
            # 否则，直接返回原始的工厂函数
            return path_factory

    else:
        # 3. 如果是静态路径，直接返回最终的 Path 对象
        #    如果此时传入了 kwargs，它们将被忽略，可以打印一个警告
        if kwargs:
            logger.warning(
                f"Warning: kwargs {kwargs} were provided for a static path key '{short_key}' and will be ignored."
            )
        return project_root / resolved_path_str


def check_project_files_exist(config: "DictConfig", *file_keys: str) -> bool:
    """
    【helixpipe 专属】检查所有指定的【项目特定】的数据文件是否存在。

    这个函数是通用 `check_paths_exist` 的一个包装器，它知道如何使用
    本项目的 `get_path` 和 `file_key` 约定。

    Args:
        config (DictConfig): 项目的Hydra配置对象。
        *file_keys (str): 一系列要检查的、点分隔的文件键。

    Returns:
        bool: 如果所有对应的文件都存在，则返回True。
    """
    try:
        # 使用一个生成器表达式，高效地将所有 file_keys 转换为 Path 对象
        paths_to_check = (get_path(config, key) for key in file_keys)

        # 调用通用的检查函数
        return generic_check_paths_exist(paths_to_check)

    except (KeyError, ValueError):
        # 如果任何一个 get_path 调用失败，都视为检查不通过
        return False
