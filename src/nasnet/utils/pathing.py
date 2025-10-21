import shutil
from collections import defaultdict
from pathlib import Path
from typing import Generator, Tuple

import research_template as rt
from omegaconf import DictConfig, ListConfig, OmegaConf
from research_template import check_paths_exist as generic_check_paths_exist


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
    print("\n--- [Setup] Verifying and setting up dataset directories... ---")

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
                print(
                    f"!!! WARNING: `force_restart` is True for variant '{variant_name}' and experiment '{experiment_name}'."
                )
                print(f"    Deleting directory: {dir_to_clean}")
                shutil.rmtree(dir_to_clean)

        # --- 2. 声明式地创建所有需要的目录 ---
        print("--> Ensuring all necessary directories exist...")

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
                # print(f"    - Skipping path key '{full_key}' (cannot be resolved in current context: {e})")
                pass

        # c. 确保每个解析出的路径的父目录都存在
        created_count = 0
        unique_parent_dirs = {p.parent for p in paths_to_ensure}
        for parent_dir in sorted(list(unique_parent_dirs)):  # 排序以便于查看日志
            if not parent_dir.exists():
                parent_dir.mkdir(parents=True, exist_ok=True)
                created_count += 1

        if created_count > 0:
            print(f"--> Successfully created {created_count} new directories.")
        else:
            print("-> All necessary directories already exist.")

        # --- 3. (可选) 检查原始文件目录是否为空 ---
        raw_dir_path = get_path(config, "raw.dummy_file_to_get_dir").parent
        if not any(raw_dir_path.iterdir()):
            print(f"-> ⚠️  WARNING: The 'raw' directory is empty: {raw_dir_path}")

        print("--- [Setup] Directory setup complete. ---")

    except (rt.ConfigKeyError, KeyError) as e:
        # 使用我们之前设计的自定义异常来提供清晰的错误报告
        missing_key = e.full_key if isinstance(e, rt.ConfigKeyError) else str(e)
        raise rt.ConfigPathError(
            message=f"Failed during directory setup due to missing config key: {missing_key}",
            missing_key=missing_key,
            file_key="N/A (during setup)",
        ) from e


def get_path(cfg: DictConfig, short_key: str, **kwargs) -> Path:
    """
    【V8 终极优雅版】直接从已解析的配置中获取路径。
    所有复杂的路径构建逻辑都已封装在自定义的 'path' 解析器中。
    """
    # 【注意】我们不再需要在这里注入任何运行时变量！
    # 因为解析器在被调用时，可以访问到完整的、最新的cfg对象。
    full_key = f"data_structure.paths.{short_key}"
    try:
        # select 会触发 ${path:...} 解析器的执行
        resolved_path_str = OmegaConf.select(cfg, full_key)
    except rt.ConfigKeyError as e:
        raise rt.ConfigPathError(
            message=f"Missing config key '{e.full_key}'",
            missing_key=e.full_key,
            file_key=full_key,
        ) from e  # 使用 `from e` 保留原始的异常链，以便深度调试

    if resolved_path_str is None:
        raise ValueError(f"Failed to find {full_key}\n")
    # 格式化 **kwargs (prefix, suffix) - 这部分仍然需要
    if kwargs:
        format_args = defaultdict(str, **kwargs)
        resolved_path_str = resolved_path_str.format_map(format_args)

    return rt.get_project_root() / resolved_path_str


def check_project_files_exist(config: "DictConfig", *file_keys: str) -> bool:
    """
    【nasnet 专属】检查所有指定的【项目特定】的数据文件是否存在。

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
