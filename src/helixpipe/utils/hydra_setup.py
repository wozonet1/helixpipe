from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from .pathing import get_path


def path_resolver(key: str, *, _root_: DictConfig) -> str:
    """
    【V4 最终版 - 无前缀】一个统一的路径解析器。
    它能根据 key 的内容智能地构建实验文件路径或全局缓存路径。

    用法:
    - ${path:raw.authoritative_dti} -> 构建实验文件路径
    - ${path:cache.feature_template} -> 构建全局缓存路径模板
    """
    cfg = _root_

    # --- 逻辑 A: 构建全局缓存的路径 ---
    if key.startswith("cache."):
        # a. 【核心变化】根据key的第一部分确定根目录
        #    key: "cache.features.template" -> parts: ["cache", "features", "template"]
        parts = key.split(".")
        cache_type = parts[1]  # "features" or "ids"

        # 使用 getattr 动态访问配置
        # e.g., cfg.global_paths.feature_cache_dir
        base_dir = Path(getattr(cfg.global_paths, f"{cache_type}_cache_dir"))

        # b. 获取文件名或模板
        filename_key_path = f"data_structure.filenames.{key}"
        filename_or_template = OmegaConf.select(cfg, filename_key_path)

        if filename_or_template is None:
            raise KeyError(
                f"Filename/template for cache key '{key}' not found at '{filename_key_path}'."
            )

        return str(base_dir / filename_or_template)

    elif key.startswith("assets."):
        base_dir = Path(cfg.global_paths.assets_dir)
        # key: "assets.uniprot_proteome_tsv" -> sub_key: "uniprot_proteome_tsv"
        sub_key = key.split(".", 1)[1]

        filename_key_path = f"data_structure.filenames.assets.{sub_key}"
        filename = OmegaConf.select(cfg, filename_key_path)

        if filename is None:
            raise KeyError(
                f"Filename for asset key '{key}' not found at '{filename_key_path}'."
            )

        return str(base_dir / filename)
    # --- 逻辑 B: 构建特定于实验的路径 (默认) ---
    else:
        # a. 动态上下文注入 (在解析器内部完成)
        variant_folder = cfg.data_params.name  # 直接使用 data_params 配置的名称
        relations_name = cfg.relations.name
        split_mode = cfg.training.coldstart.mode
        experiment_folder = f"{relations_name}-{split_mode}split"
        collection_name = cfg.dataset_collection.name

        # b. 基础目录
        base_dir = Path(cfg.global_paths.data_root) / cfg.data_structure.primary_dataset

        # c. 逐级构建路径
        parts = key.split(".")  # e.g., "processed.common.nodes_metadata"
        current_path = base_dir
        for i, part in enumerate(parts):
            current_path /= part
            if part == "processed":
                current_path /= collection_name
                current_path /= variant_folder
            elif part == "specific" and (i > 0 and parts[i - 1] == "processed"):
                current_path = current_path.parent / "experiments" / experiment_folder

        # d. 获取文件名
        filename_key_path = f"data_structure.filenames.{key}"
        filename = OmegaConf.select(cfg, filename_key_path)

        if filename is None:
            raise KeyError(
                f"Filename for key '{key}' not found at '{filename_key_path}'."
            )

        # 最终路径是目录的父级 + 文件名
        # 例如, key="processed.common.nodes_metadata", current_path 会变成
        # .../common/nodes_metadata, 我们需要其父目录 .../common/
        return str(current_path.parent / filename)


# --- 2. 注册解析器 ---
# 这是一个一次性的操作，应该在您的应用主入口执行


def register_hydra_resolvers():
    """
    注册项目中所有自定义的Hydra解析器。
    """
    resolver_name = "path"
    if not OmegaConf.has_resolver(resolver_name):
        # 【核心修正】直接传递函数本身，OmegaConf会自动处理特殊参数的注入
        OmegaConf.register_new_resolver(
            name=resolver_name,
            resolver=path_resolver,
            use_cache=False,  # 路径可能会根据上下文变化，通常不建议缓存
        )
        print(f"--> Custom Hydra resolver '{resolver_name}' registered successfully.")


def load_auxiliary_dataset(
    main_cfg: "DictConfig", dataset_name: str
) -> "pd.DataFrame | None":
    """
    一个通用的辅助函数，用于加载一个“辅助”数据集的权威DTI文件。

    它通过临时创建一个专注于该辅助数据集的配置副本来获取正确的路径，
    从而避免污染主配置。

    Args:
        main_cfg (DictConfig): 当前实验的主配置对象。
        dataset_name (str): 要加载的辅助数据集的名称 (必须与
                              `conf/data_structure/` 下的文件名对应)。

    Returns:
        pd.DataFrame | None: 如果成功加载，则返回DataFrame；否则返回None。
    """
    print(f"--> Attempting to load AUXILIARY dataset: '{dataset_name}'")

    try:
        # 1. 获取主配置的 config_path，以便 initialize_config_dir 知道去哪里找
        #    Hydra 在运行时会将原始的 .hydra 目录信息存储在主配置中
        config_path = Path(main_cfg.hydra.runtime.config_dir).parent

        # 2. 【核心技巧封装】利用Hydra的 API 临时加载辅助数据集的配置
        with hydra.initialize_config_dir(
            config_dir=str(config_path), version_base=None
        ):
            aux_cfg = hydra.compose(
                config_name="config", overrides=[f"data_structure={dataset_name}"]
            )

        # 3. 使用这个临时配置和 get_path 来获取路径
        aux_dti_path = get_path(aux_cfg, "raw.authoritative_dti")
        if not aux_dti_path.exists():
            print(
                f"    - ❌ ERROR: Authoritative file for '{dataset_name}' not found at '{aux_dti_path}'. Skipping."
            )
            return None  # 或者直接 raise FileNotFoundError
        # 4. 加载数据
        aux_df = pd.read_csv(aux_dti_path)
        print(
            f"    - Success: Loaded {len(aux_df)} interactions from '{dataset_name}'."
        )
        return aux_df

    except FileNotFoundError:
        print(
            f"    - ⚠️ WARNING: Authoritative file for '{dataset_name}' not found at '{aux_dti_path}'. Skipping."
        )
        return None
    except Exception as e:
        # 捕获其他可能的错误，如配置缺失、compose失败等
        print(
            f"    - ❌ ERROR: Failed to load auxiliary dataset '{dataset_name}'. Skipping. Reason: {e}"
        )
        return None
