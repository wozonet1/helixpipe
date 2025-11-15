# 文件: src/data_utils/data_loader_strategy.py (全新)

import importlib
import logging
from typing import cast

import hydra
import pandas as pd
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

import helixlib as hx

# 导入我们的抽象基类，用于类型检查和文档
from helixpipe.data_processing import BaseProcessor
from helixpipe.typing import AppConfig, ProcessorOutputs

logger = logging.getLogger(__name__)


def _run_processor(name: str, config: AppConfig) -> pd.DataFrame:
    """
    【V2 - 适配新架构版】
    一个动态实例化并运行任何数据处理器的辅助函数。

    它会根据给定的处理器名称，自动在 `helixpipe.data_processing.datasets`
    子模块中查找、导入并运行对应的Processor类。
    """

    # 1. 根据命名约定，动态构建类名和模块路径
    #    类名: "bindingdb" -> "BindingdbProcessor"
    class_name = f"{name.capitalize()}Processor"

    #    【核心修改】模块路径现在指向 `datasets` 子模块
    #    模块名: "bindingdb" -> "bindingdb_processor"
    module_name = f"{name}_processor"
    #    完整路径: "helixpipe.data_processing.datasets.bindingdb_processor"
    module_path = f"helixpipe.data_processing.datasets.{module_name}"

    logger.info(
        f"\n--- [Strategy] Attempting to run processor '{class_name}' from '{module_path}' ---"
    )
    try:
        # 2. 动态导入模块并获取类定义
        module = importlib.import_module(module_path)
        ProcessorClass = getattr(module, class_name)

        # 4. 实例化处理器，并将专属的config“注入”进去
        processor_instance = ProcessorClass(config=config)
        if not isinstance(processor_instance, BaseProcessor):
            raise TypeError(
                f"Class '{class_name}' found in '{module_path}' is not a valid subclass of BaseProcessor."
            )
        # 5. 调用统一的接口，执行处理流程
        df = processor_instance.process()

        if df is None:
            logger.warning(
                f"⚠️  Warning: Processor '{class_name}' returned None. Defaulting to empty DataFrame."
            )
            return pd.DataFrame()

        return df

    except (ImportError, AttributeError) as e:
        logger.error(f"❌ FATAL: Could not find or load processor for '{name}'.")
        logger.error(
            f"   - Searched for class '{class_name}' in module '{module_path}'."
        )
        logger.error(
            f"   - Please ensure the file 'src/helixpipe/data_processing/datasets/{name}_processor.py' exists "
            f"and contains the class '{class_name}'."
        )
        logger.error(f"   - Original error: {e}")
        raise RuntimeError(f"Failed to load processor '{name}'") from e

    except Exception:  # 将 Exception 放在最后，捕获所有其他异常
        logger.error(
            f"❌ FATAL: An unexpected error occurred while running processor for '{name}'."
        )
        import traceback

        traceback.print_exc()
        raise RuntimeError(f"Unexpected error in processor '{name}'")


def _compose_aux_config(config: AppConfig, aux_dataset_name: str) -> AppConfig:
    """
    【新辅助函数】一个专门负责组合辅助配置的函数。
    它可以在任何上下文（有或没有 GlobalHydra）中被安全调用。
    """
    logger.info(f"    - Composing temporary config for '{aux_dataset_name}'...")

    overrides_list = [
        f"data_structure={aux_dataset_name}",
        f"data_params={config.data_params.name}",
    ]

    global_paths_dict = OmegaConf.to_container(config.global_paths, resolve=True)
    if not isinstance(global_paths_dict, dict):
        logger.error("failed to load global paths")
        raise RuntimeError
    for key, value in global_paths_dict.items():
        key = str(key)
        overrides_list.append(f"global_paths.{key}={value}")

    return cast(
        AppConfig, hydra.compose(config_name="config", overrides=overrides_list)
    )


def load_datasets(config: AppConfig) -> ProcessorOutputs:
    """
    【V4 - 最终兼容版】根据主配置，加载主数据集和所有指定的辅助数据集。
    能够智能地在 @hydra.main 环境和独立测试环境中工作。
    """
    logger.info("=" * 80)
    logger.info("                      Executing Data Loading Strategy")
    logger.info("=" * 80)

    loaded_datasets = {}

    # 1. 加载主数据集 (逻辑不变)
    primary_dataset_name = config.data_structure.primary_dataset
    logger.info(f"--> Loading PRIMARY dataset: '{primary_dataset_name}'")
    base_df = _run_processor(primary_dataset_name, config)
    if base_df is not None and not base_df.empty:
        loaded_datasets[primary_dataset_name] = base_df

    # 2. 按需加载所有辅助数据集
    aux_dataset_names = getattr(config.dataset_collection, "auxiliary_datasets", [])

    if not aux_dataset_names:
        logger.info("\n--> No auxiliary datasets specified.")
    else:
        logger.info(
            f"\n--> Loading {len(aux_dataset_names)} AUXILIARY dataset(s): {aux_dataset_names}"
        )

        # --- 【核心修正】检查 GlobalHydra 状态 ---
        if GlobalHydra.instance().is_initialized():
            # --- 场景A: 我们在 @hydra.main 环境中 ---
            logger.debug(
                "Running within an existing GlobalHydra context (@hydra.main)."
            )
            for name in aux_dataset_names:
                aux_config = _compose_aux_config(config, name)
                aux_df = _run_processor(name, aux_config)
                if aux_df is not None and not aux_df.empty:
                    loaded_datasets[name] = aux_df
        else:
            # --- 场景B: 我们在测试环境 (或独立脚本) 中 ---
            logger.debug(
                "No GlobalHydra context found. Initializing manually for tests."
            )
            try:
                project_root = hx.get_project_root()
                config_dir_abs_path = str(project_root / "conf")
                with hydra.initialize_config_dir(
                    config_dir=config_dir_abs_path,
                    version_base=None,
                    job_name="load_aux_datasets",
                ):
                    for name in aux_dataset_names:
                        aux_config = _compose_aux_config(config, name)
                        aux_df = _run_processor(name, aux_config)
                        if aux_df is not None and not aux_df.empty:
                            loaded_datasets[name] = aux_df
            except Exception as e:
                logger.error(
                    f"FATAL: Failed during manual Hydra initialization for auxiliary datasets. Error: {e}"
                )
                raise

    logger.info("=" * 80)
    logger.info("                      Data Loading Strategy Complete")
    logger.info("=" * 80)

    return loaded_datasets
