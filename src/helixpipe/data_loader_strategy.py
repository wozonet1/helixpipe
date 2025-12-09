# src/helixpipe/data_loader_strategy.py

import importlib
import logging
from typing import Type, cast

import hydra
import pandas as pd
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

# [修改] 导入通用库
import helixlib as hx

# 导入基类用于类型检查
from helixpipe.data_processing.datasets.base_processor import BaseProcessor

# [修改] 从 typing 导入
from helixpipe.typing import AppConfig, ProcessorOutputs

logger = logging.getLogger(__name__)


def _run_processor(name: str, config: AppConfig) -> pd.DataFrame:
    """
    【V3 - 类型安全版】
    动态实例化并运行数据处理器。
    """
    # 1. 根据命名约定，动态构建类名和模块路径
    class_name = f"{name.capitalize()}Processor"
    module_name = f"{name}_processor"
    module_path = f"helixpipe.data_processing.datasets.{module_name}"

    logger.info(
        f"\n--- [Strategy] Attempting to run processor '{class_name}' from '{module_path}' ---"
    )

    try:
        # 2. 动态导入模块
        module = importlib.import_module(module_path)

        # [关键修复 for MyPy]
        # getattr 返回 Any，导致 MyPy 认为后续所有操作都是 Any。
        # 我们使用 cast 明确告诉 MyPy：这是一个 BaseProcessor 的子类。
        ProcessorClass_untyped = getattr(module, class_name)
        ProcessorClass = cast(Type[BaseProcessor], ProcessorClass_untyped)

        # 3. 运行时验证 (双重保险)
        if not issubclass(ProcessorClass, BaseProcessor):
            raise TypeError(
                f"Class '{class_name}' found in '{module_path}' is not a valid subclass of BaseProcessor."
            )

        # 4. 实例化
        # MyPy 现在知道 ProcessorClass 是 BaseProcessor 类型，所以构造函数是安全的
        processor_instance = ProcessorClass(config=config)

        # 5. 执行处理
        # MyPy 知道 .process() 返回 pd.DataFrame
        df = processor_instance.process()

        if df is None:
            logger.warning(
                f"⚠️  Warning: Processor '{class_name}' returned None. Defaulting to empty DataFrame."
            )
            return pd.DataFrame()

        return df

    except (ImportError, AttributeError) as e:
        msg = (
            f"❌ FATAL: Could not find or load processor for '{name}'.\n"
            f"   - Searched for class '{class_name}' in module '{module_path}'.\n"
            f"   - Please ensure the file exists and defines the class."
        )
        logger.error(msg)
        # [关键修复] 使用 raise 而不是 sys.exit，让 MyPy 知道这里会中断，不会返回 None
        raise RuntimeError(msg) from e

    except Exception as e:
        msg = f"❌ FATAL: An unexpected error occurred while running processor for '{name}'."
        logger.error(msg)
        # 打印堆栈以便调试
        import traceback

        logger.error(traceback.format_exc())
        # [关键修复] 抛出异常
        raise RuntimeError(msg) from e


def _compose_aux_config(config: AppConfig, aux_dataset_name: str) -> AppConfig:
    """
    组合辅助数据集的临时配置。
    """
    logger.info(f"    - Composing temporary config for '{aux_dataset_name}'...")

    # 继承当前的主 data_params 配置名，确保辅助数据集使用相同的过滤参数
    overrides_list = [
        f"data_structure={aux_dataset_name}",
        f"data_params={config.data_params.name}",
    ]

    # 将 global_paths 传递下去，确保路径解析正确
    global_paths_dict = OmegaConf.to_container(config.global_paths, resolve=True)
    if isinstance(global_paths_dict, dict):
        for key, value in global_paths_dict.items():
            overrides_list.append(f"global_paths.{key}={value}")

    # compose 返回 DictConfig，我们需要 cast 为 AppConfig 以通过类型检查
    # (实际上它们在运行时是兼容的 OmegaConf 对象)
    raw_cfg = hydra.compose(config_name="config", overrides=overrides_list)
    return cast(AppConfig, raw_cfg)


def load_datasets(config: AppConfig) -> ProcessorOutputs:
    """
    【V5 - 最终版】加载主数据集和辅助数据集。
    """
    logger.info("=" * 80)
    logger.info("                      Executing Data Loading Strategy")
    logger.info("=" * 80)

    loaded_datasets: ProcessorOutputs = {}

    # 1. 加载主数据集
    primary_dataset_name = config.data_structure.primary_dataset
    logger.info(f"--> Loading PRIMARY dataset: '{primary_dataset_name}'")

    # 这里的 try-except 是为了捕获 _run_processor 抛出的 RuntimeError
    # 并决定是否在此处终止程序 (通常是的)
    try:
        base_df = _run_processor(primary_dataset_name, config)
        if not base_df.empty:
            loaded_datasets[primary_dataset_name] = base_df
    except RuntimeError:
        # 已经在 _run_processor 中记录了日志，这里再次确认退出
        import sys

        sys.exit(1)

    # 2. 加载辅助数据集
    # 使用 getattr 安全获取，避免 dataclass 没有 .get() 方法的问题
    aux_dataset_names = getattr(config.dataset_collection, "auxiliary_datasets", [])

    if not aux_dataset_names:
        logger.info("\n--> No auxiliary datasets specified.")
    else:
        logger.info(
            f"\n--> Loading {len(aux_dataset_names)} AUXILIARY dataset(s): {aux_dataset_names}"
        )

        if GlobalHydra.instance().is_initialized():
            # 场景A: 在 @hydra.main 环境中
            logger.debug("Running within an existing GlobalHydra context.")
            for name in aux_dataset_names:
                try:
                    aux_config = _compose_aux_config(config, name)
                    aux_df = _run_processor(name, aux_config)
                    if not aux_df.empty:
                        loaded_datasets[name] = aux_df
                except RuntimeError:
                    import sys

                    sys.exit(1)
        else:
            # 场景B: 测试环境 (手动初始化)
            logger.debug("No GlobalHydra context found. Initializing manually.")
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
                        if not aux_df.empty:
                            loaded_datasets[name] = aux_df
            except Exception as e:
                logger.error(
                    f"FATAL: Failed during manual Hydra initialization. Error: {e}"
                )
                raise

    logger.info("=" * 80)
    logger.info("                      Data Loading Strategy Complete")
    logger.info("=" * 80)

    return loaded_datasets
