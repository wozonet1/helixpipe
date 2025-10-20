# 文件: src/data_utils/data_loader_strategy.py (全新)

import importlib
import sys
import pandas as pd
from omegaconf import DictConfig
from typing import List, Tuple
import hydra

# 导入我们的抽象基类，用于类型检查和文档
from data_processing.base_processor import BaseDataProcessor


def _run_processor(name: str, config: DictConfig) -> pd.DataFrame:
    """
    一个动态实例化并运行任何数据处理器的辅助函数。

    它会根据给定的处理器名称，自动查找、导入并运行对应的Processor类。

    Args:
        name (str): 处理器的名称 (例如, "bindingdb", "gtopdb")。
                    这必须与对应的文件名和类名匹配。
        config (DictConfig): 适用于该处理器的、完整的Hydra配置对象。

    Returns:
        pd.DataFrame: 由该处理器处理和返回的黄金标准DataFrame。
                      如果处理失败或无数据，则返回一个空的DataFrame。
    """
    try:
        # 1. 根据命名约定，动态构建模块和类的路径
        #    例如: "bindingdb" -> "BindingDBProcessor"
        class_name = f"{name.capitalize()}Processor"
        #    例如: "src.data_processing.bindingdb_processor"
        module_path = f"data_processing.{name}_processor"

        print(
            f"\n--- [Strategy] Attempting to run processor '{class_name}' from '{module_path}' ---"
        )

        # 2. 动态导入模块并获取类定义
        module = importlib.import_module(module_path)
        ProcessorClass = getattr(module, class_name)

        # 3. (可选但推荐) 验证该类是否是我们期望的类型
        if not issubclass(ProcessorClass, BaseDataProcessor):
            raise TypeError(
                f"'{class_name}' is not a valid subclass of BaseDataProcessor."
            )

        # 4. 实例化处理器，并将专属的config“注入”进去
        processor_instance = ProcessorClass(config=config)

        # 5. 调用统一的接口，执行处理流程！
        #    基类的process方法会自动处理缓存检查、保存和验证。
        df = processor_instance.process()

        if df is None:  # 额外的安全检查
            print(
                f"⚠️  Warning: Processor '{class_name}' returned None. Defaulting to empty DataFrame."
            )
            return pd.DataFrame()

        return df

    except (ImportError, AttributeError) as e:
        print(f"❌ FATAL: Could not find or load processor for '{name}'.")
        print(f"   - Searched for class '{class_name}' in module '{module_path}'.")
        print(
            f"   - Please ensure the file 'src/data_processing/{name}_processor.py' exists and contains the class '{class_name}'."
        )
        print(f"   - Original error: {e}")
        sys.exit(1)
    except Exception:
        print(
            f"❌ FATAL: An unexpected error occurred while running processor for '{name}'."
        )
        import traceback

        traceback.print_exc()
        sys.exit(1)


def load_datasets(config: DictConfig) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    """
    【核心策略函数】根据主配置，加载主数据集和所有指定的辅助数据集。

    这是连接'run.py'和具体'Processor'的桥梁。

    Args:
        config (DictConfig): 从主入口（如run.py）传入的、经过Hydra组合后的完整配置对象。

    Returns:
        Tuple[pd.DataFrame, List[pd.DataFrame]]:
            - base_df: 主数据集的DataFrame。
            - extra_dfs: 包含所有辅助数据集DataFrame的列表。
    """
    print("\n" + "=" * 80)
    print(" " * 22 + "Executing Data Loading Strategy")
    print("=" * 80)

    # --- 1. 确定并加载主数据集 ---
    # 主数据集由 data_structure 配置组的选择决定
    primary_dataset_name = config.data_structure.name
    print(f"--> Loading PRIMARY dataset: '{primary_dataset_name}'")

    # 直接将主config传递给_run_processor，因为它已经为主要任务配置好了
    base_df = _run_processor(primary_dataset_name, config)

    # --- 2. 按需加载所有辅助数据集 ---
    extra_dfs = []
    aux_dataset_names = config.data_params.get("auxiliary_datasets", [])

    if aux_dataset_names:
        print(
            f"\n--> Loading {len(aux_dataset_names)} AUXILIARY dataset(s): {aux_dataset_names}"
        )
        for name in aux_dataset_names:
            # 【关键步骤】为每个辅助数据集创建一个临时的、专属的配置上下文
            # 这个上下文的唯一目的，就是让 get_path 函数能够定位到正确的数据集目录
            print(f"    - Composing temporary config for '{name}'...")
            with hydra.initialize(config_path="../conf", version_base=None):
                # 我们从主'config.yaml'开始组合，然后应用两个关键的覆盖：
                # 1. 将data_structure强制切换到当前辅助数据集。
                # 2. 保留主实验的数据处理参数 (data_params)。
                aux_config = hydra.compose(
                    config_name="config",
                    overrides=[
                        f"data_structure={name}",
                        f"data_params={config.data_params.name}",
                    ],
                )

            # 将这个【为辅助数据集量身定做的aux_config】传递给_run_processor
            aux_df = _run_processor(name, aux_config)
            if not aux_df.empty:
                extra_dfs.append(aux_df)
    else:
        print("\n--> No auxiliary datasets specified.")

    print("\n" + "=" * 80)
    print(" " * 24 + "Data Loading Strategy Complete")
    print("=" * 80)

    return base_df, extra_dfs
