# 文件: src/data_utils/data_loader_strategy.py (全新)

import importlib
import sys
from typing import List, Tuple

import hydra
import pandas as pd

from helixpipe.configs import AppConfig

# 导入我们的抽象基类，用于类型检查和文档
from helixpipe.data_processing import BaseProcessor


def _run_processor(name: str, config: AppConfig) -> pd.DataFrame:
    """
    【V2 - 适配新架构版】
    一个动态实例化并运行任何数据处理器的辅助函数。

    它会根据给定的处理器名称，自动在 `helixpipe.data_processing.datasets`
    子模块中查找、导入并运行对应的Processor类。
    """
    try:
        # 1. 根据命名约定，动态构建类名和模块路径
        #    类名: "bindingdb" -> "BindingdbProcessor"
        class_name = f"{name.capitalize()}Processor"

        #    【核心修改】模块路径现在指向 `datasets` 子模块
        #    模块名: "bindingdb" -> "bindingdb_processor"
        module_name = f"{name}_processor"
        #    完整路径: "helixpipe.data_processing.datasets.bindingdb_processor"
        module_path = f"helixpipe.data_processing.datasets.{module_name}"

        print(
            f"\n--- [Strategy] Attempting to run processor '{class_name}' from '{module_path}' ---"
        )

        # 2. 动态导入模块并获取类定义
        module = importlib.import_module(module_path)
        ProcessorClass = getattr(module, class_name)

        # 3. 验证该类是否是我们期望的类型
        if not issubclass(ProcessorClass, BaseProcessor):
            raise TypeError(
                f"Class '{class_name}' found in '{module_path}' is not a valid subclass of BaseProcessor."
            )

        # 4. 实例化处理器，并将专属的config“注入”进去
        processor_instance = ProcessorClass(config=config)

        # 5. 调用统一的接口，执行处理流程
        df = processor_instance.process()

        if df is None:
            print(
                f"⚠️  Warning: Processor '{class_name}' returned None. Defaulting to empty DataFrame."
            )
            return pd.DataFrame()

        return df

    except (ImportError, AttributeError) as e:
        print(f"❌ FATAL: Could not find or load processor for '{name}'.")
        print(f"   - Searched for class '{class_name}' in module '{module_path}'.")
        print(
            f"   - Please ensure the file 'src/helixpipe/data_processing/datasets/{name}_processor.py' exists "
            f"and contains the class '{class_name}'."
        )
        print(f"   - Original error: {e}")
        sys.exit(1)

    except Exception:  # 将 Exception 放在最后，捕获所有其他异常
        print(
            f"❌ FATAL: An unexpected error occurred while running processor for '{name}'."
        )
        import traceback

        traceback.print_exc()
        sys.exit(1)


def load_datasets(config: AppConfig) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
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
    loaded_datasets = {}
    # --- 1. 确定并加载主数据集 ---
    # 主数据集由 data_structure 配置组的选择决定
    primary_dataset_name = config.data_structure.name
    print(f"--> Loading PRIMARY dataset: '{primary_dataset_name}'")
    base_df = _run_processor(primary_dataset_name, config)
    loaded_datasets[primary_dataset_name] = base_df
    # 直接将主config传递给_run_processor，因为它已经为主要任务配置好了
    base_df = _run_processor(primary_dataset_name, config)

    # --- 2. 按需加载所有辅助数据集 ---
    aux_dataset_names = config.dataset_collection.get("auxiliary_datasets", [])

    if aux_dataset_names:
        print(
            f"\n--> Loading {len(aux_dataset_names)} AUXILIARY dataset(s): {aux_dataset_names}"
        )
        # [REMOVED] 不再需要手动查找 config_dir
        # try:
        #     project_root = rt.get_project_root()
        #     config_dir = str(project_root / "conf")
        # except Exception as e:
        #     print(f"❌ 无法确定项目根目录或配置路径。错误: {e}")
        #     sys.exit(1)

        for name in aux_dataset_names:
            print(f"    - Composing temporary config for '{name}'...")

            # [REMOVED] 移除整个 with hydra.initialize_config_dir(...) 语句块
            # with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):

            # [MODIFIED] 直接在循环内部构建覆盖列表和调用 compose
            overrides_list = [
                f"data_structure={name}",
                # [IMPROVEMENT] 我们不应该只传递几个固定的上下文，
                # 而是应该让辅助配置继承主配置的大部分内容，只覆盖 data_structure。
                # 但为了最小化改动，我们先保持您原来的逻辑。
                f"data_params={config.data_params.name}",
                f"global_paths.data_root={config.global_paths.data_root}",
                f"runtime.verbose={config.runtime.verbose}",
            ]

            # 直接调用 hydra.compose，它会在当前已初始化的环境中工作
            aux_config = hydra.compose(config_name="config", overrides=overrides_list)

            # 将这个为辅助数据集量身定做的aux_config传递给_run_processor
            aux_df = _run_processor(name, aux_config)
            if not aux_df.empty:
                loaded_datasets[name] = aux_df
    else:
        print("\n--> No auxiliary datasets specified.")

    print("\n" + "=" * 80)
    print(" " * 24 + "Data Loading Strategy Complete")
    print("=" * 80)

    return loaded_datasets
