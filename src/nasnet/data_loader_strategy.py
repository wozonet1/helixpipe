# 文件: src/data_utils/data_loader_strategy.py (全新)

import importlib
import sys
from typing import List, Tuple

import hydra
import pandas as pd
import research_template as rt
from omegaconf import DictConfig

# 导入我们的抽象基类，用于类型检查和文档
from nasnet.data_processing import BaseDataProcessor


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
        # 【核心修正】在循环外，一次性地获取配置目录的绝对路径
        try:
            project_root = rt.get_project_root()
            config_dir = str(project_root / "conf")
        except Exception as e:
            print(f"❌ 无法确定项目根目录或配置路径。错误: {e}")
            sys.exit(1)

        for name in aux_dataset_names:
            print(f"    - Composing temporary config for '{name}'...")

            # 【核心修正】使用 initialize_config_dir 和绝对路径
            with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
                # 【最终修正】直接把主config中所有相关的顶层配置，都作为覆盖传递下去
                # 这确保了所有上下文都得到了保留
                overrides_list = [
                    f"data_structure={name}",
                    f"data_params={config.data_params.name}",
                    f"global_paths.data_root={config.global_paths.data_root}",  # <--- 明确传递
                    f"runtime.verbose={config.runtime.verbose}",  # <--- 传递verbose
                ]

                aux_config = hydra.compose(
                    config_name="config", overrides=overrides_list
                )

            # 将这个为辅助数据集量身定做的aux_config传递给_run_processor
            aux_df = _run_processor(name, aux_config)
            if not aux_df.empty:
                extra_dfs.append(aux_df)
    else:
        print("\n--> No auxiliary datasets specified.")

    print("\n" + "=" * 80)
    print(" " * 24 + "Data Loading Strategy Complete")
    print("=" * 80)

    return base_df, extra_dfs
