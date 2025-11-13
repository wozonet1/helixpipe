# 文件: src/helixpipe/utils/log.py (升级版)

from functools import wraps
from typing import Any

import pandas as pd


def log_pipeline_step(step_name: str, verbose: int):
    """
    一个通用的装饰器工厂，用于包装数据处理流水线中的一个步骤。
    它在执行前后打印DataFrame的大小变化。
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # 假设被包装的函数处理的第一个核心数据对象是 args[0]
            # (可能是DataFrame, dict, 或其他)
            data_in = args[0] if args else None

            # 记录输入状态
            if verbose > 0:
                print(f"\n-> Entering Step: '{step_name}'...")
                if verbose > 1 and isinstance(data_in, (pd.DataFrame, dict, list, set)):
                    try:
                        print(f"  - Input size: {len(data_in)} items")
                    except TypeError:
                        pass  # 有些对象可能没有len

            # 执行原始函数
            result = func(*args, **kwargs)

            # 记录输出状态
            if verbose > 0:
                # 假设返回值是核心数据对象
                if isinstance(result, (pd.DataFrame, dict, list, set)):
                    try:
                        print(f"  - Output size: {len(result)} items")
                    except TypeError:
                        pass
                if (
                    verbose > 1
                    and isinstance(result, pd.DataFrame)
                    and not result.empty
                ):
                    # 只有在最高verbose级别才打印样本，避免刷屏
                    print(f"  - Output sample:\n{result.head().to_string()}")

            return result

        return wrapper

    return decorator


# 原有的 @log_step 也可以保留，以兼容旧代码
def log_step(step_name: str):
    """
    (旧版) 装饰器工厂，用于自动记录数据处理步骤前后的DataFrame变化。
    它会从被装饰的函数的第一个参数 (通常是'self') 中获取verbose级别。
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
            if self.verbose > 0:
                print(f"\n-> Entering Step: '{step_name}'...")
                if self.verbose > 1:
                    print(f"  - Input rows: {len(df)}")
            result_df = func(self, df, *args, **kwargs)
            if self.verbose > 0:
                print(f"  - Output rows: {len(result_df)}")
            return result_df

    return decorator
