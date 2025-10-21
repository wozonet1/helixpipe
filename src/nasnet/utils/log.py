# 文件: src/data_processing/decorators.py (全新)

from functools import wraps

import pandas as pd


def log_step(step_name: str):
    """
    一个装饰器工厂，用于自动记录数据处理步骤前后的DataFrame变化。
    它会从被装饰的函数的第一个参数 (通常是'self'，一个Processor实例)
    中获取verbose级别。
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
            # `self` 是 Processor 实例, `df` 是输入的DataFrame

            # 在执行前记录
            if self.verbose > 0:
                print(f"\n-> Entering Step: '{step_name}'...")
                if self.verbose > 1:
                    print(f"  - Input rows: {len(df)}")
                    print(f"  - Input:\n{df.to_string()}")

            # 执行原始的处理函数
            result_df = func(self, df, *args, **kwargs)

            # 在执行后记录
            if self.verbose > 0:
                print(f"  - Output rows: {len(result_df)}")
                if self.verbose > 1 and not result_df.empty:
                    print(f"  - Output sample:\n{result_df.to_string()}")

            return result_df

        return wrapper

    return decorator
