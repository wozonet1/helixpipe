# 文件: src/data_processing/base_processor.py

from abc import ABC, abstractmethod
import pandas as pd
from configs.register_schemas import AppConfig


class BaseDataProcessor(ABC):
    """
    数据处理器抽象基类 (Abstract Base Class)。

    这是一个“契约”，规定了所有具体的数据处理器（如BindingDB, TDC, GtoPdb）
    都必须遵循的结构和接口。

    它的核心职责是：将某个特定来源的、原始、混乱的数据，处理成一个
    遵循项目内部“黄金标准”格式的、干净的DataFrame。
    """

    def __init__(self, config: AppConfig):
        """
        初始化处理器。

        Args:
            config (DictConfig): 实验的完整Hydra配置对象。
                                 处理器可以从中获取它需要的所有参数，
                                 如文件路径、过滤阈值、schema定义等。
        """
        self.config = config
        print(f"--- [{self.__class__.__name__}] Initialized. ---")

    @abstractmethod
    def process(self) -> pd.DataFrame:
        """
        执行完整数据处理流水线的主方法。

        这个方法必须被所有子类实现。它应该包含加载、过滤、净化、标准化
        等所有必要步骤。

        Returns:
            pd.DataFrame: 一个干净的、遵循项目内部黄金标准schema的DataFrame。
                          即使处理失败或没有数据，也应返回一个空的、但列名正确
                          的DataFrame，而不是None。
        """
        raise NotImplementedError

    def validate(self, df: pd.DataFrame):
        """
        一个可选但强烈推荐的步骤，用于在处理完成后进行最终的质量检验。
        """
        # 假设 validate_authoritative_dti_file 存在于 debug_utils 中
        from data_utils.debug_utils import validate_authoritative_dti_file

        print(
            f"\n--- [{self.__class__.__name__}] Running final validation on processed data... ---"
        )
        try:
            # 调用我们之前设计的通用验证器
            validate_authoritative_dti_file(self.config, df=df)
        except Exception as e:
            print(
                f"❌ FATAL: Validation failed for data processed by {self.__class__.__name__}!"
            )
            raise e
