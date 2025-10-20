# 文件: src/data_processing/base_processor.py (V3 - 终极解耦版)

from abc import ABC, abstractmethod
import pandas as pd
import research_template as rt
from data_utils.debug_utils import validate_authoritative_dti_file
from pathlib import Path
from configs.register_schemas import AppConfig


class BaseDataProcessor(ABC):
    def __init__(self, config: AppConfig):
        self.config = config
        self.verbose = config.runtime.verbose
        if self.verbose > 0:
            print(
                f"--- [{self.__class__.__name__}] Initialized (Verbose Level: {self.verbose}). ---"
            )
        else:
            print(f"--- [{self.__class__.__name__}] Initialized. ---")

    @property
    def output_path(self) -> Path:
        """
        【契约实现】直接使用self.config来动态解析路径。
        """
        return rt.get_path(self.config, "data_structure.paths.raw.authoritative_dti")

    @abstractmethod
    def _process_raw_data(self) -> pd.DataFrame:
        """
        【抽象方法】子类【必须】实现这个“工人”方法。
        """
        raise NotImplementedError

    def process(self) -> pd.DataFrame:
        """
        【模板方法】外部统一入口，封装了缓存逻辑。
        这个方法不应该被子类覆盖。
        """
        output_target = self.output_path

        if output_target.exists() and not self.config.runtime.force_restart:
            if self.verbose > 0:
                print(
                    f"--> [{self.__class__.__name__}] Cache hit! Loading from '{output_target.name}'..."
                )
            df = pd.read_csv(output_target)
            self.validate(df, output_target, is_cached=True)
            return df

        if self.verbose > 0:
            print(
                f"--> [{self.__class__.__name__}] Cache miss or force_restart=True. Starting raw data processing..."
            )

        final_df = self._process_raw_data()

        if final_df is None or final_df.empty:
            return pd.DataFrame()

        print(
            f"--> [{self.__class__.__name__}] Saving processed data to cache: '{output_target.name}'..."
        )
        rt.ensure_path_exists(output_target)
        final_df.to_csv(output_target, index=False)

        self.validate(final_df, output_target, is_cached=False)

        return final_df

    def validate(self, df: pd.DataFrame, file_path: Path, is_cached: bool = False):
        """验证工具函数，现在也接收文件路径用于删除损坏文件。"""
        # ... (validate的逻辑基本不变，只是错误处理时使用传入的file_path)
        if self.verbose == 0:  # 0级时完全跳过验证的打印
            return

        if is_cached:
            print(
                f"--- [{self.__class__.__name__}] Running quick validation on cached data... ---"
            )
        else:
            print(
                f"\n--- [{self.__class__.__name__}] Running full validation on newly processed data... ---"
            )
        try:
            validate_authoritative_dti_file(self.config, df=df, verbose=self.verbose)
        except Exception as e:
            if file_path.exists():
                file_path.unlink()
            raise e
