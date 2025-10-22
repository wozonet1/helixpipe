# 文件: src/data_processing/base_processor.py (V3 - 终极解耦版)

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
import research_template as rt

from nasnet.typing import AppConfig
from nasnet.utils import get_path, validate_authoritative_dti_file

from ..services import get_human_uniprot_whitelist, get_valid_pubchem_cids


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
        return get_path(self.config, "raw.authoritative_dti")

    @abstractmethod
    def _load_raw_data(self) -> pd.DataFrame:
        """
        【新抽象方法】子类必须实现这个方法，只负责从磁盘加载原始数据。
        """
        raise NotImplementedError

    @abstractmethod
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【新抽象方法】子类必须实现，负责将原始列名映射到内部标准列名。
        """
        raise NotImplementedError

    @abstractmethod
    def _transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【新抽象方法】子类必须实现这个方法，负责所有特定于源的转换逻辑。
        它接收的是已经过白名单过滤的DataFrame。
        """
        raise NotImplementedError

    def process(self) -> pd.DataFrame:
        """
        【V2 模板方法】外部统一入口，现在包含了ID验证的核心逻辑。
        """
        output_target = self.output_path
        if output_target.exists() and not self.config.runtime.force_restart:
            # ... (缓存命中逻辑不变)
            return pd.read_csv(output_target)

        # --- 1. 加载原始数据 ---
        raw_df = self._load_raw_data()
        if raw_df.empty:
            return pd.DataFrame()
        standardized_df = self._standardize_columns(raw_df)
        # --- 2. 【核心变化】执行全局ID验证 ---
        print(
            f"--- [{self.__class__.__name__}] Performing ID whitelist filtering... ---"
        )
        schema = self.config.data_structure.schema.internal.authoritative_dti

        # a. 验证 UniProt IDs
        all_pids = set(standardized_df[schema.protein_id].dropna().unique())
        valid_pids = get_human_uniprot_whitelist(all_pids, self.config)
        df_filtered = raw_df[raw_df[schema.protein_id].isin(valid_pids)]

        # b. 验证 PubChem CIDs
        all_cids = set(df_filtered[schema.molecule_id].dropna().unique())
        valid_cids = get_valid_pubchem_cids(all_cids, self.config)
        df_filtered = df_filtered[df_filtered[schema.molecule_id].isin(valid_cids)]

        print(f"--> After ID whitelisting, {len(df_filtered)} rows remain.")
        if df_filtered.empty:
            return pd.DataFrame()

        # --- 3. 执行特定于子类的转换逻辑 ---
        final_df = self._transform_data(df_filtered)

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
