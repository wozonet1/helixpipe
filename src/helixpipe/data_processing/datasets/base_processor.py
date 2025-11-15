# 文件: src/helixpipe/data_processing/datasets/base_processor.py (最终流水线编排版)

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Tuple

import pandas as pd

import helixlib as hx
from helixpipe.typing import AppConfig
from helixpipe.utils import get_path

logger = logging.getLogger(__name__)

Phase = Tuple[str, Callable]


class BaseProcessor(ABC):
    """
    【最终架构版 v2】数据处理器的抽象基类，使用模板方法和流水线编排风格。
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.verbose = config.runtime.verbose
        self.schema = self.config.data_structure.schema.internal.canonical_interaction
        logger.info(
            f"--- [{self.__class__.__name__}] Initialized (Verbose Level: {self.verbose}). ---"
        )

    # --- 模板方法 (Template Method) ---

    def process(self) -> pd.DataFrame:
        """
        不可覆盖的最终处理流程。它以流水线风格编排所有数据处理步骤。
        """
        output_target = get_path(self.config, "raw.authoritative_dti")
        if output_target.exists() and not self.config.runtime.force_restart:
            if self.verbose > 0:
                logger.info(
                    f"--> [Cache Hit] Loading processed data from: '{output_target.name}'"
                )
            return pd.read_csv(output_target)

        # --- 流水线定义 ---
        # 每个步骤都是一个元组 (step_name, step_function)
        pipeline: List[Phase] = [
            ("Load Raw Data", self._load_raw_data),
            ("Extract Relations", self._extract_relations),
            ("Standardize IDs", self._standardize_ids),
            ("Filter Data", self._filter_data),
            ("Finalize & Deduplicate Columns", self._finalize_columns),
        ]

        # --- 流水线执行 ---
        # 'data' 变量将在流水线中流动，其类型可能会变化
        data: Any = None

        for step_name, step_func in pipeline:
            if self.verbose > 0:
                logger.info(f"\n-> Entering Step: '{step_name}'...")
                if self.verbose > 1 and isinstance(
                    data, (pd.DataFrame, dict, list, set)
                ):
                    logger.info(f"  - Input size: {len(data)} items")

            # 执行步骤
            data = step_func(data) if data is not None else step_func()

            # 记录输出并检查是否为空
            if isinstance(data, pd.DataFrame):
                if self.verbose > 0:
                    logger.info(f"  - Output size: {len(data)} items")
                if data.empty:
                    logger.error(
                        f"  - Pipeline halted after step '{step_name}' because DataFrame became empty."
                    )
                    return pd.DataFrame()
            elif data is None or (isinstance(data, (dict, list, set)) and not data):
                logger.error(
                    f"  - Pipeline halted after step '{step_name}' because data became empty."
                )
                return pd.DataFrame()

        # 确保最终结果是DataFrame
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"The pipeline should end with a pandas DataFrame, but got {type(data)}."
            )

        final_df = data

        # --- 保存与验证 ---
        logger.info(
            f"\n--> [{self.__class__.__name__}] Saving processed data to cache: '{output_target.name}'..."
        )
        hx.ensure_path_exists(output_target)
        final_df.to_csv(output_target, index=False)

        if self.verbose > 0:
            logger.info(
                f"--- [{self.__class__.__name__}] Final authoritative file is ready. ---"
            )
            expected_cols_subset = {
                self.schema.source_id,
                self.schema.source_type,
                self.schema.target_id,
                self.schema.target_type,
                self.schema.relation_type,
                self.schema.label,
            }
            assert expected_cols_subset.issubset(set(final_df.columns))

        return final_df

    # --- 子类需要实现的抽象方法 (现在带有输入参数签名) ---

    @abstractmethod
    def _load_raw_data(self) -> Any:
        """从磁盘加载特定格式的原始数据。"""
        raise NotImplementedError

    @abstractmethod
    def _extract_relations(self, raw_data: Any) -> pd.DataFrame:
        """从原始数据中解析并提取出关系DataFrame。"""
        raise NotImplementedError

    @abstractmethod
    def _standardize_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """对DataFrame中的ID列进行标准化。"""
        raise NotImplementedError

    @abstractmethod
    def _filter_data(self, df: pd.DataFrame) -> Any:
        """应用data_params里的进行筛选"""
        raise NotImplementedError

    # --- Base类提供的通用、可复用的步骤 ---

    def _get_final_columns(self) -> List[str]:
        return [
            self.schema.source_id,
            self.schema.source_type,
            self.schema.target_id,
            self.schema.target_type,
            self.schema.relation_type,
            self.schema.label,
        ]

    def _finalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.schema.label not in df.columns:
            df[self.schema.label] = 1

        final_cols = self._get_final_columns()
        cols_to_keep = [col for col in final_cols if col in df.columns]
        final_df = df[cols_to_keep].copy()
        final_df.drop_duplicates(
            subset=[
                self.schema.source_id,
                self.schema.target_id,
                self.schema.relation_type,
            ],
            inplace=True,
            keep="first",
        )
        final_df = final_df.reset_index(drop=True)

        # --- 【核心修复】在这里进行最终的数据类型强制转换 ---
        # 只有在DataFrame不为空时才执行
        if not final_df.empty:
            # ID 列不再强制为 int，因为它们可能是字符串（如UniProt ID）
            # 我们依赖 Processor 来确保ID类型的正确性
            final_df[self.schema.source_id] = final_df[
                self.schema.source_id
            ]  # 保持原样
            final_df[self.schema.target_id] = final_df[
                self.schema.target_id
            ]  # 保持原样

            final_df[self.schema.source_type] = final_df[
                self.schema.source_type
            ].astype(str)
            final_df[self.schema.target_type] = final_df[
                self.schema.target_type
            ].astype(str)
            final_df[self.schema.relation_type] = final_df[
                self.schema.relation_type
            ].astype(str)

            # label 可以是 int (分类) 或 float (回归)，所以我们只做数值转换
            final_df[self.schema.label] = pd.to_numeric(final_df[self.schema.label])

        return final_df
