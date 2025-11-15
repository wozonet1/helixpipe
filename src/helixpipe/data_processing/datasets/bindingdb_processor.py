# 文件: src/helixpipe/data_processing/datasets/bindingdb_processor.py (最终模板方法版)

import logging
import sys
from typing import cast

import argcomplete
import pandas as pd
from hydra import compose
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import helixlib as hx
from helixpipe.configs import register_all_schemas
from helixpipe.typing import AppConfig
from helixpipe.utils import SchemaAccessor, get_path, register_hydra_resolvers

from .base_processor import BaseProcessor

logger = logging.getLogger(__name__)


class BindingdbProcessor(BaseProcessor):
    """
    【V8 - 最终纯净架构版】
    一个专门负责处理BindingDB原始数据的处理器。
    它的职责被纯化为：
    1. 从原始TSV加载数据。
    2. 将数据“翻译”成内部的、带有辅助信息的规范化交互格式。
    3. 执行BindingDB专属的“亲和力阈值”领域筛选。
    """

    def __init__(self, config: AppConfig):
        super().__init__(config)
        # 引用外部schema，用于解析原始文件
        schema_node = config.data_structure.schema.external["bindingdb"]
        self.external_schema = SchemaAccessor(schema_node)

    def _load_raw_data(self) -> pd.DataFrame:
        """
        步骤1: 从BindingDB的原始TSV文件中加载人类相关的数据。
        (此方法逻辑与之前版本基本一致)
        """
        tsv_path = get_path(self.config, "raw.raw_tsv")
        if not tsv_path.exists():
            raise FileNotFoundError(f"Raw TSV file not found at '{tsv_path}'")

        columns_to_read = list(self.external_schema.values())

        # 确保读取所有可能需要的列，包括亲和力和结构
        required_cols_for_processing = set(columns_to_read)

        chunk_iterator = pd.read_csv(
            tsv_path,
            sep="\t",
            on_bad_lines="warn",
            usecols=lambda c: c in required_cols_for_processing,
            low_memory=False,
            chunksize=100000,
        )

        loaded_chunks = []
        disable_tqdm = self.verbose == 0
        for chunk in tqdm(
            chunk_iterator,
            desc=f"    - Reading & Pre-filtering {self.__class__.__name__} Chunks",
            disable=disable_tqdm,
        ):
            # 基础的、绝对必要的预过滤
            chunk = chunk[
                chunk[self.external_schema.get_col("organism")] == "Homo sapiens"
            ].copy()
            chunk.dropna(
                subset=[
                    self.external_schema.get_col("protein_id"),
                    self.external_schema.get_col("molecule_id"),
                ],
                inplace=True,
            )
            if not chunk.empty:
                loaded_chunks.append(chunk)

        return (
            pd.concat(loaded_chunks, ignore_index=True)
            if loaded_chunks
            else pd.DataFrame()
        )

    def _extract_relations(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        步骤2: 对于BindingDB，原始数据已是关系格式，此步骤为直通。
        """
        return raw_data

    def _standardize_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        步骤3: 将原始DataFrame重塑为“规范化交互格式”，并附带所有用于下游校验的辅助列。
        """
        if self.verbose > 0:
            logger.info(
                "  - Reshaping DataFrame to canonical format with auxiliary columns..."
            )

        final_df = pd.DataFrame()

        # 核心规范化列
        entity_names = self.config.knowledge_graph.entity_types
        final_df[self.schema.source_id] = df[
            self.external_schema.get_col("molecule_id")
        ]
        final_df[self.schema.source_type] = entity_names.drug
        final_df[self.schema.target_id] = df[self.external_schema.get_col("protein_id")]
        final_df[self.schema.target_type] = entity_names.protein
        final_df[self.schema.relation_type] = (
            self.config.knowledge_graph.relation_types.default
        )

        # 附带所有下游需要的“原材料”
        final_df["structure_molecule"] = df[
            self.external_schema.get_col("molecule_sequence")
        ]
        final_df["structure_protein"] = df[
            self.external_schema.get_col("protein_sequence")
        ]

        # 附带亲和力数据以供下一步过滤
        for aff_type in [
            self.external_schema.get_col("ki"),
            self.external_schema.get_col("ic50"),
            self.external_schema.get_col("kd"),
        ]:
            if aff_type in df.columns:
                final_df[aff_type] = df[aff_type]

        return final_df

    def _filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        步骤4: 【职责纯化】只执行BindingDB领域专属的“亲和力”过滤。
        """
        if self.verbose > 0:
            logger.info("  - Applying domain-specific filter: Affinity Threshold...")

        # 1. 计算统一的亲和力值
        for aff_type in [
            self.external_schema.get_col("ki"),
            self.external_schema.get_col("ic50"),
            self.external_schema.get_col("kd"),
        ]:
            if aff_type in df.columns:
                df[aff_type] = pd.to_numeric(
                    df[aff_type].astype(str).str.replace("[><]", "", regex=True),
                    errors="coerce",
                )

        df["affinity_nM"] = (
            df[self.external_schema.get_col("ki")]
            .fillna(df[self.external_schema.get_col("kd")])
            .fillna(df[self.external_schema.get_col("ic50")])
        )

        # 2. 应用亲和力阈值过滤
        affinity_threshold = self.config.data_params.affinity_threshold_nM

        # 在过滤前，先丢弃没有计算出 affinity_nM 的行
        df.dropna(subset=["affinity_nM"], inplace=True)

        df_filtered = df[df["affinity_nM"] <= affinity_threshold].copy()

        if self.verbose > 0:
            logger.info(
                f"    - {len(df_filtered)} / {len(df)} records passed affinity filter."
            )

        return df_filtered


# --------------------------------------------------------------------------
# Config Store模式下的独立运行入口 (最终版)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # === 阶段 1: 明确定义脚本身份和命令行接口 ===
    from argparse import ArgumentParser

    from hydra import compose, initialize_config_dir

    # a. 这个脚本的固有基础配置
    #    它天生就是用来处理 bindingdb 数据结构的
    BASE_OVERRIDES = ["data_structure=bindingdb"]

    # b. 设置命令行解析器，只接收用户自定义的覆盖参数
    parser = ArgumentParser(
        description="Run the BindingDB processing pipeline with custom Hydra overrides."
    )
    # 'nargs='*' 会将所有额外的命令行参数都收集到一个列表中
    parser.add_argument(
        "user_overrides",
        nargs="*",
        help="Hydra overrides (e.g., data_params=strict_strong)",
    )
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    # c. 组合所有覆盖参数
    final_overrides = BASE_OVERRIDES + args.user_overrides

    # === 阶段 2: 注册、加载配置并打印 ===

    # a. 在所有Hydra操作之前，确保解析器已注册
    #    (hx 是 research_template)
    register_hydra_resolvers()
    register_all_schemas()
    # b. 使用 initialize_config_dir 和 compose 来构建最终的配置对象
    try:
        # get_project_root 在非Hydra应用下会使用 Path.cwd()
        # 请确保您是从项目根目录运行此脚本
        project_root = hx.get_project_root()
        config_dir = str(project_root / "conf")
    except Exception as e:
        logger.error(f"❌ 无法确定项目根目录或配置路径。错误: {e}")
        sys.exit(1)  # 明确退出

    with initialize_config_dir(
        config_dir=config_dir, version_base=None, job_name="bindingdb_process"
    ):
        cfg: DictConfig = compose(config_name="config", overrides=final_overrides)

    # c. 打印最终配置以供调试
    logger.info("\n" + "~" * 80)
    logger.info(" " * 25 + "HYDRA COMPOSED CONFIGURATION")
    logger.info("~" * 80)
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("~" * 80 + "\n")

    # === 阶段 3: 执行核心业务逻辑 ===
    cfg = cast(AppConfig, cfg)
    # a. 实例化处理器
    processor = BindingdbProcessor(config=cfg)

    # b. 运行处理流程
    #    基类的 .process() 方法会自动处理缓存、调用 _load_raw_data, _transform_data,
    #    以及最终的保存和验证。
    final_df = processor.process()

    # c. 打印最终总结
    if final_df is not None and not final_df.empty:
        logger.info(
            "\n✅ BindingDB processing complete. Final authoritative file is ready."
        )
    else:
        logger.warning("\n⚠️  BindingDB processing resulted in an empty dataset.")
