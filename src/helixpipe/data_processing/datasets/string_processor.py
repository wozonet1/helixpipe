# src/helixpipe/data_processing/datasets/string_processor.py
import logging
from typing import cast, dict

import pandas as pd

import helixlib as hx
from helixpipe.configs import register_all_schemas
from helixpipe.typing import AppConfig
from helixpipe.utils import SchemaAccessor, get_path, register_hydra_resolvers

from .base_processor import BaseProcessor

logger = logging.getLogger(__name__)


class StringProcessor(BaseProcessor):
    """
    【V3 - 最终纯净架构版】
    专门处理 STRING DB 原始数据的处理器。
    它的职责被纯化为：
    1. 从原始 .gz 文件加载 links 和 aliases 数据。
    2. 将 STRING ID 映射为 UniProt ID。
    3. 将数据“翻译”成内部的规范化交互格式。
    """

    def __init__(self, config: AppConfig) -> None:
        super().__init__(config)
        self.external_schema = SchemaAccessor(
            self.config.data_structure.schema.external["string"]
        )
        # 在初始化时，就加载并准备好作为内部状态的ID映射字典
        self._id_map: dict[str, str] = self._build_id_map()

    def _build_id_map(self) -> dict[str, str]:
        """一个私有辅助方法，用于构建并返回 STRING ID -> UniProt ID 的映射字典。"""
        aliases_path = get_path(self.config, "raw.protein_aliases")
        if not aliases_path.exists():
            raise FileNotFoundError(
                f"STRING aliases file not found at '{aliases_path}'"
            )

        if self.verbose > 0:
            logger.info(
                f"  - [StringProcessor Pre-computation] Building ID map from '{aliases_path.name}'..."
            )

        aliases_df = pd.read_csv(aliases_path, sep="\t", compression="gzip")

        string_id_col = self.external_schema.get_col("string_protein_id")
        alias_col = self.external_schema.get_col("alias")
        source_col = self.external_schema.get_col("source")

        # 使用 .rename() 清理带'#'的列名，以便后续方便地使用属性访问
        clean_string_id_col = string_id_col.lstrip("#")
        aliases_df.rename(columns={string_id_col: clean_string_id_col}, inplace=True)

        # 过滤出包含 UniProt Accession (AC) 的来源
        uniprot_aliases = aliases_df[
            aliases_df[source_col].str.contains("UniProt_AC")
        ].copy()

        # 去重并创建映射字典
        uniprot_aliases.drop_duplicates(
            subset=[clean_string_id_col, alias_col], inplace=True
        )
        id_map_df = uniprot_aliases.drop_duplicates(
            subset=[clean_string_id_col], keep="first"
        )
        id_map = pd.Series(
            id_map_df[alias_col].values, index=id_map_df[clean_string_id_col]
        ).to_dict()

        if self.verbose > 0:
            logger.info(
                f"    - Map created with {len(id_map)} unique STRING ID entries."
            )

        return id_map

    def _load_raw_data(self) -> pd.DataFrame:
        """
        步骤1: 只加载主要的'links'数据，并返回一个DataFrame。
        """
        links_path = get_path(self.config, "raw.protein_links")
        if not links_path.exists():
            raise FileNotFoundError(f"STRING links file not found at '{links_path}'")

        # 使用 usecols 提前筛选所需列
        protein1_col = self.external_schema.get_col("protein1")
        protein2_col = self.external_schema.get_col("protein2")
        score_col = self.external_schema.get_col("combined_score")

        return pd.read_csv(
            links_path,
            sep="\t",
            compression="gzip",
            usecols=[protein1_col, protein2_col, score_col],
        )

    def _extract_relations(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        步骤2: 利用内部的ID映射字典，将 STRING ID 转换为 UniProt ID。
        """
        # raw_data 现在就是 links_df
        links_df = raw_data

        protein1_col = self.external_schema.get_col("protein1")
        protein2_col = self.external_schema.get_col("protein2")

        # 使用 .map() 进行高效映射
        links_df["protein1_uniprot"] = links_df[protein1_col].map(self._id_map)
        links_df["protein2_uniprot"] = links_df[protein2_col].map(self._id_map)

        # 丢弃任何一个ID未能成功映射的交互
        return links_df.dropna(subset=["protein1_uniprot", "protein2_uniprot"]).copy()

    def _standardize_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        步骤3: 将DataFrame重塑为“规范化交互格式”。
        """
        if self.verbose > 0:
            logger.info("  - Reshaping DataFrame to canonical interaction format...")

        final_df = pd.DataFrame()
        entity_names = self.config.knowledge_graph.entity_types

        # 核心规范化列
        final_df[self.schema.source_id] = df["protein1_uniprot"]
        final_df[self.schema.source_type] = entity_names.protein
        final_df[self.schema.target_id] = df["protein2_uniprot"]
        final_df[self.schema.target_type] = entity_names.protein
        final_df[self.schema.relation_type] = (
            self.config.knowledge_graph.relation_types.ppi
        )

        # STRING数据不包含结构信息，所以我们不附带 structure_* 列

        return final_df

    def _filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        步骤4: 无需进行任何过滤。
        - 置信度过滤已在下载时完成。
        - 通用校验（如ID白名单）将在下游的 EntityValidator 中统一进行。
        """
        if self.verbose > 0:
            logger.info("  - No domain-specific filters to apply for StringProcessor.")

        return df


if __name__ == "__main__":
    from argparse import ArgumentParser

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from helixpipe.typing import AppConfig

    BASE_OVERRIDES = [
        "data_structure=string",
        "data_params=string",
    ]

    parser = ArgumentParser(description="Run the String processing pipeline.")
    parser.add_argument("user_overrides", nargs="*", help="Hydra overrides")
    args = parser.parse_args()
    final_overrides = BASE_OVERRIDES + args.user_overrides

    register_hydra_resolvers()
    register_all_schemas()
    project_root = hx.get_project_root()
    config_dir = str(project_root / "conf")

    with initialize_config_dir(
        config_dir=config_dir, version_base=None, job_name="string_process"
    ):
        cfg: "AppConfig" = cast(
            AppConfig, compose(config_name="config", overrides=final_overrides)
        )

    logger.info("\n" + "~" * 80)
    logger.info(" " * 25 + "HYDRA COMPOSED CONFIGURATION")
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("~" * 80 + "\n")

    processor = StringProcessor(config=cfg)
    final_df = processor.process()

    if final_df is not None and not final_df.empty:
        logger.info(
            f"\n✅ String processing complete. Generated {len(final_df)} final interactions."
        )
    else:
        logger.warning("\n⚠️  String processing resulted in an empty dataset.")
