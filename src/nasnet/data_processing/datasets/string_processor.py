# src/nasnet/data_processing/datasets/string_processor.py
from typing import Dict

import pandas as pd
import research_template as rt

from nasnet.configs import AppConfig, register_all_schemas
from nasnet.utils import get_path, register_hydra_resolvers

from .base_processor import BaseProcessor


class StringProcessor(BaseProcessor):
    """
    【V3 - 架构纯净版】
    专门处理 STRING DB 原始数据的处理器。
    严格遵循 BaseProcessor 的流水线职责划分。
    """

    def __init__(self, config: AppConfig):
        super().__init__(config)
        self.external_schema = self.config.data_structure.schema.external.string

        # 【修改1】在初始化时，就准备好作为内部状态的ID映射字典
        self._id_map = self._build_id_map()

    def _build_id_map(self) -> Dict[str, str]:
        """一个私有辅助方法，用于构建并返回 STRING ID -> UniProt ID 的映射字典。"""
        aliases_path = get_path(self.config, "raw.protein_aliases")
        if not aliases_path.exists():
            raise FileNotFoundError(
                f"STRING aliases file not found at '{aliases_path}'"
            )

        if self.verbose > 0:
            print(
                f"  - [StringProcessor Pre-computation] Building ID map from '{aliases_path.name}'..."
            )

        aliases_df = pd.read_csv(aliases_path, sep="\t", compression="gzip")

        string_id_col = self.external_schema.string_protein_id
        alias_col = self.external_schema.alias
        source_col = self.external_schema.source

        clean_string_id_col = string_id_col.lstrip("#")
        aliases_df.rename(columns={string_id_col: clean_string_id_col}, inplace=True)

        uniprot_aliases = aliases_df[
            aliases_df[source_col].str.contains("UniProt_AC")
        ].copy()
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
            print(f"    - Map created with {len(id_map)} unique STRING ID entries.")

        return id_map

    def _load_raw_data(self) -> pd.DataFrame:
        """
        【修改2】步骤1: 只加载主要的'links'数据，并返回一个DataFrame。
        """
        links_path = get_path(self.config, "raw.protein_links")
        if not links_path.exists():
            raise FileNotFoundError(f"STRING links file not found at '{links_path}'")

        return pd.read_csv(links_path, sep="\t", compression="gzip")

    def _extract_relations(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        步骤2: 利用内部的ID映射字典，将 STRING ID 转换为 UniProt ID。
        """
        # raw_data 现在就是 links_df
        links_df = raw_data

        protein1_col = self.external_schema.protein1
        protein2_col = self.external_schema.protein2

        # 使用 .map() 进行高效映射，并将结果存入新列
        links_df["protein1_uniprot"] = links_df[protein1_col].map(self._id_map)
        links_df["protein2_uniprot"] = links_df[protein2_col].map(self._id_map)

        # 丢弃任何一个ID未能成功映射的交互
        return links_df.dropna(subset=["protein1_uniprot", "protein2_uniprot"]).copy()

    def _standardize_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【修改3】步骤3: 严格履行职责，只进行ID列的重命名。
        """
        # 将上一步创建的临时列名，重命名为我们内部的标准列名
        return df.rename(
            columns={
                "protein1_uniprot": "protein1_id",
                "protein2_uniprot": "protein2_id",
            }
        )

    def _filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        步骤4: 添加 'relation_type' 列。
        """
        if self.verbose > 0:
            print("  - Skipping confidence score filtering (pre-filtered at download).")

        if not df.empty:
            ppi_relation_name = self.config.knowledge_graph.relation_types.ppi
            relation_type_col = self.schema.relation_type
            df[relation_type_col] = ppi_relation_name

        return df


if __name__ == "__main__":
    from argparse import ArgumentParser

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from nasnet.configs import AppConfig

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
    project_root = rt.get_project_root()
    config_dir = str(project_root / "conf")

    with initialize_config_dir(
        config_dir=config_dir, version_base=None, job_name="string_process"
    ):
        cfg: "AppConfig" = compose(config_name="config", overrides=final_overrides)

    print("\n" + "~" * 80)
    print(" " * 25 + "HYDRA COMPOSED CONFIGURATION")
    print(OmegaConf.to_yaml(cfg))
    print("~" * 80 + "\n")

    processor = StringProcessor(config=cfg)
    final_df = processor.process()

    if final_df is not None and not final_df.empty:
        print(
            f"\n✅ String processing complete. Generated {len(final_df)} final interactions."
        )
    else:
        print("\n⚠️  String processing resulted in an empty dataset.")
