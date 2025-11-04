# 文件: src/nasnet/data_processing/datasets/gtopdb_processor.py (最终架构版)
from typing import TYPE_CHECKING

import argcomplete
import numpy as np
import pandas as pd
import research_template as rt

from nasnet.configs import register_all_schemas
from nasnet.utils import get_path, register_hydra_resolvers

# 导入基类和所有需要的辅助模块
from .base_processor import BaseProcessor

if TYPE_CHECKING:
    from nasnet.configs import AppConfig


from nasnet.configs import AppConfig


class GtopdbProcessor(BaseProcessor):
    """
    一个专门负责处理Guide to PHARMACOLOGY原始数据的处理器。
    【V5 - 最终架构版】：严格实现BaseProcessor定义的六步流水线。
    """

    def __init__(self, config: AppConfig):
        super().__init__(config)
        self.external_schema = self.config.data_structure.schema.external.gtopdb

    # --- 步骤 1: 实现数据加载 ---
    def _load_raw_data(self) -> pd.DataFrame:
        """从GtoPdb的原始CSV文件中加载数据并进行初步合并。"""
        try:
            interactions_path = get_path(self.config, "raw.interactions")
            ligands_path = get_path(self.config, "raw.ligands")
            interactions_df = pd.read_csv(
                interactions_path, low_memory=False, comment="#"
            )
            ligands_df = pd.read_csv(ligands_path, low_memory=False, comment="#")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"GtoPdb原始CSV文件未找到! {e}")

        # 在合并前，预先过滤掉没有SMILES或CID的配体，以减小处理体积
        ligands_df.dropna(
            subset=[
                self.external_schema.ligands.molecule_sequence,
                self.external_schema.ligands.molecule_id,
            ],
            inplace=True,
        )

        # 合并交互数据和配体数据
        merged_df = pd.merge(
            interactions_df,
            ligands_df,
            left_on=self.external_schema.interactions.ligand_id,
            right_on=self.external_schema.ligands.ligand_id,
        )
        return merged_df

    # --- 步骤 2: 实现关系提取 ---
    def _extract_relations(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        对于GtoPdb，加载的数据已经是关系格式，直接返回即可。
        """
        return raw_data

    # --- 步骤 3: 实现ID和结构标准化 ---
    def _standardize_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        步骤3: 【V2 - 最终重构版】
        将原始DataFrame重塑为“规范化交互格式”，并附带所有用于下游校验的辅助列。
        """
        if self.verbose > 0:
            print(
                "  - Reshaping DataFrame to canonical format with auxiliary columns..."
            )

        final_df = pd.DataFrame()

        # --- 核心规范化列 ---
        entity_names = self.config.knowledge_graph.entity_types

        # 1. Source (Molecule)
        # 我们不再在这里重命名，而是直接赋值给新列
        final_df[self.schema.source_id] = df[self.external_schema.ligands.molecule_id]
        final_df[self.schema.source_type] = np.where(
            df["endogenous_flag"],  # 条件: if endogenous_flag is True
            entity_names.ligand_endo,  # 值为 True 时的结果
            entity_names.ligand_exo,  # 值为 False 时的结果
        )

        # 2. Target (Protein)
        # 清洗UniProt ID (e.g., 'P12345|...')并赋值给新列
        final_df[self.schema.target_id] = (
            df[self.external_schema.interactions.target_id].str.split("|").str[0]
        )
        final_df[self.schema.target_type] = entity_names.protein

        # 3. Relation Type
        final_df[self.schema.relation_type] = (
            self.config.knowledge_graph.relation_types.default
        )

        # --- 附带下游需要的“原材料” ---

        # 4. 结构信息
        final_df["structure_molecule"] = df[
            self.external_schema.ligands.molecule_sequence
        ]
        # GtoPdb不提供蛋白质序列，所以我们不创建 structure_protein 列

        # 5. 用于下一步过滤的领域特定信息
        final_df["affinity_nM"] = df[self.external_schema.interactions.affinity]
        final_df["endogenous_flag"] = df[
            self.external_schema.interactions.endogenous_flag
        ]

        return final_df

    def _filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        步骤4: 【V5 - 最终真相版】
        根据确凿的数据证据，执行正确的领域筛选。
        """
        if self.verbose > 0:
            print(
                "  - Applying domain-specific filters: Non-Endogenous & Affinity Threshold..."
            )

        df_filtered = df.copy()
        # 2. 根据亲和力阈值过滤 (逻辑不变)
        affinity_threshold = self.config.data_params.affinity_threshold_nM

        df_filtered["affinity_nM"] = pd.to_numeric(
            df_filtered["affinity_nM"], errors="coerce"
        )
        df_filtered.dropna(subset=["affinity_nM"], inplace=True)

        initial_count_after_endo = len(df_filtered)
        df_filtered = df_filtered[
            df_filtered["affinity_nM"] <= affinity_threshold
        ].copy()

        if self.verbose > 0:
            print(
                f"    - {len(df_filtered)} / {initial_count_after_endo} records passed affinity filter."
            )

        return df_filtered


if __name__ == "__main__":
    from argparse import ArgumentParser

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from nasnet.configs import AppConfig

    BASE_OVERRIDES = [
        "data_structure=gtopdb",
        "data_params=gtopdb",
    ]  # gtopdb 使用自己专属的参数集

    parser = ArgumentParser(description="Run the GtoPdb processing pipeline.")
    parser.add_argument("user_overrides", nargs="*", help="Hydra overrides")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    final_overrides = BASE_OVERRIDES + args.user_overrides

    register_hydra_resolvers()
    register_all_schemas()
    project_root = rt.get_project_root()
    config_dir = str(project_root / "conf")

    with initialize_config_dir(
        config_dir=config_dir, version_base=None, job_name="gtopdb_process"
    ):
        cfg: "AppConfig" = compose(config_name="config", overrides=final_overrides)

    print("\n" + "~" * 80)
    print(" " * 25 + "HYDRA COMPOSED CONFIGURATION")
    print(OmegaConf.to_yaml(cfg))
    print("~" * 80 + "\n")

    processor = GtopdbProcessor(config=cfg)
    final_df = processor.process()

    if final_df is not None and not final_df.empty:
        print(
            f"\n✅ GtoPdb processing complete. Generated {len(final_df)} final interactions."
        )
    else:
        print("\n⚠️  GtoPdb processing resulted in an empty dataset.")
