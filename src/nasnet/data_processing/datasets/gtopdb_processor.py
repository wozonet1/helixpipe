# 文件: src/nasnet/data_processing/datasets/gtopdb_processor.py (最终架构版)
from typing import TYPE_CHECKING

import argcomplete
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
        重命名ID和SMILES列，并清洗UniProt ID格式。
        """
        # 清洗UniProt ID (e.g., 'P12345|...')
        df[self.schema.protein_id] = (
            df[self.external_schema.interactions.target_id].str.split("|").str[0]
        )

        # 重命名列
        return df.rename(
            columns={
                self.external_schema.ligands.molecule_id: self.schema.molecule_id,
                self.external_schema.ligands.molecule_sequence: self.schema.molecule_sequence,
            }
        )

    # --- 步骤 4 (覆盖): 实现业务规则过滤 ---
    def _filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【覆盖实现 - 诊断版】执行特定于GtoPdb的业务规则过滤，并增加详细日志。
        """
        gtopdb_schema = self.config.data_structure.schema.external.gtopdb

        # a. 筛选内源性交互
        is_endogenous = df[gtopdb_schema.interactions.endogenous_flag]
        df = df[is_endogenous].copy()

        # b. 过滤关键信息缺失的行
        required_cols = [
            gtopdb_schema.interactions.target_id,
            gtopdb_schema.interactions.affinity,
        ]
        df.dropna(subset=required_cols, inplace=True)

        # c. 根据亲和力阈值过滤
        affinity_threshold = (
            self.config.data_params.affinity_threshold_nM
        )  # 注意：这里用了通用的阈值
        df[gtopdb_schema.interactions.affinity] = pd.to_numeric(
            df[gtopdb_schema.interactions.affinity], errors="coerce"
        )
        df.dropna(subset=[gtopdb_schema.interactions.affinity], inplace=True)
        df = df[df[gtopdb_schema.interactions.affinity] <= affinity_threshold].copy()
        if not df.empty:
            schema_config = self.config.data_structure.schema.internal.authoritative_dti
            df[schema_config.relation_type] = (
                self.config.knwoledge_graph.relation_types.default
            )
        return df


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
