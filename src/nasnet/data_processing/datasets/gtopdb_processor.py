from typing import TYPE_CHECKING

import pandas as pd
import research_template as rt

from nasnet.configs import register_all_schemas
from nasnet.data_processing.services.canonicalizer import fetch_sequences_from_uniprot
from nasnet.utils import get_path, log_step, register_hydra_resolvers

from ..services.purifiers import purify_dti_dataframe_parallel

# 导入基类和所有需要的辅助模块
from .base_processor import BaseDataProcessor

if TYPE_CHECKING:
    from nasnet.configs import AppConfig


class GtopdbProcessor(BaseDataProcessor):
    """
    一个专门负责处理Guide to PHARMACOLOGY原始数据的处理器。
    【V2 重构版】：实现了清晰的加载/转换分离，并采用了带日志的流水线步骤。
    """

    def _load_raw_data(self) -> pd.DataFrame:
        """
        【契约实现】只负责从GtoPdb的原始CSV文件中加载数据，并进行初步合并。
        """
        print(
            f"--- [{self.__class__.__name__}] Step: Loading raw data from CSVs... ---"
        )
        gtopdb_schema = self.config.data_structure.schema.external.gtopdb

        try:
            interactions_path = get_path(self.config, "raw.interactions")
            ligands_path = get_path(self.config, "raw.ligands")
            interactions_df = pd.read_csv(
                interactions_path, low_memory=False, comment="#"
            )
            ligands_df = pd.read_csv(ligands_path, low_memory=False, comment="#")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"GtoPdb原始CSV文件未找到! {e}")

        # 清洗配体信息 (只保留有SMILES和CID的)
        ligands_df.dropna(
            subset=[
                gtopdb_schema.ligands.molecule_sequence,
                gtopdb_schema.ligands.molecule_id,
            ],
            inplace=True,
        )

        # 合并交互数据和配体数据
        df = pd.merge(
            interactions_df,
            ligands_df,
            left_on=gtopdb_schema.interactions.ligand_id,
            right_on=gtopdb_schema.ligands.ligand_id,
        )
        print(f"--> Loaded and merged {len(df)} raw rows for transformation.")
        return df

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【契约实现】【V2 修正版】
        只重命名那些在 _load_raw_data 阶段就已经存在的列。
        'protein_sequence' 列此时还不存在，所以我们不在这里处理它。
        """
        if self.verbose > 0:
            print(
                f"--- [{self.__class__.__name__}] Step: Standardizing initial column names... ---"
            )

        gtopdb_schema = self.config.data_structure.schema.external.gtopdb
        internal_schema = self.config.data_structure.schema.internal.authoritative_dti

        # GtoPdb的UniProt ID可能包含多个，在这里处理
        df[internal_schema.protein_id] = (
            df[gtopdb_schema.interactions.target_id].str.split("|").str[0]
        )

        df.rename(
            columns={
                gtopdb_schema.ligands.molecule_id: internal_schema.molecule_id,
                gtopdb_schema.ligands.molecule_sequence: internal_schema.molecule_sequence,
                # 注意：不在这里重命名 'protein_sequence'
            },
            inplace=True,
        )
        return df

    # --- 数据转换的子步骤，由 @log_step 装饰 ---

    @log_step("Filter Endogenous & by Affinity")
    def _transform_step_1_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换步骤1：筛选内源性交互并根据亲和力阈值过滤。"""
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
        return df[df[gtopdb_schema.interactions.affinity] <= affinity_threshold].copy()

    @log_step("Fetch & Add Protein Sequences")
    def _transform_step_2_fetch_and_add_sequences(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """【新】转换步骤2：在所有行过滤之后，在线获取序列。"""
        internal_schema = self.config.data_structure.schema.internal.authoritative_dti

        # df 中现在已经有了 'UniProt_ID' 列
        unique_pids = df[internal_schema.protein_id].dropna().unique().tolist()
        if not unique_pids:
            return pd.DataFrame()

        uniprot_to_sequence_map = fetch_sequences_from_uniprot(unique_pids)

        # 将序列映射为一个新列，并使用【内部黄金标准】列名
        df[internal_schema.protein_sequence] = df[internal_schema.protein_id].map(
            uniprot_to_sequence_map
        )

        return df.dropna(subset=[internal_schema.protein_sequence])

    @log_step("Purify Data (SMILES/Sequence)")
    def _transform_step_3_purify(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换步骤4：调用通用的净化模块，进行深度清洗。"""
        # 注意：GtoPdb数据量小，并行可能开销更大，但为保持一致性我们仍使用并行版本
        return purify_dti_dataframe_parallel(df, self.config)

    @log_step("Finalize and De-duplicate")
    def _transform_step_4_finalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换步骤5：添加Label，清理数据类型，并进行最终去重。"""
        internal_schema = self.config.data_structure.schema.internal.authoritative_dti

        final_df = df[
            [
                internal_schema.molecule_id,
                internal_schema.protein_id,
                internal_schema.molecule_sequence,
                internal_schema.protein_sequence,
            ]
        ].copy()

        final_df[internal_schema.label] = 1
        final_df[internal_schema.molecule_id] = (
            pd.to_numeric(final_df[internal_schema.molecule_id], errors="coerce")
            .dropna()
            .astype(int)
        )

        final_df.drop_duplicates(
            subset=[internal_schema.molecule_id, internal_schema.protein_id],
            inplace=True,
        )
        return final_df

    def _transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【契约实现】GtoPdb数据转换流水线的编排器。
        """
        if self.verbose > 0:
            print(
                f"--- [{self.__class__.__name__}] Step: Transforming {len(df)} whitelisted rows... ---"
            )

        pipeline = [
            self._transform_step_1_filter,
            self._transform_step_2_fetch_and_add_sequences,
            self._transform_step_3_purify,
            self._transform_step_4_finalize,
        ]

        for step_func in pipeline:
            df = step_func(df)
            if df.empty:
                print(
                    f"  - Pipeline halted after step '{step_func.__name__}' because DataFrame became empty."
                )
                return pd.DataFrame()

        if self.verbose > 0:
            print(f"\n✅ [{self.__class__.__name__}] Transformation pipeline complete.")
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
