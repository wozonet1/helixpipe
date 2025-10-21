# 文件: src/data_processing/gtopdb_processor.py (全新/重构)

from argparse import ArgumentParser

import pandas as pd
import research_template as rt

from configs.register_schemas import register_all_schemas

# 导入我们的新基类和所有需要的辅助模块
from data_processing.base_processor import BaseDataProcessor
from data_processing.log_decorators import log_step
from data_processing.purifiers import purify_dti_dataframe_parallel
from data_utils.canonicalizer import fetch_sequences_from_uniprot

register_all_schemas()
rt.register_hydra_resolvers()


class GtopdbProcessor(BaseDataProcessor):
    # --- 将处理流程拆分为独立的、被装饰的步骤 ---

    @log_step("Load & Initial Filter")
    def _step_1_load_and_filter(self, _) -> pd.DataFrame:
        """步骤1：加载interactions.csv和ligands.csv，进行初步筛选和合并。"""
        gtopdb_schema = self.config.data_structure.schema.external.gtopdb

        try:
            interactions_path = rt.get_path(self.config, "raw.interactions")
            ligands_path = rt.get_path(self.config, "raw.ligands")
            interactions_df = pd.read_csv(
                interactions_path, low_memory=False, comment="#"
            )
            ligands_df = pd.read_csv(ligands_path, low_memory=False, comment="#")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"GtoPdb原始CSV文件未找到! {e}")

        # 1a. 筛选内源性交互
        is_endogenous = interactions_df[gtopdb_schema.interactions.endogenous_flag]
        endogenous_interactions = interactions_df[is_endogenous].copy()

        # 1b. 过滤掉关键信息缺失的行
        required_cols = [
            gtopdb_schema.interactions.target_id,
            gtopdb_schema.interactions.ligand_id,
            gtopdb_schema.interactions.affinity,
        ]
        endogenous_interactions.dropna(subset=required_cols, inplace=True)

        # 1c. 根据亲和力阈值过滤
        affinity_threshold = self.config.data_params.affinity_threshold_nM
        endogenous_interactions[gtopdb_schema.interactions.affinity] = pd.to_numeric(
            endogenous_interactions[gtopdb_schema.interactions.affinity],
            errors="coerce",
        )
        endogenous_interactions.dropna(
            subset=[gtopdb_schema.interactions.affinity], inplace=True
        )
        endogenous_interactions = endogenous_interactions[
            endogenous_interactions[gtopdb_schema.interactions.affinity]
            <= affinity_threshold
        ].copy()

        # 1d. 合并配体信息
        ligands_df.dropna(
            subset=[
                gtopdb_schema.ligands.molecule_sequence,
                gtopdb_schema.ligands.molecule_id,
            ],
            inplace=True,
        )

        merged_df = pd.merge(
            endogenous_interactions,
            ligands_df,
            left_on=gtopdb_schema.interactions.ligand_id,
            right_on=gtopdb_schema.ligands.ligand_id,
        )
        return merged_df

    @log_step("Fetch Protein Sequences")
    def _step_2_fetch_sequences(self, df: pd.DataFrame) -> pd.DataFrame:
        """步骤2：处理UniProt ID，并从网络获取蛋白质序列。"""
        gtopdb_schema = self.config.data_structure.schema.external.gtopdb

        # GtoPdb的UniProt ID可能包含多个，用'|'分隔，我们只取第一个
        df["main_protein_id"] = (
            df[gtopdb_schema.interactions.target_id].str.split("|").str[0]
        )

        unique_pids = df["main_protein_id"].dropna().unique().tolist()
        if not unique_pids:
            return pd.DataFrame()  # 如果没有有效的PID，直接返回空

        uniprot_to_sequence_map = fetch_sequences_from_uniprot(unique_pids)
        df["protein_sequence"] = df["main_protein_id"].map(uniprot_to_sequence_map)

        # 移除没有成功获取到序列的记录
        return df.dropna(subset=["protein_sequence"])

    @log_step("Standardize Columns")
    def _step_3_standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """步骤3：将列名重命名为项目内部的黄金标准。"""
        gtopdb_schema = self.config.data_structure.schema.external.gtopdb
        internal_schema = self.config.data_structure.schema.internal.authoritative_dti
        return df.rename(
            columns={
                gtopdb_schema.ligands.molecule_id: internal_schema.molecule_id,
                "main_protein_id": internal_schema.protein_id,
                gtopdb_schema.ligands.molecule_sequence: internal_schema.molecule_sequence,
                "protein_sequence": internal_schema.protein_sequence,
            }
        )

    @log_step("Purify Data (SMILES/Sequence)")
    def _step_4_purify(self, df: pd.DataFrame) -> pd.DataFrame:
        """步骤4：调用通用的净化模块，进行深度清洗。"""
        return purify_dti_dataframe_parallel(df, self.config)

    @log_step("Finalize and De-duplicate")
    def _step_5_finalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """步骤5：添加Label，清理数据类型，并进行最终去重。"""
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
        final_df[internal_schema.molecule_id] = final_df[
            internal_schema.molecule_id
        ].astype(int)

        final_df.drop_duplicates(
            subset=[internal_schema.molecule_id, internal_schema.protein_id],
            inplace=True,
        )
        return final_df

    def _process_raw_data(self) -> pd.DataFrame:
        """
        【契约实现】GtoPdb处理流水线的编排器。
        """
        df = pd.DataFrame()  # 初始空DataFrame

        df = self._step_1_load_and_filter(df)
        if df.empty:
            return df

        df = self._step_2_fetch_sequences(df)
        if df.empty:
            return df

        df = self._step_3_standardize_columns(df)
        if df.empty:
            return df

        df = self._step_4_purify(df)
        if df.empty:
            return df

        df = self._step_5_finalize(df)

        print(f"\n✅ [{self.__class__.__name__}] Raw processing pipeline complete.")
        return df


if __name__ == "__main__":
    from hydra import compose, initialize
    from omegaconf import OmegaConf

    # === 阶段 1: 明确定义脚本身份和命令行接口 ===

    # a. 这个脚本的固有基础配置
    BASE_OVERRIDES = ["data_structure=gtopdb", "data_params=gtopdb"]

    # b. 设置命令行解析器，只接收用户自定义的覆盖参数
    parser = ArgumentParser(description="Run the GtoPdb processing pipeline.")
    parser.add_argument(
        "user_overrides", nargs="*", help="Hydra overrides (e.g., training.epochs=10)"
    )
    args = parser.parse_args()

    # c. 组合所有覆盖参数
    final_overrides = BASE_OVERRIDES + args.user_overrides

    # === 阶段 2: 手动、可预测地加载配置 ===

    # a. 使用 initialize 来设置配置根目录
    with initialize(
        config_path="../../conf", version_base=None, job_name="gtopdb_process"
    ):
        # b. 使用 compose 来构建最终的配置对象
        cfg = compose(config_name="config", overrides=final_overrides)

    # === 阶段 3: 执行核心业务逻辑 ===

    print("\n" + "=" * 80)
    print(" " * 25 + "FINAL COMPOSED CONFIGURATION")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80 + "\n")

    # a. 实例化处理器 (cfg是DictConfig，Processor的__init__类型提示应为DictConfig)
    processor = GtopdbProcessor(config=cfg)

    # b. 运行处理流程
    final_df = processor.process()

    # c. 保存和验证
    if final_df is not None and not final_df.empty:
        # 【重要】get_path现在会在当前目录下解析相对路径，
        # 因为我们没有改变工作目录，这正是我们想要的简单行为。
        # 它会正确地在项目根目录下的data/gtopdb/raw/中创建文件。
        output_path = rt.get_path(cfg, "raw.authoritative_dti")
        rt.ensure_path_exists(output_path)
        final_df.to_csv(output_path, index=False)
        print(f"\n✅ Successfully saved authoritative DTI file to: {output_path}")
    else:
        print("\n⚠️  Processor returned an empty DataFrame. No file was saved.")
