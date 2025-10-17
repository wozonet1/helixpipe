# 文件: src/data_processing/gtopdb_processor.py (全新/重构)

import sys
import pandas as pd
from argparse import ArgumentParser

# 导入我们的新基类和所有需要的辅助模块
from data_processing.base_processor import BaseDataProcessor
from data_processing.purifiers import purify_dti_dataframe_parallel
from data_utils.canonicalizer import fetch_sequences_from_uniprot
import research_template as rt
from configs.register_schemas import register_all_schemas, AppConfig

register_all_schemas()
rt.register_hydra_resolvers()


class GtoPdbProcessor(BaseDataProcessor):
    """
    一个专门负责处理GuideToPHARMACOLOGY (GtoPdb)数据的处理器。
    主要用于提取高质量的、内源性的配体-靶点相互作用。
    """

    def _load_and_filter_raw_data(self) -> pd.DataFrame:
        """
        私有方法：加载并筛选原始GtoPdb数据文件。
        """
        print("--- [Step 1/3] 加载并筛选原始GtoPdb数据文件 ---")
        gtopdb_schema = self.config.data_structure.schema.external.gtopdb

        try:
            interactions_path = rt.get_path(
                self.config, "data_structure.paths.raw.interactions"
            )
            ligands_path = rt.get_path(self.config, "data_structure.paths.raw.ligands")
            interactions_df = pd.read_csv(
                interactions_path, low_memory=False, comment="#"
            )
            ligands_df = pd.read_csv(ligands_path, low_memory=False, comment="#")
        except FileNotFoundError as e:
            print(f"❌ 致命错误: GtoPdb原始CSV文件未找到! {e}")
            print(
                "   请确保 'interactions.csv' 和 'ligands.csv' 已放置在正确的raw目录下。"
            )
            sys.exit(1)

        print(
            f"-> 已加载 {len(interactions_df)} 条总交互, {len(ligands_df)} 条总配体。"
        )

        # 1. 筛选内源性交互
        is_endogenous = interactions_df[gtopdb_schema.interactions.endogenous_flag]
        endogenous_interactions = interactions_df[is_endogenous].copy()

        # 2. 过滤掉关键信息缺失的行
        required_cols = [
            gtopdb_schema.interactions.target_id,
            gtopdb_schema.interactions.ligand_id,
            gtopdb_schema.interactions.affinity,
        ]
        endogenous_interactions.dropna(subset=required_cols, inplace=True)

        # 3. 根据亲和力阈值过滤
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

        print(f"-> 初步筛选后得到 {len(endogenous_interactions)} 条高质量内源性交互。")

        # 4. 合并配体信息
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

    def process(self) -> pd.DataFrame:
        """
        【契约实现】执行完整的GtoPdb处理流程。
        """
        # 1. 加载和初步过滤
        df = self._load_and_filter_raw_data()
        if df.empty:
            return df

        gtopdb_schema = self.config.data_structure.schema.external.gtopdb
        internal_schema = self.config.data_structure.schema.internal

        # 2. 补全蛋白质序列
        print("\n--- [Step 2/3] 从UniProt在线获取蛋白质序列 ---")
        # GtoPdb的UniProt ID可能包含多个，用'|'分隔，我们只取第一个
        df["main_protein_id"] = (
            df[gtopdb_schema.interactions.target_id].str.split("|").str[0]
        )

        unique_pids = df["main_protein_id"].dropna().unique().tolist()
        uniprot_to_sequence_map = fetch_sequences_from_uniprot(unique_pids)
        df["protein_sequence"] = df["main_protein_id"].map(uniprot_to_sequence_map)

        df.dropna(subset=["protein_sequence"], inplace=True)
        print(f"-> 成功获取序列，剩余 {len(df)} 条完整记录。")

        # 3. 标准化列名并构建最终输出
        print("\n--- [Step 3/3] 标准化、净化并构建最终文件 ---")
        df.rename(
            columns={
                gtopdb_schema.ligands.molecule_id: internal_schema.molecule_id,
                "main_protein_id": internal_schema.protein_id,
                gtopdb_schema.ligands.molecule_sequence: internal_schema.molecule_sequence,
                "protein_sequence": internal_schema.protein_sequence,
            },
            inplace=True,
        )

        # 深度净化
        df_purified = purify_dti_dataframe_parallel(df, self.config)

        # 构建最终输出
        final_df = df_purified[
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

        print(
            f"\n✅ [{self.__class__.__name__}] 处理完成，最终生成 {len(final_df)} 条独特的交互对。"
        )

        self.validate(final_df)

        return final_df


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
    processor = GtoPdbProcessor(config=cfg)

    # b. 运行处理流程
    final_df = processor.process()

    # c. 保存和验证
    if final_df is not None and not final_df.empty:
        # 【重要】get_path现在会在当前目录下解析相对路径，
        # 因为我们没有改变工作目录，这正是我们想要的简单行为。
        # 它会正确地在项目根目录下的data/gtopdb/raw/中创建文件。
        output_path = rt.get_path(cfg, "data_structure.paths.raw.authoritative_dti")
        rt.ensure_path_exists(output_path)
        final_df.to_csv(output_path, index=False)
        print(f"\n✅ Successfully saved authoritative DTI file to: {output_path}")

        processor.validate(final_df)
    else:
        print("\n⚠️  Processor returned an empty DataFrame. No file was saved.")
