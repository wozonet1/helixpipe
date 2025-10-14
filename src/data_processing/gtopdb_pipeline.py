import pandas as pd
import sys
from omegaconf import DictConfig, OmegaConf
from hydra import initialize, compose

# 导入需要的工具函数
import research_template as rt
from data_utils.canonicalizer import fetch_sequences_from_uniprot
from data_utils.debug_utils import validate_authoritative_dti_file
from data_processing.purifiers import purify_dti_dataframe_parallel


def process_gtopdb_data(config: DictConfig):
    """
    解析Guide to PHARMACOLOGY的原始数据，提取高质量的内源性配体-靶点相互作用，
    并生成一个遵循项目黄金标准的DTI交互文件。
    【V2 重构版】
    """
    print("\n" + "=" * 80)
    print(" " * 15 + "开始处理 Guide to PHARMACOLOGY 数据流水线 (V2)")
    print("=" * 80 + "\n")

    # --- 1. 从配置中加载Schema和路径 ---
    gtopdb_schema = config.data_structure.schema.external.gtopdb
    internal_schema = config.data_structure.schema.internal.authoritative_dti

    # --- 2. 加载核心数据文件 ---
    print("--- [步骤 1/5] 加载原始数据文件 ---")
    try:
        interactions_df = pd.read_csv(
            rt.get_path(config, "data_structure.paths.raw.interactions"),
            low_memory=False,
            comment="#",
        )
        ligands_df = pd.read_csv(
            rt.get_path(config, "data_structure.paths.raw.ligands"),
            low_memory=False,
            comment="#",
        )
    except FileNotFoundError as e:
        print(f"❌ 致命错误: 原始CSV文件未找到! {e}")
        sys.exit(1)

    print(f"-> 已加载 {len(interactions_df)} 条总交互, {len(ligands_df)} 条总配体。")

    # --- 3. 筛选高质量的内源性配体相互作用 ---
    print("\n--- [步骤 2/5] 筛选高质量的内源性交互 ---")

    # 使用配置驱动的列名
    is_endogenous = interactions_df[gtopdb_schema.interactions.endogenous_flag]
    endogenous_interactions = interactions_df[is_endogenous].copy()

    required_cols = [
        gtopdb_schema.interactions.target_id,
        gtopdb_schema.interactions.ligand_id,
        gtopdb_schema.interactions.affinity,
    ]
    endogenous_interactions.dropna(subset=required_cols, inplace=True)

    # 亲和力筛选
    affinity_threshold = config.data_params.gtopdb_max_affinity_nM
    endogenous_interactions[gtopdb_schema.interactions.affinity] = pd.to_numeric(
        endogenous_interactions[gtopdb_schema.interactions.affinity], errors="coerce"
    )
    endogenous_interactions.dropna(
        subset=[gtopdb_schema.interactions.affinity], inplace=True
    )
    endogenous_interactions = endogenous_interactions[
        endogenous_interactions[gtopdb_schema.interactions.affinity]
        <= affinity_threshold
    ].copy()

    print(f"-> 筛选后保留 {len(endogenous_interactions)} 条高质量内源性交互。")

    # --- 4. 合并与清洗数据 ---
    print("\n--- [步骤 3/5] 合并交互与配体信息 ---")

    # 清洗配体信息
    ligands_df.dropna(
        subset=[
            gtopdb_schema.ligands.molecule_sequence,
            gtopdb_schema.ligands.molecule_id,
        ],
        inplace=True,
    )

    # 合并数据
    merged_df = pd.merge(
        endogenous_interactions,
        ligands_df,
        left_on=gtopdb_schema.interactions.ligand_id,
        right_on=gtopdb_schema.ligands.ligand_id,
    )

    # GtoPdb的UniProt ID可能包含多个，用'|'分隔，我们只取第一个作为主要ID
    merged_df["main_protein_id"] = (
        merged_df[gtopdb_schema.interactions.target_id].str.split("|").str[0]
    )

    print(f"-> 成功合并数据，得到 {len(merged_df)} 条记录。")

    # --- 5. 补全蛋白质序列 ---
    print("\n--- [步骤 4/5] 从UniProt在线获取蛋白质序列 ---")

    unique_pids = merged_df["main_protein_id"].dropna().unique().tolist()
    uniprot_to_sequence_map = fetch_sequences_from_uniprot(unique_pids)

    merged_df["protein_sequence"] = merged_df["main_protein_id"].map(
        uniprot_to_sequence_map
    )

    # 移除没有成功获取到序列的记录
    merged_df.dropna(subset=["protein_sequence"], inplace=True)
    print(f"-> 成功获取序列，剩余 {len(merged_df)} 条完整记录。")

    # --- 6. 构建并保存最终的黄金标准文件 ---
    print("\n--- [步骤 5/5] 构建并保存最终的权威DTI文件 ---")

    # a. 选取并重命名列，以符合黄金标准
    output_df = merged_df[
        [
            gtopdb_schema.ligands.molecule_id,
            "main_protein_id",
            gtopdb_schema.ligands.molecule_sequence,
            "protein_sequence",
        ]
    ].copy()

    output_df.rename(
        columns={
            gtopdb_schema.ligands.molecule_id: internal_schema.molecule_id,
            "main_protein_id": internal_schema.protein_id,
            gtopdb_schema.ligands.molecule_sequence: internal_schema.molecule_sequence,
            "protein_sequence": internal_schema.protein_sequence,
        },
        inplace=True,
    )

    # b. 添加Label列
    output_df[internal_schema.label] = 1

    # c. 清理数据类型
    output_df[internal_schema.molecule_id] = output_df[
        internal_schema.molecule_id
    ].astype(int)

    # d. 【重要】在输出前，对SMILES和序列进行一次净化
    print("--> 对最终数据进行深度净化...")
    # purify_dti_dataframe 期望的列名是 'SMILES' 和 'Sequence', 我们的内部schema正好是
    output_df = purify_dti_dataframe_parallel(output_df, config)

    # e. 去重
    output_df.drop_duplicates(
        subset=[internal_schema.molecule_id, internal_schema.protein_id], inplace=True
    )

    # f. 保存
    output_path = rt.get_path(config, "data_structure.paths.raw.authoritative_dti")
    rt.ensure_path_exists(output_path)
    output_df.to_csv(output_path, index=False)

    print(f"\n✅ 成功创建GtoPdb权威DTI文件，包含 {len(output_df)} 条独特的交互对。")
    print(f"   已保存至: {output_path}")
    print("=" * 80)

    return output_df


if __name__ == "__main__":
    rt.register_hydra_resolvers()
    with initialize(config_path="../../conf", job_name="gtopdb_process"):
        # 加载gtopdb的数据结构，以及通用的数据处理参数
        cfg = compose(
            config_name="config",
            overrides=["data_structure=gtopdb", "data_params=base"],
        )
    print("--- HYDRA COMPOSED CONFIG ---")
    print(OmegaConf.to_yaml(cfg))
    print("-----------------------------")
    # 1. 运行主流程
    final_df = process_gtopdb_data(cfg)

    # 2. 对产出物进行质检
    if not final_df.empty:
        print("\n" + "*" * 80)
        print(" " * 20 + "处理完成，现在开始对输出文件进行最终验证...")
        print("*" * 80)
        validate_authoritative_dti_file(cfg, df=final_df)
