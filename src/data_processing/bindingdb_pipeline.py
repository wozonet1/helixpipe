import sys
import pandas as pd
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from hydra import initialize, compose

# 根据您的说明，移除 'src.' 前缀
from data_processing.purifiers import purify_dti_dataframe_parallel
import research_template as rt
from data_utils.debug_utils import validate_authoritative_dti_file


def process_bindingdb_data(config: DictConfig):
    """
    处理本地已存在的BindingDB原始文件，并生成一个遵循项目黄金标准的DTI交互文件。
    【V3 最终重构版】: 基于分层配置结构，完全配置驱动。
    """
    print("\n" + "=" * 80)
    print(" " * 18 + "开始处理 BindingDB 本地数据流水线 (V3 最终版)")
    print("=" * 80 + "\n")

    # --- [核心修改] 从新的、更清晰的配置结构中获取Schema和路径 ---
    # 外部(原始文件)的schema
    external_schema = config.data.schema.external.bindingdb
    # 内部(黄金标准)的schema
    internal_schema = config.data.schema.internal.authoritative_dti

    raw_dir = rt.get_path(config, "data.files.raw.dummy_file_to_get_dir").parent

    # --- 步骤 1/4: 验证并加载本地原始TSV文件 ---
    print("--- [步骤 1/4] 验证并加载本地原始TSV文件 ---")
    try:
        tsv_path = next(raw_dir.glob(config.data.files.raw.raw_tsv))
    except StopIteration:
        print(
            f"❌ 致命错误: 在 '{raw_dir}' 中找不到 '{config.data.files.raw.raw_tsv}' 文件。"
        )
        print("   请确保您已手动下载并解压了BindingDB的TSV数据文件。")
        sys.exit(1)

    print(f"--> 成功定位数据文件: '{tsv_path.name}'")

    # 从外部Schema动态构建要读取的列列表
    columns_to_read = [
        external_schema.molecule_sequence,
        external_schema.molecule_id,
        external_schema.organism,
        external_schema.ki,
        external_schema.ic50,
        external_schema.kd,
        external_schema.protein_id,
        external_schema.protein_sequence,
    ]

    print("--> 开始分块加载并过滤大型TSV文件...")
    chunk_iterator = pd.read_csv(
        tsv_path,
        sep="\t",
        on_bad_lines="warn",
        usecols=columns_to_read,
        low_memory=False,
        chunksize=100000,
    )

    filtered_chunks = []
    for chunk in tqdm(chunk_iterator, desc="过滤交互数据块"):
        # 1. 筛选物种
        chunk = chunk[chunk[external_schema.organism] == "Homo sapiens"].copy()

        # 2. 过滤掉关键信息缺失的行
        chunk.dropna(
            subset=[
                external_schema.protein_id,
                external_schema.protein_sequence,
                external_schema.molecule_sequence,
                external_schema.molecule_id,
            ],
            inplace=True,
        )

        # 3. 标准化和合并亲和力数值
        for aff_type in [external_schema.ki, external_schema.ic50, external_schema.kd]:
            chunk[aff_type] = pd.to_numeric(
                chunk[aff_type].astype(str).str.replace(">", "").str.replace("<", ""),
                errors="coerce",
            )

        chunk["affinity_nM"] = (
            chunk[external_schema.ki]
            .fillna(chunk[external_schema.kd])
            .fillna(chunk[external_schema.ic50])
        )

        # 4. 根据亲和力阈值过滤
        affinity_threshold = config.params.affinity_threshold_nM
        chunk.dropna(subset=["affinity_nM"], inplace=True)
        chunk = chunk[chunk["affinity_nM"] <= affinity_threshold]

        if not chunk.empty:
            filtered_chunks.append(chunk)

    if not filtered_chunks:
        print("❌ 警告: 经过滤后，没有找到任何符合条件的交互数据。")
        sys.exit(0)

    df = pd.concat(filtered_chunks, ignore_index=True)
    print(f"--> 初步筛选后得到 {len(df)} 条交互作用。")

    # --- 步骤 2/4: 标准化列名为“黄金标准”格式 ---
    print("\n--- [步骤 2/4] 准备数据进行深度净化 ---")

    # 将原始(external)列名重命名为项目内部的黄金标准(internal)列名
    df.rename(
        columns={
            external_schema.molecule_id: internal_schema.molecule_id,
            external_schema.protein_id: internal_schema.protein_id,
            external_schema.molecule_sequence: internal_schema.molecule_sequence,
            external_schema.protein_sequence: internal_schema.protein_sequence,
        },
        inplace=True,
    )
    print("--> 列名已标准化，准备调用通用净化模块。")

    # --- 步骤 3/4: 调用通用净化模块进行深度清洗 ---
    # purify_dti_dataframe 内部应使用内部标准列名 "SMILES" 和 "Sequence"
    # 我们的 internal_schema 正好满足这个约定
    df_purified = purify_dti_dataframe_parallel(df, config)  # 假设净化器也需要config

    # --- 步骤 4/4: 构建最终文件并保存 ---
    print("\n--- [步骤 4/4] 构建并保存最终的权威文件 ---")

    output_df = df_purified[
        [
            internal_schema.molecule_id,
            internal_schema.protein_id,
            internal_schema.molecule_sequence,
            internal_schema.protein_sequence,
        ]
    ].copy()

    output_df[internal_schema.label] = 1
    output_df[internal_schema.molecule_id] = output_df[
        internal_schema.molecule_id
    ].astype(int)

    # 去重
    initial_count = len(output_df)
    output_df.drop_duplicates(
        subset=[internal_schema.molecule_id, internal_schema.protein_id], inplace=True
    )
    print(f"--> 去重后保留 {len(output_df)} / {initial_count} 条独特的交互对。")

    # 从配置获取最终输出路径
    output_path = rt.get_path(config, "data.files.raw.authoritative_dti")
    rt.ensure_path_exists(output_path)
    output_df.to_csv(output_path, index=False)

    print(f"\n✅ 成功创建权威DTI文件, 包含 {len(output_df)} 条独特的交互对。")
    print(f"   已保存至: {output_path}")
    print("=" * 80)

    return output_df


if __name__ == "__main__":
    # 独立运行脚本时，加载配置
    with initialize(config_path="../../conf", job_name="bindingdb_process"):
        # 这个 compose 调用现在因为我们的配置重构而变得非常强大和简洁
        cfg = compose(
            config_name="config", overrides=["data=bindingdb", "params=default"]
        )
    print("\n" + "~" * 80)
    print(" " * 25 + "HYDRA COMPOSED CONFIGURATION")
    print("~" * 80)
    # OmegaConf.to_yaml() 会将配置对象转换为一个格式化的YAML字符串
    print(OmegaConf.to_yaml(cfg))
    print("~" * 80 + "\n")
    # 1. 运行数据处理主流程
    final_df = process_bindingdb_data(cfg)

    # 2. 对输出文件进行严格质检
    print("\n" + "*" * 80)
    print(" " * 20 + "处理完成，现在开始对输出文件进行最终验证...")
    print("*" * 80)
    validate_authoritative_dti_file(cfg, df=final_df)
