# 文件: src/data_processing/bindingdb_pipeline.py (基于列名情报优化版)
import sys
import pandas as pd
from tqdm import tqdm
from omegaconf import DictConfig
from hydra import initialize, compose

# 假设您的项目结构如下
# 确保 purifiers 模块可以被正确导入
from data_processing.purifiers import purify_dti_dataframe_parallel
import research_template as rt
from data_utils.debug_utils import validate_authoritative_dti_file


def process_bindingdb_data(config: DictConfig):
    """
    处理本地已存在的BindingDB原始文件,并生成权威的DTI交互文件。
    【重构版】: 调用通用的净化模块来处理数据清洗和标准化。
    """
    print("\n" + "=" * 80)
    print(" " * 18 + "开始处理 BindingDB 本地数据流水线 (重构版)")
    print("=" * 80 + "\n")

    # --- 步骤 1/4: 验证并加载本地原始TSV文件 ---
    print("--- [步骤 1/4] 验证并加载本地原始TSV文件 ---")
    raw_dir = rt.get_path(config, "raw.dummy_file_to_get_dir").parent
    try:
        tsv_path = next(raw_dir.glob("BindingDB_All.tsv"))
    except StopIteration:
        print(f"❌ 致命错误: 在 '{raw_dir}' 中找不到 'BindingDB_All.tsv' 文件。")
        print("   请确保您已手动下载并解压了BindingDB的TSV数据文件。")
        sys.exit(1)

    print(f"--> 成功定位数据文件: '{tsv_path.name}'")

    # 定义需要读取的列，这些也可以考虑放入配置文件中
    columns_to_read = [
        "Ligand SMILES",
        "PubChem CID",
        "Target Source Organism According to Curator or DataSource",
        "Ki (nM)",
        "IC50 (nM)",
        "Kd (nM)",
        "UniProt (SwissProt) Primary ID of Target Chain 1",
        "BindingDB Target Chain Sequence 1",
    ]

    print("--> 开始分块加载并过滤大型TSV文件...")
    chunk_iterator = pd.read_csv(
        tsv_path,
        sep="\t",
        on_bad_lines="skip",
        usecols=columns_to_read,
        low_memory=False,
        chunksize=100000,
    )

    filtered_chunks = []
    for chunk in tqdm(chunk_iterator, desc="过滤交互数据块"):
        # 1. 筛选物种
        chunk = chunk[
            chunk["Target Source Organism According to Curator or DataSource"]
            == "Homo sapiens"
        ].copy()

        # 2. 过滤掉关键信息缺失的行
        chunk.dropna(
            subset=[
                "UniProt (SwissProt) Primary ID of Target Chain 1",
                "BindingDB Target Chain Sequence 1",
                "Ligand SMILES",
                "PubChem CID",
            ],
            inplace=True,
        )

        # 3. 标准化和合并亲和力数值
        for aff_type in ["Ki (nM)", "IC50 (nM)", "Kd (nM)"]:
            chunk[aff_type] = (
                chunk[aff_type].astype(str).str.replace(">", "").str.replace("<", "")
            )
            chunk[aff_type] = pd.to_numeric(chunk[aff_type], errors="coerce")

        chunk["affinity_nM"] = (
            chunk["Ki (nM)"].fillna(chunk["Kd (nM)"]).fillna(chunk["IC50 (nM)"])
        )

        # 4. 根据亲和力阈值过滤
        affinity_threshold = config.params.affinity_threshold_nM
        chunk.dropna(subset=["affinity_nM"], inplace=True)
        chunk = chunk[chunk["affinity_nM"] <= affinity_threshold]

        if not chunk.empty:
            filtered_chunks.append(chunk)

    if not filtered_chunks:
        print(
            "❌ 警告: 经过滤后，没有找到任何符合条件的交互数据。请检查过滤参数或原始数据。"
        )
        sys.exit(0)

    df = pd.concat(filtered_chunks, ignore_index=True)
    print(f"--> 初步筛选后得到 {len(df)} 条交互作用。")

    # --- 步骤 2/4: 标准化列名以进行通用清洗 ---
    print("\n--- [步骤 2/4] 准备数据进行深度净化 ---")

    # 将列名重命名为通用净化函数期望的名称 ('SMILES', 'Sequence'等)
    df.rename(
        columns={
            "Ligand SMILES": "SMILES",
            "BindingDB Target Chain Sequence 1": "Sequence",
            "PubChem CID": "PubChem_CID",
            "UniProt (SwissProt) Primary ID of Target Chain 1": "UniProt_ID",
        },
        inplace=True,
    )
    print("--> 列名已标准化，准备调用通用净化模块。")

    # --- 步骤 3/4: 调用通用净化模块进行深度清洗 ---
    # 【核心重构】将所有清洗逻辑（SMILES验证、标准化，序列验证）委托给外部函数
    n_jobs = config.runtime.cpus
    df_purified = purify_dti_dataframe_parallel(df, n_jobs=n_jobs)

    # --- 步骤 4/4: 构建最终文件并保存 ---
    print("\n--- [步骤 4/4] 构建并保存最终的权威文件 ---")

    # 从净化后的 DataFrame 中选取最终需要的列
    # 注意：此时 "SMILES" 列已经是标准化后的SMILES
    output_df = df_purified[
        [
            "PubChem_CID",
            "UniProt_ID",
            "SMILES",
            "Sequence",
        ]
    ].copy()
    output_df["Label"] = 1
    output_df["PubChem_CID"] = output_df["PubChem_CID"].astype(int)

    # 去除重复的 (药物, 靶点) 对
    initial_count = len(output_df)
    output_df.drop_duplicates(subset=["PubChem_CID", "UniProt_ID"], inplace=True)
    print(f"--> 去重后保留 {len(output_df)} / {initial_count} 条独特的交互对。")

    output_path = rt.get_path(config, "raw.dti_interactions")
    rt.ensure_path_exists(output_path)
    output_df.to_csv(output_path, index=False)

    print(f"\n✅ 成功创建权威DTI文件,包含 {len(output_df)} 条独特的交互对。")
    print(f"   已保存至: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    with initialize(config_path="../../conf", job_name="bindingdb_process"):
        cfg = compose(
            config_name="config", overrides=["data=bindingdb", "params=bindingdb"]
        )

    # 1. 运行数据处理主流程
    # process_bindingdb_data(cfg)

    # 2. 【新增】运行严格的验证流程，对刚刚生成的文件进行质检
    print("\n" + "*" * 80)
    print(" " * 20 + "处理完成，现在开始对输出文件进行最终验证...")
    print("*" * 80)
    validate_authoritative_dti_file(cfg)
