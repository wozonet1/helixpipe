# 文件: src/data_processing/bindingdb_processor.py

import sys
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf

# 导入我们新创建的基类和需要的辅助模块
from data_processing.base_processor import BaseDataProcessor
from data_processing.purifiers import (
    purify_dti_dataframe_parallel,
    filter_molecules_by_properties,
)
import research_template as rt


class BindingDBProcessor(BaseDataProcessor):
    """
    一个专门负责处理BindingDB数据的数据处理器。
    它实现了 BaseDataProcessor 定义的接口。
    """

    def _load_and_filter_raw_data(self) -> pd.DataFrame:
        """
        私有方法：负责加载和初步筛选原始BindingDB TSV文件。
        """
        # 从配置中获取Schema和路径
        external_schema = self.config.data_structure.schema.external.bindingdb

        tsv_path = rt.get_path(self.config, "data_structure.paths.raw.raw_tsv")
        if not tsv_path.exists():
            print(
                f"❌ 致命错误: 在 '{tsv_path.parent}' 中找不到 '{tsv_path.name}' 文件。"
            )
            print("   请先运行下载脚本或手动下载。")
            sys.exit(1)

        print(f"--> [Step 1/4] 成功定位数据文件: '{tsv_path}'")

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

        print("--> [Step 2/4] 开始分块加载并过滤大型TSV文件...")
        chunk_iterator = pd.read_csv(
            tsv_path,
            sep="\t",
            on_bad_lines="warn",
            usecols=columns_to_read,
            low_memory=False,
            chunksize=100000,
        )

        filtered_chunks = []
        for chunk in tqdm(chunk_iterator, desc="    Filtering Chunks"):
            # 过滤和清洗逻辑与之前完全相同
            chunk = chunk[chunk[external_schema.organism] == "Homo sapiens"].copy()
            chunk.dropna(
                subset=[
                    external_schema.protein_id,
                    external_schema.protein_sequence,
                    external_schema.molecule_sequence,
                    external_schema.molecule_id,
                ],
                inplace=True,
            )

            for aff_type in [
                external_schema.ki,
                external_schema.ic50,
                external_schema.kd,
            ]:
                chunk[aff_type] = pd.to_numeric(
                    chunk[aff_type]
                    .astype(str)
                    .str.replace(">", "")
                    .str.replace("<", ""),
                    errors="coerce",
                )
            chunk["affinity_nM"] = (
                chunk[external_schema.ki]
                .fillna(chunk[external_schema.kd])
                .fillna(chunk[external_schema.ic50])
            )

            affinity_threshold = self.config.data_params.affinity_threshold_nM
            chunk.dropna(subset=["affinity_nM"], inplace=True)
            chunk = chunk[chunk["affinity_nM"] <= affinity_threshold]

            if not chunk.empty:
                filtered_chunks.append(chunk)

        if not filtered_chunks:
            print(
                "❌ 警告: 经过滤后，没有找到任何符合条件的交互数据。将返回空DataFrame。"
            )
            return pd.DataFrame()

        df = pd.concat(filtered_chunks, ignore_index=True)
        print(f"--> 初步筛选后得到 {len(df)} 条交互作用。")
        return df

    def process(self) -> pd.DataFrame:
        """
        【契约实现】执行完整的BindingDB处理流程。
        """
        # 1. 加载和初步过滤
        df = self._load_and_filter_raw_data()
        if df.empty:
            return df  # 如果没有数据，直接返回空的DataFrame

        # 2. 标准化列名
        print("\n--> [Step 3/4] 标准化列名为项目内部黄金标准...")
        external_schema = self.config.data_structure.schema.external.bindingdb
        internal_schema = self.config.data_structure.schema.internal
        df.rename(
            columns={
                external_schema.molecule_id: internal_schema.molecule_id,
                external_schema.protein_id: internal_schema.protein_id,
                external_schema.molecule_sequence: internal_schema.molecule_sequence,
                external_schema.protein_sequence: internal_schema.protein_sequence,
            },
            inplace=True,
        )

        # 3. 深度净化和过滤
        print("\n--> [Step 4/4] 调用通用模块进行深度净化和过滤...")
        df_purified = purify_dti_dataframe_parallel(df, self.config)
        df_filtered = filter_molecules_by_properties(df_purified, self.config)

        # 4. 构建最终输出
        final_df = df_filtered[
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

        # 去重
        final_df.drop_duplicates(
            subset=[internal_schema.molecule_id, internal_schema.protein_id],
            inplace=True,
        )

        print(
            f"\n✅ [{self.__class__.__name__}] 处理完成，最终生成 {len(final_df)} 条独特的交互对。"
        )

        # 5. 【可选但推荐】在返回前进行自我验证
        self.validate(final_df)

        return final_df


# --------------------------------------------------------------------------
# Config Store模式下的独立运行入口 (最终版)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    from hydra import compose, initialize_config_dir

    # 导入我们的注册函数和顶层Config类型
    from configs.register_schemas import register_all_schemas

    # 1. 在所有Hydra操作之前，执行注册
    register_all_schemas()
    rt.register_hydra_resolvers()
    try:
        # get_project_root 在非Hydra应用下会使用 Path.cwd()
        # 所以请确保您是从项目根目录运行此脚本
        project_root = rt.get_project_root()
        config_dir = str(project_root / "conf")
    except Exception as e:
        print(f"❌ 无法确定项目根目录或配置路径。错误: {e}")
        sys.exit(1)

    # a. 定义这个脚本固有的、基础的overrides
    #    我们总是要处理 bindingdb 的数据结构
    base_overrides = ["data_structure=bindingdb"]

    # b. Hydra的Compose API会自动从 sys.argv 获取用户在命令行输入的覆盖
    #    所以我们不再需要手动处理 argparse
    print("--- Applying Hydra Overrides ---")

    with initialize_config_dir(
        config_dir=config_dir, version_base=None, job_name="bindingdb_process"
    ):
        cfg = compose(config_name="config", overrides=base_overrides)

    print("\n" + "~" * 80)
    print(" " * 25 + "HYDRA COMPOSED CONFIGURATION")
    print("~" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("~" * 80 + "\n")

    # --- 后续逻辑与之前类似，但现在cfg是一个更可靠的对象 ---

    # a. 实例化处理器 (config 现在是 DictConfig, 完全没问题)
    print("--- [Main] Instantiating processor: BindingDBProcessor ---")
    processor = BindingDBProcessor(config=cfg)

    # b. 调用 process 方法来执行数据处理
    final_df = processor.process()

    # c. 保存和验证 (逻辑不变)
    if final_df is not None and not final_df.empty:
        output_path = rt.get_path(cfg, "data_structure.paths.raw.authoritative_dti")
        rt.ensure_path_exists(output_path)
        final_df.to_csv(output_path, index=False)
        print(f"\n✅ 成功将权威DTI文件保存至: {output_path}")
        processor.validate(final_df)
    else:
        print("\n⚠️  Processor returned an empty DataFrame. No file was saved.")
