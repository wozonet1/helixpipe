# 文件: src/data_processing/bindingdb_processor.py

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
from hydra import compose

from .log_decorators import log_step

# --------------------------------------------------------------------------
# 步骤 2: 定义流水线化的BindingDB处理器
# --------------------------------------------------------------------------


class BindingDBProcessor(BaseDataProcessor):
    # --- 将处理流程拆分为独立的、被装饰的步骤 ---

    @log_step("Initial Load & Filter")
    def _step_1_load_and_filter(self, _) -> pd.DataFrame:
        """步骤1：加载原始TSV文件，并进行初步的、特定于源的过滤。"""
        external_schema = self.config.data_structure.schema.external.bindingdb

        tsv_path = rt.get_path(self.config, "data_structure.paths.raw.raw_tsv")
        if not tsv_path.exists():
            raise FileNotFoundError(f"Raw TSV file not found at '{tsv_path}'")

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

        chunk_iterator = pd.read_csv(
            tsv_path,
            sep="\t",
            on_bad_lines="warn",
            usecols=columns_to_read,
            low_memory=False,
            chunksize=100000,
        )

        filtered_chunks = []
        for chunk in tqdm(
            chunk_iterator, desc="    Filtering Chunks", disable=self.verbose == 0
        ):
            chunk = chunk[chunk[external_schema.organism] == "Homo sapiens"].copy()
            chunk.replace(r"^\s*$", pd.NA, regex=True, inplace=True)
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
            return pd.DataFrame()
        return pd.concat(filtered_chunks, ignore_index=True)

    @log_step("Standardize Columns")
    def _step_2_standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """步骤2：将列名重命名为项目内部的黄金标准。"""
        external_schema = self.config.data_structure.schema.external.bindingdb
        internal_schema = self.config.data_structure.schema.internal.authoritative_dti
        return df.rename(
            columns={
                external_schema.molecule_id: internal_schema.molecule_id,
                external_schema.protein_id: internal_schema.protein_id,
                external_schema.molecule_sequence: internal_schema.molecule_sequence,
                external_schema.protein_sequence: internal_schema.protein_sequence,
            }
        )

    @log_step("Purify Data (SMILES/Sequence)")
    def _step_3_purify(self, df: pd.DataFrame) -> pd.DataFrame:
        """步骤3：调用通用的净化模块，进行深度清洗。"""
        return purify_dti_dataframe_parallel(df, self.config)

    @log_step("Filter by Molecular Properties")
    def _step_4_filter_properties(self, df: pd.DataFrame) -> pd.DataFrame:
        """步骤4：根据分子理化性质进行过滤。"""
        return filter_molecules_by_properties(df, self.config)

    @log_step("Finalize and De-duplicate")
    def _step_5_finalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """步骤5：添加Label，清理数据类型，并进行最终去重。"""
        internal_schema = self.config.data_structure.schema.internal

        # 选取最终需要的列
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
        【契约实现】流水线编排器。
        """
        # 创建一个初始的空DataFrame，作为流水线的起点
        df = pd.DataFrame()

        # 按顺序执行流水线步骤
        df = self._step_1_load_and_filter(df)
        if df.empty:
            return df

        df = self._step_2_standardize_columns(df)
        if df.empty:
            return df

        df = self._step_3_purify(df)
        if df.empty:
            return df

        df = self._step_4_filter_properties(df)
        if df.empty:
            return df

        df = self._step_5_finalize(df)

        print(f"\n✅ [{self.__class__.__name__}] Raw processing pipeline complete.")
        return df


# --------------------------------------------------------------------------
# 独立运行的入口点 (保持我们最终的“手动Compose”模式)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # ... (这里的代码与我们之前确定的最终极简版完全一致，无需修改)
    pass

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
        raise e

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
    else:
        print("\n⚠️  Processor returned an empty DataFrame. No file was saved.")
