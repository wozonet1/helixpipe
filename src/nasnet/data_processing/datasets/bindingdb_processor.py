# 文件: src/data_processing/bindingdb_processor.py

import sys

import pandas as pd
import research_template as rt
from hydra import compose
from omegaconf import OmegaConf
from tqdm import tqdm

from nasnet.configs import register_all_schemas
from nasnet.typing import AppConfig
from nasnet.utils import get_path, log_step, register_hydra_resolvers

from ..services.purifiers import (
    filter_molecules_by_properties,
    purify_dti_dataframe_parallel,
)

# 导入我们新创建的基类和需要的辅助模块
from .base_processor import BaseDataProcessor

# --------------------------------------------------------------------------
# 步骤 2: 定义流水线化的BindingDB处理器
# --------------------------------------------------------------------------


class BindingdbProcessor(BaseDataProcessor):
    """
    一个专门负责处理BindingDB原始数据的处理器。
    【V3 最终版】：实现了清晰的加载/转换分离，并保留了带日志的流水线步骤。
    """

    def _load_raw_data(self) -> pd.DataFrame:
        """
        【契约实现】只负责从BindingDB的原始TSV文件中加载数据，并进行最基础的、
        特定于源的行过滤（例如物种）和列选择。
        """
        print(
            f"--- [{self.__class__.__name__}] Step: Loading and Standardizing raw data... ---"
        )
        external_schema = self.config.data_structure.schema.external.bindingdb

        tsv_path = get_path(self.config, "raw.raw_tsv")
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

        loaded_chunks = []
        # 根据verbose级别决定是否显示tqdm进度条
        disable_tqdm = self.config.runtime.verbose == 0
        for chunk in tqdm(
            chunk_iterator,
            desc="    - Reading & Pre-filtering Chunks",
            disable=disable_tqdm,
        ):
            # 1. 筛选物种
            chunk = chunk[chunk[external_schema.organism] == "Homo sapiens"].copy()

            # 2. 替换空字符串为空值，并过滤掉关键ID缺失的行
            chunk.replace(r"^\s*$", pd.NA, regex=True, inplace=True)
            chunk.dropna(
                subset=[
                    external_schema.protein_id,
                    external_schema.molecule_id,
                ],
                inplace=True,
            )

            if not chunk.empty:
                loaded_chunks.append(chunk)

        if not loaded_chunks:
            print("    - No valid data found after initial loading and filtering.")
            return pd.DataFrame()

        df = pd.concat(loaded_chunks, ignore_index=True)
        print(f"--> Loaded  {len(df)} raw rows for processing.")
        return df

    def _standardize_columns(self, df) -> pd.DataFrame:
        external_schema = self.config.data_structure.schema.external.bindingdb
        internal_schema = self.config.data_structure.schema.internal.authoritative_dti
        df.rename(
            columns={
                external_schema.molecule_id: internal_schema.molecule_id,
                external_schema.protein_id: internal_schema.protein_id,
                external_schema.molecule_sequence: internal_schema.molecule_sequence,
                external_schema.protein_sequence: internal_schema.protein_sequence,
            },
            inplace=True,
        )
        return df

    @log_step("Process & Filter by Affinity")
    def _transform_step_1_affinity(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换步骤1：处理亲和力数值并根据阈值进行过滤。"""
        external_schema = self.config.data_structure.schema.external.bindingdb

        for aff_type in [external_schema.ki, external_schema.ic50, external_schema.kd]:
            df[aff_type] = pd.to_numeric(
                df[aff_type].astype(str).str.replace(">", "").str.replace("<", ""),
                errors="coerce",
            )
        df["affinity_nM"] = (
            df[external_schema.ki]
            .fillna(df[external_schema.kd])
            .fillna(df[external_schema.ic50])
        )

        affinity_threshold = self.config.data_params.affinity_threshold_nM
        df.dropna(subset=["affinity_nM"], inplace=True)

        return df[df["affinity_nM"] <= affinity_threshold].copy()

    @log_step("Purify Data (SMILES/Sequence)")
    def _transform_step_3_purify(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换步骤3：调用通用的净化模块，进行并行的深度格式清洗。"""
        return purify_dti_dataframe_parallel(df, self.config)

    @log_step("Filter by Molecular Properties")
    def _transform_step_4_filter_properties(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换步骤4：根据分子理化性质进行过滤。"""
        return filter_molecules_by_properties(df, self.config)

    @log_step("Finalize and De-duplicate")
    def _transform_step_5_finalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换步骤5：添加Label，清理数据类型，并进行最终去重。"""
        internal_schema = self.config.data_structure.schema.internal.authoritative_dti

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

        # 确保ID列是整数类型，以防万一
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
        【契约实现】作为数据转换流水线的【编排器】。
        它接收已经过ID白名单过滤的DataFrame，然后按顺序调用所有转换子步骤。
        """
        if self.verbose > 0:
            print(
                f"--- [{self.__class__.__name__}] Step: Transforming {len(df)} whitelisted rows... ---"
            )

        # 定义要执行的处理步骤的流水线
        pipeline = [
            self._transform_step_1_affinity,
            self._transform_step_3_purify,
            self._transform_step_4_filter_properties,
            self._transform_step_5_finalize,
        ]

        # 依次执行流水线中的每一步
        for step_func in pipeline:
            df = step_func(df)
            if df.empty:
                # 如果在任何步骤之后DataFrame变为空，提前终止流水线
                print(
                    f"  - Pipeline halted after step '{step_func.__name__}' because DataFrame became empty."
                )
                return pd.DataFrame()

        if self.verbose > 0:
            print(f"\n✅ [{self.__class__.__name__}] Transformation pipeline complete.")
        return df


# --------------------------------------------------------------------------
# Config Store模式下的独立运行入口 (最终版)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # === 阶段 1: 明确定义脚本身份和命令行接口 ===
    from argparse import ArgumentParser

    from hydra import compose, initialize_config_dir

    # a. 这个脚本的固有基础配置
    #    它天生就是用来处理 bindingdb 数据结构的
    BASE_OVERRIDES = ["data_structure=bindingdb"]

    # b. 设置命令行解析器，只接收用户自定义的覆盖参数
    parser = ArgumentParser(
        description="Run the BindingDB processing pipeline with custom Hydra overrides."
    )
    # 'nargs='*' 会将所有额外的命令行参数都收集到一个列表中
    parser.add_argument(
        "user_overrides",
        nargs="*",
        help="Hydra overrides (e.g., data_params=strict_strong)",
    )
    args = parser.parse_args()

    # c. 组合所有覆盖参数
    final_overrides = BASE_OVERRIDES + args.user_overrides

    # === 阶段 2: 注册、加载配置并打印 ===

    # a. 在所有Hydra操作之前，确保解析器已注册
    #    (rt 是 research_template)
    register_hydra_resolvers()
    register_all_schemas()
    # b. 使用 initialize_config_dir 和 compose 来构建最终的配置对象
    try:
        # get_project_root 在非Hydra应用下会使用 Path.cwd()
        # 请确保您是从项目根目录运行此脚本
        project_root = rt.get_project_root()
        config_dir = str(project_root / "conf")
    except Exception as e:
        print(f"❌ 无法确定项目根目录或配置路径。错误: {e}")
        sys.exit(1)  # 明确退出

    with initialize_config_dir(
        config_dir=config_dir, version_base=None, job_name="bindingdb_process"
    ):
        cfg: AppConfig = compose(config_name="config", overrides=final_overrides)

    # c. 打印最终配置以供调试
    print("\n" + "~" * 80)
    print(" " * 25 + "HYDRA COMPOSED CONFIGURATION")
    print("~" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("~" * 80 + "\n")

    # === 阶段 3: 执行核心业务逻辑 ===

    # a. 实例化处理器
    processor = BindingdbProcessor(config=cfg)

    # b. 运行处理流程
    #    基类的 .process() 方法会自动处理缓存、调用 _load_raw_data, _transform_data,
    #    以及最终的保存和验证。
    final_df = processor.process()

    # c. 打印最终总结
    if final_df is not None and not final_df.empty:
        print("\n✅ BindingDB processing complete. Final authoritative file is ready.")
    else:
        print("\n⚠️  BindingDB processing resulted in an empty dataset.")
