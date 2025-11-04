# 文件: src/nasnet/data_processing/datasets/bindingdb_processor.py (最终模板方法版)

import sys

import argcomplete
import pandas as pd
import research_template as rt
from hydra import compose
from omegaconf import OmegaConf
from tqdm import tqdm

from nasnet.configs import AppConfig, register_all_schemas
from nasnet.data_processing.services import (
    filter_molecules_by_properties,
    purify_entities_dataframe_parallel,
)
from nasnet.utils import get_path, register_hydra_resolvers

from .base_processor import BaseProcessor


class BindingdbProcessor(BaseProcessor):
    """
    一个专门负责处理BindingDB原始数据的处理器。
    【V6 - 最终差异化流水线版】：
    - 标准化所有可用列（ID和结构）。
    - 在业务过滤步骤中，执行亲和力筛选和分子属性筛选。
    """

    def __init__(self, config: AppConfig):
        super().__init__(config)
        self.external_schema = self.config.data_structure.schema.external.bindingdb

    # --- 步骤 1: 实现数据加载 (不变) ---
    def _load_raw_data(self) -> pd.DataFrame:
        tsv_path = get_path(self.config, "raw.raw_tsv")
        if not tsv_path.exists():
            raise FileNotFoundError(f"Raw TSV file not found at '{tsv_path}'")

        columns_to_read = list(self.external_schema.values())
        chunk_iterator = pd.read_csv(
            tsv_path,
            sep="\t",
            on_bad_lines="warn",
            usecols=lambda c: c in columns_to_read,
            low_memory=False,
            chunksize=100000,
        )
        loaded_chunks = []
        disable_tqdm = self.verbose == 0
        for chunk in tqdm(
            chunk_iterator,
            desc="    - Reading & Pre-filtering Chunks",
            disable=disable_tqdm,
        ):
            chunk = chunk[chunk[self.external_schema.organism] == "Homo sapiens"].copy()
            chunk.dropna(
                subset=[
                    self.external_schema.protein_id,
                    self.external_schema.molecule_id,
                ],
                inplace=True,
            )
            if not chunk.empty:
                loaded_chunks.append(chunk)
        return (
            pd.concat(loaded_chunks, ignore_index=True)
            if loaded_chunks
            else pd.DataFrame()
        )

    # --- 步骤 2: 实现关系提取 (直通) ---
    def _extract_relations(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        对于BindingDB，原始数据已是关系格式，直接返回即可。
        """
        return raw_data

    # --- 步骤 3: 实现ID和结构标准化 ---
    def _standardize_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【修改】职责：将原始DataFrame直接重塑为“规范化交互格式”。
        """
        if self.verbose > 0:
            print("  - Reshaping DataFrame to canonical interaction format...")

        final_df = pd.DataFrame()

        # 直接从 external_schema 映射到 canonical_schema
        final_df[self.schema.source_id] = df[self.external_schema.molecule_id]
        # TODO: config中
        final_df[self.schema.source_type] = "molecule"
        final_df[self.schema.target_id] = df[self.external_schema.protein_id]
        final_df[self.schema.target_type] = "protein"
        final_df[self.schema.relation_type] = (
            self.config.knwoledge_graph.relation_types.default
        )

        # 【重要】保留原始结构和亲和力列，以便下一步过滤使用
        final_df["smiles"] = df[self.external_schema.molecule_sequence]
        for aff_type in [
            self.external_schema.ki,
            self.external_schema.ic50,
            self.external_schema.kd,
        ]:
            if aff_type in df.columns:
                final_df[aff_type] = df[aff_type]

        return final_df

    # --- 步骤 4 (覆盖): 实现业务规则过滤 ---
    def _filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【覆盖实现】执行特定于BindingDB的业务规则过滤：
        1. 亲和力筛选。
        2. 分子属性筛选。
        """
        # 1. 计算并筛选亲和力
        for aff_type in [
            self.external_schema.ki,
            self.external_schema.ic50,
            self.external_schema.kd,
        ]:
            if aff_type in df.columns:
                df[aff_type] = pd.to_numeric(
                    df[aff_type].astype(str).str.replace("[><]", "", regex=True),
                    errors="coerce",
                )
        df["affinity_nM"] = (
            df[self.external_schema.ki]
            .fillna(df[self.external_schema.kd])
            .fillna(df[self.external_schema.ic50])
        )

        affinity_threshold = self.config.data_params.affinity_threshold_nM
        df.dropna(subset=["affinity_nM"], inplace=True)
        df_filtered = df[df["affinity_nM"] <= affinity_threshold].copy()
        if df_filtered.empty:
            return pd.DataFrame()

        # 2. 分子属性筛选 (因为这是主数据集，我们应用此步骤)
        #    这个函数需要标准的'SMILES'列名，这在_standardize_ids中已完成
        #     需要使用SMILES,所以先purify一下,不影响之后structure_provider作为唯一事实
        df_purified = purify_entities_dataframe_parallel(
            df_filtered, config=self.config
        )
        df_final_filtered = filter_molecules_by_properties(df_purified, self.config)
        if not df_final_filtered.empty:
            schema_config = self.config.data_structure.schema.internal.authoritative_dti
            df_final_filtered[schema_config.relation_type] = (
                self.config.knwoledge_graph.relation_types.default
            )
        return df_final_filtered


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
    argcomplete.autocomplete(parser)
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
