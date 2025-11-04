# 文件: src/nasnet/data_processing/datasets/brenda_processor.py (生产就绪最终版)

import json
import pickle as pkl
import re
from typing import Any, Dict, List

import argcomplete
import pandas as pd
import research_template as rt
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tqdm import tqdm

from nasnet.configs import AppConfig, register_all_schemas
from nasnet.utils import (
    get_path,
    register_hydra_resolvers,
)

from .base_processor import BaseProcessor


class BrendaProcessor(BaseProcessor):
    """
    专门处理BRENDA JSON数据的处理器 (V4 - 流水线重构版)。
    """

    def __init__(self, config: AppConfig):
        super().__init__(config)
        self.external_schema = self.config.data_structure.schema.external.brenda
        self._name_to_cid_map: Dict[str, int] | None = None

    # --- 实现 BaseProcessor 的抽象方法 ---

    def _load_raw_data(self) -> Dict[str, Any]:
        """步骤1: 从磁盘加载原始JSON数据为字典。"""
        json_path = get_path(self.config, "raw.raw_json")
        if not json_path.exists():
            raise FileNotFoundError(f"Raw BRENDA JSON file not found at '{json_path}'")
        with open(json_path, "r", encoding="utf-8") as f:
            # 先加载整个JSON文件
            full_data = json.load(f)

        # 从加载的数据中获取 'data' 字段，如果不存在则返回空字典
        raw_data_dict = full_data.get("data", {})

        # --- 【核心新增】手动添加日志输出 ---
        num_records = len(raw_data_dict)

        # 遵循基类 process 方法中的日志风格
        # 只有在 verbose > 0 时才打印
        if self.verbose > 0:
            print(f"  - Output size: {num_records} items (EC entries)")

        # 如果加载的数据为空，也打印一个警告
        if num_records == 0:
            print("  - WARNING: Loaded 0 EC entries from the JSON file.")

        return raw_data_dict

    def _extract_relations(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        """
        【V4.2 Processor自治版】从原始字典中解析关系，并在内部完成分组与映射。
        """
        # --- 步骤 1: 在方法内部硬编码映射逻辑 ---
        # 这个字典将原始字段名映射到我们期望的【最终图边类型】
        ligand_fields: Dict[str, str] = self.external_schema.ligand_fields
        flag_names = self.config.knowledge_graph.relation_types
        # TODO: 如何解决这个转换硬编码?
        FIELD_TO_FINAL_RELATION_MAP: Dict[str, str] = {
            ligand_fields["inhibitor"]: flag_names.inhibits,
            ligand_fields["ki_value"]: flag_names.inhibits,
            ligand_fields["ic50_value"]: flag_names.inhibits,
            ligand_fields["km_value"]: flag_names.catalyzes,
            ligand_fields["turnover_number"]: flag_names.catalyzes,
        }

        # --- 步骤 2: 从配置中获取需要处理的字段列表 ---
        # ligand_fields现在只是一个我们感兴趣的字段名列表
        fields_to_process = list(self.external_schema.ligand_fields.keys())

        prot_organism_key = self.external_schema.protein_organism_key
        prot_accessions_key = self.external_schema.protein_accessions_key
        target_organism = self.external_schema.target_organism_value
        rel_type_col = self.schema.relation_type
        all_interactions = []

        disable_tqdm = self.verbose == 0
        iterator = tqdm(
            raw_data.items(), desc="   - Parsing EC entries", disable=disable_tqdm
        )

        for ec_id, entry in iterator:
            if not isinstance(entry, dict):
                continue
            # 步骤 2.1: 为当前EC条目，预先构建一个内部蛋白质ID到UniProt ID列表的映射
            protein_map: Dict[str, List[str]] = {}
            for prot_internal_id, prot_info in entry.get("protein", {}).items():
                if not isinstance(prot_info, dict):
                    continue

                organism = prot_info.get(prot_organism_key, "").lower()

                if target_organism in organism and prot_accessions_key in prot_info:
                    accessions = prot_info[prot_accessions_key]
                    if isinstance(accessions, list) and accessions:
                        protein_map[prot_internal_id] = accessions

            # 如果这个EC条目中没有找到任何人类蛋白质，则跳过后续的配体解析
            if not protein_map:
                continue

            # --- 步骤 3: 遍历我们感兴趣的字段，并使用内部映射 ---
            for field_name in fields_to_process:
                # 从内部映射字典中获取该字段对应的最终关系类型
                final_edge_type = FIELD_TO_FINAL_RELATION_MAP.get(field_name)

                # 如果某个字段没有在我们的内部映射中定义，就跳过它
                if not final_edge_type:
                    continue

                # 遍历该字段下的所有配体条目
                for item in entry.get(field_name, []):
                    if not isinstance(item, dict) or "proteins" not in item:
                        continue

                    value_str = item.get("value", "")
                    molecule_name: str = ""
                    numeric_value: float | None = None

                    # 根据字段类型，解析出分子名称和可选的数值
                    if field_name in [
                        "ki_value",
                        "ic50_value",
                        "km_value",
                        "turnover_number",
                    ]:
                        # 解析格式如 "12.3 {Aspirin}" 的字符串
                        match = re.match(r"^\s*([0-9eE\.\-]+)\s*\{(.*?)\}", value_str)
                        if match:
                            try:
                                numeric_value = float(match.group(1))
                                molecule_name = match.group(2).strip()
                            except (ValueError, IndexError):
                                # 如果数值或名称解析失败，则跳过此条目
                                continue
                        else:
                            # 无法匹配格式，跳过
                            continue
                    elif field_name == "inhibitor":
                        # inhibitor 字段直接是分子名称
                        molecule_name = value_str.strip()
                    else:
                        # 以后如果增加新的字段类型，可以在这里扩展
                        # 当前所有已知的字段都已覆盖，所以理论上不会进入这里
                        continue

                    # 如果没有有效的分子名称，则跳过
                    if not molecule_name or molecule_name == "?":
                        continue

                    # 步骤 3.1: 关联配体与蛋白质，生成交互记录
                    for prot_internal_id in item["proteins"]:
                        # 检查这个配体关联的蛋白质是否是我们已经筛选出的人类蛋白质
                        if prot_internal_id in protein_map:
                            # 一个配体可能与多个UniProt ID关联 (来自同一个蛋白质条目)
                            for uniprot_id in protein_map[prot_internal_id]:
                                all_interactions.append(
                                    {
                                        "UniProt_ID": uniprot_id,
                                        "Molecule_Name": molecule_name,
                                        # 【核心】直接赋值最终的图边类型
                                        rel_type_col: final_edge_type,
                                        "value_nM": numeric_value,
                                    }
                                )

        # 循环结束后，检查是否收集到了任何交互
        if not all_interactions:
            # 返回一个空的DataFrame，流水线将在下一步优雅地终止
            return pd.DataFrame()

        return pd.DataFrame(all_interactions).drop_duplicates()

    def _standardize_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """步骤3: 将'Molecule_Name'映射为'PubChem_CID'，并重命名列以符合内部标准。"""
        if df.empty:
            return df
        if self._name_to_cid_map is None:
            self._name_to_cid_map = self._load_name_cid_map()
        # 执行 Name -> CID 映射
        unique_names = df["Molecule_Name"].dropna().unique()
        cleaned_names = pd.Series(
            [self._clean_name(n) for n in unique_names], index=unique_names
        )
        cid_map = cleaned_names.map(self._name_to_cid_map)

        # 将映射结果 merge 回原始 df
        df = df.merge(
            cid_map.rename("PubChem_CID"),
            left_on="Molecule_Name",
            right_index=True,
            how="left",
        )

        # 重命名列
        return df.rename(columns={"UniProt_ID": self.schema.protein_id})

    def _filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """步骤4: 根据 relation_type 应用不同的亲和力/动力学阈值。"""
        if df.empty or "value_nM" not in df.columns:
            return df

        inhibitor_threshold = self.config.data_params.affinity_threshold_nM
        substrate_threshold = self.config.data_params.km_threshold_nM  # 新增参数

        # 创建一个默认通过的掩码
        pass_mask = pd.Series(True, index=df.index)

        # 对 inhibitor 应用过滤
        if inhibitor_threshold is not None:
            inhibitor_mask = df["relation_type"] == "inhibitor"
            pass_mask[inhibitor_mask] = (
                df.loc[inhibitor_mask, "value_nM"] <= inhibitor_threshold
            )

        # 对 substrate 应用过滤
        if substrate_threshold is not None:
            substrate_mask = df["relation_type"] == "substrate"
            pass_mask[substrate_mask] = (
                df.loc[substrate_mask, "value_nM"] <= substrate_threshold
            )

        return df[pass_mask]

    # --- 私有辅助方法 ---

    def _load_name_cid_map(self) -> Dict[str, int]:
        """加载本地缓存的名称到CID的映射文件。"""
        map_path = get_path(self.config, "cache.ids.brenda_name_to_cid")
        if not map_path.exists():
            raise FileNotFoundError(
                f"PubChem name-to-CID map not found at '{map_path}'."
            )
        print(f"   - Loading name-to-CID map from '{map_path}'...")
        with open(map_path, "rb") as f:
            return pkl.load(f)  # pkl 需要 import

    def _clean_name(self, name: Any) -> str:
        """清洗分子名称以提高映射成功率。"""
        if not isinstance(name, str):
            return ""
        name = name.lower()
        name = re.sub(r"\(.*?\)", "", name)
        name = name.split(";")[0]
        return name.strip()


# ... (独立的 __main__ 入口部分保持不变) ...
# --- 模仿 gtopdb_processor.py 的独立运行入口 ---
if __name__ == "__main__":
    from argparse import ArgumentParser

    BASE_OVERRIDES = ["data_structure=brenda", "data_params=brenda"]

    parser = ArgumentParser(description="Run the BRENDA processing pipeline.")
    parser.add_argument("user_overrides", nargs="*", help="Hydra overrides")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    final_overrides = BASE_OVERRIDES + args.user_overrides

    register_hydra_resolvers()
    register_all_schemas()

    project_root = rt.get_project_root()
    config_dir = str(project_root / "conf")

    with initialize_config_dir(
        config_dir=config_dir, version_base=None, job_name="brenda_process"
    ):
        cfg: AppConfig = compose(config_name="config", overrides=final_overrides)

    print("\n" + "~" * 80)
    print(" " * 25 + "HYDRA COMPOSED CONFIGURATION")
    print(OmegaConf.to_yaml(cfg))
    print("~" * 80 + "\n")

    processor = BrendaProcessor(config=cfg)
    final_df = processor.process()

    if final_df is not None and not final_df.empty:
        print(
            f"\n✅ BRENDA processing complete. Generated {len(final_df)} final interactions."
        )
    else:
        print("\n⚠️ BRENDA processing resulted in an empty dataset.")
