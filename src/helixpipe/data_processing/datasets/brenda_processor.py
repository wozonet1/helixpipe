# 文件: src/helixpipe/data_processing/datasets/brenda_processor.py (生产就绪最终版)

import json
import logging
import pickle as pkl
import re
from typing import Any, Dict, List, Union, cast

import argcomplete
import pandas as pd
import research_template as rt
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tqdm import tqdm

from helixpipe.configs import register_all_schemas
from helixpipe.typing import AppConfig
from helixpipe.utils import SchemaAccessor, get_path, register_hydra_resolvers

from .base_processor import BaseProcessor

logger = logging.getLogger(__name__)


class BrendaProcessor(BaseProcessor):
    """
    【V5 - 最终纯净架构版】
    专门处理BRENDA JSON数据的处理器。
    它的职责被纯化为：
    1. 解析复杂的JSON结构以提取蛋白质-配体关系。
    2. 将配体名称映射为PubChem CID。
    3. 应用特定于BRENDA的、基于关系类型的动力学/亲和力阈值。
    """

    def __init__(self, config: AppConfig):
        super().__init__(config)
        self.external_schema = SchemaAccessor(
            self.config.data_structure.schema.external["brenda"]
        )
        # 在初始化时，加载并准备好作为内部状态的“名称->CID”映射字典

    def _load_name_cid_map(self) -> Dict[str, int]:
        """一个私有辅助方法，用于构建并返回 Name -> CID 的映射字典。"""
        map_path = get_path(self.config, "cache.ids.brenda_name_to_cid")
        if not map_path.exists():
            raise FileNotFoundError(
                f"PubChem name-to-CID map not found at '{map_path}'. "
                "Please run the 'build_name_cid_map.py' script first."
            )
        if self.verbose > 0:
            logger.info(
                f"  - [BrendaProcessor Pre-computation] Loading name-to-CID map from '{map_path.name}'..."
            )

        with open(map_path, "rb") as f:
            name_map = cast(Dict[str, int], pkl.load(f))

        if self.verbose > 0:
            logger.info(f"    - Map loaded with {len(name_map)} unique name entries.")
        return name_map

    def _load_raw_data(self) -> Dict[str, Any]:
        """
        步骤1: 从磁盘加载原始JSON数据为字典。
        """
        json_path = get_path(self.config, "raw.raw_json")
        if not json_path.exists():
            raise FileNotFoundError(f"Raw BRENDA JSON file not found at '{json_path}'")

        with open(json_path, "r", encoding="utf-8") as f:
            return cast(Dict[str, int], json.load(f).get("data", {}))

    def _extract_relations(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        """
        步骤2: 从原始字典中解析出交互关系，并附带所有辅助信息。
        """
        all_interactions = []

        # 从配置中获取所有需要的键名
        prot_organism_key = self.external_schema.get_col("protein_organism_key")
        prot_accessions_key = self.external_schema.get_col("protein_accessions_key")
        target_organism = self.external_schema.get_col("target_organism_value")
        ligand_fields = self.external_schema.get_field("ligand_fields")

        disable_tqdm = self.verbose == 0
        iterator = tqdm(
            raw_data.items(),
            desc="    - Parsing BRENDA EC entries",
            disable=disable_tqdm,
        )

        for ec_id, entry in iterator:
            if not isinstance(entry, dict):
                continue

            protein_map = self._build_protein_map_for_ec(
                entry, prot_organism_key, target_organism, prot_accessions_key
            )
            if not protein_map:
                continue

            for field_key, relation_type in ligand_fields.items():
                for item in entry.get(field_key, []):
                    if not isinstance(item, dict) or "proteins" not in item:
                        continue

                    molecule_name, numeric_value = self._parse_ligand_value(
                        item.get("value", ""), field_key
                    )
                    if not molecule_name:
                        continue

                    for prot_internal_id in item["proteins"]:
                        if prot_internal_id in protein_map:
                            for uniprot_id in protein_map[prot_internal_id]:
                                all_interactions.append(
                                    {
                                        "protein_id": uniprot_id,
                                        "molecule_name": molecule_name,
                                        "relation_type": relation_type,
                                        "value": numeric_value,
                                    }
                                )

        return (
            pd.DataFrame(all_interactions).drop_duplicates()
            if all_interactions
            else pd.DataFrame()
        )

    def _standardize_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        步骤3: 将配体名称映射为CID，并重塑为“规范化格式+辅助列”。
        """
        if df.empty:
            return pd.DataFrame()
        self._name_to_cid_map: Dict[str, int] = self._load_name_cid_map()
        # 1. 名称到CID的映射
        df["cleaned_name"] = df["molecule_name"].apply(self._clean_name)
        df["PubChem_CID"] = df["cleaned_name"].map(self._name_to_cid_map)

        # 过滤掉无法映射CID的行
        df.dropna(subset=["PubChem_CID"], inplace=True)
        df["PubChem_CID"] = df["PubChem_CID"].astype(int)

        # 2. 重塑为规范化格式
        final_df = pd.DataFrame()
        entity_names = self.config.knowledge_graph.entity_types

        final_df[self.schema.source_id] = df["PubChem_CID"]
        final_df[self.schema.source_type] = (
            entity_names.ligand
        )  # BRENDA的分子被定义为ligand
        final_df[self.schema.target_id] = df["protein_id"]
        final_df[self.schema.target_type] = entity_names.protein

        # BRENDA的 relation_type 来自其字段名，更加精细
        final_df[self.schema.relation_type] = df["relation_type"]

        # 附带下游需要的“原材料” (BRENDA不提供分子或蛋白质结构)
        final_df["value"] = df["value"]  # 用于下一步过滤

        return final_df

    def _filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        步骤4: 应用特定于BRENDA的、基于关系类型的阈值过滤。
        """
        if df.empty or "value" not in df.columns:
            return df

        if self.verbose > 0:
            logger.info(
                "  - Applying domain-specific filters: Relation-Type based Thresholds..."
            )

        # 从配置获取阈值
        inhibitor_threshold = self.config.data_params.affinity_threshold_nM
        km_threshold = self.config.data_params.km_threshold_nM

        # 从配置获取关系类型名称
        rel_types = self.config.knowledge_graph.relation_types

        # 创建一个默认通过的掩码
        pass_mask = pd.Series(True, index=df.index)

        # 对 inhibitor/ki/ic50 类型的交互应用亲和力阈值
        inhibitor_mask = df[self.schema.relation_type].isin([rel_types.inhibits])
        if inhibitor_mask.any():
            pass_mask[inhibitor_mask] = (
                df.loc[inhibitor_mask, "value"] <= inhibitor_threshold
            )

        # 对 km_value 类型的交互应用Km值阈值
        km_mask = df[self.schema.relation_type] == "km_value"  # 假设配置中定义了
        if km_mask.any():
            pass_mask[km_mask] = df.loc[km_mask, "value"] <= km_threshold

        df_filtered = df[pass_mask]

        if self.verbose > 0:
            logger.info(
                f"    - {len(df_filtered)} / {len(df)} records passed relation-type based filters."
            )

        return df_filtered

    # --- 私有辅助方法 ---

    def _build_protein_map_for_ec(
        self, entry, org_key, target_org, acc_key
    ) -> Dict[str, List[str]]:
        """为单个EC条目构建 内部ID -> [UniProt IDs] 的映射。"""
        protein_map = {}
        for prot_id, prot_info in entry.get("protein", {}).items():
            if (
                isinstance(prot_info, dict)
                and target_org in prot_info.get(org_key, "").lower()
            ):
                accessions = prot_info.get(acc_key)
                if isinstance(accessions, list) and accessions:
                    protein_map[prot_id] = accessions
        return protein_map

    def _parse_ligand_value(
        self, value_str: str, field_key: str
    ) -> tuple[Union[str, None], Union[float, None]]:
        """解析BRENDA中复杂的配体值字符串。"""
        if "value" in field_key:
            match = re.match(r"^\s*([0-9eE.\-]+)\s*\{(.*?)\}", value_str)
            if match:
                try:
                    return match.group(2).strip(), float(match.group(1))
                except (ValueError, IndexError):
                    return None, None
        elif "inhibitor" in field_key:
            name = value_str.strip()
            return (name, None) if name and name != "?" else (None, None)
        return None, None

    def _clean_name(self, name: Any) -> str:
        """清洗分子名称以提高映射成功率。"""
        if not isinstance(name, str):
            return ""
        return re.sub(r"\(.*?\)", "", name.lower()).split(";")[0].strip()


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
        cfg = cast(AppConfig, compose(config_name="config", overrides=final_overrides))

    logger.info("\n" + "~" * 80)
    logger.info(" " * 25 + "HYDRA COMPOSED CONFIGURATION")
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("~" * 80 + "\n")

    processor = BrendaProcessor(config=cfg)
    final_df = processor.process()

    if final_df is not None and not final_df.empty:
        logger.info(
            f"\n✅ BRENDA processing complete. Generated {len(final_df)} final interactions."
        )
    else:
        logger.warning("\n⚠️ BRENDA processing resulted in an empty dataset.")
