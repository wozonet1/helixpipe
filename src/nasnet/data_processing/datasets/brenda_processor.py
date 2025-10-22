# 文件: src/nasnet/data_processing/datasets/brenda_processor.py

import json
import pickle as pkl
import re
from typing import Dict

import pandas as pd
import research_template as rt
from hydra import compose, initialize_config_dir

# 文件: src/nasnet/data_processing/datasets/brenda_processor.py (生产就绪最终版)
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from nasnet.configs import AppConfig, register_all_schemas
from nasnet.utils import (
    get_path,
    log_step,
    register_hydra_resolvers,
)

from .base_processor import BaseDataProcessor


class BrendaProcessor(BaseDataProcessor):
    """
    一个专门负责处理BRENDA原始JSON数据的处理器 (V3 - 架构兼容最终版)。
    它通过解析JSON，应用亲和力阈值，并使用本地缓存的名称->CID映射，
    来生成一个符合项目标准的交互数据集。
    """

    def _load_raw_data(self) -> pd.DataFrame:
        """
        加载并完全解析原始JSON，返回一个扁平化的DataFrame。
        """
        if self.verbose > 0:
            print(
                f"--- [{self.__class__.__name__}] Step: Loading and Parsing raw BRENDA JSON... ---"
            )

        json_path = get_path(self.config, "raw.raw_json")
        if not json_path.exists():
            raise FileNotFoundError(f"Raw BRENDA JSON file not found at '{json_path}'")

        with open(json_path, "r", encoding="utf-8") as f:
            ec_data = json.load(f).get("data", {})

        if not ec_data:
            return pd.DataFrame()

        brenda_schema = self.config.data_structure.schema.external.brenda
        ligand_fields = OmegaConf.to_container(
            brenda_schema.ligand_fields, resolve=True
        )

        all_interactions = self._parse_json_to_interactions(
            ec_data, brenda_schema, ligand_fields
        )

        if not all_interactions:
            return pd.DataFrame()

        df = pd.DataFrame(all_interactions).drop_duplicates()
        print(
            f"--> Found {len(df)} unique raw (UniProt, Molecule Name) interaction candidates."
        )
        return df

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        执行名称到CID的映射，并重命名列以符合内部schema，为白名单过滤做准备。
        """
        if self.verbose > 0:
            print(
                f"--- [{self.__class__.__name__}] Step: Standardizing columns (Name -> CID mapping)... ---"
            )

        mapped_df = self._map_names_to_cids_local(df)
        if mapped_df.empty:
            return pd.DataFrame()

        internal_schema = self.config.data_structure.schema.internal.authoritative_dti
        standardized_df = mapped_df.rename(
            columns={
                "UniProt_ID": internal_schema.protein_id,
                "PubChem_CID": internal_schema.molecule_id,
            }
        )

        required_cols = [
            internal_schema.protein_id,
            internal_schema.molecule_id,
            "value_nM",
            "relation_type",
        ]
        return standardized_df[required_cols]

    def _transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        接收已经过白名单过滤的DataFrame，执行亲和力筛选和最终格式化。
        """
        if self.verbose > 0:
            print(
                f"--- [{self.__class__.__name__}] Step: Applying final transformations to {len(df)} whitelisted rows... ---"
            )

        df_filtered = self._filter_by_affinity(df)
        if df_filtered.empty:
            return pd.DataFrame()

        df_finalized = self._finalize_dataframe(df_filtered)
        return df_finalized

    # --- 辅助方法 ---

    def _parse_json_to_interactions(
        self, ec_data: Dict, schema: DictConfig, fields: Dict
    ) -> list:
        """从原始JSON字典中解析出交互列表。"""
        all_interactions = []
        iterator = tqdm(
            ec_data.items(), desc="   - Parsing EC entries", disable=self.verbose == 0
        )

        for ec_id, entry in iterator:
            if not isinstance(entry, dict):
                continue

            protein_map = {}
            for prot_id, prot_info in entry.get("protein", {}).items():
                if not isinstance(prot_info, dict):
                    continue
                organism = prot_info.get(schema.protein_organism_key, "").lower()
                if (
                    schema.target_organism_value in organism
                    and schema.protein_accessions_key in prot_info
                ):
                    accessions = prot_info[schema.protein_accessions_key]
                    if isinstance(accessions, list) and accessions:
                        protein_map[prot_id] = accessions

            if not protein_map:
                continue

            for field_name, relation_type in fields.items():
                for item in entry.get(field_name, []):
                    if not isinstance(item, dict) or "proteins" not in item:
                        continue

                    value_str = item.get("value", "")
                    molecule_name, numeric_value = None, None

                    if field_name in [
                        "ki_value",
                        "ic50_value",
                        "km_value",
                        "turnover_number",
                    ]:
                        match = re.match(r"^\s*([0-9eE\.\-]+)\s*\{(.*?)\}", value_str)
                        if match:
                            try:
                                numeric_value = float(match.group(1))
                                molecule_name = match.group(2).strip()
                            except (ValueError, IndexError):
                                continue
                    elif field_name == "inhibitor":
                        molecule_name = value_str

                    if not molecule_name or molecule_name == "?":
                        continue

                    for prot_internal_id in item["proteins"]:
                        if prot_internal_id in protein_map:
                            for uniprot_id in protein_map[prot_internal_id]:
                                all_interactions.append(
                                    {
                                        "UniProt_ID": uniprot_id,
                                        "Molecule_Name": molecule_name,
                                        "relation_type": relation_type,
                                        "value_nM": numeric_value,
                                    }
                                )
        return all_interactions

    def _map_names_to_cids_local(self, df: pd.DataFrame) -> pd.DataFrame:
        """使用本地缓存的映射文件进行名称到CID的转换。"""
        map_path = get_path(self.config, "cache.ids.brenda_name_to_cid")
        if not map_path.exists():
            raise FileNotFoundError(
                f"Local PubChem name-to-CID map not found at '{map_path}'.\n"
                "Please run 'src/nasnet/analysis/scripts/build_name_cid_map.py' first."
            )

        with open(map_path, "rb") as f:
            name_to_cid_map = pkl.load(f)

        def clean_name(name):
            if not isinstance(name, str):
                return None
            name = name.lower()
            name = re.sub(r"\(.*?\)", "", name)
            name = name.split(";")[0]
            return name.strip()

        df["cleaned_name"] = df["Molecule_Name"].apply(clean_name)
        df["PubChem_CID"] = df["cleaned_name"].map(name_to_cid_map)

        # 记录映射成功率
        total_unique_names = df["cleaned_name"].nunique()
        mapped_names = df.dropna(subset=["PubChem_CID"])["cleaned_name"].nunique()
        if self.verbose > 0 and total_unique_names > 0:
            success_rate = (mapped_names / total_unique_names) * 100
            print(
                f"    - Name->CID Mapping: {mapped_names}/{total_unique_names} ({success_rate:.2f}%) unique names successfully mapped."
            )

        df.dropna(subset=["PubChem_CID"], inplace=True)
        if df.empty:
            return pd.DataFrame()

        df["PubChem_CID"] = df["PubChem_CID"].astype(int)

        return df.drop(columns=["cleaned_name", "Molecule_Name"])

    @log_step("Filter by Affinity")
    def _filter_by_affinity(self, df: pd.DataFrame) -> pd.DataFrame:
        """根据亲和力阈值筛选DataFrame。"""
        threshold = self.config.data_params.affinity_threshold_nM

        no_value_mask = df["value_nM"].isna()
        value_pass_mask = (df["value_nM"].notna()) & (df["value_nM"] <= threshold)

        return df[no_value_mask | value_pass_mask].copy()

    @log_step("Finalize DataFrame")
    def _finalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """格式化最终输出。"""
        internal_schema = self.config.data_structure.schema.internal.authoritative_dti

        final_df = df[
            [internal_schema.protein_id, internal_schema.molecule_id, "relation_type"]
        ].copy()
        final_df[internal_schema.label] = 1

        final_df.drop_duplicates(
            subset=[internal_schema.protein_id, internal_schema.molecule_id],
            inplace=True,
            keep="first",
        )

        final_df[internal_schema.molecule_sequence] = None
        final_df[internal_schema.protein_sequence] = None

        final_cols = [
            internal_schema.molecule_id,
            internal_schema.protein_id,
            internal_schema.molecule_sequence,
            internal_schema.protein_sequence,
            internal_schema.label,
            "relation_type",
        ]
        return final_df.reindex(columns=final_cols)


# --- 模仿 gtopdb_processor.py 的独立运行入口 ---
if __name__ == "__main__":
    from argparse import ArgumentParser

    BASE_OVERRIDES = ["data_structure=brenda", "data_params=brenda"]

    parser = ArgumentParser(description="Run the BRENDA processing pipeline.")
    parser.add_argument("user_overrides", nargs="*", help="Hydra overrides")
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
