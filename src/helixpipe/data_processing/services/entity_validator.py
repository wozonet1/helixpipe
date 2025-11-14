# src/helixpipe/data_processing/services/entity_validator.py

import logging

import pandas as pd

from helixpipe.configs import AppConfig

# 导入所有需要的底层服务
from .filter import filter_molecules_by_properties  # 我们重构后的版本
from .id_validation_service import get_human_uniprot_whitelist, get_valid_pubchem_cids
from .purifiers import (
    validate_protein_structure,
    validate_smiles_structure,
)

logger = logging.getLogger(__name__)


def _validate_molecules(
    id_series: pd.Series, structure_series: pd.Series, config: AppConfig
) -> pd.Series:
    """
    【V2 - 索引修复版】
    """
    if id_series.empty:
        return pd.Series(dtype=bool)

    # 1. ID 白名单校验
    valid_cids = get_valid_pubchem_cids(set(id_series), config)
    id_mask = id_series.isin(valid_cids)

    # 2. 结构有效性校验
    #    canonical_smiles 的索引与 id_series/structure_series 部分对齐
    canonical_smiles = validate_smiles_structure(structure_series)
    structure_mask = canonical_smiles.notna()

    # 3. 理化性质校验
    #    只对那些ID和结构都有效的分子进行最昂贵的理化性质校验
    pre_validated_mask = id_mask & structure_mask
    if not pre_validated_mask.any():
        return pd.Series(False, index=id_series.index, dtype=bool)

    smiles_to_filter = canonical_smiles[pre_validated_mask]
    property_mask = filter_molecules_by_properties(smiles_to_filter, config)

    # 4. 【核心修复】组合所有掩码
    #    创建一个全为False的最终掩码
    final_mask = pd.Series(False, index=id_series.index, dtype=bool)

    #    property_mask 的索引是 pre_validated_mask 的一个子集
    #    我们找到这些通过了最终校验的索引
    passed_indices = property_mask[property_mask].index

    #    只在最终掩码的这些位置上设置为True
    final_mask.loc[passed_indices] = True

    return final_mask


def _validate_proteins(
    id_series: pd.Series, structure_series: pd.Series, config: AppConfig
) -> pd.Series:
    """
    【V2 - 索引安全版】
    """
    if id_series.empty:
        return pd.Series(dtype=bool)

    # 1. 将输入组合成一个临时的DataFrame，以保证索引的绝对对齐
    df = pd.DataFrame({"id": id_series, "structure": structure_series})

    # 2. ID 白名单校验
    valid_pids = get_human_uniprot_whitelist(set(df["id"]), config)
    df["id_is_valid"] = df["id"].isin(valid_pids)

    # 3. 结构有效性校验
    df["structure_is_valid"] = validate_protein_structure(df["structure"])

    # 4. 组合结果，并返回一个与原始输入索引完全对齐的Series
    final_mask = df["id_is_valid"] & df["structure_is_valid"]

    # final_mask 的索引与 df 的索引一致，而 df 的索引又与 id_series/structure_series 一致
    return final_mask


def validate_and_filter_entities(
    entities_df: pd.DataFrame, config: AppConfig
) -> pd.DataFrame:
    """
    【V2 - 极限调试版】
    在调用蛋白质校验器前后增加超详细的日志。
    """
    if config.runtime.verbose > 0:
        logger.info(
            "\n--- [Entity Validator] Starting unified entity validation service... ---"
        )

    if entities_df.empty:
        return pd.DataFrame()

    entity_types = config.knowledge_graph.entity_types
    validated_indices = []

    # --- 分子部分 (我们假设这部分是正确的，暂时保持原样) ---
    molecule_mask = entities_df["entity_type"] == entity_types.molecule
    if molecule_mask.any():
        if config.runtime.verbose > 0:
            logger.info(
                f"  -> Validating {molecule_mask.sum()} '{entity_types.molecule}' entities..."
            )
        molecule_df = entities_df[molecule_mask]
        valid_molecule_mask = _validate_molecules(
            id_series=molecule_df["entity_id"],
            structure_series=molecule_df["structure"],
            config=config,
        )
        validated_indices.extend(
            valid_molecule_mask[valid_molecule_mask].index.tolist()
        )

    # --- 【核心调试区】蛋白质部分 ---
    protein_mask = entities_df["entity_type"] == entity_types.protein
    if protein_mask.any():
        if config.runtime.verbose > 0:
            logger.info(
                f"  -> Validating {protein_mask.sum()} '{entity_types.protein}' entities..."
            )

        protein_df = entities_df[protein_mask]

        # 【核心调用】
        valid_protein_mask = _validate_proteins(
            id_series=protein_df["entity_id"],
            structure_series=protein_df["structure"],
            config=config,
        )

        passed_indices = valid_protein_mask[valid_protein_mask].index.tolist()

        validated_indices.extend(passed_indices)

    # --- 重新组装 ---
    final_df = entities_df.loc[validated_indices].copy()

    if config.runtime.verbose > 0:
        logger.info(
            f"--- [Entity Validator] Complete. {len(final_df)} / {len(entities_df)} entities passed validation. ---"
        )

    return final_df
