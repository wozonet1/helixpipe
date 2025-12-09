# src/helixpipe/data_processing/services/entity_validator.py

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from helixpipe.configs import AppConfig, FilteringConfig

# 导入 filter.py 的计算和执行函数，以及常量
from helixpipe.data_processing.services.filter import (
    COL_CHECK_PAINS,
    COL_HBA,
    COL_HBD,
    COL_LOGP,
    COL_MW,
    COL_QED,
    COL_SA,
    SUFFIX_MAX,
    SUFFIX_MIN,
    apply_dynamic_filter,
    calculate_molecular_properties,
)
from helixpipe.data_processing.services.id_validation_service import (
    get_human_uniprot_whitelist,
    get_valid_pubchem_cids,
)
from helixpipe.data_processing.services.purifiers import (
    validate_protein_structure,
    validate_smiles_structure,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. 策略构建 (Policy Construction)
# ==============================================================================


def _compute_filtering_criteria(
    entities_subset: pd.DataFrame, config: AppConfig
) -> pd.DataFrame:
    """
    【核心逻辑】为每个实体动态计算其过滤标准。

    原则：来源感知放宽 (Source-Aware Relaxation)。
    1. 以全局配置为基准。
    2. 如果实体属于某个特定来源，且该来源配置更宽松，则放宽限制。

    Args:
        entities_subset: 包含 'all_sources' 列的 DataFrame。
        config: 全局 AppConfig。

    Returns:
        pd.DataFrame: 包含动态阈值的 DataFrame (索引与输入对齐)。
    """
    # 1. 准备配置源
    #    这里假设 config.data_params 已经模块化
    # TODO: 通过反射自动获取所有 processor 的配置
    source_configs: Dict[str, Optional[FilteringConfig]] = {
        "bindingdb": getattr(config.data_params.bindingdb, "filtering", None)
        if config.data_params.bindingdb
        else None,
        "brenda": getattr(config.data_params.brenda, "filtering", None)
        if config.data_params.brenda
        else None,
        "gtopdb": getattr(config.data_params.gtopdb, "filtering", None)
        if config.data_params.gtopdb
        else None,
        # 如果有新的 processor，需在此添加映射，或者通过反射自动获取
    }
    global_cfg = config.data_params.filtering

    # 2. 初始化标准 DataFrame
    criteria = pd.DataFrame(index=entities_subset.index)

    # --- 辅助函数：安全获取嵌套配置值 ---
    def get_val(
        cfg: FilteringConfig, attr: str, sub_attr: str, default: float
    ) -> float:
        if not cfg.enabled:
            return default
        sub_cfg = getattr(cfg, attr, None)
        return getattr(sub_cfg, sub_attr, default) if sub_cfg else default

    # --- 初始化：填充全局默认值 ---
    # 使用 +/- inf 作为无限制的默认值，方便后续取 min/max
    criteria[COL_MW + SUFFIX_MIN] = get_val(
        global_cfg, "molecular_weight", "min", -np.inf
    )
    criteria[COL_MW + SUFFIX_MAX] = get_val(
        global_cfg, "molecular_weight", "max", np.inf
    )

    criteria[COL_LOGP + SUFFIX_MIN] = get_val(global_cfg, "logp", "min", -np.inf)
    criteria[COL_LOGP + SUFFIX_MAX] = get_val(global_cfg, "logp", "max", np.inf)

    criteria[COL_HBD + SUFFIX_MAX] = get_val(global_cfg, "h_bond_donors", "max", np.inf)
    criteria[COL_HBA + SUFFIX_MAX] = get_val(
        global_cfg, "h_bond_acceptors", "max", np.inf
    )

    criteria[COL_QED + SUFFIX_MIN] = get_val(global_cfg, "qed", "min", -np.inf)
    criteria[COL_SA + SUFFIX_MAX] = get_val(global_cfg, "sa_score", "max", np.inf)

    # PAINS: 初始值取决于全局是否启用
    criteria[COL_CHECK_PAINS] = global_cfg.enabled and global_cfg.apply_pains_filter

    # 3. 动态放宽
    for src_name, src_cfg in source_configs.items():
        if src_cfg is None:
            continue

        # 找到属于该来源的所有行
        mask = entities_subset["all_sources"].apply(lambda s: src_name in s)

        if not mask.any():
            continue

        if not src_cfg.enabled:
            # 如果该源禁用了过滤，则解除所有限制
            # 逻辑：取最宽松的边界。既然该源说"不过滤"，那么对于属于该源的实体，
            # 无论其他源（或全局配置）怎么说，我们都给予通过的“豁免权”。

            # 1. 范围限制 (Range Filters) -> 设为无限宽
            criteria.loc[mask, COL_MW + SUFFIX_MIN] = -np.inf
            criteria.loc[mask, COL_MW + SUFFIX_MAX] = np.inf

            criteria.loc[mask, COL_LOGP + SUFFIX_MIN] = -np.inf
            criteria.loc[mask, COL_LOGP + SUFFIX_MAX] = np.inf

            # 2. 上限限制 (Max Filters) -> 设为正无穷
            criteria.loc[mask, COL_HBD + SUFFIX_MAX] = np.inf
            criteria.loc[mask, COL_HBA + SUFFIX_MAX] = np.inf
            criteria.loc[mask, COL_SA + SUFFIX_MAX] = np.inf

            # 3. 下限限制 (Min Filters) -> 设为负无穷
            criteria.loc[mask, COL_QED + SUFFIX_MIN] = -np.inf

            # 4. 布尔检查 (Boolean Checks) -> 设为不检查
            criteria.loc[mask, COL_CHECK_PAINS] = False

            continue

        # 逻辑：取最宽松的边界
        # 最小值：取更小的 (np.minimum)
        criteria.loc[mask, COL_MW + SUFFIX_MIN] = np.minimum(
            criteria.loc[mask, COL_MW + SUFFIX_MIN],
            get_val(src_cfg, "molecular_weight", "min", -np.inf),
        )
        criteria.loc[mask, COL_LOGP + SUFFIX_MIN] = np.minimum(
            criteria.loc[mask, COL_LOGP + SUFFIX_MIN],
            get_val(src_cfg, "logp", "min", -np.inf),
        )
        criteria.loc[mask, COL_QED + SUFFIX_MIN] = np.minimum(
            criteria.loc[mask, COL_QED + SUFFIX_MIN],
            get_val(src_cfg, "qed", "min", -np.inf),
        )

        # 最大值：取更大的 (np.maximum)
        criteria.loc[mask, COL_MW + SUFFIX_MAX] = np.maximum(
            criteria.loc[mask, COL_MW + SUFFIX_MAX],
            get_val(src_cfg, "molecular_weight", "max", np.inf),
        )
        criteria.loc[mask, COL_LOGP + SUFFIX_MAX] = np.maximum(
            criteria.loc[mask, COL_LOGP + SUFFIX_MAX],
            get_val(src_cfg, "logp", "max", np.inf),
        )
        criteria.loc[mask, COL_HBD + SUFFIX_MAX] = np.maximum(
            criteria.loc[mask, COL_HBD + SUFFIX_MAX],
            get_val(src_cfg, "h_bond_donors", "max", np.inf),
        )
        criteria.loc[mask, COL_HBA + SUFFIX_MAX] = np.maximum(
            criteria.loc[mask, COL_HBA + SUFFIX_MAX],
            get_val(src_cfg, "h_bond_acceptors", "max", np.inf),
        )
        criteria.loc[mask, COL_SA + SUFFIX_MAX] = np.maximum(
            criteria.loc[mask, COL_SA + SUFFIX_MAX],
            get_val(src_cfg, "sa_score", "max", np.inf),
        )

        # PAINS: 如果该源不要求检查，则设为 False (逻辑与)
        if not src_cfg.apply_pains_filter:
            criteria.loc[mask, COL_CHECK_PAINS] = False

    return criteria


# ==============================================================================
# 2. 分子校验逻辑 (Molecule Validation)
# ==============================================================================


def _validate_molecules(
    id_series: pd.Series,
    structure_series: pd.Series,
    source_series: pd.Series,
    config: AppConfig,
) -> pd.Series:
    """
    分子校验流水线：ID -> 结构 -> 属性计算 -> 动态过滤。
    """
    if id_series.empty:
        return pd.Series(dtype=bool)

    # 1. 基础校验 (ID 格式 & SMILES 语法)
    valid_cids = get_valid_pubchem_cids(set(id_series), config)
    id_mask = id_series.isin(valid_cids)

    canonical_smiles = validate_smiles_structure(structure_series)
    structure_mask = canonical_smiles.notna()

    pre_validated_mask = id_mask & structure_mask
    if not pre_validated_mask.any():
        return pd.Series(False, index=id_series.index, dtype=bool)

    # 2. 准备数据子集 (只计算基础校验通过的行)
    subset_indices = pre_validated_mask[pre_validated_mask].index
    smiles_subset = canonical_smiles[subset_indices]

    # 3. 计算分子属性 (调用 filter.py)
    props_df = calculate_molecular_properties(smiles_subset, config.runtime.cpus)

    # 注意：calculate_molecular_properties 可能会丢弃计算失败的行
    # 我们需要对齐索引
    valid_calc_indices = props_df.index

    # 4. 构建动态策略 (调用本地函数)
    #    只为计算成功的行构建策略
    criteria_df = _compute_filtering_criteria(
        pd.DataFrame({"all_sources": source_series[valid_calc_indices]}), config
    )

    # 5. 执行过滤 (调用 filter.py)
    #    props_df 和 criteria_df 索引已对齐
    passed_mask = apply_dynamic_filter(props_df, criteria_df)

    # 6. 组合最终掩码
    #    创建一个全 False 的掩码，只在 passed_mask 为 True 的位置设为 True
    final_mask = pd.Series(False, index=id_series.index, dtype=bool)
    # passed_mask[passed_mask] 得到的是值为 True 的那些索引
    final_mask.loc[passed_mask[passed_mask].index] = True

    return final_mask


# ==============================================================================
# 3. 蛋白质校验逻辑 (Protein Validation)
# ==============================================================================


def _validate_proteins(
    id_series: pd.Series, structure_series: pd.Series, config: AppConfig
) -> pd.Series:
    """
    蛋白质校验流水线：ID -> 结构字符集。
    (目前蛋白质暂无复杂的理化性质过滤需求，保持全局一致)
    """
    if id_series.empty:
        return pd.Series(dtype=bool)

    # 1. 临时 DataFrame 保证索引对齐
    df = pd.DataFrame({"id": id_series, "structure": structure_series})

    # 2. ID 白名单 (本地离线检查)
    valid_pids = get_human_uniprot_whitelist(set(df["id"]), config)
    df["id_is_valid"] = df["id"].isin(valid_pids)

    # 3. 结构有效性 (字符集检查)
    df["structure_is_valid"] = validate_protein_structure(df["structure"])

    return df["id_is_valid"] & df["structure_is_valid"]


# ==============================================================================
# 4. 主入口 (Main Entry Point)
# ==============================================================================


def validate_and_filter_entities(
    entities_df: pd.DataFrame, config: AppConfig
) -> pd.DataFrame:
    """
    中心化的实体校验与过滤服务入口。

    Args:
        entities_df: 包含 'entity_id', 'entity_type', 'structure', 'all_sources' 的 DataFrame。
        config: 全局配置。

    Returns:
        pd.DataFrame: 过滤后的纯净 DataFrame。
    """
    if config.runtime.verbose > 0:
        logger.info(
            "\n--- [Entity Validator] Starting unified entity validation service... ---"
        )

    if entities_df.empty:
        return pd.DataFrame()

    entity_types = config.knowledge_graph.entity_types
    validated_indices = []

    # --- 分子部分 ---
    # 使用正则表达式匹配所有分子类型 (drug, ligand, etc.)
    # 假设 'molecule' 是基类名，或者列出所有分子子类型
    molecule_mask = entities_df["entity_type"].isin(
        [
            entity_types.drug,
            entity_types.ligand,
            entity_types.ligand_endo,
            entity_types.ligand_exo,
        ]
    ) | (entities_df["entity_type"] == "molecule")

    if molecule_mask.any():
        if config.runtime.verbose > 0:
            logger.info(f"  -> Validating {molecule_mask.sum()} molecule entities...")
        molecule_df = entities_df[molecule_mask]

        valid_molecule_mask = _validate_molecules(
            id_series=molecule_df["entity_id"],
            structure_series=molecule_df["structure"],
            source_series=molecule_df["all_sources"],  # 关键：传入来源
            config=config,
        )
        validated_indices.extend(
            valid_molecule_mask[valid_molecule_mask].index.tolist()
        )

    # --- 蛋白质部分 ---
    protein_mask = entities_df["entity_type"] == entity_types.protein
    if protein_mask.any():
        if config.runtime.verbose > 0:
            logger.info(f"  -> Validating {protein_mask.sum()} protein entities...")

        protein_df = entities_df[protein_mask]

        valid_protein_mask = _validate_proteins(
            id_series=protein_df["entity_id"],
            structure_series=protein_df["structure"],
            config=config,
        )
        validated_indices.extend(valid_protein_mask[valid_protein_mask].index.tolist())

    # --- 重新组装 ---
    final_df = entities_df.loc[validated_indices].copy()

    if config.runtime.verbose > 0:
        logger.info(
            f"--- [Entity Validator] Complete. {len(final_df)} / {len(entities_df)} entities passed validation. ---"
        )

    return final_df
