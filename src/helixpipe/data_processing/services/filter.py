import logging
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger
from rdkit.Chem import QED, Descriptors, FilterCatalog
from rdkit.Contrib.SA_Score import sascorer  # type: ignore

# 【核心】导入SA Score的计算器和PAINS的过滤器
# 现在，我们可以直接像导入顶层模块一样导入它
from tqdm import tqdm

from helixpipe.configs import FilteringConfig

logger = logging.getLogger(__name__)
# --- 全局初始化 (在模块级别执行一次，避免在每个并行进程中重复加载) ---

# a. 初始化PAINS过滤器
#    这是一个比较耗时的操作，放在全局可以显著提速
params = FilterCatalog.FilterCatalogParams()
params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)  # type: ignore
params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B)  # type: ignore
params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C)  # type: ignore
PAINS_CATALOG = FilterCatalog.FilterCatalog(params)
RDLogger.logger().setLevel(RDLogger.CRITICAL)  # 全局关闭RDKit的冗余日志

COL_MW = "MW"
COL_LOGP = "LogP"
COL_HBD = "HBD"
COL_HBA = "HBA"
COL_QED = "QED"
COL_SA = "SA_Score"
COL_IS_PAINS = "is_pains"

# 阈值列名后缀
SUFFIX_MIN = "_min"
SUFFIX_MAX = "_max"
COL_CHECK_PAINS = "check_pains"
# 让tqdm能和pandas的apply方法优雅地协作
# 在并行化场景下，我们将主要用tqdm来包裹joblib的调用
tqdm.pandas()


def _calculate_chunk(smiles_series: pd.Series) -> pd.DataFrame:
    """
    (私有) 处理一个 SMILES 分块，计算所有理化性质。
    """
    results = []
    # 再次确保子进程中日志关闭
    RDLogger.logger().setLevel(RDLogger.CRITICAL)

    for index, smiles in smiles_series.items():
        if not isinstance(smiles, str):
            continue

        mol = Chem.MolFromSmiles(smiles)

        # 默认值为 NaN
        row: dict[str, Any] = {
            "original_index": index,
            COL_MW: np.nan,
            COL_LOGP: np.nan,
            COL_HBD: np.nan,
            COL_HBA: np.nan,
            COL_QED: np.nan,
            COL_SA: np.nan,
            COL_IS_PAINS: True,  # 默认设为 True (最坏情况)，只有计算成功且无匹配才设为 False
        }

        if mol:
            try:
                row[COL_MW] = Descriptors.MolWt(mol)
                row[COL_LOGP] = Descriptors.MolLogP(mol)  # type: ignore
                row[COL_HBD] = Descriptors.NumHDonors(mol)  # type: ignore
                row[COL_HBA] = Descriptors.NumHAcceptors(mol)  # type: ignore
                row[COL_QED] = QED.qed(mol)
                row[COL_SA] = sascorer.calculateScore(mol)
                row[COL_IS_PAINS] = PAINS_CATALOG.HasMatch(mol)
            except Exception:
                # 计算出错保持 NaN
                pass

        results.append(row)

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).set_index("original_index")


def calculate_molecular_properties(smiles_series: pd.Series, cpus: int) -> pd.DataFrame:
    """
    【公共 API】纯粹的属性计算函数。

    Args:
        smiles_series: 包含 SMILES 字符串的 Series。
        cpus: 并行计算使用的 CPU 核心数。

    Returns:
        pd.DataFrame: 包含理化性质的 DataFrame，索引与输入对齐。
                      包含列: MW, LogP, HBD, HBA, QED, SA_Score, is_pains
    """
    if smiles_series.empty:
        return pd.DataFrame()

    # 1. 准备数据
    #    只计算非空且唯一的 SMILES 以节省资源?
    #    注: 为了保持索引对齐逻辑简单，这里暂时不对 unique 进行去重，
    #    因为 IDMapper 阶段已经大致去重了。如果需要极致性能可优化。
    smiles_to_process = smiles_series.dropna()

    if smiles_to_process.empty:
        return pd.DataFrame()

    # 2. 并行计算
    n_jobs = max(1, cpus)
    # 动态决定 chunk size，避免小任务切分过细
    num_chunks = min(len(smiles_to_process) // 50 + 1, n_jobs * 4)
    chunks = np.array_split(smiles_to_process, num_chunks)

    # 使用 tqdm 显示进度
    # 这里的 total 虽然不精确，但足够看进度
    results = Parallel(n_jobs=n_jobs)(
        delayed(_calculate_chunk)(chunk)
        for chunk in tqdm(chunks, desc="Calculating Molecular Properties", leave=False)
    )
    if not results:
        return pd.DataFrame()
    # 3. 合并结果
    props_df = pd.concat([r for r in results if r is not None and not r.empty])

    # 4. 清理无效计算行 (任何关键属性为 NaN 的行都视为计算失败)
    #    注意：这里我们不根据阈值过滤，只根据"是否计算成功"过滤
    valid_mask = props_df[[COL_MW, COL_LOGP]].notna().all(axis=1)

    return props_df[valid_mask].copy()


def apply_static_filter(props: pd.DataFrame, config: FilteringConfig) -> pd.Series:
    """
    【给 Processor 使用】应用单一的、静态的过滤配置。

    此函数用于 Processor 内部的“机会主义过滤”。它接收计算好的属性 DataFrame
    和一个静态的 FilteringConfig 对象，返回一个布尔掩码。

    Args:
        props: 包含分子属性的 DataFrame (必须包含 COL_MW, COL_LOGP 等列)。
        config: 单个 FilteringConfig 对象 (例如 BindingdbParams.filtering)。

    Returns:
        pd.Series: 布尔掩码，True 表示保留，False 表示过滤。
    """
    # 0. 如果 DataFrame 为空，直接返回空 Series
    if props.empty:
        return pd.Series(dtype=bool)

    # 1. 如果配置未启用，全部保留
    if not config.enabled:
        return pd.Series(True, index=props.index)

    # 2. 初始化掩码 (默认全部为 True)
    mask = pd.Series(True, index=props.index)

    # 3. 逐个应用过滤条件
    #    注意：我们需要检查每个子配置是否存在 (is not None)
    #    以及子配置中的具体阈值是否存在 (is not None)

    # --- Molecular Weight (Range: min, max) ---
    if config.molecular_weight:
        if config.molecular_weight.min is not None:
            mask &= props[COL_MW] >= config.molecular_weight.min
        if config.molecular_weight.max is not None:
            mask &= props[COL_MW] <= config.molecular_weight.max

    # --- LogP (Range: min, max) ---
    if config.logp:
        if config.logp.min is not None:
            mask &= props[COL_LOGP] >= config.logp.min
        if config.logp.max is not None:
            mask &= props[COL_LOGP] <= config.logp.max

    # --- H-Bond Donors (Max) ---
    if config.h_bond_donors and config.h_bond_donors.max is not None:
        mask &= props[COL_HBD] <= config.h_bond_donors.max

    # --- H-Bond Acceptors (Max) ---
    if config.h_bond_acceptors and config.h_bond_acceptors.max is not None:
        mask &= props[COL_HBA] <= config.h_bond_acceptors.max

    # --- QED (Min) ---
    if config.qed and config.qed.min is not None:
        mask &= props[COL_QED] >= config.qed.min

    # --- SA Score (Max) ---
    if config.sa_score and config.sa_score.max is not None:
        mask &= props[COL_SA] <= config.sa_score.max

    # --- PAINS (Boolean) ---
    # 逻辑：如果配置要求应用 PAINS 过滤，则剔除那些 is_pains 为 True 的行
    if config.apply_pains_filter:
        mask &= ~props[COL_IS_PAINS]

    return mask


def apply_dynamic_filter(props: pd.DataFrame, criteria: pd.DataFrame) -> pd.Series:
    """
    【给 Validator 使用】应用动态的、每行不同的标准。

    Args:
        props: 分子属性表 (MW, LogP...)
        criteria: 标准表 (MW_min, MW_max, check_pains...)，索引必须与 props 对齐
    """
    # 向量化比较，极快
    mask = props[COL_MW].between(
        criteria[COL_MW + SUFFIX_MIN], criteria[COL_MW + SUFFIX_MAX]
    )
    mask &= props[COL_LOGP].between(
        criteria[COL_LOGP + SUFFIX_MIN], criteria[COL_LOGP + SUFFIX_MAX]
    )
    mask &= props[COL_HBD] <= criteria[COL_HBD + SUFFIX_MAX]
    mask &= props[COL_HBA] <= criteria[COL_HBA + SUFFIX_MAX]
    mask &= props[COL_QED] >= criteria[COL_QED + SUFFIX_MIN]
    mask &= props[COL_SA] <= criteria[COL_SA + SUFFIX_MAX]

    # PAINS 逻辑: (是PAINS分子) AND (需要检查) -> 过滤掉
    # 等价于: 保留 (~是PAINS) OR (~需要检查)
    mask &= (~props[COL_IS_PAINS]) | (~criteria[COL_CHECK_PAINS])

    return mask


def filter_molecules_by_properties(
    smiles_series: pd.Series, config: FilteringConfig
) -> pd.Series:
    """
    (Legacy Wrapper) 一站式计算并应用静态过滤。
    用于 Processor 的“机会主义过滤”。
    """
    # 假设 cpus 默认为 1 或者从哪里获取，这里简化处理
    # 实际项目中建议 Processor 直接分别调用 calculate 和 apply
    props = calculate_molecular_properties(smiles_series, cpus=4)
    mask = apply_static_filter(props, config)

    # 需要返回与原始 smiles_series 对齐的 mask
    final_mask = pd.Series(False, index=smiles_series.index)
    final_mask.loc[mask.index] = mask
    return final_mask
