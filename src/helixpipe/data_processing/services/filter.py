import logging

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger
from rdkit.Chem import QED, Descriptors, FilterCatalog
from rdkit.Contrib.SA_Score import sascorer  # type: ignore

# 【核心】导入SA Score的计算器和PAINS的过滤器
# 【核心】导入SA Score的计算器和PAINS的过滤器
# 现在，我们可以直接像导入顶层模块一样导入它
from tqdm import tqdm

from helixpipe.typing import AppConfig

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

# 让tqdm能和pandas的apply方法优雅地协作
# 在并行化场景下，我们将主要用tqdm来包裹joblib的调用
tqdm.pandas()


def _calculate_descriptors_for_chunk(smiles_series: pd.Series) -> pd.DataFrame:
    """
    【V3 - 健壮版】为SMILES块计算描述符，并保持原始索引。
    """
    results = []
    logger = RDLogger.logger()
    logger.setLevel(RDLogger.CRITICAL)
    # 【核心修正1】保留原始索引，以便后续安全合并
    for index, smiles in smiles_series.items():
        # 确保只处理字符串
        if not isinstance(smiles, str):
            continue

        mol = Chem.MolFromSmiles(smiles)

        descriptor_dict = {
            "MW": np.nan,
            "LogP": np.nan,
            "HBD": np.nan,
            "HBA": np.nan,
            "QED": np.nan,
            "SA_Score": np.nan,
            "is_pains": True,  # 默认为True，只有成功解析且无匹配的才为False
        }

        if mol:
            try:
                descriptor_dict["MW"] = Descriptors.MolWt(mol)
                descriptor_dict["LogP"] = Descriptors.MolLogP(mol)  # type: ignore
                descriptor_dict["HBD"] = Descriptors.NumHDonors(mol)  # type: ignore
                descriptor_dict["HBA"] = Descriptors.NumHAcceptors(mol)  # type: ignore
                descriptor_dict["QED"] = QED.qed(mol)
                descriptor_dict["SA_Score"] = sascorer.calculateScore(mol)
                descriptor_dict["is_pains"] = PAINS_CATALOG.HasMatch(mol)
            except Exception:
                # 容忍单个分子计算失败，保留NaN
                pass

        # 【核心修正2】将原始索引与结果一起存储
        results.append({"original_index": index, **descriptor_dict})

    if not results:
        return pd.DataFrame()

    # 将结果转换为DataFrame，并设置原始索引
    return pd.DataFrame(results).set_index("original_index")


def filter_molecules_by_properties(
    smiles_series: pd.Series, config: AppConfig
) -> pd.Series:
    """
    【V7 - 极限调试版】
    为每一步过滤增加详细的日志打印，以追踪数据变化。
    """
    filter_cfg = config.data_params.filtering
    # 【DEBUG】强制开启verbose模式，以便在测试中看到打印信息
    verbose = config.runtime.verbose
    if not filter_cfg.enabled:
        if verbose > 0:
            logger.info("--> [Molecule Filter] Disabled by config. Skipping.")
        return pd.Series(True, index=smiles_series.index, dtype=bool)

    if verbose > 0:
        logger.info(
            "\n--- [Molecule Filter] Applying property filters to SMILES Series... ---"
        )

    initial_count = len(smiles_series)
    if initial_count == 0:
        return pd.Series(dtype=bool)

    # --- 1. 并行计算所有描述符 ---
    if verbose > 0:
        logger.info(
            f"    - Step 1: Calculating descriptors for {initial_count} unique molecules..."
        )

    n_jobs = config.runtime.cpus
    num_chunks = min(max(1, initial_count // 1000), n_jobs * 4)
    # 确保即使在dropna后仍有数据可处理
    smiles_to_process = smiles_series.dropna()
    if smiles_to_process.empty:
        if verbose > 0:
            logger.info(
                "--- [Molecule Filter] Complete. 0 molecules passed (all inputs were NaN). ---"
            )
        return pd.Series(False, index=smiles_series.index, dtype=bool)

    smiles_chunks = np.array_split(smiles_to_process, num_chunks)

    with Parallel(n_jobs=n_jobs) as parallel:
        descriptor_dfs = parallel(
            delayed(_calculate_descriptors_for_chunk)(chunk) for chunk in smiles_chunks
        )
    descriptors_df = pd.concat(
        [d for d in descriptor_dfs if d is not None and not d.empty]
    )

    if verbose > 1:
        logger.debug("\n      - [DEBUG] Descriptors calculated. DataFrame sample:")
        logger.debug(descriptors_df.to_string())
        logger.debug(f"      - [DEBUG] Shape of descriptors_df: {descriptors_df.shape}")

    descriptors_df.dropna(subset=["MW", "LogP", "QED", "SA_Score"], inplace=True)

    if verbose > 1:
        logger.debug(
            "\n      - [DEBUG] Descriptors after dropping NaN core properties:"
        )
        logger.debug(descriptors_df.to_string())
        logger.debug(f"      - [DEBUG] Shape after dropna: {descriptors_df.shape}")

    if descriptors_df.empty:
        if verbose > 0:
            logger.info(
                "--- [Molecule Filter] Complete. 0 molecules passed (all failed descriptor calculation). ---"
            )
        return pd.Series(False, index=smiles_series.index, dtype=bool)

    # --- 2. 应用链式的过滤流水线 ---
    if verbose > 0:
        logger.info(
            "    - Step 2: Applying filter pipeline to generate validity mask..."
        )

    mask = pd.Series(True, index=descriptors_df.index, dtype=bool)
    if verbose > 1:
        logger.debug(f"\n        - [DEBUG] Initial mask count: {mask.sum()}")

    # a. PAINS 过滤
    if filter_cfg.apply_pains_filter:
        pains_mask = ~descriptors_df["is_pains"]
        if verbose > 1:
            logger.debug("\n        - [DEBUG] PAINS Filter Details:")
            logger.debug("          - `is_pains` column sample:")
            logger.debug(descriptors_df["is_pains"].to_string())
            logger.debug("          - `pains_mask` (~is_pains) sample:")
            logger.debug(pains_mask.to_string())

        mask &= pains_mask
        if verbose > 1:
            logger.debug(
                f"        - [DEBUG] Mask count after PAINS filter: {mask.sum()}"
            )

    # b. 分子量
    if (mw_cfg := filter_cfg.molecular_weight) is not None:
        mw_mask = descriptors_df["MW"].between(
            mw_cfg.min or -np.inf, mw_cfg.max or np.inf
        )
        if verbose > 1:
            logger.debug("\n        - [DEBUG] Molecular Weight Filter Details:")
            logger.debug(f"          - Range: min={mw_cfg.min}, max={mw_cfg.max}")
            logger.debug("          - `MW` column sample:")
            logger.debug(descriptors_df["MW"].to_string())
            logger.debug("          - `mw_mask` sample:")
            logger.debug(mw_mask.to_string())

        mask &= mw_mask
        if verbose > 1:
            logger.debug(f"        - [DEBUG] Mask count after MW filter: {mask.sum()}")

    # c. LogP
    if (logp_cfg := filter_cfg.logp) is not None:
        logp_mask = descriptors_df["LogP"].between(
            logp_cfg.min or -np.inf, logp_cfg.max or np.inf
        )
        if verbose > 1:
            logger.debug("\n        - [DEBUG] LogP Filter Details:")
            logger.debug(f"          - Range: min={logp_cfg.min}, max={logp_cfg.max}")
            logger.debug("          - `LogP` column sample:")
            logger.debug(descriptors_df["LogP"].to_string())
            logger.debug("          - `logp_mask` sample:")
            logger.debug(logp_mask.to_string())

        mask &= logp_mask
        if verbose > 1:
            logger.debug(
                f"        - [DEBUG] Mask count after LogP filter: {mask.sum()}"
            )

    # d. 氢键供体 (HBD)
    if (hbd_cfg := getattr(filter_cfg, "h_bond_donors", None)) is not None:
        hbd_mask = descriptors_df["HBD"] <= (hbd_cfg.max or np.inf)
        if verbose > 1:
            logger.debug("\n        - [DEBUG] H-Bond Donors Filter Details:")
            logger.debug(f"          - Range: max={hbd_cfg.max}")
            logger.debug("          - `HBD` column sample:")
            logger.debug(descriptors_df["HBD"].to_string())
            logger.debug("          - `hbd_mask` sample:")
            logger.debug(hbd_mask.to_string())
        mask &= hbd_mask
        if verbose > 1:
            logger.debug(f"        - [DEBUG] Mask count after HBD filter: {mask.sum()}")

    # e. 氢键受体 (HBA)
    if (hba_cfg := getattr(filter_cfg, "h_bond_acceptors", None)) is not None:
        hba_mask = descriptors_df["HBA"] <= (hba_cfg.max or np.inf)
        if verbose > 1:
            logger.debug("\n        - [DEBUG] H-Bond Acceptors Filter Details:")
            logger.debug(f"          - Range: max={hba_cfg.max}")
            logger.debug("          - `HBA` column sample:")
            logger.debug(descriptors_df["HBA"].to_string())
            logger.debug("          - `hba_mask` sample:")
            logger.debug(hba_mask.to_string())
        mask &= hba_mask
        if verbose > 1:
            logger.debug(f"        - [DEBUG] Mask count after HBA filter: {mask.sum()}")

    # f. QED 评分
    if (qed_cfg := getattr(filter_cfg, "qed", None)) is not None:
        qed_mask = descriptors_df["QED"] >= (qed_cfg.min or -np.inf)
        if verbose > 1:
            logger.debug("\n        - [DEBUG] QED Score Filter Details:")
            logger.debug(f"          - Range: min={qed_cfg.min}")
            logger.debug("          - `QED` column sample:")
            logger.debug(descriptors_df["QED"].to_string())
            logger.debug("          - `qed_mask` sample:")
            logger.debug(qed_mask.to_string())
        mask &= qed_mask
        if verbose > 1:
            logger.debug(f"        - [DEBUG] Mask count after QED filter: {mask.sum()}")

    # g. SA Score
    if (sa_score_cfg := getattr(filter_cfg, "sa_score", None)) is not None:
        sa_mask = descriptors_df["SA_Score"] <= (sa_score_cfg.max or np.inf)
        if verbose > 1:
            logger.debug("\n        - [DEBUG] SA Score Filter Details:")
            logger.debug(f"          - Range: max={sa_score_cfg.max}")
            logger.debug("          - `SA_Score` column sample:")
            logger.debug(descriptors_df["SA_Score"].to_string())
            logger.debug("          - `sa_mask` sample:")
            logger.debug(sa_mask.to_string())
        mask &= sa_mask
        if verbose > 1:
            logger.debug(
                f"        - [DEBUG] Mask count after SA Score filter: {mask.sum()}"
            )

    # --- 3. 返回最终的、与原始输入对齐的布尔掩码 ---
    final_mask = pd.Series(False, index=smiles_series.index, dtype=bool)
    passed_indices = mask[mask].index

    if verbose > 1:
        logger.debug("\n      - [DEBUG] Final internal mask (before re-indexing):")
        logger.debug(mask.to_string())
        logger.debug(
            f"      - [DEBUG] Indices that passed all filters: {passed_indices.tolist()}"
        )

    # 使用 .loc 确保即使索引不连续也能正确赋值
    final_mask.loc[passed_indices] = True

    if verbose > 0:
        num_passed = final_mask.sum()
        logger.info(
            f"--- [Molecule Filter] Complete. {num_passed} / {initial_count} molecules passed all filters. ---"
        )

    if verbose > 1:
        logger.debug(
            "\n      - [DEBUG] Final returned mask (aligned to original input):"
        )
        logger.debug(final_mask.to_string())

    return final_mask
