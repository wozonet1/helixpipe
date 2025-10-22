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

from nasnet.typing import AppConfig

# --- 全局初始化 (在模块级别执行一次，避免在每个并行进程中重复加载) ---

# a. 初始化PAINS过滤器
#    这是一个比较耗时的操作，放在全局可以显著提速
params = FilterCatalog.FilterCatalogParams()
# 我们选择最常见的 PAINS A, B, C 三个集合
params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)
params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B)
params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C)
PAINS_CATALOG = FilterCatalog.FilterCatalog(params)
print("--> [Purifiers] PAINS filter catalog initialized globally.")

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
                descriptor_dict["LogP"] = Descriptors.MolLogP(mol)
                descriptor_dict["HBD"] = Descriptors.NumHDonors(mol)
                descriptor_dict["HBA"] = Descriptors.NumHAcceptors(mol)
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


# --- 主过滤器函数 (并行版) ---


def filter_molecules_by_properties(
    df: pd.DataFrame, config: "AppConfig"
) -> pd.DataFrame:
    """
    【V3 最终版】根据配置，应用一个多阶段的、并行的分子理化性质过滤流水线。
    """
    # 1. 检查总开关
    try:
        filter_cfg = config.data_params.filtering
        if not filter_cfg.get("enabled", False):
            print("--> [Molecule Filter] Disabled by config. Skipping.")
            return df
    except Exception:
        print("--> [Molecule Filter] No 'filtering' config found. Skipping.")
        return df

    print(
        "\n--- [Molecule Filter] Applying molecular property filters (Parallel Mode)... ---"
    )
    logger = RDLogger.logger()
    logger.setLevel(RDLogger.CRITICAL)
    initial_count = len(df)
    if initial_count == 0:
        return df

    schema = config.data_structure.schema.internal.authoritative_dti
    smiles_col = schema.molecule_sequence

    # --- 2. 并行计算所有描述符 ---
    print(f"    - Step 1: Calculating descriptors for {initial_count} molecules...")

    n_jobs = config.runtime.cpus
    num_chunks = min(len(df) // 1000, n_jobs * 4) if len(df) > 1000 else n_jobs
    if num_chunks <= 0:
        num_chunks = 1

    smiles_chunks = np.array_split(df[smiles_col], num_chunks)

    with Parallel(n_jobs=n_jobs) as parallel:
        descriptor_dfs = parallel(
            delayed(_calculate_descriptors_for_chunk)(chunk)
            for chunk in tqdm(
                smiles_chunks,
                desc="      - Descriptor Chunks",
                disable=config.runtime.verbose == 0,
            )
        )

    descriptors_df = pd.concat(d for d in descriptor_dfs if not d.empty)
    df_with_props = df.join(descriptors_df)

    # 移除计算失败的分子
    df_with_props.dropna(subset=["MW", "LogP", "QED", "SA_Score"], inplace=True)

    # --- 3. 应用过滤流水线 ---
    print("    - Step 2: Applying filter pipeline...")
    filtered_df = df_with_props

    # a. PAINS 过滤
    if filter_cfg.get("apply_pains_filter", False):
        filtered_df = filtered_df[~filtered_df["is_pains"]]
        if config.runtime.verbose > 0:
            print(f"      - After PAINS filter: {len(filtered_df)} molecules remain.")

    # b. 分子量
    if (mw_cfg := filter_cfg.get("molecular_weight")) is not None:
        filtered_df = filtered_df[
            filtered_df["MW"].between(
                mw_cfg.get("min", -np.inf), mw_cfg.get("max", np.inf)
            )
        ]
        if config.runtime.verbose > 0:
            print(f"      - After MW filter: {len(filtered_df)} molecules remain.")

    # c. LogP
    if (logp_cfg := filter_cfg.get("logp")) is not None:
        filtered_df = filtered_df[
            filtered_df["LogP"].between(
                logp_cfg.get("min", -np.inf), logp_cfg.get("max", np.inf)
            )
        ]
        if config.runtime.verbose > 0:
            print(f"      - After LogP filter: {len(filtered_df)} molecules remain.")

    # d. 氢键供体
    if (hbd_cfg := filter_cfg.get("h_bond_donors")) is not None:
        filtered_df = filtered_df[filtered_df["HBD"] <= hbd_cfg.get("max", np.inf)]
        if config.runtime.verbose > 0:
            print(f"      - After HBD filter: {len(filtered_df)} molecules remain.")

    # e. 氢键受体
    if (hba_cfg := filter_cfg.get("h_bond_acceptors")) is not None:
        filtered_df = filtered_df[filtered_df["HBA"] <= hba_cfg.get("max", np.inf)]
        if config.runtime.verbose > 0:
            print(f"      - After HBA filter: {len(filtered_df)} molecules remain.")

    # f. QED 评分
    if (qed_cfg := filter_cfg.get("qed")) is not None:
        filtered_df = filtered_df[filtered_df["QED"] >= qed_cfg.get("min", -np.inf)]
        if config.runtime.verbose > 0:
            print(f"      - After QED filter: {len(filtered_df)} molecules remain.")

    # g. SA Score
    if (sa_score_cfg := filter_cfg.get("sa_score")) is not None:
        filtered_df = filtered_df[
            filtered_df["SA_Score"] <= sa_score_cfg.get("max", np.inf)
        ]
        if config.runtime.verbose > 0:
            print(
                f"      - After SA Score filter: {len(filtered_df)} molecules remain."
            )

    final_count = len(filtered_df)

    # --- 4. 清理并返回 ---
    # 只保留原始列，丢弃我们添加的属性列
    final_df = filtered_df[df.columns]

    num_removed = initial_count - final_count
    print(
        f"--- [Molecule Filter] Complete. Removed {num_removed} of {initial_count} molecules. ---"
    )

    return final_df
