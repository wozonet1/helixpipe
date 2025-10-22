import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger

# 【核心】导入SA Score的计算器和PAINS的过滤器
# 【核心】导入SA Score的计算器和PAINS的过滤器
from rdkit.Chem import QED, Descriptors, FilterCatalog

# 现在，我们可以直接像导入顶层模块一样导入它
from rdkit.Contrib.SA_Score import sascorer  # type: ignore
from tqdm import tqdm

from nasnet.typing import AppConfig

from .canonicalizer import canonicalize_smiles

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


# TODO:独立出来
def _purify_chunk(df_chunk: pd.DataFrame) -> pd.DataFrame:
    if df_chunk.empty:
        return df_chunk

    logger = RDLogger.logger()
    logger.setLevel(RDLogger.CRITICAL)

    df = df_chunk.copy()
    # 有关id的在之前已经进行过了,这里只针对SMILES/序列信息进行筛选
    # --- 1. SMILES 净化 (我们的逻辑更严格，保持不变) ---
    smiles_mask_initial = df["SMILES"].apply(
        lambda s: isinstance(s, str) and s.strip() != ""
    )
    df = df[smiles_mask_initial]
    if df.empty:
        return df

    smiles_mask_valid = df["SMILES"].apply(lambda s: Chem.MolFromSmiles(s) is not None)
    df = df[smiles_mask_valid]
    if df.empty:
        return df

    # --- 32 序列 净化 (借鉴DeepPurpose) ---
    # a. 过滤掉None/NaN或非字符串
    df.dropna(subset=["Sequence"], inplace=True)
    df = df[df["Sequence"].apply(isinstance, args=(str,))]
    if df.empty:
        return df

    # b. 【核心改进】使用DeepPurpose启发的、更完整的字符集
    VALID_SEQ_CHARS = "ACDEFGHIKLMNPQRSTVWYU"  # 标准21种氨基酸
    valid_chars_set = set(VALID_SEQ_CHARS)

    def is_valid_protein_sequence(seq: str) -> bool:
        seq_clean = "".join(seq.split()).upper()
        if not seq_clean:
            return False
        return all(char in valid_chars_set for char in seq_clean)

    valid_seq_mask = df["Sequence"].apply(is_valid_protein_sequence)
    df = df[valid_seq_mask]
    if df.empty:
        return df

    # --- 3. SMILES 标准化 (现在是最后一步) ---
    df["SMILES"] = df["SMILES"].apply(canonicalize_smiles)
    df.dropna(subset=["SMILES"], inplace=True)
    return df


def purify_dti_dataframe_parallel(df: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    """
    对一个包含DTI数据的DataFrame进行并行的深度清洗和标准化。

    Args:
        df (pd.DataFrame): 待处理的DataFrame, 必须包含'SMILES'和'Sequence'列。
        n_jobs (int): 使用的CPU核心数。-1 表示使用所有可用的核心。

    Returns:
        pd.DataFrame: 经过深度清洗和标准化的DataFrame。
    """
    n_jobs = config.runtime.cpus
    print("\n" + "-" * 80)
    print(" " * 20 + f"启动并行数据净化流程 (使用 {n_jobs} 个核心)")
    print("-" * 80)

    # 1. 将 DataFrame 分割成块
    #    选择合适的块数量，通常是CPU核心数的几倍，以确保负载均衡
    num_chunks = min(len(df) // 1000, n_jobs * 4) if len(df) > 1000 else n_jobs
    if num_chunks <= 0:
        num_chunks = 1  # 保证至少有一个chunk

    df_chunks = np.array_split(df, num_chunks)
    print(f"--> 数据已分割成 {len(df_chunks)} 个块进行并行处理。")

    # 2. 使用 Joblib 并行处理所有块
    print("--> 开始并行清洗...")

    # 使用with语句确保进程池被正确关闭
    with Parallel(n_jobs=n_jobs) as parallel:
        # delayed(_purify_chunk) 创建一个函数的“延迟”版本
        # tqdm在这里包裹整个并行任务列表，以显示总进度
        processed_chunks = parallel(
            delayed(_purify_chunk)(chunk)
            for chunk in tqdm(df_chunks, desc="处理数据块")
        )

    # 3. 合并处理后的结果
    print("--> 所有块处理完毕，正在合并结果...")
    purified_df = pd.concat(processed_chunks, ignore_index=True)

    print("\n" + "-" * 80)
    print(f"✅ 并行数据净化流程完成。最终保留 {len(purified_df)} 条记录。")
    print("-" * 80)

    return purified_df


def _calculate_descriptors_for_chunk(smiles_series: pd.Series) -> pd.DataFrame:
    """
    一个“工人”函数，为一小块(chunk)SMILES数据计算所有需要的分子描述符。
    包括：MW, LogP, HBD, HBA, QED, SA_Score, 以及是否为PAINS。
    """
    results = []
    for smiles in smiles_series:
        mol = Chem.MolFromSmiles(smiles)

        # 初始化一个包含默认“失败”值的字典
        descriptor_dict = {
            "MW": np.nan,
            "LogP": np.nan,
            "HBD": np.nan,  # Hydrogen Bond Donors
            "HBA": np.nan,  # Hydrogen Bond Acceptors
            "QED": np.nan,
            "SA_Score": np.nan,
            "is_pains": True,  # 默认视为PAINS，只有成功通过检查的才设为False
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
            except Exception as e:
                # 容忍单个分子的计算失败
                if len(smiles) < 100:  # 只打印较短的SMILES以避免刷屏
                    print(
                        f"    - WARNING: Descriptor calculation failed for SMILES '{smiles}'. Error: {e}"
                    )
                else:
                    print(
                        f"    - WARNING: Descriptor calculation failed for a long SMILES string. Error: {e}"
                    )

        results.append(descriptor_dict)

    return pd.DataFrame(results)


# --- 主过滤器函数 (并行版) ---


# TODO: 添加更多属性检查,到~1万左右
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

    descriptors_df = pd.concat(descriptor_dfs).reset_index(drop=True)
    df_with_props = df.reset_index(drop=True).join(descriptors_df)

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
