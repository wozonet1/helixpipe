import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm
from joblib import Parallel, delayed
from rdkit.Chem import Descriptors
from data_utils.canonicalizer import canonicalize_smiles
from omegaconf import DictConfig

# 让tqdm能和pandas的apply方法优雅地协作
# 在并行化场景下，我们将主要用tqdm来包裹joblib的调用
tqdm.pandas()


def _purify_chunk(df_chunk: pd.DataFrame) -> pd.DataFrame:
    """
    这是一个“工作者”函数,用于处理单个DataFrame块。
    所有核心的清洗逻辑都在这里。
    """
    # 抑制此工作进程中的RDKit日志
    logger = RDLogger.logger()
    logger.setLevel(RDLogger.CRITICAL)

    is_valid_smiles = df_chunk["SMILES"].apply(
        lambda s: Chem.MolFromSmiles(s) is not None if isinstance(s, str) else False
    )
    df_chunk = df_chunk[is_valid_smiles].copy()
    if df_chunk.empty:
        return pd.DataFrame()

    # --- 2. 标准化 (Canonicalize) SMILES ---
    df_chunk["SMILES"] = df_chunk["SMILES"].apply(canonicalize_smiles)
    df_chunk.dropna(subset=["SMILES"], inplace=True)
    if df_chunk.empty:
        return pd.DataFrame()

    # 【第一步：标准化】将所有序列转换为大写。这是关键！
    # 这样可以确保后续验证的正确性，并使输出数据保持一致。
    df_chunk["Sequence"] = df_chunk["Sequence"].str.upper()

    # 【第二步：验证和过滤】现在在大写序列上查找非法字符。
    valid_amino_acids = "ACDEFGHIKLMNPQRSTVWUY"
    invalid_char_pattern = f"[^{valid_amino_acids}]"

    # 找出不包含任何非法字符的序列
    is_sequence_valid = ~df_chunk["Sequence"].str.contains(
        invalid_char_pattern,
        regex=True,
        na=True,  # na=True 将NaN视为无效
    )

    df_chunk = df_chunk[is_sequence_valid].copy()

    return df_chunk


def purify_dti_dataframe_parallel(df: pd.DataFrame, config) -> pd.DataFrame:
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
    一个“工人”函数，为一小块(chunk)SMILES数据计算分子描述符。
    它被设计为可以在独立的进程中运行。
    """
    results = []
    for smiles in smiles_series:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            results.append({"MW": mw, "LogP": logp})
        else:
            # 对于无效的SMILES，返回NaN，以便后续可以轻松过滤掉
            results.append({"MW": np.nan, "LogP": np.nan})
    return pd.DataFrame(results)


# --- 主过滤器函数 (并行版) ---


def filter_molecules_by_properties(
    df: pd.DataFrame, config: DictConfig
) -> pd.DataFrame:
    """
    【V2 并行版】根据配置中定义的分子理化性质规则，对DataFrame进行过滤。
    """
    # 1. 检查总开关 (逻辑不变)
    try:
        filter_cfg = config.data_params.filtering
        if not filter_cfg.get("enabled", False):
            print("--> Molecular property filtering is disabled. Skipping.")
            return df
    except Exception:
        print("--> No 'filtering' config found. Skipping molecular property filtering.")
        return df

    print(
        "\n--- [Molecule Filter] Applying molecular property filters (Parallel Mode)... ---"
    )

    initial_count = len(df)
    schema = config.data_structure.schema.internal.authoritative_dti
    smiles_col = schema.molecule_sequence

    # 2. 【核心变化】并行计算所有必需的分子描述符
    print(
        f"    - Calculating molecular descriptors for {len(df)} molecules in parallel..."
    )

    # a. 确定并行核心数
    n_jobs = config.runtime.get("cpus", -1)

    # b. 将SMILES列分割成多个块(chunks)
    #    选择合适的块数量，通常是CPU核心数的几倍，以实现负载均衡
    num_chunks = min(len(df) // 1000, n_jobs * 4) if len(df) > 1000 else n_jobs
    if num_chunks <= 0:
        num_chunks = 1

    smiles_chunks = np.array_split(df[smiles_col], num_chunks)

    # c. 使用 joblib 并行执行“工人”函数
    with Parallel(n_jobs=n_jobs) as parallel:
        descriptor_dfs = parallel(
            delayed(_calculate_descriptors_for_chunk)(chunk)
            for chunk in tqdm(smiles_chunks, desc="      Descriptor Chunks")
        )

    # d. 合并结果，并将其与原始DataFrame对齐
    descriptors_df = pd.concat(descriptor_dfs).reset_index(drop=True)
    df = df.reset_index(drop=True)  # 确保原始df的索引也是连续的
    df[["MW", "LogP"]] = descriptors_df

    # 移除计算失败的分子
    df.dropna(subset=["MW", "LogP"], inplace=True)

    # 3. 逐条应用过滤规则 (逻辑不变，但在大数据上会快很多)
    print("    - Applying filters...")

    # a. 分子量
    if "molecular_weight" in filter_cfg:
        mw_cfg = filter_cfg.molecular_weight
        df = df[
            df["MW"].between(
                mw_cfg.get("min", -float("inf")), mw_cfg.get("max", float("inf"))
            )
        ]

    # b. LogP
    if "logp" in filter_cfg:
        logp_cfg = filter_cfg.logp
        df = df[
            df["LogP"].between(
                logp_cfg.get("min", -float("inf")), logp_cfg.get("max", float("inf"))
            )
        ]

    final_count = len(df)
    print(f"    - Filtering complete. {final_count} molecules remain.")

    # 4. 清理并返回 (逻辑不变)
    df.drop(columns=["MW", "LogP"], inplace=True)

    num_removed = initial_count - final_count
    print(f"--- [Molecule Filter] Complete. Removed {num_removed} molecules. ---")

    return df
