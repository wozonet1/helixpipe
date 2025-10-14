import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm
from joblib import Parallel, delayed

from data_utils.canonicalizer import canonicalize_smiles

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


def purify_dti_dataframe_parallel(df: pd.DataFrame, n_jobs: int = -1) -> pd.DataFrame:
    """
    对一个包含DTI数据的DataFrame进行并行的深度清洗和标准化。

    Args:
        df (pd.DataFrame): 待处理的DataFrame, 必须包含'SMILES'和'Sequence'列。
        n_jobs (int): 使用的CPU核心数。-1 表示使用所有可用的核心。

    Returns:
        pd.DataFrame: 经过深度清洗和标准化的DataFrame。
    """
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
