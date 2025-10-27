# src/nasnet/data_processing/services/purifiers.py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger
from tqdm import tqdm

from nasnet.typing import AppConfig

from .canonicalizer import canonicalize_smiles


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
    对一个包含DTI数据的DataFrame进行并行的深度清洗和标准化。一般在调用了structure_provider之后使用。

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
