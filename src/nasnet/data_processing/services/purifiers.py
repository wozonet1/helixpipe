# src/nasnet/data_processing/services/purifiers.py

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger
from tqdm import tqdm

from nasnet.typing import AppConfig

from .canonicalizer import canonicalize_smiles


def _purify_molecules_chunk(
    df_chunk: pd.DataFrame, smiles_col: str, verbose: int = 0
) -> pd.DataFrame:
    """【新】一个只净化分子DataFrame块的辅助函数。"""
    if df_chunk.empty:
        return df_chunk

    logger = RDLogger.logger()
    logger.setLevel(RDLogger.CRITICAL)

    df = df_chunk.copy()

    # --- SMILES 净化流程 ---
    # a. 过滤空值和非字符串
    df.dropna(subset=[smiles_col], inplace=True)
    df = df[df[smiles_col].apply(lambda s: isinstance(s, str) and s.strip() != "")]
    if df.empty:
        return df

    # b. 验证SMILES格式
    df = df[df[smiles_col].apply(lambda s: Chem.MolFromSmiles(s) is not None)]
    if df.empty:
        return df

    # c. 标准化
    df[smiles_col] = df[smiles_col].apply(canonicalize_smiles)
    df.dropna(subset=[smiles_col], inplace=True)

    return df


# TODO: 性能瓶颈
def _purify_proteins_chunk(
    df_chunk: pd.DataFrame, seq_col: str, verbose: int = 0
) -> pd.DataFrame:
    """【V2 - 带Debug日志版】一个只净化蛋白质DataFrame块的辅助函数。"""
    if df_chunk.empty:
        return df_chunk

    df = df_chunk.copy()
    initial_count = len(df)

    df.dropna(subset=[seq_col], inplace=True)
    df = df[df[seq_col].apply(isinstance, args=(str,))]

    if verbose > 1 and len(df) < initial_count:
        print(
            f"    - DEBUG (Prot): Dropped {initial_count - len(df)} rows due to NaN/non-string Sequence."
        )

    if df.empty:
        return df

    # 净化流程2: 验证氨基酸字符集
    # 【核心修改】放宽验证，允许 B, Z, X
    VALID_SEQ_CHARS = "ACDEFGHIKLMNPQRSTVWYUBZX"  # <-- 增加了 B, Z, X
    valid_chars_set = set(VALID_SEQ_CHARS)

    def find_invalid_chars(seq: str) -> str:
        seq_clean = "".join(seq.split()).upper()
        if not seq_clean:
            return "EMPTY"

        invalid_chars = {char for char in seq_clean if char not in valid_chars_set}
        return "".join(sorted(list(invalid_chars))) if invalid_chars else ""

    # 应用函数，找到所有无效字符
    df["invalid_chars"] = df[seq_col].apply(find_invalid_chars)

    # 筛选出有效的行
    valid_mask = df["invalid_chars"] == ""

    # 【核心修改】如果需要Debug，打印被丢弃的行
    if verbose > 1 and not valid_mask.all():
        dropped_df = df[~valid_mask]
        print(
            f"    - DEBUG (Prot): Dropping {len(dropped_df)} rows due to invalid sequence characters."
        )
        # 统计最常见的无效字符组合
        reason_counts = dropped_df["invalid_chars"].value_counts().head(5)
        print("      - Top reasons (invalid chars):")
        for reason, count in reason_counts.items():
            print(f"        - Chars: '{reason}' -> Count: {count}")

    # 返回净化后的DataFrame，并丢弃辅助列
    return df[valid_mask].drop(columns=["invalid_chars"])


def purify_entities_dataframe_parallel(
    df: pd.DataFrame, config: AppConfig
) -> pd.DataFrame:
    """
    【最终版】对一个包含“实体清单”的DataFrame进行并行的深度清洗。
    它通过拆分-分别净化-合并的策略，正确处理分子行和蛋白质行分离的情况。
    """
    if df.empty:
        return df

    n_jobs = config.runtime.cpus
    verbose_level = config.runtime.verbose
    schema = config.data_structure.schema.internal.authoritative_dti
    mol_id_col, smiles_col = schema.molecule_id, schema.molecule_sequence
    prot_id_col, seq_col = schema.protein_id, schema.protein_sequence

    print("\n--- [Purifier] Starting PARALLEL ENTITY purification...")
    initial_count = len(df)

    # 1. 将 DataFrame 拆分为分子和蛋白质两部分
    molecules_df = df[df[mol_id_col].notna()].copy()
    proteins_df = df[df[prot_id_col].notna()].copy()

    if verbose_level > 0:
        print(
            f"  -> Found {len(molecules_df)} molecule candidates and {len(proteins_df)} protein candidates."
        )

    # 2. 分别对两部分进行并行净化
    purified_molecules_list = []
    if not molecules_df.empty:
        num_chunks = min(len(molecules_df), max(1, n_jobs * 4))
        mol_chunks = np.array_split(molecules_df, num_chunks)
        with Parallel(n_jobs=n_jobs) as parallel:
            purified_molecules_list = parallel(
                delayed(_purify_molecules_chunk)(
                    chunk, smiles_col, verbose=verbose_level
                )
                for chunk in tqdm(
                    mol_chunks,
                    desc="  - Purifying Molecules",
                    disable=verbose_level == 0,
                )
            )

    purified_proteins_list = []
    if not proteins_df.empty:
        num_chunks = min(len(proteins_df), max(1, n_jobs * 4))
        prot_chunks = np.array_split(proteins_df, num_chunks)
        with Parallel(n_jobs=n_jobs) as parallel:
            purified_proteins_list = parallel(
                delayed(_purify_proteins_chunk)(chunk, seq_col, verbose=verbose_level)
                for chunk in tqdm(
                    prot_chunks,
                    desc="  - Purifying Proteins",
                    disable=verbose_level == 0,
                )
            )

    # 3. 合并净化后的结果
    #    使用列表推导式过滤掉并行任务可能返回的空DataFrame
    purified_molecules_df = (
        pd.concat([df for df in purified_molecules_list if not df.empty])
        if purified_molecules_list
        else pd.DataFrame()
    )
    purified_proteins_df = (
        pd.concat([df for df in purified_proteins_list if not df.empty])
        if purified_proteins_list
        else pd.DataFrame()
    )

    final_df = pd.concat(
        [purified_molecules_df, purified_proteins_df], ignore_index=True
    )

    final_count = len(final_df)
    print(
        f"--- [Purifier] ENTITY purification complete. Retained {final_count} of {initial_count} total entities. ---"
    )

    return final_df.reset_index(drop=True)


def purify_dataframe_parallel(df: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    """
    【V3 - 统一接口版】
    对一个DataFrame进行并行的、深度的结构清洗（SMILES和Sequence）。

    该函数是智能的：
    - 它会自动检测DataFrame中是否存在 SMILES 和/或 Sequence 列。
    - 只对存在的列进行净化操作。
    - 它不关心输入的DataFrame是“关系清单”还是“实体清单”。
    """
    if df.empty:
        return df

    n_jobs = config.runtime.cpus
    schema = config.data_structure.schema.internal.authoritative_dti
    smiles_col = schema.molecule_sequence
    seq_col = schema.protein_sequence

    print("\n--- [Purifier] Starting PARALLEL structure purification...")

    purified_df = df.copy()
    initial_count = len(purified_df)

    # --- 1. 净化 SMILES (如果存在) ---
    if smiles_col in purified_df.columns:
        print("  -> Purifying SMILES column...")
        mol_chunks = np.array_split(purified_df, n_jobs)
        with Parallel(n_jobs=n_jobs) as parallel:
            processed_chunks = parallel(
                delayed(_purify_molecules_chunk)(chunk, smiles_col)
                for chunk in tqdm(mol_chunks, desc="    - Molecule Chunks")
            )
        # 重新合并，只保留非空的块
        purified_df = (
            pd.concat([c for c in processed_chunks if not c.empty])
            if processed_chunks
            else pd.DataFrame()
        )
        print(f"    - Retained {len(purified_df)} rows after SMILES purification.")

    if purified_df.empty:
        print("--- [Purifier] Purification complete. No valid data remains. ---")
        return purified_df

    # --- 2. 净化 Sequence (如果存在) ---
    if seq_col in purified_df.columns:
        print("  -> Purifying Sequence column...")
        prot_chunks = np.array_split(purified_df, n_jobs)
        with Parallel(n_jobs=n_jobs) as parallel:
            processed_chunks = parallel(
                delayed(_purify_proteins_chunk)(
                    chunk, seq_col, verbose=config.runtime.verbose
                )
                for chunk in tqdm(prot_chunks, desc="    - Protein Chunks")
            )
        purified_df = (
            pd.concat([c for c in processed_chunks if not c.empty])
            if processed_chunks
            else pd.DataFrame()
        )
        print(f"    - Retained {len(purified_df)} rows after Sequence purification.")

    final_count = len(purified_df)
    print(
        f"--- [Purifier] Purification complete. Retained {final_count} of {initial_count} total rows. ---"
    )

    return purified_df.reset_index(drop=True)
