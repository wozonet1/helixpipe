import re

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from tqdm import tqdm

from data_utils.canonicalizer import canonicalize_smiles
from project_types import AppConfig

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
    # --- 1. 【新增】PubChem_CID 净化 ---
    # a. 过滤掉None/NaN值
    df.dropna(subset=["PubChem_CID"], inplace=True)
    if df.empty:
        return df

    # b. 尝试将CID转换为整数，无法转换的行将被设为NaN
    #    errors='coerce' 是这里的关键，它会把所有“坏”数据变成NaN
    df["PubChem_CID"] = pd.to_numeric(df["PubChem_CID"], errors="coerce")

    # c. 再次过滤掉转换失败的行
    df.dropna(subset=["PubChem_CID"], inplace=True)
    if df.empty:
        return df

    # d. 确保CID是整数且大于0
    df = df[df["PubChem_CID"] > 0]
    df["PubChem_CID"] = df["PubChem_CID"].astype(int)  # 确保最终是int类型
    if df.empty:
        return df

    # --- 2. SMILES 净化 (我们的逻辑更严格，保持不变) ---
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

    # --- 3. 序列 净化 (借鉴DeepPurpose) ---
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

    # --- 4. 【新增】UniProt ID 净化 ---
    # a. 过滤掉None/NaN或非字符串
    df.dropna(subset=["UniProt_ID"], inplace=True)
    df = df[df["UniProt_ID"].apply(isinstance, args=(str,))]
    if df.empty:
        return df

    # b. 使用严格的正则表达式进行格式验证
    #    这个表达式匹配P12345, Q9Y261, A0A024R1R8等标准格式
    uniprot_pattern = re.compile(
        r"([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})"
    )
    valid_uniprot_mask = df["UniProt_ID"].str.match(uniprot_pattern)
    df = df[valid_uniprot_mask]
    if df.empty:
        return df

    # --- 5. SMILES 标准化 (现在是最后一步) ---
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


# TODO: 添加更多属性检查,到~1万左右
def filter_molecules_by_properties(df: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
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
