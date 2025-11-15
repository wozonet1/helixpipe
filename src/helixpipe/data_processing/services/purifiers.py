# src/helixpipe/data_processing/services/purifiers.py

import logging
from typing import cast

import pandas as pd
from rdkit import Chem, RDLogger

logger = logging.getLogger(__name__)
# --- 全局设置 ---
RDLogger.logger().setLevel(RDLogger.CRITICAL)


def validate_smiles_structure(smiles_series: pd.Series) -> pd.Series:
    """
    【新】对一个SMILES Series进行结构和语法有效性验证，并进行标准化。

    Args:
        smiles_series (pd.Series): 包含SMILES字符串的Pandas Series。

    Returns:
        pd.Series: 一个新的Series，其索引与输入相同。
                   - 值为有效的、标准化的SMILES字符串。
                   - 对于无效的、空的或非字符串的输入，值为None。
    """
    if smiles_series.empty:
        return smiles_series

    # 1. 预处理：确保是字符串且非空
    #    我们不在这里丢弃行，而是将无效的标记为None，保持索引对齐
    cleaned_series = smiles_series.apply(
        lambda s: s.strip() if isinstance(s, str) else None
    )

    # 2. 验证SMILES化学结构有效性
    #    使用 apply + lambda，对于无效的SMILES，MolFromSmiles返回None
    mol_series = cast(pd.Series, cleaned_series.dropna().apply(Chem.MolFromSmiles))  # type: ignore
    # 3. 标准化 (Canonicalization)
    #    只对有效的分子对象进行标准化
    canonical_series = mol_series.dropna().apply(
        lambda mol: Chem.MolToSmiles(mol, canonical=True)
    )

    # 4. 将结果对齐回原始索引
    #    .reindex() 会自动用 NaN (我们稍后会处理) 填充那些在 canonical_series 中不存在的索引
    return cast(pd.Series, canonical_series.reindex(smiles_series.index))


def validate_protein_structure(
    sequence_series: pd.Series,
) -> pd.Series:
    """
    【新】对一个蛋白质序列Series进行字符集有效性验证。

    Args:
        sequence_series (pd.Series): 包含蛋白质序列的Pandas Series。

    Returns:
        pd.Series: 一个布尔掩码 (boolean mask)，其索引与输入相同。
                   值为 True 表示序列通过了字符集验证。
    """
    if sequence_series.empty:
        return pd.Series(dtype=bool)

    # 1. 定义合法的氨基酸字符集 (包括不常见的)
    VALID_SEQ_CHARS = "ACDEFGHIKLMNOPQRSTUVWXYZ"
    valid_chars_set = set(VALID_SEQ_CHARS)

    def is_valid_sequence(seq) -> None:
        if not isinstance(seq, str) or not seq:
            return False
        # 移除空格并转为大写，然后检查每个字符是否都在合法集合中
        return all(char in valid_chars_set for char in seq.strip().upper())

    # 2. 应用验证函数，直接返回布尔掩码
    return cast(pd.Series, sequence_series.apply(is_valid_sequence))
