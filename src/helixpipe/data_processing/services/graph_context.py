# 文件: src/helixpipe/data_processing/services/graph_context.py
# 图构建上下文工具函数（原 GraphBuildContext 类拆解为纯函数）

import logging
from typing import TYPE_CHECKING

import pandas as pd
import torch

from helixpipe.typing import LogicID, LogicInteractionTriple

if TYPE_CHECKING:
    from .id_mapper import IDMapper

logger = logging.getLogger(__name__)


def build_local_id_mapping(
    relevant_mol_ids: set[LogicID],
    relevant_prot_ids: set[LogicID],
) -> tuple[dict[int, int], list[int], int]:
    """
    构建全局逻辑ID → 局部0-based ID 的映射。

    排序规则：分子在前，蛋白质在后，各自按 ID 升序。

    Returns:
        (global_to_local, local_to_global, num_local_mols)
    """
    g2l: dict[int, int] = {}
    l2g: list[int] = []
    current = 0

    for gid in sorted(relevant_mol_ids):
        g2l[gid] = current
        l2g.append(gid)
        current += 1
    num_mols = current

    for gid in sorted(relevant_prot_ids):
        g2l[gid] = current
        l2g.append(gid)
        current += 1

    logger.debug(
        f"  [build_local_id_mapping] Molecules: {num_mols}, "
        f"Proteins: {current - num_mols}, Total: {current}"
    )
    return g2l, l2g, num_mols


def slice_embeddings(
    local_to_global: list[int],
    num_mols: int,
    global_mol_embeddings: torch.Tensor,
    global_prot_embeddings: torch.Tensor,
    num_total_molecules: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    根据局部ID映射，从全局 embedding 矩阵中切片出当前 fold 的局部 embedding。
    """
    if num_mols > 0:
        mol_indices = torch.tensor(local_to_global[:num_mols], dtype=torch.long)
        local_mol = global_mol_embeddings[mol_indices]
    else:
        local_mol = torch.empty(
            0,
            global_mol_embeddings.shape[1] if global_mol_embeddings.numel() > 0 else 0,
        )

    num_prots = len(local_to_global) - num_mols
    if num_prots > 0:
        prot_indices = torch.tensor(
            [gid - num_total_molecules for gid in local_to_global[num_mols:]],
            dtype=torch.long,
        )
        local_prot = global_prot_embeddings[prot_indices]
    else:
        local_prot = torch.empty(
            0,
            global_prot_embeddings.shape[1]
            if global_prot_embeddings.numel() > 0
            else 0,
        )

    logger.debug(
        f"  [slice_embeddings] Molecules: {local_mol.shape}, Proteins: {local_prot.shape}"
    )
    return local_mol, local_prot


def build_local_id_to_type(
    local_to_global: list[int],
    id_mapper: "IDMapper",
) -> dict[int, str]:
    """根据局部→全局映射，查询 IDMapper 构建局部 ID→类型的字典。"""
    result: dict[int, str] = {}
    for local_id, global_id in enumerate(local_to_global):
        meta = id_mapper.get_meta_by_logic_id(global_id)
        if meta is None:
            raise RuntimeError(f"Failed to get entity meta for logic ID: {global_id}")
        result[local_id] = meta["type"]
    return result


def convert_pairs_to_local(
    global_pairs: list[LogicInteractionTriple],
    g2l: dict[int, int],
) -> list[LogicInteractionTriple]:
    """将使用全局ID的交互对列表转换为使用局部ID。只保留两端都在映射中的对。"""
    return [(g2l[u], g2l[v], r) for u, v, r in global_pairs if u in g2l and v in g2l]


def convert_ids_to_local(
    global_ids: set[LogicID],
    g2l: dict[int, int],
) -> set[LogicID]:
    """将全局逻辑ID集合转换为局部ID集合。"""
    return {g2l[gid] for gid in global_ids if gid in g2l}


def convert_dataframe_to_global(
    local_df: pd.DataFrame,
    source_col: str,
    target_col: str,
    local_to_global: list[int],
) -> pd.DataFrame:
    """将使用局部ID的 DataFrame 转换回使用全局ID。"""
    if local_df.empty:
        return local_df
    global_df = local_df.copy()
    reverse_map = pd.Series(local_to_global)
    global_df[source_col] = global_df[source_col].map(reverse_map)
    global_df[target_col] = global_df[target_col].map(reverse_map)
    return global_df
