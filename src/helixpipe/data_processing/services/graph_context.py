# 文件: src/helixpipe/data_processing/services/graph_context.py (最终版)

import logging
from typing import TYPE_CHECKING, Dict, List, Set, Tuple

import pandas as pd
import torch

from helixpipe.configs import AppConfig

if TYPE_CHECKING:
    from .id_mapper import IDMapper

# 在模块顶部获取 logger 实例
logger = logging.getLogger(__name__)


class GraphBuildContext:
    """
    一个封装了图构建所需“局部上下文”的服务类。
    """

    def __init__(
        self,
        fold_idx: int,
        global_id_mapper: "IDMapper",
        global_mol_embeddings: torch.Tensor,
        global_prot_embeddings: torch.Tensor,
        relevant_mol_ids: Set[int],
        relevant_prot_ids: Set[int],
        config: AppConfig,
    ):
        logger.debug("\n--- [GraphBuildContext] Initializing local context... ---")
        logger.debug(
            f"    - Received {len(relevant_mol_ids)} relevant molecule IDs: {sorted(list(relevant_mol_ids))}"
        )
        logger.debug(
            f"    - Received {len(relevant_prot_ids)} relevant protein IDs: {sorted(list(relevant_prot_ids))}"
        )

        # --- 1. 构建ID映射 ---
        logger.debug("  --- Step 1: Building ID Mappings ---")

        sorted_relevant_mols = sorted(list(relevant_mol_ids))
        sorted_relevant_prots = sorted(list(relevant_prot_ids))

        self.global_to_local_id_map: Dict[int, int] = {}
        self.local_to_global_id_list: List[int] = []

        current_local_id = 0
        logger.debug("    - Mapping molecules:")
        for global_id in sorted_relevant_mols:
            self.global_to_local_id_map[global_id] = current_local_id
            self.local_to_global_id_list.append(global_id)
            logger.debug(
                f"      - Global ID {global_id} -> Local ID {current_local_id}"
            )
            current_local_id += 1

        self.num_local_mols = len(sorted_relevant_mols)

        logger.debug("    - Mapping proteins:")
        for global_id in sorted_relevant_prots:
            self.global_to_local_id_map[global_id] = current_local_id
            self.local_to_global_id_list.append(global_id)
            logger.debug(
                f"      - Global ID {global_id} -> Local ID {current_local_id}"
            )
            current_local_id += 1

        self.num_local_prots = len(sorted_relevant_prots)

        # --- 2. 筛选局部特征嵌入 ---
        if self.num_local_mols > 0:
            mol_indices = torch.tensor(sorted_relevant_mols, dtype=torch.long)
            self.local_mol_embeddings = global_mol_embeddings[mol_indices]
        else:
            self.local_mol_embeddings = torch.empty(
                0,
                global_mol_embeddings.shape[1]
                if global_mol_embeddings.numel() > 0
                else 0,
            )

        if self.num_local_prots > 0:
            prot_indices_0_based = torch.tensor(
                [gid - global_id_mapper.num_molecules for gid in sorted_relevant_prots],
                dtype=torch.long,
            )
            self.local_prot_embeddings = global_prot_embeddings[prot_indices_0_based]
        else:
            self.local_prot_embeddings = torch.empty(
                0,
                global_prot_embeddings.shape[1]
                if global_prot_embeddings.numel() > 0
                else 0,
            )

        logger.debug(
            f"  --- Step 2: Sliced local embeddings. Molecules shape: {self.local_mol_embeddings.shape}, Proteins shape: {self.local_prot_embeddings.shape}"
        )

        # --- 3. 构建局部ID到类型的映射 ---
        logger.debug("  --- Step 3: Building Local ID to Type Map ---")
        self.local_id_to_type_map: Dict[int, str] = {}
        for local_id, global_id in enumerate(self.local_to_global_id_list):
            node_type = global_id_mapper.get_node_type(global_id)
            self.local_id_to_type_map[local_id] = node_type
            logger.debug(
                f"      - Local ID {local_id} (Global {global_id}) -> Type '{node_type}'"
            )

        logger.info(
            f"[GraphBuildContext] Initialization complete for fold {fold_idx}. Local molecules: {self.num_local_mols}, Local proteins: {self.num_local_prots}."
        )

    def convert_pairs_to_local(
        self, global_pairs: List[Tuple[int, int, str]]
    ) -> List[Tuple[int, int, str]]:
        """将使用全局ID的交互对列表，转换为使用局部ID。"""
        local_pairs = []
        for u_global, v_global, rel_type in global_pairs:
            u_local = self.global_to_local_id_map.get(u_global)
            v_local = self.global_to_local_id_map.get(v_global)
            if u_local is not None and v_local is not None:
                local_pairs.append((u_local, v_local, rel_type))
        return local_pairs

    def convert_dataframe_to_global(
        self, local_df: pd.DataFrame, source_col: str, target_col: str
    ) -> pd.DataFrame:
        """将一个使用局部ID的DataFrame转换回使用全局ID。"""
        if local_df.empty:
            return local_df
        global_df = local_df.copy()
        reverse_map = pd.Series(self.local_to_global_id_list)
        global_df[source_col] = global_df[source_col].map(reverse_map)
        global_df[target_col] = global_df[target_col].map(reverse_map)
        return global_df

    def convert_ids_to_local(self, global_ids: Set[int]) -> Set[int]:
        """将一个全局逻辑ID的集合，转换为局部ID的集合。"""
        return {
            self.global_to_local_id_map[gid]
            for gid in global_ids
            if gid in self.global_to_local_id_map
        }

    def get_local_node_type(self, local_id: int) -> str:
        """根据局部ID，返回其节点类型。"""
        return self.local_id_to_type_map[local_id]

    def get_local_protein_id_offset(self) -> int:
        """返回在局部ID空间中，蛋白质ID的起始编号。"""
        return self.num_local_mols
