# 使用前向引用来避免循环导入，这是一个常见的Python类型提示技巧
from typing import TYPE_CHECKING, Dict, List, Set, Tuple

import pandas as pd
import torch

from helixpipe.configs import AppConfig

if TYPE_CHECKING:
    from .id_mapper import IDMapper


class GraphBuildContext:
    """
    一个封装了图构建所需“局部上下文”的服务类。

    它的核心职责是管理一个从“全局ID空间”到一个临时的、与特定任务
    相关的“局部ID空间”的映射和数据转换。

    一旦被实例化，其内部状态就是不可变的。
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
        """
        在构造时，立即完成所有映射和数据筛选。

        Args:
            global_id_mapper: 已经最终化的全局IDMapper实例。
            global_mol_embeddings: 完整的、使用全局分子ID索引的分子嵌入。
            global_prot_embeddings: 完整的、使用全局蛋白质ID索引的蛋白质嵌入。
            relevant_mol_ids: 本次构建任务相关的全局分子逻辑ID集合。
            relevant_prot_ids: 本次构建任务相关的全局蛋白质逻辑ID集合。
            config: 全局配置对象，用于获取verbose等级等。
        """
        if config.runtime.verbose > 0:
            print(
                "\n--- [GraphBuildContext] Initializing local context for graph building..."
            )
            print(
                f"    - Received {len(relevant_mol_ids)} relevant molecule IDs and {len(relevant_prot_ids)} relevant protein IDs."
            )

        # --- 1. 构建ID映射 ---

        # a. 排序以保证映射的确定性
        sorted_relevant_mols = sorted(list(relevant_mol_ids))
        sorted_relevant_prots = sorted(list(relevant_prot_ids))

        # b. 创建双向映射
        self.global_to_local_id_map: Dict[int, int] = {}
        self.local_to_global_id_list: List[int] = []

        # 分子局部ID从0开始
        current_local_id = 0
        for global_id in sorted_relevant_mols:
            self.global_to_local_id_map[global_id] = current_local_id
            self.local_to_global_id_list.append(global_id)
            current_local_id += 1

        self.num_local_mols = len(sorted_relevant_mols)

        # 蛋白质局部ID接在分子之后
        for global_id in sorted_relevant_prots:
            self.global_to_local_id_map[global_id] = current_local_id
            self.local_to_global_id_list.append(global_id)
            current_local_id += 1

        self.num_local_prots = len(sorted_relevant_prots)

        # --- 2. 筛选局部特征嵌入 ---
        # a. 分子嵌入
        # relevant_mol_ids 是全局逻辑ID，可以直接用作索引
        if self.num_local_mols > 0:
            mol_indices = torch.tensor(sorted_relevant_mols, dtype=torch.long)
            self.local_mol_embeddings = global_mol_embeddings[mol_indices]
        else:
            # [NEW] 健壮性处理：如果没有分子，创建一个空的张量
            self.local_mol_embeddings = torch.empty(0, global_mol_embeddings.shape[1])

        # b. 蛋白质嵌入
        # global_prot_embeddings 的索引是从0开始的，但蛋白质的全局ID是从num_molecules开始的。
        # 我们需要先将全局蛋白质ID转换为相对于protein_embeddings张量的0-based索引。
        if self.num_local_prots > 0:
            # [MODIFIED] 蛋白质索引的计算逻辑保持不变，但更加关键
            # global_prot_embeddings 的索引是从0开始的，但蛋白质的全局ID是从全局分子数量开始的。
            # 我们需要先将全局蛋白质ID，转换为相对于protein_embeddings张量的0-based索引。
            prot_indices_0_based = torch.tensor(
                [gid - global_id_mapper.num_molecules for gid in sorted_relevant_prots],
                dtype=torch.long,
            )
            self.local_prot_embeddings = global_prot_embeddings[prot_indices_0_based]
        else:
            # [NEW] 健壮性处理：如果没有蛋白质，创建一个空的张量
            self.local_prot_embeddings = torch.empty(0, global_prot_embeddings.shape[1])

        # --- 3. 构建局部ID到类型的映射 ---
        self.local_id_to_type_map: Dict[int, str] = {}
        for local_id, global_id in enumerate(self.local_to_global_id_list):
            # 从全局IDMapper获取权威的节点类型
            self.local_id_to_type_map[local_id] = global_id_mapper.get_node_type(
                global_id
            )

        if config.runtime.verbose > 0:
            print("    - Local context created successfully.")
            print(
                f"    - Local molecules: {self.num_local_mols}, Local proteins: {self.num_local_prots}"
            )

    # --- 公共转换方法 ---

    def convert_pairs_to_local(
        self, global_pairs: List[Tuple[int, int, str]]
    ) -> List[Tuple[int, int, str]]:
        """将使用全局ID的交互对列表，转换为使用局部ID。"""
        local_pairs = []
        for u_global, v_global, rel_type in global_pairs:
            u_local = self.global_to_local_id_map.get(u_global)
            v_local = self.global_to_local_id_map.get(v_global)

            # 只有当交互对的双方都在我们的相关实体集合中时，才保留它
            if u_local is not None and v_local is not None:
                local_pairs.append((u_local, v_local, rel_type))
        return local_pairs

    def convert_dataframe_to_global(
        self, local_df: pd.DataFrame, source_col: str, target_col: str
    ) -> pd.DataFrame:
        """
        将一个使用局部ID的DataFrame（如图文件、标签文件）转换回使用全局ID。
        这是一个通用的转换器。
        """
        if local_df.empty:
            return local_df

        global_df = local_df.copy()

        # 使用 .map() 进行高效的批量转换
        # Series.map() 比 apply(lambda...) 快得多
        reverse_map = pd.Series(self.local_to_global_id_list)

        global_df[source_col] = global_df[source_col].map(reverse_map)
        global_df[target_col] = global_df[target_col].map(reverse_map)

        return global_df

    def get_local_node_type(self, local_id: int) -> str:
        """根据局部ID，返回其节点类型 ('drug', 'ligand', 'protein')。"""
        return self.local_id_to_type_map[local_id]

    def get_local_protein_id_offset(self) -> int:
        """返回在局部ID空间中，蛋白质ID的起始编号。"""
        return self.num_local_mols
