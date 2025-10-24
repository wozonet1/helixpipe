# 文件: src/nasnet/data_processing/services/graph_builder.py (Builder模式重构版)

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import pandas as pd
import research_template as rt
from tqdm import tqdm

from nasnet.configs import AppConfig

from .id_mapper import IDMapper

# ==============================================================================
# 1. 定义 Builder 抽象基类 (接口)
# ==============================================================================


class GraphBuilder(ABC):
    """
    【Builder接口】
    定义了构建一个异构图所需的所有步骤的抽象方法。
    具体的实现由子类 (ConcreteBuilder) 完成。
    """

    @abstractmethod
    def add_interaction_edges(self, train_pairs: List[Tuple[int, int, str]]):
        """添加交互类型的边 (DTI, LPI, inhibits, etc.)。"""
        raise NotImplementedError

    @abstractmethod
    def add_molecule_similarity_edges(self):
        """添加所有分子间的相似性边 (D-D, L-L, D-L)。"""
        raise NotImplementedError

    @abstractmethod
    def add_protein_similarity_edges(self):
        """添加蛋白质间的相似性边 (P-P)。"""
        raise NotImplementedError

    @abstractmethod
    def get_graph(self) -> pd.DataFrame:
        """获取最终构建完成的图 DataFrame。"""
        raise NotImplementedError


# ==============================================================================
# 2. 提供 ConcreteBuilder 实现
# ==============================================================================


class HeteroGraphBuilder(GraphBuilder):
    """
    【ConcreteBuilder 实现】
    一个具体的图生成器，负责实现所有构建步骤，并维护正在构建的图的状态。
    """

    def __init__(
        self,
        config: AppConfig,
        id_mapper: IDMapper,
        dl_sim_matrix: np.ndarray,
        prot_sim_matrix: np.ndarray,
    ):
        """
        初始化具体的生成器。

        Args:
            config: 完整的Hydra配置。
            id_mapper: 已最终化的IDMapper实例。
            dl_sim_matrix: 分子相似性矩阵。
            prot_sim_matrix: 蛋白质相似性矩阵。
        """
        self.config = config
        self.id_mapper = id_mapper
        self.dl_sim_matrix = dl_sim_matrix
        self.prot_sim_matrix = prot_sim_matrix
        self.verbose = config.runtime.verbose

        # 内部状态：用于存储正在构建的边列表
        self._edges: List[List] = []
        self._graph_schema = self.config.data_structure.schema.internal.graph_output

        if self.verbose > 0:
            print("--- [HeteroGraphBuilder] Initialized. Ready to build. ---")

    def add_interaction_edges(self, train_pairs: List[Tuple[int, int, str]]):
        """
        【实现】根据 final_edge_type 和 relations.flags 添加交互边。
        """
        flags = self.config.relations.flags
        counts = defaultdict(int)

        # train_pairs已经是 (u, v, final_edge_type) 的列表
        for u, v, final_edge_type in train_pairs:
            # Processor已经完成了所有翻译工作，这里直接检查开关
            if flags.get(final_edge_type, False):
                self._edges.append([u, v, final_edge_type])
                counts[final_edge_type] += 1

        if self.verbose > 0 and counts:
            print("    - Added Interaction Edges:")
            for edge_type, count in counts.items():
                print(f"      - {count} '{edge_type}' edges.")

    def add_molecule_similarity_edges(self):
        """【实现】添加所有分子间的相似性边。"""
        self._add_similarity_edges_generic(self.dl_sim_matrix, "molecule")

    def add_protein_similarity_edges(self):
        """【实现】添加蛋白质间的相似性边。"""
        self._add_similarity_edges_generic(self.prot_sim_matrix, "protein")

    def get_graph(self) -> pd.DataFrame:
        """【实现】返回构建完成的图 DataFrame。"""
        if self.verbose > 0:
            print(
                f"--- [HeteroGraphBuilder] Finalizing graph with {len(self._edges)} total edges. ---"
            )

        return pd.DataFrame(
            self._edges,
            columns=[
                self._graph_schema.source_node,
                self._graph_schema.target_node,
                self._graph_schema.edge_type,
            ],
        )

    # --- 私有辅助方法 (将之前的相似性边逻辑封装在这里) ---
    def _add_similarity_edges_generic(self, sim_matrix: np.ndarray, entity_type: str):
        """
        一个通用的、由IDMapper驱动的相似性边添加函数。

        Args:
            sim_matrix: 要处理的相似性矩阵。
            entity_type: 实体类型, 'molecule' 或 'protein'。
        """
        id_offset = 0 if entity_type == "molecule" else self.id_mapper.num_molecules
        rows, cols = np.where(np.triu(sim_matrix, k=1))

        flags = self.config.relations.flags
        thresholds = self.config.data_params.similarity_thresholds
        edge_counts = defaultdict(int)

        disable_tqdm = self.verbose == 0
        iterator = tqdm(
            zip(rows, cols),
            total=len(rows),
            desc=f"    - Scanning '{entity_type}' similarity pairs",
            disable=disable_tqdm,
        )

        for i, j in iterator:
            similarity = sim_matrix[i, j]

            global_id_i = i + id_offset
            global_id_j = j + id_offset

            type1 = self.id_mapper.get_node_type(global_id_i)
            type2 = self.id_mapper.get_node_type(global_id_j)

            source_type, relation_prefix, target_type = (
                rt.graph_utils.get_canonical_relation(type1, type2)
            )

            # 最终的相似性边类型字符串
            final_edge_type = f"{relation_prefix}_similarity"

            # 检查总开关
            if flags.get(final_edge_type, False):
                threshold = thresholds.get(
                    relation_prefix, 1.1
                )  # 默认阈值1.1确保不会意外通过

                if similarity > threshold:
                    # 确定边的方向以符合规范
                    if type1 == source_type:
                        source_id, target_id = global_id_i, global_id_j
                    else:
                        source_id, target_id = global_id_j, global_id_i

                    self._edges.append([source_id, target_id, final_edge_type])
                    edge_counts[final_edge_type] += 1

        if self.verbose > 0 and edge_counts:
            print("    - Added Similarity Edges:")
            for edge_type, count in edge_counts.items():
                print(f"      - {count} '{edge_type}' edges.")
