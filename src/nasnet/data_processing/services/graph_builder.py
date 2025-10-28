# 文件: src/nasnet/data_processing/services/graph_builder.py (最终正确版 - 实时计算)

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Tuple

import faiss
import numpy as np
import pandas as pd
import research_template as rt
import torch

from nasnet.configs import AppConfig

from .id_mapper import IDMapper

# ==============================================================================
# 1. Builder 抽象基类 (接口) - 保持不变
# ==============================================================================


class GraphBuilder(ABC):
    """【Builder接口】定义了构建一个异构图所需的所有步骤的抽象方法。"""

    @abstractmethod
    def add_interaction_edges(self, train_pairs: List[Tuple[int, int, str]]):
        raise NotImplementedError

    @abstractmethod
    def add_molecule_similarity_edges(self):
        raise NotImplementedError

    @abstractmethod
    def add_protein_similarity_edges(self):
        raise NotImplementedError

    @abstractmethod
    def get_graph(self) -> pd.DataFrame:
        raise NotImplementedError


# ==============================================================================
# 2. ConcreteBuilder 实现 - 最终正确版
# ==============================================================================


class HeteroGraphBuilder(GraphBuilder):
    """
    【ConcreteBuilder - 实时计算版】
    在构建图的过程中，实时地、分块地从特征嵌入计算相似度。
    它不依赖任何磁盘上的相似度矩阵缓存。
    """

    def __init__(
        self,
        config: AppConfig,
        id_mapper: IDMapper,
        molecule_embeddings: torch.Tensor,
        protein_embeddings: torch.Tensor,
    ):
        """
        初始化具体的生成器。
        Args:
            config: 完整的Hydra配置。
            id_mapper: 已最终化的IDMapper实例。
            molecule_embeddings: 【必需】用于实时计算的分子特征嵌入。
            protein_embeddings: 【必需】用于实时计算的蛋白质特征嵌入。
        """
        self.config = config
        self.id_mapper = id_mapper
        self.molecule_embeddings = molecule_embeddings
        self.protein_embeddings = protein_embeddings
        self.verbose = config.runtime.verbose

        self._edges: List[List] = []
        self._graph_schema = self.config.data_structure.schema.internal.graph_output

        if self.verbose > 0:
            print(
                "--- [HeteroGraphBuilder] Initialized for on-the-fly similarity computation. ---"
            )

    def add_interaction_edges(self, train_pairs: List[Tuple[int, int, str]]):
        """【实现】根据 final_edge_type 和 relations.flags 添加交互边。"""
        flags = self.config.relations.flags
        counts = defaultdict(int)

        for u, v, final_edge_type in train_pairs:
            if flags.get(final_edge_type, False):
                self._edges.append([u, v, final_edge_type])
                counts[final_edge_type] += 1

        if self.verbose > 0 and counts:
            print("    - Added Interaction Edges:")
            for edge_type, count in counts.items():
                print(f"      - {count} '{edge_type}' edges.")

    # [MODIFIED] 更新公共方法以读取新的 'similarity_top_k' 配置
    def add_molecule_similarity_edges(self):
        """【实现】调用ANN方法计算并添加分子相似性边。"""
        self._add_similarity_edges_ann(
            entity_type="molecule",
            embeddings=self.molecule_embeddings,
            k=self.config.data_params.similarity_top_k,
        )

    # [MODIFIED] 更新公共方法
    def add_protein_similarity_edges(self):
        """【实现】调用ANN方法计算并添加蛋白质相似性边。"""
        self._add_similarity_edges_ann(
            entity_type="protein",
            embeddings=self.protein_embeddings,
            k=self.config.data_params.similarity_top_k,
        )

    # [REPLACED] 旧的 _add_similarity_edges_on_the_fly 方法被完全替换为下面的新方法
    def _add_similarity_edges_ann(
        self,
        entity_type: str,
        embeddings: torch.Tensor,
        k: int,
    ):
        """
        [NEW] 使用 Faiss (ANN) 高效计算 Top-K 相似邻居并筛选边。
        """
        print(
            f"\n    -> Calculating '{entity_type}' similarities using ANN (Faiss) for Top-{k} neighbors..."
        )

        num_embeddings = embeddings.shape[0]
        if num_embeddings < k:
            print(
                f"    - WARNING: Number of embeddings ({num_embeddings}) is less than k ({k}). Skipping ANN."
            )
            return

        embeddings_np = embeddings.cpu().detach().numpy().astype(np.float32)
        dim = embeddings_np.shape[1]

        # 1. 构建 Faiss 索引
        index = faiss.IndexFlatL2(dim)
        # 核心步骤: L2归一化，使得L2距离等价于余弦相似度
        faiss.normalize_L2(embeddings_np)
        index.add(embeddings_np)

        # 2. 搜索 Top-K 邻居 (请求k+1个，因为第一个总是实体自身)
        distances, indices = index.search(embeddings_np, k + 1)

        # 3. 添加边
        id_offset = 0 if entity_type == "molecule" else self.id_mapper.num_molecules
        flags = self.config.relations.flags
        thresholds = self.config.data_params.similarity_thresholds
        edge_counts = defaultdict(int)

        # 这个循环的复杂度是 N * K，远低于 N * N
        for i in range(num_embeddings):
            # 从索引1开始，跳过自身
            for neighbor_idx in range(1, k + 1):
                j = indices[i, neighbor_idx]

                global_id_i = i + id_offset
                global_id_j = j + id_offset

                # 避免重复边 (i,j) 和 (j,i)
                if global_id_i >= global_id_j:
                    continue

                # 从L2距离转换回余弦相似度
                similarity = 1 - 0.5 * (distances[i, neighbor_idx] ** 2)

                # 后续的类型检查和阈值过滤逻辑与您原来的一样
                type1 = self.id_mapper.get_node_type(global_id_i)
                type2 = self.id_mapper.get_node_type(global_id_j)
                source_type, relation_prefix, target_type = (
                    rt.graph_utils.get_canonical_relation(type1, type2)
                )
                final_edge_type = f"{relation_prefix}_similarity"

                if flags.get(final_edge_type, False):
                    threshold = thresholds.get(relation_prefix, 1.1)
                    if similarity > threshold:
                        source_id, target_id = (
                            (global_id_i, global_id_j)
                            if type1 == source_type
                            else (global_id_j, global_id_i)
                        )
                        self._edges.append([source_id, target_id, final_edge_type])
                        edge_counts[final_edge_type] += 1

        if self.verbose > 0 and edge_counts:
            print("    - Added ANN-based Similarity Edges:")
            for edge_type, count in edge_counts.items():
                print(f"      - {count} '{edge_type}' edges.")

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
