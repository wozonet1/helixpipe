# 文件: src/nasnet/data_processing/services/graph_builder.py (最终版)

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import pandas as pd
import research_template as rt
from tqdm import tqdm

from nasnet.configs import AppConfig
from nasnet.utils import get_path

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
# 2. ConcreteBuilder 实现 - 最终版
# ==============================================================================


class HeteroGraphBuilder(GraphBuilder):
    """
    【ConcreteBuilder 实现 - V3 解耦版】
    从分块缓存文件和交互对数据中，具体地构建异构图。
    它不再依赖于内存中的 embeddings 张量。
    """

    def __init__(
        self,
        config: AppConfig,
        id_mapper: IDMapper,
    ):
        """
        初始化具体的生成器。
        Args:
            config: 完整的Hydra配置。
            id_mapper: 已最终化的IDMapper实例。
        """
        self.config = config
        self.id_mapper = id_mapper
        self.verbose = config.runtime.verbose

        self._edges: List[List] = []
        self._graph_schema = self.config.data_structure.schema.internal.graph_output

        if self.verbose > 0:
            print(
                "--- [HeteroGraphBuilder] Initialized. Ready to build from file chunks. ---"
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

    def add_molecule_similarity_edges(self):
        """【实现】从分块缓存中加载数据并添加分子相似性边。"""
        self._add_similarity_edges_from_chunks(
            entity_type="molecule",
            chunk_dir_key="processed.common.similarity_matrices.molecule_chunks_dir",
            batch_size=self.config.data_params.get("similarity_batch_size", 1024),
        )

    def add_protein_similarity_edges(self):
        """【实现】从分块缓存中加载数据并添加蛋白质相似性边。"""
        self._add_similarity_edges_from_chunks(
            entity_type="protein",
            chunk_dir_key="processed.common.similarity_matrices.protein_chunks_dir",
            batch_size=self.config.data_params.get("similarity_batch_size", 1024),
        )

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

    def _add_similarity_edges_from_chunks(
        self,
        entity_type: str,
        chunk_dir_key: str,
        batch_size: int,
    ):
        """
        一个通用的、从磁盘分块缓存加载数据并筛选边的函数。
        """
        # 【核心修改】从 id_mapper 获取实体数量
        if entity_type == "molecule":
            num_embeddings = self.id_mapper.num_molecules
            id_offset = 0
        elif entity_type == "protein":
            num_embeddings = self.id_mapper.num_proteins
            id_offset = self.id_mapper.num_molecules
        else:
            return

        if num_embeddings == 0:
            return

        chunk_dir = get_path(self.config, chunk_dir_key)
        if not chunk_dir.exists():
            print(
                f"    - WARNING: Similarity chunk directory not found for '{entity_type}'. Skipping."
            )
            return

        chunk_template_str = self.config.data_structure.filenames.processed.common.similarity_matrices.chunk_template

        def chunk_path_factory(chunk_idx):
            return chunk_dir / chunk_template_str.format(chunk_idx=chunk_idx)

        flags = self.config.relations.flags
        thresholds = self.config.data_params.similarity_thresholds
        edge_counts = defaultdict(int)

        num_chunks = (num_embeddings + batch_size - 1) // batch_size

        disable_tqdm = self.verbose == 0
        for i in tqdm(
            range(num_chunks),
            desc=f"    - Processing '{entity_type}' sim chunks",
            disable=disable_tqdm,
        ):
            chunk_path = chunk_path_factory(i)
            if not chunk_path.exists():
                continue

            sim_sub_matrix = np.load(chunk_path)

            # 性能优化: 只处理那些可能通过阈值的候选边
            min_threshold = min(v for k, v in thresholds.items() if "similarity" in k)
            candidate_rows, candidate_cols = np.where(sim_sub_matrix > min_threshold)

            for r, c in zip(candidate_rows, candidate_cols):
                global_id_i = (i * batch_size) + r + id_offset
                global_id_j = c + id_offset

                if global_id_i >= global_id_j:
                    continue

                similarity = sim_sub_matrix[r, c]

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
            print("    - Added Similarity Edges:")
            for edge_type, count in edge_counts.items():
                print(f"      - {count} '{edge_type}' edges.")
