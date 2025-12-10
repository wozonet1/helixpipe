# 文件: src/helixpipe/data_processing/services/graph_builder.py (最终正确版 - 实时计算)

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import DefaultDict, Literal, Union, overload

import faiss
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from helixpipe.typing import AppConfig, LogicID, LogicInteractionTriple

from .graph_context import GraphBuildContext
from .relation_utils import get_similarity_relation_type

SimilarityResult = tuple[int, int, float, str]
logger = logging.getLogger(__name__)
# ==============================================================================
# 1. Builder 抽象基类 (接口) - 保持不变
# ==============================================================================


class GraphBuilder(ABC):
    """【Builder接口】定义了构建一个异构图所需的所有步骤的抽象方法。"""

    @abstractmethod
    def add_interaction_edges(self, train_pairs: list[LogicInteractionTriple]) -> None:
        raise NotImplementedError

    @abstractmethod
    def add_molecule_similarity_edges(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def add_protein_similarity_edges(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_graph(self) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def filter_background_edges_for_strict_mode(self) -> None:
        """
        If in 'strict' mode, filters out background edges that touch cold-start nodes.
        This is an in-place operation on the builder's internal state.
        """
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
        context: GraphBuildContext,
        molecule_embeddings: torch.Tensor,
        protein_embeddings: torch.Tensor,
        cold_start_entity_ids_local: Union[set[LogicID], None] = None,
    ):
        """
        初始化具体的生成器。
        Args:
            config: 完整的Hydra配置。
            context: 包含所有局部ID映射和信息的图构建上下文对象。
            molecule_embeddings: 【必需】用于实时计算的分子特征嵌入。
            protein_embeddings: 【必需】用于实时计算的蛋白质特征嵌入。
        """
        self.config = config
        self.kg_config = config.knowledge_graph
        self.context = context
        self.molecule_embeddings = molecule_embeddings
        self.protein_embeddings = protein_embeddings

        self._edges: list[list] = []
        self._graph_schema = self.config.data_structure.schema.internal.graph_output
        self._cold_start_entity_ids = cold_start_entity_ids_local

        logger.info(
            "--- [HeteroGraphBuilder] Initialized for on-the-fly similarity computation. ---"
        )

    def add_interaction_edges(self, train_pairs: list[LogicInteractionTriple]) -> None:
        """【实现】根据 final_edge_type 和 relations.flags 添加交互边。"""
        flags = self.config.relations.flags
        counts: DefaultDict[str, int] = defaultdict(int)

        for u, v, final_edge_type in train_pairs:
            if flags.get(final_edge_type, False):
                self._edges.append([u, v, final_edge_type])
                counts[final_edge_type] += 1

        if counts:
            logger.info("    - Added Interaction Edges:")
            for edge_type, count in counts.items():
                logger.info(f"      - {count} '{edge_type}' edges.")

    # [MODIFIED] 更新公共方法以读取新的 'similarity_top_k' 配置
    def add_molecule_similarity_edges(self) -> None:
        """【实现】调用ANN方法计算并添加分子相似性边。"""
        self._add_similarity_edges_ann(
            entity_type="molecule",
            embeddings=self.molecule_embeddings,
            k=self.config.data_params.similarity_top_k,
        )

    # [MODIFIED] 更新公共方法
    def add_protein_similarity_edges(self) -> None:
        """【实现】调用ANN方法计算并添加蛋白质相似性边。"""
        self._add_similarity_edges_ann(
            entity_type="protein",
            embeddings=self.protein_embeddings,
            k=self.config.data_params.similarity_top_k,
        )

    @overload
    def _add_similarity_edges_ann(
        self,
        *,  # 使用 * 强制后续参数为关键字参数，好习惯
        entity_type: str,
        embeddings: torch.Tensor,
        k: int,
        analysis_mode: Literal[True],
    ) -> list[SimilarityResult]: ...

    @overload
    def _add_similarity_edges_ann(
        self,
        *,
        entity_type: str,
        embeddings: torch.Tensor,
        k: int,
        analysis_mode: Literal[False] = False,
    ) -> None: ...

    def _add_similarity_edges_ann(
        self,
        entity_type: str,
        embeddings: torch.Tensor,
        k: int,
        analysis_mode: bool = False,  # [NEW] 新增 analysis_mode 参数
    ) -> Union[list[SimilarityResult], None]:
        """
        [V3] 使用 Faiss (ANN) 计算 Top-K 相似邻居。
        - 在正常模式下，筛选边并添加到 self._edges。
        - 在分析模式下，返回所有候选相似度对的列表。
        """
        # ... (方法上半部分的 faiss 索引构建和搜索逻辑保持不变) ...
        if not analysis_mode:
            logger.info(
                f"\n    -> Calculating '{entity_type}' similarities using ANN (Faiss) for Top-{k} neighbors..."
            )
        candidate_pairs_for_analysis: Union[list, None] = None
        num_embeddings = embeddings.shape[0]
        if num_embeddings <= k:
            return None if not analysis_mode else []
        embeddings_np = embeddings.cpu().detach().numpy().astype(np.float32)
        dim = embeddings_np.shape[1]
        index = faiss.IndexFlatL2(dim)
        faiss.normalize_L2(embeddings_np)
        index.add(embeddings_np)  # type: ignore
        distances, indices = index.search(embeddings_np, k + 1)  # type: ignore
        if distances is None or indices is None:
            return [] if analysis_mode else None
        id_offset = (
            0
            if entity_type == "molecule"
            else self.context.get_local_protein_id_offset()
        )

        edge_counts: DefaultDict[str, int] = defaultdict(int)

        logger.debug(
            f"\n    --- [DEBUG] Inside _add_similarity_edges_ann for '{entity_type}' ---"
        )
        logger.debug(
            f"    - Thresholds dictionary being used: {self.config.data_params.similarity_thresholds}"
        )
        for i in range(num_embeddings):
            for neighbor_idx in range(1, k + 1):
                j = indices[i, neighbor_idx]

                local_id_i = i + id_offset
                local_id_j = j + id_offset

                if local_id_i >= local_id_j:
                    continue

                similarity = 1 - 0.5 * (distances[i, neighbor_idx] ** 2)

                type1 = self.context.get_local_node_type(local_id_i)
                type2 = self.context.get_local_node_type(local_id_j)

                source_type, target_type, final_edge_type = (
                    get_similarity_relation_type(type1, type2, self.kg_config)
                )
                relation_prefix = f"{source_type}_{target_type}"
                source_id, target_id = (
                    (local_id_i, local_id_j)
                    if type1 == source_type
                    else (local_id_j, local_id_i)
                )

                # [MODIFIED] 核心逻辑分支
                if analysis_mode:
                    candidate_pairs_for_analysis = []
                    # 在分析模式下，不进行阈值过滤，直接收集所有计算出的相似度
                    candidate_pairs_for_analysis.append(
                        (source_id, target_id, similarity, final_edge_type)
                    )
                else:
                    # 在正常模式下，执行阈值过滤和添加边的操作
                    flags = self.config.relations.flags
                    if flags.get(final_edge_type, False):
                        thresholds = self.config.data_params.similarity_thresholds
                        threshold = getattr(
                            thresholds,
                            relation_prefix,
                            1.1,  # 提供一个默认值，以防 relation_prefix 对应的字段不存在
                        )
                        if similarity > threshold:
                            self._edges.append([source_id, target_id, final_edge_type])
                            edge_counts[final_edge_type] += 1
                        # [NEW DEBUG PRINT] 打印每一个候选边的决策过程
                        # 只打印相似度较高的，避免刷屏
                        if similarity > 0.5:
                            logger.debug(
                                f"      - Candidate Edge: ({local_id_i}, {local_id_j}), "
                                f"Type: {final_edge_type}, "
                                f"Similarity: {similarity:.4f}, "
                                f"Threshold: {threshold}, "
                                f"Passes?: {similarity > threshold}"
                            )

        if not analysis_mode:
            if edge_counts:
                logger.info("    - Added ANN-based Similarity Edges:")
                for edge_type, count in edge_counts.items():
                    logger.info(f"      - {count} '{edge_type}' edges.")
            return None  # 正常模式下无返回值
        else:
            return candidate_pairs_for_analysis  # 分析模式下返回收集的列表

    def get_graph(self) -> pd.DataFrame:
        """【实现】返回构建完成的图 DataFrame。"""

        logger.info(
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

    # [NEW] 新增的、职责专一的过滤方法
    # TODO: 精细化调控
    def filter_background_edges_for_strict_mode(self) -> None:
        """
        如果处于 'strict' 模式，则从已添加的边 (_edges) 中，
        移除所有接触到冷启动实体的背景知识边。
        这是一个 in-place 操作。
        """

        logger.info(
            "    - [Builder] Applying 'strict' cold-start filter to background edges..."
        )

        # a. 识别出所有背景知识边类型
        rel_types_dict = OmegaConf.to_container(
            self.config.knowledge_graph.relation_types, resolve=True
        )
        # 2. 获取 values 并转为 set
        # 注意：to_container 返回的是 dict | list | None，这里肯定是 dict
        if isinstance(rel_types_dict, dict):
            interaction_rel_types = set(rel_types_dict.values())
        else:
            # 防御性编程，理论上不应发生
            interaction_rel_types = set()

        # b. 遍历当前的 _edges 列表，只保留那些应该留下的
        edges_to_keep = []
        num_removed = 0
        for edge in self._edges:
            source, target, edge_type = edge

            is_background = edge_type not in interaction_rel_types
            is_touching_cold = (source in self._cold_start_entity_ids) or (
                target in self._cold_start_entity_ids
            )

            # 保留条件：(不是背景边) OR (是背景边 但不接触冷启动实体)
            if not is_background or not is_touching_cold:
                edges_to_keep.append(edge)
            else:
                num_removed += 1

        if num_removed > 0:
            logger.info(
                f"      - Removed {num_removed} background edges connected to cold-start entities."
            )

        # c. 用过滤后的列表替换掉旧的列表
        self._edges = edges_to_keep

    # [NEW] 新增的公共分析方法
    def analyze_similarities(self) -> pd.DataFrame:
        """
        [NEW] 执行一个“仅分析”的相似度计算，返回所有候选相似度对的DataFrame。
        """
        logger.info(
            "\n--- [HeteroGraphBuilder] Running in ANALYSIS-ONLY mode for similarities..."
        )

        # 调用底层方法，并传入 analysis_mode=True
        mol_sim_pairs = self._add_similarity_edges_ann(
            entity_type="molecule",
            embeddings=self.molecule_embeddings,
            k=self.config.data_params.similarity_top_k,
            analysis_mode=True,
        )

        prot_sim_pairs = self._add_similarity_edges_ann(
            entity_type="protein",
            embeddings=self.protein_embeddings,
            k=self.config.data_params.similarity_top_k,
            analysis_mode=True,
        )

        all_sim_pairs = mol_sim_pairs + prot_sim_pairs

        if not all_sim_pairs:
            return pd.DataFrame(columns=["source", "target", "similarity", "type"])

        return pd.DataFrame(
            all_sim_pairs, columns=["source", "target", "similarity", "type"]
        )
