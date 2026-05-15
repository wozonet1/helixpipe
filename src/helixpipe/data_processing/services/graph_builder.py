# 文件: src/helixpipe/data_processing/services/graph_builder.py

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

from .relation_utils import get_similarity_relation_type

SimilarityResult = tuple[int, int, float, str]
logger = logging.getLogger(__name__)


# ==============================================================================
# 1. Builder 抽象基类 (接口)
# ==============================================================================


class GraphBuilder(ABC):
    """【Builder接口】定义了构建一个异构图所需的所有步骤的抽象方法。"""

    @abstractmethod
    def build(self, train_pairs: list[LogicInteractionTriple]) -> None:
        """执行完整的图构建流程。"""
        raise NotImplementedError

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
        raise NotImplementedError


# ==============================================================================
# 2. ConcreteBuilder 实现
# ==============================================================================


class HeteroGraphBuilder(GraphBuilder):
    """
    【ConcreteBuilder - 实时计算版】
    在构建图的过程中，实时地、分块地从特征嵌入计算相似度。
    不依赖任何磁盘上的相似度矩阵缓存。
    """

    def __init__(
        self,
        config: AppConfig,
        molecule_embeddings: torch.Tensor,
        protein_embeddings: torch.Tensor,
        local_id_to_type: dict[int, str],
        protein_id_offset: int,
        cold_start_entity_ids_local: Union[set[LogicID], None] = None,
    ):
        self.config = config
        self.kg_config = config.knowledge_graph
        self.molecule_embeddings = molecule_embeddings
        self.protein_embeddings = protein_embeddings
        self.local_id_to_type = local_id_to_type
        self.protein_id_offset = protein_id_offset

        self._edges: list[list] = []
        self._graph_schema = self.config.data_structure.schema.internal.graph_output
        self._cold_start_entity_ids = cold_start_entity_ids_local

        logger.info(
            "--- [HeteroGraphBuilder] Initialized for on-the-fly similarity computation. ---"
        )

    def build(self, train_pairs: list[LogicInteractionTriple]) -> None:
        """根据配置的 flags，按预设顺序执行完整的图构建流程。"""
        if self.config.runtime.verbose > 0:
            logger.info(
                "\n--- [HeteroGraphBuilder] Starting graph construction process... ---"
            )

        # 1. 交互边
        if train_pairs:
            if self.config.runtime.verbose > 0:
                logger.info("  -> Adding interaction edges...")
            self.add_interaction_edges(train_pairs)

        # 2. 分子相似性边
        flags = self.config.relations.flags
        if (
            flags.get("drug_drug_similarity", False)
            or flags.get("ligand_ligand_similarity", False)
            or flags.get("drug_ligand_similarity", False)
        ):
            if self.config.runtime.verbose > 0:
                logger.info("  -> Adding molecule similarity edges...")
            self.add_molecule_similarity_edges()

        # 3. 蛋白质相似性边
        if flags.get("protein_protein_similarity", False):
            if self.config.runtime.verbose > 0:
                logger.info("  -> Adding protein similarity edges...")
            self.add_protein_similarity_edges()

        # 4. strict 冷启动过滤
        if self.config.training.coldstart.strictness == "strict":
            if self.config.runtime.verbose > 0:
                logger.info("  -> Applying strict cold-start filter...")
            self.filter_background_edges_for_strict_mode()

        if self.config.runtime.verbose > 0:
            logger.info(
                "--- [HeteroGraphBuilder] Graph construction process finished. ---"
            )

    def add_interaction_edges(self, train_pairs: list[LogicInteractionTriple]) -> None:
        """根据 final_edge_type 和 relations.flags 添加交互边。"""
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

    def add_molecule_similarity_edges(self) -> None:
        """调用ANN方法计算并添加分子相似性边。"""
        self._add_similarity_edges_ann(
            entity_type="molecule",
            embeddings=self.molecule_embeddings,
            k=self.config.data_params.similarity_top_k,
        )

    def add_protein_similarity_edges(self) -> None:
        """调用ANN方法计算并添加蛋白质相似性边。"""
        self._add_similarity_edges_ann(
            entity_type="protein",
            embeddings=self.protein_embeddings,
            k=self.config.data_params.similarity_top_k,
        )

    @overload
    def _add_similarity_edges_ann(
        self,
        *,
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
        analysis_mode: bool = False,
    ) -> Union[list[SimilarityResult], None]:
        """
        使用 Faiss (ANN) 计算 Top-K 相似邻居。
        - 正常模式：筛选边并添加到 self._edges。
        - 分析模式：返回所有候选相似度对的列表。
        """
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
        id_offset = 0 if entity_type == "molecule" else self.protein_id_offset

        edge_counts: DefaultDict[str, int] = defaultdict(int)

        if analysis_mode:
            candidate_pairs_for_analysis = []

        for i in range(num_embeddings):
            for neighbor_idx in range(1, k + 1):
                j = indices[i, neighbor_idx]

                local_id_i = i + id_offset
                local_id_j = j + id_offset

                if local_id_i >= local_id_j:
                    continue

                similarity = 1 - 0.5 * (distances[i, neighbor_idx] ** 2)

                type1 = self.local_id_to_type[local_id_i]
                type2 = self.local_id_to_type[local_id_j]

                source_type, target_type, final_edge_type = (
                    get_similarity_relation_type(type1, type2, self.kg_config)
                )
                relation_prefix = f"{source_type}_{target_type}"
                source_id, target_id = (
                    (local_id_i, local_id_j)
                    if type1 == source_type
                    else (local_id_j, local_id_i)
                )

                if analysis_mode:
                    candidate_pairs_for_analysis.append(
                        (source_id, target_id, similarity, final_edge_type)
                    )
                else:
                    flags = self.config.relations.flags
                    if flags.get(final_edge_type, False):
                        thresholds = self.config.data_params.similarity_thresholds
                        threshold = getattr(
                            thresholds,
                            relation_prefix,
                            1.1,
                        )
                        if similarity > threshold:
                            self._edges.append([source_id, target_id, final_edge_type])
                            edge_counts[final_edge_type] += 1
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
            return None
        else:
            return candidate_pairs_for_analysis

    def get_graph(self) -> pd.DataFrame:
        """返回构建完成的图 DataFrame。"""
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

    def filter_background_edges_for_strict_mode(self) -> None:
        """
        如果处于 'strict' 模式，从已添加的边中移除接触到冷启动实体的背景知识边。
        """
        logger.info(
            "    - [Builder] Applying 'strict' cold-start filter to background edges..."
        )

        rel_types_dict = OmegaConf.to_container(
            self.config.knowledge_graph.relation_types, resolve=True
        )
        if isinstance(rel_types_dict, dict):
            interaction_rel_types = set(rel_types_dict.values())
        else:
            interaction_rel_types = set()

        edges_to_keep = []
        num_removed = 0
        for edge in self._edges:
            source, target, edge_type = edge
            is_background = edge_type not in interaction_rel_types
            is_touching_cold = (source in self._cold_start_entity_ids) or (
                target in self._cold_start_entity_ids
            )
            if not is_background or not is_touching_cold:
                edges_to_keep.append(edge)
            else:
                num_removed += 1

        if num_removed > 0:
            logger.info(
                f"      - Removed {num_removed} background edges connected to cold-start entities."
            )
        self._edges = edges_to_keep

    def analyze_similarities(self) -> pd.DataFrame:
        """执行一个"仅分析"的相似度计算，返回所有候选相似度对的 DataFrame。"""
        logger.info(
            "\n--- [HeteroGraphBuilder] Running in ANALYSIS-ONLY mode for similarities..."
        )

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
