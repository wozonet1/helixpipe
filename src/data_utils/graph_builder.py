# 文件: src/data_utils/graph_builder.py (全新)

import pandas as pd
import numpy as np
from collections import defaultdict
from project_types import AppConfig

# 导入我们需要的自定义模块
import research_template as rt
from .id_mapper import IDMapper


class GraphBuilder:
    """
    一个专门负责从关系数据构建图结构文件的“工匠”类。

    它的核心职责是，为实验的每一折(fold)，根据配置生成一个
    包含所有背景知识边（交互、相似性等）的图结构CSV文件。
    """

    def __init__(
        self,
        config: AppConfig,
        id_mapper: IDMapper,
        dl_sim_matrix: np.ndarray,
        prot_sim_matrix: np.ndarray,
    ):
        """
        初始化GraphBuilder。

        一次性接收所有不变量（在所有fold中都相同）作为输入。

        Args:
            config (DictConfig): 完整的Hydra配置对象。
            id_mapper (IDMapper): 已初始化的IDMapper实例。
            dl_sim_matrix (np.ndarray): 药物/配体的相似度矩阵。
            prot_sim_matrix (np.ndarray): 蛋白质的相似度矩阵。
        """
        self.config = config
        self.id_mapper = id_mapper
        self.dl_sim_matrix = dl_sim_matrix
        self.prot_sim_matrix = prot_sim_matrix
        self.verbose = config.runtime.get("verbose", 1)

        print("--- [GraphBuilder] Initialized. Ready to build graphs. ---")

    def build_for_fold(self, fold_idx: int, train_pairs: list):
        """
        为指定的fold构建并保存一个完整的训练图。

        Args:
            fold_idx (int): 当前的折数 (e.g., 1, 2, ...)。
            train_pairs (list): 属于这一折【训练集】的正样本交互对列表。
                                这些将被用作图中的“背景知识”边。
        """
        if self.verbose > 0:
            print(f"\n--- [GraphBuilder] Building graph for Fold {fold_idx}... ---")

        typed_edges_list = []
        graph_schema = self.config.data_structure.schema.internal.graph_output

        # 步骤 1: 添加交互边
        self._add_interaction_edges(typed_edges_list, train_pairs)

        # 步骤 2: 添加相似性边
        self._add_similarity_edges(typed_edges_list, self.prot_sim_matrix, "protein")
        self._add_similarity_edges(typed_edges_list, self.dl_sim_matrix, "molecule")

        # (未来扩展: 在这里可以添加 _add_ppi_edges(...) 等)

        # 步骤 3: 保存最终的图文件
        graph_output_path = rt.get_path(
            self.config,
            "data_structure.paths.processed.specific.graph_template",
            prefix=f"fold_{fold_idx}",
            suffix="train",  # 文件名后缀应为train
        )

        if self.verbose > 0:
            print(
                f"--> Saving final graph structure for Fold {fold_idx} to: {graph_output_path.name}"
            )

        typed_edges_df = pd.DataFrame(
            typed_edges_list,
            columns=[
                graph_schema.source_node,
                graph_schema.target_node,
                graph_schema.edge_type,
            ],
        )

        rt.ensure_path_exists(graph_output_path)
        typed_edges_df.to_csv(graph_output_path, index=False)

        if self.verbose > 0:
            print(
                f"--> Graph for Fold {fold_idx} saved with {len(typed_edges_df)} total edges."
            )

    def _add_interaction_edges(self, edges_list: list, train_pairs: list):
        """私有方法：根据配置添加DTI和LPI交互边。"""
        flags = self.config.relations.flags
        counts = defaultdict(int)

        for u, v in train_pairs:
            # 使用id_mapper来确定源节点的类型
            source_type = self.id_mapper.get_node_type(u)  # 'drug' or 'ligand'

            if source_type == "drug" and flags.get("dp_interaction", False):
                edges_list.append([u, v, "drug_protein_interaction"])
                counts["drug_protein_interaction"] += 1
            elif source_type == "ligand" and flags.get("lp_interaction", False):
                edges_list.append([u, v, "ligand_protein_interaction"])
                counts["ligand_protein_interaction"] += 1

        if self.verbose > 0:
            for edge_type, count in counts.items():
                print(f"    - Added {count} '{edge_type}' edges.")

    def _add_similarity_edges(
        self, edges_list: list, sim_matrix: np.ndarray, entity_type: str
    ):
        """私有方法：通用的、由IDMapper驱动的相似性边添加函数。"""
        if self.verbose > 0:
            print(
                f"    - Processing {entity_type} similarity matrix (shape: {sim_matrix.shape})..."
            )

        id_offset = 0 if entity_type == "molecule" else self.id_mapper.num_molecules

        rows, cols = np.where(np.triu(sim_matrix, k=1))
        edge_counts = defaultdict(int)

        # 为了加速，预先获取配置
        flags = self.config.relations.flags
        thresholds = self.config.data_params.similarity_thresholds

        for i, j in zip(rows, cols):
            similarity = sim_matrix[i, j]

            type1 = self.id_mapper.get_node_type(i + id_offset)
            type2 = self.id_mapper.get_node_type(j + id_offset)

            relation_prefix = "_".join(sorted((type1, type2)))
            relation_flag_key = f"{relation_prefix}_similarity"

            if flags.get(relation_flag_key, False):
                threshold = thresholds.get(relation_prefix, 1.1)

                if similarity > threshold:
                    source_id = i + id_offset
                    target_id = j + id_offset
                    edges_list.append([source_id, target_id, relation_flag_key])
                    edge_counts[relation_flag_key] += 1

        if self.verbose > 1:
            for edge_type, count in edge_counts.items():
                print(f"      - Added {count} '{edge_type}' edges.")
