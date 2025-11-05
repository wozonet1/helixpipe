import unittest
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

# 导入我们需要测试的类
from nasnet.data_processing.services.graph_builder import HeteroGraphBuilder
from nasnet.data_processing.services.graph_director import GraphDirector

# --- 模拟(Mock)对象 ---


class MockGraphBuildContext:
    """
    一个模拟的GraphBuildContext，用于精确控制测试场景。
    """

    def __init__(self, num_mols_train, num_prots_train, num_mols_cold, num_prots_cold):
        self.num_local_mols = num_mols_train + num_mols_cold
        self.num_local_prots = num_prots_train + num_prots_cold

        self.mol_train_ids = set(range(num_mols_train))
        self.prot_train_ids = set(
            range(self.num_local_mols, self.num_local_mols + num_prots_train)
        )
        self.mol_cold_ids = set(range(num_mols_train, self.num_local_mols))
        self.prot_cold_ids = set(
            range(
                self.num_local_mols + num_prots_train,
                self.num_local_mols + self.num_local_prots,
            )
        )

        self.local_id_to_type_map = {}
        # 为了简化，我们假设所有分子都是'drug'类型
        for i in range(self.num_local_mols):
            self.local_id_to_type_map[i] = "drug"
        for i in range(self.num_local_mols, self.num_local_mols + self.num_local_prots):
            self.local_id_to_type_map[i] = "protein"

    def get_local_node_type(self, local_id: int) -> str:
        return self.local_id_to_type_map[local_id]

    def get_local_protein_id_offset(self) -> int:
        return self.num_local_mols


class TestHeteroGraphBuilder(unittest.TestCase):
    def setUp(self):
        """
        创建一个包含训练、测试和冷启动实体的模拟环境。
        """
        print("\n" + "=" * 80)

        # 场景: 2个训练分子, 1个冷启动分子, 2个训练蛋白, 1个冷启动蛋白
        self.context = MockGraphBuildContext(
            num_mols_train=2, num_prots_train=2, num_mols_cold=1, num_prots_cold=1
        )
        # 局部ID: mols=[0,1 (train), 2 (cold)], prots=[3,4 (train), 5 (cold)]

        # 模拟嵌入（在新的测试策略下，这些不再直接使用，但保留以备将来）
        self.mol_embeddings = torch.empty(3, 8)
        self.prot_embeddings = torch.empty(3, 8)

        # 模拟训练集交互对（使用局部ID）
        self.train_pairs = [(0, 3, "interacts_with")]  # mol 0 - prot 3

    def _get_base_config(self):
        """创建一个包含了所有必要依赖的基础配置。"""
        return OmegaConf.create(
            {
                "runtime": {"verbose": 1},
                "knowledge_graph": {"relation_types": {"default": "interacts_with"}},
                "data_params": {
                    "similarity_top_k": 10,
                    "similarity_thresholds": {"drug_drug": 0.8, "protein_protein": 0.8},
                },
                "relations": {
                    "flags": {
                        "interacts_with": True,
                        "drug_drug_similarity": True,
                        "protein_protein_similarity": True,
                    }
                },
                "training": {"coldstart": {"strictness": "informed"}},
                "data_structure": {
                    "schema": {
                        "internal": {
                            "graph_output": {
                                "source_node": "source",
                                "target_node": "target",
                                "edge_type": "edge_type",
                            }
                        }
                    }
                },
            }
        )

    @patch.object(HeteroGraphBuilder, "_add_similarity_edges_ann", autospec=True)
    def test_build_informed_mode(self, mock_add_similarity_edges):
        """
        测试点1: 在 'informed' 模式下，所有背景边（包括接触冷启动节点的）都被创建。
        """
        print("\n--- Running Test: Graph Build in 'informed' Mode ---")

        cfg = self._get_base_config()

        # 定义模拟方法的行为：直接向builder实例的_edges列表添加我们想要的边
        def add_fake_edges(self_builder, entity_type, **kwargs):
            if entity_type == "molecule":
                # 添加一条训练集内部的 drug-drug 边
                self_builder._edges.append([0, 1, "drug_drug_similarity"])
            elif entity_type == "protein":
                # 添加一条跨界的 protein-protein 边 (train <-> cold)
                self_builder._edges.append([3, 5, "protein_protein_similarity"])

        mock_add_similarity_edges.side_effect = add_fake_edges

        builder = HeteroGraphBuilder(
            config=cfg,
            context=self.context,
            molecule_embeddings=self.mol_embeddings,
            protein_embeddings=self.prot_embeddings,
            cold_start_entity_ids_local={2, 5},  # mol 2, prot 5
        )
        director = GraphDirector(cfg)

        # Director 会调用我们模拟的 _add_similarity_edges_ann
        director.construct(builder, self.train_pairs)

        # Director 在 informed 模式下不会调用过滤方法
        graph_df = builder.get_graph()

        # 预期边:
        # 1. (0, 3, 'interacts_with') - 来自 construct
        # 2. [0, 1, 'drug_drug_similarity'] - 来自 mock
        # 3. [3, 5, 'protein_protein_similarity'] - 来自 mock (泄露边)
        self.assertEqual(
            len(graph_df),
            3,
            "Should create all interaction and mocked similarity edges.",
        )

        edge_types = set(graph_df["edge_type"])
        self.assertIn(
            "protein_protein_similarity",
            edge_types,
            "Leaky P-P edge should exist in informed mode.",
        )

        print("  ✅ Test Passed: 'informed' mode correctly constructed the graph.")

    @patch.object(HeteroGraphBuilder, "_add_similarity_edges_ann", autospec=True)
    def test_build_strict_mode(self, mock_add_similarity_edges):
        """
        测试点2 (核心): 在 'strict' 模式下，接触冷启动节点的背景边被正确过滤。
        """
        print("\n--- Running Test: Graph Build in 'strict' Mode ---")

        cfg = self._get_base_config()
        OmegaConf.update(cfg, "training.coldstart.strictness", "strict")

        # 模拟方法的行为与上面完全相同
        def add_fake_edges(self_builder, entity_type, **kwargs):
            if entity_type == "molecule":
                self_builder._edges.append([0, 1, "drug_drug_similarity"])
            elif entity_type == "protein":
                self_builder._edges.append([3, 5, "protein_protein_similarity"])

        mock_add_similarity_edges.side_effect = add_fake_edges

        builder = HeteroGraphBuilder(
            config=cfg,
            context=self.context,
            molecule_embeddings=self.mol_embeddings,
            protein_embeddings=self.prot_embeddings,
            cold_start_entity_ids_local={2, 5},  # 冷启动实体是 mol 2 和 prot 5
        )
        director = GraphDirector(cfg)

        # Director 现在应该会调用 builder.filter_background_edges_for_strict_mode
        director.construct(builder, self.train_pairs)

        graph_df = builder.get_graph()

        # 预期边:
        # 1. (0, 3, 'interacts_with') - 保留
        # 2. [0, 1, 'drug_drug_similarity'] - 保留 (训练集内部)
        # 3. [3, 5, 'protein_protein_similarity'] - 被过滤 (因为接触了冷启动蛋白 5)
        self.assertEqual(
            len(graph_df),
            2,
            "Should filter background edges connected to cold-start nodes.",
        )

        edge_types = set(graph_df["edge_type"])
        self.assertIn("interacts_with", edge_types)
        self.assertIn("drug_drug_similarity", edge_types)
        self.assertNotIn(
            "protein_protein_similarity",
            edge_types,
            "Leaky P-P edge should be removed in strict mode.",
        )

        print("  ✅ Test Passed: 'strict' mode correctly filtered the graph.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
