import unittest
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

# 导入我们需要测试的类
from helixpipe.data_processing.services.graph_builder import HeteroGraphBuilder

# --- 模拟(Mock)对象 ---


class TestHeteroGraphBuilder(unittest.TestCase):
    def setUp(self):
        """
        创建一个包含训练、测试和冷启动实体的模拟环境。
        """
        print("\n" + "=" * 80)

        # 场景: 2个训练分子, 1个冷启动分子, 2个训练蛋白, 1个冷启动蛋白
        # 局部ID: mols=[0,1 (train), 2 (cold)], prots=[3,4 (train), 5 (cold)]
        self.local_id_to_type = {
            0: "drug",
            1: "drug",
            2: "drug",  # 分子
            3: "protein",
            4: "protein",
            5: "protein",  # 蛋白质
        }
        self.protein_id_offset = 3  # 分子数量

        # 模拟嵌入
        self.mol_embeddings = torch.empty(3, 8)
        self.prot_embeddings = torch.empty(3, 8)

        # 模拟训练集交互对（使用局部ID，五元组格式）
        self.train_pairs = [
            (0, 3, "interacts_with", "bindingdb", 0.85)
        ]  # mol 0 - prot 3

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
                            "canonical_interaction": {
                                "source_id": "source_id",
                                "source_type": "source_type",
                                "target_id": "target_id",
                                "target_type": "target_type",
                                "relation_type": "relation_type",
                                "source_dataset": "source_dataset",
                                "raw_score": "raw_score",
                            },
                            "graph_output": {
                                "source_node": "source",
                                "target_node": "target",
                                "edge_type": "edge_type",
                                "source_dataset": "source_dataset",
                                "score": "score",
                            },
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
                self_builder._edges.append(
                    [0, 1, "drug_drug_similarity", "computed", 0.92]
                )
            elif entity_type == "protein":
                # 添加一条跨界的 protein-protein 边 (train <-> cold)
                self_builder._edges.append(
                    [3, 5, "protein_protein_similarity", "computed", 0.88]
                )

        mock_add_similarity_edges.side_effect = add_fake_edges

        builder = HeteroGraphBuilder(
            config=cfg,
            molecule_embeddings=self.mol_embeddings,
            protein_embeddings=self.prot_embeddings,
            local_id_to_type=self.local_id_to_type,
            protein_id_offset=self.protein_id_offset,
            cold_start_entity_ids_local={2, 5},  # mol 2, prot 5
        )

        # build 会调用我们模拟的 _add_similarity_edges_ann
        builder.build(self.train_pairs)

        # informed 模式下不会调用过滤方法
        graph_df = builder.get_graph()

        # 预期边:
        # 1. (0, 3, 'interacts_with', 'bindingdb', 0.85) - 来自 build
        # 2. [0, 1, 'drug_drug_similarity', 'computed', 0.92] - 来自 mock
        # 3. [3, 5, 'protein_protein_similarity', 'computed', 0.88] - 来自 mock (泄露边)
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
        # 验证 source_dataset 和 score 列存在
        self.assertIn("source_dataset", graph_df.columns)
        self.assertIn("score", graph_df.columns)

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
                self_builder._edges.append(
                    [0, 1, "drug_drug_similarity", "computed", 0.92]
                )
            elif entity_type == "protein":
                self_builder._edges.append(
                    [3, 5, "protein_protein_similarity", "computed", 0.88]
                )

        mock_add_similarity_edges.side_effect = add_fake_edges

        builder = HeteroGraphBuilder(
            config=cfg,
            molecule_embeddings=self.mol_embeddings,
            protein_embeddings=self.prot_embeddings,
            local_id_to_type=self.local_id_to_type,
            protein_id_offset=self.protein_id_offset,
            cold_start_entity_ids_local={2, 5},  # 冷启动实体是 mol 2 和 prot 5
        )

        # build 现在会调用 filter_background_edges_for_strict_mode
        builder.build(self.train_pairs)

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

    def test_score_normalization_by_edge_type(self):
        """
        测试点3: 按 edge_type 分组的 min-max 归一化。
        - 同类型多条边：raw_score 被归一化到 [0, 1]
        - 不同 edge_type 独立归一化
        - 输出列名为 'score'（不是 'raw_score'）
        """
        print("\n--- Running Test: Score Normalization by Edge Type ---")

        cfg = self._get_base_config()

        builder = HeteroGraphBuilder(
            config=cfg,
            molecule_embeddings=self.mol_embeddings,
            protein_embeddings=self.prot_embeddings,
            local_id_to_type=self.local_id_to_type,
            protein_id_offset=self.protein_id_offset,
        )

        # 手动添加多条不同 edge_type 的边，raw_score 不同
        builder._edges = [
            [0, 3, "interacts_with", "bindingdb", 10.0],
            [1, 4, "interacts_with", "gtopdb", 100.0],
            [0, 1, "drug_drug_similarity", "computed", 0.6],
            [1, 2, "drug_drug_similarity", "computed", 0.9],
        ]

        graph_df = builder.get_graph()

        # 验证列名是 'score' 而不是 'raw_score'
        self.assertIn("score", graph_df.columns)
        self.assertNotIn("raw_score", graph_df.columns)

        # 验证 interacts_with 组归一化: min=10, max=100
        # (10-10)/(100-10) = 0.0, (100-10)/(100-10) = 1.0
        dti_rows = graph_df[graph_df["edge_type"] == "interacts_with"].sort_values(
            "source"
        )
        self.assertAlmostEqual(dti_rows["score"].iloc[0], 0.0, places=5)
        self.assertAlmostEqual(dti_rows["score"].iloc[1], 1.0, places=5)

        # 验证 drug_drug_similarity 组独立归一化: min=0.6, max=0.9
        # (0.6-0.6)/(0.9-0.6) = 0.0, (0.9-0.6)/(0.9-0.6) = 1.0
        sim_rows = graph_df[
            graph_df["edge_type"] == "drug_drug_similarity"
        ].sort_values("source")
        self.assertAlmostEqual(sim_rows["score"].iloc[0], 0.0, places=5)
        self.assertAlmostEqual(sim_rows["score"].iloc[1], 1.0, places=5)

        # 验证 source_dataset 列保留
        self.assertIn("source_dataset", graph_df.columns)

        print("  ✅ Test Passed: Scores normalized independently by edge_type.")

    def test_single_edge_gets_score_one(self):
        """
        测试点4: 单条边（某 edge_type 只有一条）归一化后 score 为 1.0。
        """
        print("\n--- Running Test: Single Edge Score ---")

        cfg = self._get_base_config()

        builder = HeteroGraphBuilder(
            config=cfg,
            molecule_embeddings=self.mol_embeddings,
            protein_embeddings=self.prot_embeddings,
            local_id_to_type=self.local_id_to_type,
            protein_id_offset=self.protein_id_offset,
        )

        builder._edges = [
            [0, 3, "interacts_with", "bindingdb", 42.0],
        ]

        graph_df = builder.get_graph()
        self.assertEqual(len(graph_df), 1)
        self.assertAlmostEqual(graph_df["score"].iloc[0], 1.0, places=5)

        print("  ✅ Test Passed: Single edge normalized to 1.0.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
