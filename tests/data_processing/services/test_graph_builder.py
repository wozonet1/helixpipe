# 文件: tests/data_processing/services/test_graph_builder.py (Builder模式测试版)

import unittest

import numpy as np
from omegaconf import OmegaConf

# 导入我们需要测试的新类和旧的依赖
from nasnet.data_processing.services.graph_builder import HeteroGraphBuilder
from nasnet.data_processing.services.graph_director import GraphDirector


# --- 模拟(Mock)对象和数据 ---
class MockIDMapper:
    """一个轻量级的IDMapper模拟对象，只提供必要的方法。"""

    def __init__(self):
        self.num_drugs = 2
        self.num_ligands = 1
        self.num_molecules = 3
        self.num_proteins = 2

    def get_node_type(self, node_id: int) -> str:
        if 0 <= node_id < 2:
            return "drug"
        if 2 <= node_id < 3:
            return "ligand"
        if 3 <= node_id < 5:
            return "protein"
        raise ValueError(f"Invalid node ID {node_id} for mock IDMapper")


# --- 测试用例类 ---
class TestGraphBuilderDirector(unittest.TestCase):
    def setUp(self):
        """为每个测试设置一个包含所有模拟对象的“沙箱”环境。"""
        # 1. 模拟 IDMapper
        self.mock_id_mapper = MockIDMapper()

        # 2. 模拟相似度矩阵
        # 药物/配体矩阵 (d0, d1, l2)
        self.mock_dl_sim = np.array(
            [
                [1.0, 0.9, 0.6],  # d0-d1 (pass, >0.8), d0-l2 (fail, <0.7)
                [0.9, 1.0, 0.95],  # d1-d0 (skip), d1-l2 (pass, >0.7)
                [0.6, 0.95, 1.0],  # l2-d0, l2-d1
            ]
        )
        # 蛋白质矩阵 (p3, p4)
        self.mock_prot_sim = np.array(
            [
                [1.0, 0.85],  # p3-p4 (pass, >0.8)
                [0.85, 1.0],
            ]
        )

        # 3. 模拟训练交互对 (包含最终关系类型)
        self.mock_train_pairs = [
            (0, 3, "interacts_with"),  # drug 0 - protein 3
            (1, 4, "inhibits"),  # drug 1 - protein 4
            (2, 4, "catalyzes"),  # ligand 2 - protein 4
        ]

    def _get_test_config(self, relations_flags: dict) -> dict:
        """创建一个模拟的、最小化的Config字典。"""
        return OmegaConf.create(
            {
                "runtime": {"verbose": 0},
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
                "data_params": {
                    "similarity_thresholds": {
                        "drug_drug": 0.8,
                        "drug_ligand": 0.7,
                        "ligand_ligand": 0.9,
                        "protein_protein": 0.8,
                    }
                },
                "relations": {"flags": relations_flags},
            }
        )

    def test_build_all_edge_types(self):
        """
        集成测试 1: 所有关系开关都打开时，图是否被正确构建。
        """
        print("\n--- Running Test: GraphBuilder with ALL edge types enabled ---")

        # 1. 准备输入：一个开启所有开关的配置
        config = self._get_test_config(
            relations_flags={
                "interacts_with": True,
                "inhibits": True,
                "catalyzes": True,
                "drug_drug_similarity": True,
                "drug_ligand_similarity": True,
                "ligand_ligand_similarity": False,  # <-- 特意关闭一个来测试
                "protein_protein_similarity": True,
            }
        )

        # 2. 实例化 Director 和 Builder
        director = GraphDirector(config)
        builder = HeteroGraphBuilder(
            config, self.mock_id_mapper, self.mock_dl_sim, self.mock_prot_sim
        )

        # 3. 执行构建过程
        director.construct(builder, self.mock_train_pairs)
        result_df = builder.get_graph()

        # (可选) 打印结果用于调试
        print("Generated DataFrame:")
        print(result_df.to_string())

        # 4. 断言最终结果
        self.assertEqual(
            len(result_df), 6, "Incorrect total number of edges generated."
        )

        edge_counts = result_df["edge_type"].value_counts()
        self.assertEqual(edge_counts.get("interacts_with", 0), 1)
        self.assertEqual(edge_counts.get("inhibits", 0), 1)
        self.assertEqual(edge_counts.get("catalyzes", 0), 1)
        self.assertEqual(edge_counts.get("drug_drug_similarity", 0), 1)
        self.assertEqual(edge_counts.get("drug_ligand_similarity", 0), 1)
        self.assertEqual(edge_counts.get("protein_protein_similarity", 0), 1)
        # 确认被关闭的边没有生成
        self.assertNotIn("ligand_ligand_similarity", edge_counts)

        print("  ✅ All edge counts are correct.")

    def test_build_only_interaction_edges(self):
        """
        集成测试 2: 只有部分交互关系开关打开时，是否只生成了对应的边。
        """
        print(
            "\n--- Running Test: GraphBuilder with ONLY specific interaction edges ---"
        )

        # 1. 准备输入：一个只开启 'inhibits' 和 'catalyzes' 的配置
        config = self._get_test_config(
            relations_flags={
                "interacts_with": False,  # <-- 关闭
                "inhibits": True,
                "catalyzes": True,
                "drug_drug_similarity": False,
                "drug_ligand_similarity": False,
                "ligand_ligand_similarity": False,
                "protein_protein_similarity": False,
            }
        )

        # 2. 实例化并执行
        director = GraphDirector(config)
        builder = HeteroGraphBuilder(
            config, self.mock_id_mapper, self.mock_dl_sim, self.mock_prot_sim
        )
        director.construct(builder, self.mock_train_pairs)
        result_df = builder.get_graph()

        # 3. 断言
        self.assertEqual(len(result_df), 2)
        edge_counts = result_df["edge_type"].value_counts()
        self.assertEqual(edge_counts.get("inhibits", 0), 1)
        self.assertEqual(edge_counts.get("catalyzes", 0), 1)
        self.assertNotIn("interacts_with", edge_counts)
        # 确认没有任何相似性边
        self.assertFalse(any("similarity" in s for s in edge_counts.index))

        print("  ✅ Correctly generated only specified interaction edges.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
