# 文件: tests/test_graph_builder.py (最终可靠版 V2)

import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from pathlib import Path
import hydra
from omegaconf import DictConfig
from project_types import AppConfig

# 导入我们需要测试的类和模块
from data_utils.graph_builder import GraphBuilder
from data_utils.id_mapper import IDMapper
import research_template as rt
from configs.register_schemas import register_all_schemas

# 在所有测试开始前，全局执行一次注册
register_all_schemas()
rt.register_hydra_resolvers()


class TestGraphBuilder(unittest.TestCase):
    def setUp(self):
        """为每个测试设置一个包含所有模拟对象的“沙箱”环境。"""
        print("\n" + "=" * 80)

        # 1. 创建一个行为可预测的模拟 IDMapper
        self.mock_id_mapper = MagicMock(spec=IDMapper)
        self.mock_id_mapper.num_drugs = 2
        self.mock_id_mapper.num_ligands = 1
        self.mock_id_mapper.num_molecules = 3
        self.mock_id_mapper.num_proteins = 2

        def get_node_type_side_effect(node_id):
            if 0 <= node_id < 2:
                return "drug"
            if 2 <= node_id < 3:
                return "ligand"
            if 3 <= node_id < 5:
                return "protein"
            raise ValueError(f"Invalid node ID {node_id} for mock IDMapper")

        self.mock_id_mapper.get_node_type.side_effect = get_node_type_side_effect

        # 2. 创建模拟的相似度矩阵
        self.mock_dl_sim = np.array(
            [
                [1.0, 0.9, 0.6],  # d0-d1 (pass, >0.8), d0-l2 (fail, <0.7)
                [0.9, 1.0, 0.95],  # d1-d0 (skip), d1-l2 (pass, >0.7)
                [0.6, 0.95, 1.0],
            ]
        )
        self.mock_prot_sim = np.array(
            [
                [1.0, 0.85],  # p3-p4 (pass, >0.8)
                [0.85, 1.0],
            ]
        )

        # 3. 创建模拟的训练交互对
        self.mock_train_pairs = [
            (0, 3),  # drug 0 - protein 3
            (2, 4),  # ligand 2 - protein 4
        ]

    def _get_test_config(self, overrides: list) -> DictConfig:
        """使用真正的Hydra Compose API来构建一个完整的、可用于测试的配置对象。"""
        with hydra.initialize(config_path="../conf", version_base=None):
            cfg = hydra.compose(config_name="config", overrides=overrides)
        return cfg

    def _run_build_and_capture_df(self, config: AppConfig) -> pd.DataFrame:
        """
        一个辅助函数，它运行GraphBuilder，并精确地捕获最终生成的DataFrame。
        """
        builder = GraphBuilder(
            config=config,
            id_mapper=self.mock_id_mapper,
            dl_sim_matrix=self.mock_dl_sim,
            prot_sim_matrix=self.mock_prot_sim,
        )

        with patch("pandas.DataFrame") as mock_pd_dataframe:
            mock_df_instance = MagicMock()
            # 模拟to_csv，防止在某些情况下它被意外调用并需要一个返回值
            mock_df_instance.to_csv.return_value = None
            mock_pd_dataframe.return_value = mock_df_instance

            with patch(
                "research_template.path_manager.get_path",
                return_value=Path("/tmp/fake_graph.csv"),
            ):
                builder.build_for_fold(fold_idx=1, train_pairs=self.mock_train_pairs)

        self.assertTrue(
            mock_pd_dataframe.called,
            "pandas.DataFrame was not called inside GraphBuilder.",
        )
        last_call_args = mock_pd_dataframe.call_args
        captured_data = last_call_args[0][0]
        captured_columns = last_call_args[1]["columns"]

        return pd.DataFrame(captured_data, columns=captured_columns)

    def test_build_all_edge_types(self):
        """
        测试场景1：所有关系开关都打开时，图是否被正确构建。
        """
        print("--- Running Test: GraphBuilder with all edge types enabled ---")

        # 使用compose API获取一个完整的、正确的config
        cfg = self._get_test_config(
            overrides=[
                "relations=test_all_true",
                "data_params=test_thresholds",
                "runtime.verbose=2",  # 开启详细日志
            ]
        )

        # 调用捕获函数
        result_df = self._run_build_and_capture_df(cfg)

        print("\nCaptured DataFrame:")
        print(result_df.to_string())

        # --- 断言 ---
        # 预期: 1 dp + 1 lp + 1 dd + 1 dl + 1 pp = 5条边
        self.assertEqual(
            len(result_df), 5, "Incorrect total number of edges generated."
        )

        edge_counts = result_df["edge_type"].value_counts()
        self.assertEqual(edge_counts.get("drug_protein_interaction", 0), 1)
        self.assertEqual(edge_counts.get("ligand_protein_interaction", 0), 1)
        self.assertEqual(edge_counts.get("drug_drug_similarity", 0), 1)
        self.assertEqual(edge_counts.get("drug_ligand_similarity", 0), 1)
        self.assertEqual(edge_counts.get("protein_protein_similarity", 0), 1)
        self.assertNotIn("ligand_ligand_similarity", edge_counts)

        print("\n✅ All edge counts are correct.")

    def test_build_only_interaction_edges(self):
        """
        测试场景2：只有交互关系开关打开时，是否只生成交互边。
        """
        print("\n--- Running Test: GraphBuilder with only interaction edges ---")

        # 获取一个只开启交互边的配置
        cfg = self._get_test_config(
            overrides=["relations=DPL_no_sim", "runtime.verbose=2"]
        )

        # 调用捕获函数
        result_df = self._run_build_and_capture_df(cfg)

        print("\nCaptured DataFrame:")
        print(result_df.to_string())

        # --- 断言 ---
        # 预期: 1 dp + 1 lp = 2条边
        self.assertEqual(len(result_df), 2)
        edge_counts = result_df["edge_type"].value_counts()
        self.assertEqual(edge_counts.get("drug_protein_interaction", 0), 1)
        self.assertEqual(edge_counts.get("ligand_protein_interaction", 0), 1)
        self.assertFalse(any("similarity" in s for s in edge_counts.index))

        print("\n✅ Correctly generated only interaction edges.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
