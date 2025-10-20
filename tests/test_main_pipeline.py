# 文件: tests/test_main_pipeline.py (全新)

import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import hydra
from omegaconf import DictConfig
import torch

# 导入我们需要测试的主函数和所有相关模块
from data_utils.data_loader_strategy import load_datasets
from data_processing.main_pipeline import process_data
from configs.register_schemas import register_all_schemas
import research_template as rt

# 在所有测试开始前，全局执行一次注册
register_all_schemas()
rt.register_hydra_resolvers()


class TestMainPipeline(unittest.TestCase):
    def setUp(self):
        """为测试创建一个完全隔离的沙箱环境。"""
        self.test_dir = Path("./test_temp_output_main_pipeline")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

        self.fake_project_root = self.test_dir
        self.fake_data_dir = self.fake_project_root / "data"
        self.fake_conf_dir = self.fake_project_root / "conf"

        # 复制伪造的原始数据到沙箱中
        shutil.copytree("tests/fake_data_v2", self.fake_data_dir)
        # 我们也需要一个 conf 目录，尽管我们主要用 overrides
        self.fake_conf_dir.mkdir()

    def tearDown(self):
        """清理测试环境。"""
        shutil.rmtree(self.test_dir)

    def _get_test_config(self, overrides: list) -> DictConfig:
        """加载配置，并强制路径指向我们的沙箱。"""
        with hydra.initialize(config_path="../conf", version_base=None):
            cfg = hydra.compose("config", overrides=["data_params=test"] + overrides)

        cfg.global_paths.data_root = str(self.fake_data_dir)
        return cfg

    # ------------------ 测试用例 ------------------

    @patch("features.sim_calculators.calculate_embedding_similarity")
    @patch("features.extractors.extract_chemberta_molecule_embeddings")
    @patch("features.extractors.extract_esm_protein_embeddings")
    def test_end_to_end_pipeline_for_bindingdb(
        self, mock_esm, mock_chemberta, mock_sim
    ):
        """
        对BindingDB数据集，进行从数据加载到图文件生成的端到端集成测试。
        """
        print("\n--- Running Test: End-to-End Main Pipeline for BindingDB ---")

        # --- 1. 准备 Mock ---
        # 让特征提取器返回一个假的、但维度正确的特征字典
        mock_esm.return_value = {
            "P12345": torch.randn(1, 128),
            "P67890": torch.randn(1, 128),
        }
        mock_chemberta.return_value = {
            1001: torch.randn(1, 128),
            1002: torch.randn(1, 128),
        }
        # 让相似度计算返回一个假的、固定的矩阵
        mock_sim.return_value = np.array([[1.0, 0.5], [0.5, 1.0]])

        # --- 2. 获取配置并运行 ---
        cfg = self._get_test_config(
            overrides=[
                "data_structure=bindingdb",
                "runtime.verbose=1",
                "training.k_folds=1",  # 为了测试简单，只跑1折
            ]
        )

        # a. 运行策略化数据加载
        base_df, extra_dfs = load_datasets(cfg)
        self.assertEqual(len(base_df), 2)  # 确认加载的数据量正确
        self.assertEqual(len(extra_dfs), 0)

        # b. 运行主处理流水线
        process_data(cfg, base_df=base_df, extra_dfs=extra_dfs)

        # --- 3. 验证产出物 ---
        print("\n--- Verifying pipeline outputs ---")

        # a. 验证图结构文件是否已生成
        graph_path = rt.get_path(
            cfg,
            "data_structure.paths.processed.specific.graph_template",
            prefix="fold_1",
            suffix="train",
        )
        self.assertTrue(graph_path.exists(), "Graph file was not created.")

        # b. 验证图结构文件的内容
        graph_df = pd.read_csv(graph_path)
        print("Generated Graph Edges:")
        print(graph_df.to_string())

        # 我们预期图中有2条dp_interaction边 (因为我们的fake data最终剩2条)
        # 并且，因为相似度是0.5，不满足默认的0.8阈值，所以不应该有相似性边
        self.assertEqual(
            len(graph_df[graph_df["edge_type"] == "drug_protein_interaction"]), 2
        )
        self.assertFalse("drug_drug_similarity" in graph_df["edge_type"].values)
        print("✅ Graph content is correct.")

        # c. 验证标签文件是否已生成
        train_labels_path = rt.get_path(
            cfg,
            "data_structure.paths.processed.specific.labels_template",
            prefix="fold_1",
            suffix="train",
        )
        test_labels_path = rt.get_path(
            cfg,
            "data_structure.paths.processed.specific.labels_template",
            prefix="fold_1",
            suffix="test",
        )
        self.assertTrue(train_labels_path.exists())
        self.assertTrue(test_labels_path.exists())
        print("✅ Label files were created.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
