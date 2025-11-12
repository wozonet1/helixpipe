# 文件: tests/pipelines/test_main_pipeline.py (全新集成测试版)
# TODO: 尚未完成
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from helixpipe.configs import register_all_schemas

# 导入我们需要测试的主函数和所有相关模块
from helixpipe.pipelines.main_pipeline import process_data
from helixpipe.utils import get_path, register_hydra_resolvers

# 全局执行一次注册
register_all_schemas()
register_hydra_resolvers()


class TestMainPipeline(unittest.TestCase):
    def setUp(self):
        """为测试创建一个完全隔离的沙箱环境。"""
        self.test_dir = Path("./test_temp_main_pipeline").resolve()
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

        # 将伪造数据复制到沙箱中
        shutil.copytree("tests/fake_data_v3", self.test_dir / "data")

    def tearDown(self):
        """清理测试环境。"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def _get_test_config(self, overrides: list) -> DictConfig:
        """加载配置，并强制路径指向我们的沙箱。"""
        # 使用真实的 compose API 来构建一个完整的配置
        with hydra.initialize(config_path="../../conf", version_base=None):
            cfg = hydra.compose("config", overrides=overrides)

        # 强制所有路径都指向沙箱
        cfg.global_paths.data_root = str(self.test_dir / "data")
        # 确保缓存路径也在这里
        cfg.global_paths.cache_root = str(self.test_dir / "data" / "cache")
        cfg.global_paths.ids_cache_dir = str(self.test_dir / "data" / "cache" / "ids")

        return cfg

    # 使用 patch 装饰器来模拟所有外部和耗时的调用
    @patch("helixpipe.data_processing.services.structure_provider.StructureProvider")
    @patch("helixpipe.features.similarity_calculators.calculate_embedding_similarity")
    @patch("helixpipe.features.extractors.extract_chemberta_molecule_embeddings")
    @patch("helixpipe.features.extractors.extract_esm_protein_embeddings")
    def test_end_to_end_pipeline_with_brenda(
        self, mock_esm, mock_chemberta, mock_sim, mock_structure_provider
    ):
        """
        对BindingDB(主)+Brenda(辅)数据集，进行从数据加载到图文件生成的端到端集成测试。
        """
        print("\n--- Running Test: End-to-End Main Pipeline with Brenda Auxiliary ---")

        # --- 1. 准备 Mock ---
        # 模拟 StructureProvider 返回固定的结构信息
        mock_sp_instance = mock_structure_provider.return_value
        mock_sp_instance.get_sequences.return_value = {
            "P01": "SEQ_P01_AUTH",
            "P02": "SEQ_P02_AUTH",
            "P03": "SEQ_P03_AUTH",
        }
        mock_sp_instance.get_smiles.return_value = {
            101: "CCO",
            102: "CCC",
            201: "CC(=O)OC1=CC=CC=C1C(=O)O",
            202: "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        }

        # 模拟特征提取器返回固定的特征字典
        # 顺序: drug(101,102), ligand(201,202) -> 4个分子
        # 蛋白: P01, P02, P03 -> 3个蛋白
        mock_chemberta.return_value = {
            101: torch.randn(128),
            102: torch.randn(128),
            201: torch.randn(128),
            202: torch.randn(128),
        }
        mock_esm.return_value = {
            "P01": torch.randn(128),
            "P02": torch.randn(128),
            "P03": torch.randn(128),
        }

        # 模拟一个精心设计的相似度矩阵
        # 第一次调用（分子）返回 4x4, 第二次（蛋白）返回 3x3
        mock_sim.side_effect = [
            np.array(
                [  # 分子: d101, d102, l201, l202
                    [1.0, 0.9, 0.5, 0.6],  # d101-d102 (pass > 0.8)
                    [0.9, 1.0, 0.75, 0.4],  # d102-l201 (pass > 0.7)
                    [0.5, 0.75, 1.0, 0.2],
                    [0.6, 0.4, 0.2, 1.0],
                ]
            ),
            np.array(
                [  # 蛋白: p01, p02, p03
                    [1.0, 0.95, 0.5],  # p01-p02 (pass > 0.9)
                    [0.95, 1.0, 0.8],
                    [0.5, 0.8, 1.0],
                ]
            ),
        ]

        # --- 2. 获取配置并运行 ---
        cfg = self._get_test_config(
            overrides=[
                "data_structure=bindingdb",
                "data_params=test_main_pipeline",
                "relations=DPL_sim",  # 打开所有边
                "data_params.auxiliary_datasets=[brenda]",  # 加载辅助数据集
                "runtime.verbose=1",
                "training.k_folds=1",  # 测试单折即可
            ]
        )

        # a. 运行主处理流水线 (黑盒调用)
        # 注意：在真实的 main.py 中，load_datasets 是在 process_data 外部调用的
        from helixpipe.data_loader_strategy import load_datasets

        base_df, extra_dfs = load_datasets(cfg)
        process_data(cfg, base_df=base_df, extra_dfs=extra_dfs)

        # --- 3. 验证最终产出物 (以fold 1为例) ---
        print("\n--- Verifying pipeline outputs for Fold 1 ---")

        # a. 验证图结构文件
        graph_path = get_path(
            cfg, "processed.specific.graph_template", prefix="fold_1", suffix="train"
        )
        self.assertTrue(graph_path.exists(), "Graph file was not created.")
        graph_df = pd.read_csv(graph_path)

        # 详细断言图的内容
        # 预期数据处理结果:
        # BindingDB: (101,P01), (102,P02) -> 关系 'interacts_with'
        # Brenda: (201,P01), (202,P01) -> 关系 'inhibits'
        # k_folds=1, 随机划分, train/test可能为空, 我们只断言相似性边
        edge_counts = graph_df["edge_type"].value_counts()

        # 预期相似性边 (根据mock_sim和test_main_pipeline.yaml):
        self.assertEqual(edge_counts.get("drug_drug_similarity", 0), 1)
        self.assertEqual(edge_counts.get("drug_ligand_similarity", 0), 1)
        self.assertEqual(edge_counts.get("protein_protein_similarity", 0), 1)
        # 交互边数量取决于随机划分，但它们应该存在
        self.assertIn("interacts_with", edge_counts)
        self.assertIn("inhibits", edge_counts)

        # b. 验证标签文件
        train_labels_path = get_path(
            cfg, "processed.specific.labels_template", prefix="fold_1", suffix="train"
        )
        test_labels_path = get_path(
            cfg, "processed.specific.labels_template", prefix="fold_1", suffix="test"
        )
        self.assertTrue(
            train_labels_path.exists(), "Train labels file was not created."
        )
        self.assertTrue(test_labels_path.exists(), "Test labels file was not created.")

        train_df = pd.read_csv(train_labels_path)
        test_df = pd.read_csv(test_labels_path)
        # 总共有 2(BDB)+2(Brenda)=4 条正样本
        # k_folds=1, test_fraction=0.2 -> train=3, test=1 (取决于stratify)
        total_positives = len(train_df) + (test_df["label"] == 1).sum()
        self.assertEqual(total_positives, 4)

        print("  ✅ All pipeline outputs are correct.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
