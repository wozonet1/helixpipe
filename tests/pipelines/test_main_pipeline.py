# 文件: tests/test_main_pipeline.py (全新)

import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

import hydra
import numpy as np
import pandas as pd
import research_template as rt
import torch
from omegaconf import DictConfig

from nasnet.configs.register_schemas import register_all_schemas

# 导入我们需要测试的主函数和所有相关模块
from nasnet.data_processing import load_datasets, process_data
from nasnet.utils import get_path, register_hydra_resolvers

# 在所有测试开始前，全局执行一次注册
register_all_schemas()
register_hydra_resolvers()


# FIXME: gtopdb的purifier会把现在的fake_data全部清除
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
        shutil.copytree("tests/fake_data_v3", self.fake_data_dir)
        # 我们也需要一个 conf 目录，尽管我们主要用 overrides
        self.fake_conf_dir.mkdir()

    def tearDown(self):
        """清理测试环境。"""
        shutil.rmtree(self.test_dir)

    def _get_test_config(self, overrides: list) -> DictConfig:
        """加载配置，并强制路径指向我们的沙箱。"""
        project_root = rt.get_project_root()
        config_dir = project_root / "conf"
        with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
            cfg = hydra.compose("config", overrides=["data_params=test"] + overrides)

        cfg.global_paths.data_root = str(self.fake_data_dir)
        return cfg

    # ------------------ 测试用例 ------------------
    @patch("data_processing.gtopdb_processor.fetch_sequences_from_uniprot")
    @patch("features.sim_calculators.calculate_embedding_similarity")
    @patch("features.extractors.extract_chemberta_molecule_embeddings")
    @patch("features.extractors.extract_esm_protein_embeddings")
    def test_end_to_end_pipeline_for_bindingdb(
        self, mock_esm, mock_chemberta, mock_sim, mock_gtopdb_fetch
    ):
        """
        对BindingDB数据集，进行从数据加载到图文件生成的端到端集成测试。
        """
        print("\n--- Running Test: End-to-End Main Pipeline for BindingDB ---")

        # --- 1. 准备 Mock ---
        # 让特征提取器返回一个假的、但维度正确的特征字典
        # 模拟特征提取器返回假的特征

        mock_esm.return_value = {
            "P01": torch.randn(128),
            "P02": torch.randn(128),
            "P03": torch.randn(128),
        }
        mock_chemberta.return_value = {
            101: torch.randn(128),
            102: torch.randn(128),
            103: torch.randn(128),
            201: torch.randn(128),
            202: torch.randn(128),
        }

        # 模拟一个精心设计的相似度矩阵
        # 顺序: drug(101,102,103), ligand(201,202) -> 5x5
        mock_sim.return_value = np.array(
            [
                # d101 d102 d103  l201  l202
                [1.0, 0.9, 0.5, 0.8, 0.4],  # d101
                [0.0, 1.0, 0.6, 0.2, 0.85],  # d102
                [0.0, 0.0, 1.0, 0.3, 0.1],  # d103
                [0.0, 0.0, 0.0, 1.0, 0.95],  # l201
                [0.0, 0.0, 0.0, 0.0, 1.0],  # l202
            ]
        )
        mock_gtopdb_fetch.return_value = {
            "P01": "FAKE_GTO_SEQ_1",
            "P03": "FAKE_GTO_SEQ_2",
        }
        # --- 2. 获取配置并运行 ---
        cfg = self._get_test_config(
            overrides=[
                "data_structure=bindingdb",
                "relations=test_all_true",  # 打开所有边
                "data_params=test_thresholds",  # 使用我们定义的阈值
                "data_params.auxiliary_datasets=[gtopdb]",  # 加载辅助数据集
                "runtime.verbose=1",
                "training.k_folds=2",  # 使用2折交叉验证
                "training.coldstart.mode=random",  # 测试stratify
            ]
        )

        # a. 运行策略化数据加载
        base_df, extra_dfs = load_datasets(cfg)

        # b. 运行主处理流水线
        process_data(cfg, base_df=base_df, extra_dfs=extra_dfs)

        # --- 3. 验证产出物 (以fold 1为例) ---
        print("\n--- Verifying pipeline outputs for Fold 1 ---")
        # a. 【核心修正】现在 extra_dfs 不再为空
        self.assertEqual(len(extra_dfs), 1, "Expected one auxiliary DataFrame.")
        # 我们伪造的 gtopdb 数据最终应该只剩 2 条
        self.assertEqual(
            len(extra_dfs[0]),
            2,
            "GtoPdb processor did not return the expected number of rows.",
        )
        # a. 验证图结构文件
        graph_path = get_path(
            cfg, "processed.specific.graph_template", prefix="fold_1", suffix="train"
        )
        self.assertTrue(graph_path.exists())
        graph_df = pd.read_csv(graph_path)

        # 详细断言图的内容
        # 由于是随机分层抽样，我们无法精确断言交互边的数量，但可以断言相似性边的数量
        edge_counts = graph_df["edge_type"].value_counts()

        # 预期相似性边 (根据mock_sim和test_thresholds.yaml):
        # - drug_drug: 1条 (101-102, 0.9 > 0.8)
        # - drug_ligand: 2条 (101-201, 0.8 > 0.7; 102-202, 0.85 > 0.7)
        # - ligand_ligand: 1条 (201-202, 0.95 > 0.9)
        self.assertEqual(edge_counts.get("drug_drug_similarity", 0), 1)
        self.assertEqual(edge_counts.get("drug_ligand_similarity", 0), 2)
        self.assertEqual(edge_counts.get("ligand_ligand_similarity", 0), 1)

        # 蛋白质相似度 (P01, P02, P03) 应该在_stage_3中计算
        # 假设我们mock_sim返回的是3x3的蛋白矩阵
        # ... 可以添加对pp_similarity的断言 ...

        # b. 验证标签文件
        # 总共有 3+2=5条交互，2折划分后，train大约2-3条，test大约2-3条
        train_labels_path = get_path(
            cfg, "processed.specific.labels_template", prefix="fold_1", suffix="train"
        )
        test_labels_path = get_path(
            cfg, "processed.specific.labels_template", prefix="fold_1", suffix="test"
        )
        self.assertTrue(train_labels_path.exists())
        self.assertTrue(test_labels_path.exists())

        train_df = pd.read_csv(train_labels_path)
        test_df = pd.read_csv(test_labels_path)
        self.assertGreater(len(train_df), 0)
        self.assertGreater(len(test_df), 0)
        self.assertEqual(len(train_df) + (len(test_df) // 2), 5)  # 训练+测试正样本=总数


if __name__ == "__main__":
    unittest.main(verbosity=2)
