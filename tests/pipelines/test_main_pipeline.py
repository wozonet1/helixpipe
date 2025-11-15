# 文件: tests/pipelines/test_main_pipeline.py (全新集成测试版)

import shutil
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import patch

import hydra
import numpy as np
import pandas as pd
import torch

from helixpipe.configs import register_all_schemas
from helixpipe.data_loader_strategy import load_datasets
from helixpipe.pipelines.main_pipeline import process_data

# 导入所有需要的真实模块
from helixpipe.typing import AppConfig
from helixpipe.utils import get_path, register_hydra_resolvers

# 全局注册
register_all_schemas()
register_hydra_resolvers()


class TestMainPipeline(unittest.TestCase):
    def setUp(self):
        """为测试创建一个完全隔离的沙箱文件环境。"""
        self.test_dir = Path("./test_temp_main_pipeline").resolve()
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        # 将包含伪造数据的目录复制到沙箱中
        shutil.copytree("tests/fake_data_v3", self.test_dir / "data")

    def tearDown(self):
        """清理测试环境。"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def _get_test_config(self, overrides: list) -> AppConfig:
        """加载配置，并强制所有路径指向我们的沙箱。"""
        with hydra.initialize(config_path="../../conf", version_base=None):
            cfg = hydra.compose("config", overrides=overrides)

        cfg.global_paths.data_root = str(self.test_dir / "data")
        cfg.global_paths.cache_root = str(self.test_dir / "data" / "cache")
        return cast(AppConfig, cfg)

    # 使用 @patch 装饰器来模拟所有外部和耗时的调用
    @patch("helixpipe.pipelines.main_pipeline.StructureProvider")
    @patch("helixpipe.pipelines.main_pipeline.extract_features")
    @patch("helixpipe.pipelines.main_pipeline.validate_and_filter_entities")
    def test_end_to_end_pipeline_with_brenda(
        self, mock_validate_entities, mock_extract_features, mock_structure_provider_cls
    ):
        print(
            "\n"
            + "=" * 80
            + "\n--- Running Test: End-to-End Main Pipeline (Correct Mocks) ---"
        )

        # --- 1. 准备 Mock ---

        # a. 模拟 StructureProvider
        mock_sp_instance = mock_structure_provider_cls.return_value
        mock_sp_instance.get_sequences.return_value = {
            "P01": "SEQ_P01_AUTH",
            "P02": "SEQ_P02_AUTH",
        }
        mock_sp_instance.get_smiles.return_value = {
            101: "CCO",
            102: "CCC",
            201: "C1=CC=C(C=C1)C(=O)O",
            202: "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        }

        # c. 模拟 extract_features
        def extract_features_side_effect(
            entity_type, config, device, authoritative_ids, **kwargs
        ):
            print(
                f"--- [MOCK] Simulating feature extraction for {len(authoritative_ids)} '{entity_type}' entities. ---"
            )
            if entity_type == "molecule":
                return {mid: torch.randn(128) for mid in authoritative_ids}
            elif entity_type == "protein":
                return {pid: torch.randn(128) for pid in authoritative_ids}
            return {}

        mock_extract_features.side_effect = extract_features_side_effect

        # --- 2. 获取配置并运行 ---
        cfg = self._get_test_config(
            overrides=[
                "data_structure=bindingdb",
                "dataset_collection=brenda_aux",  # 加载brenda作为辅助
                "relations=DPL_sim",  # 打开所有边
                "data_params=test_main_pipeline",  # 使用一个特定的测试参数集
                "training.k_folds=1",  # 测试单折即可
                "runtime.verbose=1",
            ]
        )

        def smart_validation_mock(enriched_entities_df, config):
            print(
                "--- [SMART MOCK] Simulating entity validation by checking for non-null structures. ---"
            )

            # 1. 找到所有 structure 字段不是 None 的行
            valid_mask = enriched_entities_df["structure"].notna()

            # 2. 返回这些行的拷贝
            validated_df = enriched_entities_df[valid_mask].copy()

            print(
                f"--- [SMART MOCK] {len(validated_df)} / {len(enriched_entities_df)} entities passed mock validation."
            )
            return validated_df

        mock_validate_entities.side_effect = smart_validation_mock
        # a. 模拟主入口的调用流程
        processor_outputs = load_datasets(cfg)
        process_data(cfg, processor_outputs)

        # --- 3. 验证最终产出物 (以 fold_1 为例) ---
        print("\n--- Verifying pipeline outputs for Fold 1 ---")

        # a. 验证通用文件
        nodes_path = get_path(cfg, "processed.common.nodes_metadata")
        features_path = get_path(cfg, "processed.common.node_features")
        self.assertTrue(nodes_path.exists(), "nodes.csv was not created.")
        self.assertTrue(features_path.exists(), "node_features.npy was not created.")

        nodes_df = pd.read_csv(nodes_path)
        features_np = np.load(features_path)
        self.assertEqual(len(nodes_df), 6, "Incorrect number of nodes in nodes.csv.")
        self.assertEqual(
            features_np.shape[0],
            6,
            "Incorrect number of embeddings in node_features.npy.",
        )

        # b. 验证特定于fold的文件
        graph_path = get_path(
            cfg, "processed.specific.graph_template", prefix="fold_1", suffix="train"
        )()
        train_labels_path = get_path(
            cfg, "processed.specific.labels_template", prefix="fold_1", suffix="train"
        )()
        test_labels_path = get_path(
            cfg, "processed.specific.labels_template", prefix="fold_1", suffix="test"
        )()

        self.assertTrue(graph_path.exists(), "Graph file was not created.")
        self.assertTrue(
            train_labels_path.exists(), "Train labels file was not created."
        )
        self.assertTrue(test_labels_path.exists(), "Test labels file was not created.")

        # c. (可选) 抽样检查文件内容
        graph_df = pd.read_csv(graph_path)

        self.assertGreater(len(graph_df), 0, "Graph file should not be empty.")

        train_df = pd.read_csv(train_labels_path)
        test_df = pd.read_csv(test_labels_path)
        total_positives = len(train_df) + (test_df["label"] == 1).sum()
        self.assertEqual(
            total_positives,
            2,
            "Total positive samples in train/test labels is incorrect.",
        )

        print("\n✅ End-to-End Main Pipeline test passed successfully!")


if __name__ == "__main__":
    unittest.main(verbosity=2)
