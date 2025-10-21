# 文件: tests/test_processors.py (最终极简黑盒版)

import unittest
from pathlib import Path
from unittest.mock import patch

import hydra
import research_template as rt
from omegaconf import DictConfig

from configs.register_schemas import register_all_schemas
from data_processing.bindingdb_processor import BindingdbProcessor
from data_processing.gtopdb_processor import GtopdbProcessor

register_all_schemas()
rt.register_hydra_resolvers()


class TestProcessorFramework(unittest.TestCase):
    def tearDown(self):
        """测试结束后，清理所有可能生成的缓存文件。"""
        # 我们不知道具体生成了哪些文件，所以我们重新加载配置来找到它们
        with hydra.initialize(config_path="../conf", version_base=None):
            cfg_db = hydra.compose("config", overrides=["data_structure=bindingdb"])
            cfg_gt = hydra.compose("config", overrides=["data_structure=gtopdb"])

        # 强制将data_root指向伪造数据目录
        cfg_db.global_paths.data_root = str(Path.cwd() / "tests" / "fake_data_v2")
        cfg_gt.global_paths.data_root = str(Path.cwd() / "tests" / "fake_data_v2")

        path_db = rt.get_path(cfg_db, "raw.authoritative_dti")
        path_gt = rt.get_path(cfg_gt, "raw.authoritative_dti")

        for path in [path_db, path_gt]:
            if path.exists():
                path.unlink()

    def _get_test_config(self, overrides: list) -> DictConfig:
        with hydra.initialize(config_path="../conf", version_base=None):
            cfg = hydra.compose("config", overrides=["data_params=test"] + overrides)

        cfg.global_paths.data_root = str(Path.cwd() / "tests" / "fake_data_v2")
        return cfg

    # ------------------ 测试用例 ------------------

    def test_bindingdb_pipeline(self):
        """黑盒测试：验证BindingDB处理器在无缓存和有缓存时的最终输出。"""
        print("\n--- Running Test: BindingDB Pipeline (Blackbox) ---")
        cfg = self._get_test_config(
            overrides=["data_structure=bindingdb", "runtime.verbose=2"]
        )
        processor = BindingdbProcessor(config=cfg)

        # --- 1. 无缓存运行 ---
        result_df1 = processor.process()

        # 只断言最终结果
        self.assertEqual(
            len(result_df1), 2, "Phase 1: Final output count is incorrect."
        )

        # 验证缓存文件是否已在正确的位置创建
        # get_path会根据被我们修改过的cfg来计算路径
        expected_cache_path = rt.get_path(cfg, "raw.authoritative_dti")
        self.assertTrue(expected_cache_path.exists())
        print(f"  ✅ No-cache phase passed. Cache created at: {expected_cache_path}")

        # --- 2. 有缓存运行 ---
        # 创建一个新的processor实例来模拟一次全新的运行
        processor_cached = BindingdbProcessor(config=cfg)
        # 我们需要知道_process_raw_data是否被跳过
        with patch.object(processor_cached, "_process_raw_data") as spy_process_raw:
            result_df2 = processor_cached.process()
            spy_process_raw.assert_not_called()

        self.assertEqual(
            len(result_df2), 2, "Phase 2: Cached output count is incorrect."
        )
        print("  ✅ With-cache phase passed.")

    def test_gtopdb_pipeline(self):
        """黑盒测试：验证GtoPdb处理器的最终输出。"""
        print("\n--- Running Test: GtoPdb Pipeline (Blackbox) ---")
        cfg = self._get_test_config(
            overrides=[
                "data_structure=gtopdb",
                "data_params=gtopdb",
                "runtime.verbose=1",
            ]
        )

        with patch(
            "data_processing.gtopdb_processor.fetch_sequences_from_uniprot",
            return_value={"P98765": "VALIDSEQ"},
        ) as mock_fetch:
            processor = GtopdbProcessor(config=cfg)
            result_df = processor.process()

            mock_fetch.assert_called_once()
            self.assertEqual(len(result_df), 1)
            print("  ✅ GtoPdb test passed.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
