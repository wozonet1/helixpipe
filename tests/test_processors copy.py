# 文件: tests/test_processors.py (最终健壮版 V2)

import unittest
from unittest.mock import patch
from pathlib import Path
import shutil
import hydra
from omegaconf import DictConfig

from data_processing.bindingdb_processor import BindingDBProcessor
from data_processing.gtopdb_processor import GtoPdbProcessor
import research_template as rt
from configs.register_schemas import register_all_schemas

register_all_schemas()
rt.register_hydra_resolvers()


class TestProcessorFramework(unittest.TestCase):
    def setUp(self):
        test_method_name = self.id().split(".")[-1]
        self.test_dir = Path(f"./test_temp_output_{test_method_name}")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(parents=True)

        self.fake_data_root = Path.cwd() / "tests" / "fake_data"

        # 【核心修正】在测试开始前，主动清理可能被污染的真实缓存文件
        self.cleanup_real_cache()

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        # 在测试结束后，再次清理，确保不留下任何痕迹
        self.cleanup_real_cache()

    def cleanup_real_cache(self):
        """辅助函数：删除可能在真实data目录下生成的测试缓存。"""
        # 我们需要加载配置来找到这些路径
        with hydra.initialize(config_path="../conf", version_base=None):
            cfg_db = hydra.compose("config", overrides=["data_structure=bindingdb"])
            cfg_gt = hydra.compose("config", overrides=["data_structure=gtopdb"])

        path_db = rt.get_path(cfg_db, "data_structure.paths.raw.authoritative_dti")
        path_gt = rt.get_path(cfg_gt, "data_structure.paths.raw.authoritative_dti")

        if path_db.exists():
            path_db.unlink()
        if path_gt.exists():
            path_gt.unlink()

    def _get_test_config(self, overrides: list) -> DictConfig:
        with hydra.initialize(config_path="../conf", version_base=None):
            cfg = hydra.compose("config", overrides=["data_params=test"] + overrides)
        return cfg

    # ------------------ 测试用例 ------------------

    # 【核心修正】使用更简单、更直接的patch方式
    @patch("data_processing.purifiers.purify_dti_dataframe_parallel")
    @patch("data_processing.base_processor.validate_authoritative_dti_file")
    def test_01_bindingdb_no_cache_and_with_cache(self, mock_validate, mock_purify):
        """
        【合并测试】一次性测试BindingDB的无缓存和有缓存两种情况。
        """
        print("\n--- Running Test: BindingDB (No Cache & With Cache) ---")

        # --- 准备 ---
        # 模拟purify函数的行为：让它返回一个过滤后的DataFrame
        def purify_side_effect(df, cfg):
            # 模拟过滤掉无效SMILES的行为
            return df[df["SMILES"] != "INVALID-SMILES"].copy()

        mock_purify.side_effect = purify_side_effect

        cfg = self._get_test_config(overrides=["data_structure=bindingdb"])

        # 替换 get_path 的目标路径
        expected_output_path = self.test_dir / "bindingdb_interactions.csv"
        cfg.data_structure.paths.raw.authoritative_dti = str(expected_output_path)
        cfg.data_structure.paths.raw.raw_tsv = str(
            self.fake_data_root / "bindingdb" / "raw" / "BindingDB_All.tsv"
        )

        processor = BindingDBProcessor(config=cfg)

        # --- 场景1: 无缓存 ---
        print("  -> Phase 1: Testing without cache...")
        result_df1 = processor.process()

        # 断言 (无缓存)
        self.assertEqual(len(result_df1), 1)
        self.assertEqual(result_df1.iloc[0]["PubChem_CID"], 2222)
        mock_validate.assert_called_once()
        self.assertTrue(expected_output_path.exists())
        print("  ✅ No-cache phase passed.")

        # --- 场景2: 有缓存 ---
        print("\n  -> Phase 2: Testing with cache...")
        mock_validate.reset_mock()  # 重置计数器

        with patch.object(
            processor, "_process_raw_data", wraps=processor._process_raw_data
        ) as spy_process:
            result_df2 = processor.process()

            # 断言 (有缓存)
            spy_process.assert_not_called()
            mock_validate.assert_called_once()
            self.assertEqual(len(result_df2), 1)
            print("  ✅ With-cache phase passed.")

    @patch(
        "data_processing.gtopdb_processor.fetch_sequences_from_uniprot",
        return_value={"P98765": "TESTSEQ"},
    )
    def test_02_gtopdb_processor(self, mock_fetch):
        print("\n--- Running Test: GtoPdb Processor ---")
        cfg = self._get_test_config(
            overrides=["data_structure=gtopdb", "data_params=gtopdb"]
        )

        # 替换路径
        cfg.data_structure.paths.raw.authoritative_dti = str(
            self.test_dir / "gtopdb_interactions.csv"
        )
        cfg.data_structure.paths.raw.interactions = str(
            self.fake_data_root / "gtopdb" / "raw" / "interactions.csv"
        )
        cfg.data_structure.paths.raw.ligands = str(
            self.fake_data_root / "gtopdb" / "raw" / "ligands.csv"
        )

        processor = GtoPdbProcessor(config=cfg)
        result_df = processor.process()

        mock_fetch.assert_called_once()
        self.assertEqual(len(result_df), 1)
        self.assertEqual(result_df.iloc[0]["PubChem_CID"], 5555)
        print("  ✅ GtoPdb test passed.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
