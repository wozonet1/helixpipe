# 文件: tests/test_id_validator.py (全新)

import shutil
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import hydra
import research_template as rt
from omegaconf import DictConfig

from configs.register_schemas import register_all_schemas

# 导入我们需要测试的模块和函数
from data_processing.id_validator import (
    generate_id_whitelists,
)

# 全局注册
register_all_schemas()
rt.register_hydra_resolvers()


class TestIdValidator(unittest.TestCase):
    def setUp(self):
        """为每个测试创建一个完全隔离的沙箱环境。"""
        self.test_dir = Path("./test_temp_output_id_validator")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

        # 创建一个包含 conf 和 data 的完整伪造项目结构
        self.fake_project_root = self.test_dir
        self.fake_data_dir = self.fake_project_root / "data"
        self.fake_conf_dir = self.fake_project_root / "conf"
        self.fake_conf_dir.mkdir(parents=True)

        # 在沙箱中创建伪造的原始数据
        # 我们只创建 bindingdb 的数据，用于扫描
        bindingdb_raw_dir = self.fake_data_dir / "bindingdb" / "raw"
        bindingdb_raw_dir.mkdir(parents=True)

        fake_tsv_content = (
            "Ligand SMILES\tPubChem CID\tUniProt (SwissProt) Primary ID of Target Chain 1\n"
            "CCO\t101\tP12345\n"  # -> 应该通过UniProt验证
            "CCC\t102\tP99999\n"  # -> 应该被UniProt API "拒绝" (非人类)
            "CN\tabc\tQ54321\n"  # -> 应该被PubChem本地验证过滤
            "C\t-10\t \n"  # -> 应该被PubChem本地验证过滤，且PID为空
        )
        with open(bindingdb_raw_dir / "BindingDB_All.tsv", "w") as f:
            f.write(fake_tsv_content)

    def tearDown(self):
        """清理测试环境。"""
        shutil.rmtree(self.test_dir)

    def _get_test_config(self, overrides: list) -> DictConfig:
        """加载配置，并强制路径指向我们的沙箱。"""
        with hydra.initialize(config_path="../conf", version_base=None):
            cfg = hydra.compose("config", overrides=overrides)

        cfg.global_paths.data_root = str(self.fake_data_dir)
        # 确保 rt.get_project_root() 在测试时返回我们的沙箱根目录
        # 这需要 patch get_project_root
        return cfg

    # ------------------ 测试用例 ------------------

    @patch("requests.get")
    @patch("requests.post")
    @patch("time.sleep", return_value=None)  # Patch掉time.sleep以加速测试
    def test_id_validator_workflow(self, mock_sleep, mock_post, mock_get):
        """
        端到端测试id_validator的完整流程：扫描、API调用、缓存和force_restart。
        """
        print("\n--- Running Test: ID Validator Workflow ---")

        # --- 1. 准备 Mock API 的行为 ---

        # a. 模拟 POST /run 的响应
        mock_post_response = MagicMock()
        mock_post_response.json.return_value = {"jobId": "fake_job_id"}
        mock_post.return_value = mock_post_response

        # b. 模拟 GET /status 的响应
        mock_status_response = MagicMock()
        mock_status_response.json.return_value = {
            "jobStatus": "FINISHED"
        }  # 直接返回完成状态

        # c. 模拟 GET /results 的响应
        mock_results_response = MagicMock()
        mock_results_response.json.return_value = {
            "results": [
                {
                    "from": "P12345",
                    "to": {"organism": {"taxonId": 9606}},
                },  # 人类，应该保留
                {
                    "from": "P99999",
                    "to": {"organism": {"taxonId": 10090}},
                },  # 小鼠，应该被过滤
                # Q54321 没有在返回结果中，模拟API找不到这个ID
            ]
        }

        # 让 requests.get 根据URL返回不同的模拟响应
        def get_side_effect(url, **kwargs):
            if "/status/" in url:
                return mock_status_response
            if "/results/" in url:
                return mock_results_response
            return MagicMock()  # 其他get请求返回一个默认mock

        mock_get.side_effect = get_side_effect

        # --- 2. Phase 1: 无缓存运行 ---
        print("\n  -> Phase 1: Testing without cache...")
        cfg = self._get_test_config(
            overrides=["data_structure=bindingdb", "runtime.verbose=1"]
        )

        # 在patch get_project_root的上下文中运行，确保所有路径正确
        with patch(
            "research_template.path_manager.get_project_root",
            return_value=self.fake_project_root,
        ):
            generate_id_whitelists(cfg)

            # --- 断言 (无缓存) ---
            # a. 验证API是否被正确调用
            mock_post.assert_called_once()
            self.assertIn("P12345,P99999,Q54321", mock_post.call_args[1]["data"]["ids"])

            # b. 验证输出文件内容
            uniprot_whitelist_path = rt.get_path(
                cfg, "processed.common.uniprot_whitelist"
            )
            cid_whitelist_path = rt.get_path(cfg, "processed.common.cid_whitelist")

            self.assertTrue(uniprot_whitelist_path.exists())
            with open(uniprot_whitelist_path) as f:
                uniprot_ids = f.read().strip().splitlines()
            self.assertEqual(len(uniprot_ids), 1)
            self.assertEqual(uniprot_ids[0], "P12345")

            self.assertTrue(cid_whitelist_path.exists())
            with open(cid_whitelist_path) as f:
                cids = f.read().strip().splitlines()
            self.assertEqual(len(cids), 2)
            self.assertIn("101", cids)
            self.assertIn("102", cids)

        print("  ✅ No-cache phase passed.")

        # --- 3. Phase 2: 有缓存运行 ---
        print("\n  -> Phase 2: Testing with cache...")
        mock_post.reset_mock()  # 重置API调用计数器

        with patch(
            "research_template.path_manager.get_project_root",
            return_value=self.fake_project_root,
        ):
            # 再次运行，这次应该命中缓存
            generate_id_whitelists(cfg)
            # 断言：API不应该被再次调用
            mock_post.assert_not_called()

        print("  ✅ With-cache phase passed.")

        # --- 4. Phase 3: 强制重启 ---
        print("\n  -> Phase 3: Testing with force_restart...")
        cfg_force = self._get_test_config(
            overrides=["data_structure=bindingdb", "runtime.force_restart=true"]
        )

        with patch(
            "research_template.path_manager.get_project_root",
            return_value=self.fake_project_root,
        ):
            generate_id_whitelists(cfg_force)
            # 断言：即使缓存存在，API也应该被再次调用
            mock_post.assert_called_once()

        print("  ✅ Force-restart phase passed.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
