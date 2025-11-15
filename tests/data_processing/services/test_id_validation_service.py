import shutil
import unittest
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# 导入您的项目模块
import helixlib as hx
from helixpipe.configs import register_all_schemas
from helixpipe.data_processing.services.id_validation_service import (
    get_human_uniprot_whitelist,
    get_valid_pubchem_cids,
)
from helixpipe.utils import get_path, register_hydra_resolvers


class TestIDValidationServiceOffline(unittest.TestCase):
    """
    【V2 离线版】测试 IDValidationService 的本地文件处理能力。
    """

    @classmethod
    def setUpClass(cls):
        """全局注册。"""
        register_hydra_resolvers()
        register_all_schemas()

    def setUp(self):
        """为每个测试创建一个隔离的沙箱文件环境。"""
        self.test_root = Path("./test_temp_validator").resolve()
        if self.test_root.exists():
            shutil.rmtree(self.test_root)
        self.test_root.mkdir(parents=True)

        # 加载一个重定向到沙箱的配置
        self.cfg = self._get_test_config()

        # 在内存中清除全局缓存，确保每个测试都重新加载文件
        # 这需要我们在 id_validation_service.py 中稍微修改一下
        # 假设我们有一个重置函数：reset_whitelist_cache()
        from helixpipe.data_processing.services import id_validation_service

        id_validation_service._local_human_uniprot_whitelist = None

    def tearDown(self):
        """清理沙箱。"""
        if self.test_root.exists():
            shutil.rmtree(self.test_root)

    def _get_test_config(self, overrides: list | None = None) -> DictConfig:
        """加载配置并将其路径重定向到沙箱。"""
        overrides = overrides or []
        project_root = hx.get_project_root()
        with hydra.initialize_config_dir(
            config_dir=str(project_root / "conf"),
            version_base=None,
            job_name="test_validator",
        ):
            cfg = hydra.compose(config_name="config", overrides=overrides)
            OmegaConf.set_struct(cfg, False)
            # 将 assets 目录重定向到我们的沙箱
            cfg.global_paths.assets_dir = str(self.test_root / "assets")
            OmegaConf.set_struct(cfg, True)
            return cfg

    def _create_fake_proteome_tsv(self, content: str):
        """一个辅助函数，用于在沙箱中创建伪造的蛋白质组TSV文件。"""
        # 使用 get_path 来确保路径与主逻辑一致
        proteome_path = get_path(self.cfg, "assets.uniprot_proteome_tsv")
        hx.ensure_path_exists(proteome_path)
        with open(proteome_path, "w") as f:
            f.write(content)

    # ------------------ 测试用例 ------------------

    def test_correctly_filters_proteome_file(self):
        """测试场景1: 验证函数能否正确读取并筛选一个标准的TSV文件。"""
        print("\n--- Running Test: Correctly filters proteome file ---")

        # 1. 准备伪造的TSV文件内容
        fake_tsv_content = (
            "Entry\tReviewed\tOrganism (ID)\n"
            "P05067\treviewed\t9606\n"  # 1. 应该通过
            "A0A024R1R8\treviewed\t9606\n"  # 2. 应该通过
            "P12345\tunreviewed\t9606\n"  # 3. 应该被 'reviewed' 状态过滤
            "P00534\treviewed\t10090\n"  # 4. 应该被 'Organism (ID)' 过滤 (小鼠)
            "P99999\treviewed\t9606\n"  # 5. 应该通过
        )
        self._create_fake_proteome_tsv(fake_tsv_content)

        # 2. 准备输入ID
        ids_to_check = {"P05067", "A0A024R1R8", "P12345", "P00534", "P99999", "Q11111"}

        # 3. 调用被测函数
        valid_ids = get_human_uniprot_whitelist(ids_to_check, self.cfg)

        # 4. 断言结果
        expected_ids = {"P05067", "A0A024R1R8", "P99999"}
        self.assertEqual(valid_ids, expected_ids)
        print("  ✅ Correct filtering test passed.")

    def test_handles_file_not_found(self):
        """测试场景2: 当蛋白质组TSV文件不存在时，函数是否会抛出预期的异常。"""
        print("\n--- Running Test: Handles file not found ---")

        # 1. (不创建任何文件)

        # 2. 准备输入ID
        ids_to_check = {"P05067"}

        # 3. 使用 assertRaises 来捕获并验证异常
        with self.assertRaises(FileNotFoundError) as cm:
            get_human_uniprot_whitelist(ids_to_check, self.cfg)

        # 4. (可选) 检查异常消息是否符合预期
        self.assertIn("Human proteome TSV file not found", str(cm.exception))
        print("  ✅ FileNotFoundError handling test passed.")

    def test_handles_malformed_tsv(self):
        """测试场景3: 当TSV文件格式错误时，函数是否能优雅地处理并返回空集合。"""
        print("\n--- Running Test: Handles malformed TSV ---")

        # 1. 准备一个缺少必需列的TSV文件
        fake_tsv_content = "Entry\tSomeOtherColumn\nP05067\tsome_value\n"
        self._create_fake_proteome_tsv(fake_tsv_content)

        ids_to_check = {"P05067"}

        # 2. 调用函数并断言结果为空
        valid_ids = get_human_uniprot_whitelist(ids_to_check, self.cfg)
        self.assertEqual(valid_ids, set())
        print("  ✅ Malformed TSV handling test passed.")

    def test_pubchem_cid_local_validator(self):
        """【保持不变】测试PubChem CID的本地验证逻辑。"""
        print("\n--- Running Test: PubChem CID local validator ---")
        cids_to_check = {123, "456", 789.0, "abc", -10, 0, None, " "}
        valid_cids = get_valid_pubchem_cids(cids_to_check, self.cfg)
        self.assertEqual(valid_cids, {123, 456, 789})
        print("  ✅ PubChem CID local validator test passed.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
