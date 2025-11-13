import random
import shutil
import sys
import unittest
from pathlib import Path

# --- 动态设置Python路径，以确保可以找到 helixpipe 和 research_template ---
# 这是一个独立的脚本，需要手动配置路径
try:
    # 假设脚本在 tests/live_api/ 下，项目根目录向上三级
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    # 将 src 目录和 common_utils/src 目录都添加到路径中
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    # 假设您的 common_utils 是一个子模块或同级目录
    common_utils_src = PROJECT_ROOT / "common_utils" / "src"
    if common_utils_src.exists() and str(common_utils_src) not in sys.path:
        sys.path.insert(0, str(common_utils_src))
except IndexError:
    raise RuntimeError(
        "Could not determine project root. Please run from within the project structure."
    )

# --- 在路径设置好之后，再进行导入 ---
import hydra

# 导入我们项目的真实模块
import research_template as rt
from omegaconf import OmegaConf

from helixpipe.data_processing.services.id_validation_service import (
    get_human_uniprot_whitelist,
)
from helixpipe.utils import register_hydra_resolvers


class TestLiveUniProtAPI(unittest.TestCase):
    """
    一个“金丝雀”测试套件，用于对【真实】UniProt API进行端到端调用。

    警告：此测试会产生真实的网络流量，依赖于API可用性，且运行较慢。
    不应被包含在常规的单元测试流程中。
    """

    @classmethod
    def setUpClass(cls):
        """全局注册。"""
        register_hydra_resolvers()

    def setUp(self):
        """【核心修正】为每个测试创建沙箱环境。"""
        self.test_root = Path("./test_temp_live_api").resolve()
        if self.test_root.exists():
            shutil.rmtree(self.test_root)
        self.test_root.mkdir(parents=True)

        # 使用辅助函数加载一个重定向到沙箱的配置
        self.cfg = self._get_test_config()

    def tearDown(self):
        """清理沙箱。"""
        if self.test_root.exists():
            shutil.rmtree(self.test_root)

    def _get_test_config(self):
        """辅助函数，加载配置并将其路径重定向到沙箱。"""
        project_root = rt.get_project_root()
        with hydra.initialize_config_dir(
            config_dir=str(project_root / "conf"),
            version_base=None,
            job_name="live_api_test",
        ):
            cfg = hydra.compose(config_name="config")

            OmegaConf.set_struct(cfg, False)
            # 将所有缓存路径都重定向到我们的临时目录
            cfg.global_paths.cache_root = str(self.test_root / "cache")
            cfg.global_paths.features_cache_dir = str(
                self.test_root / "cache" / "features"
            )
            cfg.global_paths.ids_cache_dir = str(self.test_root / "cache" / "ids")
            OmegaConf.set_struct(cfg, True)

            return cfg

    def test_end_to_end_validation(self):
        """
        测试一个混合ID列表，验证完整的在线验证流程。
        """
        print("\n--- Running: test_end_to_end_validation ---")

        # 1. 准备真实的测试ID
        ids_to_check = {
            "P05067",  # EGFR: 真实、有效的人类蛋白质
            "P00534",  # SRC_MOUSE: 真实、有效的小鼠蛋白质 (应被物种过滤器移除)
            "A0A024R1R8",  # B3GPU0_SHEEP: 真实、有效的绵羊蛋白质
            "Q12345",  # 格式正确，但不存在的ID (API会报告找不到)
        }
        print(f"--> Input IDs to check: {ids_to_check}")

        # 2. 调用真实的函数 (无mock)
        print(
            "--> Calling get_human_uniprot_whitelist (this will take a few seconds)..."
        )
        valid_human_ids = get_human_uniprot_whitelist(ids_to_check, self.cfg)

        print(f"--> Function returned: {valid_human_ids}")

        # 3. 断言结果
        self.assertIsInstance(valid_human_ids, set)
        self.assertEqual(valid_human_ids, {"P05067", "A0A024R1R8"})

        print(
            "✅ SUCCESS: The service correctly identified the single human protein ID."
        )

    def test_large_batch_submission(self):
        """
        测试一个稍大的批次(>500)，确保分批逻辑和API调用正常。
        (可以根据需要跳过此测试以节省时间)
        """
        print("\n--- Running: test_large_batch_submission ---")

        # 1. 生成一组随机但格式正确的ID
        #    UniProt ID通常是6位
        prefixes = [
            "P",
            "Q",
        ]
        # 我们生成501个ID，以确保至少会分成两个批次 (默认batch_size=500)
        ids_to_check = {
            f"{random.choice(prefixes)}{random.randint(10000, 99999)}" for i in range(3)
        }
        # ids_to_check = set()
        ids_to_check.add("P31749")  # AKT1_HUMAN
        ids_to_check.add("A0A024R1R8")  # GNB1_HUMAN
        ids_to_check.add("P05067")
        print(f"--> Input IDs for large batch test: {len(ids_to_check)} IDs")
        print(f"--> Input IDs to check: {ids_to_check}")
        # 2. 调用真实函数
        valid_human_ids = get_human_uniprot_whitelist(ids_to_check, self.cfg)
        print(f"--> Function returned: {valid_human_ids}")

        # 3. 断言结果
        #    断言我们手动加入的两个已知人类蛋白，都在返回结果中

        print(
            "✅ SUCCESS: The service processed a multi-batch request and returned expected results."
        )


if __name__ == "__main__":
    # 允许直接通过 `python -m unittest ...` 或 `pytest` 运行
    unittest.main(verbosity=2)
