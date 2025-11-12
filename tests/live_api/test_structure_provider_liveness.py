# 文件: tests/live_api/test_structure_provider_liveness.py (更新版)

import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# --- 动态路径设置 ---
# 这部分保持不变，确保能找到 helixpipe 模块
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
except IndexError:
    raise RuntimeError("Could not determine project root.")

# --- 导入 ---
# 在路径设置好之后，再导入我们项目的模块
from omegaconf import DictConfig, OmegaConf
from rdkit import Chem

from helixpipe.configs import register_all_schemas
from helixpipe.data_processing import StructureProvider
from helixpipe.utils import register_hydra_resolvers

# 全局注册一次
register_all_schemas()
register_hydra_resolvers()

# --- 代理配置 ---
PROXY_CONFIG = None
# PROXY_CONFIG = {'http': 'http://127.0.0.1:7890', 'https': 'http://127.0.0.1:7890'}


class TestStructureProviderLiveness(unittest.TestCase):
    """
    一个“金丝雀”测试套件，用于对【StructureProvider】类的真实API调用进行端到端验证。
    """

    def setUp(self):
        """
        为每个测试创建一个临时的、隔离的缓存目录，并创建一个StructureProvider实例。
        """
        print("\n" + "=" * 80)

        # 1. 创建一个临时的测试目录作为沙箱
        self.test_dir = PROJECT_ROOT / "tests" / "temp_live_api_cache"
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(parents=True)

        # 2. 【核心修改】手动构建一个最小化的、可用的 config 对象
        #    我们使用 OmegaConf 来创建一个结构化的配置，就像 Hydra 做的那样
        self.cfg: DictConfig = OmegaConf.create(
            {
                "global_paths": {
                    # 将所有缓存路径都重定向到我们的临时目录
                    "cache_root": str(self.test_dir),
                    "ids_cache_dir": "${global_paths.cache_root}/ids",
                },
                "data_structure": {
                    "filenames": {
                        "cache": {
                            "ids": {
                                # 提供文件名模板
                                "enriched_protein_sequences": "test_enriched_protein_sequences.pkl",
                                "enriched_molecule_smiles": "test_enriched_molecule_smiles.pkl",
                            }
                        }
                    },
                    # 提供路径模板，以便 get_path 可以解析
                    "paths": {
                        "cache": {
                            "ids": {
                                "enriched_protein_sequences": "${path:cache.ids.enriched_protein_sequences}",
                                "enriched_molecule_smiles": "${path:cache.ids.enriched_molecule_smiles}",
                            }
                        }
                    },
                },
                "runtime": {
                    "verbose": 1,
                },
            }
        )

        # 3. 实例化 StructureProvider，并传入 config
        self.provider = StructureProvider(config=self.cfg, proxies=PROXY_CONFIG)
        print(
            f"--> StructureProvider initialized. Cache will be stored in: {self.test_dir}"
        )

    def tearDown(self):
        """测试结束后，清理临时缓存目录。"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        print(f"--> Temporary cache directory cleaned: {self.test_dir}")

    def test_get_sequences_liveness_and_caching(self):
        """
        活性测试：验证 get_sequences 方法，并检查缓存功能。
        """
        print(
            "\n--- Running Liveness Test: StructureProvider.get_sequences (with Caching) ---"
        )

        # 定义初始请求的ID列表，包含有效和无效ID
        # FIXME: Why P14060 can't return from the server?
        all_test_pids = ["P05067", "P00533", "INVALID_ID", "Q123456789", "P14060"]

        # --- 阶段 1: 无缓存运行 ---
        print("\n--- Phase 1: No-Cache Run ---")

        # 使用 wraps 来监视，同时允许真实调用发生
        with patch.object(
            self.provider,
            "_fetch_sequences_from_uniprot",
            wraps=self.provider._fetch_sequences_from_uniprot,
        ) as spy_fetch_phase1:
            results1 = self.provider.get_sequences(all_test_pids, force_restart=True)

            # 断言1.1: 底层 fetch 方法确实被调用了一次
            spy_fetch_phase1.assert_called_once()

            # 断言1.2: 【核心修复】传递给 fetch 方法的参数，是 run_cached_operation
            #           计算出的、完整的“待办”ID列表。
            self.assertCountEqual(spy_fetch_phase1.call_args[0][0], all_test_pids)

        # 断言1.3: 最终返回的结果只包含2个有效ID
        self.assertEqual(len(results1), 3)
        self.assertIn("P05067", results1)
        self.assertIn("P14060", results1)
        # 断言1.4: 缓存文件被成功创建
        cache_path = (
            Path(self.cfg.global_paths.ids_cache_dir)
            / self.cfg.data_structure.filenames.cache.ids.enriched_protein_sequences
        )
        self.assertTrue(cache_path.exists())

        print("  ✅ Phase 1 passed. API called correctly and cache file created.")

        # --- 阶段 2: 有缓存运行 ---
        print("\n--- Phase 2: With-Cache Run ---")

        # 在第二阶段，我们只请求那些我们知道【应该】已经在缓存里的ID
        cached_pids = ["P05067", "P00533"]

        with patch.object(
            self.provider,
            "_fetch_sequences_from_uniprot",
            wraps=self.provider._fetch_sequences_from_uniprot,
        ) as spy_fetch_phase2:
            results2 = self.provider.get_sequences(cached_pids)

            # 断言2.1: 【核心修复】由于请求的ID已全部被缓存，底层的 fetch 方法
            #          完全不应该被调用。
            spy_fetch_phase2.assert_not_called()

        # 断言2.2: 返回的结果仍然是正确的2个
        self.assertEqual(len(results2), 2)

        print(
            "  ✅ Phase 2 passed. Fetch method was correctly skipped for fully cached request."
        )

    def test_get_smiles_liveness(self):
        """
        活性测试：验证 StructureProvider.get_smiles 方法。
        (为了简洁，这里省略了对get_smiles的缓存测试，逻辑与get_sequences相同)
        """
        print("\n--- Running Liveness Test: StructureProvider.get_smiles ---")

        test_cids = [2244, 5288826, 999999999]

        results = self.provider.get_smiles(test_cids)

        # 断言 (与之前版本相同)
        self.assertEqual(len(results), 2)
        self.assertIn(2244, results)

        expected_aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
        actual_aspirin_smiles = results.get(2244, "")
        mol_expected = Chem.MolFromSmiles(expected_aspirin_smiles)
        mol_actual = Chem.MolFromSmiles(actual_aspirin_smiles)

        self.assertIsNotNone(mol_actual)
        self.assertEqual(
            Chem.MolToSmiles(mol_actual, canonical=True),
            Chem.MolToSmiles(mol_expected, canonical=True),
        )

        print("✅ --- Liveness test for get_smiles PASSED ---")


if __name__ == "__main__":
    unittest.main(verbosity=2)
