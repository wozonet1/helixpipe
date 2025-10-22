# 文件: tests/live_api/test_structure_provider_liveness.py (全新)

import sys
import unittest
from pathlib import Path

from rdkit import Chem

# --- 动态路径设置 ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
except IndexError:
    raise RuntimeError("Could not determine project root.")

# --- 导入我们要测试的类 ---
from nasnet.data_processing.services.structure_provider import StructureProvider

# --- 代理配置 ---
# 如果不需要代理，请确保此为 None
PROXY_CONFIG = None
# PROXY_CONFIG = {'http': 'http://127.0.0.1:7890', 'https': 'http://127.0.0.1:7890'}


class TestStructureProviderLiveness(unittest.TestCase):
    """
    一个“金丝雀”测试套件，用于对【StructureProvider】类的真实API调用进行端到端验证。
    """

    def setUp(self):
        """在每个测试开始前，都创建一个新的StructureProvider实例。"""
        print("\n" + "=" * 80)
        self.provider = StructureProvider(proxies=PROXY_CONFIG)

    def test_get_sequences_liveness(self):
        """
        活性测试：验证 StructureProvider.get_sequences 方法。
        """
        print("--- Running Liveness Test: StructureProvider.get_sequences ---")

        # 1. 准备测试ID
        test_pids = [
            "P05067",  # EGFR_HUMAN (有效)
            "P00533",  # SRC_CHICK (有效)
            "INVALID_ID",  # 格式错误
            "Q123456789",  # 格式正确但不存在
        ]
        print(f"--> Input PIDs: {test_pids}")

        # 2. 调用被测方法
        results = self.provider.get_sequences(test_pids)
        print(f"--> API returned {len(results)} results.")

        # 3. 断言
        self.assertIsInstance(results, dict)

        # 验证成功的查询
        self.assertIn("P05067", results)
        self.assertIn("P00533", results)

        # 验证序列内容
        egfr_seq = results.get("P05067", "")
        self.assertIsInstance(egfr_seq, str)
        self.assertTrue(egfr_seq.startswith("M"))

        # 验证无效和不存在的ID被正确忽略
        self.assertNotIn("INVALID_ID", results)
        self.assertNotIn("Q123456789", results)

        # 验证最终数量
        self.assertEqual(len(results), 2)

        print("✅ --- Liveness test for get_sequences PASSED ---")

    def test_get_smiles_liveness(self):
        """
        活性测试：验证 StructureProvider.get_smiles 方法。
        """
        print("--- Running Liveness Test: StructureProvider.get_smiles ---")

        # 1. 准备测试ID
        test_cids = [
            2244,  # Aspirin (有效)
            5288826,  # Taxol (有效)
            999999999,  # 不存在
        ]
        print(f"--> Input CIDs: {test_cids}")

        # 2. 调用被测方法
        results = self.provider.get_smiles(test_cids)
        print(f"--> API returned {len(results)} results.")

        # 3. 断言
        self.assertIsInstance(results, dict)

        # 验证成功的查询
        self.assertIn(2244, results)
        self.assertIn(5288826, results)

        # 验证不存在的ID被正确忽略
        self.assertNotIn(999999999, results)

        # 验证最终数量
        self.assertEqual(len(results), 2)

        # 使用化学等价性验证SMILES
        expected_aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
        actual_aspirin_smiles = results.get(2244, "")

        mol_expected = Chem.MolFromSmiles(expected_aspirin_smiles)
        mol_actual = Chem.MolFromSmiles(actual_aspirin_smiles)

        self.assertIsNotNone(
            mol_actual, f"Actual SMILES string '{actual_aspirin_smiles}' is invalid."
        )

        self.assertEqual(
            Chem.MolToSmiles(mol_actual, canonical=True),
            Chem.MolToSmiles(mol_expected, canonical=True),
        )

        print("✅ --- Liveness test for get_smiles PASSED ---")


if __name__ == "__main__":
    unittest.main(verbosity=2)
