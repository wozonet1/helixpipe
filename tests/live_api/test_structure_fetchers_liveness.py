# 文件: tests/live_api/test_structure_fetchers_liveness.py

import sys
import unittest
from pathlib import Path

from rdkit import Chem

# --- 动态设置Python路径，确保可以找到 nasnet 模块 ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
except IndexError:
    raise RuntimeError("Could not determine project root.")

# --- 在路径设置好之后，再进行导入 ---
from nasnet.data_processing.services.canonicalizer import (
    fetch_sequences_from_uniprot,
    fetch_smiles_from_pubchem,
)


class TestStructureFetchersLiveness(unittest.TestCase):
    """
    一个“金丝雀”测试套件，用于对【真实】的UniProt和PubChem API进行端到端调用，
    以验证我们的核心数据获取函数是否能按预期工作。

    警告：此测试会产生真实的网络流量，依赖于API可用性，且运行较慢。
    """

    def test_fetch_sequences_from_uniprot_liveness(self):
        """
        活性测试：验证 fetch_sequences_from_uniprot 函数。
        """
        print("\n--- Running Liveness Test: fetch_sequences_from_uniprot ---")

        # 1. 准备一组包含各种情况的测试ID
        test_pids = [
            "P05067",  # EGFR_HUMAN: 一个标准、有效的、存在序列的ID
            "P00533",  # SRC_CHICK: 另一个标准ID
            "INVALID_ID",  # 一个格式完全错误的ID
            "P1209320",  # 一个格式正确但不存在的ID
        ]

        print(f"--> Input PIDs: {test_pids}")

        # 2. 调用被测函数 (无mock)
        results = fetch_sequences_from_uniprot(test_pids)
        print(f"--> API returned {len(results)} results.")

        # 3. 断言结果
        self.assertIsInstance(results, dict, "Function should return a dictionary.")

        # a. 验证成功的查询
        self.assertIn("P05067", results, "Expected to find sequence for P05067.")
        self.assertIn("P00533", results, "Expected to find sequence for P00533.")

        # b. 验证返回的序列是字符串且内容合理
        egfr_seq = results.get("P05067", "")
        self.assertIsInstance(egfr_seq, str)
        self.assertTrue(
            egfr_seq.startswith("M"), "Expected EGFR sequence to start with 'M'."
        )

        # c. 验证无效和不存在的ID被正确地忽略了
        self.assertNotIn("INVALID_ID", results, "Invalid ID should be ignored.")
        self.assertNotIn("P1209320", results, "Non-existent ID should be ignored.")

        # d. 验证最终结果的数量
        self.assertEqual(len(results), 2, "Expected exactly 2 successful results.")

        print("✅ --- Liveness test for fetch_sequences_from_uniprot PASSED ---")

    def test_fetch_smiles_from_pubchem_liveness(self):
        """
        活性测试：验证 fetch_smiles_from_pubchem 函数。
        """
        print("\n--- Running Liveness Test: fetch_smiles_from_pubchem ---")

        # 1. 准备一组测试CID
        test_cids = [
            2244,  # Aspirin: 一个标准、有效的CID
            5288826,  # Taxol: 另一个标准CID
            999999999,  # 一个不存在的CID
        ]

        print(f"--> Input CIDs: {test_cids}")

        # 2. 调用被测函数
        results = fetch_smiles_from_pubchem(test_cids)
        print(f"--> API returned {len(results)} results.")

        # 3. 断言结果
        self.assertIsInstance(results, dict, "Function should return a dictionary.")

        # a. 验证成功的查询
        self.assertIn(2244, results, "Expected to find SMILES for CID 2244 (Aspirin).")
        self.assertIn(
            5288826, results, "Expected to find SMILES for CID 5288826 (Taxol)."
        )

        # 【核心修正】使用化学等价性来验证SMILES
        expected_aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
        actual_aspirin_smiles = results.get(2244, "")

        # 1. 将两个SMILES都转换为RDKit分子对象
        mol_expected = Chem.MolFromSmiles(expected_aspirin_smiles)
        mol_actual = Chem.MolFromSmiles(actual_aspirin_smiles)

        # 2. 确保两者都能被成功解析
        self.assertIsNotNone(mol_expected, "Expected SMILES string is invalid.")
        self.assertIsNotNone(
            mol_actual, f"Actual SMILES string '{actual_aspirin_smiles}' is invalid."
        )

        # 3. 将它们都转换回同一种规范化表示再进行比较
        canon_expected = Chem.MolToSmiles(mol_expected, canonical=True)
        canon_actual = Chem.MolToSmiles(mol_actual, canonical=True)

        self.assertEqual(
            canon_actual, canon_expected, "Canonical SMILES for Aspirin do not match."
        )

        # c. 验证不存在的ID被正确地忽略了
        self.assertNotIn(999999999, results, "Non-existent CID should be ignored.")

        # d. 验证最终结果的数量
        self.assertEqual(len(results), 2, "Expected exactly 2 successful results.")

        print("✅ --- Liveness test for fetch_smiles_from_pubchem PASSED ---")


if __name__ == "__main__":
    unittest.main(verbosity=2)
