# 文件: tests/test_purifiers.py (最终权威版)

import unittest
import pandas as pd
from rdkit import Chem, RDLogger

# 导入我们需要直接测试的函数
from data_processing.purifiers import _purify_chunk
from data_utils.canonicalizer import canonicalize_smiles


class TestPurifiers(unittest.TestCase):
    def setUp(self):
        """为测试准备一个包含所有边界情况的输入DataFrame。"""
        RDLogger.logger().setLevel(RDLogger.CRITICAL)
        print("\n" + "=" * 80)

        self.input_data = {
            "SMILES": [
                "CCO",  # 1. 存活
                "CCC",  # 2. 存活
                "INVALID-SMILES",  # 3. ❌ 无效SMILES
                "CCCCCC",  # 4. ❌ 序列无效
                "C(C)O",  # 5. 存活 (非规范SMILES)
                "CC(=O)O",  # 6. ❌ 序列含非法字符
                "CCOC",  # 7. ❌ 序列为None
                "",  # 8. ❌ 空SMILES
                None,  # 9. ❌ None SMILES
            ],
            "Sequence": [
                "MKTFY",  # 1. 存活
                "ASDFG",  # 2. 存活
                "GHJKL",  # 3. (SMILES无效)
                "POIUYTX",  # 4. ❌ 含X (我们决定将其视为非法)
                "DIFFERENTSEQ",  # 5. 存活
                "VBNMZ",  # 6. ❌ 含B,Z (我们决定将其视为非法)
                None,  # 7. ❌ None序列
                "ATGC",  # 8. (SMILES无效)
                "ATGC",  # 9. (SMILES无效)
            ],
            "PubChem_CID": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        }
        self.input_df = pd.DataFrame(self.input_data)

    def test_purify_chunk_step_by_step(self):
        """
        对 _purify_chunk 函数进行精细的、分步骤的验证。
        """
        print("--- Running Test: _purify_chunk step-by-step ---")
        print("\n--- [Step 0] Initial Input DataFrame ---")
        print(self.input_df.to_string())
        self.assertEqual(len(self.input_df), 9)

        # --- 模拟第一步：SMILES净化 ---
        print("\n--- [Step 1] SMILES Purification ---")

        # a. 验证 (过滤空/None/无效)
        smiles_mask = self.input_df["SMILES"].apply(
            lambda s: isinstance(s, str)
            and s.strip() != ""
            and Chem.MolFromSmiles(s) is not None
        )
        df_after_smiles_validation = self.input_df[smiles_mask].copy()
        print("\nDataFrame after SMILES validation:")
        print(df_after_smiles_validation.to_string())
        # 预期：过滤掉第3, 8, 9条
        self.assertEqual(len(df_after_smiles_validation), 6)

        # b. 标准化
        df_after_smiles_validation["SMILES"] = df_after_smiles_validation[
            "SMILES"
        ].apply(canonicalize_smiles)
        df_after_canonicalization = df_after_smiles_validation.dropna(subset=["SMILES"])
        print("\nDataFrame after SMILES canonicalization:")
        print(df_after_canonicalization.to_string())
        # 预期：不应有任何行因标准化失败而被丢弃
        self.assertEqual(len(df_after_canonicalization), 6)

        # --- 模拟第二步：序列净化 ---
        print("\n--- [Step 2] Sequence Purification ---")

        # a. 过滤None值
        df_to_check_seq = df_after_canonicalization.dropna(subset=["Sequence"])
        print("\nDataFrame after dropping None Sequences:")
        print(df_to_check_seq.to_string())
        # 预期：过滤掉第7条
        self.assertEqual(len(df_to_check_seq), 5)

        # b. 验证字符集 (采用严格的21种氨基酸标准)
        VALID_SEQ_CHARS = "ACDEFGHIKLMNPQRSTVWYU"
        invalid_char_pattern = f"[^{VALID_SEQ_CHARS}]"
        seq_series = df_to_check_seq["Sequence"].astype(str).str.upper()
        sequence_mask = ~seq_series.str.contains(
            invalid_char_pattern, regex=True, na=False
        )

        df_after_seq_validation = df_to_check_seq[sequence_mask]
        print("\nDataFrame after Sequence validation:")
        print(df_after_seq_validation.to_string())

        # 预期：过滤掉含'X'(CID 4)和含'B','Z'(CID 6)的记录
        self.assertEqual(len(df_after_seq_validation), 3)

        # --- 最终调用真实函数进行黑盒对比 ---
        print("\n--- [Final Blackbox Test] Calling the real _purify_chunk function ---")

        # 为了测试真实函数，我们需要给它一个干净的输入副本
        input_df_for_blackbox = pd.DataFrame(self.input_data)
        final_output_df = _purify_chunk(input_df_for_blackbox)

        print("\nFinal output from _purify_chunk:")
        print(final_output_df.to_string())

        # 【最终断言】根据我们严格的净化规则，预期只有3条记录能存活
        self.assertEqual(
            len(final_output_df),
            3,
            "The real _purify_chunk function returned an incorrect number of rows.",
        )

        # 验证存活的是否是正确的记录
        expected_cids = {1, 2, 5}
        output_cids = set(final_output_df["PubChem_CID"])
        self.assertSetEqual(output_cids, expected_cids)

        # 验证'C(C)O'是否被正确标准化
        smiles_of_cid_5 = final_output_df[final_output_df["PubChem_CID"] == 5][
            "SMILES"
        ].iloc[0]
        self.assertEqual(smiles_of_cid_5, "CCO")

        print("\n✅ All assertions for purify function passed!")


if __name__ == "__main__":
    unittest.main(verbosity=2)
