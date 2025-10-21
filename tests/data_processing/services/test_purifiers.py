# 文件: tests/test_purifiers.py (最终权威版)

import unittest

import pandas as pd
from rdkit import RDLogger

# 导入我们需要直接测试的函数
from nasnet.data_processing.services.purifiers import _purify_chunk


class TestPurifiers(unittest.TestCase):
    def setUp(self):
        """为测试准备一个包含所有边界情况的输入DataFrame。"""
        RDLogger.logger().setLevel(RDLogger.CRITICAL)
        print("\n" + "=" * 80)

        self.input_data = {
            # --- 分组1: 完全有效的记录 ---
            "PubChem_CID": [
                101,  # 1. 完全有效
                102,  # 2. 完全有效
                103,  # 3. 有效, 但SMILES是非规范形式
            ],
            "SMILES": [
                "CCO",  # 1.
                "CCC",  # 2.
                "C(C)O",  # 3.
            ],
            "Sequence": [
                "MKTFY",  # 1.
                "ASDFG",  # 2.
                "DIFFERENTSEQ",  # 3.
            ],
            "UniProt_ID": [
                "P12345",  # 1.
                "Q9Y261",  # 2.
                "A0A024R1R8",  # 3.
            ],
            "Description": [
                "Should PASS all checks.",
                "Should PASS all checks.",
                "Should PASS all checks, and SMILES will be canonicalized.",
            ],
        }

        # --- 分组2: CID 无效的记录 ---
        self.input_data["PubChem_CID"] += ["abc", -10, 3.14, None, " ", 0]
        self.input_data["SMILES"] += ["C"] * 6
        self.input_data["Sequence"] += ["A"] * 6
        self.input_data["UniProt_ID"] += ["P00001"] * 6
        self.input_data["Description"] += [
            "Should be filtered by invalid CID (string).",
            "Should be filtered by invalid CID (negative).",
            "Should be filtered by invalid CID (float).",
            "Should be filtered by invalid CID (None).",
            "Should be filtered by invalid CID (whitespace).",
            "Should be filtered by invalid CID (zero).",
        ]

        # --- 分组3: SMILES 无效的记录 ---
        self.input_data["PubChem_CID"] += [201, 202, 203]
        self.input_data["SMILES"] += ["INVALID-SMILES", "", None]
        self.input_data["Sequence"] += ["A"] * 3
        self.input_data["UniProt_ID"] += ["P00002"] * 3
        self.input_data["Description"] += [
            "Should be filtered by invalid SMILES (bad format).",
            "Should be filtered by invalid SMILES (empty string).",
            "Should be filtered by invalid SMILES (None).",
        ]

        # --- 分组4: Sequence 无效的记录 ---
        self.input_data["PubChem_CID"] += [301, 302, 303]
        self.input_data["SMILES"] += ["C"] * 3
        self.input_data["Sequence"] += ["SEQ-WITH-X", "SEQ-WITH-123", None]
        self.input_data["UniProt_ID"] += ["P00003"] * 3
        self.input_data["Description"] += [
            "Should be filtered by invalid Sequence (contains X).",
            "Should be filtered by invalid Sequence (contains numbers).",
            "Should be filtered by invalid Sequence (None).",
        ]

        # --- 分组5: UniProt ID 无效的记录 ---
        self.input_data["PubChem_CID"] += [401, 402, 403]
        self.input_data["SMILES"] += ["C"] * 3
        self.input_data["Sequence"] += ["A"] * 3
        self.input_data["UniProt_ID"] += ["P1234", "INVALID-ID", None]
        self.input_data["Description"] += [
            "Should be filtered by invalid UniProt ID (too short).",
            "Should be filtered by invalid UniProt ID (bad format).",
            "Should be filtered by invalid UniProt ID (None).",
        ]

        self.input_df = pd.DataFrame(self.input_data)

    def test_purify_chunk_comprehensively(self):
        """
        对 _purify_chunk 函数进行全面的、黑盒式的输入/输出验证。
        """
        print("--- Running Test: _purify_chunk comprehensively ---")
        print("\n--- [Step 0] Comprehensive Input DataFrame ---")
        print(f"Total rows to process: {len(self.input_df)}")
        # print(self.input_df.to_string()) # (可选) 打印完整输入

        # 调用我们正在测试的真实函数
        output_df = _purify_chunk(self.input_df)

        print("\n--- [Final Step] Output DataFrame from _purify_chunk ---")
        print(output_df.to_string())

        # --- 最终断言 ---

        # 1. 断言最终的数量
        #    根据我们的设计，只有分组1中的3条记录应该存活下来。
        self.assertEqual(
            len(output_df), 3, "Final count of purified data is incorrect."
        )

        # 2. 断言存活记录的身份
        #    检查留下的记录的CID是否就是我们期望的那3个。
        expected_cids = {101, 102, 103}
        output_cids = set(output_df["PubChem_CID"])
        self.assertSetEqual(
            output_cids, expected_cids, "The CIDs in the output are not as expected."
        )

        # 3. 断言SMILES标准化的效果
        #    检查CID=103 (原始SMILES是'C(C)O') 的记录，其SMILES是否被正确标准化为'CCO'。
        smiles_of_cid_103 = output_df[output_df["PubChem_CID"] == 103]["SMILES"].iloc[0]
        self.assertEqual(
            smiles_of_cid_103, "CCO", "SMILES canonicalization failed for 'C(C)O'."
        )

        print("\n✅ All comprehensive assertions for _purify_chunk passed!")


if __name__ == "__main__":
    unittest.main(verbosity=2)
