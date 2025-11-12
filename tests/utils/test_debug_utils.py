# 文件: tests/utils/test_debug_utils.py (全新)

import unittest

import pandas as pd
from omegaconf import OmegaConf

# 导入我们需要测试的函数和颜色类（如果它在debug.py中）
from helixpipe.utils.debug import validate_authoritative_dti_file

# 模拟一个最小化的 AppConfig
MOCK_CONFIG = OmegaConf.create(
    {
        "runtime": {"verbose": 1, "seed": 42},
        "data_structure": {
            "schema": {
                "internal": {
                    "authoritative_dti": {
                        "molecule_id": "PubChem_CID",
                        "protein_id": "UniProt_ID",
                        "molecule_sequence": "SMILES",
                        "protein_sequence": "Sequence",
                        "label": "Label",
                    }
                }
            }
        },
    }
)


class TestDebugValidators(unittest.TestCase):
    def test_validate_dti_file_with_nones(self):
        """
        测试: 验证函数在SMILES和Sequence列包含None时是否能正常工作。
        """
        print(
            "\n--- Running Test: validate_authoritative_dti_file with None values ---"
        )

        # 1. 准备一个包含 None 值的 DataFrame，模拟 BrendaProcessor 的输出
        test_df = pd.DataFrame(
            {
                "PubChem_CID": [123, 456],
                "UniProt_ID": ["P12345", "Q9Y261"],
                "SMILES": [None, None],  # <-- 关键测试点
                "Sequence": [None, None],  # <-- 关键测试点
                "Label": [1, 1],
                "relation_type": ["substrate", "inhibitor"],
            }
        )

        try:
            # 2. 调用被测函数，断言它不会抛出任何异常
            validate_authoritative_dti_file(MOCK_CONFIG, df=test_df, verbose=1)
            # 如果能顺利执行到这里，说明测试通过
            print("  ✅ Test passed: Function handled None values gracefully.")
        except Exception as e:
            # 3. 如果抛出了任何异常，测试失败
            self.fail(
                f"validate_authoritative_dti_file raised an unexpected exception: {e}"
            )

    def test_validate_dti_file_with_mixed_data(self):
        """
        测试: 验证函数在混合了有效字符串和None值时是否能正常工作。
        """
        print("\n--- Running Test: validate_authoritative_dti_file with mixed data ---")

        test_df = pd.DataFrame(
            {
                "PubChem_CID": [1, 2, 3],
                "UniProt_ID": ["P00001", "P00002", "P00003"],
                "SMILES": ["CCO", None, "InvalidSMILES"],  # <-- 混合数据
                "Sequence": ["MKT", "SEQ-WITH-X", None],  # <-- 混合数据
                "Label": [1, 1, 1],
            }
        )

        # 我们预期SMILES和Sequence检查会失败，但不是因为TypeError
        with self.assertRaises(AssertionError, msg="Should fail on invalid SMILES"):
            validate_authoritative_dti_file(
                MOCK_CONFIG, df=test_df, verbose=0
            )  # verbose=0以避免大量打印

        # 修复Sequence中的错误，再次测试
        test_df.loc[1, "Sequence"] = "MKA"
        with self.assertRaises(
            AssertionError, msg="Should still fail on invalid SMILES"
        ):
            validate_authoritative_dti_file(MOCK_CONFIG, df=test_df, verbose=0)

        # 修复SMILES中的错误，现在应该能通过了
        test_df.loc[2, "SMILES"] = "CCC"
        try:
            validate_authoritative_dti_file(MOCK_CONFIG, df=test_df, verbose=1)
            print("  ✅ Test passed: Function correctly validated mixed data.")
        except Exception as e:
            self.fail(
                f"validate_authoritative_dti_file raised an unexpected exception with cleaned data: {e}"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
