# tests/data_processing/services/test_filter_service.py

import unittest

import pandas as pd
from omegaconf import OmegaConf

# 导入我们需要独立测试的函数
from helixpipe.data_processing.services.filter import filter_molecules_by_properties

# --- 模拟 (Mock) 配置 ---
# 这个配置与 test_entity_validator 中的完全一致
MOCK_CONFIG = OmegaConf.create(
    {
        "runtime": {"verbose": 0, "cpus": 1},
        "data_params": {
            "filtering": {
                "enabled": True,
                "apply_pains_filter": True,
                "molecular_weight": {"min": 100, "max": 200},
                "logp": {"min": None, "max": 3.0},
                "h_bond_donors": None,
                "h_bond_acceptors": None,
                "qed": None,
                "sa_score": None,
            }
        },
    }
)


class TestFilterService(unittest.TestCase):
    def test_filter_molecules_by_properties_logic(self):
        """
        白盒测试：精确验证 filter_molecules_by_properties 的过滤逻辑。
        """
        print("\n--- Running Test: filter_molecules_by_properties (Whitebox) ---")

        # 1. 准备一个包含所有关键测试用例的输入Series
        #    我们使用一个有意义的字符串索引，便于调试
        input_smiles = pd.Series(
            [
                "CCO",  # case_1: MW太小 (46.07)
                "C1=CC=C(C=C1)C(=O)O",  # case_2: 完全通过 (MW=122.12, LogP=1.87)
                "InvalidSMILES",  # case_3: 结构无效
                "O=C1NC(=S)S/C1=C/c2ccccc2",  # case_4: 是PAINS分子 (MW=139.11)
                "C1=CC=C(C=C1)C1=CC=CC=C1",  # case_5: LogP太高 (MW=154.21, LogP=3.9)
                None,  # case_6: 输入为None
            ],
            index=[
                "case_1_ethanol",
                "case_2_benzoic_acid",
                "case_3_invalid",
                "case_4_pains",
                "case_5_biphenyl",
                "case_6_none",
            ],
        )

        # 2. 调用被测函数
        result_mask = filter_molecules_by_properties(input_smiles, MOCK_CONFIG)

        # 3. 逐一断言每个用例的结果

        # a. 检查返回值的类型和索引
        self.assertIsInstance(
            result_mask, pd.Series, "Return value should be a Pandas Series."
        )
        self.assertTrue(result_mask.dtype == bool, "Series dtype should be boolean.")
        self.assertListEqual(
            result_mask.index.tolist(),
            input_smiles.index.tolist(),
            "Index of the mask should match the input index.",
        )

        # b. 精确断言每个case的过滤结果
        self.assertFalse(
            result_mask["case_1_ethanol"], "Case 1 (MW too low) should be filtered."
        )
        self.assertTrue(
            result_mask["case_2_benzoic_acid"], "Case 2 (Valid) should pass."
        )
        self.assertFalse(
            result_mask["case_3_invalid"], "Case 3 (Invalid SMILES) should be filtered."
        )
        self.assertFalse(
            result_mask["case_4_pains"], "Case 4 (PAINS) should be filtered."
        )
        self.assertFalse(
            result_mask["case_5_biphenyl"], "Case 5 (LogP too high) should be filtered."
        )
        self.assertFalse(
            result_mask["case_6_none"], "Case 6 (None input) should be filtered."
        )

        # c. 断言最终通过的总数
        self.assertEqual(
            result_mask.sum(), 1, "Only one molecule (case 2) should pass all filters."
        )

        print(
            "  ✅ Test passed: filter_molecules_by_properties behaves exactly as expected."
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
