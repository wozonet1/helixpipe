# 文件: tests/data_processing/services/test_filter_service.py (全新)

import unittest

import numpy as np
import pandas as pd

# 导入我们需要测试的私有函数
from nasnet.data_processing.services.filter import _calculate_descriptors_for_chunk


class TestFilterServiceInternals(unittest.TestCase):
    def test_calculate_descriptors_handles_dirty_data(self):
        """
        测试: _calculate_descriptors_for_chunk 是否能优雅地处理包含
              NaN, None, 和其他非字符串类型的输入。
        """
        print(
            "\n--- Running Test: _calculate_descriptors_for_chunk handles dirty data ---"
        )

        # 1. 准备一个包含各种“脏数据”的输入Series
        dirty_smiles = pd.Series(
            [
                "CCO",  # 有效
                np.nan,  # NaN值 (浮点数)
                "CCC",  # 有效
                None,  # None值
                "Invalid",  # 无效SMILES
                123,  # 错误类型：整数
            ]
        )

        try:
            # 2. 调用被测函数，断言它不应抛出任何异常
            result_df = _calculate_descriptors_for_chunk(dirty_smiles)

            # 3. 验证输出的正确性
            # a. 我们预期函数只处理了3个有效/无效的字符串: "CCO", "CCC", "Invalid"
            self.assertEqual(
                len(result_df), 3, "Function should process only string inputs."
            )

            # b. "CCO" 和 "CCC" 应该能成功计算出描述符 (MW不为NaN)
            #    "Invalid" 应该返回一行全是NaN的记录 (除了 is_pains)
            self.assertFalse(
                result_df.iloc[0]["MW"], "Descriptor for 'CCO' should be calculated."
            )
            self.assertFalse(
                result_df.iloc[1]["MW"], "Descriptor for 'CCC' should be calculated."
            )
            self.assertTrue(
                pd.isna(result_df.iloc[2]["MW"]),
                "Descriptor for 'Invalid' should be NaN.",
            )

            print(
                "  ✅ Test passed: Function correctly handled mixed-type input without crashing."
            )

        except Exception as e:
            self.fail(
                f"_calculate_descriptors_for_chunk raised an unexpected exception: {e}"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
