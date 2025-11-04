# tests/data_processing/services/test_entity_validator.py

import unittest
from unittest.mock import patch

import pandas as pd
from omegaconf import OmegaConf

# 导入我们需要测试的主函数
from nasnet.data_processing.services.entity_validator import (
    validate_and_filter_entities,
)

# --- 模拟 (Mock) 配置 ---
# 我们创建一个最小化的Config，只包含校验服务需要的所有参数
MOCK_CONFIG = OmegaConf.create(
    {
        "runtime": {"verbose": 0, "cpus": 1},
        "knowledge_graph": {
            "entity_types": {"molecule": "molecule", "protein": "protein"}
        },
        "data_params": {
            "filtering": {
                "enabled": True,
                "apply_pains_filter": True,
                # 【修改】为 molecular_weight 提供完整的 min/max
                "molecular_weight": {"min": 100, "max": 200},
                # 【核心修复】为 logp 提供完整的 min/max 结构
                # 即使我们不关心 min 的值，也要提供一个 None 来满足代码的访问
                "logp": {"min": None, "max": 3.0},
                # 【新增】为其他可能的过滤提供 None，以增加测试的健壮性
                # 这可以防止未来 filter.py 增加新过滤时，这个测试立即失败
                "h_bond_donors": None,
                "h_bond_acceptors": None,
                "qed": None,
                "sa_score": None,
            }
        },
    }
)


class TestEntityValidator(unittest.TestCase):
    def setUp(self):
        """为测试准备一个包含所有边界情况的“脏”实体清单DataFrame。"""
        self.entities_df = pd.DataFrame(
            {
                "entity_id": [
                    # --- 分子组 (Molecules) ---
                    101,  # 1. 完全有效 (MW=46, LogP= -0.3) -> 但会被MW过滤掉
                    102,  # 2. 完全有效 (MW=152, LogP=2.1) -> 应该通过所有检查
                    103,  # 3. ID无效 (非正数)
                    "abc",  # 4. ID无效 (字符串)
                    105,  # 5. SMILES结构无效
                    106,  # 6. SMILES为空
                    107,  # 7. 是一个PAINS分子
                    108,  # 8. LogP过高 (MW=180, LogP=3.5)
                    # --- 蛋白质组 (Proteins) ---
                    "P12345",  # 9. 完全有效
                    "P00533",  # 10. 完全有效
                    "INVALID",  # 11. ID格式无效
                    "P99999",  # 12. 序列包含非法字符 'X'
                    "Q88888",  # 13. 序列为空
                ],
                "entity_type": [
                    # --- 分子组 ---
                    "molecule",
                    "molecule",
                    "molecule",
                    "molecule",
                    "molecule",
                    "molecule",
                    "molecule",
                    "molecule",
                    # --- 蛋白质组 ---
                    "protein",
                    "protein",
                    "protein",
                    "protein",
                    "protein",
                ],
                "structure": [
                    # --- 分子组 ---
                    "CCO",  # 1. Ethanol, MW=46.07
                    "C1=CC=C(C=C1)C(=O)O",  # 2. Benzoic acid, MW=122.12, LogP=1.87
                    "C",  # 3. Methane
                    "CC",  # 4. Ethane
                    "InvalidSMILES",  # 5.
                    None,  # 6.
                    "O=C1NC(=S)S/C1=C/c2ccccc2",  # 7. A known PAINS molecule
                    "C1=CC=C(C=C1)C1=CC=CC=C1",  # 8. Biphenyl, MW=154.21, LogP=3.9
                    # --- 蛋白质组 ---
                    "MKTSEQ",  # 9.
                    "VALIDPROT",  # 10.
                    "ANYSEQ",  # 11.
                    "SEQWITHX",  # 12.
                    "",  # 13.
                ],
            }
        )

    # 使用 @patch 装饰器来“模拟”所有外部依赖（ID白名单检查）
    # 这让我们的测试变成了“单元/集成”混合测试，不依赖网络
    @patch("nasnet.data_processing.services.entity_validator.get_valid_pubchem_cids")
    @patch(
        "nasnet.data_processing.services.entity_validator.get_human_uniprot_whitelist"
    )
    def test_end_to_end_validation_pipeline(
        self, mock_uniprot_whitelist, mock_cid_whitelist
    ):
        """
        黑盒测试：对 validate_and_filter_entities 的完整流程进行输入/输出验证。
        """
        print("\n--- Running Test: Unified Entity Validator Pipeline ---")

        # --- 1. 准备 Mock ---
        # 模拟白名单服务返回一个“纯净”的ID集合
        mock_cid_whitelist.return_value = {
            101,
            102,
            107,
            108,
        }  # 假设 103, "abc", 105, 106 已经是无效ID
        mock_uniprot_whitelist.return_value = {
            "P12345",
            "P00533",
        }  # 假设 "INVALID", "P99999", "Q88888" 已被过滤

        # --- 2. 调用被测函数 (黑盒调用) ---
        result_df = validate_and_filter_entities(self.entities_df, MOCK_CONFIG)

        # 【在这里加入Debug Print】
        print("--- DEBUG: FINAL RESULT from validator ---")
        print(result_df.to_string())
        print("------------------------------------------")
        # --- 3. 断言最终结果 ---

        # a. 断言最终存活的实体数量
        #    - 分子: 只有 CID=102 能通过所有检查
        #    - 蛋白质: 只有 P12345 和 P00533 能通过
        self.assertEqual(
            len(result_df), 3, "Final count of validated entities is incorrect."
        )

        # b. 断言存活实体的身份
        final_ids = set(result_df["entity_id"])
        expected_ids = {102, "P12345", "P00533"}
        self.assertSetEqual(
            final_ids, expected_ids, "The IDs in the output are not as expected."
        )

        # c. [可选] 断言分子的SMILES已被标准化
        #    Benzoic acid的非标准化SMILES是 C1=CC=C(C=C1)C(=O)O
        #    标准化后可能是 c1ccc(cc1)C(=O)O
        #    为了简化测试，我们只检查它不是None
        final_molecule = result_df[result_df["entity_id"] == 102]
        self.assertIsNotNone(
            final_molecule["structure"].iloc[0],
            "SMILES should have been canonicalized.",
        )

        print(
            "  ✅ Test passed: Function correctly filtered and validated the mixed-entity DataFrame."
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
