# 文件: tests/data_processing/services/test_interaction_store.py (全新)

import unittest
from unittest.mock import MagicMock

import pandas as pd
from omegaconf import OmegaConf

# 假设您的项目结构允许这样导入
from nasnet.data_processing.services.id_mapper import IDMapper
from nasnet.data_processing.services.interaction_store import InteractionStore

# --- 准备测试数据和模拟对象 ---

# 模拟一个最小化的Config，只包含必要的schema
MOCK_CONFIG = OmegaConf.create(
    {
        "data_structure": {
            "schema": {
                "internal": {
                    "authoritative_dti": {
                        "molecule_id": "PubChem_CID",
                        "protein_id": "UniProt_ID",
                        "label": "Label",
                    }
                }
            }
        }
    }
)

# 模拟来自不同Processor的交互DataFrame
DF1 = pd.DataFrame(
    {"PubChem_CID": [101, 102], "UniProt_ID": ["P01", "P02"], "Label": [1, 1]}
)

DF2 = pd.DataFrame(
    {
        "PubChem_CID": [103, 104],  # 包含一个将被过滤掉的ID
        "UniProt_ID": ["P01", "P03"],  # 包含一个将被过滤掉的ID
        "Label": [1, 0],  # 包含一个负样本
    }
)


class TestInteractionStore(unittest.TestCase):
    def setUp(self):
        """为每个测试创建一个新的InteractionStore实例。"""
        self.interaction_store = InteractionStore([DF1, DF2], MOCK_CONFIG)

    def test_initialization(self):
        """测试: InteractionStore是否能被正确初始化并合并数据。"""
        print("\n--- Running Test: InteractionStore Initialization ---")

        # 初始应包含 DF1 和 DF2 的所有行 (2 + 2 = 4)
        self.assertEqual(len(self.interaction_store._interactions_df), 4)
        print("  ✅ Correctly initialized with 4 total interactions.")

    def test_filter_by_entities(self):
        """测试: filter_by_entities 方法是否能精确过滤交互。"""
        print("\n--- Running Test: InteractionStore filter_by_entities ---")

        # 定义一个“纯净”的实体ID集合
        valid_cids = {101, 102, 103}  # CID 104 是无效的
        valid_pids = {"P01", "P02"}  # PID P03 是无效的

        self.interaction_store.filter_by_entities(valid_cids, valid_pids)

        filtered_df = self.interaction_store.get_all_interactions_df()

        # 预期结果:
        # (101, P01) -> 保留
        # (102, P02) -> 保留
        # (103, P01) -> 保留
        # (104, P03) -> 移除 (因为104和P03都不在有效集合中)
        self.assertEqual(
            len(filtered_df), 3, "Incorrect number of interactions after filtering."
        )

        # 进一步检查留下的交互是否正确
        remaining_pairs = set(
            zip(filtered_df["PubChem_CID"], filtered_df["UniProt_ID"])
        )
        expected_pairs = {(101, "P01"), (102, "P02"), (103, "P01")}
        self.assertSetEqual(remaining_pairs, expected_pairs)
        print("  ✅ Correctly filtered interactions based on valid entity sets.")

    def test_get_mapped_positive_pairs(self):
        """测试: 能否正确地将正样本交互映射为逻辑ID。"""
        print("\n--- Running Test: InteractionStore get_mapped_positive_pairs ---")

        # 1. 创建一个模拟的、已经“最终化”的IDMapper
        mock_id_mapper = MagicMock(spec=IDMapper)
        mock_id_mapper.is_finalized = True
        mock_id_mapper.cid_to_id = {101: 0, 102: 1, 103: 2}  # 模拟逻辑ID映射
        mock_id_mapper.uniprot_to_id = {"P01": 10, "P02": 11}

        # 2. 调用被测方法
        # 内部的df包含4条记录：(101,P01,1), (102,P02,1), (103,P01,1), (104,P03,0)
        # 只有前3条是正样本
        pairs_list, pairs_set = self.interaction_store.get_mapped_positive_pairs(
            mock_id_mapper
        )

        # 3. 断言结果
        self.assertEqual(len(pairs_list), 3)
        self.assertEqual(len(pairs_set), 3)

        # 检查转换后的逻辑ID对是否正确
        expected_pairs = {
            (0, 10),
            (1, 11),
            (2, 10),
        }  # (101,P01)->(0,10), (102,P02)->(1,11), (103,P01)->(2,10)
        self.assertSetEqual(pairs_set, expected_pairs)
        print("  ✅ Correctly mapped positive interaction pairs to logic IDs.")

    def test_get_mapped_positive_pairs_unfinalized_mapper(self):
        """测试: 当传入一个未最终化的IDMapper时，是否会按预期抛出异常。"""
        print("\n--- Running Test: InteractionStore handles unfinalized mapper ---")

        unfinalized_mapper = MagicMock(spec=IDMapper)
        unfinalized_mapper.is_finalized = False

        # 使用 assertRaises 来捕获并验证预期的异常
        with self.assertRaises(RuntimeError):
            self.interaction_store.get_mapped_positive_pairs(unfinalized_mapper)

        print("  ✅ Correctly raised RuntimeError for unfinalized IDMapper.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
