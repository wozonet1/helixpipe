import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
from omegaconf import OmegaConf

from helixpipe.configs import register_all_schemas
from helixpipe.data_processing.services.label_generator import SupervisionFileManager

# 导入所有需要的模块
from helixpipe.typing import AppConfig

# --- 模拟依赖 (Mocks) ---


class MockInteractionStore:
    def __init__(self, df):
        self.dataframe = df
        # 模拟schema访问，以便被测代码可以访问 self.store._schema.source_id
        schema_mock = MagicMock()
        schema_mock.source_id = "s_id"
        schema_mock.target_id = "t_id"
        self._schema = schema_mock

    def get_mapped_positive_pairs(self, id_mapper):
        df_mapped = self.dataframe.copy()
        df_mapped["s_id"] = df_mapped["s_id"].map(id_mapper.auth_id_to_logic_id_map)
        df_mapped["t_id"] = df_mapped["t_id"].map(id_mapper.auth_id_to_logic_id_map)
        # 返回一个包含关系类型的元组列表
        return [(row.s_id, row.t_id, "rel") for row in df_mapped.itertuples()]


class MockIDMapper:
    def __init__(self):
        self.auth_id_to_logic_id_map = {
            "d1": 0,
            "d2": 1,
            "d3": 2,
            "p1": 10,
            "p2": 11,
            "p3": 12,
        }


class MockSelectorExecutor:
    def __init__(self, id_mapper):
        self.id_mapper = id_mapper

    def select_entities(self, selector):
        if selector and selector.entity_types == ["drug"]:
            return {"d1", "d2", "d3"}
        if selector and selector.entity_types == ["protein"]:
            return {"p1", "p2", "p3"}
        return set()


register_all_schemas()
MOCK_CONFIG: AppConfig = OmegaConf.create(
    {
        "runtime": {"verbose": 0, "seed": 42},
        "training": {
            "coldstart": {
                "evaluation_scope": {
                    "source_selector": {"entity_types": ["drug"]},
                    "target_selector": {"entity_types": ["protein"]},
                }
            }
        },
        "data_structure": {
            "schema": {
                "internal": {
                    "labeled_edges_output": {
                        "source_node": "source",
                        "target_node": "target",
                        "label": "label",
                    },
                    "canonical_interaction": {"source_id": "s_id", "target_id": "t_id"},
                }
            }
        },
    }
)


class TestSupervisionFileManager(unittest.TestCase):
    @patch("helixpipe.data_processing.services.label_generator.get_path")
    def setUp(self, mock_get_path):
        """为所有测试准备一个通用的 manager 实例。"""
        print("\n" + "=" * 80)
        self.mock_id_mapper = MockIDMapper()
        self.mock_executor = MockSelectorExecutor(self.mock_id_mapper)
        self.global_positive_set = {("d1", "p1"), ("d2", "p2")}
        mock_path_factory = MagicMock()
        mock_path_factory.return_value = MagicMock(name="fake_path_object")
        mock_get_path.return_value = mock_path_factory
        # 创建一个 manager 实例，它将在多个测试中被使用
        self.manager = SupervisionFileManager(
            fold_idx=1,
            config=MOCK_CONFIG,
            id_mapper=self.mock_id_mapper,
            executor=self.mock_executor,
            global_positive_set=self.global_positive_set,
        )

    def test_prepare_train_df(self):
        """测试点1: _prepare_train_df 能否创建正确的训练DataFrame。"""
        print("--- Running Test: _prepare_train_df ---")

        # 准备输入
        train_store = MockInteractionStore(
            pd.DataFrame({"s_id": ["d1"], "t_id": ["p1"]})
        )

        # 直接调用核心逻辑方法
        train_df = self.manager._prepare_train_df(train_store)

        # 直接断言返回值
        self.assertIsInstance(train_df, pd.DataFrame)
        self.assertEqual(train_df.shape, (1, 2))
        self.assertNotIn("label", train_df.columns)

        # 验证ID映射
        self.assertEqual(train_df["source"].iloc[0], 0)  # d1 -> 0
        self.assertEqual(train_df["target"].iloc[0], 10)  # p1 -> 10
        print("  ✅ Passed.")

    def test_prepare_test_df(self):
        """测试点2: _prepare_test_df 能否创建正确的、包含负样本的测试DataFrame。"""
        print("--- Running Test: _prepare_test_df ---")

        # 准备输入
        test_store = MockInteractionStore(
            pd.DataFrame({"s_id": ["d2"], "t_id": ["p2"]})
        )

        # 直接调用核心逻辑方法
        test_df = self.manager._prepare_test_df(test_store)

        # 断言返回值
        self.assertIsInstance(test_df, pd.DataFrame)
        self.assertEqual(test_df.shape, (2, 3))  # 1个正样本 + 1个负样本
        self.assertIn("label", test_df.columns)
        self.assertEqual(test_df["label"].sum(), 1)  # 应该只有一个标签为1的行

        # 验证正样本内容
        pos_sample = test_df[test_df["label"] == 1].iloc[0]
        self.assertEqual(pos_sample["source"], 1)  # d2 -> 1
        self.assertEqual(pos_sample["target"], 11)  # p2 -> 11

        # 验证负样本内容
        neg_sample = test_df[test_df["label"] == 0].iloc[0]
        self.assertIn(neg_sample["source"], [0, 1, 2])  # 逻辑ID for d1, d2, d3
        self.assertIn(neg_sample["target"], [10, 11, 12])  # 逻辑ID for p1, p2, p3
        print("  ✅ Passed.")

    def test_prepare_df_with_empty_store(self):
        """测试点3: 验证当输入store为空时，方法能否健壮地返回空DataFrame。"""
        print("--- Running Test: Handling Empty Stores ---")

        empty_store = MockInteractionStore(pd.DataFrame())

        # 测试训练方法
        train_df_empty = self.manager._prepare_train_df(empty_store)
        self.assertIsInstance(train_df_empty, pd.DataFrame)
        self.assertTrue(train_df_empty.empty)
        self.assertIn("source", train_df_empty.columns)  # 检查表头是否正确

        # 测试测试方法
        test_df_empty = self.manager._prepare_test_df(empty_store)
        self.assertIsInstance(test_df_empty, pd.DataFrame)
        self.assertTrue(test_df_empty.empty)
        self.assertIn("label", test_df_empty.columns)  # 检查表头是否正确
        print("  ✅ Passed.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
