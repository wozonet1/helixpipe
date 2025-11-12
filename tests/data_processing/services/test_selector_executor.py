import unittest

import pandas as pd

# 导入我们需要测试的类和相关的dataclass
from nasnet.configs import EntitySelectorConfig, InteractionSelectorConfig
from nasnet.data_processing.services.selector_executor import SelectorExecutor

# --- 模拟 (Mock) 依赖 ---


class MockIDMapper:
    """一个为 SelectorExecutor 测试量身定做的 Mock IDMapper。"""

    def __init__(self):
        self.meta_data = {
            101: {"type": "drug", "sources": {"bindingdb"}},
            102: {"type": "drug", "sources": {"gtopdb"}},
            201: {"type": "ligand", "sources": {"brenda"}},
            "P01": {"type": "protein", "sources": {"bindingdb", "stringdb"}},
            "P02": {"type": "protein", "sources": {"brenda"}},
            999: None,  # 一个没有元信息的实体
        }

    def get_meta_by_auth_id(self, auth_id):
        return self.meta_data.get(auth_id)

    def is_molecule(self, entity_type):
        return entity_type in ["drug", "ligand"]

    def is_protein(self, entity_type):
        return entity_type == "protein"


class TestSelectorExecutor(unittest.TestCase):
    def setUp(self):
        print("\n" + "=" * 80)
        self.mock_id_mapper = MockIDMapper()
        self.executor = SelectorExecutor(self.mock_id_mapper, verbose=True)
        self.df = pd.DataFrame(
            {
                "source": [101, 102, 201, "P01", 999],
                "target": ["P01", "P02", "P02", "P02", 101],
                "rel_type": [
                    "interacts_with",
                    "interacts_with",
                    "inhibits",
                    "ppi",
                    "unknown",
                ],
            },
            index=["inter_1", "inter_2", "inter_3", "inter_4", "inter_5"],
        )
        self.source_col = "source"
        self.target_col = "target"
        self.relation_col = "rel_type"  # 【新增】

    def test_no_selector(self):
        print("--- Running Test: No Selector (Match All) ---")
        selector = InteractionSelectorConfig()
        mask = self.executor.get_interaction_match_mask(
            self.df, selector, self.source_col, self.target_col, self.relation_col
        )
        self.assertTrue(mask.all())
        print("  ✅ Passed.")

    def test_by_relation_type(self):
        print("--- Running Test: By Relation Type ---")
        selector = InteractionSelectorConfig(
            relation_types=["interacts_with", "inhibits"]
        )
        mask = self.executor.get_interaction_match_mask(
            self.df, selector, self.source_col, self.target_col, self.relation_col
        )
        self.assertEqual(mask.sum(), 3)
        self.assertTrue(mask.loc[["inter_1", "inter_2", "inter_3"]].all())
        self.assertFalse(mask.loc[["inter_4", "inter_5"]].any())
        print("  ✅ Passed.")

    def test_by_source_properties_with_bidirectional(self):
        print("--- Running Test: By Source Properties with Bidirectional Match ---")
        selector = InteractionSelectorConfig(
            source_selector=EntitySelectorConfig(entity_types=["drug"])
        )
        mask = self.executor.get_interaction_match_mask(
            self.df, selector, self.source_col, self.target_col, self.relation_col
        )
        # 应该匹配 inter_1, inter_2 (正向) 和 inter_5 (反向: target是drug)
        self.assertEqual(mask.sum(), 3)
        self.assertTrue(mask.loc[["inter_1", "inter_2", "inter_5"]].all())
        print("  ✅ Passed.")

    def test_complex_bidirectional_match(self):
        print("--- Running Test: Complex Bidirectional Match ---")
        selector = InteractionSelectorConfig(
            source_selector=EntitySelectorConfig(from_sources=["brenda"]),
            target_selector=EntitySelectorConfig(meta_types=["protein"]),
        )
        mask = self.executor.get_interaction_match_mask(
            self.df, selector, self.source_col, self.target_col, self.relation_col
        )
        # 应该匹配 inter_3 (正向) 和 inter_4 (反向)
        self.assertEqual(mask.sum(), 2)
        self.assertTrue(mask.loc[["inter_3", "inter_4"]].all())
        print("  ✅ Passed.")

    def test_strict_handling_for_unknown_meta(self):
        """【最终修正断言版】"""
        print(
            "\n" + "=" * 80 + "\n--- Running Final Corrected Test: Strict Handling ---"
        )

        # 场景a: 选择器对未知实体提出了要求
        selector_a = InteractionSelectorConfig(
            source_selector=EntitySelectorConfig(meta_types=["protein"])
        )
        mask_a = self.executor.get_interaction_match_mask(
            self.df, selector_a, self.source_col, self.target_col, self.relation_col
        )

        # 【核心修正】: 我们的严谨分析表明，正确的预期结果是 4！
        # inter_1, inter_2, inter_3 通过反向匹配
        # inter_4 通过正向匹配
        # inter_5 (含999) 被严格过滤
        self.assertEqual(mask_a.sum(), 4)
        self.assertTrue(mask_a.loc[["inter_1", "inter_2", "inter_3", "inter_4"]].all())
        self.assertFalse(mask_a["inter_5"])
        print(
            "  ✅ Passed (a): Correctly rejects unknown meta AND performs correct bidirectional match."
        )

        # 场景b: 保持不变
        selector_b = InteractionSelectorConfig(
            source_selector=EntitySelectorConfig(entity_types=["drug"]),
            target_selector=None,
        )
        mask_b = self.executor.get_interaction_match_mask(
            self.df, selector_b, self.source_col, self.target_col, self.relation_col
        )
        self.assertEqual(mask_b.sum(), 3)
        self.assertTrue(mask_b.loc[["inter_1", "inter_2", "inter_5"]].all())
        print(
            "  ✅ Passed (b): Correctly handles None selector in bidirectional match."
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
