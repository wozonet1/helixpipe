import unittest

import pandas as pd

# 导入我们需要测试的类和相关的dataclass
from helixpipe.configs import EntitySelectorConfig, InteractionSelectorConfig
from helixpipe.data_processing.services.selector_executor import SelectorExecutor

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

    def get_all_final_ids(self):
        return list(self.meta_data.keys())

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

    def test_select_entities(self):
        """测试点: SelectorExecutor.select_entities 方法。"""
        print("--- Running Test: select_entities ---")

        # 场景a: 按 entity_types 筛选
        selector_a = EntitySelectorConfig(entity_types=["drug", "ligand"])
        result_a = self.executor.select_entities(selector_a)
        # 预期: 101, 102, 201
        self.assertSetEqual(result_a, {101, 102, 201})
        print("  ✅ Passed (a): Correctly selects by entity_types.")

        # 场景b: 按 meta_types 筛选
        selector_b = EntitySelectorConfig(meta_types=["molecule"])
        result_b = self.executor.select_entities(selector_b)
        # 预期: 101, 102, 201 (drug 和 ligand 都属于 molecule)
        self.assertSetEqual(result_b, {101, 102, 201})
        print("  ✅ Passed (b): Correctly selects by meta_types.")

        # 场景c: 按 from_sources 筛选
        selector_c = EntitySelectorConfig(from_sources=["brenda", "stringdb"])
        result_c = self.executor.select_entities(selector_c)
        # 预期: 201 (来自brenda), P01 (来自stringdb), P02 (来自brenda)
        self.assertSetEqual(result_c, {201, "P01", "P02"})
        print("  ✅ Passed (c): Correctly selects by from_sources.")

        # 场景d: 组合筛选
        selector_d = EntitySelectorConfig(
            entity_types=["protein"], from_sources=["bindingdb"]
        )
        result_d = self.executor.select_entities(selector_d)
        # 预期: 只有 P01 既是 protein 又来自 bindingdb
        self.assertSetEqual(result_d, {"P01"})
        print("  ✅ Passed (d): Correctly handles combined selectors.")

        # 场景e: 空选择器，应返回所有
        selector_e = EntitySelectorConfig()
        result_e = self.executor.select_entities(selector_e)
        # 预期: 所有实体 (包括 999，因为严格模式只在有要求时才过滤)
        self.assertSetEqual(result_e, set(self.mock_id_mapper.get_all_final_ids()))
        print("  ✅ Passed (e): Correctly returns all entities for an empty selector.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
