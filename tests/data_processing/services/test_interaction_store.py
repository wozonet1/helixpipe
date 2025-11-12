import unittest

import pandas as pd
from omegaconf import OmegaConf

# 导入我们需要测试的类和相关的dataclass
from nasnet.configs import (
    AppConfig,
    EntitySelectorConfig,
    InteractionSelectorConfig,
    register_all_schemas,
)
from nasnet.data_processing.services.interaction_store import InteractionStore

# --- 模拟 (Mock) 依赖 ---


# 1. 模拟一个已经最终化的 IDMapper
class MockFinalizedIDMapper:
    def __init__(self):
        self.auth_id_to_logic_id_map = {
            101: 0,  # drug from bindingdb
            102: 1,  # drug from bindingdb
            201: 2,  # ligand from brenda
            "P01": 3,  # protein from bindingdb
            "P02": 4,  # protein from brenda
            "P03": 5,  # protein from stringdb
        }
        self.meta_data = {
            101: {"type": "drug", "sources": {"bindingdb"}},
            102: {"type": "drug", "sources": {"bindingdb"}},
            201: {"type": "ligand", "sources": {"brenda"}},
            "P01": {"type": "protein", "sources": {"bindingdb"}},
            "P02": {"type": "protein", "sources": {"brenda"}},
            "P03": {"type": "protein", "sources": {"stringdb"}},
        }

    def get_meta_by_auth_id(self, auth_id):
        return self.meta_data.get(auth_id)

    def is_molecule(self, entity_type):
        return entity_type in ["drug", "ligand"]

    def is_protein(self, entity_type):
        return entity_type == "protein"


# 2. 模拟一个最小化的 AppConfig
# 我们需要先注册schemas，这样OmegaConf才能正确创建嵌套对象
register_all_schemas()

MOCK_CONFIG: AppConfig = OmegaConf.create(
    {
        "runtime": {"verbose": 0},
        "knowledge_graph": {
            # entity_meta 用于 is_molecule/is_protein 判断
            "entity_meta": {
                "drug": {"metatype": "molecule"},
                "ligand": {"metatype": "molecule"},
                "protein": {"metatype": "protein"},
            }
        },
        "data_structure": {
            "schema": {
                "internal": {
                    "canonical_interaction": {
                        "source_id": "source_id",
                        "source_type": "source_type",
                        "target_id": "target_id",
                        "target_type": "target_type",
                        "relation_type": "relation_type",
                    }
                }
            }
        },
    }
)


class TestInteractionStore(unittest.TestCase):
    def setUp(self):
        """为所有测试准备一个通用的 InteractionStore 实例。"""
        print("\n" + "=" * 80)
        # 准备模拟的 processor_outputs
        self.processor_outputs = {
            "bindingdb": pd.DataFrame(
                {
                    "source_id": [101, 102],
                    "source_type": ["drug", "drug"],
                    "target_id": ["P01", "P01"],
                    "target_type": ["protein", "protein"],
                    "relation_type": ["interacts_with", "interacts_with"],
                }
            ),
            "brenda": pd.DataFrame(
                {
                    "source_id": [201],
                    "source_type": ["ligand"],
                    "target_id": ["P02"],
                    "target_type": ["protein"],
                    "relation_type": ["inhibits"],
                }
            ),
            # 添加一个包含无效实体(999)的交互，用于测试过滤
            "stringdb": pd.DataFrame(
                {
                    "source_id": ["P01", 999],
                    "source_type": ["protein", "protein"],
                    "target_id": ["P03", "P01"],
                    "target_type": ["protein", "protein"],
                    "relation_type": ["ppi", "ppi"],
                }
            ),
        }
        self.store = InteractionStore(self.processor_outputs, MOCK_CONFIG)
        self.mock_id_mapper = MockFinalizedIDMapper()

    def test_initialization_and_len(self):
        """测试点1: 初始化是否能正确聚合所有交互。"""
        print("--- Running Test: Initialization ---")
        # bindingdb(2) + brenda(1) + stringdb(2) = 5
        self.assertEqual(len(self.store), 5)
        # 验证内部DataFrame是否包含了来源标签
        internal_df = self.store.dataframe
        self.assertIn("__source_dataset__", internal_df.columns)
        self.assertSetEqual(
            set(internal_df["__source_dataset__"].unique()),
            {"bindingdb", "brenda", "stringdb"},
        )
        print("  ✅ Passed.")

    def test_get_all_entity_auth_ids(self):
        """测试点2: 能否正确提取所有唯一的权威ID。"""
        print("--- Running Test: Get All Entity IDs ---")
        expected_ids = {101, 102, 201, "P01", "P02", "P03", 999}
        actual_ids = self.store.get_all_entity_auth_ids()
        self.assertSetEqual(actual_ids, expected_ids)
        print("  ✅ Passed.")

    def test_filter_by_entities(self):
        """测试点3: 过滤功能是否能移除包含无效实体的交互。"""
        print("--- Running Test: Filter by Entities ---")
        valid_ids = {101, 102, 201, "P01", "P02", "P03"}  # 999 是无效ID
        pure_store = self.store.filter_by_entities(valid_ids)

        # 原始store不应被改变
        self.assertEqual(len(self.store), 5)
        # 新store应只包含4条纯净交互
        self.assertEqual(len(pure_store), 4)

        remaining_ids = pure_store.get_all_entity_auth_ids()
        self.assertNotIn(999, remaining_ids)
        print("  ✅ Passed.")

    def test_get_mapped_positive_pairs(self):
        """测试点4: 在过滤后，能否正确映射为逻辑ID对。"""
        print("--- Running Test: Get Mapped Positive Pairs ---")
        valid_ids = {101, 102, 201, "P01", "P02", "P03"}
        pure_store = self.store.filter_by_entities(valid_ids)

        mapped_pairs = pure_store.get_mapped_positive_pairs(self.mock_id_mapper)
        self.assertEqual(len(mapped_pairs), 4)

        # 转换为集合以便于断言，不关心顺序
        mapped_pairs_set = {tuple(p) for p in mapped_pairs}

        # 预期 (权威ID) -> (逻辑ID)
        # (101, P01) -> (0, 3)
        # (102, P01) -> (1, 3)
        # (201, P02) -> (2, 4)
        # (P01, P03) -> (3, 5)
        expected_pairs_set = {
            (0, 3, "interacts_with"),
            (1, 3, "interacts_with"),
            (2, 4, "inhibits"),
            (3, 5, "ppi"),
        }
        self.assertSetEqual(mapped_pairs_set, expected_pairs_set)
        print("  ✅ Passed.")

    def test_query_api(self):
        """测试点5: 核心查询API的功能是否正确。"""
        print("--- Running Test: Query API ---")

        # 场景a: 只查询 'inhibits' 类型的交互
        selector_a = InteractionSelectorConfig(relation_types=["inhibits"])
        result_a = self.store.query(selector_a, self.mock_id_mapper)
        self.assertEqual(len(result_a), 1)
        self.assertEqual(result_a.dataframe["relation_type"].iloc[0], "inhibits")
        print("  ✅ Passed (a): Query by relation_type.")

        # 场景b: 查询所有源自 'bindingdb' 的 'drug' 的交互
        selector_b = InteractionSelectorConfig(
            source_selector=EntitySelectorConfig(
                entity_types=["drug"], from_sources=["bindingdb"]
            )
        )
        result_b = self.store.query(selector_b, self.mock_id_mapper)
        # (101, P01) 和 (102, P01)
        self.assertEqual(len(result_b), 2)
        print("  ✅ Passed (b): Query by source entity properties.")

        # 场景c: 查询所有目标是 'protein' 且关系是 'ppi' 的交互 (双向匹配)
        selector_c = InteractionSelectorConfig(
            source_selector=EntitySelectorConfig(meta_types=["protein"]),
            target_selector=EntitySelectorConfig(meta_types=["protein"]),
            relation_types=["ppi"],
        )
        result_c = self.store.query(selector_c, self.mock_id_mapper)
        # (P01, P03) 和 (999, P01)
        self.assertEqual(len(result_c), 2)
        print("  ✅ Passed (c): Query with bidirectional matching.")

        # 场景d: 组合查询 - 源是 bindingdb 的 drug, 目标是 bindingdb 的 protein
        selector_d = InteractionSelectorConfig(
            source_selector=EntitySelectorConfig(from_sources=["bindingdb"]),
            target_selector=EntitySelectorConfig(from_sources=["bindingdb"]),
        )
        result_d = self.store.query(selector_d, self.mock_id_mapper)
        # 只有 (101,P01) 和 (102,P01) 满足
        self.assertEqual(len(result_d), 2)
        print("  ✅ Passed (d): Complex query with source and target properties.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
