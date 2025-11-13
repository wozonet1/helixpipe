import unittest
from unittest.mock import patch

import pandas as pd
from omegaconf import OmegaConf

# 导入重命名后的包
from helixpipe.configs import (
    AppConfig,
    InteractionSelectorConfig,
    register_all_schemas,
)
from helixpipe.data_processing.services.interaction_store import InteractionStore


# --- 模拟依赖 (Mocks) ---
class MockIDMapper:
    # 这个Mock现在非常简单，因为Executor会处理大部分逻辑
    def get_meta_by_auth_id(self, auth_id):
        if isinstance(auth_id, str) and auth_id.startswith("P"):
            return {"type": "protein"}
        elif isinstance(auth_id, int):
            return {"type": "drug"}
        return None

    def is_molecule(self, entity_type):
        return entity_type == "drug"

    def is_protein(self, entity_type):
        return entity_type == "protein"


register_all_schemas()
MOCK_CONFIG: AppConfig = OmegaConf.create(
    {
        "runtime": {"verbose": 0},
        "knowledge_graph": {
            "entity_meta": {
                "drug": {"metatype": "molecule", "priority": 0},
                "protein": {"metatype": "protein", "priority": 10},
            }
        },
        "data_structure": {
            "schema": {
                "internal": {
                    "canonical_interaction": {
                        "source_id": "s_id",
                        "source_type": "s_type",
                        "target_id": "t_id",
                        "target_type": "t_type",
                        "relation_type": "rel_type",
                    }
                }
            }
        },
    }
)


class TestInteractionStore(unittest.TestCase):
    def test_canonicalization_on_init(self):
        """测试点1: 初始化时是否正确执行了交互规范化排序。"""
        print("\n--- Running Test: Canonicalization on Initialization ---")
        processor_outputs = {
            "test": pd.DataFrame(
                {
                    "s_id": ["P02", "P03", 102],
                    "s_type": ["protein", "protein", "drug"],
                    "t_id": [101, "P01", "P01"],
                    "t_type": ["drug", "protein", "protein"],
                    "rel_type": ["needs_swap", "id_swap", "no_swap"],
                }
            )
        }

        store = InteractionStore(processor_outputs, MOCK_CONFIG)
        df = store.dataframe.set_index("rel_type")

        # 验证优先级交换
        row_swapped = df.loc["needs_swap"]
        self.assertEqual(row_swapped.s_id, 101)
        self.assertEqual(row_swapped.t_id, "P02")

        # 验证同质ID交换
        row_id_swapped = df.loc["id_swap"]
        self.assertEqual(row_id_swapped.s_id, "P01")
        self.assertEqual(row_id_swapped.t_id, "P03")

        # 验证无需交换
        row_no_swap = df.loc["no_swap"]
        self.assertEqual(row_no_swap.s_id, 102)
        self.assertEqual(row_no_swap.t_id, "P01")
        print("  ✅ Passed.")

    def test_filter_by_entities(self):
        """测试点2: filter_by_entities 功能。"""
        print("\n--- Running Test: Filter by Entities ---")
        store = InteractionStore(
            {
                "test": pd.DataFrame(
                    {
                        "s_id": [1, 99],
                        "t_id": [2, 1],
                        "rel_type": ["a", "b"],
                        "s_type": ["d", "d"],
                        "t_type": ["p", "d"],
                    }
                )
            },
            MOCK_CONFIG,
        )
        pure_store = store.filter_by_entities({1, 2})
        self.assertEqual(len(pure_store), 1)
        # 经过规范化排序后，(2,1)会变成(1,2)
        self.assertEqual(pure_store.dataframe.s_id.iloc[0], 1)
        self.assertEqual(pure_store.dataframe.t_id.iloc[0], 2)
        print("  ✅ Passed.")

    # 使用 @patch 来模拟 SelectorExecutor
    @patch("helixpipe.data_processing.services.interaction_store.SelectorExecutor")
    def test_query_delegates_to_executor(self, MockSelectorExecutor):
        """测试点3: query方法是否正确地委托给SelectorExecutor。"""
        print("\n--- Running Test: Query Delegation ---")

        mock_executor_instance = MockSelectorExecutor.return_value
        mock_return_mask = pd.Series([True, False], index=[0, 1])
        mock_executor_instance.get_interaction_match_mask.return_value = (
            mock_return_mask
        )

        store = InteractionStore(
            {
                "test": pd.DataFrame(
                    {
                        "s_id": [1, 2],
                        "t_id": [3, 4],
                        "rel_type": ["a", "b"],
                        "s_type": ["d", "d"],
                        "t_type": ["p", "p"],
                    }
                )
            },
            MOCK_CONFIG,
        )
        selector = InteractionSelectorConfig()
        mock_id_mapper = MockIDMapper()

        result_store = store.query(selector, mock_id_mapper)

        MockSelectorExecutor.assert_called_once_with(mock_id_mapper, verbose=False)
        mock_executor_instance.get_interaction_match_mask.assert_called_once()

        self.assertEqual(len(result_store), 1)
        self.assertEqual(result_store.dataframe.s_id.iloc[0], 1)
        print("  ✅ Passed.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
