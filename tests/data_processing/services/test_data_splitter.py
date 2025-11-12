# tests/data_processing/services/test_data_splitter.py

import unittest
from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf

from nasnet.configs.training import ColdstartConfig, EntitySelectorConfig
from nasnet.data_processing.services.splitter import DataSplitter


# --- 【最终版 V3】MockIDMapper ---
class MockIDMapper:
    """
    一个模拟的IDMapper，现在精确地模拟了 get_ids_by_selector 的行为。
    """

    def __init__(self):
        self.logic_id_to_type_map = {
            0: "drug",
            1: "drug",
            100: "protein",
            101: "protein",
        }

        # 【核心修正】: 我们不再简单地mock get_ids_by_selector，
        # 而是给它一个真实的、能够响应不同selector的实现。
        self.get_ids_by_selector = MagicMock(side_effect=self._selector_side_effect)

    def _selector_side_effect(self, selector: EntitySelectorConfig):
        """这个函数模拟真实的 get_ids_by_selector 的行为。"""
        # DataSplitter 在 presplit 时会调用这个
        if selector.meta_types == ["molecule"]:
            return {0, 1}  # 所有 drug 的 ID
        if selector.meta_types == ["protein"]:
            return {100, 101}  # 所有 protein 的 ID

        # DataSplitter 在准备冷启动实体池时也会调用这个
        # 我们的测试场景下，pool_scope 也是 meta_types=['molecule']，所以上面的if会命中

        return set()  # 默认返回空集合

    # 剩下的方法 splitter 在 V6 中不再直接使用，但保留无妨
    def is_molecule(self, t):
        return t == "drug"

    def is_protein(self, t):
        return t == "protein"


class TestDataSplitter(unittest.TestCase):
    @patch("nasnet.data_processing.services.splitter.train_test_split")
    def test_strictness_modes_for_background_edges(self, mock_train_test_split):
        print("\n--- Running Test: DataSplitter Strictness Modes (V3 Corrected) ---")

        mock_id_mapper = MockIDMapper()
        # 【核心修正】: 我们需要确保 splitter 在请求冷启动池时，get_ids_by_selector 也能正确响应
        # 在我们的测试中，pool_scope 和 presplit 的 source_scope 是一样的，所以没问题。

        mock_train_test_split.return_value = ([1], [0])

        all_pairs = [
            (0, 100, "interacts_with"),  # -> evaluable -> test
            (1, 101, "interacts_with"),  # -> evaluable -> train
            (0, 1, "drug_drug_similarity"),  # -> background
        ]

        pool_scope = EntitySelectorConfig(meta_types=["molecule"])
        eval_scope = (
            EntitySelectorConfig(meta_types=["molecule"]),
            EntitySelectorConfig(meta_types=["protein"]),
        )

        # --- 测试 'strict' 模式 ---
        print("  -> Testing 'strict' mode...")
        cfg_strict = ColdstartConfig(
            pool_scope=pool_scope,
            evaluation_scope=eval_scope,
            strictness="strict",
            test_fraction=0.5,
        )
        conf_strict = OmegaConf.create(
            {
                "training": {"coldstart": cfg_strict, "k_folds": 1},
                "runtime": {"verbose": 0},
                "data_structure": {"primary_dataset": ""},
            }
        )
        # 这里的 all_pairs 会被正确地 presplit
        # _evaluable_pairs = [(0, 100, ...), (1, 101, ...)]
        # _background_pairs = [(0, 1, ...)]
        splitter_strict = DataSplitter(conf_strict, all_pairs, mock_id_mapper, 42)
        (
            _,
            train_graph_edges_strict,
            train_labels_strict,
            test_pairs_strict,
            _,
        ) = next(iter(splitter_strict))

        # 断言现在应该完全正确
        self.assertCountEqual(test_pairs_strict, [(0, 100, "interacts_with")])
        self.assertCountEqual(train_labels_strict, [(1, 101, "interacts_with")])
        self.assertCountEqual(
            train_graph_edges_strict,
            [(1, 101, "interacts_with")],
            "Strict mode failed: Leaky background edge was NOT removed.",
        )
        print("    ✅ 'strict' mode behaved as expected.")

        # --- 测试 'informed' 模式 ---
        print("  -> Testing 'informed' mode...")
        cfg_informed = ColdstartConfig(
            pool_scope=pool_scope,
            evaluation_scope=eval_scope,
            strictness="informed",
            test_fraction=0.5,
        )
        conf_informed = OmegaConf.create(
            {
                "training": {"coldstart": cfg_informed, "k_folds": 1},
                "runtime": {"verbose": 0},
                "data_structure": {"primary_dataset": ""},
            }
        )
        splitter_informed = DataSplitter(conf_informed, all_pairs, mock_id_mapper, 42)
        (
            _,
            train_graph_edges_informed,
            train_labels_informed,
            test_pairs_informed,
            _,
        ) = next(iter(splitter_informed))

        self.assertCountEqual(test_pairs_informed, [(0, 100, "interacts_with")])
        self.assertCountEqual(train_labels_informed, [(1, 101, "interacts_with")])
        self.assertCountEqual(
            train_graph_edges_informed,
            [(1, 101, "interacts_with"), (0, 1, "drug_drug_similarity")],
            "Informed mode failed: Leaky background edge was NOT kept.",
        )
        print("    ✅ 'informed' mode behaved as expected.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
