# tests/data_processing/services/test_label_generator.py

import unittest
from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf

from nasnet.configs.training import ColdstartConfig, EntitySelectorConfig

# 导入我们需要测试的类和相关的dataclass
from nasnet.data_processing.services.label_generator import SupervisionFileManager


# --- 模拟 (Mock) 的 IDMapper ---
class MockIDMapper:
    def __init__(self):
        # 这个 get_meta_by_logic_id 将在每个测试中被 mock，所以这里的实现不重要
        self.get_meta_by_logic_id = MagicMock()
        # get_ids_by_selector 也将被 mock
        self.get_ids_by_selector = MagicMock()


class TestSupervisionFileManager(unittest.TestCase):
    def setUp(self):
        """准备一个通用的测试环境。"""
        self.mock_id_mapper = MockIDMapper()
        self.seed = 42
        self.fold_idx = 1
        # 一个简单的正样本集合，用于负采样时的碰撞检查
        self.global_pos_set = {(0, 100), (1, 101)}

        # 准备一个包含多种交互类型的、划分好的测试集
        self.test_pairs_global = [
            (
                0,
                100,
                "rel_protagonist_dti",
            ),  # 主角DTI (drug from bindingdb -> protein from bindingdb)
            (
                1,
                200,
                "rel_mixed_dti",
            ),  # 混合DTI (drug from bindingdb -> protein from stringdb)
            (10, 101, "rel_lpi"),  # LPI (ligand -> protein)
            (100, 101, "rel_ppi"),  # PPI (protein -> protein)
        ]

        # Mock get_meta_by_logic_id 的行为
        def meta_side_effect(logic_id):
            meta_map = {
                0: {"type": "drug", "sources": {"bindingdb"}},
                1: {"type": "drug", "sources": {"bindingdb"}},
                10: {"type": "ligand", "sources": {"gtopdb"}},
                100: {"type": "protein", "sources": {"bindingdb"}},
                101: {"type": "protein", "sources": {"bindingdb"}},
                200: {"type": "protein", "sources": {"stringdb"}},  # 这是一个配角蛋白
            }
            return meta_map.get(logic_id)

        self.mock_id_mapper.get_meta_by_logic_id.side_effect = meta_side_effect

    # 我们需要mock get_path 和 DataFrame.to_csv，因为我们不想在测试中真正写入文件
    @patch("nasnet.data_processing.services.label_generator.get_path")
    @patch("pandas.DataFrame.to_csv")
    def test_evaluation_scope_filtering(self, mock_to_csv, mock_get_path):
        """
        测试场景1: 验证 _save_test_labels 是否能根据 evaluation_scope 正确过滤正样本。
        """
        print("\n--- Running Test: Evaluation Scope Filtering ---")

        # 1. 配置: 只评估主角DTI
        source_selector = EntitySelectorConfig(
            entity_types=["drug"], from_sources=["bindingdb"]
        )
        target_selector = EntitySelectorConfig(
            meta_types=["protein"], from_sources=["bindingdb"]
        )
        coldstart_cfg = ColdstartConfig(
            evaluation_scope=(source_selector, target_selector)
        )
        config = OmegaConf.create(
            {
                "training": {"coldstart": coldstart_cfg},
                "runtime": {"verbose": 1},
                "knowledge_graph": {
                    "entity_meta": {"protein": {"metatype": "protein"}}
                },  # 提供 meta_types 需要的元信息
                "data_structure": {
                    "schema": {
                        "internal": {
                            "labeled_edges_output": {
                                "source_node": "s",
                                "target_node": "t",
                                "label": "l",
                            }
                        }
                    }
                },
            }
        )

        # 2. Mock 依赖
        # 负采样部分我们暂时不关心，返回空列表
        self.mock_id_mapper.get_ids_by_selector.return_value = []
        mock_get_path.return_value = MagicMock()  # 让 get_path 返回一个模拟对象

        # 3. 初始化和运行
        manager = SupervisionFileManager(
            self.fold_idx, config, self.mock_id_mapper, self.global_pos_set, self.seed
        )
        manager._save_test_labels(self.test_pairs_global)

        # 4. 断言
        # a. 检查传递给 to_csv 的DataFrame的内容
        self.assertTrue(mock_to_csv.called, "DataFrame.to_csv should have been called.")

        # 获取 to_csv 被调用时的第一个参数 (即 self, 也就是那个DataFrame)
        saved_df = mock_to_csv.call_args[0][0]

        # 预期只有 (0, 100) 这个主角DTI被保留为正样本
        self.assertEqual(
            len(saved_df[saved_df["l"] == 1]),
            1,
            "Only one positive pair should pass the filter.",
        )

        saved_pos_pair = saved_df[saved_df["l"] == 1]
        self.assertEqual(saved_pos_pair["s"].iloc[0], 0)
        self.assertEqual(saved_pos_pair["t"].iloc[0], 100)

        print("  ✅ Passed.")

    @patch("nasnet.data_processing.services.label_generator.get_path")
    @patch("pandas.DataFrame.to_csv")
    def test_selector_driven_negative_sampling(self, mock_to_csv, mock_get_path):
        """
        测试场景2: 验证 _perform_negative_sampling 是否使用 selector 来构建正确的采样池。
        """
        print("\n--- Running Test: Selector-Driven Negative Sampling ---")

        # 1. 配置: 同上，负采样池也应该由主角DTI的 selector 决定
        source_selector = EntitySelectorConfig(
            entity_types=["drug"], from_sources=["bindingdb"]
        )
        target_selector = EntitySelectorConfig(
            meta_types=["protein"], from_sources=["bindingdb"]
        )
        coldstart_cfg = ColdstartConfig(
            evaluation_scope=(source_selector, target_selector)
        )
        config = OmegaConf.create(
            {
                "training": {"coldstart": coldstart_cfg},
                "runtime": {"verbose": 1},
                # ... (其他与上一个测试相同的配置)
            }
        )

        # 2. Mock 依赖
        # 【核心】命令 get_ids_by_selector 返回我们预设的主角实体池
        def selector_side_effect(selector):
            if selector.entity_types == ["drug"]:
                return [0, 1]  # 主角药物
            if selector.meta_types == ["protein"]:
                return [100, 101]  # 主角蛋白
            return []

        self.mock_id_mapper.get_ids_by_selector.side_effect = selector_side_effect
        mock_get_path.return_value = MagicMock()

        # 3. 初始化和运行
        manager = SupervisionFileManager(
            self.fold_idx, config, self.mock_id_mapper, self.global_pos_set, self.seed
        )

        # a. 只保留一个正样本，以请求一个负样本
        one_pos_pair = [(0, 100, "rel")]
        manager._save_test_labels(one_pos_pair)

        # b. 获取保存的DataFrame
        saved_df = mock_to_csv.call_args[0][0]
        neg_pairs_df = saved_df[saved_df["l"] == 0]

        # 4. 断言
        self.assertEqual(len(neg_pairs_df), 1, "Should generate one negative pair.")

        neg_source = neg_pairs_df["s"].iloc[0]
        neg_target = neg_pairs_df["t"].iloc[0]

        # 验证生成的负样本的源和目标，都来自于我们预设的主角池
        self.assertIn(
            neg_source, [0, 1], "Negative sample source must be a protagonist drug."
        )
        self.assertIn(
            neg_target,
            [100, 101],
            "Negative sample target must be a protagonist protein.",
        )

        print("  ✅ Passed.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
