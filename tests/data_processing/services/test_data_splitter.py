import logging
import unittest
from typing import cast

import pandas as pd
from omegaconf import OmegaConf

# 导入所有需要的真实模块和配置类
from helixpipe.configs import (
    AppConfig,
    EntitySelectorConfig,
    register_all_schemas,
)
from helixpipe.configs.training import ColdstartConfig
from helixpipe.data_processing.services.interaction_store import InteractionStore
from helixpipe.data_processing.services.selector_executor import SelectorExecutor
from helixpipe.data_processing.services.splitter import DataSplitter

logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(name)s: %(message)s")


# --- 模拟最底层的数据源 ---
class MockIDMapper:
    def __init__(self):
        self.auth_id_to_logic_id_map = {
            "d1": 0,
            "d2": 1,
            "d3": 2,
            "d4": 3,  # 4 drugs
            "p1": 4,
            "p2": 5,
            "p3": 6,  # 3 proteins
        }
        self.meta_data = {
            "d1": {"type": "drug", "sources": {"db1"}},
            "d2": {"type": "drug", "sources": {"db1"}},
            "d3": {"type": "drug", "sources": {"db2"}},
            "d4": {"type": "drug", "sources": {"db2"}},
            "p1": {"type": "protein", "sources": {"db1"}},
            "p2": {"type": "protein", "sources": {"db2"}},
            "p3": {"type": "protein", "sources": {"db1", "db2"}},
        }

    def get_meta_by_auth_id(self, auth_id):
        return self.meta_data.get(auth_id)

    def get_all_final_ids(self):
        return list(self.meta_data.keys())

    def is_molecule(self, entity_type):
        return entity_type == "drug"

    def is_protein(self, entity_type):
        return entity_type == "protein"


# 注册并创建基础配置
register_all_schemas()
MOCK_BASE_CONFIG: AppConfig = cast(
    AppConfig,
    OmegaConf.create(
        {
            "runtime": {"verbose": 0},
            "knowledge_graph": {
                "entity_meta": {"drug": {"priority": 0}, "protein": {"priority": 10}}
            },
            "data_structure": {
                "primary_dataset": "db1",
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
                },
            },
        }
    ),
)


class TestDataSplitter(unittest.TestCase):
    def setUp(self):
        """准备一个通用的测试环境。"""
        print("\n" + "=" * 80)

        # 准备一个包含主角DTI和背景知识PPI的数据集
        processor_outputs = {
            "db1": pd.DataFrame(
                {  # 主数据集
                    "s_id": ["d1", "d2"],
                    "s_type": ["drug"] * 2,
                    "t_id": ["p1", "p1"],
                    "t_type": ["protein"] * 2,
                    "rel_type": ["interacts_with"] * 2,
                }
            ),
            "db2": pd.DataFrame(
                {  # 辅助数据集
                    "s_id": ["d3"],
                    "s_type": ["drug"],
                    "t_id": ["p2"],
                    "t_type": ["protein"],
                    "rel_type": ["interacts_with"],
                }
            ),
            "stringdb": pd.DataFrame(
                {  # 背景知识
                    "s_id": ["p1", "p2"],
                    "s_type": ["protein"] * 2,
                    "t_id": ["p2", "d4"],
                    "t_type": ["protein", "drug"],  # p2-d4 是一个会泄露信息的边
                    "rel_type": ["ppi", "interacts_with"],
                }
            ),
        }
        self.store = InteractionStore(processor_outputs, MOCK_BASE_CONFIG)
        self.mock_id_mapper = MockIDMapper()
        self.executor = SelectorExecutor(self.mock_id_mapper)  # pyright: ignore[reportArgumentType]

    def test_hot_start_split(self):
        """测试点1: 热启动（随机划分交互）"""
        print("--- Running Test: Hot-Start Split (Single Run) ---")

        coldstart_cfg = ColdstartConfig(
            mode="warm",
            pool_scope=EntitySelectorConfig(),  # 空的 pool_scope 触发热启动
            test_fraction=0.5,
            # evaluation_scope 走默认，只评估主角DTI (d1-p1, d2-p1)
        )
        test_config = MOCK_BASE_CONFIG.copy()
        OmegaConf.update(test_config, "training.coldstart", coldstart_cfg)
        OmegaConf.update(test_config, "training.k_folds", 1)  # 单次运行

        splitter = DataSplitter(
            test_config, self.store, self.mock_id_mapper, self.executor, seed=42
        )

        # 预分流检查
        self.assertEqual(
            len(splitter._evaluable_store),
            5,
            "Should identify 5 protagonist DTIs as evaluable.",
        )
        self.assertEqual(
            len(splitter._background_store),
            0,
            "Should identify 0 interactions as background.",
        )

        fold, (train_graph, train_labels, test, cold_ids) = next(iter(splitter))

        # 热启动，test_fraction=0.5，2个可评估DTI，1个进训练，1个进测试
        self.assertEqual(
            len(train_labels), 2, "Train labels should contain 1 evaluable interaction."
        )
        self.assertEqual(
            len(test), 3, "Test set should contain 1 evaluable interaction."
        )
        # 训练图 = 1(train_label) + 3(background) = 4
        self.assertEqual(
            len(train_graph),
            2,
            "Train graph should contain train labels + all background.",
        )
        self.assertEqual(len(cold_ids), 0, "No cold start IDs in hot-start mode.")
        print("  ✅ Passed.")

    def test_cold_start_strict_split(self):
        """测试点2: 冷启动 'strict' 模式"""
        print("--- Running Test: Cold-Start 'strict' Split ---")

        coldstart_cfg = ColdstartConfig(
            pool_scope=EntitySelectorConfig(
                from_sources=["db2"]
            ),  # 在 db2 的实体 (d3, d4, p2, p3) 上冷启动
            test_fraction=0.5,
            strictness="strict",
        )
        test_config = MOCK_BASE_CONFIG.copy()
        OmegaConf.update(test_config, "training.coldstart", coldstart_cfg)
        OmegaConf.update(test_config, "training.k_folds", 1)

        # 在 db2 的实体上划分，假设 d4, p2 进入测试集
        # (train_test_split with seed=42 and a sorted list will produce a deterministic result)
        # sorted(['d3', 'd4', 'p2', 'p3']) -> test_entities = {'p2', 'd4'}

        splitter = DataSplitter(
            test_config, self.store, self.mock_id_mapper, self.executor, seed=42
        )

        fold, (train_graph, train_labels, test, cold_ids) = next(iter(splitter))

        # 可评估的DTI (d1-p1, d2-p1) 不涉及冷启动实体，全部进入训练标签
        self.assertEqual(len(train_labels), 2)
        # 测试集为空，因为可评估DTI都不涉及冷启动实体
        self.assertEqual(len(test), 0)

        self.assertEqual(len(train_graph), 4)

        # 验证冷启动ID是否被正确识别 (逻辑ID)
        expected_cold_ids_logic = {
            self.mock_id_mapper.auth_id_to_logic_id_map["p3"],
            self.mock_id_mapper.auth_id_to_logic_id_map["d4"],
        }
        self.assertSetEqual(cold_ids, expected_cold_ids_logic)
        print("  ✅ Passed.")

    def test_cold_start_informed_split(self):
        """测试点3: 冷启动 'informed' 模式"""
        print("--- Running Test: Cold-Start 'informed' Split ---")

        coldstart_cfg = ColdstartConfig(
            pool_scope=EntitySelectorConfig(from_sources=["db2"]),
            test_fraction=0.5,
            strictness="informed",
        )
        test_config = MOCK_BASE_CONFIG.copy()
        OmegaConf.update(test_config, "training.coldstart", coldstart_cfg)
        OmegaConf.update(test_config, "training.k_folds", 1)

        splitter = DataSplitter(
            test_config, self.store, self.mock_id_mapper, self.executor, seed=42
        )
        fold, (train_graph, train_labels, test, cold_ids) = next(iter(splitter))

        # train_labels 和 test 结果与 strict 模式相同
        self.assertEqual(len(train_labels), 2)
        self.assertEqual(len(test), 0)

        # 训练图 = 2(train_labels) + (经过informed过滤的background)
        # 背景知识: d3-p2, p1-p2, p2-d4
        # 在 informed 模式下，所有这些泄露信息的边都被保留
        self.assertEqual(len(train_graph), 5)
        print("  ✅ Passed.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
