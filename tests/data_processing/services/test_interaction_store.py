import logging
import unittest
from unittest.mock import patch

import pandas as pd
from omegaconf import OmegaConf

# 导入重命名后的包
from helixpipe.configs import (
    AppConfig,
    EntitySelectorConfig,
    InteractionSelectorConfig,
    register_all_schemas,
)
from helixpipe.configs.data_params import (
    DownstreamSamplingConfig,
    StratifiedSamplingConfig,
    StratumConfig,
    UniformSamplingConfig,
)
from helixpipe.data_processing.services import InteractionStore, SelectorExecutor

logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")


class MockIDMapper:
    """
    一个为 SelectorExecutor 提供元数据支持的 Mock IDMapper。
    它的实现必须足够精确，以支持真实的 Executor 运行。
    """

    def get_meta_by_auth_id(self, auth_id):
        # 根据ID的范围和类型，返回模拟的元数据
        if isinstance(auth_id, int):
            if 0 <= auth_id < 10:
                return {"type": "drug"}
            if 10 <= auth_id < 15:
                return {"type": "ligand"}
        if isinstance(auth_id, str) and auth_id.startswith("P"):
            return {"type": "protein"}
        return None

    def is_molecule(self, entity_type):
        return entity_type in ["drug", "ligand"]

    def is_protein(self, entity_type):
        return entity_type == "protein"


register_all_schemas()
MOCK_CONFIG: AppConfig = OmegaConf.create(
    {
        "runtime": {"verbose": 2},
        "knowledge_graph": {
            "entity_meta": {
                "drug": {"metatype": "molecule", "priority": 0},
                "ligand": {"metatype": "molecule", "priority": 1},
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

        MockSelectorExecutor.assert_called_once_with(mock_id_mapper, verbose=True)
        mock_executor_instance.get_interaction_match_mask.assert_called_once()

        self.assertEqual(len(result_store), 1)
        self.assertEqual(result_store.dataframe.s_id.iloc[0], 1)
        print("  ✅ Passed.")

    # 【新增的测试方法】

    def test_apply_sampling_strategy(self):
        """测试点: apply_sampling_strategy 方法的完整编排逻辑。"""
        print("\n" + "=" * 80)
        print("--- Running Test: apply_sampling_strategy ---")

        # --- 1. 准备输入数据 ---
        # 10条drug交互, 5条ligand交互, 3条ppi交互
        processor_outputs = {
            "source1": pd.DataFrame(
                {
                    "s_id": list(range(10)),
                    "s_type": ["drug"] * 10,
                    "t_id": [f"P{i}" for i in range(10)],
                    "t_type": ["protein"] * 10,
                    "rel_type": ["dti"] * 10,
                }
            ),
            "source2": pd.DataFrame(
                {
                    "s_id": list(range(10, 15)),
                    "s_type": ["ligand"] * 5,
                    "t_id": [f"P{i}" for i in range(10, 15)],
                    "t_type": ["protein"] * 5,
                    "rel_type": ["lti"] * 5,
                }
            ),
            "source_ppi": pd.DataFrame(
                {
                    "s_id": [f"P{i}" for i in range(20, 23)],
                    "s_type": ["protein"] * 3,
                    "t_id": [f"P{i}" for i in range(30, 33)],
                    "t_type": ["protein"] * 3,
                    "rel_type": ["ppi"] * 3,
                }
            ),
        }
        store = InteractionStore(processor_outputs, MOCK_CONFIG)
        mock_id_mapper = MockIDMapper()

        # 【核心变化】创建一个真实的 Executor 实例
        executor = SelectorExecutor(mock_id_mapper)
        # --- 2. 定义一个复杂的采样配置 ---
        sampling_config = DownstreamSamplingConfig(
            enabled=True,
            stratified_sampling=StratifiedSamplingConfig(
                enabled=True,
                strata=[
                    StratumConfig(
                        name="ligands_anchor",
                        selector=InteractionSelectorConfig(
                            source_selector=EntitySelectorConfig(
                                entity_types=["ligand"]
                            )
                        ),
                        fraction=1.0,
                    ),
                    StratumConfig(
                        name="drugs_target",
                        selector=InteractionSelectorConfig(
                            source_selector=EntitySelectorConfig(entity_types=["drug"])
                        ),
                        fraction=None,
                        ratio_to="ligands_anchor",
                        ratio=0.4,
                    ),
                    StratumConfig(
                        name="ppi_fraction",
                        selector=InteractionSelectorConfig(relation_types=["ppi"]),
                        fraction=1 / 3,
                    ),
                ],
            ),
            uniform_sampling=UniformSamplingConfig(enabled=True, fraction=0.5),
        )
        # 将这个配置更新到我们的 MOCK_CONFIG 中
        test_config = MOCK_CONFIG.copy()
        OmegaConf.update(test_config, "data_params.sampling", sampling_config)

        # --- 4. 调用被测方法 ---
        sampled_store = store.apply_sampling_strategy(
            config=test_config, executor=executor, seed=42
        )

        # --- 5. 断言结果 ---

        # a. 验证分层采样的中间结果
        # ligand: 5 * 1.0 = 5 条
        # drug: 5 (anchor_count) * 0.4 = 2 条
        # ppi: 3 * 1/3 = 1 条
        # unclassified: 0 条
        # 分层采样后总数: 5 + 2 + 1 = 8 条

        # b. 验证统一采样的最终结果
        # 8 * 0.5 = 4 条
        self.assertEqual(
            len(sampled_store), 4, "Final count after uniform sampling is incorrect."
        )

        # c. 验证最终结果的内容构成是否大致符合预期
        # 由于采样是随机的，我们不能精确断言内容，但可以检查类型的构成
        # 预期最终结果中：ligand 约 5*0.5=2.5条, drug 约 2*0.5=1条, ppi 约 1*0.5=0.5条
        # 这是一个概率问题，我们可以检查类型的存在性

        # ppi 数量少，可能被采样掉，不做强断言
        # self.assertIn("protein", type_counts)

        print("  ✅ Passed.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
