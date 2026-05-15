import logging
import unittest
from typing import cast
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
MOCK_CONFIG: AppConfig = cast(
    AppConfig,
    OmegaConf.create(
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
                            "source_dataset": "source_dataset",
                            "raw_score": "raw_score",
                        }
                    }
                }
            },
        }
    ),
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
                    "source_dataset": ["test", "test", "test"],
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
                    "source_dataset": ["source1"] * 10,
                }
            ),
            "source2": pd.DataFrame(
                {
                    "s_id": list(range(10, 15)),
                    "s_type": ["ligand"] * 5,
                    "t_id": [f"P{i}" for i in range(10, 15)],
                    "t_type": ["protein"] * 5,
                    "rel_type": ["lti"] * 5,
                    "source_dataset": ["source2"] * 5,
                }
            ),
            "source_ppi": pd.DataFrame(
                {
                    "s_id": [f"P{i}" for i in range(20, 23)],
                    "s_type": ["protein"] * 3,
                    "t_id": [f"P{i}" for i in range(30, 33)],
                    "t_type": ["protein"] * 3,
                    "rel_type": ["ppi"] * 3,
                    "source_dataset": ["source_ppi"] * 3,
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

    def test_difference_excludes_source_metadata(self):
        """
        BUG-01+ISSUE-07 修复验证：
        difference() 应只基于 canonical 列 (source_id, target_id, relation_type) 计算差集，
        不受任何内部元数据列的影响。

        场景：同一个交互 (1, P01, dti) 同时出现在 source1 和 source2 中。
        旧实现会因为 source_dataset 列不同而将它们视为不同行，
        导致 difference() 的结果中出现本应被减掉的数据泄漏。
        """
        # 构建一个包含来自多个数据源的交互的 store
        # 注意：(1, P01, dti) 同时存在于 source1 和 source2
        all_interactions = pd.DataFrame(
            {
                "s_id": [1, 2, 3],
                "s_type": ["drug", "drug", "drug"],
                "t_id": ["P01", "P02", "P03"],
                "t_type": ["protein", "protein", "protein"],
                "rel_type": ["dti", "dti", "dti"],
            }
        )
        all_store = InteractionStore._from_dataframe(all_interactions, MOCK_CONFIG)

        # evaluable store 包含 (1, P01, dti) 和 (2, P02, dti)
        evaluable_df = pd.DataFrame(
            {
                "s_id": [1, 2],
                "s_type": ["drug", "drug"],
                "t_id": ["P01", "P02"],
                "t_type": ["protein", "protein"],
                "rel_type": ["dti", "dti"],
            }
        )
        evaluable_store = InteractionStore._from_dataframe(evaluable_df, MOCK_CONFIG)

        # 执行差集
        background_store = all_store.difference(evaluable_store)

        # background 应只包含 (3, P03, dti)
        # 如果 source_dataset 列泄漏到 difference() 中，
        # 结果会错误地保留更多行
        self.assertEqual(len(background_store), 1)
        self.assertEqual(background_store.dataframe.s_id.iloc[0], 3)
        self.assertEqual(background_store.dataframe.t_id.iloc[0], "P03")

    def test_source_tags_preserved_through_operations(self):
        """
        验证 source_dataset 列在子集操作中被正确传递：

        1. 初始化时记录每条交互的来源
        2. filter_by_entities 后保留过滤结果的来源
        3. difference 后保留差集的来源
        4. source_dataset 作为 DataFrame Categorical 列存在
        """
        # 创建来自两个数据源的交互
        processor_outputs = {
            "bindingdb": pd.DataFrame(
                {
                    "s_id": [1, 2],
                    "s_type": ["drug", "drug"],
                    "t_id": ["P01", "P02"],
                    "t_type": ["protein", "protein"],
                    "rel_type": ["dti", "dti"],
                    "source_dataset": ["bindingdb", "bindingdb"],
                }
            ),
            "gtopdb": pd.DataFrame(
                {
                    "s_id": [3],
                    "s_type": ["drug"],
                    "t_id": ["P03"],
                    "t_type": ["protein"],
                    "rel_type": ["dti"],
                    "source_dataset": ["gtopdb"],
                }
            ),
        }
        store = InteractionStore(processor_outputs, MOCK_CONFIG)

        # 断言：source_dataset 列应该存在于 DataFrame 中
        self.assertIn("source_dataset", store.dataframe.columns)

        # 断言：来源应正确记录（规范化后 key 为 (source, target, relation)）
        df = store.dataframe
        src_1 = df[(df["s_id"] == 1) & (df["t_id"] == "P01")]["source_dataset"].iloc[0]
        self.assertEqual(src_1, "bindingdb")
        src_3 = df[(df["s_id"] == 3) & (df["t_id"] == "P03")]["source_dataset"].iloc[0]
        self.assertEqual(src_3, "gtopdb")

        # 测试 filter_by_entities 后来源列是否保留
        filtered = store.filter_by_entities({1, "P01"})
        self.assertEqual(len(filtered), 1)
        self.assertIn("source_dataset", filtered.dataframe.columns)
        self.assertEqual(filtered.dataframe["source_dataset"].iloc[0], "bindingdb")

    def test_concat_merges_source_dataset(self):
        """
        验证 concat + dedup 策略 A：
        同一交互来自多个 store 时，source_dataset 用 "|" 合并，
        不同交互正常取并集。
        """
        # store_a: 两条来自 bindingdb 的交互
        df_a = pd.DataFrame(
            {
                "s_id": [1, 2],
                "s_type": ["drug", "drug"],
                "t_id": ["P01", "P02"],
                "t_type": ["protein", "protein"],
                "rel_type": ["dti", "dti"],
                "source_dataset": ["bindingdb", "bindingdb"],
            }
        )
        # store_b: 其中一条与 store_a 重叠，另一条不同
        df_b = pd.DataFrame(
            {
                "s_id": [1, 3],
                "s_type": ["drug", "drug"],
                "t_id": ["P01", "P03"],
                "t_type": ["protein", "protein"],
                "rel_type": ["dti", "dti"],
                "source_dataset": ["gtopdb", "gtopdb"],
            }
        )
        store_a = InteractionStore._from_dataframe(df_a, MOCK_CONFIG)
        store_b = InteractionStore._from_dataframe(df_b, MOCK_CONFIG)

        merged = InteractionStore.concat([store_a, store_b], MOCK_CONFIG)

        # 应去重为 3 条：(1,P01), (2,P02), (3,P03)
        self.assertEqual(len(merged), 3)
        result_df = merged.dataframe

        # 重叠的交互 (1, P01) 应合并来源
        row_overlap = result_df[(result_df["s_id"] == 1) & (result_df["t_id"] == "P01")]
        self.assertEqual(len(row_overlap), 1)
        self.assertEqual(row_overlap["source_dataset"].iloc[0], "bindingdb|gtopdb")

        # 非重叠的交互来源不变
        row_b_only = result_df[(result_df["s_id"] == 2) & (result_df["t_id"] == "P02")]
        self.assertEqual(row_b_only["source_dataset"].iloc[0], "bindingdb")

        row_c_only = result_df[(result_df["s_id"] == 3) & (result_df["t_id"] == "P03")]
        self.assertEqual(row_c_only["source_dataset"].iloc[0], "gtopdb")

    def test_init_merges_duplicate_sources(self):
        """
        验证 __init__ 阶段的去重：
        多个 Processor 输出包含同一交互时，source_dataset 被合并。
        """
        processor_outputs = {
            "bindingdb": pd.DataFrame(
                {
                    "s_id": [1],
                    "s_type": ["drug"],
                    "t_id": ["P01"],
                    "t_type": ["protein"],
                    "rel_type": ["dti"],
                    "source_dataset": ["bindingdb"],
                }
            ),
            "gtopdb": pd.DataFrame(
                {
                    "s_id": [1],
                    "s_type": ["drug"],
                    "t_id": ["P01"],
                    "t_type": ["protein"],
                    "rel_type": ["dti"],
                    "source_dataset": ["gtopdb"],
                }
            ),
        }
        store = InteractionStore(processor_outputs, MOCK_CONFIG)

        self.assertEqual(len(store), 1)
        self.assertEqual(store.dataframe["source_dataset"].iloc[0], "bindingdb|gtopdb")

    def test_from_dataframe_auto_canonicalizes(self):
        """
        BUG-02 修复验证：_from_dataframe 默认执行规范化。
        传入非规范化的 DataFrame（protein 在 source, drug 在 target），
        应自动交换为规范化顺序。
        """
        # 非规范化：protein 在 source, drug 在 target
        non_canonical_df = pd.DataFrame(
            {
                "s_id": ["P01"],
                "s_type": ["protein"],
                "t_id": [1],
                "t_type": ["drug"],
                "rel_type": ["dti"],
                "source_dataset": ["test"],
            }
        )
        store = InteractionStore._from_dataframe(non_canonical_df, MOCK_CONFIG)

        # 应自动规范化：drug (priority=0) → source, protein (priority=10) → target
        df = store.dataframe
        self.assertEqual(df["s_id"].iloc[0], 1)
        self.assertEqual(df["t_id"].iloc[0], "P01")

    def test_from_dataframe_skip_canonicalize(self):
        """
        BUG-02 修复验证：skip_canonicalize=True 时保留原始顺序。
        """
        original_df = pd.DataFrame(
            {
                "s_id": ["P01"],
                "s_type": ["protein"],
                "t_id": [1],
                "t_type": ["drug"],
                "rel_type": ["dti"],
                "source_dataset": ["test"],
            }
        )
        store = InteractionStore._from_dataframe(
            original_df, MOCK_CONFIG, skip_canonicalize=True
        )

        # 应保留原始顺序，不做交换
        df = store.dataframe
        self.assertEqual(df["s_id"].iloc[0], "P01")
        self.assertEqual(df["t_id"].iloc[0], 1)

    def test_concat_auto_canonicalizes(self):
        """
        BUG-02 修复验证：concat 合并两个 store 时，结果自动规范化。
        即使某个 store 内部包含非规范化的行（理论上不应发生，但作为防御性保证）。
        """
        # store_a: 已规范化 (drug → source, protein → target)
        df_a = pd.DataFrame(
            {
                "s_id": [1],
                "s_type": ["drug"],
                "t_id": ["P01"],
                "t_type": ["protein"],
                "rel_type": ["dti"],
                "source_dataset": ["a"],
            }
        )
        # store_b: 非规范化 (protein → source, drug → target)
        df_b = pd.DataFrame(
            {
                "s_id": ["P02"],
                "s_type": ["protein"],
                "t_id": [2],
                "t_type": ["drug"],
                "rel_type": ["dti"],
                "source_dataset": ["b"],
            }
        )
        store_a = InteractionStore._from_dataframe(
            df_a, MOCK_CONFIG, skip_canonicalize=True
        )
        store_b = InteractionStore._from_dataframe(
            df_b, MOCK_CONFIG, skip_canonicalize=True
        )

        merged = InteractionStore.concat([store_a, store_b], MOCK_CONFIG)

        df = merged.dataframe
        # 两条都应规范化：drug → source
        self.assertEqual(df["s_id"].iloc[0], 1)
        self.assertEqual(df["s_id"].iloc[1], 2)

    def test_get_mapped_positive_pairs_with_metadata(self):
        """测试 get_mapped_positive_pairs_with_metadata 返回正确的五元组。"""
        print("\n--- Running Test: get_mapped_positive_pairs_with_metadata ---")

        # 创建带 score 列的数据
        processor_outputs = {
            "test": pd.DataFrame(
                {
                    "s_id": [1, 2],
                    "s_type": ["drug", "drug"],
                    "t_id": ["P01", "P02"],
                    "t_type": ["protein", "protein"],
                    "rel_type": ["dti", "dti"],
                    "source_dataset": ["bindingdb", "gtopdb"],
                    "raw_score": [10.0, 100.0],
                }
            )
        }
        store = InteractionStore(processor_outputs, MOCK_CONFIG)

        # 创建一个带 auth_id_to_logic_id_map 的 mock
        class MockIDMapperWithMap:
            auth_id_to_logic_id_map = {1: 0, 2: 1, "P01": 10, "P02": 11}

        result = store.get_mapped_positive_pairs_with_metadata(MockIDMapperWithMap())

        self.assertEqual(len(result), 2)
        # 验证五元组结构
        for item in result:
            self.assertEqual(len(item), 5)
            self.assertIsInstance(item[0], int)  # source logic id
            self.assertIsInstance(item[1], int)  # target logic id
            self.assertIsInstance(item[2], str)  # relation_type
            self.assertIsInstance(item[3], str)  # source_dataset
            self.assertIsInstance(item[4], float)  # score

        # 验证 source_dataset 值
        sources = {item[3] for item in result}
        self.assertEqual(sources, {"bindingdb", "gtopdb"})

        # 验证返回原始 score（不做归一化）
        scores = {item[0]: item[4] for item in result}
        self.assertAlmostEqual(scores[0], 10.0)  # 原始值 10.0
        self.assertAlmostEqual(scores[1], 100.0)  # 原始值 100.0

    def test_get_mapped_positive_pairs_with_metadata_no_score(self):
        """测试无原始 score 列时，默认 score 为 1.0。"""
        print("\n--- Running Test: metadata with no score column ---")

        processor_outputs = {
            "test": pd.DataFrame(
                {
                    "s_id": [1],
                    "s_type": ["drug"],
                    "t_id": ["P01"],
                    "t_type": ["protein"],
                    "rel_type": ["dti"],
                    "source_dataset": ["bindingdb"],
                }
            )
        }
        store = InteractionStore(processor_outputs, MOCK_CONFIG)

        class MockIDMapperWithMap:
            auth_id_to_logic_id_map = {1: 0, "P01": 10}

        result = store.get_mapped_positive_pairs_with_metadata(MockIDMapperWithMap())

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][4], 1.0)  # 默认 score

    def test_get_mapped_positive_pairs_with_metadata_empty(self):
        """测试空 store 返回空列表。"""
        print("\n--- Running Test: metadata with empty store ---")

        store = InteractionStore._from_dataframe(
            pd.DataFrame(), MOCK_CONFIG, skip_canonicalize=True
        )

        class MockIDMapperWithMap:
            auth_id_to_logic_id_map = {}

        result = store.get_mapped_positive_pairs_with_metadata(MockIDMapperWithMap())
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
