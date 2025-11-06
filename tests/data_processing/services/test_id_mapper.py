import unittest

import pandas as pd
from omegaconf import OmegaConf

# 导入我们需要测试的类
from nasnet.data_processing.services.id_mapper import IDMapper


class TestIDMapperV5(unittest.TestCase):
    def setUp(self):
        """
        创建一个共享的、包含所有必需部分的配置对象。
        """
        print("\n" + "=" * 80)

        self.config = OmegaConf.create(
            {
                "runtime": {"verbose": 0},
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
                "knowledge_graph": {
                    "entity_types": {
                        "drug": "drug",
                        "ligand": "ligand",
                        "protein": "protein",
                        "gene": "gene",
                    },
                    "entity_meta": {
                        "drug": {"metatype": "molecule", "priority": 0},
                        "ligand": {"metatype": "molecule", "priority": 1},
                        "protein": {"metatype": "protein", "priority": 10},
                        "gene": {"metatype": "protein", "priority": 11},
                    },
                },
            }
        )
        print("--> Test setup complete.")

    def test_initialization_and_aggregation(self):
        """
        测试点1: __init__ 是否能正确地从多个 processor_outputs 中聚合 types 和 sources。
        """
        print("\n--- Running Test: Initialization and Metadata Aggregation ---")

        # 模拟来自不同 Processor 的输出
        processor_outputs = {
            "bindingdb": pd.DataFrame(
                {
                    "source_id": [101, "P01"],
                    "source_type": ["drug", "protein"],
                    "target_id": ["P01", "P02"],
                    "target_type": ["protein", "protein"],
                }
            ),
            "stringdb": pd.DataFrame(
                {
                    "source_id": ["P01"],
                    "source_type": ["protein"],
                    "target_id": ["P03"],
                    "target_type": ["protein"],
                }
            ),
            "brenda": pd.DataFrame(
                {
                    "source_id": [101],
                    "source_type": ["ligand"],
                    "target_id": ["P02"],
                    "target_type": ["protein"],
                }
            ),
        }

        mapper = IDMapper(processor_outputs, self.config)

        # 验证内部的 _collected_entities 字典
        collected = mapper._collected_entities
        self.assertEqual(len(collected), 4, "Should collect 5 unique entities.")

        # 深入检查几个关键实体
        self.assertSetEqual(collected[101]["types"], {"drug", "ligand"})
        self.assertSetEqual(collected[101]["sources"], {"bindingdb", "brenda"})

        self.assertSetEqual(collected["P01"]["types"], {"protein"})
        self.assertSetEqual(collected["P01"]["sources"], {"bindingdb", "stringdb"})

        self.assertSetEqual(collected["P03"]["types"], {"protein"})
        self.assertSetEqual(collected["P03"]["sources"], {"stringdb"})

        print("  ✅ Test Passed: Correctly aggregated types and sources.")

    def test_is_molecule_is_protein(self):
        """
        测试点2: is_molecule 和 is_protein 方法是否由 config 驱动。
        """
        print("\n--- Running Test: Config-driven is_molecule/is_protein ---")

        mapper = IDMapper({}, self.config)  # 用空数据初始化即可

        self.assertTrue(mapper.is_molecule("drug"))
        self.assertTrue(mapper.is_molecule("ligand"))
        self.assertTrue(mapper.is_protein("protein"))
        self.assertTrue(mapper.is_protein("gene"))
        self.assertFalse(mapper.is_molecule("protein"))
        self.assertFalse(mapper.is_protein("drug"))

        print("  ✅ Test Passed: Type checking methods are correctly driven by config.")

    def test_finalization_and_type_merging(self):
        """
        测试点3: finalize_with_valid_entities 是否能正确执行类型合并和ID分配。
        """
        print("\n--- Running Test: Finalization with Type Merging ---")

        processor_outputs = {
            "source1": pd.DataFrame(
                {
                    "source_id": [101, 102],
                    "source_type": ["drug", "protein"],
                    "target_id": ["P01", "P01"],
                    "target_type": ["protein", "protein"],
                }
            ),
            "source2": pd.DataFrame(
                {
                    "source_id": [101],
                    "source_type": ["ligand"],
                    "target_id": ["P02"],
                    "target_type": ["protein"],
                }
            ),
        }
        mapper = IDMapper(processor_outputs, self.config)

        # 假设校验后，所有实体都有效
        valid_ids = {101, 102, "P01", "P02"}
        with self.assertRaises(ValueError) as context:
            mapper.finalize_with_valid_entities(valid_ids)
        # 3. [可选] 检查异常消息中是否包含了我们期望的、有用的信息
        self.assertIn("ID format mismatch", str(context.exception))
        self.assertIn("'protein'", str(context.exception))
        self.assertIn("ID: 102", str(context.exception))

        print("  ✅ Test Passed: Correctly raised ValueError on inconsistent data.")

    def test_query_apis(self):
        """
        测试点4: 测试新的查询API get_entity_meta 和 get_ids_by_filter。
        """
        print("\n--- Running Test: New Query APIs ---")

        processor_outputs_clean = {
            "bindingdb": pd.DataFrame(
                {
                    "source_id": [101],
                    "source_type": ["drug"],
                    "target_id": ["P01"],
                    "target_type": ["protein"],
                }
            ),
            "stringdb": pd.DataFrame(
                {
                    "source_id": ["P01"],
                    "source_type": ["protein"],
                    "target_id": ["P02"],
                    "target_type": ["protein"],
                }
            ),
        }
        mapper = IDMapper(processor_outputs_clean, self.config)
        mapper.finalize_with_valid_entities({101, "P01", "P02"})

        # a. 测试 get_entity_meta
        # 假设 drug 0, protein 1, 2
        logic_id_p01 = mapper.auth_id_to_logic_id_map["P01"]
        meta_p01 = mapper.get_entity_meta(logic_id_p01)

        self.assertIsNotNone(meta_p01)
        self.assertEqual(meta_p01["type"], "protein")
        self.assertSetEqual(meta_p01["sources"], {"bindingdb", "stringdb"})

        # b. 测试 get_ids_by_filter
        # 获取所有来自 stringdb 的实体
        string_ids_logic = mapper.get_ids_by_filter(
            lambda meta: "stringdb" in meta["sources"]
        )
        self.assertEqual(len(string_ids_logic), 2)  # P01 和 P02
        self.assertIn(mapper.auth_id_to_logic_id_map["P01"], string_ids_logic)
        self.assertIn(mapper.auth_id_to_logic_id_map["P02"], string_ids_logic)

        # 获取所有 drug 实体
        drug_ids_logic = mapper.get_ids_by_filter(lambda meta: meta["type"] == "drug")
        self.assertEqual(len(drug_ids_logic), 1)
        self.assertEqual(drug_ids_logic[0], mapper.auth_id_to_logic_id_map[101])

        print("  ✅ Test Passed: Query APIs work as expected.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
