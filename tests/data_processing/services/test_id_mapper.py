# tests/data_processing/services/test_id_mapper.py

import unittest

import pandas as pd
from omegaconf import OmegaConf

# 导入我们需要测试的类和相关的dataclass
from helixpipe.data_processing.services.id_mapper import IDMapper

# --- 模拟 (Mock) 配置 ---
MOCK_CONFIG = OmegaConf.create(
    {
        "runtime": {"verbose": 2},
        "knowledge_graph": {
            "entity_meta": {
                "drug": {"metatype": "molecule", "priority": 0},
                "ligand": {"metatype": "molecule", "priority": 1},
                "protein": {"metatype": "protein", "priority": 10},
                "gene": {"metatype": "protein", "priority": 11},
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
                    }
                }
            }
        },
    }
)


class TestIDMapperV5(unittest.TestCase):
    def test_initialization_and_metadata_aggregation(self):
        print("\n--- Running Test: V5 Initialization and Metadata Aggregation ---")
        processor_outputs = {
            "bindingdb": pd.DataFrame(
                {
                    "source_id": [101],
                    "source_type": ["drug"],
                    "target_id": ["P01"],
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
        mapper = IDMapper(processor_outputs, MOCK_CONFIG)
        collected = mapper._collected_entities

        # 唯一实体: 101, P01, P02
        self.assertEqual(len(collected), 3)
        self.assertSetEqual(collected[101]["types"], {"drug", "ligand"})
        self.assertSetEqual(collected[101]["sources"], {"bindingdb", "brenda"})
        self.assertSetEqual(collected["P01"]["sources"], {"bindingdb"})
        print("  ✅ Passed.")

    def test_finalization_with_type_merging_and_ids(self):
        print("\n--- Running Test: V5 Finalization with Type Merging ---")
        processor_outputs = {
            "source1": pd.DataFrame(
                {
                    "source_id": [101, 102],
                    "source_type": ["ligand", "ligand"],  # 【修复】102现在是ligand
                    "target_id": ["P01", "P01"],
                    "target_type": ["protein", "protein"],
                }
            ),
            "source2": pd.DataFrame(
                {
                    "source_id": [101],
                    "source_type": ["drug"],
                    "target_id": ["P02"],
                    "target_type": ["protein"],
                }
            ),
        }
        mapper = IDMapper(processor_outputs, MOCK_CONFIG)
        mapper.finalize_with_valid_entities({101, 102, "P01", "P02"})
        meta_101 = mapper.get_meta_by_auth_id(101)
        self.assertEqual(meta_101["type"], "drug")

        self.assertEqual(mapper.get_num_entities("drug"), 1)
        self.assertEqual(mapper.get_num_entities("protein"), 2)
        self.assertEqual(mapper.num_total_entities, 4)

        self.assertEqual(mapper.auth_id_to_logic_id_map[101], 0)
        self.assertIn(mapper.auth_id_to_logic_id_map["P01"], {1, 2})
        print("  ✅ Passed.")

    def test_finalization_raises_error_on_mixed_id_formats(self):
        print("\n--- Running Test: V5 Finalization handles mixed ID types ---")
        processor_outputs = {
            "source1": pd.DataFrame(
                {
                    "source_id": ["P01", 102],
                    "source_type": ["protein", "protein"],
                    "target_id": ["P02", "P02"],
                    "target_type": ["protein", "protein"],
                }
            ),
        }
        mapper = IDMapper(processor_outputs, MOCK_CONFIG)
        with self.assertRaisesRegex(
            ValueError, "ID format mismatch detected for entity type 'protein'"
        ):
            mapper.finalize_with_valid_entities({"P01", 102, "P02"})
        print("  ✅ Passed.")

    def test_query_apis(self):
        print("\n--- Running Test: V5 Query APIs ---")
        mapper = self._create_complex_finalized_mapper()

        # a. get_entity_meta
        meta_p01 = mapper.get_meta_by_logic_id(mapper.auth_id_to_logic_id_map["P01"])
        self.assertEqual(meta_p01["type"], "protein")
        self.assertSetEqual(meta_p01["sources"], {"bindingdb", "stringdb"})

        # b. get_ids_by_filter
        stringdb_ids = mapper.get_ids_by_filter(
            lambda meta: "stringdb" in meta["sources"]
        )
        # P01, P02, P03, P04 来自 stringdb -> 逻辑ID 1, 2, 3, 4
        self.assertCountEqual(stringdb_ids, [1, 2, 3, 4])

        drug_ids = mapper.get_ids_by_filter(lambda meta: meta["type"] == "drug")
        self.assertCountEqual(drug_ids, [0])
        print("  ✅ Passed.")

    def _create_complex_finalized_mapper(self) -> IDMapper:
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
                    "source_id": ["P01", "P02"],
                    "source_type": ["protein", "protein"],
                    "target_id": ["P03", "P04"],
                    "target_type": ["protein", "protein"],
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
        mapper = IDMapper(processor_outputs, MOCK_CONFIG)
        mapper.finalize_with_valid_entities({101, "P01", "P02", "P03", "P04"})
        return mapper


if __name__ == "__main__":
    unittest.main(verbosity=2)
