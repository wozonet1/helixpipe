# tests/data_processing/services/test_id_mapper.py

import unittest

import pandas as pd
from omegaconf import OmegaConf

# 导入我们需要测试的类
from nasnet.data_processing.services.id_mapper import IDMapper

# --- 模拟 (Mock) 配置 ---
MOCK_CONFIG = OmegaConf.create(
    {
        "runtime": {"verbose": 0},
        "knowledge_graph": {
            "entity_types": {
                "drug": "drug",
                "ligand": "ligand",
                "protein": "protein",
                "molecule": "molecule",
            },
            "type_mapping_strategy": None,  # 默认不进行映射
            "type_merge_priority": {
                "drug": 0,
                "ligand": 1,
                "molecule": 9,
                "protein": 10,
            },
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


class TestIDMapper(unittest.TestCase):
    def test_initialization_and_collection(self):
        """测试场景1: IDMapper能否从输入中正确收集所有唯一的(id, type)对。"""
        print("\n--- Running Test: IDMapper Initialization & Collection ---")

        interactions = pd.DataFrame(
            {
                "source_id": [101, 102, 101, "P01"],
                "source_type": ["drug", "ligand", "drug", "protein"],
                "target_id": ["P01", "P01", "P02", "P02"],
                "target_type": ["protein", "protein", "protein", "protein"],
                "relation_type": ["rel1", "rel2", "rel3", "rel4"],
            }
        )

        mapper = IDMapper(interactions, MOCK_CONFIG)

        # 预期收集到的唯一对: (101, 'drug'), (102, 'ligand'), ('P01', 'protein'), ('P02', 'protein')
        self.assertEqual(len(mapper._collected_entities), 4)

        collected_set = {tuple(rec) for rec in mapper._collected_entities}
        expected_set = {
            (101, "drug"),
            (102, "ligand"),
            ("P01", "protein"),
            ("P02", "protein"),
        }
        self.assertSetEqual(collected_set, expected_set)
        print("  ✅ Passed.")

    def test_type_mapping_strategy(self):
        """测试场景2: type_mapping_strategy(消融实验)是否能正确工作。"""
        print("\n--- Running Test: Type Mapping Strategy (Ablation) ---")

        # 创建一个启用了类型映射的配置
        config = MOCK_CONFIG.copy()
        OmegaConf.update(
            config,
            "knowledge_graph.type_mapping_strategy",
            {"drug": "molecule", "ligand": "molecule"},
        )

        interactions = pd.DataFrame(
            {
                "source_id": [101, 102],
                "source_type": ["drug", "ligand"],
                "target_id": ["P01", "P01"],
                "target_type": ["protein", "protein"],
            }
        )

        mapper = IDMapper(interactions, config)

        # 预期: (101, 'drug') -> (101, 'molecule'), (102, 'ligand') -> (102, 'molecule')
        # 最终唯一对: (101, 'molecule'), (102, 'molecule'), ('P01', 'protein')
        self.assertEqual(len(mapper._collected_entities), 3)
        collected_set = {tuple(rec) for rec in mapper._collected_entities}
        expected_set = {(101, "molecule"), (102, "molecule"), ("P01", "protein")}
        self.assertSetEqual(collected_set, expected_set)
        print("  ✅ Passed.")

    def test_type_merging_and_finalization(self):
        """测试场景3: 核心功能 - 类型合并和最终的ID分配是否正确。"""
        print("\n--- Running Test: Type Merging & Finalization ---")

        interactions = pd.DataFrame(
            {
                "source_id": [101, 101, 102, 103, "P01"],
                "source_type": ["drug", "ligand", "ligand", "molecule", "protein"],
                "target_id": ["P01", "P02", "P01", "P02", "P02"],
                "target_type": ["protein", "protein", "protein", "protein", "protein"],
            }
        )

        mapper = IDMapper(interactions, MOCK_CONFIG)
        mapper.finalize_mappings()

        # 1. 验证类型合并结果
        # ID 101: 出现为 drug(0) 和 ligand(1)，最终应为 drug
        # ID 102: 只有 ligand(1)
        # ID 103: 只有 molecule(9)
        self.assertEqual(mapper._final_entity_map[101], "drug")
        self.assertEqual(mapper._final_entity_map[102], "ligand")
        self.assertEqual(mapper._final_entity_map[103], "molecule")

        # 2. 验证实体分组和数量
        self.assertListEqual(sorted(mapper.entities_by_type["drug"]), [101])
        self.assertListEqual(sorted(mapper.entities_by_type["ligand"]), [102])
        self.assertListEqual(sorted(mapper.entities_by_type["molecule"]), [103])
        self.assertListEqual(sorted(mapper.entities_by_type["protein"]), ["P01", "P02"])

        self.assertEqual(mapper.get_num_entities("drug"), 1)
        self.assertEqual(mapper.get_num_entities("ligand"), 1)
        self.assertEqual(mapper.get_num_entities("molecule"), 1)

        # 3. 验证ID分配的连续性和顺序 (drug -> ligand -> molecule -> protein)
        # drug (1个): ID 0
        # ligand (1个): ID 1
        # molecule (1个): ID 2
        # protein (2个): ID 3, 4
        self.assertEqual(mapper.entity_to_id_maps["drug"][101], 0)
        self.assertEqual(mapper.entity_to_id_maps["ligand"][102], 1)
        self.assertEqual(mapper.entity_to_id_maps["molecule"][103], 2)
        self.assertEqual(mapper.entity_to_id_maps["protein"]["P01"], 3)
        self.assertEqual(mapper.entity_to_id_maps["protein"]["P02"], 4)

        # 4. 验证反向映射
        self.assertEqual(mapper.logic_id_to_type_map[2], "molecule")
        self.assertEqual(mapper.logic_id_to_auth_id_map[4], "P02")
        print("  ✅ Passed.")

    def test_get_mapped_positive_pairs(self):
        """测试场景4: get_mapped_positive_pairs 是否能正确将DataFrame转换为逻辑ID对。"""
        print("\n--- Running Test: get_mapped_positive_pairs ---")

        interactions = pd.DataFrame(
            {
                "source_id": [101, 102, "P01"],
                "source_type": ["drug", "ligand", "protein"],
                "target_id": ["P01", "P01", "P02"],
                "target_type": ["protein", "protein", "protein"],
                "relation_type": ["inhibits", "binds_to", "associated_with"],
            }
        )

        mapper = IDMapper(interactions, MOCK_CONFIG)
        mapper.finalize_mappings()

        # drug(101): 0; ligand(102): 1; protein(P01, P02): 2, 3

        pairs_with_type, pairs_set = mapper.get_mapped_positive_pairs(interactions)

        self.assertEqual(len(pairs_with_type), 3)
        self.assertEqual(len(pairs_set), 3)

        expected_pairs_with_type = [
            (0, 2, "inhibits"),
            (1, 2, "binds_to"),
            (2, 3, "associated_with"),
        ]

        # to_records返回的是numpy记录数组，需要转换为元组列表
        self.assertCountEqual(
            [tuple(rec) for rec in pairs_with_type], expected_pairs_with_type
        )
        print("  ✅ Passed.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
