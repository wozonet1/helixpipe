import unittest
from unittest.mock import patch

from omegaconf import OmegaConf

# 导入我们需要测试的类
from nasnet.data_processing.services.splitter import DataSplitter

# --- [MODIFIED] 模拟(Mock)对象和数据 ---


class MockIDMapper:
    """
    一个更复杂的IDMapper模拟对象，用于测试位置无知的划分。
    """

    def __init__(self):
        self.entity_types = ["drug", "ligand", "protein"]

        self.entities_by_type = {
            "drug": [0, 1],
            "ligand": [2, 3],
            "protein": [10, 11, 12, 13],
        }

        self.logic_id_to_type_map = {}
        for type_name, ids in self.entities_by_type.items():
            for entity_id in ids:
                self.logic_id_to_type_map[entity_id] = type_name

    def get_entity_types(self):
        return self.entity_types

    def get_ordered_ids(self, entity_type: str):
        return self.entities_by_type.get(entity_type, [])

    def is_molecule(self, entity_type: str) -> bool:
        return entity_type in ["drug", "ligand"]

    def is_protein(self, entity_type: str) -> bool:
        return entity_type == "protein"

    def get_logic_id_to_type_map(self):
        return self.logic_id_to_type_map


# --- 全局的模拟数据 ---
mock_id_mapper = MockIDMapper()

# 创建一个复杂的交互对列表
# 2个 drug-prot, 2个 ligand-prot, 1个 prot-prot
MOCK_PAIRS = [
    # DTI
    (0, 10, "interacts_with"),  # (drug, protein)
    (1, 11, "interacts_with"),  # (drug, protein)
    # LPI
    (2, 12, "interacts_with"),  # (ligand, protein)
    (3, 13, "interacts_with"),  # (ligand, protein)
    # PPI
    (10, 11, "associated_with"),  # (protein, protein)
]


class TestDataSplitter(unittest.TestCase):
    def test_molecule_cold_split_heterogeneous(self):
        """
        测试点1: 在分子冷启动模式下，是否能正确划分一个包含多种分子类型的交互列表。
        """
        print("\n--- Running Test: Molecule Cold-Split on Heterogeneous Pairs ---")

        cfg = OmegaConf.create(
            {
                "training": {
                    "k_folds": 1,  # 单次划分
                    "coldstart": {"mode": "molecule", "test_fraction": 0.5},
                },
            }
        )

        # 种子固定，使得 train_test_split 的结果可预测
        # 分子ID: [0, 1, 2, 3]。test_fraction=0.5 -> 2个测试分子
        # 假设 train_test_split 会选择 [1, 2] 作为测试分子
        splitter = DataSplitter(cfg, MOCK_PAIRS, mock_id_mapper, seed=42)

        _, _, test_pairs, cold_start_mol_ids = next(iter(splitter))

        # 预期:
        # (0, 10, 'interacts_with') -> train
        # (1, 11, 'interacts_with') -> test (因为分子1在测试集)
        # (2, 12, 'interacts_with') -> test (因为分子2在测试集)
        # (3, 13, 'interacts_with') -> train
        # (10, 11, 'associated_with') -> train (不含任何分子)

        self.assertEqual(len(test_pairs), 2)
        self.assertEqual(
            len(cold_start_mol_ids), 2, "Should identify 2 cold-start molecules."
        )

        test_mol_ids_from_pairs = set()
        logic_id_to_type = mock_id_mapper.get_logic_id_to_type_map()
        for u, v, _ in test_pairs:
            if mock_id_mapper.is_molecule(logic_id_to_type[u]):
                test_mol_ids_from_pairs.add(u)
            if mock_id_mapper.is_molecule(logic_id_to_type[v]):
                test_mol_ids_from_pairs.add(v)

        self.assertEqual(
            test_mol_ids_from_pairs,
            cold_start_mol_ids,
            "All molecules in test pairs must be from the cold-start set.",
        )

        print("  ✅ Test Passed: Correctly isolated and returned molecule-cold IDs.")

    @patch("nasnet.data_processing.services.splitter.train_test_split")
    def test_protein_cold_split_handles_ppi(self, mock_train_test_split):
        """
        测试点2 (核心): 在蛋白质冷启动模式下，是否能正确处理 protein-protein 交互。
        """
        print("\n--- Running Test: Protein Cold-Split with PPI ---")

        cfg = OmegaConf.create(
            {
                "training": {
                    "k_folds": 1,
                    "coldstart": {"mode": "protein", "test_fraction": 0.5},
                },
            }
        )
        train_entities = [10, 13]
        test_entities = [11, 12]
        mock_train_test_split.return_value = (train_entities, test_entities)
        # 蛋白质ID: [10, 11, 12, 13]。test_fraction=0.5 -> 2个测试蛋白
        # 假设 train_test_split (seed=1) 会选择 [11, 12] 作为测试蛋白
        splitter = DataSplitter(cfg, MOCK_PAIRS, mock_id_mapper, seed=1)

        _, train_pairs, test_pairs, cold_start_prot_ids = next(iter(splitter))

        # 预期:
        # (0, 10, 'interacts_with') -> train
        # (1, 11, 'interacts_with') -> test (因为蛋白11在测试集)
        # (2, 12, 'interacts_with') -> test (因为蛋白12在测试集)
        # (3, 13, 'interacts_with') -> train
        # (10, 11, 'associated_with') -> test (因为蛋白11在测试集) <-- 核心验证点！

        self.assertEqual(len(train_pairs), 2, "Incorrect number of training pairs.")
        self.assertEqual(len(test_pairs), 3, "Incorrect number of test pairs.")

        test_prot_ids_found = set()
        for u, v, _ in test_pairs:
            if mock_id_mapper.is_protein(mock_id_mapper.logic_id_to_type_map[u]):
                test_prot_ids_found.add(u)
            if mock_id_mapper.is_protein(mock_id_mapper.logic_id_to_type_map[v]):
                test_prot_ids_found.add(v)

        # 验证 (10, 11) 这条边确实被分到了测试集
        self.assertEqual(
            len(cold_start_prot_ids), 2, "Should identify 2 cold-start proteins."
        )
        self.assertEqual(
            cold_start_prot_ids,
            {11, 12},
            "The returned cold-start protein IDs are not as expected.",
        )  # 依赖于mock的train_test_split
        print("  ✅ Test Passed: Correctly handled PPI pair during protein cold-split.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
