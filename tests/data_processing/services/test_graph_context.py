import logging
import unittest

import pandas as pd
import torch

from helixpipe.configs import register_all_schemas
from helixpipe.data_processing.services.graph_context import (
    build_local_id_mapping,
    build_local_id_to_type,
    convert_dataframe_to_global,
    convert_ids_to_local,
    convert_pairs_to_local,
    slice_embeddings,
)

# 导入所有需要的真实模块和配置类
from helixpipe.typing import AppConfig

# FIXME: helper导入
from tests.helper import create_test_config

# --- 模拟依赖 (Mocks) ---
logging.basicConfig(level=logging.DEBUG)


class MockIDMapper:
    """一个为 graph_context 测试量身定做的 Mock IDMapper。"""

    def __init__(self, num_mols, num_prots):
        self.num_molecules = num_mols
        self.num_proteins = num_prots

        self.logic_id_to_type_map = {}

        # 定义一个明确的 drug/ligand 划分，与我们的测试用例保持一致
        drug_ids = {1, 3, 8}
        ligand_ids = {9}

        for i in range(num_mols):
            if i in drug_ids:
                self.logic_id_to_type_map[i] = "drug"
            elif i in ligand_ids:
                self.logic_id_to_type_map[i] = "ligand"
            else:
                self.logic_id_to_type_map[i] = "drug"

        for i in range(num_mols, num_mols + num_prots):
            self.logic_id_to_type_map[i] = "protein"

    def get_meta_by_logic_id(self, logic_id: int) -> dict:
        """返回实体的元数据，包含 type 字段。"""
        entity_type = self.logic_id_to_type_map.get(logic_id, "unknown")
        return {"type": entity_type}

    def is_molecule(self, entity_type: str) -> bool:
        return entity_type in ["drug", "ligand"]

    def is_protein(self, entity_type: str) -> bool:
        return entity_type == "protein"


register_all_schemas()
MOCK_CONFIG: AppConfig = create_test_config({"runtime": {"verbose": 0}})


class TestGraphContext(unittest.TestCase):
    def setUp(self):
        """准备一个通用的测试环境。"""
        print("\n" + "=" * 80)

        # 1. 模拟一个全局环境
        self.global_num_mols = 10
        self.global_num_prots = 5
        self.mock_global_id_mapper = MockIDMapper(
            self.global_num_mols, self.global_num_prots
        )

        # 模拟全局特征嵌入 (10个分子, 5个蛋白, 每个8维)
        self.global_mol_embeddings = torch.randn(self.global_num_mols, 8)
        self.global_prot_embeddings = torch.randn(self.global_num_prots, 8)

        # 2. 定义本次"施工"相关的实体范围 (使用全局逻辑ID)
        # 相关的分子ID: 1, 3, 8 (drug), 9 (ligand)
        # 相关的蛋白ID: 10, 12 (全局蛋白ID从10开始)
        self.relevant_mol_ids = {1, 3, 8, 9}
        self.relevant_prot_ids = {10, 12}

        # 3. 构建映射
        print("--- Running Test: build_local_id_mapping ---")
        self.g2l, self.l2g, self.num_local_mols = build_local_id_mapping(
            self.relevant_mol_ids, self.relevant_prot_ids
        )

    def test_initialization_counts_and_mappings(self):
        """测试点1: 初始化后，局部实体数量和ID映射是否正确。"""
        # a. 验证数量
        self.assertEqual(self.num_local_mols, 4)
        self.assertEqual(len(self.l2g) - self.num_local_mols, 2)

        # b. 验证从全局到局部的映射
        # 全局分子ID 1, 3, 8, 9 应该被映射为局部ID 0, 1, 2, 3
        self.assertEqual(self.g2l[1], 0)
        self.assertEqual(self.g2l[3], 1)
        self.assertEqual(self.g2l[8], 2)
        self.assertEqual(self.g2l[9], 3)
        # 全局蛋白ID 10, 12 应该接在后面，被映射为局部ID 4, 5
        self.assertEqual(self.g2l[10], 4)
        self.assertEqual(self.g2l[12], 5)

        # c. 验证从局部到全局的映射
        self.assertEqual(self.l2g[1], 3)  # local 1 -> global 3
        self.assertEqual(self.l2g[4], 10)  # local 4 -> global 10
        print("  ✅ Passed (1): Counts and ID mappings are correct.")

    def test_local_embeddings_slicing(self):
        """测试点2: 局部的特征嵌入是否被正确地筛选出来。"""
        local_mol_emb, local_prot_emb = slice_embeddings(
            self.l2g,
            self.num_local_mols,
            self.global_mol_embeddings,
            self.global_prot_embeddings,
            self.global_num_mols,
        )

        # a. 验证分子嵌入
        self.assertEqual(local_mol_emb.shape, (4, 8))
        # 验证内容：局部嵌入的第一行，应该等于全局嵌入中索引为1的那一行
        # sorted_relevant_mols is [1, 3, 8, 9]
        self.assertTrue(torch.equal(local_mol_emb[0], self.global_mol_embeddings[1]))
        self.assertTrue(torch.equal(local_mol_emb[3], self.global_mol_embeddings[9]))

        # b. 验证蛋白嵌入
        self.assertEqual(local_prot_emb.shape, (2, 8))
        # 验证内容：全局蛋白ID 10, 12 对应于 global_prot_embeddings 的索引 0, 2
        # sorted_relevant_prots is [10, 12]
        self.assertTrue(torch.equal(local_prot_emb[0], self.global_prot_embeddings[0]))
        self.assertTrue(torch.equal(local_prot_emb[1], self.global_prot_embeddings[2]))
        print("  ✅ Passed (2): Local embeddings are sliced correctly.")

    def test_local_type_mapping(self):
        """测试点3: 局部的ID到类型的映射是否正确。"""
        local_id_to_type = build_local_id_to_type(self.l2g, self.mock_global_id_mapper)
        # 局部ID 2 -> 全局ID 8 (drug)
        self.assertEqual(local_id_to_type[2], "drug")
        # 局部ID 3 -> 全局ID 9 (ligand)
        self.assertEqual(local_id_to_type[3], "ligand")
        # 局部ID 5 -> 全局ID 12 (protein)
        self.assertEqual(local_id_to_type[5], "protein")
        print("  ✅ Passed (3): Local type mapping is correct.")

    def test_id_conversion_methods(self):
        """测试点4: ID空间转换工具函数是否工作正常。"""
        print("--- Running Test: ID Conversion Functions ---")

        # a. 测试 convert_pairs_to_local
        global_pairs = [
            (1, 10, "rel1"),  # 应该被保留并转换为 (0, 4)
            (8, 12, "rel2"),  # 应该被保留并转换为 (2, 5)
            (1, 99, "rel3"),  # 应该被丢弃，因为99不在相关集合中
        ]
        local_pairs = convert_pairs_to_local(global_pairs, self.g2l)
        self.assertEqual(len(local_pairs), 2)
        self.assertIn((0, 4, "rel1"), local_pairs)
        self.assertIn((2, 5, "rel2"), local_pairs)

        # b. 测试 convert_dataframe_to_global
        local_df = pd.DataFrame({"source": [0, 2], "target": [4, 5]})
        global_df = convert_dataframe_to_global(local_df, "source", "target", self.l2g)
        self.assertEqual(global_df["source"].iloc[0], 1)
        self.assertEqual(global_df["target"].iloc[1], 12)

        # c. 测试 convert_ids_to_local
        global_id_set = {1, 12, 99}
        local_id_set = convert_ids_to_local(global_id_set, self.g2l)
        self.assertSetEqual(local_id_set, {0, 5})
        print("  ✅ Passed (4): All ID conversion functions work correctly.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
