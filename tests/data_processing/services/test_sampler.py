# 文件: tests/data_processing/services/test_sampler.py (最终版)

import math
import unittest
from typing import List, Tuple

from omegaconf import OmegaConf

# 导入我们新的、统一的采样函数
from nasnet.data_processing.services.sampler import sample_interactions

# --- 模拟(Mock)对象和数据 (与之前版本相同) ---


class MockIDMapper:
    """一个轻量级的IDMapper模拟对象。"""

    def __init__(self, num_drugs: int, num_ligands: int):
        self.num_drugs = num_drugs
        self.num_ligands = num_ligands
        self.num_molecules = num_drugs + num_ligands


# 全局模拟数据
NUM_DRUGS, NUM_LIGANDS = 100, 10
mock_id_mapper = MockIDMapper(NUM_DRUGS, NUM_LIGANDS)
MOCK_DRUG_PAIRS = [(i, 1000 + i, "interacts_with") for i in range(NUM_DRUGS)]
MOCK_LIGAND_PAIRS = [(NUM_DRUGS + i, 2000 + i, "inhibits") for i in range(NUM_LIGANDS)]
MOCK_ALL_PAIRS = MOCK_DRUG_PAIRS + MOCK_LIGAND_PAIRS


# --- 测试用例类 ---
class TestInteractionSampler(unittest.TestCase):
    def test_sampling_disabled(self):
        """场景1: 当采样开关关闭时，应返回所有原始数据。"""
        print("\n--- Running Test: Sampling Disabled ---")
        cfg = OmegaConf.create(
            {"data_params": {"sampling": {"enabled": False}}, "runtime": {"seed": 42}}
        )
        sampled_pairs, _ = sample_interactions(MOCK_ALL_PAIRS, mock_id_mapper, cfg)
        self.assertEqual(len(sampled_pairs), len(MOCK_ALL_PAIRS))
        print("  ✅ Passed.")

    def test_uniform_sampling_only(self):
        """场景2: 只进行统一采样 (fraction)。"""
        print("\n--- Running Test: Uniform Sampling Only ---")
        cfg = OmegaConf.create(
            {
                "data_params": {"sampling": {"enabled": True, "fraction": 0.5}},
                "runtime": {"seed": 42},
            }
        )
        sampled_pairs, _ = sample_interactions(MOCK_ALL_PAIRS, mock_id_mapper, cfg)
        expected_len = math.ceil(len(MOCK_ALL_PAIRS) * 0.5)
        self.assertEqual(len(sampled_pairs), expected_len)
        print("  ✅ Passed.")

    def test_stratified_sampling_only(self):
        """场景3: 只进行分层采样 (ratio)。"""
        print("\n--- Running Test: Stratified Sampling Only (1:1 ratio) ---")
        cfg = OmegaConf.create(
            {
                "data_params": {
                    "sampling": {"enabled": True, "drug_to_ligand_ratio": 1.0}
                },
                "runtime": {"seed": 42},
            }
        )
        sampled_pairs, _ = sample_interactions(MOCK_ALL_PAIRS, mock_id_mapper, cfg)
        expected_len = NUM_LIGANDS + int(NUM_LIGANDS * 1.0)  # 10 ligands + 10 drugs
        self.assertEqual(len(sampled_pairs), expected_len)
        self._assert_ratio(sampled_pairs, 1.0)
        print("  ✅ Passed.")

    def test_sequential_sampling_stratified_then_uniform(self):
        """【核心】场景4: 测试串联采样，先分层，再统一。"""
        print("\n--- Running Test: Sequential Sampling (Stratified then Uniform) ---")
        cfg = OmegaConf.create(
            {
                "data_params": {
                    "sampling": {
                        "enabled": True,
                        "drug_to_ligand_ratio": 5.0,  # 5:1 ratio
                        "fraction": 0.5,  # 再取其中的50%
                    }
                },
                "runtime": {"seed": 42},
            }
        )

        sampled_pairs, _ = sample_interactions(MOCK_ALL_PAIRS, mock_id_mapper, cfg)

        # 步骤1 (分层) 的中间结果: 10 (L) + 10*5 (D) = 60 pairs
        intermediate_len = NUM_LIGANDS + int(NUM_LIGANDS * 5.0)
        # 步骤2 (统一) 的最终结果: 60 * 0.5 = 30 pairs
        expected_len = math.ceil(intermediate_len * 0.5)

        self.assertEqual(len(sampled_pairs), expected_len)

        # 验证最终结果的比例是否仍然近似于5:1
        # 预期: 5 (L) + 25 (D) = 30 pairs
        num_ligands_in_sample = self._count_ligands(sampled_pairs)
        num_drugs_in_sample = self._count_drugs(sampled_pairs)

        self.assertEqual(num_ligands_in_sample, math.ceil(NUM_LIGANDS * 0.5))
        self.assertEqual(num_drugs_in_sample, math.ceil((NUM_LIGANDS * 5.0) * 0.5))
        print("  ✅ Passed.")

    def test_stratified_sampling_ratio_overflow(self):
        """场景5: 边缘情况 - 分层采样的比例超过上限。"""
        print("\n--- Running Test: Stratified Sampling (Ratio Overflow) ---")
        cfg = OmegaConf.create(
            {
                "data_params": {
                    "sampling": {"enabled": True, "drug_to_ligand_ratio": 100.0}
                },
                "runtime": {"seed": 42},
            }
        )
        sampled_pairs, _ = sample_interactions(MOCK_ALL_PAIRS, mock_id_mapper, cfg)
        expected_len = NUM_LIGANDS + NUM_DRUGS  # 应使用所有 drug
        self.assertEqual(len(sampled_pairs), expected_len)
        self.assertEqual(self._count_drugs(sampled_pairs), NUM_DRUGS)
        print("  ✅ Passed.")

    def test_no_ligands_for_stratified_sampling(self):
        """场景6: 边缘情况 - 进行分层采样时，没有ligand数据。"""
        print("\n--- Running Test: Stratified Sampling (No Ligands) ---")
        cfg = OmegaConf.create(
            {
                "data_params": {
                    "sampling": {"enabled": True, "drug_to_ligand_ratio": 1.0}
                },
                "runtime": {"seed": 42},
            }
        )
        # 只传入 drug pairs
        sampled_pairs, _ = sample_interactions(MOCK_DRUG_PAIRS, mock_id_mapper, cfg)

        # 预期: 无法进行分层，应返回所有 drug pairs
        self.assertEqual(len(sampled_pairs), NUM_DRUGS)
        self.assertEqual(self._count_ligands(sampled_pairs), 0)
        print("  ✅ Passed.")

    # --- 辅助断言方法 ---
    def _count_drugs(self, pairs: List[Tuple]) -> int:
        return sum(1 for u, v, r in pairs if u < mock_id_mapper.num_drugs)

    def _count_ligands(self, pairs: List[Tuple]) -> int:
        return sum(1 for u, v, r in pairs if u >= mock_id_mapper.num_drugs)

    def _assert_ratio(self, pairs: List[Tuple], ratio: float):
        num_l = self._count_ligands(pairs)
        num_d = self._count_drugs(pairs)
        if num_l > 0:
            self.assertAlmostEqual(num_d / num_l, ratio)


if __name__ == "__main__":
    unittest.main(verbosity=2)
