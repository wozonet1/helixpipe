import pytest
from omegaconf import OmegaConf

# 导入要测试的类
from data_utils.splitters import DataSplitter

# --- 模拟(Mock)对象和数据 ---


class MockIDMapper:
    """一个轻量级的IDMapper模拟对象，只提供必要的方法和属性。"""

    def __init__(self, num_mols, num_prots):
        self.molecule_to_id = {f"CID_{i}": i for i in range(num_mols)}
        self.protein_to_id = {f"PID_{i}": i + num_mols for i in range(num_prots)}


# 全局的模拟数据，供所有测试函数使用
NUM_MOLS, NUM_PROTS = 10, 10
mock_id_mapper = MockIDMapper(NUM_MOLS, NUM_PROTS)
# 创建一个 10x10 的交互网格，总共100个交互对
MOCK_PAIRS = [
    (m, p)
    for m in mock_id_mapper.molecule_to_id.values()
    for p in mock_id_mapper.protein_to_id.values()
]

# --- 测试函数 ---


def test_k5_random_split():
    """测试 5-fold 随机划分 (热启动)。"""
    cfg = OmegaConf.create(
        {
            "training": {
                "k_folds": 5,
                "coldstart": {"mode": "random", "test_fraction": 0.3},
            },
            "runtime": {"seed": 42},
        }
    )
    splitter = DataSplitter(cfg, MOCK_PAIRS, mock_id_mapper)

    all_test_pairs_from_folds = []
    fold_count = 0
    for fold_idx, train_pairs, test_pairs in splitter:
        assert len(train_pairs) == 80  # 100 * 4/5
        assert len(test_pairs) == 20  # 100 * 1/5
        all_test_pairs_from_folds.extend(test_pairs)
        fold_count += 1

    assert fold_count == 5  # 确保迭代了5次
    # 检查所有折的测试集是否能无重复、无遗漏地重构原始数据集
    assert len(set(all_test_pairs_from_folds)) == len(MOCK_PAIRS)
    assert sorted(all_test_pairs_from_folds) == sorted(MOCK_PAIRS)
    print("\n✅ test_k5_random_split: PASSED")


def test_k5_molecule_cold_split():
    """测试 5-fold 分子冷启动划分。"""
    cfg = OmegaConf.create(
        {
            "training": {
                "k_folds": 5,
                "coldstart": {"mode": "molecule", "test_fraction": 0.3},
            },
            "runtime": {"seed": 42},
        }
    )
    splitter = DataSplitter(cfg, MOCK_PAIRS, mock_id_mapper)

    all_test_mols_seen = set()
    total_test_pairs_count = 0
    for fold_idx, train_pairs, test_pairs in splitter:
        test_mols_in_fold = {p[0] for p in test_pairs}
        train_mols_in_fold = {p[0] for p in train_pairs}

        # 每一折应该包含 10/5=2 个测试分子
        assert len(test_mols_in_fold) == 2
        # 训练集中的分子不应该出现在测试集中 (冷启动的核心)
        assert not (train_mols_in_fold & test_mols_in_fold)
        # 每个测试分子对应10个蛋白质交互
        assert len(test_pairs) == 2 * NUM_PROTS

        all_test_mols_seen.update(test_mols_in_fold)
        total_test_pairs_count += len(test_pairs)

    # 5折下来，所有分子都应该被作为测试集出现过一次
    assert all_test_mols_seen == set(mock_id_mapper.molecule_to_id.values())
    assert total_test_pairs_count == len(MOCK_PAIRS)
    print("✅ test_k5_molecule_cold_split: PASSED")


def test_single_split_protein_cold():
    """测试 k=1 的单次蛋白质冷启动划分。"""
    cfg = OmegaConf.create(
        {
            "training": {
                "k_folds": 1,
                "coldstart": {"mode": "protein", "test_fraction": 0.3},
            },
            "runtime": {"seed": 42},
        }
    )
    splitter = DataSplitter(cfg, MOCK_PAIRS, mock_id_mapper)

    splits = list(splitter)  # 将迭代器转换为列表
    assert len(splits) == 1  # 确认只产生了一次划分

    fold_idx, train_pairs, test_pairs = splits[0]

    test_prots_in_fold = {p[1] for p in test_pairs}

    # 测试蛋白质的数量应该是 10 * 0.3 = 3
    assert len(test_prots_in_fold) == 3
    # 测试交互对的数量应该是 3 * 10 (每个蛋白有10个分子交互)
    assert len(test_pairs) == 3 * NUM_MOLS
    # 训练交互对的数量应该是 (10-3) * 10 = 70
    assert len(train_pairs) == (NUM_PROTS - 3) * NUM_MOLS
    print("✅ test_single_split_protein_cold: PASSED")


if __name__ == "__main__":
    print("--- Running DataSplitter Tests ---")
    test_k5_random_split()
    test_k5_molecule_cold_split()
    test_single_split_protein_cold()
    print("\n--- All DataSplitter tests completed successfully. ---")
