import pandas as pd
from hydra import initialize, compose

# 导入我们要测试的类
from data_utils.id_mapper import IDMapper

# --- 准备测试数据 ---
# 我们可以创建一些小型的、人造的DataFrame来模拟真实数据
# 覆盖各种边缘情况
MOCK_DF_1 = pd.DataFrame(
    {
        "PubChem_CID": [101, 102, 103],
        "UniProt_ID": ["P1", "P2", "P1"],
        "SMILES": ["C", "CC", "CCC"],
        "Sequence": ["MA", "MV", "MAA"],
        "Label": [1, 1, 1],
    }
)

MOCK_DF_2 = pd.DataFrame(
    {
        "PubChem_CID": [102, 104],  # 包含一个重复的CID (102)
        "UniProt_ID": ["P3", "P2"],  # 包含一个重复的PID (P2)
        "SMILES": ["CCO", "CN"],
        "Sequence": ["MS", "MVC"],
        "Label": [1, 1],
    }
)

# --- 开始编写测试函数 ---


def test_initialization():
    """测试IDMapper是否能被正确初始化。"""
    with initialize(config_path="../conf", job_name="test_id_mapper"):
        cfg = compose(config_name="config")  # 加载默认配置

    mapper = IDMapper([MOCK_DF_1], cfg)

    assert mapper.num_molecules == 3
    assert mapper.num_proteins == 2
    assert mapper.num_total_entities == 5
    print("\n✅ test_initialization: PASSED")


def test_merging_logic():
    """测试IDMapper是否能正确合并来自多个DataFrame的实体。"""
    with initialize(config_path="../conf", job_name="test_id_mapper"):
        cfg = compose(config_name="config")

    mapper = IDMapper([MOCK_DF_1, MOCK_DF_2], cfg)

    # 预期结果：
    # CIDs: {101, 102, 103, 104} -> 4个分子
    # PIDs: {P1, P2, P3} -> 3个蛋白
    assert mapper.num_molecules == 4
    assert mapper.num_proteins == 3
    assert mapper.num_total_entities == 7
    print("✅ test_merging_logic: PASSED")


def test_id_ranges_and_consistency():
    """测试逻辑ID的分配是否正确（连续、分区）。"""
    with initialize(config_path="../conf", job_name="test_id_mapper"):
        cfg = compose(config_name="config")

    mapper = IDMapper([MOCK_DF_1], cfg)

    # 分子ID应该从0开始，连续
    assert sorted(mapper.cid_to_id.values()) == [0, 1, 2]
    # 蛋白质ID应该从分子的数量之后开始，连续
    assert sorted(mapper.uniprot_to_id.values()) == [3, 4]

    # 检查反向映射
    assert len(mapper.id_to_cid) == 3
    assert len(mapper.id_to_uniprot) == 2
    print("✅ test_id_ranges_and_consistency: PASSED")


def test_ordered_list_generation():
    """测试 get_ordered_... 方法返回的列表顺序是否正确。"""
    with initialize(config_path="../conf", job_name="test_id_mapper"):
        cfg = compose(config_name="config")

    mapper = IDMapper([MOCK_DF_1], cfg)

    # 获取排序后的CID: [101, 102, 103]
    sorted_cids = mapper._sorted_cids

    # 获取SMILES列表
    ordered_smiles = mapper.get_ordered_smiles()

    # 验证列表中的SMILES顺序是否与排序后的CID顺序一致
    expected_smiles = [
        MOCK_DF_1[MOCK_DF_1.PubChem_CID == cid].SMILES.iloc[0] for cid in sorted_cids
    ]
    assert ordered_smiles == expected_smiles
    print("✅ test_ordered_list_generation: PASSED")


if __name__ == "__main__":
    # 允许我们像普通脚本一样运行这个测试文件
    print("--- Running IDMapper Tests ---")
    test_initialization()
    test_merging_logic()
    test_id_ranges_and_consistency()
    test_ordered_list_generation()
    print("\n--- All tests completed. ---")
