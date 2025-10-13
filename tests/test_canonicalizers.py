# 文件：tests/test_canonicalizers.py
import os
from src.data_utils.canonicalizer import (
    canonicalize_smiles_to_cid,
    canonicalize_sequences_to_uniprot,
)
import research_template as rt

project_root = rt.get_project_root()
# 定义一个临时的缓存文件夹，用于本次测试，避免污染真实数据
TEST_CACHE_DIR = project_root / "tests" / "temp_cache"
TEST_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def test_smiles_to_cid():
    """
    测试SMILES到CID的转换函数。
    """
    print("\n" + "=" * 80)
    print(" " * 25 + "TESTING: SMILES to PubChem CID")
    print("=" * 80)

    # 1. 准备测试数据
    test_smiles = [
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        "c1ccccc1",  # Benzene
        "invalid_smiles_string",  # 一个无效的SMILES
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin (重复)
        None,  # 空值
        "CCO",  # Ethanol (与 OCC 等价)
        "OCC",  # Ethanol (另一个写法)
    ]

    # 2. 定义我们期望的正确答案
    expected_cid_map = {
        "CC(=O)Oc1ccccc1C(=O)O": 2244,
        "c1ccccc1": 241,
        "CCO": 702,
        "OCC": 702,
    }

    # 3. 定义缓存路径
    cache_file = TEST_CACHE_DIR / "smiles_test.pkl"
    if cache_file.exists():
        os.remove(cache_file)  # 确保每次测试都从一个干净的状态开始

    # 4. 调用被测试的函数 (第一次运行，会触发API调用和缓存写入)
    print("\n--- First run (cold cache) ---")
    result_map = canonicalize_smiles_to_cid(test_smiles, cache_path=cache_file)

    # 5. 验证结果 (使用assert)
    assert len(result_map) == len(expected_cid_map), (
        f"Expected {len(expected_cid_map)} results, but got {len(result_map)}"
    )

    for smiles, expected_cid in expected_cid_map.items():
        assert smiles in result_map, f"Expected SMILES '{smiles}' not found in results"
        assert result_map[smiles] == expected_cid, (
            f"For SMILES '{smiles}', expected CID {expected_cid}, but got {result_map[smiles]}"
        )

    assert "invalid_smiles_string" not in result_map, "Invalid SMILES should be ignored"

    print("✅ SMILES to CID: First run validation PASSED!")

    # 6. (可选但推荐) 测试缓存机制
    print("\n--- Second run (warm cache) ---")
    # 再次调用，这次它应该直接从缓存加载
    cached_result_map = canonicalize_smiles_to_cid(test_smiles, cache_path=cache_file)
    assert cached_result_map == result_map, (
        "Cached result does not match the original result"
    )
    print("✅ SMILES to CID: Cache mechanism validation PASSED!")

    # 7. 清理
    os.remove(cache_file)


def test_sequence_to_uniprot():
    """
    测试蛋白质序列到UniProt ID的转换函数。
    """
    print("\n" + "=" * 80)
    print(" " * 25 + "TESTING: Sequence to UniProt ID")
    print("=" * 80)

    # 1. 准备测试数据
    test_sequences = [
        "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEKLFNSLGK",  # 一个已知的、会成功的序列
        "A_VERY_FAKE_SEQUENCE_XYZ",  # 一个会失败的序列
    ]

    # 2. 定义期望的答案
    expected_uniprot_map = {"MKTAYIAKQRQISFVKSHFSRQLEERLGLIEKLFNSLGK": "P62158"}

    # 3. 定义缓存路径
    cache_file = TEST_CACHE_DIR / "sequence_test.pkl"
    if cache_file.exists():
        os.remove(cache_file)

    # 4. 调用被测试的函数
    result_map = canonicalize_sequences_to_uniprot(
        test_sequences, cache_path=cache_file
    )

    # 5. 验证结果
    assert len(result_map) == len(expected_uniprot_map), (
        f"Expected {len(expected_uniprot_map)} results, but got {len(result_map)}"
    )

    for seq, expected_id in expected_uniprot_map.items():
        assert result_map[seq] == expected_id, (
            f"For sequence '{seq[:10]}...', expected ID {expected_id}, but got {result_map[seq]}"
        )

    print("✅ Sequence to UniProt ID validation PASSED!")

    # 清理
    os.remove(cache_file)


if __name__ == "__main__":
    try:
        test_smiles_to_cid()
        test_sequence_to_uniprot()
        print("\n🎉 ALL TESTS PASSED! 🎉")
    finally:
        # 确保无论测试成功与否，都清理临时缓存文件夹
        import shutil

        if TEST_CACHE_DIR.exists():
            shutil.rmtree(TEST_CACHE_DIR)
            print(f"\n🧹 Cleaned up temporary cache directory: {TEST_CACHE_DIR}")
