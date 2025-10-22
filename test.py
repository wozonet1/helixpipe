# 文件: minimal_pubchem_test.py (放置于项目根目录)

import sys
from pathlib import Path

# --- 动态路径设置 ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parent
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
except IndexError:
    raise RuntimeError("Could not determine project root.")

# --- 导入 ---
from nasnet.data_processing.services.canonicalizer import fetch_smiles_from_pubchem


def run_minimal_test():
    """
    一个最小化的、独立的测试，用于验证 PubChem SMILES 获取功能。
    """
    print("--- [Minimal PubChem Test] Starting ---")

    # 1. 准备测试ID
    test_cids = [
        2244,  # Aspirin
        5288826,  # Taxol
        999999999,  # 不存在的ID
    ]
    print(f"--> Input CIDs: {test_cids}")

    # 2. 调用函数，明确指定不使用代理 (proxies=None)
    results = fetch_smiles_from_pubchem(test_cids, proxies=None)

    print(f"--> Function returned {len(results)} results.")
    print(f"--> Results dict: {results}")

    # 3. 手动断言，验证结果
    success = True

    # a. 检查Aspirin
    if 2244 not in results:
        print("❌ FAIL: CID 2244 (Aspirin) not found in results.")
        success = False
    elif "CC(=O)OC1=CC=CC=C1C(=O)O" not in results[2244]:
        print(f"❌ FAIL: SMILES for Aspirin is incorrect. Got: {results[2244]}")
        success = False
    else:
        print("✅ PASS: Found correct SMILES for Aspirin.")

    # b. 检查Taxol
    if 5288826 not in results:
        print("❌ FAIL: CID 5288826 (Taxol) not found in results.")
        success = False
    else:
        print("✅ PASS: Found SMILES for Taxol.")

    # c. 检查不存在的ID
    if 999999999 in results:
        print("❌ FAIL: Non-existent CID 999999999 was unexpectedly found.")
        success = False
    else:
        print("✅ PASS: Non-existent CID was correctly ignored.")

    # d. 检查总数
    if len(results) != 2:
        print(f"❌ FAIL: Expected 2 results, but got {len(results)}.")
        success = False
    else:
        print("✅ PASS: Total number of results is correct.")

    print("\n--- [Minimal PubChem Test] Finished ---")
    if success:
        print(
            "\n🎉🎉🎉 FINAL RESULT: SUCCESS! The fetcher is working correctly. 🎉🎉🎉"
        )
    else:
        print(
            "\n🔥🔥🔥 FINAL RESULT: FAILURE. Please check the error messages above. 🔥🔥🔥"
        )


if __name__ == "__main__":
    run_minimal_test()
