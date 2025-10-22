# 文件: exploration_brenda.py (放置于项目根目录)

import json
import sys
from pathlib import Path

# --- [核心修正] 配置 ---
# 路径已根据您的项目结构更新
BRENDA_JSON_PATH = Path("data/brenda/raw/brenda_2025_1.json")  # <-- 已更正


def explore_brenda_structure():
    """
    加载BRENDA JSON文件并打印单个酶条目的结构。
    """
    print("--- [BRENDA Explorer V2] Starting ---")

    if not BRENDA_JSON_PATH.exists():
        print(f"❌ ERROR: BRENDA file not found at: {BRENDA_JSON_PATH}")
        print("   Please ensure the file exists at the specified path.")
        sys.exit(1)

    # --- [核心修正] 1. 直接加载JSON文件 ---
    print(f"--> Loading '{BRENDA_JSON_PATH.name}'... (This may take a moment)")
    try:
        # 使用标准的 open() 读取未压缩的JSON文件
        with open(BRENDA_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        print("--> File loaded successfully.")
    except Exception as e:
        print(f"❌ ERROR: Failed to load JSON file. Reason: {e}")
        return

    # --- 2. 验证顶层键 (逻辑不变) ---
    top_level_keys = list(data.keys())
    print("\n--- Top-Level Keys ---")
    print(top_level_keys)
    assert "release" in top_level_keys
    assert "version" in top_level_keys
    assert "data" in top_level_keys
    print("✅ Top-level structure is as expected.")

    # --- 3. 检查 'data' 对象 (逻辑不变) ---
    enzyme_data = data.get("data", {})
    print("\n--- 'data' Object Inspection ---")
    print(f"Type of 'data' object: {type(enzyme_data)}")

    if isinstance(enzyme_data, dict):
        num_enzymes = len(enzyme_data)
        print(f"Number of enzyme entries found: {num_enzymes}")

        ec_numbers = list(enzyme_data.keys())
        print(f"First 5 EC numbers: {ec_numbers[:5]}")

        # --- 4. 深入研究单个酶 (逻辑不变) ---
        target_ec = "1.1.1.1"  # Alcohol dehydrogenase
        if target_ec in enzyme_data:
            print(f"\n--- Structure of a Single Enzyme (EC: {target_ec}) ---")
            single_enzyme_entry = enzyme_data[target_ec]
            print(json.dumps(single_enzyme_entry, indent=2))
        else:
            print(
                f"⚠️ WARNING: Target EC number '{target_ec}' not found. Printing the first available entry instead."
            )
            if ec_numbers:
                first_ec = ec_numbers[0]
                print(f"\n--- Structure of a Single Enzyme (EC: {first_ec}) ---")
                print(json.dumps(enzyme_data[first_ec], indent=2))


if __name__ == "__main__":
    explore_brenda_structure()
