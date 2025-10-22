# 文件: src/nasnet/analysis/scripts/analyze_brenda_schema.py (最终探索版)

import json
import sys
from collections import defaultdict

import research_template as rt

# --- 配置 ---
# 这个脚本是独立的，所以我们硬编码路径。请确保它相对于项目根目录是正确的。
PROJECT_ROOT = rt.get_project_root()
BRENDA_JSON_PATH = PROJECT_ROOT / "data/brenda/raw/brenda_2025_1.json"


def analyze_brenda():
    """
    系统性地分析BRENDA JSON文件的结构，重点关注人类蛋白质和UniProt ID的可用性。
    """
    print("--- [BRENDA Schema Analyzer] Starting ---")

    if not BRENDA_JSON_PATH.exists():
        print(f"❌ ERROR: BRENDA file not found at: {BRENDA_JSON_PATH}")
        sys.exit(1)

    # --- 1. 加载数据 ---
    print(f"--> Loading '{BRENDA_JSON_PATH.name}'...")
    try:
        with open(BRENDA_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f).get("data", {})
        print(f"--> File loaded. Found {len(data)} total EC entries.")
    except Exception as e:
        print(f"❌ ERROR: Failed to load JSON file. Reason: {e}")
        return

    # --- 2. 初始化统计变量 ---
    ec_with_human_count = 0
    human_protein_entries_total = 0
    human_with_accessions = 0
    human_without_accessions = 0
    source_field_counts = defaultdict(int)

    # 用于抽样打印的列表
    samples_with_accessions = []
    samples_without_accessions = []

    # --- 3. 遍历所有EC条目进行分析 ---
    print("--> Analyzing all EC entries for 'Homo sapiens' protein data...")
    for ec_id, entry in data.items():
        if not isinstance(entry, dict) or "protein" not in entry:
            continue

        found_human_in_ec = False
        for prot_internal_id, prot_info in entry.get("protein", {}).items():
            if not isinstance(prot_info, dict):
                continue

            organism = prot_info.get("organism", "").lower()

            if "homo sapiens" in organism:
                found_human_in_ec = True
                human_protein_entries_total += 1

                # 核心分析逻辑
                if "accessions" in prot_info and prot_info["accessions"]:
                    human_with_accessions += 1
                    source = prot_info.get("source", "N/A")
                    source_field_counts[source] += 1
                    # 存储样本以供后续打印
                    if len(samples_with_accessions) < 5:
                        samples_with_accessions.append(
                            {
                                "ec_id": ec_id,
                                "internal_id": prot_internal_id,
                                "data": prot_info,
                            }
                        )
                else:
                    human_without_accessions += 1
                    if len(samples_without_accessions) < 5:
                        samples_without_accessions.append(
                            {
                                "ec_id": ec_id,
                                "internal_id": prot_internal_id,
                                "data": prot_info,
                            }
                        )

        if found_human_in_ec:
            ec_with_human_count += 1

    # --- 4. 打印最终的侦察报告 ---
    print("\n" + "=" * 80)
    print(" " * 25 + "BRENDA HUMAN PROTEIN REPORT")
    print("=" * 80)

    print("\n--- [Overall Statistics] ---")
    print(f"Total EC Entries Analyzed:        {len(data)}")
    print(f"EC Entries Mentioning 'Homo sapiens': {ec_with_human_count}")

    print("\n--- [Human Protein Entry Details] ---")
    print(f"Total 'Homo sapiens' Protein Entries: {human_protein_entries_total}")
    print(f"  - Entries WITH 'accessions' key:    {human_with_accessions}")
    print(f"  - Entries WITHOUT 'accessions' key: {human_without_accessions}")

    if human_with_accessions > 0:
        success_rate = (human_with_accessions / human_protein_entries_total) * 100
        print(f"  - Direct UniProt ID Availability: {success_rate:.2f}%")

    print("\n--- ['source' Field Distribution (for entries with accessions)] ---")
    if not source_field_counts:
        print("  No 'source' field found in any entry with accessions.")
    else:
        for source, count in source_field_counts.items():
            print(f"  - '{source}': {count} times")

    print("\n" + "=" * 80)
    print(" " * 30 + "SAMPLE ENTRIES")
    print("=" * 80)

    print("\n--- [SAMPLES] Human Protein Entries WITH 'accessions' key ---")
    if not samples_with_accessions:
        print("  None found.")
    else:
        for sample in samples_with_accessions:
            print(f"  EC: {sample['ec_id']}, Internal ID: {sample['internal_id']}")
            print(json.dumps(sample["data"], indent=4))
            print("-" * 20)

    print("\n--- [SAMPLES] Human Protein Entries WITHOUT 'accessions' key ---")
    if not samples_without_accessions:
        print("  None found.")
    else:
        for sample in samples_without_accessions:
            print(f"  EC: {sample['ec_id']}, Internal ID: {sample['internal_id']}")
            print(json.dumps(sample["data"], indent=4))
            print("-" * 20)

    print("\n--- [Analyzer] Finished ---")


if __name__ == "__main__":
    analyze_brenda()
