# 文件: src/helixpipe/analysis/scripts/analyze_uniprot_id_formats.py (全新)

import re
import sys

import pandas as pd
from tqdm import tqdm

import helixlib as hx

# --- 动态路径设置 ---
try:
    PROJECT_ROOT = hx.get_project_root()
except IndexError:
    raise RuntimeError("Could not determine project root.")

# --- 配置 ---
UNIPROT_WHITELIST_PATH = (
    PROJECT_ROOT / "data" / "assets" / "uniprotkb_proteome_UP000005640.tsv"
)

# --- 定义我们要测试的正则表达式 ---
REGEX_PATTERNS = {
    "Strict (Official)": re.compile(
        r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$",
        re.IGNORECASE,
    ),
    "General (Simplified)": re.compile(r"^[A-Z][0-9][A-Z0-9]{4,9}$", re.IGNORECASE),
    "Loose (Len 6-10)": re.compile(r"^[A-Z0-9]{6,10}$", re.IGNORECASE),
}


def analyze_formats() -> None:
    """
    分析权威白名单中的UniProt ID格式，并评估不同正则表达式的覆盖率。
    """
    print("--- [UniProt ID Format Analyzer] Starting ---")

    if not UNIPROT_WHITELIST_PATH.exists():
        print(
            f"❌ ERROR: UniProt whitelist file not found at: {UNIPROT_WHITELIST_PATH}"
        )
        sys.exit(1)

    # --- 1. 加载并提取所有唯一的、真实的ID ---
    print(f"--> Loading real UniProt IDs from '{UNIPROT_WHITELIST_PATH.name}'...")
    df = pd.read_csv(
        UNIPROT_WHITELIST_PATH, sep="\t", usecols=["Entry", "Reviewed", "Organism (ID)"]
    )
    df_human_reviewed = df[
        (df["Organism (ID)"] == 9606) & (df["Reviewed"] == "reviewed")
    ]
    all_ids = df_human_reviewed["Entry"].unique().tolist()
    total_ids = len(all_ids)
    print(f"--> Found {total_ids} unique, reviewed, human UniProt IDs to analyze.")

    # --- 2. 使用每个正则表达式进行匹配和统计 ---
    results = {}
    exceptions = set(all_ids)  # 先假设所有ID都是例外

    for name, pattern in REGEX_PATTERNS.items():
        print(f"\n--> Testing Regex: '{name}'")
        matched_ids = set()
        for pid in tqdm(all_ids, desc="   - Matching"):
            if pattern.match(pid):
                matched_ids.add(pid)

        coverage = (len(matched_ids) / total_ids) * 100 if total_ids > 0 else 0
        results[name] = {"matched_count": len(matched_ids), "coverage_%": coverage}

        # 如果是严格模式，找出它未能匹配的ID
        if name == "Strict (Official)":
            exceptions = set(all_ids) - matched_ids

    # --- 3. 打印总结报告 ---
    print("\n" + "=" * 80)
    print(" " * 20 + "UniProt ID Regex Coverage Report")
    print("=" * 80)

    report_df = pd.DataFrame.from_dict(results, orient="index")
    print(report_df.to_string(float_format="%.2f"))

    print("\n" + "=" * 80)
    print(" " * 20 + "Analysis of 'Strict (Official)' Failures")
    print("=" * 80)

    if not exceptions:
        print(
            "\n✅ SUCCESS: The 'Strict (Official)' regex covered 100% of the IDs in the whitelist!"
        )
        print(
            "   This means we can safely use the most precise regex for API validation."
        )
    else:
        print(
            f"\n⚠️ WARNING: The 'Strict (Official)' regex failed to match {len(exceptions)} IDs."
        )
        print(
            "   These IDs are present in the official reviewed human proteome but do not match the standard format."
        )
        print("\n--- Sample of Unmatched IDs (up to 20) ---")
        for i, pid in enumerate(list(exceptions)[:20]):
            print(f"  - {pid}")

        print("\n--- Recommendation ---")
        print(
            "   Consider using the 'General (Simplified)' regex for a better balance of precision and coverage,"
        )
        print("   or investigate why these exceptions exist in the UniProt database.")

    print("\n--- [Analyzer] Finished ---")


if __name__ == "__main__":
    analyze_formats()
