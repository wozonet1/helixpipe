# src/helixpipe/analysis/scripts/analyze_gtopdb_endogenous.py

import sys

import pandas as pd
import research_template as rt

# --- 脚本配置 ---
PROJECT_ROOT = rt.get_project_root()

# 定义GtoPdb的原始数据目录
GTOPDB_RAW_DIR = PROJECT_ROOT / "data" / "gtopdb" / "raw"
INTERACTIONS_FILE = GTOPDB_RAW_DIR / "interactions.csv"

# 我们要调查的目标列
TARGET_COLUMN = "Endogenous"


def analyze_endogenous_column():
    """
    一个独立的分析脚本，专门用于深度分析 GtoPdb 'interactions.csv' 文件中的
    'Endogenous' 列的内容、格式和分布。
    """
    print("\n" + "=" * 80)
    print("🔬 GtoPdb 'Endogenous' Column Analyzer")
    print("=" * 80)

    # 1. 检查文件是否存在
    if not INTERACTIONS_FILE.exists():
        print(f"❌ 错误: 文件未找到于 -> {INTERACTIONS_FILE}")
        print("   请确保您已将 'interactions.csv' 文件放置在正确的目录。")
        sys.exit(1)

    # 2. 加载数据 (只加载我们需要的列以提高效率)
    try:
        print(
            f"--> 正在加载 '{INTERACTIONS_FILE.name}' (仅加载 '{TARGET_COLUMN}' 列)..."
        )
        # 使用 usecols 参数，极大地加速了文件读取
        df = pd.read_csv(
            INTERACTIONS_FILE, low_memory=False, comment="#", usecols=[TARGET_COLUMN]
        )
        print(f"✅ 文件加载成功，共 {len(df)} 行。")
    except ValueError:
        print(
            f"❌ 错误: 在 '{INTERACTIONS_FILE.name}' 中找不到名为 '{TARGET_COLUMN}' 的列。"
        )
        print("   请检查原始文件的列名是否正确。")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 错误: 读取文件时发生未知错误: {e}")
        sys.exit(1)

    # 3. 核心分析：使用 value_counts() 获取所有唯一值及其计数
    print("\n--- [1. 唯一值及其分布] ---")

    # .value_counts() 是我们最强大的侦察工具
    # dropna=False 会将 NaN 值也作为一个类别进行统计
    value_distribution = df[TARGET_COLUMN].value_counts(dropna=False)

    if value_distribution.empty:
        print("该列不包含任何数据。")
    else:
        print("'Endogenous' 列中所有唯一值及其出现的次数：")
        print(value_distribution.to_string())

    # 4. 衍生分析：检查是否存在大小写或前后空格问题
    print("\n--- [2. 格式与一致性检查] ---")

    # 创建一个经过清洗的Series (去除前后空格，转为小写)
    cleaned_series = df[TARGET_COLUMN].str.strip().str.lower()
    cleaned_distribution = cleaned_series.value_counts(dropna=False)

    print("经过“清洗”(去除空格、转为小写)后的唯一值及其分布：")
    print(cleaned_distribution.to_string())

    if (
        len(value_distribution) == len(cleaned_distribution)
        and (value_distribution.index == cleaned_distribution.index).all()
    ):
        print("\n[结论] -> 数据格式非常干净！不存在大小写混合或前后有空格的问题。")
    else:
        print(
            "\n[结论] -> 数据格式存在不一致！原始值和清洗后的值有差异，建议在代码中使用清洗后的值进行比较。"
        )

    # 5. 最终建议
    print("\n--- [3. 最终行动建议] ---")
    print(
        "基于以上分析，您在 `GtopdbProcessor` 的 `_filter_data` 方法中应该使用以下逻辑："
    )

    # 假设最常见的值是 'No'
    most_common_value = cleaned_distribution.index[0]
    if most_common_value == "no":
        print("✅ 方案A (推荐用于药物发现): 保留非内源性交互。")
        print("   df_filtered = df[df['endogenous_flag_normalized'] != 'yes']")
        print(
            "   (或者更安全地: df_filtered = df[df['endogenous_flag_normalized'] == 'no'])"
        )
    elif most_common_value == "yes":
        print("✅ 方案B (如果您确认需要内源性交互): 保留内源性交互。")
        print("   df_filtered = df[df['endogenous_flag_normalized'] == 'yes']")
    else:
        print(
            "🟡 警告: 最常见的值既不是 'yes' 也不是 'no'。请根据上面的分布情况，自行决定正确的过滤逻辑。"
        )

    print("=" * 80)


if __name__ == "__main__":
    analyze_endogenous_column()
