# 文件: src/nasnet/analysis/scripts/benchmark_api_batch_sizes.py (全新)

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# --- 动态路径设置 ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
except IndexError:
    raise RuntimeError("Could not determine project root.")

# --- 导入我们的fetcher函数 ---
from nasnet.data_processing.services.canonicalizer import (
    fetch_sequences_from_uniprot,
    fetch_smiles_from_pubchem,
)

# --- 代理配置 ---
# 如果需要，在这里设置您的代理
PROXY_CONFIG = None
# PROXY_CONFIG = {'http': 'http://127.0.0.1:7890', 'https': 'http://127.0.0.1:7890'}

# ==============================================================================
# 1. 针对具体API的Benchmark“工人”函数
# ==============================================================================


def benchmark_uniprot(batch_size: int, num_trials: int, id_pool: list) -> dict:
    """对 UniProt API 进行基准测试。"""
    print(f"\n--- Testing UniProt | Batch Size: {batch_size} ({num_trials} trials) ---")
    timings = []
    success_counts = 0

    for i in range(num_trials):
        id_batch = random.sample(id_pool, k=min(batch_size, len(id_pool)))
        start_time = time.time()

        results = fetch_sequences_from_uniprot(id_batch)

        duration = time.time() - start_time
        timings.append(duration)
        success_counts += len(results)

        print(
            f"  - Trial {i + 1}/{num_trials}: Fetched {len(results)}/{len(id_batch)} sequences in {duration:.2f}s"
        )
        time.sleep(1)  # 每次试验间休息

    avg_time = np.mean(timings) if timings else float("inf")
    avg_success_rate = (
        (success_counts / (num_trials * batch_size)) * 100
        if num_trials * batch_size > 0
        else 0
    )

    return {
        "batch_size": batch_size,
        "avg_time_s": avg_time,
        "success_rate_%": avg_success_rate,
    }


def benchmark_pubchem(batch_size: int, num_trials: int, id_pool: list) -> dict:
    """对 PubChem API 进行基准测试。"""
    print(f"\n--- Testing PubChem | Batch Size: {batch_size} ({num_trials} trials) ---")
    timings = []
    success_counts = 0

    for i in range(num_trials):
        id_batch = random.sample(id_pool, k=min(batch_size, len(id_pool)))
        start_time = time.time()

        results = fetch_smiles_from_pubchem(
            id_batch, batch_size=batch_size, proxies=PROXY_CONFIG
        )

        duration = time.time() - start_time
        timings.append(duration)
        success_counts += len(results)

        print(
            f"  - Trial {i + 1}/{num_trials}: Fetched {len(results)}/{len(id_batch)} SMILES in {duration:.2f}s"
        )
        time.sleep(1)

    avg_time = np.mean(timings) if timings else float("inf")
    avg_success_rate = (
        (success_counts / (num_trials * batch_size)) * 100
        if num_trials * batch_size > 0
        else 0
    )

    return {
        "batch_size": batch_size,
        "avg_time_s": avg_time,
        "success_rate_%": avg_success_rate,
    }


# ==============================================================================
# 2. ID池生成器
# ==============================================================================


def get_uniprot_id_pool(size=2000) -> list:
    """生成一组随机但格式有效的UniProt ID。"""
    print(f"\n--> Generating a pool of {size} random UniProt IDs...")
    prefixes = ["P", "Q", "O"]
    # 生成更多样的ID格式
    pool = {
        f"{random.choice(prefixes)}{random.randint(10000, 99999)}"
        for _ in range(size // 2)
    }
    pool.update(
        {
            f"A0A{random.randint(100, 999)}R{random.randint(1, 9)}R{random.randint(1, 9)}"
            for _ in range(size // 2)
        }
    )
    return list(pool)


def get_pubchem_cid_pool(size=2000) -> list:
    """生成一组随机的PubChem CID。"""
    print(f"\n--> Generating a pool of {size} random PubChem CIDs...")
    # 在一个常见的范围内生成随机整数
    return [random.randint(1, 100000) for _ in range(size)]


# ==============================================================================
# 3. 主协调函数
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark API batch sizes for UniProt and PubChem."
    )
    parser.add_argument(
        "target", choices=["uniprot", "pubchem"], help="The API to benchmark."
    )
    args = parser.parse_args()

    print("=" * 80)
    print(" " * 20 + f"STARTING BATCH SIZE BENCHMARK FOR: {args.target.upper()}")
    print("=" * 80)

    # --- 配置测试参数 ---
    BATCH_SIZES_TO_TEST = [25, 50, 100, 200, 400, 500]
    NUM_TRIALS_PER_SIZE = 3

    if args.target == "uniprot":
        id_pool = get_uniprot_id_pool()
        benchmark_func = benchmark_uniprot
    else:  # pubchem
        id_pool = get_pubchem_cid_pool()
        benchmark_func = benchmark_pubchem

    # --- 运行基准测试 ---
    all_results = []
    for size in BATCH_SIZES_TO_TEST:
        result = benchmark_func(size, NUM_TRIALS_PER_SIZE, id_pool)
        all_results.append(result)

    # --- 打印最终报告 ---
    print("\n\n" + "=" * 80)
    print(" " * 30 + "Benchmark Summary")
    print("=" * 80)

    results_df = pd.DataFrame(all_results)
    # 计算吞吐量 (每秒处理的ID数)
    results_df["throughput_id_per_sec"] = (
        results_df["batch_size"] / results_df["avg_time_s"]
    )

    print(results_df.to_string(index=False, float_format="%.2f"))

    # --- 推荐最佳选择 ---
    # 筛选出成功率高于95%的选项
    reliable_options = results_df[results_df["success_rate_%"] > 95]
    if not reliable_options.empty:
        best_choice = reliable_options.loc[
            reliable_options["throughput_id_per_sec"].idxmax()
        ]
        print("\n" + "=" * 80)
        print(f"🏆 Recommended Batch Size: {int(best_choice['batch_size'])}")
        print(
            "   This size offers the best throughput while maintaining a high success rate (>95%)."
        )
    else:
        print("\n" + "=" * 80)
        print(
            "⚠️ No batch size achieved a high success rate. Consider smaller sizes or check network stability."
        )


if __name__ == "__main__":
    main()
