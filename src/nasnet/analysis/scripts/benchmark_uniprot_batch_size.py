import random
import time
from typing import Tuple

import numpy as np
import requests
from requests.adapters import HTTPAdapter, Retry

# ==============================================================================
# 1. 从 id_validation_service.py 中复制并简化API调用函数
#    我们在这里创建独立的、最小化的版本，以确保测试的隔离性
# ==============================================================================

API_URL = "https://rest.uniprot.org"
POLLING_INTERVAL = 3
REQUEST_TIMEOUT = 60  # 给与足够长的单次请求超时时间

# 创建一个带重试的、无代理的会话
session = requests.Session()
retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))
session.proxies = {
    "http": None,
    "https": None,
}


def submit_job(ids: str) -> Tuple[str, None]:
    """提交任务，如果失败则返回None。"""
    try:
        response = session.post(
            f"{API_URL}/idmapping/run",
            data={"from": "UniProtKB_AC-ID", "to": "UniProtKB", "ids": ids},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json().get("jobId")
    except requests.RequestException as e:
        print(f"    - ❌ Submit Error: {e}")
        return None


def check_status(job_id: str) -> bool:
    """检查任务状态，如果失败则返回False。"""
    try:
        while True:
            response = session.get(
                f"{API_URL}/idmapping/status/{job_id}", timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            j = response.json()
            if "jobStatus" in j:
                if j["jobStatus"] == "RUNNING":
                    time.sleep(POLLING_INTERVAL)
                elif j["jobStatus"] == "FINISHED":
                    return True
                else:  # FAILED, ERROR etc.
                    print(f"    -  Job failed with status: {j['jobStatus']}")
                    return False
            else:
                return True  # Assume success if no status
    except requests.RequestException as e:
        print(f"    - ❌ Status Check Error: {e}")
        return False


# ==============================================================================
# 2. 主测试逻辑
# ==============================================================================


def benchmark_batch_size(batch_size: int, num_trials: int, test_ids: list) -> dict:
    """
    对给定的 batch_size 进行多次测试，并返回统计结果。
    """
    print(f"\n--- Testing Batch Size: {batch_size} (running {num_trials} trials) ---")

    success_count = 0
    timings = []

    for i in range(num_trials):
        start_time = time.time()
        # 从我们的ID池中随机抽取一批ID进行测试
        id_batch = random.sample(test_ids, k=min(batch_size, len(test_ids)))

        job_id = submit_job(",".join(id_batch))

        if job_id and check_status(job_id):
            # 我们不需要获取结果，只要任务能成功完成即可
            success_count += 1
            duration = time.time() - start_time
            timings.append(duration)
            print(f"  - Trial {i + 1}/{num_trials}: ✅ SUCCESS in {duration:.2f}s")
        else:
            print(f"  - Trial {i + 1}/{num_trials}: ❌ FAILED")

        time.sleep(1)  # 在每次试验之间稍作停顿

    success_rate = (success_count / num_trials) * 100
    avg_time = np.mean(timings) if timings else float("inf")

    return {
        "batch_size": batch_size,
        "success_rate": success_rate,
        "avg_time_s": avg_time,
    }


def main():
    """
    主函数，编排整个基准测试流程。
    """
    print("=" * 80)
    print(" " * 15 + "UniProt ID Mapping API Batch Size Benchmark")
    print("=" * 80)

    # --- 配置测试参数 ---
    # 定义我们要测试的一系列 batch_size
    BATCH_SIZES_TO_TEST = [3, 10, 25, 50, 100, 200, 400, 500]
    # 每个 batch_size 测试多少次以获得可靠的统计数据
    NUM_TRIALS_PER_SIZE = 5

    # --- 生成测试ID池 ---
    # 我们需要一个足够大的ID池，从中随机抽样
    print("\n--> Generating a pool of 1000 random (but valid format) UniProt IDs...")
    prefixes = ["P", "Q", "O"]
    test_id_pool = list(
        {
            f"{random.choice(prefixes)}{random.randint(10000, 99999)}"
            for _ in range(1000)
        }
    )
    print(f"--> ID pool created with {len(test_id_pool)} unique IDs.")

    # --- 运行基准测试 ---
    all_results = []
    for size in BATCH_SIZES_TO_TEST:
        result = benchmark_batch_size(size, NUM_TRIALS_PER_SIZE, test_id_pool)
        all_results.append(result)

    # --- 打印最终的总结报告 ---
    print("\n\n" + "=" * 80)
    print(" " * 30 + "Benchmark Summary")
    print("=" * 80)
    print(f"{'Batch Size':<15} | {'Success Rate':<15} | {'Avg. Time (s)':<15}")
    print("-" * 50)

    best_choice = None
    max_throughput = 0

    for res in all_results:
        rate = res["success_rate"]
        avg_time = res["avg_time_s"]
        print(f"{res['batch_size']:<15} | {rate:<15.1f}% | {avg_time:<15.2f}")

        # --- 寻找最佳选择 ---
        # 我们寻找一个成功率高，且“吞吐量”（每秒处理的ID数）最大的尺寸
        if rate > 99.0:  # 必须是高成功率
            throughput = res["batch_size"] / avg_time if avg_time > 0 else 0
            if throughput > max_throughput:
                max_throughput = throughput
                best_choice = res["batch_size"]

    print("=" * 80)
    if best_choice:
        print(f"\n🏆 Recommended Batch Size: {best_choice}")
        print(
            "   This size offers the best throughput while maintaining a high success rate."
        )
    else:
        print(
            "\n⚠️ No batch size achieved a high success rate. Consider using a smaller size or checking network stability."
        )


if __name__ == "__main__":
    main()
