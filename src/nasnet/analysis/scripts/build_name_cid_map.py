# 文件: src/nasnet/analysis/scripts/build_name_cid_map.py

import gzip
import pickle as pkl
import time
from pathlib import Path

from tqdm import tqdm

# --- 配置 ---
# PubChem同义词源文件路径 (根据您的项目结构)
SOURCE_SYNONYM_FILE = Path("data/assets/CID-Synonym-filtered.gz")

# 我们希望生成的本地映射文件的输出路径
OUTPUT_MAP_FILE = Path("data/cache/ids/pubchem_name_to_cid.pkl")


def build_map():
    """
    解析PubChem的同义词文件，构建一个 name -> cid 的映射字典。
    注意：这是一个耗时且消耗内存的操作，但只需运行一次。
    """
    print("--- [PubChem Mapper Builder] Starting ---")
    if not SOURCE_SYNONYM_FILE.exists():
        print(f"❌ ERROR: PubChem synonym file not found at '{SOURCE_SYNONYM_FILE}'.")
        print(
            "   Please download 'CID-Synonym-filtered.gz' from the PubChem FTP server."
        )
        return

    # 确保输出目录存在
    OUTPUT_MAP_FILE.parent.mkdir(parents=True, exist_ok=True)

    print(f"--> Parsing '{SOURCE_SYNONYM_FILE.name}'... This will take a long time.")

    name_to_cid = {}
    start_time = time.time()

    # 使用流式读取，避免将几十GB的文件一次性读入内存
    with gzip.open(SOURCE_SYNONYM_FILE, "rt", encoding="utf-8") as f:
        # 使用tqdm来可视化处理进度 (需要知道总行数，如果不知道可以不加total)
        # 估算行数以提供进度条，实际行数可能略有不同
        estimated_lines = 300_000_000
        for line in tqdm(f, total=estimated_lines, desc="   - Processing lines"):
            try:
                cid_str, name = line.strip().split("\t")
                # 使用小写名称作为键，以实现不区分大小写的查找
                # 我们只为每个名称存储第一个遇到的CID，这足够满足我们的需求
                if name.lower() not in name_to_cid:
                    name_to_cid[name.lower()] = int(cid_str)
            except ValueError:
                # 忽略格式不正确的行
                continue

    end_time = time.time()
    print(f"\n--> Parsing complete in {end_time - start_time:.2f} seconds.")
    print(f"--> Found {len(name_to_cid)} unique names.")

    print(f"--> Saving map to '{OUTPUT_MAP_FILE}'...")
    with open(OUTPUT_MAP_FILE, "wb") as f:
        pkl.dump(name_to_cid, f)

    print("✅ --- [PubChem Mapper Builder] Finished successfully! ---")


if __name__ == "__main__":
    build_map()
