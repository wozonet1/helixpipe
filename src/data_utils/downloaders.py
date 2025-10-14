# 文件: src/data_utils/downloaders.py

import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from omegaconf import DictConfig
from hydra import initialize, compose
import argparse

import research_template as rt

# --- 私有辅助函数 ---


def _download_file(url: str, output_path: Path):
    """一个通用的文件下载函数，带进度条和“跳过已存在文件”的逻辑。"""
    if output_path.exists():
        print(f"--> 找到已存在的文件: '{output_path.name}'。跳过下载。")
        return

    print(f"--> 正在从 {url} 下载 '{output_path.name}'...")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        with (
            open(output_path, "wb") as f,
            tqdm(
                desc=output_path.name,
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar,
        ):
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                bar.update(size)
    except requests.exceptions.RequestException as e:
        print(f"❌ 致命错误: 下载 {output_path.name} 失败。错误: {e}")
        if output_path.exists():
            output_path.unlink()  # 清理不完整的文件
        raise


def _get_dynamic_urls_for_bindingdb() -> dict:
    """动态构建BindingDB的下载链接。"""
    now = datetime.now()
    year_month = f"{now.year}{now.month:02d}"
    base_url = "https://www.bindingdb.org/rwd/bind/downloads"

    # 注意：我们将解压后的文件名也定义在这里，以保持一致性
    return {
        "interactions": {
            "url": f"{base_url}/BindingDB_All_{year_month}_tsv.zip",
            "zip_name": f"BindingDB_All_{year_month}_tsv.zip",
            "inner_name": "BindingDB_All.tsv",
        },
        "sequences": {
            "url": f"{base_url}/BindingDBTargetSequences.fasta",
            "file_name": "BindingDBTargetSequences.fasta",
        },
    }


# --- 公开的下载器函数 ---


def download_bindingdb_data(config: DictConfig):
    """为BindingDB数据集下载所有必需的原始文件。"""
    print("\n" + "=" * 80)
    print(" " * 25 + "启动 BindingDB 数据下载器")
    print("=" * 80 + "\n")

    # 确定要保存到的 raw 目录
    raw_dir = rt.get_path(config, "raw.dummy_file_to_get_dir").parent
    rt.ensure_path_exists(raw_dir / "dummy.txt")

    urls = _get_dynamic_urls_for_bindingdb()

    # 下载并解压交互数据
    interactions_info = urls["interactions"]
    zip_path = raw_dir / interactions_info["zip_name"]
    _download_file(interactions_info["url"], zip_path)

    # 检查是否需要解压
    unzipped_path = raw_dir / interactions_info["inner_name"]
    if not unzipped_path.exists() and zip_path.exists():
        print(f"--> 正在解压 '{zip_path.name}'...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extract(interactions_info["inner_name"], raw_dir)
        print("--> 解压完成。")

    # 下载序列数据
    sequences_info = urls["sequences"]
    fasta_path = raw_dir / sequences_info["file_name"]
    _download_file(sequences_info["url"], fasta_path)

    print("\n✅ BindingDB 数据下载任务完成。")
    print("=" * 80)


# --- 主程序入口 ---


def main():
    """主函数，用于从命令行运行下载器。"""
    parser = argparse.ArgumentParser(description="数据集下载器")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["bindingdb", "tdc"],
        help="选择要下载的数据集。",
    )
    args = parser.parse_args()

    with initialize(config_path="../../conf", job_name=f"{args.dataset}_download"):
        cfg = compose(config_name="config", overrides=[f"data={args.dataset}"])

    if args.dataset == "bindingdb":
        download_bindingdb_data(cfg)
    elif args.dataset == "tdc":
        print("TDC下载逻辑将在 tdc_pipeline.py 中实现，因为它通常与处理紧密耦合。")
        # 在这里可以调用一个专门的 tdc_downloader 函数
        pass


if __name__ == "__main__":
    main()
