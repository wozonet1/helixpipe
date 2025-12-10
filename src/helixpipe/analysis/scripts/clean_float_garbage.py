import logging
import sys
from pathlib import Path

import helixlib as hx

# --- 配置日志 ---
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()


def scan_and_clean(project_root: Path, delete_mode: bool = False):
    """
    扫描 data/cache 目录，寻找并清理文件名中包含 '.0.' 的浮点数命名文件。
    """
    # 定义我们要检查的目标目录
    target_dirs = [
        project_root / "data/cache/features",
        project_root / "data/cache/ids",
    ]

    total_found = 0
    total_deleted = 0
    total_size_mb = 0.0

    logger.info("=" * 60)
    logger.info(
        f"MODE: {'🔴 DELETION (DANGER)' if delete_mode else '🟢 ANALYSIS (DRY RUN)'}"
    )
    logger.info("=" * 60)

    for target_dir in target_dirs:
        if not target_dir.exists():
            logger.warning(f"Directory not found, skipping: {target_dir}")
            continue

        logger.info(f"Scanning: {target_dir} ...")

        # 递归查找所有文件
        # rglob("*") 会遍历子文件夹
        files = [f for f in target_dir.rglob("*") if f.is_file()]

        for file_path in files:
            filename = file_path.name

            # --- 核心匹配逻辑 ---
            # 匹配: 12345.0.pt, P12345.0.pkl 等
            # 我们主要针对特征文件 (.pt) 和可能的 ID 缓存 (.pkl)
            if filename.endswith(".0.pt") or filename.endswith(".0.pkl"):
                file_size = file_path.stat().st_size / (1024 * 1024)  # MB
                total_found += 1
                total_size_mb += file_size

                if delete_mode:
                    try:
                        file_path.unlink()
                        total_deleted += 1
                        # 每删除 1000 个打印一次进度，避免刷屏
                        if total_deleted % 1000 == 0:
                            logger.info(f"  ... Deleted {total_deleted} files ...")
                    except OSError as e:
                        logger.error(f"  ❌ Failed to delete {filename}: {e}")
                else:
                    # 分析模式：只打印前 10 个作为样本
                    if total_found <= 10:
                        logger.info(f"  [Found] {filename} ({file_size:.2f} MB)")

    # --- 总结报告 ---
    logger.info("-" * 60)
    logger.info("SUMMARY REPORT")
    logger.info("-" * 60)
    if delete_mode:
        logger.info(f"Total Files Deleted: {total_deleted}")
        logger.info(f"Reclaimed Space:     {total_size_mb:.2f} MB")
    else:
        logger.info(f"Total Garbage Files Found: {total_found}")
        logger.info(f"Total Wasted Space:        {total_size_mb:.2f} MB")
        if total_found > 0:
            logger.info("-" * 60)
            logger.info("💡 To delete these files, run:")
            logger.info(f"   python {sys.argv[0]} --delete")
    logger.info("=" * 60)


if __name__ == "__main__":
    # 自动定位项目根目录 (假设脚本在 scripts/ 下)

    project_root = hx.get_project_root()

    # 检查命令行参数
    delete_mode = "--delete" in sys.argv

    scan_and_clean(project_root, delete_mode)
