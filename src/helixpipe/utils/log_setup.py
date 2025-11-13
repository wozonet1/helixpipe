# src/helixpipe/utils/log_setup.py (新文件)

import logging
import sys

from helixpipe.configs import AppConfig


def setup_logging(config: AppConfig):
    """
    根据Hydra配置，初始化整个项目的日志系统。
    """
    log_level_map = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG,
    }
    log_level = log_level_map.get(config.runtime.verbose, logging.INFO)

    # 获取根logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # 移除所有已存在的处理器，避免重复打印
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 创建一个控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)

    # 创建格式化器
    formatter = logging.Formatter(
        "%(asctime)s - [%(levelname)s] - [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)

    # 将处理器添加到根logger
    root_logger.addHandler(console_handler)

    logging.info("Logging system initialized.")
