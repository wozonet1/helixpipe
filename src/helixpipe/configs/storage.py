# 文件: src/configs/training.py

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StorageConfig:
    """
    【结构化配置】定义所有与模型训练和评估相关的参数。
    """

    name: str = "default"
    port: int = 8080
    host: str = "localhost"
    enabled: bool = True
