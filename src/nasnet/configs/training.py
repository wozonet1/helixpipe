# 文件: src/configs/training.py

from dataclasses import dataclass, field


@dataclass
class ColdstartConfig:
    """定义冷启动的配置。"""

    mode: str = "drug"
    test_fraction: float = 0.2


@dataclass
class TrainingConfig:
    """
    【结构化配置】定义所有与模型训练和评估相关的参数。
    """

    learning_rate: float = 0.0005
    epochs: int = 200
    negative_sampling_ratio: float = 1.0
    weight_decay: float = 1e-5
    k_folds: int = 5
    batch_size: int = 512
    coldstart: ColdstartConfig = field(default_factory=ColdstartConfig)
