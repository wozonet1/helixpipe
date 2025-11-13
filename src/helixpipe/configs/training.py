# 文件: src/configs/training.py

from dataclasses import dataclass, field
from typing import List, Optional

from .selectors import EntitySelectorConfig, InteractionSelectorConfig


@dataclass
class ColdstartConfig:
    """【V2 - 策略定义版】"""

    # warm or cold
    mode: str = "cold"
    # 注意对于实体来说,这是实体的比例,而非边的比例
    test_fraction: float = 0.2

    # 【核心修改】'pool_scope' 现在是一个结构化的选择器
    pool_scope: EntitySelectorConfig = field(
        default_factory=lambda: EntitySelectorConfig(
            # 默认情况下，不对实体池做任何限制
        )
    )
    # 【核心修改】'evaluation_scope' 也是一个结构化的选择器
    # 它定义了“边”的选择，所以需要两个选择器
    evaluation_scope: Optional[InteractionSelectorConfig] = None

    strictness: str = "strict"
    # 'allowed_leakage_types' 是一个白名单，只在 strictness='strict' 时生效
    # 默认情况下，严格模式不允许任何背景边泄露
    allowed_leakage_types: List[str] = field(default_factory=list)


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
