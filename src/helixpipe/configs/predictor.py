# 文件: src/configs/predictor.py

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class PredictorConfig:
    """
    【结构化配置】定义模型(预测器)的名称和超参数。
    """

    _target_: str = "src.configs.predictor.PredictorConfig"

    name: str = "default_predictor"
    # params 是一个开放字典，因为不同模型的参数不同
    # 并且它会包含一个 _target_ 字段用于Hydra实例化
    params: Dict[str, Any] = field(default_factory=dict)
