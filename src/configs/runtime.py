# 文件: src/configs/runtime.py

from dataclasses import dataclass


@dataclass
class RuntimeConfig:
    """
    【结构化配置】定义与运行环境相关的参数，这些参数不影响实验的数学结果。
    """

    _target_: str = "src.configs.runtime.RuntimeConfig"

    seed: int = 514
    cpus: int = 4
    gpu: str = "cuda:0"
    force_restart: bool = False
    skip_data_proc: bool = False
    debug: bool = True
    validate_every_n_epochs: int = 10
    fold_idx: int = 1  # 由sweeper覆盖
