# 文件: src/configs/runtime.py

from dataclasses import dataclass


@dataclass
class RuntimeConfig:
    """
    【结构化配置】定义与运行环境相关的参数，这些参数不影响实验的数学结果。
    """

    seed: int = 514
    cpus: int = 100
    gpu: str = "cuda:2"
    force_restart: bool = False
    skip_data_proc: bool = False
    debug: bool = True
    validate_every_n_epochs: int = 10

    # fold_idx 由sweeper覆盖，给一个默认值
    fold_idx: int = 1

    # 您之前的配置中有 trian_loader_cpus (有拼写错误)，这里修正为 train
    train_loader_cpus: int = 16
    test_loader_cpus: int = 8
