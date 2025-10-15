# 文件: src/configs/relations.py

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class RelationsConfig:
    """
    【结构化配置】定义要在异构图中包含哪些类型的边（关系）。
    """

    _target_: str = "src.configs.relations.RelationsConfig"

    name: str = "default_relations"
    flags: Dict[str, bool] = field(
        default_factory=lambda: {
            "dp_interaction": True,
            "lp_interaction": False,
            "pp_similarity": False,
            "dd_similarity": False,
            "dl_similarity": False,
            "ll_similarity": False,
        }
    )
