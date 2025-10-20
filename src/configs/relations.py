# 文件: src/configs/relations.py

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class RelationsConfig:
    """
    【结构化配置】定义要在异构图中包含哪些类型的边（关系）。
    """

    _target_: str = "configs.relations.RelationsConfig"

    name: str = "default_relations"
    flags: Dict[str, bool] = field(
        default_factory=lambda: {
            "drug_protein_interaction": True,
            "ligand_protein_interaction": False,
            "protein_protein_similarity": False,
            "drug_drug_similarity": False,
            "drug_ligand_similarity": False,
            "ligand_ligand_similarity": False,
        }
    )
