# 文件: src/nasnet/configs/relations.py (最终扁平化版)

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class RelationNames:
    """定义所有已知的【原始】关系类型名称的注册表。"""

    # 通用/默认的原始关系类型
    default: str = "interacts_with"

    # 来自 BRENDA 的原始关系类型
    inhibits: str = "inhibits"
    catalyzes: str = "catalyzes"

    # 来自 STRING DB 的原始关系类型 (为未来准备)
    ppi: str = "physical_association"


@dataclass
class RelationsConfig:
    """
    【结构化配置 - 扁平化版】定义图的边关系类型、全局开关和映射规则。

    这个配置决定了最终异构图的结构和边的语义。
    """

    # name: 用于识别这套关系配置的名称，例如 "ppi_and_similarity"。
    name: str = "default_relations"
    names: RelationNames = field(default_factory=RelationNames)
    # flags: 一个全局的开关字典。
    # - 键 (key): 最终要生成到图文件中的【边类型字符串】。
    # - 值 (value): 布尔值 (true/false)，决定是否启用该类型的边。
    # 这里的键名是下游模型(如PyG HeteroData)将直接消费的名称。
    flags: Dict[str, bool] = field(
        default_factory=lambda: {
            # 默认的DTI/LPI关系
            "interacts_with": True,
            # 来自BRENDA的细粒度关系
            "inhibits": False,
            "catalyzes": False,
            # 来自STRING DB的PPI关系
            "associated_with": False,
            # 相似性关系
            "drug_drug_similarity": False,
            "ligand_ligand_similarity": False,
            "drug_ligand_similarity": False,
            "protein_protein_similarity": False,
        }
    )
