# src/nasnet/configs/knowledge_graph.py

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class EntityTypeNames:
    """定义项目中所有已知实体类型的权威名称字符串。"""

    molecule: str = "molecule"
    protein: str = "protein"
    gene: str = "gene"
    disease: str = "disease"
    drug: str = "drug"
    ligand: str = "ligand"
    ligand_endo: str = "endogenous_ligand"  # 内源性配体
    ligand_exo: str = "exogenous_ligand"  # 外源性配体


@dataclass
class RelationTypeNames:
    """定义项目中所有已知“原始”关系类型的权威名称字符串。"""

    default: str = "interacts_with"
    inhibits: str = "inhibits"
    catalyzes: str = "catalyzes"
    ppi: str = "physical_association"


@dataclass
class KnowledgeGraphConfig:
    """
    【新增】定义知识图谱的逻辑/语义层。
    它描述了图中有哪些类型的实体和概念。
    """

    # 实体类型名称的官方注册表
    entity_types: EntityTypeNames = field(default_factory=EntityTypeNames)
    relation_types: RelationTypeNames = field(default_factory=RelationTypeNames)
    type_merge_priority: Dict[str, int] = field(
        default_factory=lambda: {
            # 分子亚型
            "drug": 0,  # 最高优先级
            "exogenous_ligand": 1,  # 外源性配体，次之
            "ligand": 2,  # 普通配体，再次之
            "endogenous_ligand": 3,  # 内源性配体，优先级最低的分子
            "molecule": 9,  # 通用分子类型，优先级非常低
            # 蛋白质亚型 (为未来预留)
            "enzyme": 10,
            "receptor": 11,
            "protein": 19,  # 通用蛋白质类型
            # 其他类型
            "gene": 20,
            "disease": 30,
        }
    )
    type_mapping_strategy: Optional[Dict[str, str]] = None
