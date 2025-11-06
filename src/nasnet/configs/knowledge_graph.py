# src/nasnet/configs/knowledge_graph.py

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class EntityMetaConfig:
    metatype: str  # 'molecule', 'protein', 'gene', etc.
    priority: int


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
    entity_meta: Dict[str, EntityMetaConfig] = field(
        default_factory=lambda: {
            # 分子大类
            "drug": EntityMetaConfig(metatype="molecule", priority=0),
            "exogenous_ligand": EntityMetaConfig(metatype="molecule", priority=1),
            "ligand": EntityMetaConfig(metatype="molecule", priority=2),
            "endogenous_ligand": EntityMetaConfig(metatype="molecule", priority=3),
            "molecule": EntityMetaConfig(metatype="molecule", priority=9),
            # 蛋白质/基因大类
            "enzyme": EntityMetaConfig(metatype="protein", priority=10),
            "receptor": EntityMetaConfig(metatype="protein", priority=11),
            "protein": EntityMetaConfig(metatype="protein", priority=19),
            "gene": EntityMetaConfig(metatype="gene", priority=20),
            # 其他大类
            "disease": EntityMetaConfig(metatype="disease", priority=30),
        }
    )
    type_mapping_strategy: Optional[Dict[str, str]] = None
