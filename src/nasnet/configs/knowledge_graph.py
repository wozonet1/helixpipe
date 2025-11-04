# src/nasnet/configs/knowledge_graph.py

from dataclasses import dataclass, field


@dataclass
class EntityTypeNames:
    """定义项目中所有已知实体类型的权威名称字符串。"""

    molecule: str = "molecule"
    protein: str = "protein"
    gene: str = "gene"
    disease: str = "disease"
    drug: str = "drug"
    ligand: str = "ligand"


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
