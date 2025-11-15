# src/helixpipe/utils/relation_utils.py (新文件或移动后的文件)

from typing import TYPE_CHECKING, Dict, Tuple

if TYPE_CHECKING:
    from helixpipe.configs.knowledge_graph import KnowledgeGraphConfig


def get_canonical_tuple(
    type1: str, type2: str, priority_map: Dict[str, int]
) -> Tuple[str, str]:
    """根据优先级，只返回一个规范化的 (source_type, target_type) 元组。"""
    priority1 = priority_map.get(type1, 999)
    priority2 = priority_map.get(type2, 999)

    if priority1 <= priority2:
        return type1, type2
    else:
        return type2, type1


def get_similarity_relation_type(
    type1: str,
    type2: str,
    kg_config: "KnowledgeGraphConfig",
) -> Tuple[str, str, str]:
    """根据配置模板，生成标准的相似性关系类型字符串。"""
    priority_map = {name: meta.priority for name, meta in kg_config.entity_meta.items()}
    source_type, target_type = get_canonical_tuple(type1, type2, priority_map)

    # 从配置中获取模板
    template = kg_config.relation_templates.similarity

    # 使用模板进行格式化
    return (
        source_type,
        target_type,
        template.format(source_type=source_type, target_type=target_type),
    )
