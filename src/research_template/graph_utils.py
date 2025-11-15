# 文件: src/research_template/graph_utils.py (新增函数)
# 文件: src/research_template/graph_utils.py (升级版)
from typing import Tuple


def get_canonical_relation(type1: str, type2: str) -> Tuple[str, str, str]:
    """
    【升级版】根据预定义的优先级，为两种节点类型生成一个
    规范的、有向的关系元组。

    Returns:
        Tuple[str, str, str]: 一个元组 (source_type, relation_prefix, target_type)
                              例如: ("drug", "drug_ligand", "ligand")
    """
    NODE_TYPE_PRIORITY = ["drug", "ligand", "protein", "disease", "gene"]
    priority_map = {name: i for i, name in enumerate(NODE_TYPE_PRIORITY)}

    priority1 = priority_map.get(type1, 999)
    priority2 = priority_map.get(type2, 999)

    if priority1 <= priority2:
        source_type, target_type = type1, type2
    else:
        source_type, target_type = type2, type1

    relation_prefix = f"{source_type}_{target_type}"

    return source_type, relation_prefix, target_type
