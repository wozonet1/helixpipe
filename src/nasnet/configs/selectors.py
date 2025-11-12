# 新构想: src/nasnet/configs/selectors.py (创建一个新文件)

from dataclasses import dataclass
from typing import List, Optional


# 这是我们已有的实体选择器
@dataclass
class EntitySelectorConfig:
    entity_types: Optional[List[str]] = None
    meta_types: Optional[List[str]] = None
    from_sources: Optional[List[str]] = None


# 【全新】这是我们的边选择器
@dataclass
class InteractionSelectorConfig:
    """定义一个用于筛选交互（边）的规则集合。"""

    # 规则1: 对边的“源”节点进行筛选
    source_selector: Optional[EntitySelectorConfig] = None

    # 规则2: 对边的“目标”节点进行筛选
    target_selector: Optional[EntitySelectorConfig] = None

    # 规则3: 对边的“关系类型”进行筛选
    relation_types: Optional[List[str]] = None

    # 逻辑: 一个交互必须同时满足所有非None的选择器规则。
