# src/helixpipe/typing.py

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, Literal, Set, Tuple, Union

import pandas as pd
import torch

from .configs import AppConfig, EntitySelectorConfig, InteractionSelectorConfig

# ==============================================================================
# 1. 基础类型别名 (Primitive Type Aliases)
# ==============================================================================
# 这些别名用于增强代码的可读性，明确区分不同类型的ID。

# 权威ID (Authoritative ID)，可以是来自原始数据源的字符串或整数。
# 例如: 'P05067' (UniProt ID), 2244 (PubChem CID)。
CID = int

PID = str

AuthID = Union[CID, PID]

# 逻辑ID (Logic ID)，是项目中统一分配的、从0开始的连续整数。
LogicID = int

# 关系类型 (Relation Type)，表示边的类型的字符串。
RelationType = str

SMILES = str

ProteinSequence = str
# ==============================================================================
# 2. 核心数据结构别名 (Core Data Structure Aliases)
# ==============================================================================
# 这些别名定义了项目中核心的数据结构，如交互元组、字典等。

# --- 交互元组 (Interaction Tuples) ---

# 使用【权威ID】表示的交互元组: (source_auth_id, target_auth_id, relation_type)
AuthInteractionTuple = Tuple[AuthID, AuthID, RelationType]

# 使用【逻辑ID】表示的交互三元组: (source_logic_id, target_logic_id, relation_type)
# 这是图构建和模型训练中常用的格式。
LogicInteractionTriple = Tuple[LogicID, LogicID, RelationType]


# --- 字典类型 (Dictionary Types) ---

# 特征提取器的输出格式: {权威ID: 特征张量}
FeatureDict = Dict[AuthID, torch.Tensor]

# 所有 Processor 聚合后的输出格式: {数据集名称: DataFrame}
ProcessorOutputs = Dict[str, pd.DataFrame]

# ==============================================================================
# 3. 函数签名与复杂组合别名 (Function Signatures & Compositions)
# ==============================================================================
# 这些别名用于注解复杂的函数签名，尤其是那些返回函数或复杂元组的。

# `get_path` 函数返回的路径工厂 (Path Factory) 的类型。
PathFactory = Callable[..., Path]

# `get_path` 函数的联合返回类型。
PathLike = Union[Path, PathFactory]

# `DataSplitter` 的 `__next__` 方法返回的复杂元组类型。
# 注意: 使用字符串'InteractionStore'来避免循环导入问题。
SplitResult = Tuple[
    "InteractionStore",  # noqa: F821 # type: ignore
    "InteractionStore",  # noqa: F821 # type: ignore
    "InteractionStore",  # noqa: F821 # type: ignore
    Set[LogicID],
]


# ==============================================================================
# 5. TemplateKey (用于 `pathing.py` 的 @overload)
# ==============================================================================
# ATTENTION: MANUAL MAINTENANCE REQUIRED
# -------------------------------------
# 这个Literal类型用于帮助MyPy静态地识别哪些路径键会返回一个Callable。
# 每当你在YAML配置中新增或修改一个【路径模板】(包含"{...}")时，
# 都需要在这里手动更新这个列表。
# -------------------------------------
TemplateKey = Literal[
    "processed.specific.graph_template",
    "processed.specific.labels_template",
    "cache.features.template",
]
if TYPE_CHECKING:
    # 这个块中的代码只会被 MyPy 等类型检查器执行，
    # Python 解释器在运行时会完全忽略它。
    # 在这里，我们导入所有被前向引用 ("...") 的类型。
    from helixpipe.data_processing.services.interaction_store import InteractionStore

__all__ = [
    "AppConfig",
    "EntitySelectorConfig",
    "InteractionSelectorConfig",
    "CID",
    "PID",
    "AuthID",
    "LogicID",
    "RelationType",
    "SMILES",
    "ProteinSequence",
    "AuthInteractionTuple",
    "LogicInteractionTriple",
    "FeatureDict",
    "ProcessorOutputs",
    "PathFactory",
    "PathLike",
    "SplitResult",
    "TemplateKey",
]
