# 文件: src/configs/register_schemas.py (V4.0 - 最终稳定版)

import dataclasses
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, ValidationError

from .analysis import AnalysisConfig
from .data_params import DataParamsConfig

# --- 1. 静态地、明确地导入所有Config组件 ---
# 这使得AppConfig可以被静态分析，并被外部导入
from .data_structure import DataStructureConfig
from .dataset_collection import DatasetCollectionConfig
from .global_paths import GlobalPathsConfig
from .knowledge_graph import KnowledgeGraphConfig
from .relations import RelationsConfig
from .runtime import RuntimeConfig
from .training import TrainingConfig
from .validators import ValidatorsConfig


# --------------------------------------------------------------------------
# 2. 静态地定义顶层的 AppConfig
#    这个类现在可以被项目中的任何其他文件导入！
# --------------------------------------------------------------------------
@dataclass
class AppConfig(DictConfig):
    """
    【最终版】顶层聚合配置的静态定义。
    它作为整个配置结构的唯一“真理之源”和可导入的类型。
    """

    defaults: list[Any] = field(default_factory=list)

    # --- 明确列出所有配置组字段 ---
    data_structure: DataStructureConfig = field(default_factory=DataStructureConfig)
    data_params: DataParamsConfig = field(default_factory=DataParamsConfig)
    relations: RelationsConfig = field(default_factory=RelationsConfig)

    validators: ValidatorsConfig = field(default_factory=ValidatorsConfig)
    dataset_collection: DatasetCollectionConfig = field(
        default_factory=DatasetCollectionConfig
    )
    training: TrainingConfig = field(default_factory=TrainingConfig)
    # --- 明确列出所有顶层字段 ---
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    global_paths: GlobalPathsConfig = field(default_factory=GlobalPathsConfig)

    knowledge_graph: KnowledgeGraphConfig = field(default_factory=KnowledgeGraphConfig)


# --------------------------------------------------------------------------
# 3. 自动化的注册函数 (职责简化)
# --------------------------------------------------------------------------
def register_all_schemas() -> None:
    """
    【最终版】将静态定义的AppConfig及其所有组件注册到Config Store。
    它还包含一个检查步骤，以确保AppConfig没有遗漏任何configs目录下的模块。
    """
    try:
        cs = ConfigStore.instance()

        # a. 注册主schema
        cs.store(name="base_app_schema", node=AppConfig)

        # b. 自动发现并注册所有【组】schema
        #    我们通过 inspect AppConfig 的字段来找到它们
        grouped_schemas = {
            "data_structure": DataStructureConfig,
            "data_params": DataParamsConfig,
            "relations": RelationsConfig,
            "analysis": AnalysisConfig,
            "validators": ValidatorsConfig,
            "dataset_collection": DatasetCollectionConfig,
            "training": TrainingConfig,
            # 注意：training, runtime等顶层节点没有“组”的概念，所以不在这里注册
        }

        for group_name, schema_cls in grouped_schemas.items():
            schema_name = f"base_{group_name}"
            cs.store(name=schema_name, group=group_name, node=schema_cls)
    # 【核心修改】捕获更具体的错误，并提供更精准的建议
    except ValidationError as e:
        print("\n" + "=" * 80)
        print(
            "❌ FATAL: Hydra Schema Validation Failed. Your dataclass definitions are likely incompatible."
        )
        print("=" * 80)
        print(f"   Error Type: {type(e).__name__}")
        print(f"   Error Message: {e}")

        # 【新增】针对 Literal 类型的专属提示
        if "Unexpected type annotation: Literal" in str(e):
            print("\n" + "-" * 30 + " [ SPECIFIC SUGGESTION ] " + "-" * 30)
            print(
                "   This is a known compatibility issue with older versions of OmegaConf."
            )
            print("   To fix this, you can either:")
            print(
                "     1. (Quick Fix): In the dataclass mentioned above, replace `Literal[...]` with `str`."
            )
            print(
                "     2. (Long-term): Upgrade your `omegaconf` and `hydra-core` libraries to the latest versions."
            )
            print("-" * 80)

        print("\n--- Full Stack Trace ---")
        traceback.print_exc()
        print("------------------------")
        sys.exit(1)

    # c. (可选但推荐) 自动验证，确保没有遗漏
    _validate_completeness()

    print("--> All structured config schemas registered successfully.")


def _validate_completeness() -> None:
    """一个辅助函数，用于检查AppConfig是否包含了configs目录下的所有定义。"""
    defined_fields = {f.name for f in dataclasses.fields(AppConfig)}

    found_modules = set()
    for f in Path(__file__).parent.glob("*.py"):
        if f.name.startswith(("_", "register")) or f.name.startswith("selector"):
            continue
        found_modules.add(f.stem)

    missing_in_app_config = found_modules - defined_fields
    if missing_in_app_config:
        print("=" * 80)
        print(
            "⚠️  CONFIG SCHEMA WARNING: The following modules were found in 'src/configs/' but are missing as fields in 'AppConfig':"
        )
        for missing in sorted(list(missing_in_app_config)):
            print(f"   - {missing}")
        print(
            "   Please add them to 'src/configs/register_schemas.py' -> AppConfig dataclass."
        )
        print("=" * 80)
