# 文件: src/configs/register_schemas.py (V4.0 - 最终稳定版)

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List
import dataclasses
from hydra.core.config_store import ConfigStore

# --- 1. 静态地、明确地导入所有Config组件 ---
# 这使得AppConfig可以被静态分析，并被外部导入
from .data_structure import DataStructureConfig
from .data_params import DataParamsConfig
from .relations import RelationsConfig
from .predictor import PredictorConfig
from .training import TrainingConfig
from .runtime import RuntimeConfig
from .analysis import AnalysisConfig
from .global_paths import GlobalPathsConfig
from .mlflow import MlflowConfig


# --------------------------------------------------------------------------
# 2. 静态地定义顶层的 AppConfig
#    这个类现在可以被项目中的任何其他文件导入！
# --------------------------------------------------------------------------
@dataclass
class AppConfig:
    """
    【最终版】顶层聚合配置的静态定义。
    它作为整个配置结构的唯一“真理之源”和可导入的类型。
    """

    defaults: List[Any] = field(default_factory=list)

    # --- 明确列出所有配置组字段 ---
    data_structure: DataStructureConfig = field(default_factory=DataStructureConfig)
    data_params: DataParamsConfig = field(default_factory=DataParamsConfig)
    relations: RelationsConfig = field(default_factory=RelationsConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)

    # --- 明确列出所有顶层字段 ---
    training: TrainingConfig = field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    global_paths: GlobalPathsConfig = field(default_factory=GlobalPathsConfig)
    mlflow: MlflowConfig = field(default_factory=MlflowConfig)

    # hydra节点我们依然让它保持为 untyped dict
    hydra: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# 3. 自动化的注册函数 (职责简化)
# --------------------------------------------------------------------------
def register_all_schemas():
    """
    【最终版】将静态定义的AppConfig及其所有组件注册到Config Store。
    它还包含一个检查步骤，以确保AppConfig没有遗漏任何configs目录下的模块。
    """
    cs = ConfigStore.instance()

    # a. 注册主schema
    cs.store(name="base_app_schema", node=AppConfig)

    # b. 自动发现并注册所有【组】schema
    #    我们通过 inspect AppConfig 的字段来找到它们
    grouped_schemas = {
        "data_structure": DataStructureConfig,
        "data_params": DataParamsConfig,
        "relations": RelationsConfig,
        "predictor": PredictorConfig,
        "analysis": AnalysisConfig,
        # 注意：training, runtime等顶层节点没有“组”的概念，所以不在这里注册
    }

    for group_name, schema_cls in grouped_schemas.items():
        schema_name = f"base_{group_name}"
        cs.store(name=schema_name, group=group_name, node=schema_cls)

    # c. (可选但推荐) 自动验证，确保没有遗漏
    _validate_completeness()

    print("--> All structured config schemas registered successfully.")


def _validate_completeness():
    """一个辅助函数，用于检查AppConfig是否包含了configs目录下的所有定义。"""
    defined_fields = {f.name for f in dataclasses.fields(AppConfig)}

    found_modules = set()
    for f in Path(__file__).parent.glob("*.py"):
        if f.name.startswith(("_", "register")):
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
