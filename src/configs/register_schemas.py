# 文件: src/configs/register_schemas.py (V3.1 - 完全自动化版)

import importlib
from pathlib import Path
from dataclasses import field, make_dataclass
from typing import Dict, Any, List
from hydra.core.config_store import ConfigStore
import dataclasses  # 显式导入

# --- 全局变量，用于存储发现的schemas ---
# 顶层schemas (没有配置组)
_top_level_schemas: Dict[str, type] = {}
# 带组的schemas
_grouped_schemas: Dict[str, type] = {}

# 约定：哪些配置文件应该被视为顶层配置
TOP_LEVEL_CONFIG_NAMES = ["global_paths", "mlflow", "hydra", "training", "runtime"]


def _discover_and_load_schemas():
    """自动扫描、导入，并根据约定区分“顶层”和“带组”的schemas。"""
    if _top_level_schemas or _grouped_schemas:  # 避免重复扫描
        return

    for f in Path(__file__).parent.glob("*.py"):
        if f.name.startswith(("_", "register")):
            continue

        module_name = f.stem
        try:
            module = importlib.import_module(f".{module_name}", package=__package__)
            class_name = (
                f"{''.join(word.capitalize() for word in module_name.split('_'))}Config"
            )

            if hasattr(module, class_name):
                schema_class = getattr(module, class_name)
                if dataclasses.is_dataclass(schema_class):
                    # 【核心修正】根据约定进行区分
                    if module_name in TOP_LEVEL_CONFIG_NAMES:
                        _top_level_schemas[module_name] = schema_class
                    else:
                        _grouped_schemas[module_name] = schema_class
        except ImportError as e:
            print(f"⚠️ Warning: Could not process schema in '{f.name}': {e}")


def _create_dynamic_app_config() -> type:
    """基于自动发现的所有schema，动态地创建一个顶层的AppConfig dataclass。"""

    # 1. 动态构建所有字段
    all_fields = [("defaults", List[Any], field(default_factory=list))]

    # a. 添加所有顶层字段
    for name, schema_cls in _top_level_schemas.items():
        all_fields.append((name, schema_cls, field(default_factory=schema_cls)))

    # b. 添加所有带组的字段
    for name, schema_cls in _grouped_schemas.items():
        all_fields.append((name, schema_cls, field(default_factory=schema_cls)))

    # 2. 动态创建AppConfig类
    AppConfig = make_dataclass("AppConfig", fields=all_fields)
    return AppConfig


def register_all_schemas():
    """【自动化版】将项目中所有的dataclass schema注册到Config Store。"""
    # 1. 运行发现
    _discover_and_load_schemas()
    # 2. 动态创建AppConfig
    AppConfig = _create_dynamic_app_config()

    cs = ConfigStore.instance()

    # 3. 注册主schema
    cs.store(name="base_app_schema", node=AppConfig)

    # 4. 自动注册所有【带组】的子schema
    for group_name, schema_cls in _grouped_schemas.items():
        schema_name = f"base_{group_name}"
        cs.store(name=schema_name, group=group_name, node=schema_cls)

    print("--> All structured config schemas dynamically discovered and registered.")
