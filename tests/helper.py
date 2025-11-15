# tests/helpers.py

from typing import Any, cast

from omegaconf import OmegaConf

from helixpipe.configs import AppConfig


def create_test_config(overrides: dict[str, Any] = None) -> AppConfig:
    """
    创建一个用于测试的基础 AppConfig 对象，并允许覆盖。
    """
    # 1. 使用 structured API 创建一个符合 schema 的基础对象
    base_config = OmegaConf.structured(AppConfig)

    # 2. 定义一套适用于大多数测试的、最小化的默认覆盖
    default_overrides = {
        "runtime": {"verbose": 0},
        # ... 其他测试环境的通用设置 ...
    }

    # 3. 合并默认覆盖和用户传入的覆盖
    final_overrides = OmegaConf.merge(default_overrides, overrides or {})

    # 4. 将覆盖合并到基础配置中
    config = OmegaConf.merge(base_config, final_overrides)

    # 5. 使用 cast 返回一个类型检查器满意的对象
    return cast(AppConfig, config)
