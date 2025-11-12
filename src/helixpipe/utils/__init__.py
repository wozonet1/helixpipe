# src/helixpipe/utils/__init__.py


"""helixpipe.utils - A collection of project-specific, reusable utility modules.

This package provides common tools for debugging, downloading, Hydra setup,
logging, and path management that are specific to the 'helixpipe' project."""


# --- 从各个子模块中“提升”最常用、最高阶的函数和类 ---

# 从 debug.py 中提升核心的验证函数
# 假设 validate_authoritative_dti_file 是最常用的
from .debug import (
    run_online_validation,
    validate_authoritative_dti_file,
    validate_config_with_data,
)

# 从 download.py 中提升核心下载函数
# 假设 download_bindingdb_data 是一个高级接口
from .download import download_bindingdb_data

# 从 hydra_setup.py 中提升注册函数
# path_resolver 是内部实现，通常不需要在包级别暴露
from .hydra_setup import register_hydra_resolvers

# 从 log.py 中提升装饰器
from .log import log_step

# 从 pathing.py 中提升核心路径获取函数
from .pathing import get_path

# --- 使用 __all__ 明确定义公共API ---
# 这份列表定义了当其他模块执行 `from helixpipe.utils import *` 时，
# 应该导入哪些名称。这也是对包的公共API的一种“文档”。
__all__ = [
    # debug.py
    validate_authoritative_dti_file,
    run_online_validation,
    validate_config_with_data,
    # download.py
    download_bindingdb_data,
    # hydra_setup.py
    register_hydra_resolvers,
    # log.py
    log_step,
    # pathing.py
    get_path,
]
