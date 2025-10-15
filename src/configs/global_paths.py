# 文件: src/configs/global_paths.py

from dataclasses import dataclass


@dataclass
class GlobalPathsConfig:
    data_root: str = "data"
    feature_cache_root: str = "data/global_feature_cache"
