# 文件: src/configs/global_paths.py

from dataclasses import dataclass


@dataclass
class GlobalPathsConfig:
    data_root: str = "data"
    cache_root: str = "${global_paths.data_root}/cache"
    features_cache_dir: str = "${global_paths.cache_root}/features"
    ids_cache_dir: str = "${global_paths.cache_root}/ids"
    assets_dir: str = "${global_paths.data_root}/assets"
