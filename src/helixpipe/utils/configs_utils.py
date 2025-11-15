# src/helixpipe/utils/config_utils.py (新文件或现有文件)

from typing import Any, Iterable

from omegaconf import DictConfig, OmegaConf


class SchemaAccessor:
    def __init__(self, schema_node: DictConfig):
        # schema_node 就是 self.config.data_structure.schema.external['gtopdb']
        self._node = schema_node

    def get_col(self, key_path: str) -> str:
        """
        安全地获取一个列名配置，并确保它是一个字符串。
        """
        try:
            value = OmegaConf.select(self._node, key_path)
            if not isinstance(value, str):
                raise TypeError(
                    f"Expected a string for schema key '{key_path}', but got {type(value).__name__}."
                )
            return value
        except Exception as e:
            # 可以在这里抛出一个更具体的自定义异常
            raise KeyError(f"Failed to resolve schema key '{key_path}': {e}") from e

    def get_field(self, key_path: str) -> dict[str, Any]:
        """
        安全地获取一个列名配置，并确保它是一个字符串。
        """
        try:
            value = OmegaConf.select(self._node, key_path)
            if not isinstance(value, dict):
                raise TypeError(
                    f"Expected a string for schema key '{key_path}', but got {type(value).__name__}."
                )
            return value
        except Exception as e:
            # 可以在这里抛出一个更具体的自定义异常
            raise KeyError(f"Failed to resolve schema key '{key_path}': {e}") from e

    def values(self) -> Iterable[Any]:
        """
        直接调用内部 OmegaConf 节点的 .values() 方法。
        注意：这只返回顶层的值，不会递归。
        """
        # DictConfig.values() 返回一个 ValuesView，它是一个 Iterable
        return self._node.values()
