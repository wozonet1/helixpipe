# 文件: src/nasnet/data_processing/services/selector_executor.py (终极版 V4)

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import pandas as pd

if TYPE_CHECKING:
    from nasnet.configs import EntitySelectorConfig, InteractionSelectorConfig

    from .id_mapper import IDMapper


class SelectorExecutor:
    def __init__(self, id_mapper: IDMapper, verbose: bool = False):
        self.id_mapper = id_mapper
        self.verbose = verbose  # 保持 verbose 以便未来调试

    def get_interaction_match_mask(
        self,
        df: pd.DataFrame,
        selector: InteractionSelectorConfig,
        source_col: str,
        target_col: str,
        relation_col: str,
    ) -> pd.Series:
        """
        【V4 - 逻辑无懈可击版】
        """
        if df.empty:
            return pd.Series(dtype=bool)

        # 1. 关系类型过滤 (保持不变)
        final_mask = pd.Series(True, index=df.index)
        if selector.relation_types:
            final_mask &= df[relation_col].isin(selector.relation_types)

        if not final_mask.any():
            return final_mask

        # 2. 实体过滤
        if selector.source_selector or selector.target_selector:
            df_to_check = df[final_mask]

            # --- 正向匹配掩码 ---
            s_matches_s = self._get_entity_column_match_mask(
                df_to_check[source_col], selector.source_selector
            )
            t_matches_t = self._get_entity_column_match_mask(
                df_to_check[target_col], selector.target_selector
            )
            match_forward = s_matches_s & t_matches_t

            # --- 反向匹配掩码 ---
            s_matches_t = self._get_entity_column_match_mask(
                df_to_check[source_col], selector.target_selector
            )
            t_matches_s = self._get_entity_column_match_mask(
                df_to_check[target_col], selector.source_selector
            )
            match_backward = s_matches_t & t_matches_s

            # --- 【核心修正】将结果安全地写回主掩码 ---
            # 只有在子掩码为False的地方，才将主掩码更新为False
            final_mask.loc[match_forward.index] &= match_forward | match_backward

        return final_mask

    def _get_entity_column_match_mask(
        self, entity_id_column: pd.Series, selector: EntitySelectorConfig | None
    ) -> pd.Series:
        # 这个辅助方法已经是正确的，保持不变
        if selector is None or not any(selector.__dict__.values()):
            return pd.Series(True, index=entity_id_column.index)

        unique_ids = entity_id_column.dropna().unique()
        meta_cache = {
            auth_id: self.id_mapper.get_meta_by_auth_id(auth_id)
            for auth_id in unique_ids
        }
        match_map = {
            uid: self._entity_meta_matches_selector(meta, selector)
            for uid, meta in meta_cache.items()
        }

        return entity_id_column.map(match_map).fillna(False)

    def _entity_meta_matches_selector(
        self, meta: Dict | None, selector: EntitySelectorConfig | None
    ) -> bool:
        # 这个辅助方法已经是正确的“严格模式”，保持不变
        if selector is None or not any(selector.__dict__.values()):
            return True
        if meta is None:
            return False
        if selector.entity_types and meta.get("type") not in selector.entity_types:
            return False
        if selector.meta_types:
            entity_type = meta.get("type")
            is_meta_type_match = (
                self.id_mapper.is_molecule(entity_type)
                and "molecule" in selector.meta_types
            ) or (
                self.id_mapper.is_protein(entity_type)
                and "protein" in selector.meta_types
            )
            if not is_meta_type_match:
                return False
        if selector.from_sources and meta.get("sources", set()).isdisjoint(
            selector.from_sources
        ):
            return False
        return True
