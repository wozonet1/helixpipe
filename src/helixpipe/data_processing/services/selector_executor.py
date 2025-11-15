# 文件: src/helixpipe/data_processing/services/selector_executor.py (终极版 V4)

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Set

import pandas as pd

if TYPE_CHECKING:
    from helixpipe.typing import (
        AuthID,
        EntitySelectorConfig,
        InteractionSelectorConfig,
    )

    from .id_mapper import IDMapper

logger = logging.getLogger(__name__)


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
    ) -> pd.Series[bool]:
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
    ) -> pd.Series[bool]:
        # 这个辅助方法已经是正确的，保持不变
        if selector is None or not any(selector.__dict__.values()):
            return pd.Series(True, index=entity_id_column.index)

        unique_ids = entity_id_column.dropna().unique()
        meta_cache = {
            auth_id: self.id_mapper.get_meta_by_auth_id(auth_id)
            for auth_id in unique_ids
        }
        match_map = {}
        for uid, meta in meta_cache.items():
            if meta is None:
                # 核心规则：如果一个实体没有元数据，它不可能匹配任何有具体要求的选择器。
                # 只有当选择器为空时，它才可能“通过”（返回True）。
                match_map[uid] = False
            else:
                # 只有在 meta 存在时，才调用纯净版的匹配函数
                match_map[uid] = self._entity_meta_matches_selector(meta, selector)

        return entity_id_column.map(match_map).fillna(False)

    def _entity_meta_matches_selector(
        self, meta: Dict, selector: EntitySelectorConfig
    ) -> bool:
        # 这个辅助方法已经是正确的“严格模式”，保持不变
        if selector.entity_types and meta.get("type") not in selector.entity_types:
            return False
        if selector.meta_types:
            entity_type = meta.get("type")
            if entity_type is None:
                return False
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

    def select_entities(self, selector: EntitySelectorConfig | None) -> Set[AuthID]:
        """
        根据实体选择器，从 IDMapper 中筛选出匹配的【权威ID】集合。
        """
        if self.verbose:
            logger.debug(
                f"--- [Executor] Starting select_entities with selector: {selector}"
            )

        # 1. 获取所有最终化的实体ID作为全集
        all_final_auth_ids = self.id_mapper.get_all_final_ids()
        if not all_final_auth_ids:
            return set()

        # 2. 如果选择器为空，直接返回全集
        if selector is None or not any(selector.__dict__.values()):
            if self.verbose:
                logger.debug(
                    f"Selector is empty, returning all {len(all_final_auth_ids)} entities."
                )
            return set(all_final_auth_ids)

        # 3. 遍历全集，应用筛选逻辑
        matching_ids = set()
        for auth_id in all_final_auth_ids:
            # a. 获取元数据
            meta = self.id_mapper.get_meta_by_auth_id(auth_id)
            # [修改] 在这里处理 None
            if meta is None:
                continue  # 跳过后续的匹配
            # b. 调用我们已经测试过的、可靠的匹配函数
            if self._entity_meta_matches_selector(meta, selector):
                matching_ids.add(auth_id)

        if self.verbose:
            logger.debug(f"Found {len(matching_ids)} matching entities.")

        return matching_ids
