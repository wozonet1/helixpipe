from __future__ import annotations  # 允许类方法返回自身的类型提示

import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .selector_executor import SelectorExecutor

if TYPE_CHECKING:
    from helixpipe.configs import InteractionSelectorConfig
    from helixpipe.configs.data_structure import CanonicalInteractionSchema
    from helixpipe.typing import AppConfig, AuthID, LogicInteractionTriple

    from .id_mapper import IDMapper
import logging

logger = logging.getLogger(__name__)


class InteractionStore:
    """
    一个管理所有交互（边）关系的服务。

    它作为交互数据的中央仓库和查询引擎，与IDMapper（实体管理器）协同工作。
    它的核心数据是使用【权威ID】的DataFrame。
    查询方法遵循不可变模式，返回新的InteractionStore实例。
    """

    def __init__(
        self, processor_outputs: dict[str, pd.DataFrame], config: AppConfig
    ) -> None:
        """
        通过聚合所有 Processor 的输出来初始化，并立即规范化交互对。
        来源信息由 BaseProcessor 在输出时盖章为 source_dataset 列，
        此处只做 concat + 规范化。
        """
        self._config = config
        self._schema = config.data_structure.schema.internal.canonical_interaction
        self._verbose = config.runtime.verbose

        # 1. 聚合所有交互（source_dataset 列已由 BaseProcessor 盖章）
        all_dfs = [df.copy() for df in processor_outputs.values() if not df.empty]

        if not all_dfs:
            self._df = pd.DataFrame()
            logger.error("--- [InteractionStore] Initialized with an empty DataFrame.")
            return

        aggregated_df = pd.concat(all_dfs, ignore_index=True)

        # 防御性检查：兼容旧缓存 CSV 中可能缺少 source_dataset 列的情况
        src_col = self._schema.source_dataset
        if src_col not in aggregated_df.columns:
            logger.warning(
                f"    - '{src_col}' column missing from aggregated data. "
                "Filling with 'unknown'. Re-run processors with force_restart to fix."
            )
            aggregated_df[src_col] = "unknown"

        # 去重：同一交互来自多个数据源时合并 source_dataset
        aggregated_df = self._deduplicate_and_merge_source(aggregated_df, self._schema)

        aggregated_df[src_col] = aggregated_df[src_col].astype("category")
        logger.info(
            f"--- [InteractionStore] Initialized with {len(aggregated_df)} total raw interactions."
        )

        # 2. 规范化交互对（source/target 交换时，source_dataset 随行一起交换）
        self._df = self._canonicalize_interactions(aggregated_df)

    def _canonicalize_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        (私有) 根据配置的实体优先级，对交互对进行规范化排序。
        规则:
        1. 异质交互：priority 低的在 source, priority 高的在 target。
        2. 同质交互：ID 字典序小的在 source, 字典序大的在 target。
        """
        logger.info("    - Canonicalizing interaction pairs...")
        if df.empty:
            return df

        s_id_col, s_type_col = self._schema.source_id, self._schema.source_type
        t_id_col, t_type_col = self._schema.target_id, self._schema.target_type

        # 1. 准备优先级映射
        entity_meta = self._config.knowledge_graph.entity_meta
        # 创建一个从 entity_type -> priority 的字典，对于未知类型给予一个高优先级
        priority_map = {
            entity_type: meta.priority for entity_type, meta in entity_meta.items()
        }
        unknown_priority = 999

        # 2. 将优先级应用到DataFrame的每一行
        source_priorities = df[s_type_col].map(priority_map).fillna(unknown_priority)
        target_priorities = df[t_type_col].map(priority_map).fillna(unknown_priority)

        # 3. 找出所有需要交换的行
        # 条件1: target的优先级比source高 (e.g., source是protein, target是drug)
        # 条件2: 优先级相同 (同质交互)，但target的ID字典序比source小
        # 我们需要确保ID列是字符串类型以便于比较
        should_swap_mask = (source_priorities > target_priorities) | (
            (source_priorities == target_priorities)
            & (df[s_id_col].astype(str) > df[t_id_col].astype(str))
        )

        num_to_swap = should_swap_mask.sum()
        logger.info(
            f"      - Found {num_to_swap} / {len(df)} interactions to be swapped."
        )

        if num_to_swap == 0:
            return df

        # 4. 高效地执行交换
        # 创建一个DataFrame的副本以避免SettingWithCopyWarning
        df_swapped = df.copy()

        # a. 提取需要交换的行的 source 和 target 数据
        source_to_swap = df_swapped.loc[should_swap_mask, [s_id_col, s_type_col]].values
        target_to_swap = df_swapped.loc[should_swap_mask, [t_id_col, t_type_col]].values

        # b. 将它们互相赋值
        df_swapped.loc[should_swap_mask, [s_id_col, s_type_col]] = target_to_swap
        df_swapped.loc[should_swap_mask, [t_id_col, t_type_col]] = source_to_swap

        logger.info("    - Canonicalization complete.")
        return df_swapped

    @staticmethod
    def _deduplicate_and_merge_source(
        df: pd.DataFrame, schema: CanonicalInteractionSchema
    ) -> pd.DataFrame:
        """
        (静态私有) 基于 canonical 列去重，并合并 source_dataset。
        同一条交互来自多个数据源时，source_dataset 用 "|" 拼接（如 "bindingdb|gtopdb"）。
        """
        src_col = schema.source_dataset
        s_col = schema.source_id
        t_col = schema.target_id
        r_col = schema.relation_type

        if src_col not in df.columns or df.empty:
            return df

        dedup_cols = [s_col, t_col, r_col]
        num_before = len(df)

        def _merge_sources(series: pd.Series) -> str:
            return "|".join(sorted(set(series.astype(str))))

        deduped_df = df.groupby(dedup_cols, sort=False, as_index=False).agg(
            {col: "first" for col in df.columns if col not in dedup_cols}
            | {src_col: _merge_sources}
        )

        # 恢复列顺序和 category 类型
        deduped_df = deduped_df[df.columns]
        deduped_df[src_col] = deduped_df[src_col].astype("category")

        num_after = len(deduped_df)
        if num_before != num_after:
            logger.info(
                f"      - Deduplicated: {num_before} → {num_after} "
                f"({num_before - num_after} merged rows)."
            )
        return deduped_df

    @classmethod
    def _from_dataframe(
        cls,
        df: pd.DataFrame,
        config: AppConfig,
        *,
        skip_canonicalize: bool = False,
    ) -> InteractionStore:
        """
        私有工厂方法，通过已有的 DataFrame 创建新实例。

        Args:
            skip_canonicalize: 为 True 时跳过规范化（调用方保证输入已规范化）。
                               默认 False，自动执行规范化以维护不变量。
        """
        store = cls.__new__(cls)
        store._config = config
        store._schema = config.data_structure.schema.internal.canonical_interaction
        store._verbose = config.runtime.verbose
        if skip_canonicalize or df.empty:
            store._df = df
        else:
            store._df = store._canonicalize_interactions(df)
        return store

    @classmethod
    def concat(
        cls, stores: list[InteractionStore], config: AppConfig
    ) -> InteractionStore:
        """
        【类方法】将一个InteractionStore实例的列表合并成一个单一的、新的实例。

        Args:
            stores (list[InteractionStore]): 要合并的InteractionStore实例列表。
            config (AppConfig): 全局配置对象，用于初始化新的store实例。

        Returns:
            InteractionStore: 一个包含所有输入store中交互的新实例。
        """
        if not stores:
            return cls._from_dataframe(pd.DataFrame(), config, skip_canonicalize=True)

        # 提取每个store内部的DataFrame
        dataframes_to_concat = [
            store.dataframe for store in stores if not store.dataframe.empty
        ]

        if not dataframes_to_concat:
            return cls._from_dataframe(pd.DataFrame(), config, skip_canonicalize=True)

        # 使用pandas.concat进行高效合并（source_dataset 列自动跟随）
        concatenated_df = pd.concat(dataframes_to_concat, ignore_index=True)

        # 去重：同一交互来自多个 store 时合并 source_dataset
        schema = config.data_structure.schema.internal.canonical_interaction
        concatenated_df = cls._deduplicate_and_merge_source(concatenated_df, schema)

        return cls._from_dataframe(concatenated_df, config)

    def sample(
        self,
        fraction: float | None = None,
        n: int | None = None,
        seed: int | None = None,
    ) -> InteractionStore:
        """
        对内部的交互进行随机采样，并返回一个新的、经过采样的InteractionStore实例。

        智能处理 n 和 fraction 参数，n 优先。
        """
        if self._df.empty:
            return self

        if n is not None:
            sampled_df = self._df.sample(
                n=n, random_state=seed, replace=False, ignore_index=True
            )
        elif fraction is not None:
            if fraction >= 1.0:
                logger.warning("fraction >=1.0, skipping sampling...")
                return self._from_dataframe(
                    self._df.copy(), self._config, skip_canonicalize=True
                )
            sampled_df = self._df.sample(
                frac=fraction, random_state=seed, replace=False, ignore_index=True
            )
        else:
            logger.warning("no parameters was specified to sample")
            return self._from_dataframe(
                self._df.copy(), self._config, skip_canonicalize=True
            )

        logger.info(
            f"    - [InteractionStore.sample] Sampled {len(sampled_df)} interactions from {len(self._df)}."
        )

        return self._from_dataframe(sampled_df, self._config, skip_canonicalize=True)

    def difference(self, other: InteractionStore) -> InteractionStore:
        """
        【不可变操作 - 集合运算】
        返回一个新的InteractionStore，包含在当前store中但不在other store中的交互。

        这对于计算 a - b (例如，(所有交互) - (可评估交互) = 背景知识) 非常有用。
        匹配仅基于 canonical 交互列 (source_id, target_id, relation_type)，
        不受 source_dataset 等元数据列影响。

        Args:
            other (InteractionStore): 要减去的另一个store。

        Returns:
            InteractionStore: 一个代表差集的新实例。
        """
        if self._df.empty:
            return self
        if other._df.empty:
            return self._from_dataframe(
                self._df.copy(), self._config, skip_canonicalize=True
            )

        # 只基于 canonical 列构建匹配键，忽略 source_dataset 等元数据列
        s_col, t_col, r_col = (
            self._schema.source_id,
            self._schema.target_id,
            self._schema.relation_type,
        )

        self_keys = set(
            zip(
                self._df[s_col].astype(str),
                self._df[t_col].astype(str),
                self._df[r_col],
            )
        )
        other_keys = set(
            zip(
                other._df[s_col].astype(str),
                other._df[t_col].astype(str),
                other._df[r_col],
            )
        )

        diff_keys = self_keys - other_keys

        if not diff_keys:
            return self._from_dataframe(
                pd.DataFrame(columns=self._df.columns),
                self._config,
                skip_canonicalize=True,
            )

        # 用 canonical 列做布尔索引，保留行的所有列（含 source_dataset）
        key_col = (
            self._df[s_col].astype(str)
            + "|"
            + self._df[t_col].astype(str)
            + "|"
            + self._df[r_col].astype(str)
        )
        diff_mask = key_col.isin({"|".join(k) for k in diff_keys})

        difference_df = self._df[diff_mask].reset_index(drop=True)

        return self._from_dataframe(difference_df, self._config, skip_canonicalize=True)

    # --- 核心API ---

    def get_all_entity_auth_ids(self) -> set[AuthID]:
        """
        【提供给IDMapper】
        从所有交互中，提取出一个包含所有唯一实体权威ID的集合。
        """
        if self._df.empty:
            return set()

        source_ids = set(self._df[self._schema.source_id].dropna().unique())
        target_ids = set(self._df[self._schema.target_id].dropna().unique())
        return source_ids.union(target_ids)

    def filter_by_entities(self, valid_entity_ids: set[AuthID]) -> InteractionStore:
        """
        【不可变操作】
        接收一个纯净的实体ID集合，返回一个只包含"纯净"交互的新InteractionStore实例。
        """
        if self._df.empty:
            return self

        source_is_valid = self._df[self._schema.source_id].isin(valid_entity_ids)
        target_is_valid = self._df[self._schema.target_id].isin(valid_entity_ids)

        pure_df = self._df[source_is_valid & target_is_valid].copy()

        logger.info(
            f"    - [InteractionStore] Filtered by entities. {len(pure_df)} / {len(self._df)} interactions remain."
        )

        return self._from_dataframe(pure_df, self._config, skip_canonicalize=True)

    def get_mapped_positive_pairs(
        self, id_mapper: IDMapper
    ) -> list[LogicInteractionTriple]:
        """
        【提供给下游】
        将内部的、纯净的交互DataFrame，映射为使用【逻辑ID】的元组列表。
        这个方法应该在 self 和 id_mapper 都被最终化之后调用。
        """
        if self._df.empty:
            return []

        # 使用IDMapper的映射字典进行高效的批量转换
        source_logic_ids = self._df[self._schema.source_id].map(
            id_mapper.auth_id_to_logic_id_map
        )
        target_logic_ids = self._df[self._schema.target_id].map(
            id_mapper.auth_id_to_logic_id_map
        )

        # 健壮性检查：检查是否有映射失败的情况
        if source_logic_ids.isna().any() or target_logic_ids.isna().any():
            num_failed = source_logic_ids.isna().sum() + target_logic_ids.isna().sum()
            raise ValueError(
                f"InteractionStore contains {num_failed} IDs that are not in the finalized IDMapper. This indicates a logic error in the pipeline."
            )

        return list(
            zip(
                source_logic_ids.astype(int),
                target_logic_ids.astype(int),
                self._df[self._schema.relation_type],
            )
        )

    # --- 查询API (为Sampler和Splitter准备) ---

    def query(
        self, selector: InteractionSelectorConfig, id_mapper: IDMapper
    ) -> InteractionStore:
        """
        根据一个复杂的交互选择器，返回一个新的、只包含匹配结果的InteractionStore。

        此方法将所有复杂的匹配逻辑完全委托给 SelectorExecutor 服务。
        """
        if self._df.empty:
            logger.warning("    - [Store.query] Store is empty, returning self.")
            return self
        if selector is None:
            logger.warning("    - [Store.query] Selector is None, returning self.")
            return self

        executor = SelectorExecutor(id_mapper, verbose=self._verbose > 1)

        final_mask = executor.get_interaction_match_mask(
            df=self._df,
            selector=selector,
            source_col=self._schema.source_id,
            target_col=self._schema.target_id,
            relation_col=self._schema.relation_type,
        )

        logger.info(
            f"    - [Store.query] Executed selector, {final_mask.sum()} / {len(self._df)} interactions matched."
        )

        filtered_df = self._df.loc[final_mask]

        return self._from_dataframe(filtered_df, self._config, skip_canonicalize=True)

    def __len__(self) -> int:
        return len(self._df)

    @property
    def dataframe(self) -> pd.DataFrame:
        """提供对内部DataFrame的只读访问。"""
        return self._df.copy()

    def apply_sampling_strategy(
        self, config: AppConfig, executor: SelectorExecutor, seed: int
    ) -> InteractionStore:
        """
        根据配置，以流水线方式对Store中的交互执行一个完整的采样策略，
        并返回一个新的、经过采样的InteractionStore实例。

        执行顺序:
        1. (如果启用) 分层采样 (Stratified Sampling)。
        2. (如果启用) 在第一步的结果上，进行统一采样 (Uniform Sampling)。
        """
        sampling_cfg = config.data_params.sampling
        rng = np.random.default_rng(seed)
        logger.debug("--- [apply_sampling_strategy] START ---")
        logger.debug(f"Initial store size: {len(self._df)}")

        if not sampling_cfg.enabled or self._df.empty:
            logger.debug("Sampling disabled or store is empty. Returning self.")
            return self

        working_store = self

        # --- 阶段1: 分层采样 ---
        if sampling_cfg.stratified_sampling.enabled:
            working_store = self._apply_stratified_sampling(
                working_store, sampling_cfg.stratified_sampling, executor, rng
            )
            logger.debug(f"Store size after stratified sampling: {len(working_store)}")

        # --- 阶段2: 统一采样 ---
        if sampling_cfg.uniform_sampling.enabled:
            uniform_cfg = sampling_cfg.uniform_sampling
            if uniform_cfg.fraction < 1.0:
                logger.debug(
                    f"Applying Uniform Sampling with fraction: {uniform_cfg.fraction}"
                )

                len_before_uniform = len(working_store)
                # 检查以避免在空DataFrame上采样
                if len_before_uniform > 0:
                    raw_num_to_sample = len_before_uniform * uniform_cfg.fraction
                    num_to_sample = math.ceil(raw_num_to_sample)
                    logger.debug(
                        f"Calculation for uniform sampling: ceil({len_before_uniform} * {uniform_cfg.fraction}) = ceil({raw_num_to_sample}) = {num_to_sample}"
                    )

                    working_store = working_store.sample(
                        n=num_to_sample, seed=int(rng.integers(1_000_000))
                    )
                    logger.debug(
                        f"Store size after uniform sampling: {len(working_store)}"
                    )
                else:
                    logger.debug(
                        "Working store is empty before uniform sampling. Skipping."
                    )

        logger.debug(
            f"--- [apply_sampling_strategy] END --- Final size: {len(working_store)}"
        )
        return working_store

    def _apply_stratified_sampling(
        self,
        store: InteractionStore,
        stratified_cfg,
        executor: SelectorExecutor,
        rng: np.random.Generator,
    ) -> InteractionStore:
        logger.debug("  --- [_apply_stratified_sampling] START ---")

        # 1. 分层与归属
        strata_dfs = {}
        strata_counts = {}
        unclassified_mask = pd.Series(True, index=store.dataframe.index)

        for stratum_cfg in stratified_cfg.strata:
            mask = executor.get_interaction_match_mask(
                df=store.dataframe,
                selector=stratum_cfg.selector,
                source_col=self._schema.source_id,
                target_col=self._schema.target_id,
                relation_col=self._schema.relation_type,
            )
            # 确保一个交互只属于第一个匹配到的stratum
            mask &= unclassified_mask

            strata_dfs[stratum_cfg.name] = store.dataframe[mask]
            strata_counts[stratum_cfg.name] = mask.sum()
            unclassified_mask &= ~mask

        logger.debug(f"  Initial strata counts: {strata_counts}")

        final_sampled_dfs = []

        # 2. 第一轮循环：处理所有使用 fraction 的层 (包括锚点层)
        logger.debug("  --- First Pass: Fraction-based strata ---")
        for stratum_cfg in stratified_cfg.strata:
            if stratum_cfg.ratio_to is None:
                stratum_df = strata_dfs[stratum_cfg.name]
                if not stratum_df.empty:
                    len_before = len(stratum_df)
                    frac = (
                        stratum_cfg.fraction
                        if stratum_cfg.fraction is not None
                        else 1.0
                    )
                    raw_num = len_before * frac
                    num_to_sample = math.ceil(raw_num)

                    sampled_df = stratum_df.sample(n=num_to_sample, random_state=rng)
                    final_sampled_dfs.append(sampled_df)
                    # 更新计数值，为第二轮做准备
                    strata_counts[stratum_cfg.name] = len(sampled_df)
                    logger.debug(
                        f"    - Stratum '{stratum_cfg.name}': ceil({len_before} * {frac}) = ceil({raw_num}) = {num_to_sample}. Sampled {len(sampled_df)}."
                    )
                else:
                    logger.debug(
                        f"    - Stratum '{stratum_cfg.name}': Empty, nothing to sample."
                    )

        # 3. 第二轮循环：处理所有使用 ratio_to 的层
        logger.debug("  --- Second Pass: Ratio-based strata ---")
        logger.debug(f"  Strata counts before second pass: {strata_counts}")
        for stratum_cfg in stratified_cfg.strata:
            if stratum_cfg.ratio_to is not None:
                anchor_name = stratum_cfg.ratio_to
                if anchor_name not in strata_counts:
                    raise ValueError(
                        f"Anchor stratum '{anchor_name}' not found or was processed out of order."
                    )

                anchor_count = strata_counts[anchor_name]
                stratum_df = strata_dfs[stratum_cfg.name]

                if not stratum_df.empty:
                    len_before = len(stratum_df)
                    ratio = stratum_cfg.ratio if stratum_cfg.ratio is not None else 1.0
                    raw_num = anchor_count * ratio
                    num_to_sample = math.ceil(raw_num)
                    num_to_sample = min(num_to_sample, len_before)  # 不能超过上限

                    sampled_df = stratum_df.sample(n=num_to_sample, random_state=rng)
                    final_sampled_dfs.append(sampled_df)
                    logger.debug(
                        f"    - Stratum '{stratum_cfg.name}': ceil({anchor_count} * {ratio}) = ceil({raw_num}) = {num_to_sample}. Sampled {len(sampled_df)}."
                    )
                else:
                    logger.debug(
                        f"    - Stratum '{stratum_cfg.name}': Empty, nothing to sample."
                    )

        # 4. 收尾工作
        unclassified_df = store.dataframe[unclassified_mask]
        if not unclassified_df.empty:
            final_sampled_dfs.append(unclassified_df)
            logger.debug(f"  - Kept {len(unclassified_df)} unclassified interactions.")

        if not final_sampled_dfs:
            logger.debug(
                "  No data left after all sampling strata. Returning empty store."
            )
            return self._from_dataframe(
                pd.DataFrame(), self._config, skip_canonicalize=True
            )

        final_df = pd.concat(final_sampled_dfs, ignore_index=True)
        logger.debug(f"  Total size after concatenating all strata: {len(final_df)}")

        # 使用 frac=1 打乱顺序
        final_df_shuffled = final_df.sample(frac=1, random_state=rng).reset_index(
            drop=True
        )

        logger.debug("  --- [_apply_stratified_sampling] END ---")
        return self._from_dataframe(
            final_df_shuffled, self._config, skip_canonicalize=True
        )
