from __future__ import annotations  # 允许类方法返回自身的类型提示

from typing import TYPE_CHECKING, Any, Dict, List, Set, Tuple

import pandas as pd

if TYPE_CHECKING:
    from helixpipe.configs import (
        AppConfig,
        InteractionSelectorConfig,
    )

    from .id_mapper import IDMapper


class InteractionStore:
    """
    一个管理所有交互（边）关系的服务。

    它作为交互数据的中央仓库和查询引擎，与IDMapper（实体管理器）协同工作。
    它的核心数据是使用【权威ID】的DataFrame。
    查询方法遵循不可变模式，返回新的InteractionStore实例。
    """

    def __init__(self, processor_outputs: Dict[str, pd.DataFrame], config: AppConfig):
        """
        【V2 - 带规范化排序版】
        在流水线早期，通过聚合所有Processor的输出来初始化，并立即对交互对进行规范化。
        """
        self._config = config
        self._schema = config.data_structure.schema.internal.canonical_interaction
        self._verbose = config.runtime.verbose

        # 1. 聚合所有交互 (逻辑不变)
        all_dfs = []
        # ... (聚合 processor_outputs 的代码保持不变)
        for source_dataset, df in processor_outputs.items():
            if not df.empty:
                df_copy = df.copy()
                df_copy["__source_dataset__"] = source_dataset
                all_dfs.append(df_copy)

        if not all_dfs:
            self._df = pd.DataFrame()
            if self._verbose > 0:
                print("--- [InteractionStore] Initialized with an empty DataFrame.")
            return

        aggregated_df = pd.concat(all_dfs, ignore_index=True)
        if self._verbose > 0:
            print(
                f"--- [InteractionStore] Initialized with {len(aggregated_df)} total raw interactions."
            )

        # 2. 【核心新增】调用规范化方法
        self._df = self._canonicalize_interactions(aggregated_df)

    def _canonicalize_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        (私有) 根据配置的实体优先级，对交互对进行规范化排序。
        规则:
        1. 异质交互：priority 低的在 source, priority 高的在 target。
        2. 同质交互：ID 字典序小的在 source, 字典序大的在 target。
        """
        if self._verbose > 0:
            print("    - Canonicalizing interaction pairs...")
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
        if self._verbose > 0:
            print(
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

        if self._verbose > 0:
            print("    - Canonicalization complete.")
        return df_swapped

    @classmethod
    def _from_dataframe(cls, df: pd.DataFrame, config: AppConfig) -> InteractionStore:
        """一个私有的工厂方法，用于通过已有的DataFrame创建新实例 (用于查询方法)。"""
        store = cls.__new__(cls)
        store._config = config
        store._schema = config.data_structure.schema.internal.canonical_interaction
        store._verbose = config.runtime.verbose
        store._df = df
        return store

    @classmethod
    def concat(
        cls, stores: List[InteractionStore], config: AppConfig
    ) -> InteractionStore:
        """
        【类方法】将一个InteractionStore实例的列表合并成一个单一的、新的实例。

        Args:
            stores (List[InteractionStore]): 要合并的InteractionStore实例列表。
            config (AppConfig): 全局配置对象，用于初始化新的store实例。

        Returns:
            InteractionStore: 一个包含所有输入store中交互的新实例。
        """
        if not stores:
            # 如果输入列表为空，返回一个空的InteractionStore
            return cls._from_dataframe(pd.DataFrame(), config)

        # 提取每个store内部的DataFrame
        dataframes_to_concat = [
            store.dataframe for store in stores if not store.dataframe.empty
        ]

        if not dataframes_to_concat:
            return cls._from_dataframe(pd.DataFrame(), config)

        # 使用pandas.concat进行高效合并
        concatenated_df = pd.concat(dataframes_to_concat, ignore_index=True)

        # 使用工厂方法创建并返回新实例
        return cls._from_dataframe(concatenated_df, config)

    def sample(
        self, fraction: float = 1.0, n: int | None = None, seed: int | None = None
    ) -> InteractionStore:
        """
        【不可变操作】
        对内部的交互进行随机采样，并返回一个新的、经过采样的InteractionStore实例。

        支持按比例(fraction)或按绝对数量(n)进行采样。如果同时提供，n优先。

        Args:
            fraction (float): 要采样的比例 (0.0 to 1.0)。默认为 1.0 (不采样)。
            n (int | None): 要采样的绝对数量。如果提供，则忽略fraction。
            seed (int | None): 用于采样的随机种子，以保证可复现性。

        Returns:
            InteractionStore: 一个只包含采样后交互的新实例。
        """
        if self._df.empty:
            return self

        # 如果采样比例为1.0且没有指定n，则无需采样，直接返回自身拷贝
        if n is None and fraction >= 1.0:
            return self._from_dataframe(self._df.copy(), self._config)

        # 使用pandas内置的高效 .sample() 方法
        sampled_df = self._df.sample(
            n=n, frac=fraction, random_state=seed, replace=False, ignore_index=True
        )

        if self._verbose > 1:
            print(
                f"    - [InteractionStore.sample] Sampled {len(sampled_df)} interactions from {len(self._df)}."
            )

        return self._from_dataframe(sampled_df, self._config)

    def difference(self, other: InteractionStore) -> InteractionStore:
        """
        【不可变操作 - 集合运算】
        返回一个新的InteractionStore，包含在当前store中但不在other store中的交互。

        这对于计算 a - b (例如，(所有交互) - (可评估交互) = 背景知识) 非常有用。
        匹配基于所有列的完全匹配。

        Args:
            other (InteractionStore): 要减去的另一个store。

        Returns:
            InteractionStore: 一个代表差集的新实例。
        """
        if self._df.empty:
            return self
        if other._df.empty:
            return self._from_dataframe(self._df.copy(), self._config)

        # 为了高效地计算差集，我们将DataFrame转换为元组集合
        # 注意：这假设DataFrame的行序不重要
        current_set = set(self._df.itertuples(index=False, name=None))
        other_set = set(other._df.itertuples(index=False, name=None))

        difference_set = current_set - other_set

        if not difference_set:
            return self._from_dataframe(pd.DataFrame(), self._config)

        # 将结果转换回DataFrame
        difference_df = pd.DataFrame(list(difference_set), columns=self._df.columns)

        return self._from_dataframe(difference_df, self._config)

    # --- 核心API ---

    def get_all_entity_auth_ids(self) -> Set[Any]:
        """
        【提供给IDMapper】
        从所有交互中，提取出一个包含所有唯一实体权威ID的集合。
        """
        if self._df.empty:
            return set()

        source_ids = set(self._df[self._schema.source_id].dropna().unique())
        target_ids = set(self._df[self._schema.target_id].dropna().unique())
        return source_ids.union(target_ids)

    def filter_by_entities(self, valid_entity_ids: Set[Any]) -> InteractionStore:
        """
        【不可变操作】
        接收一个纯净的实体ID集合，返回一个只包含“纯净”交互的新InteractionStore实例。
        """
        if self._df.empty:
            return self

        source_is_valid = self._df[self._schema.source_id].isin(valid_entity_ids)
        target_is_valid = self._df[self._schema.target_id].isin(valid_entity_ids)

        pure_df = self._df[source_is_valid & target_is_valid].copy()

        if self._verbose > 0:
            print(
                f"    - [InteractionStore] Filtered by entities. {len(pure_df)} / {len(self._df)} interactions remain."
            )

        return self._from_dataframe(pure_df, self._config)

    def get_mapped_positive_pairs(
        self, id_mapper: IDMapper
    ) -> List[Tuple[int, int, str]]:
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
        【不可变操作】
        根据一个复杂的交互选择器，返回一个新的、只包含匹配结果的InteractionStore。
        """
        if self._df.empty or selector is None:
            return self

        # 1. 初始掩码：假设所有交互都通过
        final_mask = pd.Series(True, index=self._df.index)

        # 2. 逐一应用选择器规则

        # a. 按关系类型过滤 (最快，先执行)
        if selector.relation_types:
            final_mask &= self._df[self._schema.relation_type].isin(
                selector.relation_types
            )

        # b. 按源/目标节点过滤 (如果DataFrame已经很小，可以提前筛选以加速)
        if not final_mask.any():
            return self._from_dataframe(self._df.iloc[0:0], self._config)

        if selector.source_selector or selector.target_selector:
            # 【核心变化】我们不创建 subset_df，而是直接将当前的 final_mask 传入
            # _get_match_mask，让它只在需要的地方进行计算。
            entity_match_mask = self._get_match_mask(
                self._df, selector, id_mapper, initial_mask=final_mask
            )
            final_mask &= entity_match_mask

        # 3. 使用最终掩码筛选DataFrame，并创建新实例
        filtered_df = self._df[final_mask].copy()
        return self._from_dataframe(filtered_df, self._config)

    def __len__(self) -> int:
        return len(self._df)

    @property
    def dataframe(self) -> pd.DataFrame:
        """提供对内部DataFrame的只读访问。"""
        return self._df.copy()
