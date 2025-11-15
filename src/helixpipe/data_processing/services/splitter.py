import logging
from typing import Iterator, Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from helixpipe.typing import (
    AppConfig,
    EntitySelectorConfig,
    InteractionSelectorConfig,
    SplitResult,
)

from .id_mapper import IDMapper
from .interaction_store import InteractionStore
from .selector_executor import SelectorExecutor

logger = logging.getLogger(__name__)


class DataSplitter:
    """
    【V7 - "InteractionStore" 驱动版】
    一个智能的、由策略驱动的数据分区引擎。它消费InteractionStore对象，
    并为交叉验证的每个fold，产出划分好的InteractionStore子集。
    """

    def __init__(
        self,
        config: AppConfig,
        store: "InteractionStore",
        id_mapper: "IDMapper",
        executor: "SelectorExecutor",
        seed: int,
    ):
        """
        初始化DataSplitter，并执行核心的预处理步骤。
        """
        logger.info(
            "--- [DataSplitter V7] Initializing with InteractionStore and Executor..."
        )
        self.config = config
        self.store = store
        self.id_mapper = id_mapper
        self.executor = executor
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.coldstart_cfg = config.training.coldstart
        self.num_folds = config.training.k_folds
        self.is_cold_start = self.coldstart_cfg.mode == "cold"

        # 只有在确定是冷启动模式下，才去初始化 scope
        if self.is_cold_start:
            self._initialize_cold_start_scopes()

        self._evaluable_store, self._background_store = (
            self._presplit_by_evaluation_scope()
        )

        if self.is_cold_start:
            # 冷启动时，我们划分的是实体ID（权威ID）
            self._items_to_split: Union[list, pd.DataFrame] = sorted(
                list(self.executor.select_entities(self.coldstart_cfg.pool_scope))
            )
        else:
            # 热启动时，我们划分的是可评估的交互 DataFrame
            self._items_to_split = self._evaluable_store.dataframe

        self._iterator: Union[Iterator, None] = None
        logger.info(
            f"Splitter ready. Found {len(self._evaluable_store)} evaluable pairs and {len(self._background_store)} background pairs."
        )

    def _initialize_cold_start_scopes(self):
        """
        (私有) 检查并智能地修正 coldstart 配置中的 `pool_scope` 和 `evaluation_scope`。

        - 如果 pool_scope 未定义，默认为对“分子”元类型进行冷启动。
        - 如果 evaluation_scope 未定义，默认为评估“主角DTI”（即源自主要数据集的 drug-protein 交互）。
        """

        # --- 1. 修正 pool_scope (决定冷启动划分的对象) ---
        pool_scope = self.coldstart_cfg.pool_scope

        # 检查 pool_scope 是否为“空”的默认配置
        is_pool_scope_default = not (
            pool_scope.entity_types or pool_scope.meta_types or pool_scope.from_sources
        )

        if is_pool_scope_default:  # 只有在交叉验证时才应用默认冷启动
            logger.info(
                "  - [DataSplitter] `pool_scope` is default. Auto-configuring for 'molecule' metatype cold-start."
            )
            # 直接修改传入的 config 对象
            self.coldstart_cfg.pool_scope.meta_types = ["molecule"]

        # --- 2. 修正 evaluation_scope (决定最终评估的交互) ---
        if self.coldstart_cfg.evaluation_scope is None:
            # a. 从配置中获取主数据集的名称
            try:
                primary_dataset = self.config.data_structure.primary_dataset
            except Exception:
                # 如果找不到主数据集名称，无法设定默认值，这是一个错误
                raise ValueError(
                    "`data_structure.primary_dataset` must be defined to set a default evaluation_scope."
                )

            logger.info(
                f"  - [DataSplitter] `evaluation_scope` is not defined. "
                f"Auto-configuring to evaluate 'protagonist DTI' from primary dataset '{primary_dataset}'."
            )

            # b. 创建默认的选择器
            # 默认源: 是 'molecule' 元类型，并且必须来自主数据集
            source_selector = EntitySelectorConfig(
                meta_types=["molecule"], from_sources=[primary_dataset]
            )
            # 默认目标: 是 'protein' 元类型，并且也必须来自主数据集
            target_selector = EntitySelectorConfig(
                meta_types=["protein"], from_sources=[primary_dataset]
            )

            # c. 组合成 InteractionSelectorConfig 并赋值
            self.coldstart_cfg.evaluation_scope = InteractionSelectorConfig(
                source_selector=source_selector,
                target_selector=target_selector,
                # 默认情况下，我们不按关系类型进行限制
                relation_types=None,
            )

    def _presplit_by_evaluation_scope(
        self,
    ) -> tuple["InteractionStore", "InteractionStore"]:
        """使用 store.query() 和 store.difference() 高效分流。"""
        logger.info("Pre-splitting pairs into 'evaluable' and 'background' sets...")
        evaluation_scope = self.coldstart_cfg.evaluation_scope
        if evaluation_scope is None:
            raise RuntimeError("evaluation_scope is not specified")
        evaluable_store = self.store.query(evaluation_scope, self.id_mapper)
        background_store = self.store.difference(evaluable_store)
        return evaluable_store, background_store

    def _prepare_iterator(self):
        """根据 self.is_cold_start 的状态，初始化正确的sklearn迭代器。"""
        if self.num_folds <= 1 or len(self._items_to_split) == 0:
            self._iterator = iter([None])  # 单次运行或没有数据可分
            return

        if self.is_cold_start:
            # 冷启动：在实体列表上进行KFold
            if not isinstance(self._items_to_split, list):
                raise TypeError(
                    f"Cold-start mode requires a list of items to split, but got {type(self._items_to_split).__name__}"
                )
            kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)
            self._iterator = iter(kf.split(self._items_to_split))
        else:  # 热启动：在交互DataFrame上进行分层KFold
            if not isinstance(self._items_to_split, pd.DataFrame):
                raise TypeError(
                    f"Hot-start mode requires a DataFrame of interactions, but got {type(self._items_to_split).__name__}"
                )
            df = self._items_to_split
            # 使用 target 节点的权威ID作为分层依据
            y_stratify = df[self.store._schema.target_id]
            class_counts = y_stratify.value_counts()

            # 如果某个类别的样本数少于fold数，无法进行分层，回退到普通KFold
            if min(class_counts) < self.num_folds:
                logger.warning(
                    "Cannot use StratifiedKFold due to small class sizes. Falling back to KFold."
                )
                kf = KFold(
                    n_splits=self.num_folds, shuffle=True, random_state=self.seed
                )
                self._iterator = iter(kf.split(df))
            else:
                skf = StratifiedKFold(
                    n_splits=self.num_folds, shuffle=True, random_state=self.seed
                )
                self._iterator = iter(skf.split(df, y_stratify))

    def _split_data(self, split_result) -> SplitResult:
        """
        【V2 - 极限调试版】
        执行实际的数据切分，操作InteractionStore对象，并打印详细的日志。
        """
        logger.debug("--- [_split_data] START ---")

        train_eval_store: Optional[InteractionStore] = None
        test_store: Optional[InteractionStore] = None
        cold_start_entity_ids_auth: set = set()

        evaluable_df = self._evaluable_store.dataframe
        logger.debug(f"Evaluable store contains {len(evaluable_df)} interactions.")

        if self.is_cold_start and isinstance(self._items_to_split, list):
            # --- 冷启动逻辑 ---
            logger.debug("Mode: Cold-Start")
            if self._items_to_split:
                if self.num_folds > 1 and split_result:
                    _, test_indices = split_result
                    test_entity_ids = {self._items_to_split[i] for i in test_indices}
                elif self.num_folds <= 1:
                    # 单次运行
                    train_entities, test_entities = train_test_split(
                        self._items_to_split,
                        test_size=self.coldstart_cfg.test_fraction,
                        random_state=self.seed,
                        shuffle=True,
                    )
                    test_entity_ids = set(test_entities)
                else:  # 交叉验证但 split_result 为 None
                    test_entity_ids = set()
            else:  # 没有可划分的实体
                test_entity_ids = set()

            cold_start_entity_ids_auth = test_entity_ids
            logger.debug(
                f"Identified {len(cold_start_entity_ids_auth)} cold-start auth IDs: {cold_start_entity_ids_auth if len(cold_start_entity_ids_auth) < 10 else '...'}"
            )

            if not evaluable_df.empty:
                s_col, t_col = (
                    self._evaluable_store._schema.source_id,
                    self._evaluable_store._schema.target_id,
                )
                is_in_test = evaluable_df[s_col].isin(test_entity_ids) | evaluable_df[
                    t_col
                ].isin(test_entity_ids)

                test_df = evaluable_df[is_in_test]
                train_eval_df = evaluable_df[~is_in_test]
            else:
                test_df = pd.DataFrame()
                train_eval_df = pd.DataFrame()

            test_store = InteractionStore._from_dataframe(test_df, self.config)
            train_eval_store = InteractionStore._from_dataframe(
                train_eval_df, self.config
            )
        elif isinstance(self._items_to_split, pd.DataFrame):
            # --- 热启动逻辑 ---
            logger.debug("Mode: Hot-Start")
            if self._items_to_split.empty:
                logger.debug("Items to split is empty. Returning all empty stores.")
                return (
                    self._from_empty(),
                    self._from_empty(),
                    self._from_empty(),
                    set(),
                )

            if self.num_folds > 1 and split_result:
                train_indices, test_indices = split_result
                train_eval_df = self._items_to_split.iloc[train_indices]
                test_df = self._items_to_split.iloc[test_indices]
            else:  # 单次运行
                df = self._items_to_split
                y_stratify = df[self.store._schema.target_id]
                try:
                    train_eval_df, test_df = train_test_split(
                        df,
                        test_size=self.coldstart_cfg.test_fraction,
                        random_state=self.seed,
                        stratify=y_stratify,
                        shuffle=True,
                    )
                except ValueError:
                    logger.warning(
                        "Stratification failed in train_test_split. Falling back to random split."
                    )
                    train_eval_df, test_df = train_test_split(
                        df,
                        test_size=self.coldstart_cfg.test_fraction,
                        random_state=self.seed,
                        shuffle=True,
                    )

            test_store = InteractionStore._from_dataframe(test_df, self.config)
            train_eval_store = InteractionStore._from_dataframe(
                train_eval_df, self.config
            )
        if train_eval_store is None or test_store is None:
            raise RuntimeError("train_eval_store is None or test_store is None")
        logger.debug(
            f"Split results: train_eval_store size={len(train_eval_store)}, test_store size={len(test_store)}"
        )

        # --- 组装最终产物 ---
        final_test_store = test_store
        final_train_labels_store = train_eval_store

        train_graph_stores_to_concat = [final_train_labels_store]
        logger.debug(
            f"Initially, train_graph_stores_to_concat has 1 store (the train_labels_store with {len(final_train_labels_store)} interactions)."
        )

        if len(self._background_store) > 0:
            logger.debug(
                f"Found {len(self._background_store)} background interactions to process."
            )

            if not self.is_cold_start:
                logger.debug(
                    "Hot-start mode: Adding all background interactions to train graph."
                )
                train_graph_stores_to_concat.append(self._background_store)

            elif self.coldstart_cfg.strictness == "informed":
                logger.debug(
                    "Cold-start 'informed' mode: Adding all background interactions to train graph."
                )
                train_graph_stores_to_concat.append(self._background_store)

            elif self.coldstart_cfg.strictness == "strict":
                logger.debug(
                    "Cold-start 'strict' mode: Filtering background interactions..."
                )
                bg_df = self._background_store.dataframe
                s_col, t_col = (
                    self._background_store._schema.source_id,
                    self._background_store._schema.target_id,
                )

                is_leaky = bg_df[s_col].isin(cold_start_entity_ids_auth) | bg_df[
                    t_col
                ].isin(cold_start_entity_ids_auth)
                safe_bg_df = bg_df[~is_leaky]

                logger.debug(
                    f"Found {is_leaky.sum()} leaky interactions. Keeping {len(safe_bg_df)} safe background interactions."
                )

                if not safe_bg_df.empty:
                    safe_bg_store = InteractionStore._from_dataframe(
                        safe_bg_df, self.config
                    )
                    train_graph_stores_to_concat.append(safe_bg_store)
        else:
            logger.debug("No background interactions to process.")

        final_train_graph_store = InteractionStore.concat(
            train_graph_stores_to_concat, self.config
        )
        logger.debug(
            f"Final train_graph_store size after concatenation: {len(final_train_graph_store)}"
        )

        cold_start_entity_ids_logic = {
            self.id_mapper.auth_id_to_logic_id_map[auth_id]
            for auth_id in cold_start_entity_ids_auth
            if auth_id in self.id_mapper.auth_id_to_logic_id_map
        }

        logger.debug("--- [_split_data] END ---")
        return (
            final_train_graph_store,
            final_train_labels_store,
            final_test_store,
            cold_start_entity_ids_logic,
        )

    def __iter__(self) -> "DataSplitter":
        self.fold_idx = 1
        self._prepare_iterator()
        return self

    def __next__(
        self,
    ) -> tuple[int, SplitResult]:
        if self.fold_idx > self.num_folds:
            raise StopIteration
        if self._iterator is None:
            raise RuntimeError(
                "Iterator has not been initialized. Did you forget to use this class in a for loop?"
            )
        try:
            split_result = next(self._iterator)
        except StopIteration:
            raise StopIteration

        (
            final_train_graph_store,
            final_train_labels_store,
            final_test_store,
            cold_start_entity_ids_logic,
        ) = self._split_data(split_result)

        logger.info(
            f"[Splitter Fold {self.fold_idx}] "
            f"Train Graph: {len(final_train_graph_store)} edges | "
            f"Train Labels: {len(final_train_labels_store)} | "
            f"Test set: {len(final_test_store)}"
        )

        result = (
            self.fold_idx,
            (
                final_train_graph_store,
                final_train_labels_store,
                final_test_store,
                cold_start_entity_ids_logic,
            ),
        )
        self.fold_idx += 1
        return result

    def _from_empty(self) -> "InteractionStore":
        """(私有) 辅助函数，创建一个空的InteractionStore。"""
        return InteractionStore._from_dataframe(pd.DataFrame(), self.config)
