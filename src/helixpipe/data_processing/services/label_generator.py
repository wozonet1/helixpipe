import logging
from typing import TYPE_CHECKING, List, Set, Tuple

import numpy as np
import pandas as pd

import helixlib as hx
from helixpipe.typing import AppConfig
from helixpipe.utils import get_path

if TYPE_CHECKING:
    from helixpipe.typing import AuthID, LogicID

    from .id_mapper import IDMapper
    from .interaction_store import InteractionStore
    from .selector_executor import SelectorExecutor
logger = logging.getLogger(__name__)

LogicInteractionPair = Tuple[LogicID, LogicID]
AuthInteractionPair = Tuple[AuthID, AuthID]


class SupervisionFileManager:
    """
    【V4 - 逻辑与I/O分离版】
    为一个Fold生成并持久化所有模型监督文件的服务。
    核心逻辑被封装在返回DataFrame的私有方法中，以便于独立测试。
    """

    def __init__(
        self,
        fold_idx: int,
        config: AppConfig,
        id_mapper: "IDMapper",
        executor: "SelectorExecutor",
        global_positive_set: Set[LogicInteractionPair],
    ):
        """
        在构造时，接收所有需要的服务和全局状态。
        """
        self.fold_idx = fold_idx
        self.config = config
        self.id_mapper = id_mapper
        self.executor = executor
        self.global_positive_set = global_positive_set
        self.rng = np.random.default_rng(config.runtime.seed + fold_idx)

        self.labels_schema = config.data_structure.schema.internal.labeled_edges_output
        self.labels_path_factory = get_path(
            config, "processed.specific.labels_template"
        )
        self.verbose = config.runtime.verbose

    def generate_and_save(
        self,
        train_labels_store: "InteractionStore",
        test_store: "InteractionStore",
    ):
        """
        【编排器】调用核心逻辑方法创建DataFrame，并执行文件I/O操作。
        """
        if self.verbose > 0:
            logger.info(
                f"--- [SupervisionManager Fold {self.fold_idx}] Generating label files... ---"
            )

        # 1. 调用逻辑方法生成DataFrame
        train_df = self._prepare_train_df(train_labels_store)
        test_df = self._prepare_test_df(test_store)

        # 2. 执行文件写入
        train_labels_path = self.labels_path_factory(
            prefix=f"fold_{self.fold_idx}", suffix="train"
        )
        hx.ensure_path_exists(train_labels_path)
        train_df.to_csv(train_labels_path, index=False)
        if self.verbose > 0:
            logger.info(
                f"  - Saved {len(train_df)} positive training labels to '{train_labels_path.name}'."
            )

        test_labels_path = self.labels_path_factory(
            prefix=f"fold_{self.fold_idx}", suffix="test"
        )
        hx.ensure_path_exists(test_labels_path)
        test_df.to_csv(test_labels_path, index=False)
        if self.verbose > 0:
            pos_count = (test_df["label"] == 1).sum() if "label" in test_df else 0
            neg_count = len(test_df) - pos_count
            ratio = neg_count / pos_count if pos_count > 0 else 0
            logger.info(
                f"  - Saved {len(test_df)} labeled test pairs (1:{ratio:.1f} ratio) to '{test_labels_path.name}'."
            )

    def _prepare_train_df(self, train_labels_store: "InteractionStore") -> pd.DataFrame:
        """
        【核心逻辑】创建训练DataFrame (纯内存操作)。
        """
        logger.debug("--- [_prepare_train_df] START ---")
        if train_labels_store is None or train_labels_store.dataframe.empty:
            logger.warning(
                "Input 'train_labels_store' is empty. Creating an empty train DataFrame."
            )
            return pd.DataFrame(
                columns=[self.labels_schema.source_node, self.labels_schema.target_node]
            )

        train_pairs_logic = train_labels_store.get_mapped_positive_pairs(self.id_mapper)
        logger.debug(
            f"Mapped to {len(train_pairs_logic)} logic ID pairs: {train_pairs_logic}"
        )

        train_df = pd.DataFrame(
            [(u, v) for u, v, _ in train_pairs_logic],
            columns=[self.labels_schema.source_node, self.labels_schema.target_node],
        )
        logger.debug(f"Prepared train DataFrame with shape: {train_df.shape}")
        logger.debug("--- [_prepare_train_df] END ---")
        return train_df

    def _prepare_test_df(self, test_store: "InteractionStore") -> pd.DataFrame:
        """
        【核心逻辑】创建测试DataFrame，包含负采样 (纯内存操作)。
        """
        logger.debug("--- [_prepare_test_df] START ---")
        if test_store is None or test_store.dataframe.empty:
            logger.warning(
                f"Test store for fold {self.fold_idx} is empty. Creating an empty test DataFrame."
            )
            return pd.DataFrame(
                columns=[
                    self.labels_schema.source_node,
                    self.labels_schema.target_node,
                    self.labels_schema.label,
                ]
            )

        schema = self.config.data_structure.schema.internal.canonical_interaction
        positive_pairs_auth = [
            (row[schema.source_id], row[schema.target_id])
            for _, row in test_store.dataframe.iterrows()
        ]
        logger.debug(
            f"Extracted {len(positive_pairs_auth)} positive pairs (auth IDs): {positive_pairs_auth}"
        )

        negative_pairs_auth = self._perform_negative_sampling(len(positive_pairs_auth))

        pos_df = pd.DataFrame(positive_pairs_auth, columns=["s_id", "t_id"])
        pos_df["label"] = 1
        neg_df = pd.DataFrame(negative_pairs_auth, columns=["s_id", "t_id"])
        neg_df["label"] = 0
        logger.debug(
            f"Positive samples DF shape: {pos_df.shape}, Negative samples DF shape: {neg_df.shape}"
        )

        labeled_df_auth = pd.concat([pos_df, neg_df])

        labeled_df_auth["source"] = labeled_df_auth["s_id"].map(
            self.id_mapper.auth_id_to_logic_id_map
        )
        labeled_df_auth["target"] = labeled_df_auth["t_id"].map(
            self.id_mapper.auth_id_to_logic_id_map
        )

        final_df = labeled_df_auth[["source", "target", "label"]].dropna().astype(int)
        logger.debug(
            f"Final DataFrame after mapping and cleaning (shape: {final_df.shape})"
        )

        if final_df.empty:
            logger.warning("Final test DataFrame is empty after mapping and cleaning.")
            return final_df

        final_df_shuffled = final_df.sample(frac=1, random_state=self.rng).reset_index(
            drop=True
        )
        logger.debug(f"Prepared test DataFrame with shape {final_df_shuffled.shape}")
        logger.debug("--- [_prepare_test_df] END ---")
        return final_df_shuffled

    def _perform_negative_sampling(
        self, num_to_sample: int
    ) -> List[AuthInteractionPair]:
        """
        (私有) 执行配置驱动的负采样，返回权威ID对列表。
        """
        # ... (此方法的代码与我们之前确定的版本完全一致，无需修改) ...
        # (为了完整性，这里粘贴一份)
        if num_to_sample == 0:
            return []

        logger.debug("  - Performing negative sampling for test set...")

        neg_sampling_scope = self.config.training.coldstart.evaluation_scope

        if neg_sampling_scope is None:
            raise RuntimeError("evaluation_scope is not specified")

        source_pool_auth = self.executor.select_entities(
            neg_sampling_scope.source_selector
        )
        target_pool_auth = self.executor.select_entities(
            neg_sampling_scope.target_selector
        )

        if not source_pool_auth or not target_pool_auth:
            logger.warning(
                "Negative sampling pools are empty. Cannot generate negative samples."
            )
            return []

        negative_pairs_auth: list[AuthInteractionPair] = []
        source_list = list(source_pool_auth)
        target_list = list(target_pool_auth)

        max_tries = num_to_sample * 100
        tries = 0
        while len(negative_pairs_auth) < num_to_sample and tries < max_tries:
            source_id = self.rng.choice(source_list)
            target_id = self.rng.choice(target_list)

            # 使用全局正样本集合进行碰撞检查 (需要检查两个方向，因为集合是无向的)
            if (source_id, target_id) not in self.global_positive_set and (
                target_id,
                source_id,
            ) not in self.global_positive_set:
                negative_pairs_auth.append((source_id, target_id))

            tries += 1

        if len(negative_pairs_auth) < num_to_sample:
            logger.warning(
                f"Could only generate {len(negative_pairs_auth)} / {num_to_sample} negative samples after {max_tries} tries."
            )

        logger.debug(
            f"    - Successfully generated {len(negative_pairs_auth)} negative samples."
        )
        return negative_pairs_auth
