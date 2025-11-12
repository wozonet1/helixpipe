# 使用前向引用来避免循环导入
from typing import TYPE_CHECKING, List, Set, Tuple

import numpy as np
import pandas as pd
import research_template as rt
from tqdm import tqdm

from nasnet.configs import AppConfig

if TYPE_CHECKING:
    from nasnet.data_processing import IDMapper


class SupervisionFileManager:
    """
    【V2 - 全局空间版】
    一个专门负责为一个Fold生成并持久化所有模型监督文件的服务类。
    它现在直接接收和处理【全局ID】对，并能为任意目标边类型生成负样本。
    """

    def __init__(
        self,
        fold_idx: int,
        config: AppConfig,
        global_id_mapper: "IDMapper",
        global_positive_pairs_set: Set[Tuple[int, int]],
        seed: int,
    ):
        """
        在构造时，接收所有需要的全局信息。
        """
        self.fold_idx = fold_idx
        self.config = config
        self.global_id_mapper = global_id_mapper
        self.global_positive_pairs_set = global_positive_pairs_set
        self.verbose = config.runtime.verbose
        self.rng = np.random.default_rng(seed)

        self.labels_schema = config.data_structure.schema.internal.labeled_edges_output
        self.labels_path_factory = rt.get_path(
            config, "processed.specific.labels_template"
        )

    def generate_and_save(
        self,
        train_pairs_global: List[Tuple[int, int, str]],
        test_pairs_global: List[Tuple[int, int, str]],
    ):
        """
        一个高层次的公共方法，编排所有生成和保存的步骤。
        """
        if self.verbose > 0:
            print(f"    -> Generating label files for Fold {self.fold_idx}...")

        self._save_train_labels(train_pairs_global)
        self._save_test_labels(test_pairs_global)

    def _save_train_labels(self, train_pairs_global: List[Tuple[int, int, str]]):
        """处理训练标签文件 (仅正样本)，直接使用全局ID。"""

        train_labels_path = self.labels_path_factory(
            prefix=f"fold_{self.fold_idx}", suffix="train"
        )

        # 直接使用全局ID对创建DataFrame
        train_pairs_global_df = pd.DataFrame(
            [(u, v) for u, v, _ in train_pairs_global],
            columns=[self.labels_schema.source_node, self.labels_schema.target_node],
        )

        rt.ensure_path_exists(train_labels_path)
        train_pairs_global_df.to_csv(train_labels_path, index=False)

        if self.verbose > 0:
            print(
                f"      - Saved {len(train_pairs_global_df)} positive training pairs to '{train_labels_path.name}'."
            )

    def _save_test_labels(self, test_pairs_global: List[Tuple[int, int, str]]):
        """处理测试标签文件 (正样本 + 负样本)，使用类型感知的负采样。"""

        test_labels_path = self.labels_path_factory(
            prefix=f"fold_{self.fold_idx}", suffix="test"
        )

        test_pairs_for_eval_global = [(u, v) for u, v, _ in test_pairs_global]

        if not test_pairs_for_eval_global:
            if self.verbose > 0:
                print(
                    "      - WARNING: No positive pairs for the test set. Saving an empty label file."
                )
            pd.DataFrame(
                columns=[
                    self.labels_schema.source_node,
                    self.labels_schema.target_node,
                    self.labels_schema.label,
                ]
            ).to_csv(test_labels_path, index=False)
            return

        # 从配置中获取目标边类型
        edge_cfg = self.config.training.target_edge
        target_edge_type = (
            edge_cfg.source_type,
            edge_cfg.relation_type,
            edge_cfg.target_type,
        )

        # 调用新的、类型感知的负采样方法
        negative_pairs_global = self._perform_negative_sampling(
            num_to_sample=len(test_pairs_for_eval_global),
            target_edge_type=target_edge_type,
        )

        # 组合正负样本 (所有操作都在全局ID空间)
        test_pos_global_df = pd.DataFrame(
            test_pairs_for_eval_global,
            columns=[self.labels_schema.source_node, self.labels_schema.target_node],
        )
        test_pos_global_df[self.labels_schema.label] = 1

        neg_df_global = pd.DataFrame(
            negative_pairs_global,
            columns=[self.labels_schema.source_node, self.labels_schema.target_node],
        )
        neg_df_global[self.labels_schema.label] = 0

        labeled_df_global = (
            pd.concat([test_pos_global_df, neg_df_global], ignore_index=True)
            .sample(frac=1, random_state=self.rng)
            .reset_index(drop=True)
        )

        rt.ensure_path_exists(test_labels_path)
        labeled_df_global.to_csv(test_labels_path, index=False)

        if self.verbose > 0:
            ratio = (
                len(neg_df_global) / len(test_pos_global_df)
                if len(test_pos_global_df) > 0
                else 0
            )
            print(
                f"      - Saved {len(labeled_df_global)} labeled test pairs (1:{ratio:.0f} pos/neg ratio) to '{test_labels_path.name}'."
            )

    def _perform_negative_sampling(
        self, num_to_sample: int, target_edge_type: Tuple[str, str, str]
    ) -> List[Tuple[int, int]]:
        """
        【V2 - 类型感知版】
        在全局ID空间，为指定的目标边类型，执行随机负采样。
        """
        if self.verbose > 1:
            print(
                f"      - Performing negative sampling for edge type: {target_edge_type}"
            )

        negative_pairs_global = []

        source_type, _, target_type = target_edge_type

        # 动态地从 IDMapper 获取源和目标实体的“采样池”
        try:
            # 依赖于 IDMapper V4 的 entities_by_type 属性
            source_pool = self.global_id_mapper.entities_by_type[source_type]
            target_pool = self.global_id_mapper.entities_by_type[target_type]
        except KeyError as e:
            print(
                f"    - WARNING: Cannot perform negative sampling. Entity type '{e.args[0]}' not found in IDMapper."
            )
            return []

        if not source_pool or not target_pool:
            print(
                "      - WARNING: Not enough entities in source/target pool to generate negative samples."
            )
            return []

        # 通用的采样循环
        sampling_strategy = self.config.data_params.negative_sampling_strategy
        disable_tqdm = self.verbose == 0
        with tqdm(
            total=num_to_sample,
            desc=f"      - Neg Sampling for Test ({sampling_strategy})",
            disable=disable_tqdm,
        ) as pbar:
            while len(negative_pairs_global) < num_to_sample:
                source_id_global = self.rng.choice(source_pool)
                target_id_global = self.rng.choice(target_pool)

                if (
                    source_id_global,
                    target_id_global,
                ) not in self.global_positive_pairs_set:
                    # 额外检查：对于同质边(如PPI)，确保 u != v
                    if (
                        source_type == target_type
                        and source_id_global == target_id_global
                    ):
                        continue
                    negative_pairs_global.append((source_id_global, target_id_global))
                    pbar.update(1)

        return negative_pairs_global
