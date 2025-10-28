# 使用前向引用来避免循环导入
from typing import TYPE_CHECKING, List, Set, Tuple

import numpy as np
import pandas as pd
import research_template as rt
from tqdm import tqdm

from nasnet.configs import AppConfig
from nasnet.utils import get_path

if TYPE_CHECKING:
    # [MODIFIED] 不再需要 GraphBuildContext，因为它现在只在全局空间工作
    from .id_mapper import IDMapper


class SupervisionFileManager:
    """
    【V2 - 全局空间版】
    一个专门负责为一个Fold生成并持久化所有模型监督文件的服务类。
    它现在直接接收和处理【全局ID】对。
    """

    # [MODIFIED] __init__ 的签名已更新
    def __init__(
        self,
        fold_idx: int,
        config: AppConfig,
        # [REMOVED] 不再需要 context 对象
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
        self.labels_path_factory = get_path(
            config, "processed.specific.labels_template"
        )

    # [MODIFIED] generate_and_save 的签名已更新
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

    # [MODIFIED] _save_train_labels 的实现已简化
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

    # [MODIFIED] _save_test_labels 的实现已简化
    def _save_test_labels(self, test_pairs_global: List[Tuple[int, int, str]]):
        """处理测试标签文件 (正样本 + 负样本)，直接使用全局ID。"""

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

        # 负采样在全局空间进行
        negative_pairs_global = self._perform_negative_sampling(
            num_to_sample=len(test_pairs_for_eval_global)
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

    # [MODIFIED] _perform_negative_sampling 方法的签名和实现保持不变，因为它本来就在全局空间工作
    def _perform_negative_sampling(self, num_to_sample: int) -> List[Tuple[int, int]]:
        """在全局ID空间执行负采样。"""
        negative_pairs_global = []

        all_molecule_ids_global = list(self.global_id_mapper.molecule_to_id.values())
        all_protein_ids_global = list(self.global_id_mapper.protein_to_id.values())

        if not all_molecule_ids_global or not all_protein_ids_global:
            print("      - WARNING: Not enough entities to generate negative samples.")
            return []

        sampling_strategy = self.config.data_params.negative_sampling_strategy
        disable_tqdm = self.verbose == 0
        with tqdm(
            total=num_to_sample,
            desc=f"      - Neg Sampling for Test ({sampling_strategy})",
            disable=disable_tqdm,
        ) as pbar:
            while len(negative_pairs_global) < num_to_sample:
                mol_id_global = self.rng.choice(all_molecule_ids_global)
                prot_id_global = self.rng.choice(all_protein_ids_global)

                if (
                    mol_id_global,
                    prot_id_global,
                ) not in self.global_positive_pairs_set:
                    negative_pairs_global.append((mol_id_global, prot_id_global))
                    pbar.update(1)

        return negative_pairs_global
