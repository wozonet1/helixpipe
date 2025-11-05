from collections import Counter
from typing import TYPE_CHECKING, Iterator, List, Set, Tuple, Union

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from nasnet.configs import AppConfig

# 使用前向引用进行类型提示
if TYPE_CHECKING:
    from .id_mapper import IDMapper


class DataSplitter:
    """
    【V5 - 位置无知最终版】
    一个封装了所有数据划分策略的类。
    它现在通过查询IDMapper的类型判断接口，动态地、逐个地检查交互对中的
    每个实体，不再依赖任何关于实体位置的硬编码假设。
    """

    def __init__(
        self,
        config: AppConfig,
        positive_pairs: List[Tuple[int, int, str]],
        id_mapper: "IDMapper",
        seed: int,
    ):
        """
        初始化DataSplitter。
        """
        print("--- [DataSplitter V5] Initializing for position-agnostic splitting...")
        self.config = config
        self.positive_pairs = positive_pairs
        self.id_mapper = id_mapper
        self.seed = seed

        self.split_mode = config.training.coldstart.mode
        self.num_folds = config.training.k_folds
        self.test_fraction = config.training.coldstart.test_fraction

        self.entities_to_split: List[Union[int, Tuple]]
        self._iterator: Iterator

        self._prepare_iterator()
        print(
            f"--> Splitter ready for {self.num_folds}-fold splitting with mode: '{self.split_mode}'."
        )

    def _prepare_iterator(self):
        """根据配置，初始化正确的sklearn迭代器或设置单次运行模式。"""
        if self.num_folds > 1:
            if self.split_mode == "molecule":
                all_molecule_ids = []
                for entity_type in self.id_mapper.get_entity_types():
                    if self.id_mapper.is_molecule(entity_type):
                        all_molecule_ids.extend(
                            self.id_mapper.get_ordered_ids(entity_type)
                        )

                self.entities_to_split = sorted(all_molecule_ids)
                kf = KFold(
                    n_splits=self.num_folds, shuffle=True, random_state=self.seed
                )
                self._iterator = iter(kf.split(self.entities_to_split))

            elif self.split_mode == "protein":
                all_protein_ids = []
                for entity_type in self.id_mapper.get_entity_types():
                    if self.id_mapper.is_protein(entity_type):
                        all_protein_ids.extend(
                            self.id_mapper.get_ordered_ids(entity_type)
                        )

                self.entities_to_split = sorted(all_protein_ids)
                kf = KFold(
                    n_splits=self.num_folds, shuffle=True, random_state=self.seed
                )
                self._iterator = iter(kf.split(self.entities_to_split))

            else:  # "random" mode (热启动)
                self.entities_to_split = self.positive_pairs
                dummy_y = [p[1] for p in self.positive_pairs]
                class_counts = Counter(dummy_y)

                if min(class_counts.values()) < self.num_folds:
                    print(
                        "    - WARNING: Cannot perform stratified K-Fold. Falling back to regular K-Fold."
                    )
                    kf = KFold(
                        n_splits=self.num_folds, shuffle=True, random_state=self.seed
                    )
                    self._iterator = iter(kf.split(self.entities_to_split))
                else:
                    print("    - Using Stratified K-Fold for 'random' mode splitting.")
                    skf = StratifiedKFold(
                        n_splits=self.num_folds, shuffle=True, random_state=self.seed
                    )
                    self._iterator = iter(skf.split(self.entities_to_split, dummy_y))
        else:  # k=1, single split mode
            self._iterator = iter([None])

    def _split_data(
        self, split_result: Union[Tuple[List[int], List[int]], None]
    ) -> Tuple[List, List, Set]:
        """根据一次迭代的结果，执行实际的数据切分逻辑。"""

        train_pairs: List[Tuple[int, int, str]]
        test_pairs: List[Tuple[int, int, str]]
        cold_start_entity_ids: Set[int] = set()  # 默认为空集
        # --- K-Fold (k > 1) Logic ---
        if self.num_folds > 1:
            train_indices, test_indices = split_result
            if self.split_mode in ["molecule", "protein"]:
                test_entity_ids = {self.entities_to_split[i] for i in test_indices}
                cold_start_entity_ids = test_entity_ids  # <-- 我们需要的信息
                train_pairs = []
                test_pairs = []

                logic_id_to_type = self.id_mapper.get_logic_id_to_type_map()

                for pair in self.positive_pairs:
                    u, v, _ = pair
                    u_type = logic_id_to_type[u]
                    v_type = logic_id_to_type[v]

                    is_involved = False
                    if self.split_mode == "molecule":
                        if (
                            self.id_mapper.is_molecule(u_type) and u in test_entity_ids
                        ) or (
                            self.id_mapper.is_molecule(v_type) and v in test_entity_ids
                        ):
                            is_involved = True
                    elif self.split_mode == "protein":
                        if (
                            self.id_mapper.is_protein(u_type) and u in test_entity_ids
                        ) or (
                            self.id_mapper.is_protein(v_type) and v in test_entity_ids
                        ):
                            is_involved = True

                    if is_involved:
                        test_pairs.append(pair)
                    else:
                        train_pairs.append(pair)

            else:  # "random" mode
                train_pairs = [self.entities_to_split[i] for i in train_indices]
                test_pairs = [self.entities_to_split[i] for i in test_indices]

        # --- Single Split (k = 1) Logic ---
        else:
            if self.split_mode in ["molecule", "protein"]:
                all_entities = []
                is_target_type = (
                    self.id_mapper.is_molecule
                    if self.split_mode == "molecule"
                    else self.id_mapper.is_protein
                )
                for entity_type in self.id_mapper.get_entity_types():
                    if is_target_type(entity_type):
                        all_entities.extend(self.id_mapper.get_ordered_ids(entity_type))

                entity_list = sorted(all_entities)
                train_entities, test_entities = train_test_split(
                    entity_list, test_size=self.test_fraction, random_state=self.seed
                )
                test_entity_ids = set(test_entities)
                cold_start_entity_ids = test_entity_ids  # <-- 我们需要的信息
                train_pairs = []
                test_pairs = []
                logic_id_to_type = self.id_mapper.get_logic_id_to_type_map()

                for pair in self.positive_pairs:
                    u, v, _ = pair
                    u_type = logic_id_to_type[u]
                    v_type = logic_id_to_type[v]

                    is_involved = (is_target_type(u_type) and u in test_entity_ids) or (
                        is_target_type(v_type) and v in test_entity_ids
                    )

                    if is_involved:
                        test_pairs.append(pair)
                    else:
                        train_pairs.append(pair)

            else:  # "random" mode
                labels = [p[1] for p in self.positive_pairs]
                try:
                    train_pairs, test_pairs = train_test_split(
                        self.positive_pairs,
                        test_size=self.test_fraction,
                        random_state=self.seed,
                        stratify=labels,
                    )
                except ValueError:
                    print(
                        "    - WARNING: Cannot perform stratified train-test split. Falling back to regular split."
                    )
                    train_pairs, test_pairs = train_test_split(
                        self.positive_pairs,
                        test_size=self.test_fraction,
                        random_state=self.seed,
                        stratify=None,
                    )
        return train_pairs, test_pairs, cold_start_entity_ids

    def __iter__(self) -> "DataSplitter":
        """让这个类本身成为一个迭代器，返回自己。"""
        self.fold_idx = 1
        self._prepare_iterator()
        return self

    def __next__(
        self,
    ) -> Tuple[int, List[Tuple[int, int, str]], List[Tuple[int, int, str]], Set[int]]:
        """在每次迭代时，返回一个元组 (fold_idx, train_pairs, test_pairs, cold_start_entity_ids)。"""
        if self.fold_idx > self.num_folds:
            raise StopIteration

        split_result = next(self._iterator)
        train_pairs, test_pairs, cold_start_entity_ids = self._split_data(split_result)

        result = (self.fold_idx, train_pairs, test_pairs, cold_start_entity_ids)
        self.fold_idx += 1
        return result
