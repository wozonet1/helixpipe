from collections import Counter
from typing import TYPE_CHECKING, Iterator, List, Tuple, Union

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from nasnet.configs import AppConfig

# 使用前向引用进行类型提示
if TYPE_CHECKING:
    from .id_mapper import IDMapper


class DataSplitter:
    """
    【V3 - 全局空间划分版】
    一个封装了所有数据划分策略的类。

    它在初始化时接收全局数据，并作为一个迭代器，为每个Fold按需生成
    使用【全局逻辑ID】的训练集和测试集划分。
    """

    def __init__(
        self,
        config: AppConfig,
        positive_pairs: List[Tuple[int, int, str]],
        id_mapper: "IDMapper",  # [MODIFIED] 重新依赖于全局 IDMapper
        seed: int,
    ):
        """
        初始化DataSplitter。

        Args:
            config (AppConfig): 全局配置对象。
            positive_pairs (List[Tuple[int, int, str]]): 包含所有正样本全局逻辑ID对和关系类型的列表。
            id_mapper (IDMapper): 已最终化的【全局】IDMapper对象。
            seed (int): 用于本次划分的随机种子。
        """
        print("--- [DataSplitter V3] Initializing for global ID space splitting...")
        self.config = config
        self.positive_pairs = positive_pairs
        self.id_mapper = id_mapper  # [MODIFIED] 存储全局 id_mapper
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
                # [MODIFIED] 从全局 id_mapper 获取所有分子的全局逻辑ID
                self.entities_to_split = sorted(
                    list(self.id_mapper.molecule_to_id.values())
                )
                kf = KFold(
                    n_splits=self.num_folds, shuffle=True, random_state=self.seed
                )
                self._iterator = iter(kf.split(self.entities_to_split))

            elif self.split_mode == "protein":
                # [MODIFIED] 从全局 id_mapper 获取所有蛋白质的全局逻辑ID
                self.entities_to_split = sorted(
                    list(self.id_mapper.protein_to_id.values())
                )
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
    ) -> Tuple[List, List]:
        """根据一次迭代的结果，执行实际的数据切分逻辑。"""
        # --- K-Fold (k > 1) Logic ---
        if self.num_folds > 1:
            train_indices, test_indices = split_result
            if self.split_mode in ["molecule", "protein"]:
                test_entity_ids = {self.entities_to_split[i] for i in test_indices}
                entity_idx = 0 if self.split_mode == "molecule" else 1

                train_pairs = [
                    p
                    for p in self.positive_pairs
                    if p[entity_idx] not in test_entity_ids
                ]
                test_pairs = [
                    p for p in self.positive_pairs if p[entity_idx] in test_entity_ids
                ]
            else:  # "random" mode
                train_pairs = [self.entities_to_split[i] for i in train_indices]
                test_pairs = [self.entities_to_split[i] for i in test_indices]

        # --- Single Split (k = 1) Logic ---
        else:
            if self.split_mode in ["molecule", "protein"]:
                # [MODIFIED] 从全局 id_mapper 获取实体列表
                entity_list = (
                    sorted(list(self.id_mapper.molecule_to_id.values()))
                    if self.split_mode == "molecule"
                    else sorted(list(self.id_mapper.protein_to_id.values()))
                )
                train_entities, test_entities = train_test_split(
                    entity_list, test_size=self.test_fraction, random_state=self.seed
                )
                test_entity_ids = set(test_entities)
                entity_idx = 0 if self.split_mode == "molecule" else 1

                train_pairs = [
                    p
                    for p in self.positive_pairs
                    if p[entity_idx] not in test_entity_ids
                ]
                test_pairs = [
                    p for p in self.positive_pairs if p[entity_idx] in test_entity_ids
                ]
            else:  # "random" mode
                labels = [p[1] for p in self.positive_pairs]
                try:
                    # 尝试分层划分
                    train_pairs, test_pairs = train_test_split(
                        self.positive_pairs,
                        test_size=self.test_fraction,
                        random_state=self.seed,
                        stratify=labels,
                    )
                except ValueError:
                    # 如果分层失败 (例如某个类别样本太少)，则降级为普通划分
                    print(
                        "    - WARNING: Cannot perform stratified train-test split. Falling back to regular split."
                    )
                    train_pairs, test_pairs = train_test_split(
                        self.positive_pairs,
                        test_size=self.test_fraction,
                        random_state=self.seed,
                        stratify=None,
                    )
        return train_pairs, test_pairs

    def __iter__(self) -> "DataSplitter":
        """让这个类本身成为一个迭代器，返回自己。"""
        self.fold_idx = 1
        self._prepare_iterator()  # 每次开始新迭代时，都重新准备内部的sklearn迭代器
        return self

    def __next__(
        self,
    ) -> Tuple[int, List[Tuple[int, int, str]], List[Tuple[int, int, str]]]:
        """在每次迭代时，返回一个元组 (fold_idx, train_pairs, test_pairs)。"""
        if self.fold_idx > self.num_folds:
            raise StopIteration

        split_result = next(self._iterator)
        train_pairs, test_pairs = self._split_data(split_result)

        result = (self.fold_idx, train_pairs, test_pairs)
        self.fold_idx += 1
        return result
