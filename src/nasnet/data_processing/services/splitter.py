from collections import Counter
from typing import TYPE_CHECKING, Iterator, List, Tuple, Union

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from nasnet.typing import AppConfig

# 使用前向引用进行类型提示，避免循环导入问题
if TYPE_CHECKING:
    from .id_mapper import IDMapper


class DataSplitter:
    """
    一个封装了所有数据划分策略的类。

    它在初始化时接收所有必要的数据和配置，然后可以作为一个迭代器，
    为每个Fold按需生成划分好的训练集和测试集。
    """

    def __init__(
        self,
        config: AppConfig,
        positive_pairs: List[Tuple[int, int]],
        id_mapper: "IDMapper",
    ):
        """
        初始化DataSplitter。

        Args:
            config (DictConfig): 全局配置对象。
            positive_pairs (List[Tuple[int, int]]): 包含所有正样本逻辑ID对的列表。
            id_mapper (IDMapper): 已初始化的IDMapper对象。
        """
        print("--- [DataSplitter] Initializing... ---")
        self.config = config
        self.positive_pairs = positive_pairs
        self.id_mapper = id_mapper

        self.split_mode = config.training.coldstart.mode
        self.num_folds = config.training.k_folds
        self.seed = config.runtime.seed
        self.test_fraction = config.training.coldstart.test_fraction

        self.entities_to_split: List[int]
        self._iterator: Iterator

        # 在初始化时就准备好迭代器，使其 ready to use
        self._prepare_iterator()
        print(
            f"--> Splitter ready for {self.num_folds}-fold splitting with mode: '{self.split_mode}'."
        )

    def _prepare_iterator(self):
        """
        根据配置，私下里初始化正确的sklearn迭代器或设置单次运行模式。
        """
        if self.num_folds > 1:
            if self.split_mode == "molecule":
                # 【核心变化】从 id_mapper 获取所有分子的逻辑ID
                self.entities_to_split = sorted(
                    list(self.id_mapper.molecule_to_id.values())
                )
                kf = KFold(
                    n_splits=self.num_folds, shuffle=True, random_state=self.seed
                )
                self._iterator = iter(kf.split(self.entities_to_split))
            elif self.split_mode == "protein":
                # 【核心变化】从 id_mapper 获取所有蛋白质的逻辑ID
                self.entities_to_split = sorted(
                    list(self.id_mapper.protein_to_id.values())
                )
                kf = KFold(
                    n_splits=self.num_folds, shuffle=True, random_state=self.seed
                )
                self._iterator = iter(kf.split(self.entities_to_split))
            else:  # "random" mode (热启动)
                self.entities_to_split = self.positive_pairs
                # 使用蛋白质ID进行分层，确保每折的蛋白质分布相似
                dummy_y = [p[1] for p in self.positive_pairs]
                class_counts = Counter(dummy_y)

                # 如果最小的类别数量小于K折的数量，则无法进行分层
                if min(class_counts.values()) < self.num_folds:
                    print(
                        f"    - WARNING: Cannot perform stratified K-Fold because some proteins appear "
                        f"fewer than {self.num_folds} times. Falling back to regular K-Fold."
                    )
                    # 2. 降级到普通的 KFold
                    kf = KFold(
                        n_splits=self.num_folds, shuffle=True, random_state=self.seed
                    )
                    self._iterator = iter(kf.split(self.entities_to_split))
                else:
                    # 3. 如果可以，则继续使用 StratifiedKFold
                    print("    - Using Stratified K-Fold for 'random' mode splitting.")
                    skf = StratifiedKFold(
                        n_splits=self.num_folds, shuffle=True, random_state=self.seed
                    )
                    self._iterator = iter(skf.split(self.entities_to_split, dummy_y))
        else:  # k=1, single split mode
            # 创建一个只包含一个None元素的迭代器，以触发一次循环
            self._iterator = iter([None])

    def _split_data(
        self, split_result: Union[Tuple[List[int], List[int]], None]
    ) -> Tuple[List, List]:
        """
        根据一次迭代的结果 (train/test indices)，执行实际的数据切分逻辑。
        """
        if self.num_folds > 1:  # K-Fold logic
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
        else:  # Single Split logic (k=1)
            if self.split_mode in ["molecule", "protein"]:
                # 【核心变化】同样从 id_mapper 获取实体列表
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
                label_counts = Counter(labels)
                if min(label_counts.values()) < 2:
                    print(
                        "    - WARNING: Cannot perform stratified train-test split. Falling back to regular split."
                    )
                    train_pairs, test_pairs = train_test_split(
                        self.positive_pairs,
                        test_size=self.test_fraction,
                        random_state=self.seed,
                        stratify=None,
                    )
                else:
                    train_pairs, test_pairs = train_test_split(
                        self.positive_pairs,
                        test_size=self.test_fraction,
                        random_state=self.seed,
                        stratify=labels,
                    )
        return train_pairs, test_pairs

    def __iter__(self) -> "DataSplitter":
        """
        让这个类本身成为一个迭代器，返回自己。
        每次开始新的迭代时，都重新准备内部的sklearn迭代器。
        """
        self.fold_idx = 1
        self._prepare_iterator()
        return self

    def __next__(self) -> Tuple[int, List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        在每次迭代时，返回一个元组 (fold_idx, train_pairs, test_pairs)。
        这是 for 循环的核心。
        """
        if self.fold_idx > self.num_folds:
            raise StopIteration

        # 从内部迭代器获取下一次的划分索引
        split_result = next(self._iterator)

        # 使用这些索引来执行数据切分
        train_pairs, test_pairs = self._split_data(split_result)

        result = (self.fold_idx, train_pairs, test_pairs)
        self.fold_idx += 1
        return result
