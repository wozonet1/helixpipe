# src/nasnet/data_processing/services/splitter.py

from collections import Counter
from typing import TYPE_CHECKING, Iterator, List, Set, Tuple, Union

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from nasnet.configs import AppConfig
from nasnet.configs.training import EntitySelectorConfig  # 确保导入

# 使用前向引用进行类型提示
if TYPE_CHECKING:
    from .id_mapper import IDMapper


class DataSplitter:
    """
    【V6 - 终极版: 评估范围感知的划分器】
    它在划分前，会根据 evaluation_scope 配置预先分离出“可评估交互”和“背景知识”，
    确保划分比例的精确性，并使职责更加清晰。
    """

    def __init__(
        self,
        config: AppConfig,
        positive_pairs: List[Tuple[int, int, str]],
        id_mapper: "IDMapper",
        seed: int,
    ):
        """
        初始化DataSplitter，并执行核心的预处理步骤。
        """
        print("--- [DataSplitter V6] Initializing with evaluation-scope awareness...")
        self.config = config
        self.id_mapper = id_mapper
        self.seed = seed

        self.coldstart_cfg = config.training.coldstart
        self.num_folds = config.training.k_folds
        pool_scope = self.coldstart_cfg.pool_scope
        self.is_cold_start = (
            False
            if (
                pool_scope.entity_types is None
                and pool_scope.meta_types is None
                and pool_scope.from_sources is None
            )
            else True
        )
        # 在 __init__ 中执行一次智能修正
        self._initialize_scopes()

        # 【核心重构】预先分离“可评估交互”和“背景知识”
        self._evaluable_pairs, self._background_pairs = (
            self._presplit_by_evaluation_scope(positive_pairs)
        )

        # 后续所有划分，都只针对 evaluable_pairs
        self._pairs_to_split = self._evaluable_pairs

        # 初始化内部状态变量
        self._entities_to_split: List[Union[int, Tuple]] = []
        self._iterator: Union[Iterator, None] = None

        print(
            f"--> Splitter ready. Found {len(self._evaluable_pairs)} evaluable pairs and {len(self._background_pairs)} background knowledge pairs."
        )

    def _initialize_scopes(self):
        """检查并智能地修正 coldstart 配置中所有与 scope 相关的部分。"""
        # --- 修正 pool_scope ---
        pool_scope = self.coldstart_cfg.pool_scope
        is_pool_scope_default = (
            pool_scope.entity_types is None
            and pool_scope.meta_types is None
            and pool_scope.from_sources is None
        )
        if is_pool_scope_default:
            # 默认进行分子冷启动
            if self.config.runtime.verbose > 0:
                print(
                    "  - [DataSplitter] `pool_scope` is default. Auto-configuring to 'molecule' metatype cold-start."
                )
            self.coldstart_cfg.pool_scope.meta_types = ["molecule"]
            self.is_cold_start = True  # 更新状态

        # --- 修正 evaluation_scope ---
        if self.coldstart_cfg.evaluation_scope is None:
            primary_dataset = self.config.data_structure.primary_dataset
            if self.config.runtime.verbose > 0:
                print(
                    f"  - [DataSplitter] `evaluation_scope` is default. Auto-configuring to 'protagonist DTI' from primary dataset '{primary_dataset}'."
                )

            source_selector = EntitySelectorConfig(
                meta_types=["molecule"], from_sources=[primary_dataset]
            )
            target_selector = EntitySelectorConfig(
                meta_types=["protein"], from_sources=[primary_dataset]
            )
            relation_types = ["interaction"]
            self.coldstart_cfg.evaluation_scope = (
                source_selector,
                target_selector,
                relation_types,
            )

    # FIXME: 使用interaction_store 筛选
    def _presplit_by_evaluation_scope(
        self, all_pairs: List[Tuple[int, int, str]]
    ) -> Tuple[List, List]:
        """
        【核心修正】使用 IDMapper 的批量查询API来高效地划分。
        """
        evaluation_scope = self.coldstart_cfg.evaluation_scope

        if self.config.runtime.verbose > 0:
            print(
                "  - Pre-splitting pairs into 'evaluable' and 'background' sets (efficiently)..."
            )

        evaluable, background = [], []
        source_selector, target_selector = evaluation_scope

        # 1. 一次性获取所有满足条件的源实体和目标实体的ID集合
        source_pool = set(self.id_mapper.get_ids_by_selector(source_selector))
        target_pool = set(self.id_mapper.get_ids_by_selector(target_selector))

        # 2. 遍历交互，使用高效的 'in' 操作进行判断
        for pair in all_pairs:
            u, v, _ = pair

            # 检查 (u, v) 是否符合 (source, target) 或 (target, source) 的规则
            u_in_source = u in source_pool
            v_in_target = v in target_pool
            u_in_target = u in target_pool
            v_in_source = v in source_pool

            if (u_in_source and v_in_target) or (u_in_target and v_in_source):
                evaluable.append(pair)
            else:
                background.append(pair)

        return evaluable, background

    def _prepare_iterator(self):
        """根据 self.is_cold_start 的状态，初始化正确的sklearn迭代器。"""
        if self.num_folds <= 1:
            self._iterator = iter([None])
            return

        if self.is_cold_start:
            entity_pool_ids = self.id_mapper.get_ids_by_selector(
                self.coldstart_cfg.pool_scope
            )
            self._entities_to_split = sorted(entity_pool_ids)
            kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)
            self._iterator = iter(kf.split(self._entities_to_split))
        else:  # 热启动
            self._entities_to_split = self._pairs_to_split
            if not self._entities_to_split:
                self._iterator = iter([])  # 如果没有可评估的对，则迭代器为空
                return
            dummy_y = [p[1] for p in self._entities_to_split]
            class_counts = Counter(dummy_y)
            if len(class_counts) <= 1 or min(class_counts.values()) < self.num_folds:
                kf = KFold(
                    n_splits=self.num_folds, shuffle=True, random_state=self.seed
                )
                self._iterator = iter(kf.split(self._entities_to_split))
            else:
                skf = StratifiedKFold(
                    n_splits=self.num_folds, shuffle=True, random_state=self.seed
                )
                self._iterator = iter(skf.split(self._entities_to_split, dummy_y))

    def _split_data(
        self, split_result: Union[Tuple[List[int], List[int]], None]
    ) -> Tuple[List, List, Set, Set]:
        """
        【V8.1 - 最终修正版】
        执行实际的数据切分逻辑，并根据 'strictness' 精确地处理背景知识。
        """
        train_eval_pairs: List[Tuple[int, int, str]]
        test_eval_pairs: List[Tuple[int, int, str]]
        cold_start_entity_ids: Set[int] = set()

        if self.is_cold_start:
            # --- 冷启动逻辑 ---
            if self.num_folds > 1:  # K-Fold
                if not self._entities_to_split:
                    return self._pairs_to_split, [], set(), set()
                train_indices, test_indices = split_result
                test_entity_ids = {self._entities_to_split[i] for i in test_indices}
            else:  # Single Split
                entity_list = sorted(
                    list(
                        set(
                            self.id_mapper.get_ids_by_selector(
                                self.coldstart_cfg.pool_scope
                            )
                        )
                    )
                )
                if not entity_list or not self._pairs_to_split:
                    return self._pairs_to_split, [], set(), set()
                _, test_entities = train_test_split(
                    entity_list,
                    test_size=self.coldstart_cfg.test_fraction,
                    random_state=self.seed,
                )
                test_entity_ids = set(test_entities)

            cold_start_entity_ids = test_entity_ids
            train_eval_pairs, test_eval_pairs = [], []
            logic_id_to_type = self.id_mapper.logic_id_to_type_map
            # 动态推断划分的元类型
            target_metatype = (
                self.coldstart_cfg.pool_scope.meta_types[0]
                if self.coldstart_cfg.pool_scope.meta_types
                else "molecule"
            )
            is_target_metatype = (
                self.id_mapper.is_molecule
                if target_metatype == "molecule"
                else self.id_mapper.is_protein
            )

            for pair in self._pairs_to_split:
                u, v, _ = pair
                is_involved = (
                    is_target_metatype(logic_id_to_type.get(u, ""))
                    and u in test_entity_ids
                ) or (
                    is_target_metatype(logic_id_to_type.get(v, ""))
                    and v in test_entity_ids
                )
                if is_involved:
                    test_eval_pairs.append(pair)
                else:
                    train_eval_pairs.append(pair)
        else:
            # --- 热启动逻辑 ---
            if not self._pairs_to_split:
                return [], [], set()
            if self.num_folds > 1:
                train_indices, test_indices = split_result
                train_eval_pairs = [self._entities_to_split[i] for i in train_indices]
                test_eval_pairs = [self._entities_to_split[i] for i in test_indices]
            else:
                labels = [p[1] for p in self._pairs_to_split]
                try:
                    train_eval_pairs, test_eval_pairs = train_test_split(
                        self._pairs_to_split,
                        test_size=self.coldstart_cfg.test_fraction,
                        random_state=self.seed,
                        stratify=labels,
                    )
                except ValueError:
                    train_eval_pairs, test_eval_pairs = train_test_split(
                        self._pairs_to_split,
                        test_size=self.coldstart_cfg.test_fraction,
                        random_state=self.seed,
                        stratify=None,
                    )

        # 1. 定义最终的测试集 (保持不变)
        final_test_pairs = test_eval_pairs

        # 2. 定义最终的训练【标签】(监督信号)
        #    它只包含从可评估集中划分出来的训练部分。
        final_train_labels = train_eval_pairs

        # 3. 定义最终的训练【图结构边】
        #    它包含了训练标签，并根据strictness有条件地包含背景知识。
        final_train_graph_edges = list(train_eval_pairs)  # 显式拷贝

        if not self.is_cold_start:
            # 热启动时，所有背景知识都可以安全加入
            final_train_graph_edges.extend(self._background_pairs)
        else:
            # 冷启动时，需要根据 strictness 对背景知识进行筛选
            strictness_mode = self.coldstart_cfg.strictness
            if self.config.runtime.verbose > 0:
                print(
                    f"  - Handling {len(self._background_pairs)} background pairs with strictness='{strictness_mode}'..."
                )

            leaky_background_pairs_count = 0
            safe_background_pairs = []

            for pair in self._background_pairs:
                u, v, _ = pair
                is_leaky = (u in cold_start_entity_ids) or (v in cold_start_entity_ids)

                if not is_leaky:
                    safe_background_pairs.append(pair)
                elif strictness_mode == "informed":
                    # 在 informed 模式下，泄露的边也被加入图结构
                    safe_background_pairs.append(pair)
                    leaky_background_pairs_count += 1
                else:  # strict mode
                    leaky_background_pairs_count += 1

            final_train_graph_edges.extend(safe_background_pairs)
            # ... (日志打印逻辑可以相应调整)

        # 【修改2】返回四个值
        return (
            final_train_graph_edges,
            final_train_labels,
            final_test_pairs,
            cold_start_entity_ids,
        )

    def __iter__(self) -> "DataSplitter":
        self.fold_idx = 1
        self._prepare_iterator()
        return self

    def __next__(
        self,
    ) -> Tuple[
        int,
        List[Tuple[int, int, str]],
        List[Tuple[int, int, str]],
        List[Tuple[int, int, str]],
        Set[int],
    ]:
        """
        【修改】返回一个包含分离的训练图边和训练标签的元组。
        """
        if self.fold_idx > self.num_folds:
            raise StopIteration

        try:
            split_result = next(self._iterator)
        except StopIteration:
            raise StopIteration

        # 【修改3】解包四个返回值
        (
            final_train_graph_edges,
            final_train_labels,
            final_test_pairs,
            cold_start_entity_ids,
        ) = self._split_data(split_result)

        if self.config.runtime.verbose > 0:
            # 更新日志，使其反映新的数据结构
            print(
                f"  - [Splitter Fold {self.fold_idx}] "
                f"Train Graph Edges: {len(final_train_graph_edges)} | "
                f"Train Labels: {len(final_train_labels)} | "
                f"Test Labels: {len(final_test_pairs)}"
            )

        # 【修改4】返回五个值的元组
        result = (
            self.fold_idx,
            final_train_graph_edges,
            final_train_labels,
            final_test_pairs,
            cold_start_entity_ids,
        )
        self.fold_idx += 1
        return result
