# src/nasnet/data_processing/services/id_mapper.py

from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

import pandas as pd
import research_template as rt

from nasnet.configs import AppConfig
from nasnet.utils import get_path


class IDMapper:
    """
    【V4 - 通用实体注册表版】
    一个纯粹的、通用的实体身份管理器。
    它接收一个或多个纯净的“规范化交互DataFrame”，并为所有实体分配和管理逻辑ID。
    核心功能包括：
    - 动态处理任意实体类型。
    - 基于配置的“类型映射策略”，用于控制语义粒度（例如，用于消融实验）。
    - 基于配置的“类型优先级合并”，用于解决同一实体在不同数据源中的类型冲突。
    """

    def __init__(self, pure_interactions_df: pd.DataFrame, config: AppConfig):
        """
        构造函数只负责从纯净的交互数据中收集所有原始的 (id, type) 对。
        """
        print("--- [IDMapper V4] Initializing with pure interaction data...")
        self._config = config
        self._schema = config.data_structure.schema.internal.canonical_interaction

        # 1. 收集所有唯一的 (id, type) 对
        source_entities = pure_interactions_df[
            [self._schema.source_id, self._schema.source_type]
        ].rename(
            columns={self._schema.source_id: "id", self._schema.source_type: "type"}
        )
        target_entities = pure_interactions_df[
            [self._schema.target_id, self._schema.target_type]
        ].rename(
            columns={self._schema.target_id: "id", self._schema.target_type: "type"}
        )

        collected_df = pd.concat([source_entities, target_entities]).drop_duplicates()

        self._collected_entities = collected_df.to_records(index=False)

        # 2. (可选) 应用类型映射策略
        mapping_strategy = self._config.knowledge_graph.type_mapping_strategy
        if mapping_strategy:
            if self._config.runtime.verbose > 0:
                print(
                    "  - Applying type mapping strategy for semantic granularity control..."
                )

            mapped_records = []
            for entity_id, entity_type in self._collected_entities:
                mapped_type = mapping_strategy.get(entity_type, entity_type)
                mapped_records.append((entity_id, mapped_type))

            self._collected_entities = (
                pd.DataFrame(mapped_records, columns=["id", "type"])
                .drop_duplicates()
                .to_records(index=False)
            )

            if self._config.runtime.verbose > 0:
                print(
                    f"    - Remapped entities. New unique (id, type) pairs: {len(self._collected_entities)}"
                )

        print(
            f"--> Collected {len(self._collected_entities)} final unique (id, type) pairs to be processed."
        )

        # 3. 初始化内部状态变量
        self.is_finalized = False
        self._final_entity_map: Dict[
            Any, str
        ] = {}  # { entity_id -> final_entity_type }
        self.entities_by_type: Dict[str, List] = defaultdict(list)
        self.entity_to_id_maps: Dict[str, Dict] = defaultdict(dict)
        self.id_to_entity_maps: Dict[str, Dict] = defaultdict(dict)
        self.logic_id_to_type_map: Dict[int, str] = {}
        self.logic_id_to_auth_id_map: Dict[int, Any] = {}

    def finalize_mappings(self):
        """
        在所有实体数据被收集和预处理后，执行类型合并和最终的ID分配。
        由于entity(类型映射或者清洗)本身可能相对容易改变,而逻辑id映射又是不应该被改变的,应该是一次性的,
        (虽然这不是计算上昂贵的),所以我们为了一种逻辑上的严谨性,
        设计一个"锁",确保两种id转换是永远满足不定式,不可修改的
        """
        if self.is_finalized:
            print("--> [IDMapper V4] Mappings already finalized. Skipping.")
            return

        print("\n--- [IDMapper V4] Finalizing all ID mappings with type merging... ---")

        type_priority = self._config.knowledge_graph.type_merge_priority

        # 1. 实现类型合并策略
        for entity_id, entity_type in self._collected_entities:
            current_type = self._final_entity_map.get(entity_id)
            if current_type is None:
                self._final_entity_map[entity_id] = entity_type
            else:
                current_priority = type_priority.get(current_type, 999)
                new_priority = type_priority.get(entity_type, 999)
                if new_priority < current_priority:
                    self._final_entity_map[entity_id] = entity_type

        # 2. 按最终确定的类型对实体进行分组
        for entity_id, final_type in self._final_entity_map.items():
            self.entities_by_type[final_type].append(entity_id)

        # 3. 分配连续的逻辑ID
        current_id = 0
        sorted_types = sorted(self.entities_by_type.keys())

        for entity_type in sorted_types:
            # 排序以保证映射的确定性
            sorted_entities = sorted(self.entities_by_type[entity_type])
            self.entities_by_type[entity_type] = sorted_entities

            type_map = {
                auth_id: i + current_id for i, auth_id in enumerate(sorted_entities)
            }

            self.entity_to_id_maps[entity_type] = type_map
            self.id_to_entity_maps[entity_type] = {
                i: auth_id for auth_id, i in type_map.items()
            }

            for auth_id, logic_id in type_map.items():
                self.logic_id_to_type_map[logic_id] = entity_type
                self.logic_id_to_auth_id_map[logic_id] = auth_id

            current_id += len(sorted_entities)

        self.is_finalized = True
        print("--- [IDMapper V4] Finalization complete. Final counts: ---")
        for entity_type in sorted_types:
            print(
                f"  - Found {self.get_num_entities(entity_type)} unique '{entity_type}' entities."
            )
        print(f"  - Total entities: {self.num_total_entities}")

    # --- 公共 Getter 和属性 ---

    def get_num_entities(self, entity_type: str) -> int:
        """获取指定类型的实体数量。"""
        return len(self.entities_by_type.get(entity_type, []))

    @property
    def num_total_entities(self) -> int:
        """获取所有实体的总数。"""
        return len(self._final_entity_map)

    def get_ordered_ids(self, entity_type: str) -> List:
        """获取指定类型的、排序后的权威ID列表。"""
        if not self.is_finalized:
            raise RuntimeError("IDMapper must be finalized first.")
        return self.entities_by_type.get(entity_type, [])

    def get_entity_types(self) -> List[str]:
        """获取所有最终存在的实体类型列表。"""
        return sorted(self.entities_by_type.keys())

    def get_logic_id_to_type_map(self) -> Dict[int, str]:
        """获取 逻辑ID -> 实体类型 的映射。"""
        return self.logic_id_to_type_map

    def is_molecule(self, entity_type: str) -> bool:
        """检查一个实体类型是否属于“分子”大类。"""
        # 这是一个简单的、基于字符串的约定
        return (
            "molecule" in entity_type
            or "drug" in entity_type
            or "ligand" in entity_type
        )

    def is_protein(self, entity_type: str) -> bool:
        """检查一个实体类型是否属于“蛋白质”大类。"""
        return "protein" in entity_type

    # --- 核心映射方法 ---

    def get_mapped_positive_pairs(
        self, pure_interactions_df: pd.DataFrame
    ) -> Tuple[List[Tuple[int, int, str]], Set[Tuple[int, int]]]:
        """
        将纯净的交互DataFrame高效地映射为逻辑ID对。
        """
        if not self.is_finalized:
            raise RuntimeError("IDMapper must be finalized before mapping pairs.")

        if self._config.runtime.verbose > 0:
            print("  - Mapping pure interactions to logic ID pairs...")

        # 1. 创建一个全局的 权威ID -> 逻辑ID 映射
        global_auth_to_logic_id_map = {}
        for type_map in self.entity_to_id_maps.values():
            global_auth_to_logic_id_map.update(type_map)

        # 2. 使用 .map() 进行高效映射
        source_logic_ids = pure_interactions_df[self._schema.source_id].map(
            global_auth_to_logic_id_map
        )
        target_logic_ids = pure_interactions_df[self._schema.target_id].map(
            global_auth_to_logic_id_map
        )

        # 3. 将结果与关系类型组合
        result_df = (
            pd.DataFrame(
                {
                    "source": source_logic_ids,
                    "target": target_logic_ids,
                    "rel": pure_interactions_df[self._schema.relation_type],
                }
            )
            .dropna()
            .astype({"source": int, "target": int})
        )

        # 4. 转换为所需的元组和集合格式
        mapped_pairs_with_type = list(result_df.to_records(index=False))
        mapped_pairs_set = set(zip(result_df["source"], result_df["target"]))

        if self._config.runtime.verbose > 0:
            print(
                f"    - Successfully mapped {len(mapped_pairs_set)} unique interaction pairs."
            )

        return mapped_pairs_with_type, mapped_pairs_set

    def save_maps_for_debugging(self, entities_with_structures: pd.DataFrame):
        """
        【V2 - 适配新架构版】将IDMapper内部的所有核心映射关系保存到磁盘 (nodes.csv)。
        现在需要外部传入包含结构信息的DataFrame。
        """
        if not self.is_finalized:
            print(
                "--> [IDMapper] Warning: Mappings not finalized. Skipping save_maps_for_debugging."
            )
            return

        print("\n--- [IDMapper V4] Saving internal maps for debugging...")

        nodes_schema = self._config.data_structure.schema.internal.nodes_output
        node_data = []

        # 遍历所有已分配逻辑ID的实体
        for logic_id, auth_id in self.logic_id_to_auth_id_map.items():
            entity_type = self.logic_id_to_type_map[logic_id]
            node_data.append(
                {
                    nodes_schema.global_id: logic_id,
                    nodes_schema.node_type: entity_type,
                    nodes_schema.authoritative_id: auth_id,
                }
            )

        if not node_data:
            print("--> No data to save for debugging.")
            return

        nodes_df = pd.DataFrame(node_data).sort_values(by=nodes_schema.global_id)

        # 【核心修改】将结构信息从外部DataFrame合并进来
        # 创建一个可用于合并的实体ID列
        entities_with_structures["auth_id_for_merge"] = entities_with_structures[
            "entity_id"
        ]
        nodes_df = pd.merge(
            nodes_df,
            entities_with_structures[["auth_id_for_merge", "structure"]],
            left_on="authoritative_id",
            right_on="auth_id_for_merge",
            how="left",
        ).drop(columns=["auth_id_for_merge"])

        # 重命名 'structure' 列以符合 schema
        nodes_df.rename(columns={"structure": nodes_schema.structure}, inplace=True)

        output_path = get_path(self._config, "processed.common.nodes_metadata")
        rt.ensure_path_exists(output_path)
        nodes_df.to_csv(output_path, index=False)
        print(f"--> Debug file 'nodes.csv' saved to: {output_path}")
