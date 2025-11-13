from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set

import pandas as pd
import research_template as rt

# TODO: 考虑解耦
from helixpipe.configs import AppConfig, EntitySelectorConfig
from helixpipe.configs.knowledge_graph import EntityMetaConfig
from helixpipe.utils import get_path

from .selector_executor import SelectorExecutor


class IDMapper:
    """
    【V5.1 - 元数据中心版】
    一个纯粹的、通用的实体身份管理器，能够追踪每个实体的来源和多重类型，
    并提供丰富的、由配置驱动的查询API。
    """

    def __init__(self, processor_outputs: Dict[str, pd.DataFrame], config: AppConfig):
        """
        构造函数从 Processor 的输出字典中，聚合所有实体的元信息，
        并预处理实体类型的元数据。
        """
        print(
            "--- [IDMapper V5.1] Initializing with metadata-aware entity aggregation..."
        )
        self._config = config
        self._schema = config.data_structure.schema.internal.canonical_interaction

        # --- 1. 预处理实体元类型 ---
        self._molecule_subtypes = set()
        self._protein_subtypes = set()
        entity_meta_config = config.knowledge_graph.entity_meta

        # 遍历所有已注册的实体类型及其元信息
        for entity_type, meta in entity_meta_config.items():
            if meta.metatype == "molecule":
                self._molecule_subtypes.add(entity_type)
            elif meta.metatype in [
                "protein",
                "gene",
            ]:  # 我们可以定义哪些元类型属于蛋白质大类
                self._protein_subtypes.add(entity_type)

        if self._config.runtime.verbose > 1:
            print(
                f"    - [DEBUG] Identified molecule subtypes: {self._molecule_subtypes}"
            )
            print(
                f"    - [DEBUG] Identified protein subtypes: {self._protein_subtypes}"
            )

        # --- 2. 聚合实体元信息 ---
        self._collected_entities: Dict[Any, Dict[str, Set]] = defaultdict(
            lambda: {"types": set(), "sources": set()}
        )
        for source_dataset, df in processor_outputs.items():
            if df.empty:
                continue

            source_df = df[[self._schema.source_id, self._schema.source_type]]
            for entity_id, entity_type in source_df.itertuples(index=False, name=None):
                self._collected_entities[entity_id]["types"].add(entity_type)
                self._collected_entities[entity_id]["sources"].add(source_dataset)

            target_df = df[[self._schema.target_id, self._schema.target_type]]
            for entity_id, entity_type in target_df.itertuples(index=False, name=None):
                self._collected_entities[entity_id]["types"].add(entity_type)
                self._collected_entities[entity_id]["sources"].add(source_dataset)

        if self._config.runtime.verbose > 0:
            print(
                f"--> Collected metadata for {len(self._collected_entities)} unique entities across all sources."
            )

        # --- 3. 初始化内部状态变量 ---
        self.is_finalized = False
        self._final_entity_map: Dict[Any, Dict[str, Any]] = {}
        self.entities_by_type: Dict[str, List] = defaultdict(list)
        self._logic_id_to_type_map: Dict[int, str] = {}
        self._logic_id_to_auth_id_map: Dict[int, Any] = {}
        self._auth_id_to_logic_id_map: Dict[Any, int] = {}

    def finalize_with_valid_entities(self, valid_entity_ids: Set[Any]):
        """
        接收一个纯净的实体ID集合，并只为这些实体执行类型合并和最终的ID分配。
        """
        verbose = self._config.runtime.verbose
        if self.is_finalized:
            print("--> [IDMapper V5.1] Mappings already finalized. Skipping.")
            return

        print(
            "\n--- [IDMapper V5.1] Finalizing mappings for valid entities only... ---"
        )

        entity_meta_config = self._config.knowledge_graph.entity_meta

        # 1. 类型合并，但只针对有效实体
        for entity_id in valid_entity_ids:
            if entity_id in self._collected_entities:
                meta = self._collected_entities[entity_id]
                entity_types = meta["types"]
                if not entity_types:
                    continue

                final_type = min(
                    entity_types,
                    key=lambda t: entity_meta_config.get(
                        t, EntityMetaConfig(metatype="unknown", priority=999)
                    ).priority,
                )
                self._final_entity_map[entity_id] = {
                    "type": final_type,
                    "sources": meta["sources"],
                }
                if verbose > 1:
                    print(
                        f"      - ID: {entity_id:<10} | Types: {str(entity_types):<30} | Final Type: {final_type}"
                    )
        # 2. 按最终确定的类型对实体进行分组
        for entity_id, final_meta in self._final_entity_map.items():
            final_type = final_meta["type"]
            self.entities_by_type[final_type].append(entity_id)
        if verbose > 1:
            for t, ents in self.entities_by_type.items():
                print(
                    f"      - Type: {t:<10} | Count: {len(ents)} | Sample IDs: {ents[:3]}"
                )
        # 3. 分配连续的逻辑ID
        current_id = 0
        sorted_types = sorted(self.entities_by_type.keys())
        if verbose > 1:
            print(f"      - ID assignment order: {sorted_types}")
        for entity_type in sorted_types:
            try:
                # 尝试进行排序。如果类型混合，这里会失败。
                sorted_entities = sorted(self.entities_by_type[entity_type])
            except TypeError:
                # [NEW] 捕获TypeError，并包装成一个信息量更大的ValueError
                # 我们可以做一个快速检查，找出罪魁祸首
                culprits = []
                for entity_id in self.entities_by_type[entity_type]:
                    if (
                        self.is_molecule(entity_type) and not isinstance(entity_id, int)
                    ) or (
                        self.is_protein(entity_type) and not isinstance(entity_id, str)
                    ):
                        culprits.append(
                            f"(ID: {entity_id}, Type: {type(entity_id).__name__})"
                        )

                raise ValueError(
                    f"ID format mismatch detected for entity type '{entity_type}'. "
                    f"This type expects a specific ID format, but received mixed types. "
                    f"This indicates an error in the upstream data validation. "
                    f"Problematic entities (sample): {culprits[:5]}"
                )
            self.entities_by_type[entity_type] = sorted_entities
            if verbose > 1:
                print(
                    f"        - Assigning IDs for type '{entity_type}' (starts at {current_id})..."
                )
            for logic_id, auth_id in enumerate(sorted_entities, start=current_id):
                # 构建logic_id -> type 的map
                self._logic_id_to_type_map[logic_id] = entity_type
                # 加入总体的auth_id <->logic_id map中
                self._logic_id_to_auth_id_map[logic_id] = auth_id
                self._auth_id_to_logic_id_map[auth_id] = logic_id
            current_id += len(sorted_entities)

        self.is_finalized = True
        print("--- [IDMapper V5.1] Finalization complete. Final counts: ---")
        for entity_type in sorted_types:
            print(
                f"  - Found {self.get_num_entities(entity_type)} unique '{entity_type}' entities."
            )
        print(f"  - Total entities: {self.num_total_entities}")
        if verbose > 1:
            print(self._auth_id_to_logic_id_map)
            print(self._logic_id_to_auth_id_map)

    def build_entity_manifest(self) -> pd.DataFrame:
        """
        根据收集到的原始实体信息，构建一个用于下游丰富化和校验的“实体清单”DataFrame。
        """
        print("--- [IDMapper V5.1] Building initial entity manifest...")
        manifest_data = []
        entity_meta_config = self._config.knowledge_graph.entity_meta

        for entity_id, meta in self._collected_entities.items():
            primary_type = min(
                meta["types"],
                key=lambda t: entity_meta_config.get(
                    t, EntityMetaConfig(metatype="unknown", priority=999)
                ).priority,
            )
            manifest_data.append(
                {
                    "entity_id": entity_id,
                    "entity_type": primary_type,
                    "all_types": list(meta["types"]),
                    "all_sources": list(meta["sources"]),
                    "structure": None,
                }
            )
        return pd.DataFrame(manifest_data)

    def save_maps_to_csv(self, entities_with_structures: pd.DataFrame):
        """
        将IDMapper内部的所有核心映射关系，结合外部提供的结构信息，保存到'nodes.csv'。
        """
        if not self.is_finalized:
            print(
                "--> [IDMapper] Warning: Mappings not finalized. Skipping saving debug maps."
            )
            return

        print("\n--- [IDMapper V5.1] Saving final nodes metadata ('nodes.csv')... ---")

        nodes_schema = self._config.data_structure.schema.internal.nodes_output
        node_data = []

        for logic_id, auth_id in self._logic_id_to_auth_id_map.items():
            meta = self._final_entity_map[auth_id]
            node_data.append(
                {
                    nodes_schema.global_id: logic_id,
                    nodes_schema.node_type: meta["type"],
                    nodes_schema.authoritative_id: auth_id,
                    "sources": ",".join(sorted(list(meta["sources"]))),
                }
            )

        if not node_data:
            print("--> No finalized data to save for debugging.")
            return

        nodes_df = pd.DataFrame(node_data).sort_values(by=nodes_schema.global_id)

        structure_map = entities_with_structures.set_index("entity_id")["structure"]
        nodes_df[nodes_schema.structure] = nodes_df[nodes_schema.authoritative_id].map(
            structure_map
        )
        nodes_df[nodes_schema.structure].fillna("", inplace=True)  # 确保没有NaN

        output_path = get_path(self._config, "processed.common.nodes_metadata")
        rt.ensure_path_exists(output_path)
        nodes_df.to_csv(output_path, index=False)
        print(f"--> Core metadata file 'nodes.csv' saved to: {output_path}")

    # --- 公共 Getter 和查询 API ---
    def get_meta_by_auth_id(self, auth_id: Any) -> Optional[Dict[str, Any]]:
        """【新增】根据【权威ID】，获取一个实体最终的元信息。"""
        if not self.is_finalized:
            raise RuntimeError("IDMapper must be finalized first.")

        return self._final_entity_map.get(auth_id)

    def get_meta_by_logic_id(self, logic_id: int) -> Optional[Dict[str, Any]]:
        """根据逻辑ID，获取一个实体最终的元信息（最终类型、来源等）。"""
        if not self.is_finalized:
            raise RuntimeError("IDMapper must be finalized first.")
        auth_id = self._logic_id_to_auth_id_map.get(logic_id)
        return self._final_entity_map.get(auth_id) if auth_id else None

    def get_all_final_ids(self) -> List[Any]:
        """返回所有【最终纯净的】权威实体ID。"""
        if not self.is_finalized:
            raise RuntimeError("IDMapper must be finalized first.")
        return list(self._final_entity_map.keys())

    def get_ids_by_filter(self, filter_func: Callable[[Dict], bool]) -> List[Any]:
        """一个通用的查询接口，返回所有元信息满足过滤条件的【逻辑ID】。"""
        if not self.is_finalized:
            raise RuntimeError("IDMapper must be finalized before filtering.")

        return [
            self._auth_id_to_logic_id_map[entity_id]
            for entity_id, meta in self._final_entity_map.items()
            if filter_func(meta)
        ]

    def get_ids_by_selector(self, selector: EntitySelectorConfig) -> List[int]:
        """
        【V2 - Executor委托版】
        一个强大的查询接口，它将解析Selector的任务完全委托给SelectorExecutor。
        """
        if not self.is_finalized:
            raise RuntimeError("IDMapper must be finalized before using selectors.")

        # 1. 实例化 Executor (即用即创)
        #    注意：Executor 需要一个 IDMapper 实例，我们把自己传进去
        executor = SelectorExecutor(self)

        # 2. 调用 Executor 获取匹配的【权威ID】集合
        matching_auth_ids = executor.select_entities(selector)

        # 3. 将权威ID集合转换为逻辑ID列表
        #    使用列表推导式和字典的 .get() 避免因ID不存在而崩溃
        logic_ids = [
            self._auth_id_to_logic_id_map.get(auth_id)
            for auth_id in matching_auth_ids
            if auth_id in self._auth_id_to_logic_id_map
        ]

        return logic_ids

    @property
    def num_molecules(self) -> int:
        """【V2】返回所有“分子”大类的实体总数。"""
        if not self.is_finalized:
            raise RuntimeError("IDMapper must be finalized before querying counts.")

        # 遍历所有已知的实体类型，累加所有属于“分子”元类型的实体数量
        count = 0
        for entity_type in self.entity_types:
            if self.is_molecule(entity_type):
                count += self.get_num_entities(entity_type)
        return count

    @property
    def num_proteins(self) -> int:
        """【V2】返回所有“蛋白质/基因”大类的实体总数。"""
        if not self.is_finalized:
            raise RuntimeError("IDMapper must be finalized before querying counts.")

        count = 0
        for entity_type in self.entity_types:
            if self.is_protein(entity_type):
                count += self.get_num_entities(entity_type)
        return count

    @property
    def num_total_entities(self) -> int:
        """【V2】返回所有实体总数。"""
        if not self.is_finalized:
            raise RuntimeError("IDMapper must be finalized before querying counts.")
        return len(self._logic_id_to_auth_id_map)

    def get_num_entities(self, entity_type: str) -> int:
        """
        获取指定【最终】实体类型的实体数量。

        Args:
            entity_type (str): 要查询的最终实体类型字符串 (e.g., 'drug', 'protein')。

        Returns:
            int: 该类型的实体数量。如果类型不存在，返回0。
        """
        if not self.is_finalized:
            raise RuntimeError(
                "IDMapper must be finalized before querying entity counts."
            )
        return len(self.entities_by_type.get(entity_type, []))

    def get_ordered_ids(self, entity_type: str) -> List:
        if not self.is_finalized:
            raise RuntimeError("IDMapper must be finalized first.")
        return self.entities_by_type.get(entity_type, [])

    @property
    def entity_types(self) -> List[str]:
        """获取所有最终存在的、排序后的实体类型列表。"""
        if not self.is_finalized:
            raise RuntimeError("IDMapper must be finalized first.")
        return sorted(self.entities_by_type.keys())

    @property
    def logic_id_to_auth_id_map(self) -> Dict[int, Any]:
        """获取 逻辑ID -> 权威ID 的映射字典。"""
        if not self.is_finalized:
            raise RuntimeError("IDMapper must be finalized first.")
        return self._logic_id_to_auth_id_map

    @property
    def logic_id_to_type_map(self) -> Dict[int, str]:
        """获取 逻辑ID -> 实体类型 的映射字典。"""
        if not self.is_finalized:
            raise RuntimeError("IDMapper must be finalized first.")
        return self._logic_id_to_type_map

    @property
    def auth_id_to_logic_id_map(self) -> Dict[Any, int]:
        """获取 权威ID -> 逻辑ID 的映射字典。"""
        if not self.is_finalized:
            raise RuntimeError("IDMapper must be finalized first.")
        return self._auth_id_to_logic_id_map

    def is_molecule(self, entity_type: str) -> bool:
        """通过查询配置，检查一个实体类型是否属于“分子”大类。"""
        return entity_type in self._molecule_subtypes

    def is_protein(self, entity_type: str) -> bool:
        """通过查询配置，检查一个实体类型是否属于“蛋白质/基因”大类。"""
        return entity_type in self._protein_subtypes
