# 文件: src/nasnet/data_processing/services/id_mapper.py (最终架构版)

from typing import Dict, List, Set

import pandas as pd
import research_template as rt

from nasnet.configs import AppConfig
from nasnet.utils import get_path


class IDMapper:
    """
    【V3 - 最终架构版】一个纯粹的实体身份管理器。
    职责：
    1. 从关系数据中聚合所有唯一的实体ID。
    2. 接收外部提供的结构信息并存储。
    3. 在所有数据准备好后，最终化并为所有实体分配和管理连续的逻辑ID。
    """

    def __init__(self, interaction_dfs: List[pd.DataFrame], config: AppConfig):
        """
        【职责简化】构造函数现在只负责从交互数据中【收集】所有唯一的实体ID。
        """
        print("--- [IDMapper] Initializing by collecting unique entity IDs...")
        self._config = config
        schema = self._config.data_structure.schema.internal.authoritative_dti

        # --- 1. 收集所有唯一的权威ID ---
        all_cids: Set[int] = set()
        all_pids: Set[str] = set()

        for df in interaction_dfs:
            all_cids.update(
                pd.to_numeric(df[schema.molecule_id], errors="coerce")
                .dropna()
                .astype(int)
                .unique()
            )
            all_pids.update(df[schema.protein_id].dropna().unique())

        self._all_cids = all_cids
        self._all_pids = all_pids

        # --- 2. 初始化空的结构信息字典 ---
        # 结构信息将由外部的 StructureProvider 填充
        self._cid_to_smiles: Dict[int, str] = {cid: None for cid in self._all_cids}
        self._uniprot_to_sequence: Dict[str, str] = {
            pid: None for pid in self._all_pids
        }

        # --- 3. 延迟最终的映射构建 ---
        # 最终的逻辑ID映射将在结构数据被注入和净化后，由一个专门的方法完成。
        self.is_finalized = False
        self.drug_to_id: Dict[int, int] = {}
        self.ligand_to_id: Dict[int, int] = {}
        self.protein_to_id: Dict[str, int] = {}
        self._drug_cids_source: Set[int] = set()  # 用一个新的属性来接收drug身份信息

        print(
            f"--> Collected {len(self._all_cids)} unique CIDs and {len(self._all_pids)} unique PIDs."
        )

    def set_drug_cids(self, drug_cids: Set[int]):
        """
        【新增方法】由外部调用者注入“drug”的身份定义。
        """
        self._drug_cids_source = drug_cids
        if self._config.runtime.verbose > 0:
            print(
                f"--> [IDMapper] Received definition for {len(drug_cids)} source drug CIDs."
            )

    # --- 数据注入与状态更新接口 ---
    def update_sequences(self, sequence_map: Dict[str, str]):
        """用外部获取的序列数据更新内部字典。"""
        self._uniprot_to_sequence.update(sequence_map)
        print(
            f"--> [IDMapper] Updated,now contains {len(sequence_map)} protein sequences."
        )

    def update_smiles(self, smiles_map: Dict[int, str]):
        """用外部获取的SMILES数据更新内部字典。"""
        self._cid_to_smiles.update(smiles_map)
        print(
            f"--> [IDMapper] Updated, now contains {len(smiles_map)} molecule SMILES."
        )

    def update_from_dataframe(self, df: pd.DataFrame):
        """
        使用一个净化后的DataFrame来更新内部的结构字典。
        这个方法在全局净化后被调用，在 finalization 之前。
        """
        print(
            "\n--- [IDMapper] Updating internal structures from purified DataFrame... ---"
        )
        schema = self._config.data_structure.schema.internal.authoritative_dti

        purified_mols = df.dropna(subset=[schema.molecule_id])
        self._cid_to_smiles = pd.Series(
            purified_mols[schema.molecule_sequence].values,
            index=purified_mols[schema.molecule_id].astype(int),
        ).to_dict()

        purified_prots = df.dropna(subset=[schema.protein_id])
        self._uniprot_to_sequence = pd.Series(
            purified_prots[schema.protein_sequence].values,
            index=purified_prots[schema.protein_id],
        ).to_dict()

        # 清理掉不存在的ID
        self._all_cids = set(self._cid_to_smiles.keys())
        self._all_pids = set(self._uniprot_to_sequence.keys())

        print(
            f"--> Internal structures updated. Total CIDs: {len(self._all_cids)}, Total PIDs: {len(self._all_pids)}."
        )

    def finalize_mappings(self):
        """
        在所有结构数据被注入和净化后，执行最终的ID分配和别名,反向映射构建。
        """
        if self.is_finalized:
            print("--> [IDMapper] Mappings already finalized. Skipping.")
            return

        print("\n--- [IDMapper] Finalizing all ID mappings... ---")

        # 1. 根据最终纯净的实体列表，重新确定drug/ligand身份
        purified_cids = set(self._cid_to_smiles.keys())
        self._drug_cids = self._drug_cids_source & purified_cids
        self._ligand_cids = purified_cids - self._drug_cids

        self._sorted_drugs = sorted(list(self._drug_cids))
        self._sorted_ligands = sorted(list(self._ligand_cids))
        self._sorted_proteins = sorted(list(self._uniprot_to_sequence.keys()))

        # 2. 分配逻辑ID
        current_id = 0
        self.drug_to_id = {
            cid: i + current_id for i, cid in enumerate(self._sorted_drugs)
        }
        current_id += len(self._sorted_drugs)
        self.ligand_to_id = {
            cid: i + current_id for i, cid in enumerate(self._sorted_ligands)
        }
        current_id += len(self._sorted_ligands)
        self.protein_to_id = {
            pid: i + current_id for i, pid in enumerate(self._sorted_proteins)
        }

        # 3. 创建便捷的别名和反向映射,为之后引入其他外部ID(除了CID,PID)体系做准备
        self.molecule_to_id = {**self.drug_to_id, **self.ligand_to_id}
        self.cid_to_id = self.molecule_to_id
        self.uniprot_to_id = self.protein_to_id
        self.id_to_drug = {i: cid for cid, i in self.drug_to_id.items()}
        self.id_to_ligand = {i: cid for cid, i in self.ligand_to_id.items()}
        self.id_to_protein = {i: pid for pid, i in self.protein_to_id.items()}
        self.id_to_cid = {**self.id_to_drug, **self.id_to_ligand}
        self.id_to_uniprot = self.id_to_protein

        self.is_finalized = True
        print("--- [IDMapper] Finalization complete. Final counts: ---")
        print(f"  - Found {self.num_drugs} unique drugs.")
        print(f"  - Found {self.num_ligands} unique pure ligands.")
        print(f"  - Found {self.num_proteins} unique proteins.")
        print(f"  - Total entities: {self.num_total_entities}")

    # --- 公共 Getter 和属性 ---

    def get_all_pids(self) -> List[str]:
        return list(self._all_pids)

    def get_all_cids(self) -> List[int]:
        return list(self._all_cids)

    def get_ordered_cids(self) -> List[int]:
        if not self.is_finalized:
            raise RuntimeError(
                "Mappings must be finalized before getting ordered lists."
            )
        return self._sorted_drugs + self._sorted_ligands

    def get_ordered_pids(self) -> List[str]:
        if not self.is_finalized:
            raise RuntimeError(
                "Mappings must be finalized before getting ordered lists."
            )
        return self._sorted_proteins

    @property
    def num_drugs(self) -> int:
        return len(self._drug_cids) if hasattr(self, "_drug_cids") else 0

    @property
    def num_ligands(self) -> int:
        return len(self._ligand_cids) if hasattr(self, "_ligand_cids") else 0

    @property
    def num_molecules(self) -> int:
        return self.num_drugs + self.num_ligands

    @property
    def num_proteins(self) -> int:
        return len(self.protein_to_id)

    @property
    def num_total_entities(self) -> int:
        return self.num_molecules + self.num_proteins

    def get_ordered_smiles(self) -> List[str]:
        if not self.is_finalized:
            raise RuntimeError(
                "Mappings must be finalized before getting ordered lists."
            )
        ordered_cids = self._sorted_drugs + self._sorted_ligands
        return [self._cid_to_smiles.get(cid, "") for cid in ordered_cids]

    def get_ordered_sequences(self) -> List[str]:
        if not self.is_finalized:
            raise RuntimeError(
                "Mappings must be finalized before getting ordered lists."
            )
        return [self._uniprot_to_sequence.get(pid, "") for pid in self._sorted_proteins]

    def get_node_type(self, logic_id: int) -> str:
        if not self.is_finalized:
            raise RuntimeError("Mappings must be finalized before getting node type.")
        if logic_id < self.num_drugs:
            return "drug"
        elif logic_id < self.num_molecules:
            return "ligand"
        else:
            return "protein"

    # TODO: 性能瓶颈
    def to_dataframe(self) -> pd.DataFrame:
        print("--- [IDMapper] Exporting internal state to DataFrame... ---")
        schema = self._config.data_structure.schema.internal.authoritative_dti

        all_entities = []
        for cid, smiles in self._cid_to_smiles.items():
            all_entities.append(
                {
                    schema.molecule_id: cid,
                    schema.molecule_sequence: smiles,
                    schema.protein_id: None,
                    schema.protein_sequence: None,
                }
            )
        for pid, sequence in self._uniprot_to_sequence.items():
            all_entities.append(
                {
                    schema.molecule_id: None,
                    schema.molecule_sequence: None,
                    schema.protein_id: pid,
                    schema.protein_sequence: sequence,
                }
            )

        if not all_entities:
            return pd.DataFrame()
        df = pd.DataFrame(all_entities)
        print(f"--> Exported {len(df)} total unique entities (molecules and proteins).")
        return df

    def save_nodes_metadata(self):
        """
        【V2 - Schema驱动版】将IDMapper内部的所有核心映射关系保存到磁盘 (nodes.csv)。
        """
        if not self.is_finalized:
            print("--> [IDMapper] Warning: Mappings not finalized. Stop saving.")
            return

        print("\n--- [IDMapper] Saving final nodes metadata ('nodes.csv')... ---")

        # --- 1. 从配置中获取 nodes.csv 的 schema ---
        nodes_schema = self._config.data_structure.schema.internal.nodes_output

        node_data = []

        # --- 2. 使用 schema 中的列名来构建数据 ---
        for cid, logic_id in self.cid_to_id.items():
            node_type = "drug" if cid in self._drug_cids else "ligand"
            node_data.append(
                {
                    nodes_schema.global_id: logic_id,
                    nodes_schema.node_type: node_type,
                    nodes_schema.authoritative_id: cid,
                    nodes_schema.structure: self._cid_to_smiles.get(cid),
                }
            )

        for pid, logic_id in self.protein_to_id.items():
            node_data.append(
                {
                    nodes_schema.global_id: logic_id,
                    nodes_schema.node_type: "protein",
                    nodes_schema.authoritative_id: pid,
                    nodes_schema.structure: self._uniprot_to_sequence.get(pid),
                }
            )

        if not node_data:
            print("--> No data to save for debugging.")
            return

        nodes_df = (
            pd.DataFrame(node_data)
            .sort_values(by=nodes_schema.global_id)
            .reset_index(drop=True)
        )

        # --- 3. 保存文件 ---
        output_path = get_path(self._config, "processed.common.nodes_metadata")
        rt.ensure_path_exists(output_path)
        nodes_df.to_csv(output_path, index=False)
        print(f"--> Core metadata file 'nodes.csv' saved to: {output_path}")
