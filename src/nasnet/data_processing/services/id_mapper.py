from typing import Dict, List, Set, Tuple

import pandas as pd
import research_template as rt
from tqdm import tqdm

from nasnet.typing import AppConfig
from nasnet.utils import get_path


class IDMapper:
    """
    一个在运行时动态创建的、负责管理所有ID映射关系的对象。
    它将权威ID (PubChem CID, UniProt ID) 映射到模型友好的连续逻辑ID (0, 1, 2, ...)。
    """

    def __init__(
        self, base_df: pd.DataFrame, extra_dfs: List[pd.DataFrame], config: AppConfig
    ):
        """
        通过一个基础DataFrame和多个扩展DataFrame来初始化ID映射器。
        【V2 优化版】
        """
        print("--- [IDMapper] Initializing with entity type distinction...")
        self._config = config
        schema = self._config.data_structure.schema.internal.authoritative_dti

        # --- 0. 初始化所有实例属性 ---
        # 将所有属性的定义都放在最前面，是一个好的编程习惯
        self._cid_to_smiles: Dict[int, str] = {}
        self._uniprot_to_sequence: Dict[str, str] = {}

        # --- 1. 收集并区分 Drug 和 Ligand 的权威ID (CIDs) ---

        base_cids = set(base_df[schema.molecule_id].dropna().unique())

        extra_cids = set()
        for df in extra_dfs:
            extra_cids.update(df[schema.molecule_id].dropna().unique())

        self._drug_cids: Set[int] = base_cids
        self._ligand_cids: Set[int] = extra_cids - base_cids

        # --- 2. 合并数据并收集实体与序列映射 ---
        all_dfs = [base_df] + (extra_dfs if extra_dfs else [])
        merged_df = pd.concat(all_dfs, ignore_index=True)

        # a. 收集所有唯一的蛋白质ID
        all_uniprot_ids: Set[str] = set(merged_df[schema.protein_id].dropna().unique())

        # b. 【优化】一次性构建 CID -> SMILES 映射
        mol_map_df = (
            merged_df[[schema.molecule_id, schema.molecule_sequence]]
            .dropna()
            .drop_duplicates(subset=[schema.molecule_id])
        )
        self._cid_to_smiles = pd.Series(
            mol_map_df[schema.molecule_sequence].values,
            index=mol_map_df[schema.molecule_id],
        ).to_dict()

        # c. 【优化】一次性构建 UniProt ID -> Sequence 映射
        prot_map_df = (
            merged_df[[schema.protein_id, schema.protein_sequence]]
            .dropna()
            .drop_duplicates(subset=[schema.protein_id])
        )
        self._uniprot_to_sequence = pd.Series(
            prot_map_df[schema.protein_sequence].values,
            index=prot_map_df[schema.protein_id],
        ).to_dict()

        # --- 3. 创建从权威ID到逻辑ID的有序映射 ---
        self._sorted_drugs = sorted(list(self._drug_cids))
        self._sorted_ligands = sorted(list(self._ligand_cids))
        self._sorted_proteins = sorted(list(all_uniprot_ids))

        # 逻辑ID分配: [drugs, ligands, proteins]
        current_id = 0
        self.drug_to_id: Dict[int, int] = {
            cid: i + current_id for i, cid in enumerate(self._sorted_drugs)
        }
        current_id += len(self._sorted_drugs)

        self.ligand_to_id: Dict[int, int] = {
            cid: i + current_id for i, cid in enumerate(self._sorted_ligands)
        }
        current_id += len(self._sorted_ligands)

        self.protein_to_id: Dict[str, int] = {
            pid: i + current_id for i, pid in enumerate(self._sorted_proteins)
        }

        # --- 4. 创建统一和反向的映射 (别名) ---
        self.molecule_to_id = {**self.drug_to_id, **self.ligand_to_id}
        self.cid_to_id = self.molecule_to_id
        self.uniprot_to_id = self.protein_to_id

        self.id_to_drug: Dict[int, int] = {i: cid for cid, i in self.drug_to_id.items()}
        self.id_to_ligand: Dict[int, int] = {
            i: cid for cid, i in self.ligand_to_id.items()
        }
        self.id_to_protein: Dict[int, str] = {
            i: pid for pid, i in self.protein_to_id.items()
        }

        self.id_to_cid = {**self.id_to_drug, **self.id_to_ligand}
        self.id_to_uniprot = self.id_to_protein

        # --- 5. 打印总结信息 ---
        print("--- [IDMapper] Initialization complete. ---")
        print(f"  - Found {self.num_drugs} unique drugs.")
        print(f"  - Found {self.num_ligands} unique pure ligands.")
        print(f"  - Found {self.num_proteins} unique proteins.")
        print(f"  - Total entities: {self.num_total_entities}")

    # --- 6. 属性和辅助方法 ---
    @property
    def num_drugs(self) -> int:
        return len(self._drug_cids)

    @property
    def num_ligands(self) -> int:
        return len(self._ligand_cids)

    @property
    def num_molecules(self) -> int:
        return self.num_drugs + self.num_ligands

    @property
    def num_proteins(self) -> int:
        return len(self._sorted_proteins)

    @property
    def num_total_entities(self) -> int:
        return self.num_molecules + self.num_proteins

    @property
    def sorted_drug_cids(self) -> List[int]:
        return self._sorted_drugs

    @property
    def sorted_ligand_cids(self) -> List[int]:
        return self._sorted_ligands

    @property
    def sorted_protein_ids(self) -> List[str]:
        return self._sorted_proteins

    def get_ordered_smiles(self) -> List[str]:
        """按 [drugs, ligands] 的逻辑ID顺序返回SMILES列表。"""
        ordered_cids = self._sorted_drugs + self._sorted_ligands
        return [self._cid_to_smiles[cid] for cid in ordered_cids]

    def get_ordered_sequences(self) -> List[str]:
        """按逻辑ID的顺序返回蛋白质序列列表。"""
        return [self._uniprot_to_sequence[pid] for pid in self._sorted_proteins]

    def get_node_type(self, logic_id: int) -> str:
        """根据给定的逻辑ID，返回其节点类型 ('drug', 'ligand', 'protein')。"""
        if logic_id < self.num_drugs:
            return "drug"
        elif logic_id < self.num_molecules:
            return "ligand"
        else:
            return "protein"

    def save_maps_for_debugging(self, config: AppConfig):
        """
        将IDMapper内部的所有核心映射关系保存到磁盘，
        以便于人类阅读、调试和后续分析。
        这将生成我们熟悉的 nodes.csv 文件。
        """
        print("\n--- [IDMapper] Saving internal maps for debugging...")

        # 1. 构造一个包含所有节点信息的列表
        node_data = []

        # 处理分子
        for cid, logic_id in self.cid_to_id.items():
            node_data.append(
                {
                    "global_id": logic_id,
                    "node_type": "molecule",  # 简化类型名
                    "authoritative_id": cid,
                    "sequence_or_smiles": self._cid_to_smiles.get(cid),
                }
            )

        # 处理蛋白质
        for pid, logic_id in self.uniprot_to_id.items():
            node_data.append(
                {
                    "global_id": logic_id,
                    "node_type": "protein",
                    "authoritative_id": pid,
                    "sequence_or_smiles": self._uniprot_to_sequence.get(pid),
                }
            )

        # 2. 转换为DataFrame并排序
        nodes_df = pd.DataFrame(node_data)
        nodes_df.sort_values(by="global_id", inplace=True)

        # 3. 保存
        # 这个路径现在也由配置驱动
        output_path = get_path(config, "processed.common.nodes_metadata")
        rt.ensure_path_exists(output_path)
        nodes_df.to_csv(output_path, index=False)

        print(f"--> Debug file 'nodes.csv' saved to: {output_path}")

    def map_pairs(
        self, dataframes: List[pd.DataFrame]
    ) -> Tuple[List[Tuple[int, int]], Set[Tuple[int, int]]]:
        """
        【新增】接收一个或多个标准的DataFrame，提取所有正样本交互，
        并将其转换为逻辑ID对。

        Args:
            dataframes (List[pd.DataFrame]): 包含交互数据的DataFrame列表。

        Returns:
            Tuple[List, Set]:
                - 一个包含所有正样本逻辑ID对的列表。
                - 一个包含所有正样本逻辑ID对的集合，用于快速查找。
        """
        print("--- [IDMapper] Mapping interaction pairs to logic IDs...")
        schema = self._config.data_structure.schema.internal.authoritative_dti

        if not dataframes:
            return [], set()

        merged_df = pd.concat(dataframes, ignore_index=True)
        positive_df = merged_df[merged_df[schema.label] == 1].copy()
        positive_df.drop_duplicates(
            subset=[schema.molecule_id, schema.protein_id], inplace=True
        )

        positive_pairs = []
        for row in tqdm(
            positive_df.itertuples(),
            total=len(positive_df),
            desc="[IDMapper] Converting pairs",
        ):
            cid = getattr(row, schema.molecule_id)
            pid = getattr(row, schema.protein_id)

            mol_logic_id = self.cid_to_id.get(cid)
            prot_logic_id = self.uniprot_to_id.get(pid)

            if mol_logic_id is not None and prot_logic_id is not None:
                positive_pairs.append((mol_logic_id, prot_logic_id))

        positive_pairs_set = set(positive_pairs)
        print(f"--> Found {len(positive_pairs_set)} unique positive pairs.")

        return positive_pairs, positive_pairs_set
