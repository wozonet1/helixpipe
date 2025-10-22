# 文件: src/nasnet/data_processing/services/interaction_store.py (全新)

from typing import List, Set, Tuple

import pandas as pd
from tqdm import tqdm

from nasnet.configs import AppConfig

from .id_mapper import IDMapper


class InteractionStore:
    """
    一个负责存储、管理和过滤所有交互关系（边）的类。
    """

    def __init__(self, interaction_dfs: List[pd.DataFrame], config: AppConfig):
        self._config = config
        self._schema = config.data_structure.schema.internal.authoritative_dti

        print(
            f"--- [InteractionStore] Initializing with {len(interaction_dfs)} interaction DataFrame(s)..."
        )

        # 合并所有输入的交互数据为一个内部DataFrame
        self._interactions_df = pd.concat(interaction_dfs, ignore_index=True)

        print(
            f"--> Stored a total of {len(self._interactions_df)} raw interaction records."
        )

    def filter_by_entities(self, valid_cids: Set[int], valid_pids: Set[str]):
        """
        根据一组有效的实体ID，过滤内部的交互列表。
        这是一个in-place操作，会修改自身的状态。
        """
        initial_count = len(self._interactions_df)
        print(
            f"--- [InteractionStore] Filtering {initial_count} interactions against purified entity list..."
        )

        self._interactions_df.dropna(
            subset=[self._schema.molecule_id, self._schema.protein_id], inplace=True
        )

        self._interactions_df = self._interactions_df[
            self._interactions_df[self._schema.molecule_id].isin(valid_cids)
            & self._interactions_df[self._schema.protein_id].isin(valid_pids)
        ]

        final_count = len(self._interactions_df)
        print(f"--> Filtering complete. {final_count} interactions remain.")

    def get_all_interactions_df(self) -> pd.DataFrame:
        """返回当前状态下的交互DataFrame。"""
        return self._interactions_df.copy()

    def get_mapped_positive_pairs(
        self, id_mapper: IDMapper
    ) -> Tuple[List[Tuple[int, int]], Set[Tuple[int, int]]]:
        """
        使用一个【已最终化】的IDMapper，将存储的正样本交互转换为逻辑ID对。
        """
        if not id_mapper.is_finalized:
            raise RuntimeError("IDMapper must be finalized before mapping pairs.")

        # 只处理正样本
        positive_df = self._interactions_df[
            self._interactions_df[self._schema.label] == 1
        ].copy()
        positive_df.drop_duplicates(
            subset=[self._schema.molecule_id, self._schema.protein_id], inplace=True
        )

        positive_pairs = []
        # tqdm描述可以更具体
        iterator = tqdm(
            positive_df.itertuples(),
            total=len(positive_df),
            desc="[InteractionStore] Mapping pairs",
        )

        for row in iterator:
            cid = getattr(row, self._schema.molecule_id)
            pid = getattr(row, self._schema.protein_id)

            mol_logic_id = id_mapper.cid_to_id.get(cid)
            prot_logic_id = id_mapper.uniprot_to_id.get(pid)

            if mol_logic_id is not None and prot_logic_id is not None:
                positive_pairs.append((mol_logic_id, prot_logic_id))

        positive_pairs_set = set(positive_pairs)
        print(f"--> Found {len(positive_pairs_set)} unique valid positive pairs.")
        return positive_pairs, positive_pairs_set
