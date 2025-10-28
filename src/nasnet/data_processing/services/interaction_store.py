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
        # [NEW DEFENSIVE CODE]
        rel_type_col = self._schema.relation_type
        default_rel_type = self._config.relations.names.default

        # 检查 'relation_type' 列是否存在，如果不存在则创建
        if rel_type_col not in self._interactions_df.columns:
            self._interactions_df[rel_type_col] = default_rel_type
            print(
                f"    - WARNING: '{rel_type_col}' column was missing. Filled with default value: '{default_rel_type}'."
            )
        else:
            # 如果列存在，检查其中是否有NaN值
            nan_count = self._interactions_df[rel_type_col].isna().sum()
            if nan_count > 0:
                self._interactions_df[rel_type_col] = self._interactions_df[
                    rel_type_col
                ].fillna(default_rel_type)
                print(
                    f"    - WARNING: Found and filled {nan_count} NaN values in '{rel_type_col}' column with default value."
                )

        # [NEW] 强制转换为字符串类型，彻底杜绝类型问题
        self._interactions_df[rel_type_col] = self._interactions_df[
            rel_type_col
        ].astype(str)

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
    ) -> Tuple[List[Tuple[int, int, str]], Set[Tuple[int, int]]]:
        """
        【升级版】使用一个【已最终化】的IDMapper，将存储的正样本交互
        转换为包含【最终图边类型】的逻辑ID对。

        Returns:
            Tuple[List[Tuple[int, int, str]], Set[Tuple[int, int]]]:
            - 一个元组列表，每个元组是 (分子逻辑ID, 蛋白逻辑ID, 最终关系类型字符串)。
            - 一个元组集合，用于快速进行存在性检查 (不包含关系类型)。
        """
        if not id_mapper.is_finalized:
            raise RuntimeError("IDMapper must be finalized before mapping pairs.")

        # 从 internal_schema 获取标准列名
        mol_id_col = self._schema.molecule_id
        prot_id_col = self._schema.protein_id
        label_col = self._schema.label
        rel_type_col = self._schema.relation_type

        # 1. 筛选正样本并去重
        positive_df = self._interactions_df[
            self._interactions_df[label_col] == 1
        ].copy()
        positive_df.drop_duplicates(subset=[mol_id_col, prot_id_col], inplace=True)

        # 2. 遍历并执行映射
        positive_pairs_with_type: List[Tuple[int, int, str]] = []

        # 准备一个默认值，以防 'relation_type' 列意外丢失
        default_rel_type = self._config.relations.names.default

        iterator = tqdm(
            positive_df.itertuples(index=False),  # index=False 效率更高
            total=len(positive_df),
            desc="[InteractionStore] Mapping pairs with relation types",
        )

        for row in iterator:
            cid = getattr(row, mol_id_col)
            pid = getattr(row, prot_id_col)
            # 【核心】安全地获取 relation_type，如果不存在则使用默认值
            rel_type = getattr(row, rel_type_col, default_rel_type)

            mol_logic_id = id_mapper.cid_to_id.get(cid)
            prot_logic_id = id_mapper.uniprot_to_id.get(pid)

            if mol_logic_id is not None and prot_logic_id is not None:
                positive_pairs_with_type.append((mol_logic_id, prot_logic_id, rel_type))

        # 3. 生成一个不带类型的集合，用于下游的快速查找 (例如负采样)
        positive_pairs_set = {(u, v) for u, v, _ in positive_pairs_with_type}

        if self._config.runtime.verbose > 0:
            print(
                f"--> [InteractionStore] Found {len(positive_pairs_set)} unique valid positive pairs with types."
            )

        return positive_pairs_with_type, positive_pairs_set
