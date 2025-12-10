# 文件: scripts/debug_stage7.py

import logging
from typing import cast

import hydra
import numpy as np
import pandas as pd
import torch

from helixpipe.configs import register_all_schemas
from helixpipe.data_processing import IDMapper, InteractionStore
from helixpipe.pipelines.main_pipeline import _stage7_split_and_build_graphs
from helixpipe.typing import AppConfig, AuthID
from helixpipe.utils import get_path, register_hydra_resolvers, setup_logging

logger = logging.getLogger(__name__)


def rehydrate_id_mapper(config: AppConfig, nodes_df: pd.DataFrame) -> IDMapper:
    """
    【核心黑魔法】从 nodes.csv '复活' 一个已 Finalized 的 IDMapper 对象。
    这避免了重新运行复杂的聚合逻辑。
    """
    logger.info("--> Rehydrating IDMapper from nodes.csv...")

    # 1. 创建一个空的 Mapper (传入空字典绕过初始化逻辑)
    mapper = IDMapper({}, config)

    # 2. 手术式注入内部状态
    # nodes.csv 列: global_id, node_type, authoritative_id, sources

    # 注入 _logic_id_to_auth_id_map
    mapper._logic_id_to_auth_id_map = pd.Series(
        nodes_df.authoritative_id.values, index=nodes_df.global_id
    ).to_dict()

    # 注入 _auth_id_to_logic_id_map
    mapper._auth_id_to_logic_id_map = pd.Series(
        nodes_df.global_id.values, index=nodes_df.authoritative_id
    ).to_dict()

    # 注入 _logic_id_to_type_map
    mapper._logic_id_to_type_map = pd.Series(
        nodes_df.node_type.values, index=nodes_df.global_id
    ).to_dict()

    # 注入 _final_entity_map (包含 sources)
    mapper._final_entity_map = {}
    for row in nodes_df.itertuples():
        sources_set = (
            set(str(row.sources).split(",")) if pd.notna(row.sources) else set()
        )
        safe_auth_id = cast(AuthID, row.authoritative_id)

        mapper._final_entity_map[safe_auth_id] = {
            "type": row.node_type,
            "sources": sources_set,
        }

    # 重建 entities_by_type 索引
    for row in nodes_df.itertuples():
        safe_auth_id = cast(AuthID, row.authoritative_id)
        mapper.entities_by_type[cast(str, row.node_type)].append(safe_auth_id)

    # 标记为已完成
    mapper.is_finalized = True

    logger.info(f"    - IDMapper restored. Total entities: {mapper.num_total_entities}")
    return mapper


@hydra.main(config_path="../../../../conf", config_name="config", version_base=None)
def fast_debug_stage7(cfg: AppConfig) -> None:
    setup_logging(cfg)
    logger.info("=== STARTING FAST DEBUG: STAGE 7 ONLY ===")

    # 1. 检查必要文件是否存在
    nodes_path = get_path(cfg, "processed.common.nodes_metadata")
    features_path = get_path(cfg, "processed.common.node_features")
    dti_path = get_path(cfg, "raw.authoritative_dti")

    if not (nodes_path.exists() and features_path.exists() and dti_path.exists()):
        logger.error(
            "❌ Missing intermediate files! You must run the full pipeline ONCE to generate caches."
        )
        return

    # 2. 加载数据
    logger.info("--> Loading intermediate files...")
    nodes_df = pd.read_csv(nodes_path)
    # 处理 authoritative_id 的类型，防止 int 被读成 float/str
    # 这是一个常见坑，我们尝试根据 node_type 智能转换
    # 简单起见，这里假设它可能是混合类型，IDMapper 重建时会处理

    # 3. 复活 IDMapper
    id_mapper = rehydrate_id_mapper(cfg, nodes_df)

    # 4. 复活 InteractionStore (模拟 Stage 5 的过滤)
    logger.info("--> Rehydrating InteractionStore...")
    raw_dti_df = pd.read_csv(dti_path)
    store = InteractionStore._from_dataframe(raw_dti_df, cfg)

    # 关键：必须执行和 Main Pipeline 一样的过滤步骤
    valid_ids = set(id_mapper.get_all_final_ids())
    pure_store = store.filter_by_entities(valid_ids)

    # 5. 复活特征 (Tensor 切片)
    logger.info("--> Loading Embeddings...")
    features_np = np.load(features_path)
    features_tensor = torch.from_numpy(features_np)

    num_mols = id_mapper.num_molecules
    mol_emb = features_tensor[:num_mols]
    prot_emb = features_tensor[num_mols:]

    logger.info(f"    - Mol Emb: {mol_emb.shape}, Prot Emb: {prot_emb.shape}")

    # 6. 【核心】直接调用 Stage 7 函数
    logger.info("\n>>> JUMPING DIRECTLY TO STAGE 7 >>>")

    _stage7_split_and_build_graphs(
        config=cfg,
        id_mapper=id_mapper,
        pure_interaction_store=pure_store,
        molecule_embeddings=mol_emb,
        protein_embeddings=prot_emb,
    )

    logger.info("=== FAST DEBUG COMPLETE ===")


if __name__ == "__main__":
    register_all_schemas()
    register_hydra_resolvers()
    fast_debug_stage7()
