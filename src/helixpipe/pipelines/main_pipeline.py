from typing import Dict, Tuple

import numpy as np
import pandas as pd
import research_template as rt
import torch

# (确保在文件顶部有以下 imports)
from helixpipe.configs import AppConfig
from helixpipe.data_processing import (
    DataSplitter,
    GraphBuildContext,
    GraphDirector,
    IDMapper,
    StructureProvider,
    SupervisionFileManager,
    sample_interactions,
    validate_and_filter_entities,
)
from helixpipe.data_processing.services.graph_builder import HeteroGraphBuilder
from helixpipe.features import extract_features
from helixpipe.utils import get_path


def process_data(config: AppConfig, processor_outputs: Dict[str, pd.DataFrame]):
    """
    【V5 - 最终完整编排版】数据处理的总编排器。
    """
    if not processor_outputs:
        raise ValueError(
            "Cannot process data: Input processor_outputs dictionary is empty."
        )

    # === 实例化共享服务 ===
    provider = StructureProvider(config, proxies=None)

    # === STAGE 1: 数据聚合 (Aggregation) ===
    id_mapper = IDMapper(processor_outputs=processor_outputs, config=config)

    # === STAGE 2: 实体清单构建 (Entity Manifest Construction) ===
    # 【调用新版函数】
    raw_entities_df = id_mapper.build_entity_manifest()
    if raw_entities_df.empty:
        print("\n⚠️  No entities found after aggregation. Halting pipeline.")
        return

    # === STAGE 3: 实体丰富化 ===
    enriched_entities_df = _stage3_enrich_entities(raw_entities_df, config, provider)

    # === STAGE 4: 中心化校验 ===
    valid_entities_df = _stage4_validate_entities(enriched_entities_df, config)
    if valid_entities_df.empty:
        print("\n⚠️  No entities passed validation. Halting pipeline.")
        return

    # === STAGE 5: 最终交互过滤 ===
    pure_interactions_df = _stage5_finalize_and_filter(
        id_mapper, valid_entities_df, processor_outputs, config
    )
    if pure_interactions_df.empty:
        print("\n⚠️  No interactions remained after filtering. Halting pipeline.")
        return

    molecule_embeddings, protein_embeddings = _stage6_generate_features(
        config=config,
        id_mapper=id_mapper,
        entities_with_structures=valid_entities_df,
        restart_flag=config.runtime.force_restart,
    )

    # === STAGE 8: 最终产物生成 (图 & 标签) ===
    # 启动包含采样、划分、图构建的最终下游流水线。
    _stage7_split_and_build_graphs(
        config=config,
        id_mapper=id_mapper,
        pure_interactions_df=pure_interactions_df,
        molecule_embeddings=molecule_embeddings,
        protein_embeddings=protein_embeddings,
    )

    print("\n✅ All data processing stages (up to dispatching) completed successfully!")


def _stage2_build_entity_manifest(
    all_interactions_df: pd.DataFrame, config: AppConfig
) -> pd.DataFrame:
    """
    【新 - Stage 2】从所有交互记录中，构建一个纯粹的、无结构的“实体清单”。

    这个函数执行一个简单的“提取”和“去重”操作。
    """
    if config.runtime.verbose > 0:
        print(
            "\n--- [Pipeline Stage 2] Building a unified (structureless) entity manifest... ---"
        )

    if all_interactions_df.empty:
        if config.runtime.verbose > 0:
            print(
                "  - Input interactions are empty. Returning an empty entity manifest."
            )
        return pd.DataFrame()

    schema = config.data_structure.schema.internal.canonical_interaction

    # 1. 从 source 列提取 (id, type) 对
    source_entities = all_interactions_df[
        [schema.source_id, schema.source_type]
    ].rename(columns={schema.source_id: "entity_id", schema.source_type: "entity_type"})

    # 2. 从 target 列提取 (id, type) 对
    target_entities = all_interactions_df[
        [schema.target_id, schema.target_type]
    ].rename(columns={schema.target_id: "entity_id", schema.target_type: "entity_type"})

    # 3. 合并并去除完全重复的行
    unique_entities_df = (
        pd.concat([source_entities, target_entities])
        .drop_duplicates()
        .reset_index(drop=True)
    )

    if config.runtime.verbose > 0:
        print(
            f"  - Found {len(unique_entities_df)} unique (id, type) pairs across all data sources."
        )

    return unique_entities_df


def _stage3_enrich_entities(
    raw_entities_df: pd.DataFrame, config: AppConfig, provider: StructureProvider
) -> pd.DataFrame:
    """
    【新】Stage 3: 对缺少结构信息的实体，统一调用 StructureProvider 进行在线丰富化。
    """
    if config.runtime.verbose > 0:
        print(
            "\n--- [Pipeline Stage 3] Enriching entities with missing structures... ---"
        )

    enriched_df = raw_entities_df.copy()
    entities_to_enrich = enriched_df[enriched_df["structure"].isna()]

    if entities_to_enrich.empty:
        if config.runtime.verbose > 0:
            print("  - No missing structures to enrich. Skipping.")
        return enriched_df

    if config.runtime.verbose > 0:
        print(
            f"  - Found {len(entities_to_enrich)} entities requiring structure enrichment."
        )

    entity_types = config.knowledge_graph.entity_types
    restart_flag = config.runtime.force_restart

    # --- 丰富化分子 (SMILES) ---
    molecules_to_enrich = entities_to_enrich[
        entities_to_enrich["entity_type"] == entity_types.molecule
    ]
    if not molecules_to_enrich.empty:
        cids_to_fetch = molecules_to_enrich["entity_id"].tolist()
        smiles_map = provider.get_smiles(cids_to_fetch, force_restart=restart_flag)
        if smiles_map:
            enriched_df["structure"].fillna(
                enriched_df["entity_id"].map(smiles_map), inplace=True
            )

    # --- 丰富化蛋白质 (Sequences) ---
    proteins_to_enrich = entities_to_enrich[
        entities_to_enrich["entity_type"] == entity_types.protein
    ]
    if not proteins_to_enrich.empty:
        pids_to_fetch = proteins_to_enrich["entity_id"].tolist()
        sequence_map = provider.get_sequences(pids_to_fetch, force_restart=restart_flag)
        if sequence_map:
            enriched_df["structure"].fillna(
                enriched_df["entity_id"].map(sequence_map), inplace=True
            )

    if config.runtime.verbose > 0:
        final_with_structure = enriched_df["structure"].notna().sum()
        print(
            f"  - Enrichment complete. {final_with_structure} entities now have structures."
        )

    return enriched_df


def _stage4_validate_entities(
    enriched_entities_df: pd.DataFrame, config: AppConfig
) -> pd.DataFrame:
    """
    【新】Stage 4: 将完整的实体清单送入中心化校验服务。
    """
    if config.runtime.verbose > 0:
        print(
            "\n--- [Pipeline Stage 4] Performing centralized entity validation... ---"
        )

    return validate_and_filter_entities(enriched_entities_df, config)


def _stage5_finalize_and_filter(
    id_mapper: IDMapper,
    valid_entities_df: pd.DataFrame,
    processor_outputs: Dict[str, pd.DataFrame],
    config: AppConfig,
) -> pd.DataFrame:
    """
    【新】Stage 5: 封装了IDMapper最终化和交互过滤两个紧密相关的步骤。
    """
    if config.runtime.verbose > 0:
        print(
            "\n--- [Pipeline Stage 5] Finalizing IDMapper and filtering interactions... ---"
        )

    # a. 使用纯净实体ID集合，来最终化IDMapper的状态
    valid_entity_ids = set(valid_entities_df["entity_id"])
    id_mapper.finalize_with_valid_entities(valid_entity_ids)

    # b. 保存包含所有元信息的 nodes.csv
    id_mapper.save_maps_to_csv(valid_entities_df)

    # c. 过滤原始的交互列表
    all_interactions_df = pd.concat(list(processor_outputs.values()), ignore_index=True)

    schema = config.data_structure.schema.internal.canonical_interaction

    source_is_valid = all_interactions_df[schema.source_id].isin(valid_entity_ids)
    target_is_valid = all_interactions_df[schema.target_id].isin(valid_entity_ids)

    pure_interactions_df = all_interactions_df[source_is_valid & target_is_valid]

    if config.runtime.verbose > 0:
        print(
            f"  - Filtering complete. {len(pure_interactions_df)} / {len(all_interactions_df)} interactions remain."
        )

    return pure_interactions_df.reset_index(drop=True)


def _stage6_generate_features(
    config: AppConfig,
    id_mapper: IDMapper,
    entities_with_structures: pd.DataFrame,
    restart_flag: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    【V2 - 通用亚型版】为所有纯净实体生成或加载特征嵌入。

    该函数通过动态查询IDMapper来处理任意数量的分子和蛋白质亚型。
    """
    print(
        "\n--- [Pipeline Stage 7] Generating or loading node features for all subtypes... ---"
    )

    final_features_path = get_path(config, "processed.common.node_features")

    # --- 缓存检查 (逻辑不变) ---
    if final_features_path.exists() and not restart_flag:
        print(
            f"--> [Cache Hit] Loading final aggregated features from '{final_features_path.name}'."
        )
        features_np = np.load(final_features_path)

        if len(features_np) != id_mapper.num_total_entities:
            print(
                f"    - ⚠️ WARNING: Cached feature file size ({len(features_np)}) does not match IDMapper count ({id_mapper.num_total_entities}). Regenerating."
            )
        else:
            num_molecules = id_mapper.num_molecules
            molecule_embeddings = torch.from_numpy(features_np[:num_molecules])
            protein_embeddings = torch.from_numpy(features_np[num_molecules:])
            print("--> Features loaded successfully.")
            return molecule_embeddings, protein_embeddings

    print("--> [Cache Miss/Regenerate] Starting feature calculation process...")

    # --- 1. “分类打包”工作流 ---

    # a. 准备“篮子”和“打包清单”
    all_molecule_ids = []
    all_protein_ids = []

    # b. 向IDMapper“问询”所有存在的实体类型，并进行“大类分拣”
    entity_types = id_mapper.entity_types
    for entity_type in entity_types:
        if id_mapper.is_molecule(entity_type):
            all_molecule_ids.extend(id_mapper.get_ordered_ids(entity_type))
        elif id_mapper.is_protein(entity_type):
            all_protein_ids.extend(id_mapper.get_ordered_ids(entity_type))

    # c. 根据“打包清单”提取“货物”（结构信息）
    #    使用 .set_index() 创建一个高效的查找映射
    structure_map = entities_with_structures.set_index("entity_id")["structure"]

    # 使用 .map() 或列表推导式高效、有序地提取结构
    ordered_smiles = [structure_map.get(mid) for mid in all_molecule_ids]
    ordered_sequences = [structure_map.get(pid) for pid in all_protein_ids]

    # --- 2. 调用“打包工人”（特征提取器） ---

    device = torch.device(config.runtime.gpu if torch.cuda.is_available() else "cpu")

    molecule_features_dict = {}
    if all_molecule_ids:
        molecule_features_dict = extract_features(
            entity_type="molecule",  # 使用通用大类名称
            config=config,
            device=device,
            authoritative_ids=all_molecule_ids,
            sequences_or_smiles=ordered_smiles,
            force_regenerate=restart_flag,
        )

    protein_features_dict = {}
    if all_protein_ids:
        protein_features_dict = extract_features(
            entity_type="protein",  # 使用通用大类名称
            config=config,
            device=device,
            authoritative_ids=all_protein_ids,
            sequences_or_smiles=ordered_sequences,
            force_regenerate=restart_flag,
        )

    # --- 3. 按顺序组装最终的“集装箱”（特征矩阵） ---

    ordered_mol_embeddings = [
        molecule_features_dict.get(mid) for mid in all_molecule_ids
    ]
    ordered_prot_embeddings = [
        protein_features_dict.get(pid) for pid in all_protein_ids
    ]

    # 健壮性检查
    if any(e is None for e in ordered_mol_embeddings):
        raise RuntimeError("Molecule feature extraction failed for some entities.")
    if any(e is None for e in ordered_prot_embeddings):
        raise RuntimeError("Protein feature extraction failed for some entities.")

    molecule_embeddings = (
        torch.stack(ordered_mol_embeddings)
        if ordered_mol_embeddings
        else torch.empty(0, 0)
    )
    protein_embeddings = (
        torch.stack(ordered_prot_embeddings)
        if ordered_prot_embeddings
        else torch.empty(0, 0)
    )

    # --- 4. 维度对齐与保存 (逻辑不变) ---

    if (
        molecule_embeddings.numel() > 0
        and protein_embeddings.numel() > 0
        and molecule_embeddings.shape[1] != protein_embeddings.shape[1]
    ):
        print("    - WARNING: Embedding dimensions differ. Applying linear projection.")
        proj = torch.nn.Linear(
            molecule_embeddings.shape[1], protein_embeddings.shape[1]
        ).to(device)
        molecule_embeddings = proj(molecule_embeddings.to(device)).cpu()

    # 拼接并保存
    all_feature_embeddings = (
        torch.cat([molecule_embeddings, protein_embeddings], dim=0)
        .cpu()
        .detach()
        .numpy()
    )
    rt.ensure_path_exists(final_features_path)
    np.save(final_features_path, all_feature_embeddings)
    print(
        f"--> Final aggregated features saved to: '{final_features_path.name}'. Shape: {all_feature_embeddings.shape}"
    )

    return molecule_embeddings, protein_embeddings


def _stage7_split_and_build_graphs(
    config: AppConfig,
    id_mapper: IDMapper,  # 这是一个已经最终化的、纯净的IDMapper
    pure_interactions_df: pd.DataFrame,  # 这是一个纯净的、使用【权威ID】的交互列表
    molecule_embeddings: torch.Tensor,
    protein_embeddings: torch.Tensor,
):
    """
    【V9 - 最终版】
    负责数据采样、划分、图构建和标签生成的总指挥。
    """
    print(
        "\n--- [Pipeline Stage 8] Orchestrating sampling, splitting, and graph/label generation... ---"
    )

    stage8_rng = np.random.default_rng(config.runtime.seed)
    schema = config.data_structure.schema.internal.canonical_interaction

    # --- 1. [NEW] 将权威ID交互对，转换为逻辑ID交互对 ---
    # 我们直接在这里，一次性地完成映射
    # IDMapper 现在有一个 auth_id -> logic_id 的全局映射
    source_logic_ids = pure_interactions_df[schema.source_id].map(
        id_mapper.auth_id_to_logic_id_map
    )
    target_logic_ids = pure_interactions_df[schema.target_id].map(
        id_mapper.auth_id_to_logic_id_map
    )
    if source_logic_ids.isna().any() or target_logic_ids.isna().any():
        raise ValueError(
            "Failed to map some authoritative IDs to logic IDs. Check for inconsistencies."
        )
    # 组合成 (u, v, rel_type) 的元组列表
    all_positive_pairs_with_type = list(
        zip(
            source_logic_ids,
            target_logic_ids,
            pure_interactions_df[schema.relation_type],
        )
    )

    # --- 2. 采样 (现在输入的是【逻辑ID】对) ---
    sampled_pairs_with_type, sampled_pairs_set_global = sample_interactions(
        all_positive_pairs=all_positive_pairs_with_type,
        id_mapper=id_mapper,  # Sampler 仍然需要 id_mapper 来判断类型
        config=config,
        seed=stage8_rng.integers(1_000_000),
    )
    if not sampled_pairs_with_type:
        print(
            "⚠️  WARNING: Sampling resulted in an empty dataset. Halting graph building."
        )
        return

    # --- 3. 实例化分裂器和指挥者 (逻辑不变) ---
    data_splitter = DataSplitter(
        config=config,
        positive_pairs=sampled_pairs_with_type,
        id_mapper=id_mapper,
        seed=stage8_rng.integers(1_000_000),
    )
    director = GraphDirector(config)
    graph_output_path_template = get_path(
        config, "processed.specific.graph_template", suffix="train"
    )

    # 4. 循环每个Fold进行图构建
    for fold_idx, global_train_pairs, global_test_pairs in data_splitter:
        print(
            f"\n{'=' * 30} PROCESSING FOLD {fold_idx} / {config.training.k_folds} {'=' * 30}"
        )

        # a. 定义此Fold的相关实体（仅基于训练集）
        relevant_mol_ids = set()
        relevant_prot_ids = set()
        logic_id_to_type = id_mapper.logic_id_to_type_map

        for u, v, _ in global_train_pairs:
            # 检查源节点
            u_type = logic_id_to_type.get(u)
            if u_type:
                if id_mapper.is_molecule(u_type):
                    relevant_mol_ids.add(u)
                elif id_mapper.is_protein(u_type):
                    relevant_prot_ids.add(u)

            # 检查目标节点
            v_type = logic_id_to_type.get(v)
            if v_type:
                if id_mapper.is_molecule(v_type):
                    relevant_mol_ids.add(v)
                elif id_mapper.is_protein(v_type):
                    relevant_prot_ids.add(v)

        # b. 创建局部上下文
        context = GraphBuildContext(
            fold_idx=fold_idx,
            global_id_mapper=id_mapper,
            global_mol_embeddings=molecule_embeddings,
            global_prot_embeddings=protein_embeddings,
            relevant_mol_ids=relevant_mol_ids,
            relevant_prot_ids=relevant_prot_ids,
            config=config,
        )
        local_train_pairs = context.convert_pairs_to_local(global_train_pairs)

        # c. 实例化Builder并构建图
        builder = HeteroGraphBuilder(
            config=config,
            context=context,
            molecule_embeddings=context.local_mol_embeddings,
            protein_embeddings=context.local_prot_embeddings,
        )
        director.construct(builder, local_train_pairs)
        local_graph_df = builder.get_graph()

        # d. 将图转换回全局ID空间并保存
        graph_schema = config.data_structure.schema.internal.graph_output
        global_graph_df = context.convert_dataframe_to_global(
            local_graph_df,
            source_col=graph_schema.source_node,
            target_col=graph_schema.target_node,
        )

        # 【补完】获取路径并保存文件
        graph_output_path = graph_output_path_template(prefix=f"fold_{fold_idx}")
        rt.ensure_path_exists(graph_output_path)
        global_graph_df.to_csv(graph_output_path, index=False)

        if config.runtime.verbose > 0:
            print(
                f"--> Graph for Fold {fold_idx} saved to '{graph_output_path.name}' with {len(global_graph_df)} edges."
            )

        # e. 生成监督文件
        label_manager = SupervisionFileManager(
            fold_idx=fold_idx,
            config=config,
            global_id_mapper=id_mapper,
            global_positive_pairs_set=sampled_pairs_set_global,
            seed=config.runtime.seed,
        )
        label_manager.generate_and_save(global_train_pairs, global_test_pairs)
