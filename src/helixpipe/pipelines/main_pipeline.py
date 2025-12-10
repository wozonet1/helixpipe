import logging
from typing import Sequence, cast

import numpy as np
import pandas as pd
import torch

import helixlib as hx
from helixpipe.data_processing import (
    DataSplitter,
    GraphBuildContext,
    GraphDirector,
    HeteroGraphBuilder,
    IDMapper,
    InteractionStore,
    SelectorExecutor,
    StructureProvider,
    SupervisionFileManager,
    validate_and_filter_entities,
)
from helixpipe.features import extract_features

# (确保在文件顶部有以下 imports)
from helixpipe.typing import (
    CID,
    PID,
    AppConfig,
    AuthID,
    FeatureDict,
    ProcessorOutputs,
)
from helixpipe.utils import get_path

logger = logging.getLogger(__name__)


def process_data(config: AppConfig, processor_outputs: ProcessorOutputs) -> None:
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
    interaction_store = InteractionStore(
        processor_outputs=processor_outputs, config=config
    )
    # === STAGE 2: 实体清单构建 (Entity Manifest Construction) ===
    # 【调用新版函数】
    raw_entities_df = id_mapper.build_entity_manifest()
    if raw_entities_df.empty:
        logger.warning("\n⚠️  No entities found after aggregation. Halting pipeline.")
        return

    # === STAGE 3: 实体丰富化 ===
    enriched_entities_df = _stage3_enrich_entities(raw_entities_df, config, provider)

    # === STAGE 4: 中心化校验 ===
    valid_entities_df = _stage4_validate_entities(enriched_entities_df, config)
    if valid_entities_df.empty:
        logger.warning("\n⚠️  No entities passed validation. Halting pipeline.")
        return

    # === STAGE 5: 最终交互过滤 ===
    valid_entity_ids = set(valid_entities_df["entity_id"])

    # a. IDMapper 最终化
    id_mapper.finalize_with_valid_entities(valid_entity_ids)
    id_mapper.save_maps_to_csv(valid_entities_df)

    # b. InteractionStore 最终化 (过滤)
    pure_interaction_store = interaction_store.filter_by_entities(valid_entity_ids)

    if len(pure_interaction_store) == 0:
        logger.warning("No interactions remained after filtering. Halting pipeline.")
        return

    molecule_embeddings, protein_embeddings = _stage6_generate_features(
        config=config,
        id_mapper=id_mapper,
        entities_with_structures=valid_entities_df,
    )

    # === STAGE 7: 最终产物生成 (图 & 标签) ===
    # 启动包含采样、划分、图构建的最终下游流水线。
    _stage7_split_and_build_graphs(
        config=config,
        id_mapper=id_mapper,
        pure_interaction_store=pure_interaction_store,  # <--- 传递 store 对象
        molecule_embeddings=molecule_embeddings,
        protein_embeddings=protein_embeddings,
    )

    logger.info(
        "\n✅ All data processing stages (up to dispatching) completed successfully!"
    )


def _stage3_enrich_entities(
    raw_entities_df: pd.DataFrame, config: AppConfig, provider: StructureProvider
) -> pd.DataFrame:
    logger.info(
        "\n--- [Pipeline Stage 3] Enriching entities with missing structures... ---"
    )
    enriched_df = raw_entities_df.copy()

    # 筛选出需要丰富化的实体 (structure 列为空)
    to_enrich_mask = enriched_df["structure"].isna()
    if not to_enrich_mask.any():
        logger.info("No missing structures to enrich. Skipping.")
        return enriched_df

    entities_to_enrich = enriched_df[to_enrich_mask]
    logger.info(
        f"Found {len(entities_to_enrich)} entities requiring structure enrichment."
    )

    refetch_flag = config.runtime.force_refetch_structures

    # --- 丰富化分子 (SMILES) ---
    mol_mask = entities_to_enrich["entity_type"].str.contains(
        "molecule|drug|ligand", na=False
    )
    if mol_mask.any():
        # 【核心修正】在这里进行类型转换和清洗
        cids_series = entities_to_enrich.loc[mol_mask, "entity_id"]

        # 1. 使用 pd.to_numeric 将其转换为数值类型，无法转换的变为 NaN
        cids_numeric = pd.to_numeric(cids_series, errors="coerce")

        # 2. 丢弃所有 NaN
        cids_cleaned = cids_numeric.dropna()

        # 3. 转换为整数，并去除重复
        cids_to_fetch = [int(cid) for cid in cids_cleaned.unique()]

        logger.debug(f"Found {len(cids_to_fetch)} unique, valid integer CIDs to fetch.")

        if cids_to_fetch:
            smiles_map = provider.get_smiles(cids_to_fetch, force_restart=refetch_flag)
            if smiles_map:
                mapped_smiles = enriched_df.loc[to_enrich_mask, "entity_id"].map(
                    smiles_map
                )
                # 使用 .fillna() 将新获取的SMILES填充到原有的structure列中
                enriched_df["structure"].fillna(mapped_smiles, inplace=True)

    # --- 丰富化蛋白质 (Sequences) ---
    prot_mask = entities_to_enrich["entity_type"] == "protein"
    if prot_mask.any():
        pids_to_fetch = (
            entities_to_enrich.loc[prot_mask, "entity_id"].dropna().unique().tolist()
        )
        if pids_to_fetch:
            sequence_map = provider.get_sequences(
                pids_to_fetch, force_restart=refetch_flag
            )
            if sequence_map:
                mapped_sequences = enriched_df.loc[to_enrich_mask, "entity_id"].map(
                    sequence_map
                )
                enriched_df["structure"].fillna(mapped_sequences, inplace=True)

    final_with_structure = enriched_df["structure"].notna().sum()
    logger.info(
        f"Enrichment complete. {final_with_structure} / {len(enriched_df)} entities now have structures."
    )
    return enriched_df


def _stage4_validate_entities(
    enriched_df: pd.DataFrame, config: AppConfig
) -> pd.DataFrame:
    """
    【Pipeline Stage 4】中心化校验。
    应用所有过滤规则（全局 + 来源感知）并执行在线一致性检查。
    """
    logger.info("\n--- [Pipeline Stage 4] Validating and filtering entities... ---")

    # 直接调用服务层函数
    # 此时 enriched_df 必须包含 'entity_id', 'structure' 和 'all_sources' 列
    valid_df = validate_and_filter_entities(enriched_df, config)

    # 打印统计信息
    total_input = len(enriched_df)
    total_valid = len(valid_df)
    removed = total_input - total_valid

    if total_valid == 0:
        logger.warning("⚠️  Validation resulted in 0 valid entities!")
    else:
        logger.info(
            f"Validation complete.\n"
            f"    - Input: {total_input}\n"
            f"    - Valid: {total_valid}\n"
            f"    - Removed: {removed} ({removed / total_input:.1%} rejection rate)"
        )

    return valid_df


def _stage6_generate_features(
    config: AppConfig,
    id_mapper: IDMapper,
    entities_with_structures: pd.DataFrame,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    【V2 - 通用亚型版】为所有纯净实体生成或加载特征嵌入。

    该函数通过动态查询IDMapper来处理任意数量的分子和蛋白质亚型。
    """
    logger.info(
        "\n--- [Pipeline Stage 6] Generating or loading node features for all subtypes... ---"
    )

    final_features_path = get_path(config, "processed.common.node_features")
    feature_flag = config.runtime.force_regenerate_features
    # --- 缓存检查 (逻辑不变) ---
    if final_features_path.exists() and not feature_flag:
        logger.info(
            f"--> [Cache Hit] Loading final aggregated features from '{final_features_path.name}'."
        )
        features_np = np.load(final_features_path)

        if len(features_np) != id_mapper.num_total_entities:
            logger.warning(
                f"    - ⚠️ WARNING: Cached feature file size ({len(features_np)}) does not match IDMapper count ({id_mapper.num_total_entities}). Regenerating."
            )
        else:
            num_molecules = id_mapper.num_molecules
            molecule_embeddings = torch.from_numpy(features_np[:num_molecules])
            protein_embeddings = torch.from_numpy(features_np[num_molecules:])
            logger.info("--> Features loaded successfully.")
            return molecule_embeddings, protein_embeddings

    logger.info("--> [Cache Miss/Regenerate] Starting feature calculation process...")

    # --- 1. “分类打包”工作流 ---

    # [修复] 直接使用这些列表，它们的顺序是正确的，无需再排序
    ordered_molecule_ids: list[CID] = []
    ordered_protein_ids: list[PID] = []

    entity_types = id_mapper.entity_types
    for entity_type in entity_types:
        if id_mapper.is_molecule(entity_type):
            ordered_molecule_ids.extend(
                cast(list[CID], id_mapper.get_ordered_ids(entity_type))
            )
        elif id_mapper.is_protein(entity_type):
            ordered_protein_ids.extend(
                cast(list[PID], id_mapper.get_ordered_ids(entity_type))
            )

    structure_map = entities_with_structures.set_index("entity_id")["structure"]

    # [修复] 确保 SMILES/序列列表的顺序与 ID 列表的顺序严格一致
    ordered_smiles = [structure_map.get(mid) for mid in ordered_molecule_ids]
    ordered_sequences = [structure_map.get(pid) for pid in ordered_protein_ids]

    # --- 2. 调用特征提取器 ---
    device = torch.device(config.runtime.gpu if torch.cuda.is_available() else "cpu")

    molecule_features_dict = {}
    if ordered_molecule_ids:
        molecule_features_dict = extract_features(
            entity_type="molecule",
            config=config,
            device=device,
            # [修复] 传入有序的 ID 列表
            authoritative_ids=ordered_molecule_ids,
            sequences_or_smiles=ordered_smiles,
            force_regenerate=feature_flag,
        )

    protein_features_dict = {}
    if ordered_protein_ids:
        protein_features_dict = extract_features(
            entity_type="protein",
            config=config,
            device=device,
            # [修复] 传入有序的 ID 列表
            authoritative_ids=ordered_protein_ids,
            sequences_or_smiles=ordered_sequences,
            force_regenerate=feature_flag,
        )

    # --- 3. 按顺序组装最终的特征矩阵 (您的优秀重构) ---
    def extract_ordered_embeddings(
        ordered_ids: Sequence[AuthID], features_dict: FeatureDict, meta: str
    ) -> list[torch.Tensor]:
        ordered_embeddings = []
        for entity_id in ordered_ids:
            feature = features_dict.get(entity_id)
            if feature is None:
                # 提供了更具体的错误信息
                raise RuntimeError(
                    f"Feature extraction failed for {meta} entity with ID: {entity_id}."
                )
            ordered_embeddings.append(feature)
        return ordered_embeddings

    ordered_mol_embeddings = extract_ordered_embeddings(
        ordered_ids=ordered_molecule_ids,
        features_dict=molecule_features_dict,
        meta="molecule",
    )
    ordered_prot_embeddings = extract_ordered_embeddings(
        ordered_ids=ordered_protein_ids,
        features_dict=protein_features_dict,
        meta="protein",
    )
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
        logger.warning(
            "    - WARNING: Embedding dimensions differ. Applying linear projection."
        )
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
    hx.ensure_path_exists(final_features_path)
    np.save(final_features_path, all_feature_embeddings)
    logger.info(
        f"--> Final aggregated features saved to: '{final_features_path.name}'. Shape: {all_feature_embeddings.shape}"
    )

    return molecule_embeddings, protein_embeddings


def _stage7_split_and_build_graphs(
    config: AppConfig,
    id_mapper: IDMapper,
    pure_interaction_store: InteractionStore,
    molecule_embeddings: torch.Tensor,
    protein_embeddings: torch.Tensor,
):
    """
    【V11 - 最终集成版】
    负责数据采样、划分、图构建和标签生成的总指挥。
    完全基于 InteractionStore 和 Executor-driven 服务进行编排。
    """
    logger.info(
        "--- [Pipeline Stage 7] Orchestrating Sampling, Splitting, and Graph/Label Generation... ---"
    )
    stage7_rng = np.random.default_rng(config.runtime.seed)

    # 1. 在所有下游任务开始前，一次性实例化 SelectorExecutor
    executor = SelectorExecutor(id_mapper, verbose=config.runtime.verbose > 1)

    # 2. 对纯净的交互数据执行采样策略
    sampled_store = pure_interaction_store.apply_sampling_strategy(
        config=config, executor=executor, seed=int(stage7_rng.integers(1_000_000))
    )
    if len(sampled_store) == 0:
        logger.warning(
            "Sampling resulted in an empty dataset. Halting downstream tasks."
        )
        return

    # 3. 实例化数据划分器
    data_splitter = DataSplitter(
        config=config,
        store=sampled_store,
        id_mapper=id_mapper,
        executor=executor,
        seed=int(stage7_rng.integers(1_000_000)),
    )

    # 4. 准备 Director 和文件路径工厂
    director = GraphDirector(config)
    graph_output_path_factory = get_path(config, "processed.specific.graph_template")

    # 5. 获取【采样后】的全局正样本集合，用于所有fold的负采样碰撞检查
    #    这个集合包含的是【权威ID】对
    global_positive_pairs_auth = {
        (row[sampled_store._schema.source_id], row[sampled_store._schema.target_id])
        for _, row in sampled_store.dataframe.iterrows()
    }

    # 6. 循环每个Fold，执行图构建和标签生成
    for fold_idx, (
        train_graph_store,
        train_labels_store,
        test_store,
        cold_start_entity_ids_logic,
    ) in data_splitter:
        logger.info(
            f"\n{'=' * 30} PROCESSING FOLD {fold_idx} / {config.training.k_folds} {'=' * 30}"
        )

        # a. 实例化 SupervisionFileManager
        label_manager = SupervisionFileManager(
            fold_idx=fold_idx,
            config=config,
            id_mapper=id_mapper,
            executor=executor,
            global_positive_set=global_positive_pairs_auth,  # <--- 传入权威ID集合
        )

        # b. 【先行】生成标签文件
        #    这一步现在非常干净，只负责I/O
        label_manager.generate_and_save(
            train_labels_store=train_labels_store,
            test_store=test_store,
        )

        # c. 为图构建准备上下文 (GraphBuildContext)
        #    从 train_graph_store 中提取所有相关的实体ID（权威ID）
        relevant_auth_ids = train_graph_store.get_all_entity_auth_ids()
        relevant_logic_ids = {
            id_mapper.auth_id_to_logic_id_map[auth_id] for auth_id in relevant_auth_ids
        }

        # 将逻辑ID按分子/蛋白质分类
        relevant_mol_ids_logic = set()
        for lid in relevant_logic_ids:
            node_type = id_mapper.logic_id_to_type_map.get(lid)
            if node_type is not None and id_mapper.is_molecule(node_type):
                relevant_mol_ids_logic.add(lid)

        relevant_prot_ids_logic = set()
        for lid in relevant_logic_ids:
            node_type = id_mapper.logic_id_to_type_map.get(lid)
            if node_type is not None and id_mapper.is_protein(node_type):
                relevant_prot_ids_logic.add(lid)

        context = GraphBuildContext(
            fold_idx=fold_idx,
            global_id_mapper=id_mapper,
            global_mol_embeddings=molecule_embeddings,
            global_prot_embeddings=protein_embeddings,
            relevant_mol_ids=relevant_mol_ids_logic,
            relevant_prot_ids=relevant_prot_ids_logic,
            config=config,
        )

        # d. 将权威ID的训练图交互，转换为局部ID
        train_graph_pairs_logic = train_graph_store.get_mapped_positive_pairs(id_mapper)
        local_train_pairs = context.convert_pairs_to_local(train_graph_pairs_logic)

        # e. 实例化Builder并构建图
        builder = HeteroGraphBuilder(
            config=config,
            context=context,
            molecule_embeddings=context.local_mol_embeddings,
            protein_embeddings=context.local_prot_embeddings,
            cold_start_entity_ids_local=context.convert_ids_to_local(
                cold_start_entity_ids_logic
            ),
        )
        director.construct(builder, local_train_pairs)
        local_graph_df = builder.get_graph()

        # f. 将图转换回全局ID空间并保存
        graph_schema = config.data_structure.schema.internal.graph_output
        global_graph_df = context.convert_dataframe_to_global(
            local_graph_df,
            source_col=graph_schema.source_node,
            target_col=graph_schema.target_node,
        )

        graph_output_path = graph_output_path_factory(
            prefix=f"fold_{fold_idx}", suffix="train"
        )
        hx.ensure_path_exists(graph_output_path)
        global_graph_df.to_csv(graph_output_path, index=False)

        logger.info(
            f"--> Graph for Fold {fold_idx} saved to '{graph_output_path.name}' with {len(global_graph_df)} edges."
        )
