from typing import List

import numpy as np
import pandas as pd
import research_template as rt
import torch

# (确保在文件顶部有以下 imports)
from nasnet.configs import AppConfig
from nasnet.data_processing import (
    DataSplitter,
    GraphBuildContext,
    GraphDirector,
    IDMapper,
    InteractionStore,
    StructureProvider,
    SupervisionFileManager,
    purify_entities_dataframe_parallel,
    sample_interactions,
)
from nasnet.data_processing.services.graph_builder import HeteroGraphBuilder
from nasnet.features import extract_features
from nasnet.utils import get_path


def process_data(
    config: AppConfig, base_df: pd.DataFrame, extra_dfs: List[pd.DataFrame]
):
    """
    【V3 重构版】数据处理的总编排器。
    接收一个DataFrame列表作为输入，然后驱动所有后续处理。
    """
    # 先是对三个主要类进行初始化(InteractionStore, StructureProvider, IDMapper)
    restart_flag = config.runtime.get("force_restart", False)
    if base_df is None or base_df.empty:
        raise ValueError("Cannot process data: Input dataframe list is empty.")
    all_raw_interactions_dfs = [base_df] + (extra_dfs if extra_dfs else [])
    interaction_store = InteractionStore(all_raw_interactions_dfs, config)
    structure_provider = StructureProvider(config, proxies=None)
    id_mapper = IDMapper(all_raw_interactions_dfs, config)
    schema = config.data_structure.schema.internal.authoritative_dti
    # a. 从 base_df (主数据集) 中提取所有唯一的分子ID
    drug_cids_from_base = set(
        pd.to_numeric(base_df[schema.molecule_id], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    )
    # b. 调用 IDMapper 的新方法，将这个定义注入进去
    id_mapper.set_drug_cids(drug_cids_from_base)
    # === Stage 2: 结构丰富化 (Enrichment) ===
    all_pids = list(id_mapper.get_all_pids())
    all_cids = list(id_mapper.get_all_cids())
    # FIXME:解决每次都提示那一个
    complete_sequences = structure_provider.get_sequences(
        all_pids, force_restart=restart_flag
    )
    complete_smiles = structure_provider.get_smiles(
        all_cids, force_restart=restart_flag
    )

    # 注入数据
    id_mapper.update_sequences(complete_sequences)
    id_mapper.update_smiles(complete_smiles)
    # --- 后续阶段现在接收 id_mapper 对象 ---
    entities_to_purify_df = id_mapper.to_dataframe()

    #    b. 净化：将这个导出的DataFrame交给一个外部的、专门的净化函数处理。
    purified_entities_df = purify_entities_dataframe_parallel(
        entities_to_purify_df, config
    )

    #    c. 【调用 update_from_dataframe】: 将净化后的结果DataFrame“同步”回IDMapper。
    id_mapper.update_from_dataframe(purified_entities_df)
    id_mapper.finalize_mappings()
    # 直接进入下游处理，现在的id_mapper已经是最终状态
    id_mapper.save_nodes_metadata()

    interaction_store.filter_by_entities(
        valid_cids=set(id_mapper.get_all_cids()),
        valid_pids=set(id_mapper.get_all_pids()),
    )
    # 初始化完成，进入下游阶段

    molecule_embeddings, protein_embeddings = _stage_2_generate_features(
        config, id_mapper=id_mapper, restart_flag=restart_flag
    )
    # TODO:隔离cache与纯计算?
    _stage_4_split_data_and_build_graphs(
        config,
        id_mapper,
        interaction_store,  # <-- 传递整个 store
        molecule_embeddings,
        protein_embeddings,
        master_seed=config.runtime.seed,
    )

    print("\n✅ All data processing stages completed successfully!")


def _stage_2_generate_features(
    config: AppConfig, id_mapper: IDMapper, restart_flag: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    【V4 - 最终精简版】阶段2：为所有实体生成或加载特征嵌入。

    该函数执行以下步骤：
    1. 检查最终的聚合特征文件 (`node_features.npy`) 是否存在。
       - 如果存在且不强制重启，则直接加载并返回。
    2. 如果缓存未命中：
       a. 从 IDMapper 获取所有纯净实体的有序ID和结构列表。
       b. 确定计算设备 (CPU/GPU)。
       c. 直接调用 `extract_features` 分派函数，分别为分子和蛋白质提取特征。
          (这个函数内部有更细粒度的“每个实体一个文件”的缓存机制)。
       d. 根据 IDMapper 的顺序，将返回的特征字典重新组装成有序的张量。
       e. (可选) 对齐不同模态的特征维度。
       f. 将所有特征拼接成一个大的 numpy 数组并保存到 `node_features.npy`。
    3. 返回分子和蛋白质的特征张量。
    """
    print("\n--- [Stage 2] Generating or loading node features... ---")

    final_features_path = get_path(config, "processed.common.node_features")

    # --- 缓存检查：检查最终的 .npy 文件是否存在 ---
    if final_features_path.exists() and not restart_flag:
        print(
            f"--> [Cache Hit] Loading final aggregated features from '{final_features_path.name}'."
        )
        features_np = np.load(final_features_path)
        num_molecules = id_mapper.num_molecules

        # 确保加载的数组大小与IDMapper中的实体数量一致
        if len(features_np) != id_mapper.num_total_entities:
            print(
                f"    - ⚠️ WARNING: Cached feature file size ({len(features_np)}) does not match "
                f"IDMapper entity count ({id_mapper.num_total_entities}). Regenerating features."
            )
        else:
            molecule_embeddings = torch.from_numpy(features_np[:num_molecules])
            protein_embeddings = torch.from_numpy(features_np[num_molecules:])
            print("--> Features loaded successfully.")
            return molecule_embeddings, protein_embeddings

    # --- 如果最终文件不存在或大小不匹配，则开始计算 ---
    print("--> [Cache Miss/Regenerate] Starting feature calculation process...")

    # 1. 确定计算设备
    device = torch.device(config.runtime.gpu if torch.cuda.is_available() else "cpu")

    # 2. 从 IDMapper 获取有序的权威ID和结构列表
    #    假设 IDMapper 提供了这些公共方法来获取排序后的列表
    ordered_cids = id_mapper.get_ordered_cids()
    ordered_smiles = id_mapper.get_ordered_smiles()
    ordered_pids = id_mapper.get_ordered_pids()
    ordered_sequences = id_mapper.get_ordered_sequences()

    # 3. 直接调用 extract_features 分派函数
    molecule_features_dict = extract_features(
        entity_type="molecule",
        config=config,
        device=device,
        authoritative_ids=ordered_cids,
        sequences_or_smiles=ordered_smiles,
        force_regenerate=restart_flag,
    )

    protein_features_dict = extract_features(
        entity_type="protein",
        config=config,
        device=device,
        authoritative_ids=ordered_pids,
        sequences_or_smiles=ordered_sequences,
        force_regenerate=restart_flag,
    )

    # 4. 根据 IDMapper 的顺序，从字典中组装最终的有序张量
    print("\n--> Assembling final feature matrices from dictionaries...")
    ordered_mol_embeddings = [molecule_features_dict.get(cid) for cid in ordered_cids]
    ordered_prot_embeddings = [protein_features_dict.get(pid) for pid in ordered_pids]

    # 健壮性检查：确保所有实体都成功提取了特征
    if any(e is None for e in ordered_mol_embeddings):
        failed_cids = [
            cid for cid, emb in zip(ordered_cids, ordered_mol_embeddings) if emb is None
        ]
        raise RuntimeError(
            f"Molecule feature extraction failed for CIDs: {failed_cids[:5]}"
        )

    if any(e is None for e in ordered_prot_embeddings):
        failed_pids = [
            pid
            for pid, emb in zip(ordered_pids, ordered_prot_embeddings)
            if emb is None
        ]
        raise RuntimeError(
            f"Protein feature extraction failed for PIDs: {failed_pids[:5]}"
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

    # 5. (可选) 维度对齐
    if (
        molecule_embeddings.numel() > 0
        and protein_embeddings.numel() > 0
        and molecule_embeddings.shape[1] != protein_embeddings.shape[1]
    ):
        print(
            f"    - WARNING: Molecule ({molecule_embeddings.shape[1]}D) and protein ({protein_embeddings.shape[1]}D) embedding dimensions differ. Applying linear projection to molecules."
        )
        proj = torch.nn.Linear(
            molecule_embeddings.shape[1], protein_embeddings.shape[1]
        ).to(device)
        molecule_embeddings = proj(molecule_embeddings.to(device)).cpu()

    # 6. 拼接并保存到最终的 .npy 缓存文件
    if molecule_embeddings.numel() == 0 and protein_embeddings.numel() == 0:
        print(
            "    - WARNING: No features were generated for any entity. Saving an empty feature file."
        )
        all_feature_embeddings = np.array([])
    else:
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


def _stage_4_split_data_and_build_graphs(
    config: AppConfig,
    id_mapper: IDMapper,
    interaction_store: InteractionStore,
    molecule_embeddings: torch.Tensor,
    protein_embeddings: torch.Tensor,
    master_seed: int = 42,
):
    """
    【V6 - 上下文服务重构版】
    负责数据采样、划分、图构建调度和标签文件生成的总指挥。
    它现在使用 GraphBuildContext 来封装所有局部上下文管理逻辑。
    """
    print(
        "\n--- [Stage 4] Sampling, splitting, building graphs (with local context)... ---"
    )
    stage4_rng = np.random.default_rng(config.runtime.seed)
    # --- 步骤 1: 采样 (逻辑不变) ---
    all_positive_pairs_with_type, _ = interaction_store.get_mapped_positive_pairs(
        id_mapper
    )
    if not all_positive_pairs_with_type:
        print("⚠️  WARNING: No positive interaction pairs found. Halting Stage 4.")
        return

    sampled_pairs_with_type, sampled_pairs_set_global = sample_interactions(
        all_positive_pairs=all_positive_pairs_with_type,
        id_mapper=id_mapper,
        config=config,
        seed=stage4_rng.integers(114514),
    )
    if not sampled_pairs_with_type:
        print("⚠️  WARNING: Sampling resulted in an empty dataset. Halting Stage 4.")
        return

    data_splitter = DataSplitter(
        config=config,
        positive_pairs=sampled_pairs_with_type,
        id_mapper=id_mapper,  # 传递全局 id_mapper 以供验证使用
        seed=stage4_rng.integers(1919),  # 从主RNG获取一个新种子
    )
    director = GraphDirector(config)
    graph_output_path_template = get_path(
        config,
        "processed.specific.graph_template",
        suffix="train",
    )
    # --- [MODIFIED] 步骤 5: 循环每个Fold，但在局部空间内构建图 ---
    for fold_idx, global_train_pairs, global_test_pairs in data_splitter:
        print(
            f"\n{'=' * 30} PROCESSING FOLD {fold_idx} / {config.training.k_folds} {'=' * 30}"
        )
        # 4. [MOVED & MODIFIED] 在循环内部，为当前Fold创建专属上下文
        #    只使用【训练集】的交互来定义相关实体
        relevant_mol_ids = {u for u, v, _ in global_train_pairs}
        relevant_prot_ids = {v for u, v, _ in global_train_pairs}

        context = GraphBuildContext(
            global_id_mapper=id_mapper,
            fold_idx=fold_idx,
            global_mol_embeddings=molecule_embeddings,
            global_prot_embeddings=protein_embeddings,
            relevant_mol_ids=relevant_mol_ids,
            relevant_prot_ids=relevant_prot_ids,
            config=config,
        )
        local_train_pairs = context.convert_pairs_to_local(global_train_pairs)
        # a. 实例化 Builder，并注入【局部】上下文信息
        #    注意: 我们不再传递完整的 id_mapper，而是传递 context 提供的局部信息
        builder = HeteroGraphBuilder(
            config=config,
            context=context,
            molecule_embeddings=context.local_mol_embeddings,
            protein_embeddings=context.local_prot_embeddings,
        )

        # b. 指挥者在【局部】空间内构建训练图
        director.construct(builder, local_train_pairs)
        local_graph_df = builder.get_graph()

        # c. [NEW] 将图转换回【全局】ID空间
        global_graph_df = context.convert_dataframe_to_global(
            local_graph_df,
            source_col=config.data_structure.schema.internal.graph_output.source_node,
            target_col=config.data_structure.schema.internal.graph_output.target_node,
        )

        graph_output_path = graph_output_path_template(prefix=f"fold_{fold_idx}")
        rt.ensure_path_exists(graph_output_path)
        global_graph_df.to_csv(graph_output_path, index=False)

        if config.runtime.verbose > 0:
            print(
                f"--> Graph for Fold {fold_idx} saved to '{graph_output_path.name}' with {len(global_graph_df)} edges."
            )

        # f. 委托 SupervisionFileManager 处理所有标签文件
        label_manager = SupervisionFileManager(
            fold_idx=fold_idx,
            config=config,
            global_id_mapper=id_mapper,
            global_positive_pairs_set=sampled_pairs_set_global,
            seed=stage4_rng.integers(1_000_000),
        )

        # 使用最终简化版的接口
        label_manager.generate_and_save(
            train_pairs_global=global_train_pairs, test_pairs_global=global_test_pairs
        )
