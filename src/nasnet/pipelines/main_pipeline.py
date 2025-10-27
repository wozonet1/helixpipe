import random
from typing import List, Set, Tuple

import numpy as np
import pandas as pd
import research_template as rt
import torch
from tqdm import tqdm

# (确保在文件顶部有以下 imports)
from nasnet.configs import AppConfig
from nasnet.data_processing import (
    DataSplitter,
    GraphDirector,
    IDMapper,
    InteractionStore,
    StructureProvider,
    purify_dti_dataframe_parallel,
)
from nasnet.data_processing.services.graph_builder import HeteroGraphBuilder
from nasnet.features import extract_features, sim_calculators
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
    # === Stage 2: 结构丰富化 (Enrichment) ===
    all_pids = list(id_mapper.uniprot_to_id.keys())
    all_cids = list(id_mapper.cid_to_id.keys())

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
    purified_entities_df = purify_dti_dataframe_parallel(entities_to_purify_df, config)

    #    c. 【调用 update_from_dataframe】: 将净化后的结果DataFrame“同步”回IDMapper。
    id_mapper.update_from_dataframe(purified_entities_df)
    id_mapper.finalize_mappings()
    # 直接进入下游处理，现在的id_mapper已经是最终状态
    if config.runtime.verbose > 0:
        id_mapper.save_maps_for_debugging(config)

    interaction_store.filter_by_entities(
        valid_cids=set(id_mapper.molecule_to_id.values()),
        valid_pids=set(id_mapper.protein_to_id.values()),
    )
    # 初始化完成，进入下游阶段

    molecule_embeddings, protein_embeddings = _stage_2_generate_features(
        config, id_mapper=id_mapper, restart_flag=restart_flag
    )

    dl_similarity_matrix, prot_similarity_matrix = (
        _stage_3_calculate_similarity_matrices(
            config,
            molecule_embeddings,
            protein_embeddings,
            restart_flag=restart_flag,
        )
    )
    _stage_4_split_data_and_build_graphs(
        config,
        id_mapper,
        interaction_store,  # <-- 传递整个 store
        dl_similarity_matrix,
        prot_similarity_matrix,
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


def _stage_3_calculate_similarity_matrices(
    config: AppConfig,
    molecule_embeddings: torch.Tensor,
    protein_embeddings: torch.Tensor,
    restart_flag: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    【V3.1 模板化重构版】计算所有必需的相似性矩阵。
    """
    print("\n--- [Stage 3] Calculating similarity matrices... ---")
    verbose_level = config.runtime.verbose

    # 1. 计算分子相似性矩阵
    # a. 首先，获取最终的缓存文件路径
    mol_sim_path = get_path(config, "processed.common.similarity_matrices.molecule")

    dl_similarity_matrix = rt.run_cached_calculation(
        cache_path=mol_sim_path,
        calculation_func=lambda: sim_calculators.calculate_embedding_similarity(
            molecule_embeddings
        ),
        force_restart=restart_flag,
        operation_name="Molecule Similarity Matrix",
        verbose=verbose_level,
    )

    # --- 2. 计算蛋白质相似性矩阵 (同理) ---
    prot_sim_path = get_path(config, "processed.common.similarity_matrices.protein")

    prot_similarity_matrix = rt.run_cached_calculation(
        cache_path=prot_sim_path,
        calculation_func=lambda: sim_calculators.calculate_embedding_similarity(
            protein_embeddings
        ),
        force_restart=restart_flag,
        operation_name="Protein Similarity Matrix",
        verbose=verbose_level,
    )

    return dl_similarity_matrix, prot_similarity_matrix


def _stage_4_split_data_and_build_graphs(
    config: AppConfig,
    id_mapper: IDMapper,
    interaction_store: InteractionStore,
    dl_sim_matrix: np.ndarray,
    prot_sim_matrix: np.ndarray,
):
    """
    【V2 - Builder模式版】
    阶段4的总调度函数。负责数据划分、图构建调度和标签文件生成。

    该函数执行以下步骤：
    1. 从 InteractionStore 获取所有带有最终关系类型的正样本对。
    2. 使用 DataSplitter 将这些交互对划分为 K 个折叠(fold)。
    3. 遍历每个 fold:
        a. 实例化一个全新的、干净的 HeteroGraphBuilder。
        b. 使用 GraphDirector 根据配置来指挥 Builder 构建训练图。
        c. 保存构建好的图文件。
        d. 为该 fold 生成并保存训练和测试所需的标签文件。
    """
    print(
        "\n--- [Stage 4] Splitting data, building graphs, and generating labels... ---"
    )

    # 1. 从 InteractionStore 获取所有带有【最终关系类型】的正样本对
    positive_pairs_with_type, positive_pairs_set = (
        interaction_store.get_mapped_positive_pairs(id_mapper)
    )

    # 如果没有任何正样本对，就没有必要继续下去
    if not positive_pairs_with_type:
        print(
            "⚠️  WARNING: No positive interaction pairs found after processing. Halting graph construction."
        )
        return

    # 2. 实例化数据划分器和图构建指挥者 (这两个组件在所有 fold 中可复用)
    data_splitter = DataSplitter(config, positive_pairs_with_type, id_mapper)
    director = GraphDirector(config)

    # 3. 循环遍历每个 Fold，执行完整的构建和保存流程
    for fold_idx, train_pairs, test_pairs in data_splitter:
        print(
            f"\n{'=' * 30} PROCESSING FOLD {fold_idx} / {config.training.k_folds} {'=' * 30}"
        )

        # a. 为当前 fold 实例化一个全新的、干净的 Builder
        #    Builder是有状态的，每一折都必须是一个新实例
        builder = HeteroGraphBuilder(
            config=config,
            id_mapper=id_mapper,
            dl_sim_matrix=dl_sim_matrix,
            prot_sim_matrix=prot_sim_matrix,
        )

        # b. 指挥者根据配置来指挥 Builder 构建训练图
        #    只将【训练集】的交互边作为图的背景知识传入
        director.construct(builder, train_pairs)

        # c. 从 Builder 获取最终构建完成的图
        graph_df = builder.get_graph()

        # d. 保存图文件
        graph_output_path = get_path(
            config,
            "processed.specific.graph_template",
            prefix=f"fold_{fold_idx}",
            suffix="train",
        )
        rt.ensure_path_exists(graph_output_path)
        graph_df.to_csv(graph_output_path, index=False)

        if config.runtime.verbose > 0:
            print(
                f"--> Graph for Fold {fold_idx} saved to '{graph_output_path.name}' with {len(graph_df)} total edges."
            )

        # e. 调用辅助函数，为该 fold 生成并保存标签文件 (train 和 test)
        _generate_and_save_label_files_for_fold(
            fold_idx=fold_idx,
            train_positive_pairs=train_pairs,
            test_positive_pairs=test_pairs,
            positive_pairs_set=positive_pairs_set,
            id_mapper=id_mapper,
            config=config,
        )


def _generate_and_save_label_files_for_fold(
    fold_idx: int,
    train_positive_pairs: List[Tuple[int, int, str]],  # <-- 签名已更新
    test_positive_pairs: List[Tuple[int, int, str]],  # <-- 签名已更新
    positive_pairs_set: Set[Tuple[int, int]],
    id_mapper: IDMapper,
    config: AppConfig,
):
    """
    【V2 - 健壮版】
    一个只负责为单一一折(fold)生成和保存【监督标签】文件的辅助函数。
    它现在能正确处理包含关系类型的三元组输入。

    - 训练标签文件 (train): 仅包含正样本对 (u, v)，用于 LinkNeighborLoader 的监督。
    - 测试标签文件 (test): 包含正样本和按1:1比例采样的负样本，用于最终评估。
    """
    verbose = config.runtime.verbose
    if verbose > 0:
        print(f"    -> Generating label files for Fold {fold_idx}...")

    # 从配置中获取文件名模板和列名 schema
    labels_template_key = "processed.specific.labels_template"
    labels_schema = config.data_structure.schema.internal.labeled_edges_output

    # --- 1. 保存训练标签文件 (仅包含正样本) ---
    train_labels_path = get_path(
        config, labels_template_key, prefix=f"fold_{fold_idx}", suffix="train"
    )

    # 【核心修改】从三元组中提取 (u, v) 对
    train_pairs_for_loader = [(u, v) for u, v, _ in train_positive_pairs]

    train_df = pd.DataFrame(
        train_pairs_for_loader,
        columns=[labels_schema.source_node, labels_schema.target_node],
    )

    rt.ensure_path_exists(train_labels_path)
    train_df.to_csv(train_labels_path, index=False)

    if verbose > 0:
        print(
            f"      - Saved {len(train_df)} positive training pairs to '{train_labels_path.name}'."
        )

    # --- 2. 生成并保存测试标签文件 (包含正负样本) ---
    test_labels_path = get_path(
        config, labels_template_key, prefix=f"fold_{fold_idx}", suffix="test"
    )

    # 【核心修改】从三元组中提取 (u, v) 对
    test_pairs_for_eval = [(u, v) for u, v, _ in test_positive_pairs]

    # a. 处理没有测试正样本的边缘情况
    if not test_pairs_for_eval:
        if verbose > 0:
            print(
                "      - WARNING: No positive pairs for the test set. Saving an empty label file."
            )
        # 创建一个空的但带有正确表头的DataFrame并保存
        pd.DataFrame(
            columns=[
                labels_schema.source_node,
                labels_schema.target_node,
                labels_schema.label,
            ]
        ).to_csv(test_labels_path, index=False)
        return  # 提前结束函数

    # b. 生成负样本
    negative_pairs: List[Tuple[int, int]] = []
    # 从 id_mapper 获取所有可能的分子和蛋白质ID
    all_molecule_ids = list(id_mapper.molecule_to_id.values())
    all_protein_ids = list(id_mapper.protein_to_id.values())

    # 检查是否有足够的实体进行采样
    if not all_molecule_ids or not all_protein_ids:
        print(
            "      - WARNING: Not enough entities to generate negative samples. Test set will only contain positive samples."
        )
    else:
        sampling_strategy = config.data_params.negative_sampling_strategy
        num_neg_to_sample = len(test_pairs_for_eval)

        disable_tqdm = verbose == 0
        with tqdm(
            total=num_neg_to_sample,
            desc=f"      - Neg Sampling for Test ({sampling_strategy})",
            disable=disable_tqdm,
        ) as pbar:
            # 持续采样直到满足数量要求
            while len(negative_pairs) < num_neg_to_sample:
                mol_id = random.choice(all_molecule_ids)
                prot_id = random.choice(all_protein_ids)

                # 检查生成的对是否已经是已知的正样本 (使用传入的 set 进行高效查找)
                if (mol_id, prot_id) not in positive_pairs_set:
                    negative_pairs.append((mol_id, prot_id))
                    pbar.update(1)

    # c. 组合正负样本并保存
    pos_df = pd.DataFrame(
        test_pairs_for_eval,
        columns=[labels_schema.source_node, labels_schema.target_node],
    )
    pos_df[labels_schema.label] = 1

    neg_df = pd.DataFrame(
        negative_pairs, columns=[labels_schema.source_node, labels_schema.target_node]
    )
    neg_df[labels_schema.label] = 0

    # 合并、打乱顺序、重置索引
    labeled_df = (
        pd.concat([pos_df, neg_df], ignore_index=True)
        .sample(frac=1, random_state=config.runtime.seed)
        .reset_index(drop=True)
    )

    rt.ensure_path_exists(test_labels_path)
    labeled_df.to_csv(test_labels_path, index=False)

    if verbose > 0:
        ratio = len(neg_df) / len(pos_df) if len(pos_df) > 0 else 0
        print(
            f"      - Saved {len(labeled_df)} labeled test pairs (1:{ratio:.0f} pos/neg ratio) to '{test_labels_path.name}'."
        )
