import random
import numpy as np
import pandas as pd
import pickle as pkl
import research_template as rt
from tqdm import tqdm
from project_types import AppConfig
import torch
from data_utils.id_mapper import IDMapper
from data_utils.graph_builder import GraphBuilder
from typing import List, Dict
from data_utils.splitters import DataSplitter
from features import extractors
from features import sim_calculators


def process_data(
    config: AppConfig, base_df: pd.DataFrame, extra_dfs: List[pd.DataFrame]
):
    """
    【V3 重构版】数据处理的总编排器。
    接收一个DataFrame列表作为输入，然后驱动所有后续处理。
    """

    restart_flag = config.runtime.get("force_restart", False)
    if base_df is None or base_df.empty:
        raise ValueError("Cannot process data: Input dataframe list is empty.")

    id_mapper = IDMapper(base_df, extra_dfs, config)

    if config.runtime.verbose > 0:
        id_mapper.save_maps_for_debugging(config)

    # --- 后续阶段现在接收 id_mapper 对象 ---

    molecule_embeddings, protein_embeddings = _stage_2_generate_features(
        config, id_mapper, restart_flag=restart_flag
    )

    dl_similarity_matrix, prot_similarity_matrix = (
        _stage_3_calculate_similarity_matrices(
            config,
            molecule_embeddings,
            protein_embeddings,
            restart_flag=restart_flag,
        )
    )
    dataframes_to_process = [base_df] + extra_dfs
    _stage_4_split_data_and_build_graphs(
        config,
        id_mapper,
        dataframes_to_process,
        dl_similarity_matrix,
        prot_similarity_matrix,
    )
    print("\n✅ All data processing stages completed successfully!")


def _stage_2_generate_features(
    config: AppConfig, id_mapper: IDMapper, restart_flag: bool
) -> tuple:
    """
    【V3 重构版】根据IDMapper提供的信息，为所有实体生成特征嵌入。
    """
    print("\n--- [Stage 2] Generating node features... ---")

    final_features_path = rt.get_path(config, "processed.common.node_features")

    if not final_features_path.exists() or restart_flag:
        # 1. 从 id_mapper 获取有序的权威ID和序列/SMILES列表
        ordered_cids = id_mapper._sorted_drugs + id_mapper._sorted_ligands
        ordered_smiles = id_mapper.get_ordered_smiles()
        ordered_pids = id_mapper._sorted_proteins
        ordered_sequences = id_mapper.get_ordered_sequences()

        # 2. 调用新的特征提取器，获取特征字典
        molecule_features_dict = _generate_or_load_embeddings(
            config, "molecule", ordered_cids, ordered_smiles, restart_flag
        )
        protein_features_dict = _generate_or_load_embeddings(
            config, "protein", ordered_pids, ordered_sequences, restart_flag
        )

        # 3. 【核心变化】根据 id_mapper 的顺序，从字典中组装最终的有序张量
        print("--> Assembling final feature matrices from dictionaries...")
        # a. 组装分子嵌入
        ordered_mol_embeddings = [molecule_features_dict[cid] for cid in ordered_cids]
        molecule_embeddings = (
            torch.stack(ordered_mol_embeddings)
            if ordered_mol_embeddings
            else torch.empty(0)
        )

        # b. 组装蛋白质嵌入
        ordered_prot_embeddings = [protein_features_dict[pid] for pid in ordered_pids]
        protein_embeddings = (
            torch.stack(ordered_prot_embeddings)
            if ordered_prot_embeddings
            else torch.empty(0)
        )

        if molecule_embeddings.shape[1] != protein_embeddings.shape[1]:
            print(
                f"Warning: Molecule ({molecule_embeddings.shape[1]}) and \
                    protein ({protein_embeddings.shape[1]}) embedding dims differ."
            )
            # 简单的线性投影作为临时解决方案
            proj = torch.nn.Linear(
                molecule_embeddings.shape[1], protein_embeddings.shape[1]
            ).to(molecule_embeddings.device)
            molecule_embeddings = proj(molecule_embeddings)

        all_feature_embeddings = (
            torch.cat([molecule_embeddings, protein_embeddings], dim=0)
            .cpu()
            .detach()
            .numpy()
        )
        rt.ensure_path_exists(final_features_path)
        np.save(final_features_path, all_feature_embeddings)
        print(
            f"--> Final SOTA features saved to: {final_features_path}. Shape: {all_feature_embeddings.shape}"
        )
    else:
        # 加载逻辑需要根据 id_mapper 确认维度是否匹配
        print("\n--> Found existing features file. Loading from cache...")
        features_np = np.load(final_features_path)
        num_molecules = id_mapper.num_molecules
        molecule_embeddings_np = features_np[:num_molecules]
        protein_embeddings_np = features_np[num_molecules:]

        molecule_embeddings = torch.from_numpy(molecule_embeddings_np)
        protein_embeddings = torch.from_numpy(protein_embeddings_np)

    return molecule_embeddings, protein_embeddings


def _generate_or_load_embeddings(
    config: AppConfig,
    entity_type: str,  # "protein" or "molecule"
    authoritative_ids: list,
    sequences_or_smiles: list,
    restart_flag: bool,
) -> Dict[str, torch.Tensor]:  # <-- 返回值变为字典
    """
    【V2 全局缓存版】高阶辅助函数/动态分派器。

    它会自动：
    1. 从config中读取该实体类型的专属配置 (模型名称, 提取器函数名等)。
    2. 动态地从`features.extractors`模块中获取并调用正确的提取器函数。
    3. 将所有必要的参数（包括权威ID和序列）传递给提取器。

    Returns:
        Dict[str, torch.Tensor]: 一个从权威ID映射到特征张量的字典。
    """
    print(f"\n--> Dispatching feature extraction for '{entity_type}'...")

    # 1. 读取配置 (现在从 data_structure 读取)
    try:
        entity_cfg = config.data_params.feature_extractors[entity_type]
        device = torch.device(
            config.runtime.gpu if torch.cuda.is_available() else "cpu"
        )
        extractor_func_name = entity_cfg.extractor_function
    except KeyError as e:
        raise KeyError(
            f"Configuration error for entity_type '{entity_type}'. Missing key: {e}"
        )

    # 2. 动态获取提取器函数
    try:
        extractor_function = getattr(extractors, extractor_func_name)
    except AttributeError:
        raise AttributeError(
            f"Extractor function '{extractor_func_name}' not found in 'features.extractors' module."
        )
    kwargs_for_extractor = {
        "authoritative_ids": authoritative_ids,
        "config": config,
        "device": device,
        "force_regenerate": restart_flag,
    }

    # 根据实体类型，添加正确的、具有语义的关键字参数
    if entity_type == "protein":
        kwargs_for_extractor["sequences"] = sequences_or_smiles
    elif entity_type == "molecule":
        kwargs_for_extractor["smiles_list"] = sequences_or_smiles
    else:
        raise ValueError(f"Unknown entity_type for feature extraction: {entity_type}")

    # 4. 【核心变化】使用 ** 操作符解包字典，进行调用
    features_dict = extractor_function(**kwargs_for_extractor)

    return features_dict


def _stage_3_calculate_similarity_matrices(
    config: AppConfig,
    molecule_embeddings: torch.Tensor,
    protein_embeddings: torch.Tensor,
    restart_flag: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    checkpoint_files_dict = {
        "molecule_similarity_matrix": "processed.common.similarity_matrices.molecule",
        "protein_similarity_matrix": "processed.common.similarity_matrices.protein",
    }
    if (
        not rt.check_files_exist(config, *checkpoint_files_dict.values())
        or restart_flag
    ):
        print("\n--- [Stage 3b] Calculating similarity matrices... ---")
        dl_similarity_matrix = sim_calculators.calculate_embedding_similarity(
            embeddings=molecule_embeddings, batch_size=1024
        )
        prot_similarity_matrix = sim_calculators.calculate_embedding_similarity(
            embeddings=protein_embeddings, batch_size=1024
        )
        # --- 保存相似度矩阵 ---
        mol_sim_path = rt.get_path(
            config, "processed.common.similarity_matrices.molecule"
        )
        # 【核心修正】在写入文件之前，确保其父目录存在！
        rt.ensure_path_exists(mol_sim_path)
        # 现在可以安全地写入了
        with open(mol_sim_path, "wb") as f:
            pkl.dump(dl_similarity_matrix, f)

        # b. 获取蛋白质相似度矩阵的路径
        prot_sim_path = rt.get_path(
            config, "processed.common.similarity_matrices.protein"
        )
        # 【核心修正】同样确保目录存在
        rt.ensure_path_exists(prot_sim_path)
        with open(prot_sim_path, "wb") as f:
            pkl.dump(prot_similarity_matrix, f)
        print("-> Similarity matrices saved successfully.")
    else:
        print("\n--- [Stage 3b] Loading similarity matrices from cache... ---")
        dl_similarity_matrix = pkl.load(
            open(
                rt.get_path(
                    config, checkpoint_files_dict["molecule_similarity_matrix"]
                ),
                "rb",
            )
        )
        prot_similarity_matrix = pkl.load(
            open(
                rt.get_path(config, checkpoint_files_dict["protein_similarity_matrix"]),
                "rb",
            )
        )
    return dl_similarity_matrix, prot_similarity_matrix


def _stage_4_split_data_and_build_graphs(
    config: AppConfig,
    id_mapper: IDMapper,
    all_dataframes: List[pd.DataFrame],
    dl_sim_matrix: np.ndarray,
    prot_sim_matrix: np.ndarray,
):
    """
    调度函数，负责Stage 3的所有工作。
    它现在只做两件事：收集正样本，然后将其交给总指挥官处理。
    """
    # 1. 收集全局的正样本对
    print(
        "\n--- [Stage 4] Collecting pairs, splitting data, and building graphs... ---"
    )

    # ==========================================================================
    # 1. 收集和转换全局的正样本对 (取代了旧的 _collect_positive_pairs)
    # ==========================================================================
    positive_pairs, positive_pairs_set = id_mapper.map_pairs(all_dataframes)

    data_splitter = DataSplitter(config, positive_pairs, id_mapper)
    graph_builder = GraphBuilder(config, id_mapper, dl_sim_matrix, prot_sim_matrix)
    # 3. 循环遍历 DataSplitter 生成的每个Fold的划分结果
    for fold_idx, train_pairs, test_pairs in data_splitter:
        print(
            f"\n{'=' * 30} PROCESSING FOLD {fold_idx} / {config.training.k_folds} {'=' * 30}"
        )

        # a. 委托GraphBuilder构建和保存该折的图文件
        graph_builder.build_for_fold(fold_idx, train_pairs)

        # b. 保存该折的标签文件 (这个逻辑仍然可以保留在main_pipeline中)
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
    train_positive_pairs: list,
    test_positive_pairs: list,
    positive_pairs_set: set,
    id_mapper: IDMapper,
    config: AppConfig,
):
    """
    一个只负责为单一一折(fold)生成和保存【监督标签】文件的辅助函数。
    """
    verbose = config.runtime.get("verbose", 1)
    if verbose > 0:
        print(f"    -> Generating label files for Fold {fold_idx}...")

    labels_template_key = "processed.specific.labels_template"
    graph_schema = config.data_structure.schema.internal.graph_output
    labels_schema = config.data_structure.schema.internal.labeled_edges_output

    # --- 1. 保存训练标签文件 (仅包含正样本) ---
    #    这些边将用于LinkNeighborLoader的监督学习。
    train_labels_path = rt.get_path(
        config, labels_template_key, prefix=f"fold_{fold_idx}", suffix="train"
    )
    train_df = pd.DataFrame(
        train_positive_pairs,
        columns=[graph_schema.source_node, graph_schema.target_node],
    )
    rt.ensure_path_exists(train_labels_path)
    train_df.to_csv(train_labels_path, index=False)

    if verbose > 0:
        print(
            f"      - Saved {len(train_df)} positive training pairs to '{train_labels_path.name}'."
        )

    # --- 2. 生成并保存测试标签文件 (包含正负样本) ---
    #    这些边将用于最终的模型评估。
    test_labels_path = rt.get_path(
        config, labels_template_key, prefix=f"fold_{fold_idx}", suffix="test"
    )

    if not test_positive_pairs:
        # 如果测试集没有正样本，创建一个空的带表头的csv文件
        if verbose > 0:
            print(
                "      - Warning: No positive pairs for the test set. Saving an empty label file."
            )
        pd.DataFrame(
            columns=[
                labels_schema.source_node,
                labels_schema.target_node,
                labels_schema.label,
            ]
        ).to_csv(test_labels_path, index=False)
        return

    # a. 生成负样本
    negative_pairs = []
    all_molecule_ids = list(id_mapper.molecule_to_id.values())
    all_protein_ids = list(id_mapper.protein_to_id.values())

    sampling_strategy = config.data_params.negative_sampling_strategy

    with tqdm(
        total=len(test_positive_pairs),
        desc=f"      Neg Sampling for Test ({sampling_strategy})",
        disable=verbose == 0,
    ) as pbar:
        while len(negative_pairs) < len(test_positive_pairs):
            mol_idx = random.choice(all_molecule_ids)
            p_idx = random.choice(all_protein_ids)

            # 检查生成的对是否已经是已知的正样本
            if (mol_idx, p_idx) not in positive_pairs_set:
                negative_pairs.append((mol_idx, p_idx))
                pbar.update(1)

    # b. 组合正负样本并保存
    pos_df = pd.DataFrame(
        test_positive_pairs,
        columns=[labels_schema.source_node, labels_schema.target_node],
    )
    pos_df[labels_schema.label] = 1

    neg_df = pd.DataFrame(
        negative_pairs, columns=[labels_schema.source_node, labels_schema.target_node]
    )
    neg_df[labels_schema.label] = 0

    labeled_df = (
        pd.concat([pos_df, neg_df], ignore_index=True)
        .sample(frac=1, random_state=config.runtime.seed)
        .reset_index(drop=True)
    )

    rt.ensure_path_exists(test_labels_path)
    labeled_df.to_csv(test_labels_path, index=False)

    if verbose > 0:
        print(
            f"      - Saved {len(labeled_df)} labeled test pairs (1:{len(negative_pairs) // len(pos_df)} pos/neg ratio) to '{test_labels_path.name}'."
        )
