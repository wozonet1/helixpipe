import random
import numpy as np
import pandas as pd
import pickle as pkl
import research_template as rt
from omegaconf import DictConfig
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

import torch
from data_utils.id_mapper import IDMapper
from typing import List, Dict
from data_utils.splitters import DataSplitter
from features import extractors
from features import sim_calculators


def process_data(
    config: DictConfig, base_df: pd.DataFrame, extra_dfs: List[pd.DataFrame]
):
    """
    【V3 重构版】数据处理的总编排器。
    接收一个DataFrame列表作为输入，然后驱动所有后续处理。
    """

    restart_flag = config.runtime.get("force_restart", False)
    if base_df is None or base_df.empty:
        raise ValueError("Cannot process data: Input dataframe list is empty.")

    id_mapper = IDMapper(base_df, extra_dfs, config)

    if config.runtime.debug:
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
    config: DictConfig, id_mapper: IDMapper, restart_flag: bool
) -> tuple:
    """
    【V3 重构版】根据IDMapper提供的信息，为所有实体生成特征嵌入。
    """
    print("\n--- [Stage 2] Generating node features... ---")

    final_features_path = rt.get_path(
        config, "data_structure.paths.processed.common.node_features"
    )

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
    config: "DictConfig",
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
    config: DictConfig,
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
        pkl.dump(
            dl_similarity_matrix,
            open(
                rt.get_path(
                    config, checkpoint_files_dict["molecule_similarity_matrix"]
                ),
                "wb",
            ),
        )
        pkl.dump(
            prot_similarity_matrix,
            open(
                rt.get_path(config, checkpoint_files_dict["protein_similarity_matrix"]),
                "wb",
            ),
        )
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
    config: DictConfig,
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

    # 3. 循环遍历 DataSplitter 生成的每个Fold的划分结果
    for fold_idx, train_pairs, test_pairs in data_splitter:
        print(
            f"\n{'=' * 35} PROCESSING FOLD {fold_idx} / {config.training.k_folds} {'=' * 35}"
        )

        # 4. 为每个Fold调用“工人”函数
        #    这些“工人”函数现在是 _stage_4_ 的私有辅助函数
        _process_single_fold(
            fold_idx=fold_idx,
            train_positive_pairs=train_pairs,
            test_positive_pairs=test_pairs,
            positive_pairs_set=positive_pairs_set,
            config=config,
            id_mapper=id_mapper,
            dl_sim_matrix=dl_sim_matrix,
            prot_sim_matrix=prot_sim_matrix,
        )


def _process_single_fold(
    fold_idx: int,
    train_positive_pairs: list,
    test_positive_pairs: list,
    positive_pairs_set: set,
    config: "DictConfig",
    id_mapper: "IDMapper",
    dl_sim_matrix: np.ndarray,
    prot_sim_matrix: np.ndarray,
):
    """
    【V3 重构版】负责处理单一一折的所有文件生成任务。
    签名更新，以接收和传递 id_mapper。
    """
    print(f"    -> Fold {fold_idx}: Starting file generation...")

    # 1. 保存训练标签文件 (只有正样本)
    labels_template_key = "data_structure.paths.processed.specific.labels_template"
    train_labels_path = rt.get_path(
        config, labels_template_key, prefix=f"fold_{fold_idx}", suffix="train"
    )

    graph_schema = config.data_structure.schema.internal.graph_output
    pos_df = pd.DataFrame(
        train_positive_pairs,
        columns=[graph_schema.source_node, graph_schema.target_node],
    )

    rt.ensure_path_exists(train_labels_path)
    pos_df.to_csv(train_labels_path, index=False)
    print(
        f"    - Saved {len(pos_df)} training supervision pairs to '{train_labels_path.name}'."
    )

    # 2. 生成并保存测试标签文件
    # 【核心变化】将 id_mapper 传递给下一层
    test_labels_path = rt.get_path(
        config, labels_template_key, prefix=f"fold_{fold_idx}", suffix="test"
    )
    _generate_and_save_labeled_set_for_test(
        positive_pairs=test_positive_pairs,
        positive_pairs_set=positive_pairs_set,
        id_mapper=id_mapper,  # <-- 传递
        config=config,
        output_path=test_labels_path,
    )

    # 3. 构建并保存该Fold的训练图
    # 【核心变化】将 id_mapper 传递给下一层
    _build_graph_for_fold(
        fold_idx=fold_idx,
        train_positive_pairs=train_positive_pairs,
        config=config,
        id_mapper=id_mapper,  # <-- 传递
        dl_sim_matrix=dl_sim_matrix,
        prot_sim_matrix=prot_sim_matrix,
    )


def _generate_and_save_labeled_set_for_test(
    positive_pairs: list,
    positive_pairs_set: set,
    id_mapper: "IDMapper",
    config: "DictConfig",
    output_path: Path,
):
    """
    【V3 重构版】为测试集生成带标签的文件 (1:1 正负样本)。
    负采样现在完全由 IDMapper 提供实体ID池。
    """
    if not positive_pairs:
        print(
            "    - Warning: No positive pairs provided for test set generation. Saving an empty file."
        )
        pd.DataFrame(columns=["source", "target", "label"]).to_csv(
            output_path, index=False
        )
        return

    print(
        f"    - Generating labeled test set with {len(positive_pairs)} positive pairs..."
    )

    negative_pairs = []

    # --- 【核心变化】 ---
    # 直接从 id_mapper 获取所有分子和蛋白质的逻辑ID列表作为采样池
    all_molecule_ids = list(id_mapper.molecule_to_id.values())
    all_protein_ids = list(id_mapper.protein_to_id.values())

    sampling_strategy = config.data_params.negative_sampling_strategy

    with tqdm(
        total=len(positive_pairs),
        desc=f"      Neg Sampling ({sampling_strategy})",
        leave=False,
    ) as pbar:
        # 负采样循环的逻辑保持不变
        while len(negative_pairs) < len(positive_pairs):
            # 目前只实现了 uniform 策略，可以轻松扩展
            mol_idx = random.choice(all_molecule_ids)
            p_idx = random.choice(all_protein_ids)

            # 检查生成的对是否已经是已知的正样本
            # 我们不再需要 sorted()，因为 positive_pairs_set 已经是 (mol_id, prot_id) 格式
            if (mol_idx, p_idx) not in positive_pairs_set:
                negative_pairs.append((mol_idx, p_idx))
                pbar.update(1)

    # --- 保存文件 ---
    # 使用配置中定义的 schema 来确保列名正确
    labels_schema = config.data_structure.schema.internal.labeled_edges_output

    pos_df = pd.DataFrame(
        positive_pairs, columns=[labels_schema.source_node, labels_schema.target_node]
    )
    pos_df[labels_schema.label] = 1

    neg_df = pd.DataFrame(
        negative_pairs, columns=[labels_schema.source_node, labels_schema.target_node]
    )
    neg_df[labels_schema.label] = 0

    # 合并、打乱并保存
    labeled_df = (
        pd.concat([pos_df, neg_df], ignore_index=True)
        .sample(frac=1, random_state=config.runtime.seed)
        .reset_index(drop=True)
    )

    rt.ensure_path_exists(output_path)
    labeled_df.to_csv(output_path, index=False)
    print(f"    - Saved {len(labeled_df)} labeled test pairs to '{output_path.name}'.")


def _build_graph_for_fold(
    fold_idx: int,
    train_positive_pairs: list,
    config: DictConfig,
    drug2index: dict,
    ligand2index: dict,
    prot2index: dict,
    dl2index: dict,
    dl_sim_matrix: np.ndarray,
    prot_sim_matrix: np.ndarray,
):
    """
    为一个特定的Fold，构建并保存其专属的、用于GNN编码器训练的图谱文件。

    这个图谱将包含：
    1. 一部分训练集中的D-P/L-P交互，作为“背景知识”。
    2. 所有与训练集节点相关的“背景知识”相似性边。
    它【绝对不会】包含任何将用于【监督】的交互边。
    """
    print(f"\n--- [GRAPH] Building FULL training graph for Fold {fold_idx}... ---")
    relations_config = config.relations.flags
    graph_template_key = "processed.specific.graph_template"

    typed_edges_list = []

    # --- 2. [核心修改] 将“背景知识交互边”加入图谱 ---
    dp_added_as_background = 0
    lp_added_as_background = 0
    for u, v in train_positive_pairs:
        is_drug = u < len(drug2index)
        if is_drug:
            if relations_config.get("dp_interaction", True):
                typed_edges_list.append([u, v, "drug_protein_interaction"])
                dp_added_as_background += 1
        else:  # is_ligand
            if relations_config.get("lp_interaction", True):
                typed_edges_list.append([u, v, "ligand_protein_interaction"])
                lp_added_as_background += 1

    print(
        f"    -> Added {dp_added_as_background} D-P and {lp_added_as_background} L-P interactions."
    )

    # --- 3. 添加相似性边 (这部分现在是增量，而不是全部) ---

    # a. 为蛋白质相似性做准备
    prot_local_id_to_type = {i: "protein" for i in range(len(prot2index))}
    prot_relation_rules = {
        ("protein", "protein"): {"edge_type": "protein_protein_similarity"}
    }

    _add_similarity_edges(
        typed_edges_list=typed_edges_list,
        sim_matrix=prot_sim_matrix,
        id_to_type_map=prot_local_id_to_type,
        relation_rules=prot_relation_rules,
        config=config,
        id_offset=len(dl2index),
    )

    # b. 为小分子相似性做准备
    mol_local_id_to_type = {i: "drug" for i, smi in enumerate(drug2index.keys())}
    mol_local_id_to_type.update(
        {i + len(drug2index): "ligand" for i, smi in enumerate(ligand2index.keys())}
    )
    mol_relation_rules = {
        ("drug", "drug"): {"edge_type": "drug_drug_similarity"},
        ("ligand", "ligand"): {"edge_type": "ligand_ligand_similarity"},
        ("drug", "ligand"): {
            "edge_type": "drug_ligand_similarity",
            "source_priority": "drug",
        },
    }

    _add_similarity_edges(
        typed_edges_list=typed_edges_list,
        sim_matrix=dl_sim_matrix,
        id_to_type_map=mol_local_id_to_type,
        relation_rules=mol_relation_rules,
        config=config,
        id_offset=0,
    )

    # --- 4. 保存最终的图文件 ---
    graph_output_path = rt.get_path(
        config, graph_template_key, prefix=f"fold_{fold_idx}"
    )
    print(
        f"\n--> Saving final graph structure for Fold {fold_idx} to: {graph_output_path}"
    )

    typed_edges_df = pd.DataFrame(
        typed_edges_list, columns=["source", "target", "edge_type"]
    )
    typed_edges_df.to_csv(graph_output_path, index=False, header=True)

    print(f"-> Total edges in the final training graph: {len(typed_edges_df)}")


# ... (其他 imports 和函数) ...


def _add_similarity_edges(
    typed_edges_list: list,
    sim_matrix: np.ndarray,
    id_mapper: "IDMapper",
    config: "DictConfig",
    entity_type: str,  # 'molecule' or 'protein'
):
    """
    【V3 重构版】通用的、由IDMapper驱动的相似性边添加函数。

    Args:
        typed_edges_list (list): 将要被就地修改的、包含所有图边的列表。
        sim_matrix (np.ndarray): 待处理的相似性矩阵。
        id_mapper (IDMapper): 初始化的IDMapper，提供所有ID和类型信息。
        config (DictConfig): 全局配置对象。
        entity_type (str): 正在处理的实体类型 ('molecule' or 'protein')。
    """
    print(
        f"    - Processing {entity_type} similarity matrix of shape {sim_matrix.shape}..."
    )

    # --- 1. 根据实体类型，确定ID偏移量 ---
    # 这是之前作为参数传入，现在由函数自己计算的核心信息
    id_offset = 0
    if entity_type == "protein":
        id_offset = id_mapper.num_molecules
    elif entity_type != "molecule":
        # 如果实体类型未知，则直接返回，不执行任何操作
        return

    # --- 2. 遍历相似性矩阵的上三角，提高效率 ---
    # np.triu(..., k=1) 会将对角线和下三角的元素置为0
    rows, cols = np.where(np.triu(sim_matrix, k=1))

    edge_counts = defaultdict(int)

    # 使用zip打包进行高效迭代
    for i, j in zip(rows, cols):
        similarity = sim_matrix[i, j]

        # a. 将局部索引 (i, j) 转换为全局逻辑ID
        global_id_i = i + id_offset
        global_id_j = j + id_offset

        # b. 【核心变化】使用 id_mapper 获取节点类型
        type1 = id_mapper.get_node_type(global_id_i)
        type2 = id_mapper.get_node_type(global_id_j)

        # c. 根据类型确定关系前缀，例如 "drug_drug", "drug_ligand"
        #    使用 sorted 确保 ("drug", "ligand") 和 ("ligand", "drug") 得到相同的前缀
        relation_prefix = "_".join(sorted((type1, type2)))

        # d. 检查这个关系类型是否被启用
        relation_flag_key = f"{relation_prefix}_similarity"
        if not config.relations.flags.get(relation_flag_key, False):
            continue

        # e. 从配置中获取该关系类型的特定阈值
        threshold = config.data_params.similarity_thresholds.get(relation_prefix, 1.1)

        # f. 如果相似度超过阈值，则添加边
        if similarity > threshold:
            edge_type = relation_flag_key

            # g. 【核心变化】处理边的方向性 (特别是 drug-ligand)
            #    我们约定 drug 总是 source (ID较小)
            #    由于 i < j，且 drug 的 ID 范围小于 ligand，所以 global_id_i 总是 source
            #    对于同类型边，顺序无所谓，i < j 保证了我们只添加一个方向
            source_id, target_id = global_id_i, global_id_j

            typed_edges_list.append([source_id, target_id, edge_type])
            edge_counts[edge_type] += 1

    # --- 3. 打印总结报告 ---
    for edge_type, count in edge_counts.items():
        print(f"      - Added {count} '{edge_type}' edges.")
