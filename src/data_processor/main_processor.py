import random
import numpy as np
import pandas as pd
import pickle as pkl
from research_template import get_path, check_files_exist
import research_template as rt
from omegaconf import DictConfig
from pathlib import Path
from tqdm import tqdm
from features.feature_extractors import (
    MoleculeGCN,
    ProteinGCN,
    extract_molecule_features,
    extract_protein_features,
    canonicalize_smiles,
)
from features.similarity_calculators import (
    calculate_drug_similarity,
    calculate_protein_similarity,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


def process_data(config: DictConfig):
    restart_flag = config.runtime.get("force_restart", False)
    full_df = pd.read_csv(get_path(config, "raw.dti_interactions"))
    (
        drug2index,
        ligand2index,
        prot2index,
        dl2index,
        final_smiles_list,
        final_proteins_list,
    ) = _stage_1_load_and_index_entities(config, full_df, restart_flag)

    dl_similarity_matrix, prot_similarity_matrix, _ = (
        _stage_2_generate_features_and_similarities(
            config,
            drug2index,
            ligand2index,
            prot2index,
            final_smiles_list,
            final_proteins_list,
            restart_flag,
        )
    )
    _stage_3_split_data_and_build_graphs(
        config,
        full_df,
        drug2index=drug2index,
        prot2index=prot2index,
        ligand2index=ligand2index,
        dl2index=dl2index,
        dl_sim_matrix=dl_similarity_matrix,
        prot_sim_matrix=prot_similarity_matrix,
    )


def _stage_1_load_and_index_entities(
    config: DictConfig, full_df: pd.DataFrame, restart_flag: bool = False
) -> tuple:
    # 1a. 加载基础数据集的实体
    data_config = config["data"]
    print("--- [Stage 1a] Loading base entities from full_df.csv ---")
    checkpoint_files_dict = {
        "drug": "processed.indexes.drug",
        "ligand": "processed.indexes.ligand",
        "protein": "processed.indexes.protein",
        "nodes": "processed.nodes_metadata",
    }
    if not check_files_exist(config, *checkpoint_files_dict.values()) or restart_flag:
        # 使用 .unique().tolist() 更高效
        unique_smiles = full_df["SMILES"].dropna().unique()
        base_drugs_list = [
            canonicalize_smiles(s)
            for s in tqdm(unique_smiles, desc="Canonicalizing Drug SMILES", leave=False)
        ]
        base_drugs_list = [s for s in base_drugs_list if s is not None]
        base_proteins_list = full_df["Protein"].unique().tolist()

        # 1b. 根据开关，加载GtoPdb的“扩展包”实体
        extra_ligands_list = []
        extra_proteins_list = []
        if data_config["use_gtopdb"]:
            print(
                "\n--- [Stage 1b] GtoPdb integration ENABLED. Loading extra entities... ---"
            )
            try:
                gtopdb_ligands_df = pd.read_csv(
                    get_path(config, "gtopdb.processed.ligands"),
                    header=None,
                    names=["CID", "SMILES"],
                )
                extra_ligands_list = gtopdb_ligands_df["SMILES"].unique().tolist()
                print(f"-> Found {len(extra_ligands_list)} unique ligands from GtoPdb.")

                gtopdb_proteins_df = pd.read_csv(
                    get_path(config, "gtopdb.processed.proteins"),
                    header=None,
                    names=["UniProt", "Sequence"],
                )
                extra_proteins_list = gtopdb_proteins_df["Sequence"].unique().tolist()
                print(
                    f"-> Found {len(extra_proteins_list)} unique proteins from GtoPdb."
                )

            except FileNotFoundError:
                print(
                    "Warning: Processed GtoPdb files not found. Continuing without GtoPdb data."
                )
        else:
            print("\n--- [Stage 1b] GtoPdb integration DISABLED. ---")
        print(
            f"DEBUG_1A: Initial unique drugs = {len(set(base_drugs_list))}, proteins = {len(set(base_proteins_list))}"
        )
        if data_config["use_gtopdb"]:
            print(
                f"DEBUG_1B: Initial unique extra ligands = {len(set(extra_ligands_list))}, extra proteins = {len(set(extra_proteins_list))}"
            )

        # In data_proc.py, replace the entire "region index"

        print("\n--- [Stage 2] Creating and saving index files... ---")

        # 1. 首先，获取所有 drug 和 ligand 的唯一SMILES集合
        unique_drug_smiles = set(base_drugs_list)
        unique_ligand_smiles = set(extra_ligands_list)

        # 2. [核心修复] 对所有分子进行统一规划和去重
        # 我们定义一个优先级规则：如果一个分子同时是drug和ligand，我们视其为drug。
        # 首先确定纯粹的ligand（只在ligand列表里出现）
        pure_ligand_smiles = unique_ligand_smiles - unique_drug_smiles

        # 所有的drug SMILES（包括那些也可能是ligand的）
        all_drug_smiles = unique_drug_smiles

        # 3. 分别对两组SMILES列表进行排序，以保证处理顺序的确定性
        sorted_unique_drugs = sorted(list(all_drug_smiles))
        sorted_unique_ligands = sorted(list(pure_ligand_smiles))
        # 2. 初始化索引字典和节点元数据列表
        drug2index = {}
        ligand2index = {}
        node_data = []
        current_id = 0

        for smile in sorted_unique_drugs:
            drug2index[smile] = current_id
            node_data.append({"node_id": current_id, "node_type": "drug"})
            current_id += 1

        for smile in sorted_unique_ligands:
            ligand2index[smile] = current_id
            node_data.append({"node_id": current_id, "node_type": "ligand"})
            current_id += 1

        # --- 统一处理所有蛋白质 (这部分逻辑不变) ---
        final_proteins_list = sorted(
            list(set(base_proteins_list + extra_proteins_list))
        )
        protein_start_index = current_id  # ID从最后一个小分子之后开始
        prot2index = {
            seq: i + protein_start_index for i, seq in enumerate(final_proteins_list)
        }
        for seq, idx in prot2index.items():
            node_data.append({"node_id": idx, "node_type": "protein"})
        print(
            f"DEBUG_2: Total indexed entities = drug({len(drug2index)}) + ligand({len(ligand2index)}) + protein({len(prot2index)}) = {len(drug2index) + len(ligand2index) + len(prot2index)}"
        )
        print(f"DEBUG_2: Total rows in node_data list = {len(node_data)}")
        # endregion index

        # --- 保存所有文件 ---
        pkl.dump(
            drug2index, open(get_path(config, checkpoint_files_dict["drug"]), "wb")
        )
        pkl.dump(
            ligand2index, open(get_path(config, checkpoint_files_dict["ligand"]), "wb")
        )
        pkl.dump(
            prot2index, open(get_path(config, checkpoint_files_dict["protein"]), "wb")
        )
        # 6. 保存节点元数据文件
        AllNode_df = pd.DataFrame(node_data)  # node_data已经包含了所有类型
        AllNode_df.to_csv(
            get_path(config, checkpoint_files_dict["nodes"]), index=False, header=True
        )
        print("-> Index and metadata files saved successfully.")

    else:
        print("\n--- [Stage 2] Loading indices and metadata from cache... ---")
        # Reuse the list defined above. No more duplication.
        drug_idx_path = get_path(config, checkpoint_files_dict["drug"])
        ligand_idx_path = get_path(config, checkpoint_files_dict["ligand"])
        prot_idx_path = get_path(config, checkpoint_files_dict["protein"])

        drug2index = pkl.load(open(drug_idx_path, "rb"))
        ligand2index = pkl.load(open(ligand_idx_path, "rb"))
        prot2index = pkl.load(open(prot_idx_path, "rb"))

    # --- 为后续步骤准备最终的、完整的实体列表 ---
    # 这些列表现在是从索引字典的键中动态生成的，而不是作为中间变量传来传去
    final_smiles_list = sorted(list(drug2index.keys())) + sorted(
        list(ligand2index.keys())
    )
    final_proteins_list = sorted(list(prot2index.keys()))
    dl2index = {**drug2index, **ligand2index}  # 统一的小分子索引字典
    print(f"DEBUG_3: Length of final_smiles_list = {len(final_smiles_list)}")
    print(f"DEBUG_3: Length of final_proteins_list = {len(final_proteins_list)}")
    print(
        f"DEBUG_3: Total nodes based on final_lists = {len(final_smiles_list) + len(final_proteins_list)}"
    )
    # endregion
    return (
        drug2index,
        ligand2index,
        prot2index,
        dl2index,
        final_smiles_list,
        final_proteins_list,
    )


def _stage_2_generate_features_and_similarities(
    config: DictConfig,
    drug2index: dict,
    ligand2index: dict,
    prot2index: dict,
    final_smiles_list,
    final_proteins_list,
    restart_flag: bool = False,
    device="cpu",
) -> tuple:
    checkpoint_files_dict = {
        "molecule_similarity_matrix": "processed.similarity_matrices.molecule",
        "protein_similarity_matrix": "processed.similarity_matrices.protein",
        "node_features": "processed.node_features",
    }
    if not check_files_exist(config, *checkpoint_files_dict.values()) or restart_flag:
        print("\n--- [Stage 3] Generating features and similarity matrices... ---")
        print("--> Initializing feature extraction models...")
        # Using your GCNLayer, but a standard GCNConv would be better.
        # .eval() is crucial: it disables dropout and other training-specific layers.
        molecule_feature_extractor = MoleculeGCN(5, 128).to(device).eval()
        protein_feature_extractor = ProteinGCN().to(device).eval()

        drug_embeddings = []
        # Use the new function signature in the loop
        for d in tqdm(
            final_smiles_list, desc="Extracting Molecule Features", leave=False
        ):
            embedding = extract_molecule_features(d, molecule_feature_extractor, device)
            drug_embeddings.append(embedding.cpu().detach().numpy())

        protein_embeddings = []
        for p in tqdm(
            final_proteins_list, desc="Extracting Protein Features", leave=False
        ):
            embedding = extract_protein_features(p, protein_feature_extractor, device)
            protein_embeddings.append(embedding.cpu().detach().numpy())

        # prot_similarity_matrix = cosine_similarity(protein_embeddings)
        dl_similarity_matrix = calculate_drug_similarity(final_smiles_list)
        # notice:用U与C一样的方式计算相似度
        prot_similarity_matrix = calculate_protein_similarity(
            final_proteins_list, config.runtime["cpus"]
        )

        pkl.dump(
            dl_similarity_matrix,
            open(
                get_path(config, checkpoint_files_dict["molecule_similarity_matrix"]),
                "wb",
            ),
        )
        pkl.dump(
            prot_similarity_matrix,
            open(
                get_path(config, checkpoint_files_dict["protein_similarity_matrix"]),
                "wb",
            ),
        )
        features_df = pd.concat(
            [pd.DataFrame(drug_embeddings), pd.DataFrame(protein_embeddings)], axis=0
        )
        features_df.to_csv(
            get_path(config, checkpoint_files_dict["node_features"]),
            index=False,
            header=False,
        )
        print(f"DEBUG_4: Shape of final features_df = {features_df.shape}")
        # --- The most critical assertion ---
        num_nodes_from_stage2 = len(drug2index) + len(ligand2index) + len(prot2index)
        assert features_df.shape[0] == num_nodes_from_stage2, (
            f"FATAL: Feature matrix length ({features_df.shape[0]}) does not match indexed node count ({num_nodes_from_stage2})!"
        )
    else:
        # 如果文件已存在，则加载它们以供后续步骤使用
        print(
            "\n--- [Stage 3] Loading features and similarity matrices from cache... ---"
        )
        dl_similarity_matrix = pkl.load(
            open(
                get_path(config, checkpoint_files_dict["molecule_similarity_matrix"]),
                "rb",
            )
        )
        prot_similarity_matrix = pkl.load(
            open(
                get_path(config, checkpoint_files_dict["protein_similarity_matrix"]),
                "rb",
            )
        )
        features_df = pd.read_csv(
            get_path(config, checkpoint_files_dict["node_features"]), header=None
        )
    return dl_similarity_matrix, prot_similarity_matrix, features_df


def _stage_3_split_data_and_build_graphs(
    config: DictConfig,
    full_df: pd.DataFrame,
    drug2index: dict,
    ligand2index: dict,
    prot2index: dict,
    dl2index: dict,
    dl_sim_matrix: np.ndarray,
    prot_sim_matrix: np.ndarray,
):
    """
    调度函数，负责Stage 3的所有工作。
    它现在只做两件事：收集正样本，然后将其交给总指挥官处理。
    """
    # 1. 收集全局的正样本对
    positive_pairs, positive_pairs_normalized_set = _collect_positive_pairs(
        config, full_df, dl2index, prot2index
    )

    # 2. 调用总指挥官，完成所有后续工作
    _process_all_folds(
        config=config,
        positive_pairs=positive_pairs,
        positive_pairs_normalized_set=positive_pairs_normalized_set,
        drug2index=drug2index,
        ligand2index=ligand2index,
        prot2index=prot2index,
        dl2index=dl2index,
        dl_sim_matrix=dl_sim_matrix,
        prot_sim_matrix=prot_sim_matrix,
    )


def _process_all_folds(
    config: DictConfig,
    positive_pairs: list,
    positive_pairs_normalized_set: set,
    drug2index: dict,
    ligand2index: dict,
    prot2index: dict,
    dl2index: dict,
    dl_sim_matrix: np.ndarray,
    prot_sim_matrix: np.ndarray,
):
    """
    一个【总指挥】函数，负责协调完成所有的分割、标签保存和图谱构建任务。
    它循环遍历每一折，并为每一折调用专属的处理函数。
    """
    print("\n--- [CONTROLLER] Starting data processing for all folds... ---")

    eval_config = config.training.evaluation
    split_mode = eval_config.mode
    seed = config.runtime.seed
    num_folds = eval_config.k_folds

    # --- 1. 准备分割迭代器 ---
    # 这部分逻辑决定了我们将如何循环（K-Fold或Single Split）
    if num_folds > 1:
        entities_to_split = None
        if split_mode in ["drug", "protein"]:
            if split_mode == "drug":
                entities_to_split = sorted(
                    list(set([p[0] for p in positive_pairs if p[0] < len(drug2index)]))
                )
            else:  # protein
                entities_to_split = sorted(list(set([p[1] for p in positive_pairs])))
            kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
            split_iterator = kf.split(entities_to_split)
        elif split_mode == "random":
            entities_to_split = positive_pairs
            dummy_y = [
                p[1] for p in positive_pairs
            ]  # Stratify by protein to keep distribution
            skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
            split_iterator = skf.split(entities_to_split, dummy_y)
        else:
            raise ValueError(f"Unknown split_mode '{split_mode}' in config.")
    else:
        # 对于 k=1 的情况，我们创建一个只包含一个None元素的迭代器，以触发一次循环
        split_iterator = [None]
        entities_to_split = (
            positive_pairs  # 占位符，实际在_split_data_for_single_fold中处理
        )

    # --- 2. 主循环 ---
    # 遍历每一次分割，并调用下一层处理函数
    for fold_idx, split_result in enumerate(split_iterator, 1):
        print("\n" + "=" * 80)
        print(" " * 25 + f"PROCESSING FOLD {fold_idx} / {num_folds}")
        print("=" * 80)

        # --- A. 分割数据 ---
        # 调用只负责“分割”的纯函数
        train_positive_pairs, test_positive_pairs = _split_data_for_single_fold(
            split_result=split_result,
            num_folds=num_folds,
            split_mode=split_mode,
            drug2index=drug2index,
            positive_pairs=positive_pairs,
            entities_to_split=entities_to_split,
            config=config,
        )

        # --- B. 处理这一折的所有后续工作 ---
        # 调用负责“内部再分割、保存标签、构建图”的函数
        _process_single_fold(
            fold_idx=fold_idx,
            train_positive_pairs=train_positive_pairs,
            test_positive_pairs=test_positive_pairs,
            positive_pairs_normalized_set=positive_pairs_normalized_set,
            config=config,
            drug2index=drug2index,
            ligand2index=ligand2index,
            prot2index=prot2index,
            dl2index=dl2index,
            dl_sim_matrix=dl_sim_matrix,
            prot_sim_matrix=prot_sim_matrix,
        )


def _split_data_for_single_fold(
    split_result: tuple,
    num_folds: int,
    split_mode: str,
    positive_pairs: list,
    entities_to_split: list,
    config: DictConfig,
    drug2index: dict,  # 需要 drug2index 来区分药物和配体
) -> tuple[list, list]:
    """
    一个只负责【数据分割】的纯函数。

    根据传入的策略（K-Fold/Single Split, Cold/Warm Start），为单一一折
    返回训练集和测试集的正样本对列表。

    Args:
        split_result (tuple or None): KFold迭代器返回的结果。对于k=1，它为None。
        num_folds (int): 总折数 (k)。
        split_mode (str): 分割模式 ('drug', 'protein', 'random')。
        positive_pairs (list): 全局的、所有正样本对的列表。
        entities_to_split (list): 用于分割的实体列表（在冷启动时）或边列表（在热启动时）。
        config (DictConfig): 全局配置对象，用于获取seed和test_fraction。
        drug2index (dict): 药物索引，用于在冷启动时精确识别药物实体。

    Returns:
        tuple[list, list]: (train_positive_pairs, test_positive_pairs)
    """
    train_positive_pairs, test_positive_pairs = [], []
    seed = config.runtime.seed
    test_fraction = config.training.evaluation.test_fraction

    if num_folds > 1:
        # --- 路径 1: K-Fold (k > 1) 逻辑 ---
        train_indices, test_indices = split_result
        if split_mode in ["drug", "protein"]:
            # 冷启动：根据实体索引来划分边
            test_entity_ids = set([entities_to_split[i] for i in test_indices])
            entity_idx = 0 if split_mode == "drug" else 1
            train_positive_pairs = [
                p for p in positive_pairs if p[entity_idx] not in test_entity_ids
            ]
            test_positive_pairs = [
                p for p in positive_pairs if p[entity_idx] in test_entity_ids
            ]
        else:  # random (热启动)
            # 热启动：直接根据边的索引来划分
            train_positive_pairs = [entities_to_split[i] for i in train_indices]
            test_positive_pairs = [entities_to_split[i] for i in test_indices]
    else:
        # --- 路径 2: Single Split (k = 1) 逻辑 ---
        if split_mode in ["drug", "protein"]:
            # 冷启动：先分割实体，再划分边
            entity_idx = 0 if split_mode == "drug" else 1

            # [优化] 确保实体列表的提取是正确的
            if split_mode == "drug":
                # 只分割那些被定义为“药物”的实体
                entity_list = sorted(
                    list(set([p[0] for p in positive_pairs if p[0] < len(drug2index)]))
                )
            else:  # protein
                entity_list = sorted(list(set([p[1] for p in positive_pairs])))

            train_entities, test_entities = train_test_split(
                entity_list, test_size=test_fraction, random_state=seed
            )
            test_entity_ids = set(test_entities)
            train_positive_pairs = [
                p for p in positive_pairs if p[entity_idx] not in test_entity_ids
            ]
            test_positive_pairs = [
                p for p in positive_pairs if p[entity_idx] in test_entity_ids
            ]
        else:  # random (热启动)
            # 热启动：直接用train_test_split分割边
            # stratify可以保证训练集和测试集中的蛋白质分布大致相同，让评估更稳定
            labels = [p[1] for p in positive_pairs]  # 使用 protein_id 作为分层依据
            train_positive_pairs, test_positive_pairs = train_test_split(
                positive_pairs,
                test_size=test_fraction,
                random_state=seed,
                stratify=labels,
            )

    print(
        f"    -> Split complete: {len(train_positive_pairs)} TRAIN positives, {len(test_positive_pairs)} TEST positives."
    )
    return train_positive_pairs, test_positive_pairs


def _process_single_fold(
    fold_idx: int,
    train_positive_pairs: list,
    test_positive_pairs: list,
    positive_pairs_normalized_set: set,
    config: DictConfig,
    drug2index: dict,
    ligand2index: dict,
    prot2index: dict,
    dl2index: dict,
    dl_sim_matrix: np.ndarray,
    prot_sim_matrix: np.ndarray,
):
    """
    一个负责【处理单折所有后续工作】的函数。

    它接收分割好的训练/测试正样本对，然后执行：
    1. 对训练集进行内部的“背景/监督”二次分割。
    2. 为“监督集”和“测试集”生成并保存带标签的CSV文件。
    3. 使用“背景集”和相似性矩阵，构建并保存训练图谱。
    """
    print(
        f"    -> Fold {fold_idx}: Received {len(train_positive_pairs)} train and {len(test_positive_pairs)} test positives."
    )

    # --- 1. 对训练集，进行第二次“内部”分割 (Inductive Link Prediction) ---
    supervision_ratio = config.params.get("supervision_ratio", 0.2)

    # 确保即使训练集很小，也能分出至少一个监督样本
    if len(train_positive_pairs) > 1:
        train_graph_edges, train_supervision_edges = train_test_split(
            train_positive_pairs,
            test_size=supervision_ratio,
            random_state=config.runtime.seed,
        )
    else:  # 如果训练集只有一个样本，则无法分割
        train_graph_edges = train_positive_pairs
        train_supervision_edges = []  # 没有监督边，模型将只从重构损失中学习
        print("    -> WARNING: Too few training samples to create a supervision set.")

    print(
        f"    -> Internal split: {len(train_graph_edges)} edges for graph topology, "
        f"{len(train_supervision_edges)} edges for supervision."
    )

    # --- 2. 为【监督边】和【测试边】生成并保存带标签的文件 ---
    labels_template_key = "processed.link_prediction_labels_template"
    eval_config = config.training.evaluation

    # a. 训练标签文件 (基于监督边)
    train_labels_path = rt.get_path(
        config,
        labels_template_key,
        split_suffix=f"_fold{fold_idx}{eval_config.train_file_suffix}",
    )
    _generate_and_save_labeled_set(
        positive_pairs=train_supervision_edges,
        positive_pairs_normalized_set=positive_pairs_normalized_set,
        dl2index=dl2index,
        prot2index=prot2index,
        config=config,
        output_path=train_labels_path,
    )

    # b. 测试标签文件 (基于测试边)
    test_labels_path = rt.get_path(
        config,
        labels_template_key,
        split_suffix=f"_fold{fold_idx}{eval_config.test_file_suffix}",
    )
    _generate_and_save_labeled_set(
        positive_pairs=test_positive_pairs,
        positive_pairs_normalized_set=positive_pairs_normalized_set,
        dl2index=dl2index,
        prot2index=prot2index,
        config=config,
        output_path=test_labels_path,
    )
    print(f"    -> Saved labeled edges for Fold {fold_idx} successfully.")

    # --- 3. 使用【背景边】和【相似性矩阵】，构建并保存训练图谱 ---
    _build_graph_for_fold(
        fold_idx=fold_idx,
        train_graph_edges=train_graph_edges,  # <-- 使用分割出的背景边
        config=config,
        drug2index=drug2index,
        ligand2index=ligand2index,
        prot2index=prot2index,
        dl2index=dl2index,
        dl_sim_matrix=dl_sim_matrix,
        prot_sim_matrix=prot_sim_matrix,
    )


def _generate_and_save_labeled_set(
    positive_pairs: list,
    positive_pairs_normalized_set: set,
    dl2index: dict,
    prot2index: dict,
    config: DictConfig,
    output_path: Path,
):
    print(
        f"    -> Generating labeled set with {len(positive_pairs)} positive pairs for: {output_path.name}..."
    )

    negative_pairs = []
    all_molecule_ids = list(dl2index.values())
    all_protein_ids = list(prot2index.values())

    # Get the strategy from config
    sampling_strategy = config.params.negative_sampling_strategy

    with tqdm(
        total=len(positive_pairs),
        desc=f"Negative Sampling ({sampling_strategy})",
        leave=False,  # Set leave to False for cleaner logging inside a loop
    ) as pbar:
        if sampling_strategy == "popular":
            # --- Popularity-Biased Strategy ---
            mol_degrees = {}
            prot_degrees = {}
            # [MODIFICATION] Popularity is calculated based on ALL positive pairs for a stable distribution
            for (
                u,
                v,
            ) in positive_pairs_normalized_set:  # TODO: 目前使用全局popular负采样
                mol_degrees[u] = mol_degrees.get(u, 0) + 1
                prot_degrees[v] = prot_degrees.get(v, 0) + 1

            mol_weights = [mol_degrees.get(mol_id, 1) for mol_id in all_molecule_ids]
            prot_weights = [prot_degrees.get(prot_id, 1) for prot_id in all_protein_ids]

            while len(negative_pairs) < len(positive_pairs):
                dl_idx = random.choices(all_molecule_ids, weights=mol_weights, k=1)[0]
                p_idx = random.choices(all_protein_ids, weights=prot_weights, k=1)[0]
                normalized_candidate = tuple(sorted((dl_idx, p_idx)))

                # Check against the set of ALL positives to prevent generating a known interaction
                if normalized_candidate not in positive_pairs_normalized_set:
                    negative_pairs.append((dl_idx, p_idx))
                    pbar.update(1)

        elif sampling_strategy == "uniform":
            # --- Uniform Random Strategy ---
            while len(negative_pairs) < len(positive_pairs):
                dl_idx = random.choice(all_molecule_ids)
                p_idx = random.choice(all_protein_ids)
                normalized_candidate = tuple(sorted((dl_idx, p_idx)))

                if normalized_candidate not in positive_pairs_normalized_set:
                    negative_pairs.append((dl_idx, p_idx))
                    pbar.update(1)
        else:
            raise ValueError(
                f"Unknown negative_sampling_strategy: '{sampling_strategy}'"
            )

    print(f"        -> Generated {len(negative_pairs)} negative samples.")

    # --- [THIS PART IS ALSO MOVED FROM YOUR ORIGINAL CODE] ---
    # Save the unified, labeled edge file
    pos_df = pd.DataFrame(positive_pairs, columns=["source", "target"])
    pos_df["label"] = 1
    neg_df = pd.DataFrame(negative_pairs, columns=["source", "target"])
    neg_df["label"] = 0
    labeled_df = (
        pd.concat([pos_df, neg_df], ignore_index=True)
        .sample(frac=1)
        .reset_index(drop=True)
    )

    rt.ensure_path_exists(output_path)
    labeled_df.to_csv(output_path, index=False, header=True)


def _build_graph_for_fold(
    fold_idx: int,
    train_graph_edges: list,  # <-- [关键] 现在接收的是“背景知识”交互边
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
    print(f"\n--- [GRAPH] Building training graph for Fold {fold_idx}... ---")

    eval_config = config.training.evaluation
    split_mode = eval_config.mode
    params_config = config.params
    relations_config = config.relations.flags
    graph_template_key = "processed.typed_edge_list_template"

    # --- 1. 定义当前Fold的训练节点 ---
    # 这个定义现在是基于传入的 train_graph_edges，因为它们定义了图的“骨架”
    train_node_ids = set()
    if split_mode == "drug":
        # 在药物冷启动中，训练节点 = 所有蛋白 + 所有配体 + 【只在背景知识中出现的】药物
        train_drug_ids = set([p[0] for p in train_graph_edges])
        train_node_ids = (
            set(prot2index.values())
            .union(set(ligand2index.values()))
            .union(train_drug_ids)
        )
    elif split_mode == "protein":
        # 在蛋白冷启动中，训练节点 = 所有药物 + 所有配体 + 【只在背景知识中出现的】蛋白
        train_protein_ids = set([p[1] for p in train_graph_edges])
        train_node_ids = set(dl2index.values()).union(train_protein_ids)
    else:  # random (热启动)
        # 在热启动中，所有节点都可以被认为是训练节点
        train_node_ids = set(dl2index.values()).union(set(prot2index.values()))

    print(
        f"    -> This training graph will be built upon {len(train_node_ids)} allowed nodes."
    )

    typed_edges_list = []

    # --- 2. [核心修改] 将“背景知识交互边”加入图谱 ---
    dp_added_as_background = 0
    lp_added_as_background = 0
    for u, v in train_graph_edges:
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
        f"    -> Added {dp_added_as_background} D-P and {lp_added_as_background} L-P interactions as background knowledge."
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
        train_node_ids=train_node_ids,
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
        train_node_ids=train_node_ids,
        id_to_type_map=mol_local_id_to_type,
        relation_rules=mol_relation_rules,
        config=config,
        id_offset=0,
    )

    # --- 4. 保存最终的图文件 ---
    graph_output_path = rt.get_path(
        config, graph_template_key, split_suffix=f"_fold{fold_idx}"
    )
    print(
        f"\n--> Saving final graph structure for Fold {fold_idx} to: {graph_output_path}"
    )

    typed_edges_df = pd.DataFrame(
        typed_edges_list, columns=["source", "target", "edge_type"]
    )
    typed_edges_df.to_csv(graph_output_path, index=False, header=True)

    print(f"-> Total edges in the final training graph: {len(typed_edges_df)}")


def _add_similarity_edges(
    typed_edges_list: list,
    sim_matrix: np.ndarray,
    train_node_ids: set,
    id_to_type_map: dict,
    relation_rules: dict,
    config: DictConfig,
    id_offset: int = 0,  # [核心新增] ID偏移量，默认为0
):
    """
    一个【最终版】的、通用的、可扩展的辅助函数，用于从一个相似度矩阵中添加多种类型的边。
    它现在可以动态地为每种关系查找专属的阈值。
    """
    print(f"--> Processing similarity matrix of shape {sim_matrix.shape}...")

    counts = {rule["edge_type"]: 0 for rule in relation_rules.values()}

    # --- [核心逻辑] 现在我们遍历规则，而不是遍历矩阵中的所有边 ---
    # 这使得我们可以为每个规则（即每种边类型）应用不同的阈值
    for rule_key, rule in relation_rules.items():
        edge_type = rule["edge_type"]
        relation_flag = edge_type.replace("_similarity", "")

        # 1. 检查配置开关
        if not config.relations.flags.get(relation_flag, True):
            continue

        # 2. [核心修改] 动态构造阈值参数的键名，并从配置中读取
        threshold_key = (
            f"{rule_key[0]}_{rule_key[1]}"  # e.g., 'drug_drug', 'drug_ligand'
        )
        try:
            threshold = config.params.similarity_thresholds[threshold_key]
        except Exception:
            print(
                f"    - WARNING: Threshold key 'params.similarity_thresholds.{threshold_key}' not found. Skipping '{edge_type}'."
            )
            continue

        print(f"    -> Processing '{edge_type}' with threshold > {threshold}...")

        # 3. 寻找超过阈值的边
        rows, cols = np.where(sim_matrix > threshold)

        # 4. 添加边
        for i, j in zip(rows, cols):
            if i >= j:
                continue
            global_id_i = i + id_offset
            global_id_j = j + id_offset

            # [关键] 过滤步骤现在使用全局ID
            if global_id_i not in train_node_ids or global_id_j not in train_node_ids:
                continue

            # 查询类型（现在使用局部索引）
            type1 = id_to_type_map.get(i)
            type2 = id_to_type_map.get(j)

            if not type1 or not type2:
                continue

            if tuple(sorted((type1, type2))) != rule_key:
                continue  # 这条边不属于当前规则处理的范畴

            # 根据规则决定边的方向
            source, target = (
                (i, j) if type1 == rule.get("source_priority", type1) else (j, i)
            )

            typed_edges_list.append([source, target, edge_type])
            counts[edge_type] += 1

    for edge_type, count in counts.items():
        if count > 0:
            print(f"    - Added {count} '{edge_type}' edges.")


def _collect_positive_pairs(
    config: DictConfig, full_df: pd.DataFrame, dl2index: dict, prot2index: dict
) -> list:
    data_config = config["data"]
    eval_config = config.training.evaluation
    split_mode = eval_config.mode
    print(f"--> Preparing and splitting positive pairs with mode: '{split_mode}'...")
    print(
        "--> Labeled train/test files not found or restart forced. Generating new splits..."
    )
    positive_pairs_normalized_set = set()

    # Filter the DataFrame for positive labels first
    positive_interactions_df = full_df[full_df["Y"] == 1].copy()

    # Pre-compute a mapping from raw to canonical SMILES to avoid re-computation
    raw_smiles_in_pos = positive_interactions_df["SMILES"].dropna().unique()
    canonical_map = {s: canonicalize_smiles(s) for s in raw_smiles_in_pos}
    positive_interactions_df["canonical_smiles"] = positive_interactions_df[
        "SMILES"
    ].map(canonical_map)

    for _, row in tqdm(
        positive_interactions_df.iterrows(),
        total=len(positive_interactions_df),
        desc="Processing Interaction Pairs",
    ):
        canonical_s, protein = row["canonical_smiles"], row["Protein"]
        if pd.notna(canonical_s) and canonical_s in dl2index and protein in prot2index:
            d_idx, p_idx = dl2index[canonical_s], prot2index[protein]
            normalized_pair = tuple(sorted((d_idx, p_idx)))
            positive_pairs_normalized_set.add(normalized_pair)

    # Scan interactions from GtoPdb if enabled
    if data_config["use_gtopdb"]:
        print("--> Scanning positive interactions from GtoPdb...")
        gtopdb_edges_df = pd.read_csv(
            rt.get_path(config, "gtopdb.processed.interactions"),
            header=None,
            names=["Sequence", "SMILES", "Affinity"],
        )
        for _, row in tqdm(
            gtopdb_edges_df.iterrows(),
            total=len(gtopdb_edges_df),
            desc="GtoPdb Pairs",
        ):
            smiles, sequence = row["SMILES"], row["Sequence"]
            if smiles in dl2index and sequence in prot2index:
                l_idx, p_idx = dl2index[smiles], prot2index[sequence]
                normalized_pair = tuple(sorted((l_idx, p_idx)))
                positive_pairs_normalized_set.add(normalized_pair)

    positive_pairs = list(positive_pairs_normalized_set)
    print(
        f"-> Found {len(positive_pairs)} unique, normalized positive pairs after cleaning."
    )
    return positive_pairs, positive_pairs_normalized_set
