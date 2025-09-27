import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch_geometric.data import HeteroData
import torch
from encoders.ndls_homo_encoder import NDLS_Homo_Encoder
from predictors.gbdt_predictor import GBDT_Link_Predictor
import research_template as rt
from omegaconf import DictConfig
import torch.utils.data
import mlflow
import traceback
import hydra  # noqa: F401
from tqdm import tqdm
from predictors.rgcn_link_predictor import RGCNLinkPredictor
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.loader import LinkNeighborLoader
from encoders.ndls_homo_utils import convert_hetero_to_homo
from data_utils.loaders import (
    load_graph_data_for_fold,
    load_labels_for_fold,
    convert_df_to_local_tensors,  # noqa: F401
    create_global_to_local_maps,
)

target_edge_type = ("drug", "drug_protein_interaction", "protein")


def run_workflow_for_fold(config, hetero_graph, train_df, test_df, device, maps):
    """根据配置分派并执行相应的工作流"""
    paradigm = config.training.paradigm

    if paradigm == "two_stage":
        return run_two_stage_workflow(config, hetero_graph, train_df, test_df, device)
    elif paradigm == "end_to_end":
        return run_end_to_end_workflow(
            config, hetero_graph, train_df, test_df, device, maps
        )
    else:
        raise ValueError(f"Unknown paradigm: {paradigm}")


# region two stage
def run_two_stage_workflow(config, hetero_graph, train_df, test_df, device):
    print("\n--- Running Two-Stage (Encoder + Predictor) Workflow ---")
    # 5a. 运行编码器
    encoder_name = config.encoder.name
    if encoder_name == "ndls_homo":
        adj, features, _ = convert_hetero_to_homo(hetero_graph)
        encoder = NDLS_Homo_Encoder(config, device)
        encoder.fit(adj, features)
        node_embeddings = encoder.get_embeddings()
    else:
        raise NotImplementedError(f"Encoder '{encoder_name}' not supported.")

    # 5b. 运行预测器
    predictor_name = config.predictor.name
    if predictor_name == "gbdt":
        predictor = GBDT_Link_Predictor(config)
        print(predictor.params)
        # [核心变化] 将加载好的数据框直接传进去
        return predictor.predict(node_embeddings, train_df, test_df)
    else:
        raise NotImplementedError(f"Predictor '{predictor_name}' not supported.")


# end region


# region e2e
def run_end_to_end_workflow(
    config: DictConfig,
    hetero_graph: HeteroData,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    device: torch.device,
    maps: dict,
):
    print("\n--- Running End-to-End (Mini-Batch Inductive) Workflow ---")
    target_edge_type = ("drug", "drug_protein_interaction", "protein")
    # --- 1. [数据净化] 确保所有图数据都在CPU上，且类型正确 ---
    print("--> Purifying and validating HeteroData object before Loader...")

    # a. 将图的所有张量都显式地移动到CPU
    for store in hetero_graph.stores:
        for key, value in store.items():
            if torch.is_tensor(value):
                store[key] = value.to("cpu")

    # b. 显式地将所有edge_index转换为torch.long类型
    for edge_type in hetero_graph.edge_types:
        hetero_graph[edge_type].edge_index = hetero_graph[edge_type].edge_index.long()

    print("--> HeteroData object successfully purified on CPU.")
    # --- [核心修复] 在调用Loader之前，将所有监督边，从全局ID转换为局部ID ---
    print(
        "--> Preparing supervision edges with LOCAL indices for LinkNeighborLoader..."
    )
    src_type, _, dst_type = target_edge_type

    # a. 训练监督边 (只使用正样本)
    train_pos_df = train_df[train_df["label"] == 1]
    train_src_local = torch.tensor(
        [maps[src_type][gid] for gid in train_pos_df["source"]], dtype=torch.long
    )
    train_dst_local = torch.tensor(
        [maps[dst_type][gid] for gid in train_pos_df["target"]], dtype=torch.long
    )
    train_edge_label_index_local = torch.stack([train_src_local, train_dst_local])

    # b. 测试监督边
    test_src_local = torch.tensor(
        [maps[src_type][gid] for gid in test_df["source"]], dtype=torch.long
    )
    test_dst_local = torch.tensor(
        [maps[dst_type][gid] for gid in test_df["target"]], dtype=torch.long
    )
    test_edge_label_index_local = torch.stack([test_src_local, test_dst_local])
    test_labels = torch.from_numpy(test_df["label"].values).float()

    # --- [最终的断言] 验证我们转换后的局部ID，没有越界 ---
    assert train_src_local.max() < hetero_graph[src_type].num_nodes
    assert train_dst_local.max() < hetero_graph[dst_type].num_nodes
    assert test_src_local.max() < hetero_graph[src_type].num_nodes
    assert test_dst_local.max() < hetero_graph[dst_type].num_nodes
    print("✅ Local ID validation successful.")

    # --- 实例化数据加载器 (现在所有ID都是自洽的局部ID) ---
    train_loader = LinkNeighborLoader(
        data=hetero_graph,
        num_neighbors=[-1] * config.predictor.params.num_layers,
        edge_label_index=(
            target_edge_type,
            train_edge_label_index_local,
        ),  # <-- 使用局部ID
        batch_size=config.training.get("batch_size", 512),
        shuffle=True,
        neg_sampling_ratio=1.0,
        num_workers=0,
    )
    # --- 2. 模型实例化 ---
    # 我们继续使用手动实例化，因为它最清晰、最可控
    print("--> Instantiating model...")
    model = RGCNLinkPredictor(
        hidden_channels=config.predictor.params.hidden_channels,
        out_channels=config.predictor.params.out_channels,
        num_layers=config.predictor.params.num_layers,
        dropout=config.predictor.params.dropout,
        metadata=hetero_graph.metadata(),
    ).to(device)
    print("--> Model instantiation successful!")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)

    # --- 3. [数据净化] 准备监督边，并确保类型正确 ---
    print("--> Preparing supervision edges with correct data types...")
    # 我们硬编码的target_edge_type，与模型内部一致
    target_edge_type = ("drug", "drug_protein_interaction", "protein")

    # a. 训练监督边 (只使用正样本，Loader会自动处理负采样)
    train_pos_df = train_df[train_df["label"] == 1]
    train_edge_label_index = (
        torch.from_numpy(train_pos_df[["source", "target"]].values)
        .t()
        .contiguous()
        .long()
    )  # <-- [关键] 强制为.long()

    # b. 测试监督边
    test_edge_label_index = (
        torch.from_numpy(test_df[["source", "target"]].values).t().contiguous().long()
    )  # <-- [关键] 强制为.long()
    test_labels = torch.from_numpy(
        test_df["label"].values
    ).float()  # <-- [关键] 强制为.float()

    # --- 4. [最终的断言] 在调用Loader之前，做最后一次检查 ---
    assert hetero_graph.is_undirected(), (
        "FATAL: Graph is not undirected before passing to Loader!"
    )

    # --- 5. 实例化数据加载器 ---
    print("--> Initializing LinkNeighborLoader in single-process mode...")
    train_loader = LinkNeighborLoader(
        data=hetero_graph,
        num_neighbors=[-1] * config.predictor.params.num_layers,
        edge_label_index=(target_edge_type, train_edge_label_index),
        batch_size=config.training.get("batch_size", 512),
        shuffle=True,
        neg_sampling_ratio=1.0,
        num_workers=0,  # <-- [关键] 使用单进程进行调试
    )

    # --- 6. 训练循环 ---
    print("--> Starting mini-batch model training...")
    for epoch in tqdm(range(config.training.epochs), desc="Epochs"):
        total_loss = 0
        total_examples = 0
        model.train()
        for batch in tqdm(train_loader, desc="Batches", leave=False):
            batch = batch.to(device)
            optimizer.zero_grad()

            z_dict = model.forward(batch.x_dict, batch.edge_index_dict)
            scores = model.decode(z_dict, batch[target_edge_type].edge_label_index)
            labels = batch[target_edge_type].edge_label.float()

            loss = F.binary_cross_entropy_with_logits(scores, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * scores.size(0)
            total_examples += scores.size(0)

        # [优化] 避免除以零的错误
        if total_examples > 0:
            avg_loss = total_loss / total_examples
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Avg Loss = {avg_loss:.4f}")

    # --- 7. 评估 ---
    print("--> Starting model evaluation...")
    test_loader = LinkNeighborLoader(
        data=hetero_graph,
        num_neighbors=[-1] * config.predictor.params.num_layers,
        edge_label_index=(
            target_edge_type,
            test_edge_label_index_local,
        ),  # <-- 使用局部ID
        edge_label=test_labels,
        batch_size=config.training.get("batch_size", 512),
        shuffle=False,
        neg_sampling_ratio=0.0,
        num_workers=0,
    )

    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            batch = batch.to(device)
            z_dict = model.forward(batch.x_dict, batch.edge_index_dict)
            scores = model.decode(z_dict, batch[target_edge_type].edge_label_index)

            all_preds.append(torch.sigmoid(scores).cpu())
            all_labels.append(batch[target_edge_type].edge_label.cpu())

    preds = torch.cat(all_preds, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()

    # [优化] 增加一个检查，防止在没有正或负样本时AUC计算报错
    if len(np.unique(labels)) < 2:
        print(
            "    -> WARNING: Test set contains only one class. AUC/AUPR cannot be computed."
        )
        auc, aupr = 0.0, 0.0
    else:
        auc = roc_auc_score(labels, preds)
        aupr = average_precision_score(labels, preds)

    return {"auc": auc, "aupr": aupr}


# end region

# region main


def train(config: DictConfig):
    tracker = rt.MLflowTracker(config)
    # --- Start of protected block for MLflow ---
    try:
        # ===================================================================
        # 2. Setup: Start MLflow Run, Set Environment, and Get Config
        # ===================================================================
        tracker.start_run()  # This also logs all relevant parameters
        device = torch.device(
            config["runtime"]["gpu"] if torch.cuda.is_available() else "cpu"
        )

        # Get the primary switches for the experiment from config
        training_config = config["training"]
        encoder_name = config.encoder.name
        # Use .get() for the optional predictor
        predictor_name = config.predictor.name

        paradigm = training_config.paradigm
        data_variant = "gtopdb" if config.data.use_gtopdb else "baseline"

        # 3. Startup Log
        print("\n" + "=" * 80)
        print(" " * 20 + "Starting DTI Prediction Experiment")
        print("=" * 80)
        print("Configuration loaded for this run:")
        print(f"  - Paradigm (Inferred): '{paradigm}'")
        print(f"  - Primary Dataset:     '{config['data']['primary_dataset']}'")
        print(f"  - Data Variant:        '{data_variant}'")
        print(f"  - Encoder:             '{encoder_name}'")
        print(f"  - Predictor:           '{predictor_name or 'N/A'}'")
        print(f"  - Use Relations:   '{config.relations.name}'")
        print(f"  - Seed:                {config['runtime']['seed']}")
        print(f"  - Device:              {device}")
        print("=" * 80 + "\n")

        k_folds = config.training.evaluation.k_folds
        all_fold_aucs = []
        all_fold_auprs = []

        for fold_idx in range(1, k_folds + 1):
            print("\n" + "#" * 80)
            print(f"#{' ' * 28}PROCESSING FOLD {fold_idx} / {k_folds}{' ' * 28}#")
            print("#" * 80 + "\n")
            global_to_local_maps = create_global_to_local_maps(config)
            # 1. Load data specific to the current fold
            hetero_data = load_graph_data_for_fold(
                config, fold_idx, global_to_local_maps
            )
            train_df, test_df = load_labels_for_fold(config, fold_idx)

            # 2. --- Workflow Dispatcher (based on paradigm) ---
            fold_results = run_workflow_for_fold(
                config, hetero_data, train_df, test_df, device, global_to_local_maps
            )

            # 3. Store results for the current fold
            if fold_results:
                # Assuming the predictor returns a dictionary like {'auc': 0.9, 'aupr': 0.8}
                print(
                    f"--> Fold {fold_idx} Results: AUC = {fold_results['auc']:.4f}, AUPR = {fold_results['aupr']:.4f}"
                )
                all_fold_aucs.append(fold_results["auc"])
                all_fold_auprs.append(fold_results["aupr"])

        # 4. After the loop, log the aggregated cross-validation results
        ### [MODIFIED] ###
        # Logging now happens once at the very end with the CV results.
        if all_fold_aucs and all_fold_auprs:
            print("\n" + "=" * 80)
            print(" " * 25 + "Cross-Validation Summary")
            print("=" * 80)
            print(
                f"  - Mean AUC:  {np.mean(all_fold_aucs):.4f} +/- {np.std(all_fold_aucs):.4f}"
            )
            print(
                f"  - Mean AUPR: {np.mean(all_fold_auprs):.4f} +/- {np.std(all_fold_auprs):.4f}"
            )
            tracker.log_cv_results(all_fold_aucs, all_fold_auprs)

    except Exception as e:
        # ... (error handling remains the same) ...
        print(f"\n!!! FATAL ERROR: Experiment run failed: {e}")
        if tracker.is_active:
            mlflow.set_tag("run_status", "FAILED")
            mlflow.log_text(traceback.format_exc(), "error_traceback.txt")
        raise
    finally:
        # ===================================================================
        # 7. Teardown: Always end the MLflow run
        # ===================================================================
        tracker.end_run()
        print("\n" + "=" * 80)
        print(" " * 27 + "Experiment Run Finished")
        print("=" * 80 + "\n")


# end region
