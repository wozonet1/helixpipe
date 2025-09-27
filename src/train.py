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
from data_utils import loaders, debug_utils, preparers, transforms

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
        adj, features, _ = transforms.convert_hetero_to_homo(hetero_graph)
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
    (
        hetero_graph,
        train_edge_label_index_local,
        test_edge_label_index_local,
        test_labels,
    ) = preparers.prepare_e2e_data(
        hetero_graph, train_df, test_df, maps, target_edge_type
    )

    # [重构点 2] 一行代码运行所有可选的调试检查 (可以安全地注释掉这一行来提速)
    if config.runtime.get("debug", False):  # 建议在配置中加一个开关
        debug_utils.run_optional_diagnostics(hetero_graph)

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

    # --- 5. 实例化数据加载器 ---
    print("--> Initializing LinkNeighborLoader in single-process mode...")
    train_loader = LinkNeighborLoader(
        data=hetero_graph,
        num_neighbors=[-1] * config.predictor.params.num_layers,
        edge_label_index=(target_edge_type, train_edge_label_index_local),
        batch_size=config.training.get("batch_size", 512),
        shuffle=True,
        neg_sampling_ratio=1.0,
        num_workers=config.runtime.cpus,
    )
    test_loader = LinkNeighborLoader(
        data=hetero_graph,
        num_neighbors=[-1] * config.predictor.params.num_layers,
        edge_label_index=(
            target_edge_type,
            test_edge_label_index_local,
        ),
        edge_label=test_labels,
        batch_size=config.training.get("batch_size", 512),
        shuffle=False,
        neg_sampling_ratio=0.0,
        num_workers=config.runtime.cpus,
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
            global_to_local_maps = loaders.create_global_to_local_maps(config)
            # 1. Load data specific to the current fold
            hetero_data = loaders.load_graph_data_for_fold(
                config, fold_idx, global_to_local_maps
            )
            train_df, test_df = loaders.load_labels_for_fold(config, fold_idx)

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
