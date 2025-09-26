import numpy as np
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

from encoders.ndls_homo_utils import convert_hetero_to_homo
from data_utils.loaders import (
    load_graph_data_for_fold,
    load_labels_for_fold,
    convert_df_to_local_tensors,
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
def run_end_to_end_workflow(config, hetero_graph, train_df, test_df, device, maps):
    print("\n--- Running End-to-End (RGCN) Workflow ---")
    hetero_graph = hetero_graph.to(device)

    try:
        print("--> Attempting manual instantiation...")
        model = RGCNLinkPredictor(
            # 从config中手动读取超参数
            hidden_channels=config.predictor.params.hidden_channels,
            out_channels=config.predictor.params.out_channels,
            num_layers=config.predictor.params.num_layers,
            dropout=config.predictor.params.dropout,
            # 传入我们刚刚打印的、确凿无疑的metadata
            metadata=hetero_graph.metadata(),
        )
        print("--> Manual instantiation SUCCEEDED!")
        model.to(device)
    except AssertionError as e:
        print("\n!!! CAPTURED ASSERTION ERROR !!!")
        # 启用HYDRA_FULL_ERROR=1后，这里将打印出完整的、源自PyG的Traceback
        import traceback

        traceback.print_exc()
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # 即使出错，我们也手动结束，防止其他错误干扰
        raise e
    print("--> Model instantiation successful!")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)

    print("--> Preparing full-batch training & testing tensors with LOCAL indices...")
    # [验证] 我们可以打印一个映射来检查它的正确性
    print(
        "DEBUG: Drug global to local map sample:",
        list(maps["drug"].items())[:5],
    )

    # a. 转换训练集 (后续逻辑几乎不变)
    src_type, _, dst_type = target_edge_type
    train_edge_label_index, train_edge_label = convert_df_to_local_tensors(
        train_df, maps, src_type, dst_type, device
    )

    # b. 转换测试集
    test_edge_label_index, test_labels = convert_df_to_local_tensors(
        test_df, maps, src_type, dst_type, device
    )
    test_labels = test_df["label"].values
    # --- [DEBUG PROBE 1] Verifying Tensor Shapes Before Training ---
    print("\n" + "=" * 50)
    print(" " * 15 + "DEBUG PROBE 1: DATA VERIFICATION")
    print("=" * 50)
    print(f"Total nodes in HeteroData object: {hetero_graph.num_nodes}")
    # 打印每种节点类型的数量
    for node_type in hetero_graph.node_types:
        print(f"  - Nodes of type '{node_type}': {hetero_graph[node_type].num_nodes}")

    print(f"\nTotal edges in training graph topology: {hetero_graph.num_edges}")
    # 打印每种结构边的数量
    for edge_type in hetero_graph.edge_types:
        print(f"  - Edges of type '{edge_type}': {hetero_graph[edge_type].num_edges}")

    print(
        f"\nShape of train_edge_label_index (for loss calculation): {train_edge_label_index.shape}"
    )
    print(f"Shape of train_edge_label (for loss calculation): {train_edge_label.shape}")

    # 验证与原始DataFrame的行数是否匹配
    assert train_edge_label_index.shape[1] == len(train_df)
    assert train_edge_label.shape[0] == len(train_df)

    print(
        f"\nShape of test_edge_label_index (for evaluation): {test_edge_label_index.shape}"
    )
    print(f"Number of test_labels (for evaluation): {len(test_labels)}")
    assert test_edge_label_index.shape[1] == len(test_df)
    print("=" * 50 + "\n")

    # --- 4. 训练循环 (现在是按Epoch，而不是按Batch) ---
    print("--> Starting full-batch model training...")
    for epoch in tqdm(range(config.training.epochs), desc="Epochs"):
        model.train()
        optimizer.zero_grad()

        # [核心调用] 直接调用我们之前写的get_loss，非常简单
        loss = model.get_loss(hetero_graph, train_edge_label_index, train_edge_label)

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    # --- 5. 评估 ---
    print("--> Starting model evaluation...")
    model.eval()
    with torch.no_grad():
        # [核心调用] 直接调用我们之前写的inference
        preds = model.inference(hetero_graph, test_edge_label_index)
        preds = preds.cpu().numpy()

    auc = roc_auc_score(test_labels, preds)
    aupr = average_precision_score(test_labels, preds)

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
<<<<<<< HEAD
                config,
                fold_idx,
=======
                config, fold_idx, global_to_local_maps
>>>>>>> 5b499c6 (data_proc采用了辅助函数,以及归纳式训练,现在又是0.5了)
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
