import numpy as np
import torch.nn.functional as F
from torch_geometric.data import HeteroData
import torch
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
from data_utils.debug_utils import run_optional_diagnostics
from data_utils.preparers import prepare_e2e_data

target_edge_type = ("drug", "drug_protein_interaction", "protein")


@torch.no_grad()  # 使用装饰器，确保函数内所有操作不计算梯度
def evaluate(
    model,
    loader,
    device,
    target_edge_type,
):
    """
    Performs evaluation on a given data loader and returns performance metrics.
    """
    model.eval()  # 切换到评估模式
    all_preds = []
    all_labels = []

    for batch in loader:  # 这里可以加一个小的tqdm
        batch = batch.to(device)
        z_dict = model.forward(batch.x_dict, batch.edge_index_dict)
        scores = model.decode(z_dict, batch[target_edge_type].edge_label_index)

        all_preds.append(torch.sigmoid(scores).cpu())
        all_labels.append(batch[target_edge_type].edge_label.cpu())

    preds = torch.cat(all_preds, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()

    if len(np.unique(labels)) < 2:
        return 0.0, 0.0  # 返回默认值
    else:
        auc = roc_auc_score(labels, preds)
        aupr = average_precision_score(labels, preds)
        return auc, aupr


def run_end_to_end_workflow(
    config: DictConfig,
    hetero_graph: HeteroData,  # <-- 已经是净化过的、使用局部ID的图
    train_edge_label_index: torch.Tensor,  # <-- 已经是局部ID
    test_edge_label_index: torch.Tensor,  # <-- 已经是局部ID
    test_labels: torch.Tensor,
    device: torch.device,
    tracker: rt.MLflowTracker,
):
    """
    【新版】执行端到端工作流。
    假定所有传入的数据对象都已经是经过预处理的、可以直接使用的Tensor。
    """
    print("\n--- Running Cleaned End-to-End Workflow ---")
    target_edge_type = (
        "drug",
        "drug_protein_interaction",
        "protein",
    )  # 同样可以从config读取

    # --- 1. 模型实例化 ---
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

    # --- 2. 实例化数据加载器 ---
    print("--> Initializing LinkNeighborLoader in single-process mode...")
    train_loader = LinkNeighborLoader(
        data=hetero_graph,
        num_neighbors=[-1] * config.predictor.params.num_layers,
        edge_label_index=(target_edge_type, train_edge_label_index),
        batch_size=config.training.get("batch_size", 512),  # TODO:确定合适
        shuffle=True,
        neg_sampling_ratio=1.0,
        num_workers=config.training.get("trian_loader_cpus", 8),
    )
    test_loader = LinkNeighborLoader(
        data=hetero_graph,
        num_neighbors=[-1] * config.predictor.params.num_layers,
        edge_label_index=(
            target_edge_type,
            test_edge_label_index,
        ),
        edge_label=test_labels,
        batch_size=config.training.get("batch_size", 512),
        shuffle=False,
        neg_sampling_ratio=0.0,
        num_workers=config.training.get("test_loader_cpus", 4),
    )
    # --- 6. 训练循环 ---
    print("--> Starting mini-batch model training...")
    best_val_aupr = 0
    best_epoch_results = {}
    epoch_pbar = tqdm(range(config.training.epochs), desc="Starting Training")
    validation_freq = config.training.get("validate_every_n_epochs", 10)
    for epoch in epoch_pbar:
        model.train()
        total_loss = 0
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            z_dict = model.forward(batch.x_dict, batch.edge_index_dict)
            scores = model.decode(z_dict, batch[target_edge_type].edge_label_index)
            labels = batch[target_edge_type].edge_label.float()

            loss = F.binary_cross_entropy_with_logits(scores, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * scores.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        tracker.log_training_metric(epoch=epoch, loss=avg_loss)
        epoch_pbar.set_description(f"Epoch {epoch:03d} | Avg Loss: {avg_loss:.4f}")

        # --- 7. 评估 ---
        if (epoch + 1) % validation_freq == 0:
            # 调用我们新建的评估函数
            val_auc, val_aupr = evaluate(model, test_loader, device, target_edge_type)
            tracker.log_validation_metrics(epoch=epoch, auc=val_auc, aupr=val_aupr)
            # 更新进度条，现在包含验证集分数
            epoch_pbar.set_description(
                f"Epoch {epoch:03d} | Avg Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f} | Val AUPR: {val_aupr:.4f}"
            )

            # 检查并保存最佳模型的结果
            if val_aupr > best_val_aupr:
                best_val_aupr = val_aupr
                best_epoch_results = {"auc": val_auc, "aupr": val_aupr, "epoch": epoch}
                # (可选) 在这里可以保存模型权重
                # torch.save(model.state_dict(), 'best_model.pth')
        else:
            # 如果不是验证周期，只更新loss
            epoch_pbar.set_description(f"Epoch {epoch:03d} | Avg Loss: {avg_loss:.4f}")
    print("\n--- Training Finished ---")
    print(
        f"Best validation AUPR: {best_epoch_results.get('aupr', 0):.4f} at epoch {best_epoch_results.get('epoch', -1)}"
    )

    # 返回在验证集上性能最好的那个epoch的结果
    return best_epoch_results


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

        predictor_name = config.predictor.name
        data_variant = "gtopdb" if config.data.use_gtopdb else "baseline"

        # 3. Startup Log
        print("\n" + "=" * 80)
        print(" " * 20 + "Starting DTI Prediction Experiment")
        print("=" * 80)
        print("Configuration loaded for this run:")
        print(f"  - Primary Dataset:     '{config['data']['primary_dataset']}'")
        print(f"  - Data Variant:        '{data_variant}'")
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
            (
                hetero_graph,
                train_edge_label_index,
                test_edge_label_index,
                test_labels,
                maps,  # maps可能不再需要，取决于run_workflow的实现
            ) = prepare_e2e_data(config, fold_idx)
            # [修改3] (可选) 一行代码，运行所有诊断
            if config.runtime.get("run_diagnostics", False):
                # 假设run_optional_diagnostics现在只需要graph
                run_optional_diagnostics(hetero_graph)

            # [修改4] 调用工作流函数，传递已经准备好的数据
            # 注意：我们现在传递的是已经处理好的、可以直接使用的Tensor
            fold_results = run_end_to_end_workflow(
                config=config,
                hetero_graph=hetero_graph,
                train_edge_label_index=train_edge_label_index,
                test_edge_label_index=test_edge_label_index,
                test_labels=test_labels,
                device=device,
                tracker=tracker,
            )

            if fold_results:
                all_fold_aucs.append(fold_results["auc"])
                all_fold_auprs.append(fold_results["aupr"])

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
