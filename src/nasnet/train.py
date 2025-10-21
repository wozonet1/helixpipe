import traceback

import hydra  # noqa: F401
import mlflow
import numpy as np
import research_template as rt
import torch
import torch.nn.functional as F
import torch.utils.data
from data_utils.debug_utils import run_optional_diagnostics
from data_utils.preparers import prepare_e2e_data
from omegaconf import DictConfig
from predictors.rgcn_link_predictor import RGCNLinkPredictor
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from tqdm import tqdm

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
        print(f"Evaluation Results - AUC: {auc:.4f}, AUPR: {aupr:.4f}")
        return auc, aupr


def run_end_to_end_workflow(
    config: DictConfig,
    hetero_graph: HeteroData,
    train_edge_label_index: torch.Tensor,
    test_edge_label_index: torch.Tensor,
    test_labels: torch.Tensor,
    device: torch.device,
    tracker: rt.MLflowTracker,
    fold_idx: int,
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
    in_channels_dict = {
        node_type: hetero_graph[node_type].x.shape[1]
        for node_type in hetero_graph.node_types
    }
    model = RGCNLinkPredictor(
        in_channels_dict=in_channels_dict,
        hidden_channels=config.predictor.params.hidden_channels,
        out_channels=config.predictor.params.out_channels,
        num_layers=config.predictor.params.num_layers,
        dropout=config.predictor.params.dropout,
        metadata=hetero_graph.metadata(),
    ).to(device)
    print("--> Model instantiation successful!")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    warmup_scheduler = LinearLR(
        optimizer, start_factor=1e-4, end_factor=1.0, total_iters=5
    )

    # b. 主调度器：在剩余的 (total_epochs - 5) 个epoch内，按余弦曲线衰减
    main_scheduler = CosineAnnealingLR(optimizer, T_max=config.training.epochs - 5)

    # c. 将它们组合成一个顺序调度器
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[5]
    )
    # --- 2. 实例化数据加载器 ---
    print("--> Initializing LinkNeighborLoader in single-process mode...")
    train_loader = LinkNeighborLoader(
        data=hetero_graph,
        num_neighbors=[-1] * config.predictor.params.num_layers,
        edge_label_index=(target_edge_type, train_edge_label_index),
        batch_size=config.training.get("batch_size", 512),
        shuffle=True,
        neg_sampling_ratio=1.0,
        num_workers=config.runtime.get("trian_loader_cpus", 8),
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
        num_workers=config.runtime.get("test_loader_cpus", 4),
    )
    # --- 6. 训练循环 ---
    print("--> Starting mini-batch model training...")
    best_val_aupr = 0
    best_epoch_results = {}
    epoch_pbar = tqdm(
        range(config.training.epochs), desc="Starting Training", leave=False
    )
    validation_freq = config.runtime.get("validate_every_n_epochs", 10)
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
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        tracker.log_training_metric(
            epoch=epoch,
            value=current_lr,
            fold_idx=fold_idx,
            metric_name="learning_rate",
        )  # 记录学习率
        avg_loss = total_loss / len(train_loader.dataset)
        tracker.log_training_metric(
            epoch=epoch, value=avg_loss, fold_idx=fold_idx, metric_name="loss"
        )
        epoch_pbar.set_description(f"Epoch {epoch:03d} | Avg Loss: {avg_loss:.4f}")

        # --- 7. 评估 ---
        if (epoch + 1) % validation_freq == 0:
            # 调用我们新建的评估函数
            val_auc, val_aupr = evaluate(model, test_loader, device, target_edge_type)
            tracker.log_validation_metrics(
                epoch=epoch, auc=val_auc, aupr=val_aupr, fold_idx=fold_idx
            )
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
    tracker.log_best_fold_result(best_epoch_results, fold_idx)


# end region

# region main


def train(config: DictConfig):
    tracker = rt.MLflowTracker(config)
    # --- Start of protected block for MLflow ---
    try:
        tracker.start_run()
        device = torch.device(
            config["runtime"]["gpu"] if torch.cuda.is_available() else "cpu"
        )
        fold_idx = config.runtime.fold_idx
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
        print(f"  - Cold Start:          {config.training.coldstart.mode}")
        print("=" * 80 + "\n")

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
        run_end_to_end_workflow(
            config=config,
            hetero_graph=hetero_graph,
            train_edge_label_index=train_edge_label_index,
            test_edge_label_index=test_edge_label_index,
            test_labels=test_labels,
            device=device,
            tracker=tracker,
            fold_idx=fold_idx,
        )

    except Exception as e:
        # ... (error handling remains the same) ...
        print(f"\n!!! FATAL ERROR: Experiment run failed: {e}")
        if tracker.is_active:
            mlflow.set_tag("run_status", "FAILED")
            mlflow.log_text(traceback.format_exc(), "error_traceback.txt")
        raise
    finally:
        tracker.end_run()
