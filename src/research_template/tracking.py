# ==============================================================================
#                 Experiment Tracker for MLflow (Final Version)
# ==============================================================================
#
# 职责:
#   - 封装所有与MLflow交互的逻辑。
#   - 自动记录Hydra配置参数。
#   - 提供用于记录时间序列指标（用于绘图）的方法。
#   - 内部聚合多折交叉验证的结果。
#   - 在运行结束时，自动计算并记录最终的CV摘要指标作为参数。
#
# ==============================================================================

from pathlib import Path

import mlflow
import numpy as np
from flatdict import FlatDict
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf


class MLflowTracker:
    """
    【最终版】一个为MLflow设计的、有状态的、与Hydra配置完全同步的包装类。
    """

    def __init__(self, config: DictConfig):
        """
        初始化追踪器，并为内部状态聚合做准备。
        """
        self.config = config
        mlflow_config = config.get("mlflow")
        if not mlflow_config:
            self.is_active = False
            return

        # --- 运行名称生成 ---
        try:
            # 一个能自动反映核心配置的、动态的运行名称
            self.custom_run_name = (
                f"{config.data_params.primary_dataset}_"
                f"{config.relations.name}_"
                f"{config.training.coldstart.mode}"
            )
        except Exception:
            self.custom_run_name = "configuration_error"

        # --- [核心] 初始化内部状态，用于聚合多折交叉验证的结果 ---
        self._all_fold_best_aucs: list[float] = []
        self._all_fold_best_auprs: list[float] = []

        # --- MLflow环境设置 ---
        project_root = Path(get_original_cwd())
        absolute_tracking_path = project_root / mlflow_config.tracking_uri
        final_tracking_uri = f"file://{str(absolute_tracking_path)}"
        mlflow.set_tracking_uri(final_tracking_uri)
        mlflow.set_experiment(mlflow_config.experiment_name)
        self.is_active = True
        print(
            f"--> MLflowTracker initialized for experiment: '{mlflow_config.experiment_name}'"
        )

    def _log_active_workflow_params(self):
        """
        【最终版】分块、带前缀地记录所有Hydra配置参数，完美映射您的YAML文件结构。
        """
        if not self.is_active:
            return

        print("--> Logging all workflow parameters to MLflow...")

        # --- 1. 记录运行时参数 ---
        # 例如: runtime_seed, runtime_cpus, runtime_gpu
        if hasattr(self.config, "runtime"):
            runtime_params = OmegaConf.to_container(self.config.runtime, resolve=True)
            mlflow.log_params({f"runtime_{k}": v for k, v in runtime_params.items()})

        # --- 2. 记录数据处理的核心参数 (来自 params/*.yaml) ---
        # 例如: data_proc_similarity_thresholds.protein_protein
        if hasattr(self.config, "params"):
            data_proc_params = OmegaConf.to_container(self.config.params, resolve=True)
            flat_proc_params = FlatDict(data_proc_params, delimiter=".")
            mlflow.log_params({f"param_{k}": v for k, v in flat_proc_params.items()})

        # --- 3. 记录完整的训练与评估方案 (来自顶层的 training 块) ---
        # 例如: train_learning_rate, train_evaluation.mode
        if hasattr(self.config, "training"):
            training_params = OmegaConf.to_container(self.config.training, resolve=True)
            flat_training_params = FlatDict(training_params, delimiter=".")
            mlflow.log_params(
                {f"train_{k}": v for k, v in flat_training_params.items()}
            )

        # --- 4. 记录核心组件的超参数 (Predictor, a.k.a. Model) ---
        # 例如: prd_hidden_channels, prd_num_heads
        if hasattr(self.config, "predictor"):
            # 我们通常关心的是 predictor.params 里的内容
            if hasattr(self.config.predictor, "params"):
                predictor_params = OmegaConf.to_container(
                    self.config.predictor.params, resolve=True
                )
                # 移除PyG实例化需要，但日志中无用的_target_
                predictor_params.pop("_target_", None)
                mlflow.log_params({f"prd_{k}": v for k, v in predictor_params.items()})

        # --- 5. 记录最高层级的、用于识别实验身份的核心“标签” ---
        # 这些是没有前缀的，因为它们是用于筛选和分组的核心标识符
        identifiers = {
            "use_gtopdb": self.config.data.use_gtopdb,
            "relations": self.config.relations.name,
            "eval_mode": self.config.training.coldstart.mode,
            "model_name": self.config.predictor.name,
        }
        mlflow.log_params(identifiers)

        print("--> Parameter logging complete.")

    def start_run(self):
        """
        启动一个新的MLflow运行，并记录所有参数。
        """
        if not self.is_active:
            return
        mlflow.start_run(run_name=self.custom_run_name)
        print(f"--> MLflow run started. Name: '{self.custom_run_name}'")
        self._log_active_workflow_params()

    # --- 时间序列指标记录 (用于绘制曲线) ---

    def log_training_metric(
        self, epoch: int, value: float, fold_idx: int, metric_name: str = "loss"
    ):
        """
        记录每个epoch的某个metric。
        """
        if not self.is_active:
            return
        mlflow.log_metric(
            key=f"fold_{fold_idx}_train_{metric_name}", value=value, step=epoch
        )

    def log_validation_metrics(
        self, epoch: int, auc: float, aupr: float, fold_idx: int
    ):
        """
        记录每个验证周期的验证集性能。
        """
        if not self.is_active:
            return
        mlflow.log_metric(key=f"fold_{fold_idx}_val_auc", value=auc, step=epoch)
        mlflow.log_metric(key=f"fold_{fold_idx}_val_aupr", value=aupr, step=epoch)

    # --- 摘要/最终结果记录 (作为参数，保持Metrics视图整洁) ---

    def log_best_fold_result(self, fold_results: dict, fold_idx: int):
        """
        记录单一一折的最佳性能结果到内部列表，并将其作为【参数】记录到MLflow。
        """
        if not self.is_active or not fold_results:
            return

        best_auc = fold_results.get("auc", 0)
        best_aupr = fold_results.get("aupr", 0)
        best_epoch = fold_results.get("epoch", -1)

        # 1. 在内部聚合结果
        self._all_fold_best_aucs.append(best_auc)
        self._all_fold_best_auprs.append(best_aupr)

        # 2. 将单折的最佳结果，作为独立的【参数】记录
        # 使用'z_fold_'前缀进行排序和组织
        mlflow.log_param(f"z_fold_{fold_idx}_best_auc", f"{best_auc:.4f}")
        mlflow.log_param(f"z_fold_{fold_idx}_best_aupr", f"{best_aupr:.4f}")
        mlflow.log_param(f"z_fold_{fold_idx}_best_epoch", best_epoch)

        print(f"--> Logged Best Result for Fold {fold_idx} as parameters.")

    def _log_cv_summary_results(self):
        """
        【内部方法】计算并记录交叉验证的最终摘要结果，作为【参数】。
        """
        if not self.is_active or not self._all_fold_best_aucs:
            return

        print("\n" + "=" * 80)
        print(" " * 25 + "Cross-Validation Summary")
        print("=" * 80)

        mean_auc = np.mean(self._all_fold_best_aucs)
        std_auc = np.std(self._all_fold_best_aucs)
        mean_aupr = np.mean(self._all_fold_best_auprs)
        std_aupr = np.std(self._all_fold_best_auprs)

        print(f"  - Mean AUC:  {mean_auc:.4f} +/- {std_auc:.4f}")
        print(f"  - Mean AUPR: {mean_aupr:.4f} +/- {std_aupr:.4f}")

        print("--> Logging final CV summary metrics to MLflow as PARAMETERS...")
        # 使用'z_cv_'前缀进行排序和组织
        mlflow.log_param("z_cv_mean_auc", f"{mean_auc:.4f}")
        mlflow.log_param("z_cv_std_auc", f"{std_auc:.4f}")
        mlflow.log_param("z_cv_mean_aupr", f"{mean_aupr:.4f}")
        mlflow.log_param("z_cv_std_aupr", f"{std_aupr:.4f}")

    def end_run(self):
        """
        在结束MLflow运行之前，自动计算并记录交叉验证的摘要结果。
        """
        if not self.is_active:
            return

        self._log_cv_summary_results()

        mlflow.end_run()
        print("--> MLflow run ended.")
