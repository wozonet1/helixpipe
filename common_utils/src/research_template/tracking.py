import mlflow
from pathlib import Path
from omegaconf import OmegaConf


class MLflowTracker:
    """
    A wrapper class for MLflow to standardize experiment tracking.
    """

    def __init__(self, config: dict):
        # ... (This part is unchanged) ...
        self.config = config
        mlflow_config = config.get("mlflow", {})
        if not mlflow_config:
            self.is_active = False
            return
        # 1. [NEW] Construct the custom run name
        try:
            # Get key parameters from config
            data_variant = "gtopdb" if config.data.use_gtopdb else "baseline"
            encoder = config.encoder.name

            # Combine them into a descriptive name
            custom_run_name = f"{config.data.primary_dataset}_{data_variant}-{config.relations.name}-{encoder}"

        except Exception:
            # Fallback to default naming if config structure is unexpected
            custom_run_name = "config_error"
        self.custom_run_name = custom_run_name
        tracking_uri = mlflow_config.get("tracking_uri", "mlruns")
        tracking_path = Path(tracking_uri)

        # 1. Check if the provided path is already absolute
        if not tracking_path.is_absolute():
            # 2. If it's relative, resolve it relative to the current working directory
            #    (which should be the project root)
            tracking_path = Path.cwd() / tracking_path

        # 3. Now that we have a guaranteed absolute path, we can safely convert it to a URI
        mlflow.set_tracking_uri(tracking_path.as_uri())
        experiment_name = mlflow_config.get("experiment_name", "Default Experiment")
        mlflow.set_experiment(experiment_name)
        self.is_active = True
        print(
            f"--> MLflowTracker initialized. Logging to experiment: '{experiment_name}'"
        )

    def _log_active_workflow_params(self):
        """
        [REFACTORED] A centralized and smart way to log all relevant parameters from a
        flat, component-based Hydra config.
        """
        print("--> Logging parameters to MLflow...")

        # --- 1. Log simple, top-level parameter blocks ---
        # These are blocks with simple key-value pairs
        mlflow.log_params(OmegaConf.to_container(self.config.runtime, resolve=True))
        mlflow.log_params(OmegaConf.to_container(self.config.params, resolve=True))
        mlflow.log_params({"k_folds": self.config.training.k_folds})

        # --- 2. Log key identifiers of the experiment ---
        # This section defines the "what" of the experiment

        # Determine the data variant for clear logging
        data_variant = "gtopdb" if self.config.data.use_gtopdb else "baseline"

        # Check if a predictor is defined to determine the paradigm
        paradigm = self.config.training.paradigm
        predictor_name = (
            self.config.predictor.name if hasattr(self.config, "predictor") else "N/A"
        )

        workflow_identifiers = {
            "dataset": self.config.data.primary_dataset,
            "data_variant": data_variant,
            "paradigm": paradigm,
            "encoder": self.config.encoder.name,
            "predictor": predictor_name,
        }
        mlflow.log_params(workflow_identifiers)

        # --- 3. Log the 'include_relations' switches with a prefix ---
        # Correctly access the relations switches from the composed config
        mlflow.log_param("relations", self.config.relations.name)

        # --- 4. Log hyperparameters for the ACTIVE components ---
        # This is the most significant change: we access component configs directly at the top level.

        # Log Encoder Hyperparameters
        # OmegaConf.to_container converts the config object to a plain dict
        encoder_params = OmegaConf.to_container(self.config.encoder, resolve=True)
        encoder_params.pop("name", None)  # Remove the name key as it's already logged
        mlflow.log_params({f"enc_{k}": v for k, v in encoder_params.items()})

        # Log Predictor Hyperparameters (only if a predictor is defined)
        if hasattr(self.config, "predictor"):
            predictor_params = OmegaConf.to_container(
                self.config.predictor, resolve=True
            )
            predictor_params.pop("name", None)
            mlflow.log_params({f"prd_{k}": v for k, v in predictor_params.items()})

        print("--> Parameter logging complete.")

    def start_run(self):
        """
        Starts a new MLflow run and logs all relevant parameters.
        This method is now much cleaner.
        """
        if not self.is_active:
            return

        mlflow.start_run(run_name=self.custom_run_name)
        print(f"--> MLflow run started. Name: '{self.custom_run_name}'")

        # [MODIFIED] Simply call the centralized logging method
        self._log_active_workflow_params()

    def log_cv_results(self, aucs: list, auprs: list):
        """
        Logs the results from a cross-validation run.
        It logs the mean and standard deviation of the metrics.

        Args:
            aucs (list): A list of AUC scores from each fold.
            auprs (list): A list of AUPR scores from each fold.
        """
        if not self.is_active:
            return

        import numpy as np

        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        mean_aupr = np.mean(auprs)
        std_aupr = np.std(auprs)

        print("--> Logging final metrics to MLflow...")
        mlflow.log_metric("mean_auc", mean_auc)
        mlflow.log_metric("std_auc", std_auc)
        mlflow.log_metric("mean_aupr", mean_aupr)
        mlflow.log_metric("std_aupr", std_aupr)

    def end_run(self):
        """Ends the current MLflow run."""
        if not self.is_active:
            return

        mlflow.end_run()
        print("--> MLflow run ended.")
