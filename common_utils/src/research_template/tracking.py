import mlflow
from pathlib import Path
from datetime import datetime


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
            # Get current timestamp in a clean format
            timestamp = datetime.now().strftime(
                "%Y%m%d-%H%M%S"
            )  # e.g., "20231028-153000"

            # Get key parameters from config
            training_config = self.config.get("training", {})
            data_variant = training_config.get("data_variant", "unknown_data")
            encoder = training_config.get("encoder", "unknown_encoder")

            # Combine them into a descriptive name
            custom_run_name = f"{timestamp}_{data_variant}_{encoder}"

        except Exception:
            # Fallback to default naming if config structure is unexpected
            custom_run_name = f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
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
        [PRIVATE METHOD] A centralized and smart way to log all relevant parameters.
        This method intelligently navigates the config to log only what's needed for the active run.
        """
        print("--> Logging parameters to MLflow...")

        # 1. Log non-training related, general configs
        mlflow.log_params(self.config.get("runtime", {}))
        # We can be more selective about what to log from the huge data config
        data_params_to_log = {
            "primary_dataset": self.config.get("data", {}).get("primary_dataset"),
            "use_gtopdb_in_datagen": self.config.get("data", {}).get("use_gtopdb"),
        }
        mlflow.log_params(data_params_to_log)
        mlflow.log_params(self.config.get("params", {}))  # Log data processing params

        # 2. Get the active workflow's blueprint
        training_config = self.config.get("training", {})
        # --- [MODIFIED] Logic to log based on the new config structure ---

        # 1. Get component names directly from the top level
        encoder_name = training_config.get("encoder")
        predictor_name = training_config.get("predictor")  # Will be None if not present

        # 2. Determine paradigm by convention
        paradigm = "two_stage" if predictor_name else "end_to_end"

        # 3. Build the params_to_log dictionary
        params_to_log = {
            "paradigm": paradigm,
            "data_variant": training_config.get("data_variant"),
            "encoder": encoder_name,
            "predictor": predictor_name or "N/A",  # Use 'N/A' if None
            "k_folds": training_config.get("k_folds"),
        }

        # 4. Dynamically find and add component hyperparameters (logic is the same)
        if encoder_name:
            encoder_params = training_config.get("encoders", {}).get(encoder_name, {})
            params_to_log.update({f"enc_{k}": v for k, v in encoder_params.items()})

        if predictor_name:
            predictor_params = training_config.get("predictors", {}).get(
                predictor_name, {}
            )
            params_to_log.update({f"prd_{k}": v for k, v in predictor_params.items()})

        if paradigm == "end_to_end":
            e2e_params = training_config.get("end_to_end", {})
            params_to_log.update(e2e_params)

        mlflow.log_params(params_to_log)

    def start_run(self):
        """
        Starts a new MLflow run and logs all relevant parameters.
        This method is now much cleaner.
        """
        if not self.is_active:
            return

        mlflow.start_run(run_name=self.cutom_run_name)
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
