import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from joblib import Parallel, delayed

# 从通用模板库导入
import research_template.path_manager as rt


class GBDT_Link_Predictor:
    """
    Performs link prediction using a Gradient Boosting model on pre-computed node embeddings.
    This class encapsulates the K-Fold cross-validation and evaluation process.
    """

    def __init__(self, config: dict):
        """
        Initializes the predictor with configuration.
        """
        self.config = config
        self.params = config.predictor
        self.runtime_params = config["runtime"]
        self.training_params = config["training"]

    @staticmethod
    def _train_and_evaluate_warm_start_fold(
        model, X_train, y_train, X_test, y_test, fold_num, k_folds
    ):
        """Helper function to process a single fold of cross-validation."""
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        fold_auc = roc_auc_score(y_test, y_pred_proba)
        fold_aupr = average_precision_score(y_test, y_pred_proba)

        # We can still print progress from within the parallel jobs
        print(
            f"    Fold {fold_num}/{k_folds} - AUC: {fold_auc:.4f}, AUPR: {fold_aupr:.4f}"
        )

        return fold_auc, fold_aupr

    def predict(self, node_embeddings: pd.DataFrame):
        split_mode = self.config.params.split_config.mode

        if split_mode == "random":
            print("--- [Predictor] Starting WARM-START Cross-Validation Workflow ---")
            return self._run_warm_start_cv(node_embeddings)
        elif split_mode in ["drug", "protein", "pair"]:
            print(
                f"--- [Predictor] Starting COLD-START ('{split_mode}') Evaluation Workflow ---"
            )
            return self._run_cold_start_eval(node_embeddings)
        else:
            raise ValueError(f"Unknown split_mode '{split_mode}' in config.")

    def _run_warm_start_cv(self, node_embeddings):
        k_folds = self.training_params["k_folds"]
        seed = self.runtime_params["seed"]

        # 1. [MODERNIZED] Load labeled edges from the unified file
        print("--> Loading link prediction labels...")
        labeled_edges_path = rt.get_path(
            self.config, "processed.link_prediction_labels"
        )
        labeled_edges_df = pd.read_csv(labeled_edges_path)

        X_pairs = labeled_edges_df[["source", "target"]].values
        y_labels = labeled_edges_df["label"].values

        # 2. [CORE LOGIC] Construct the feature matrix X for the classifier
        print("--> Constructing feature matrix for the classifier...")
        # Use .iloc for integer-location based indexing, which is faster and safer
        # Assuming node_embeddings DataFrame index is the node ID
        X_source = node_embeddings.iloc[X_pairs[:, 0]].values
        X_target = node_embeddings.iloc[X_pairs[:, 1]].values
        X_features = np.concatenate([X_source, X_target], axis=1)
        print(f"--> Constructed training data X with shape: {X_features.shape}")

        # 3. [CORE LOGIC] Perform Stratified K-Fold Cross-Validation
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

        aucs, auprs = [], []

        print(f"--> Starting {k_folds}-Fold Cross-Validation...")

        model = GradientBoostingClassifier(
            n_estimators=self.params["n_estimators"],
            max_depth=self.params["max_depth"],
            subsample=self.params.get("subsample"),  # Use .get for optional params
            learning_rate=self.params.get("learning_rate", 0.1),
            random_state=seed,
            verbose=1,
        )
        print(model.get_params())
        parallel_results = Parallel(n_jobs=self.runtime_params["cpus"])(
            delayed(self._train_and_evaluate_warm_start_fold)(
                model,
                X_features[train_idx],
                y_labels[train_idx],
                X_features[test_idx],
                y_labels[test_idx],
                fold + 1,
                k_folds,
            )
            for fold, (train_idx, test_idx) in enumerate(
                skf.split(X_features, y_labels)
            )
        )
        aucs, auprs = zip(*parallel_results)
        aucs = list(aucs)
        auprs = list(auprs)
        # 4. Print final aggregated results
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        mean_aupr = np.mean(auprs)
        std_aupr = np.std(auprs)

        print("\n" + "=" * 50)
        print("    GBDT Link Prediction Final Results")
        print("=" * 50)
        print(f"Mean AUC over {k_folds} folds: {mean_auc:.4f} ± {std_auc:.4f}")
        print(f"Mean AUPR over {k_folds} folds: {mean_aupr:.4f} ± {std_aupr:.4f}")
        print("=" * 50)
        final_results = {
            "aucs": aucs,
            "auprs": auprs,
            "mean_auc": mean_auc,
            "mean_aupr": mean_aupr,
            "std_auc": std_auc,
            "std_aupr": std_aupr,
        }

        # Return this dictionary so the main function can receive it
        return final_results

    def _run_cold_start_eval(self, node_embeddings: pd.DataFrame):
        """
        Performs link prediction on a pre-defined cold-start train/test split.
        """
        seed = self.runtime_params.seed
        split_config = self.config.params.split_config

        # 1. Load the pre-split train and test labeled edge files
        print("--> Loading pre-split train and test sets for cold-start evaluation...")
        train_path = rt.get_path(
            self.config,
            "processed.link_prediction_labels_template",
            split_suffix=split_config.train_file_suffix,
        )
        test_path = rt.get_path(
            self.config,
            "processed.link_prediction_labels_template",
            split_suffix=split_config.test_file_suffix,
        )

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        # 2. Construct feature matrices for TRAIN and TEST sets separately
        print("--> Constructing feature matrices...")

        # Training features
        X_train_pairs = train_df[["source", "target"]].values
        y_train = train_df["label"].values
        X_train_source = node_embeddings.iloc[X_train_pairs[:, 0]].values
        X_train_target = node_embeddings.iloc[X_train_pairs[:, 1]].values
        X_train = np.concatenate([X_train_source, X_train_target], axis=1)

        # Test features
        X_test_pairs = test_df[["source", "target"]].values
        y_test = test_df["label"].values
        X_test_source = node_embeddings.iloc[X_test_pairs[:, 0]].values
        X_test_target = node_embeddings.iloc[X_test_pairs[:, 1]].values
        X_test = np.concatenate([X_test_source, X_test_target], axis=1)

        print(f"--> Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        # 3. Train a SINGLE model on the entire training set
        print("--> Training GBDT model on the entire cold-start training set...")
        model = GradientBoostingClassifier(
            # ... (get parameters from self.params as before) ...
            random_state=seed,
            verbose=1,
        )
        model.fit(X_train, y_train)

        # 4. Evaluate on the cold test set
        print("--> Evaluating model on the cold-start test set...")
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        auc_score = roc_auc_score(y_test, y_pred_proba)
        aupr_score = average_precision_score(y_test, y_pred_proba)

        print("\n" + "=" * 50)
        print("    GBDT Link Prediction COLD-START Results")
        print("=" * 50)
        print(f"AUC on cold test set: {auc_score:.4f}")
        print(f"AUPR on cold test set: {aupr_score:.4f}")
        print("=" * 50)

        # The result is a single set of scores, not a list from CV
        final_results = {
            "aucs": [auc_score],
            "auprs": [aupr_score],
            "mean_auc": auc_score,
            "mean_aupr": aupr_score,
            "std_auc": 0.0,
            "std_aupr": 0.0,
        }
        return final_results
