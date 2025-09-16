import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from joblib import Parallel, delayed

# 从通用模板库导入
from research_template.path_manager import get_path


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
    def _train_and_evaluate_fold(
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
        """
        The main method to run the entire link prediction cross-validation pipeline.

        Args:
            node_embeddings (pd.DataFrame): A DataFrame of node embeddings, where the index
                                            corresponds to the global node ID.
        """
        print(
            "--- [Predictor] Running GBDT Cross-Validation for Link Prediction... ---"
        )

        k_folds = self.training_params["k_folds"]
        seed = self.runtime_params["seed"]

        # 1. [MODERNIZED] Load labeled edges from the unified file
        print("--> Loading link prediction labels...")
        labeled_edges_path = get_path(self.config, "processed.link_prediction_labels")
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
            subsample=self.params.get("subsample", 1.0),  # Use .get for optional params
            learning_rate=self.params.get("learning_rate", 0.1),
            random_state=seed,
            verbose=1,
        )
        parallel_results = Parallel(n_jobs=self.runtime_params["cpus"])(
            delayed(self._train_and_evaluate_fold)(
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
