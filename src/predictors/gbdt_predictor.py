import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

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
        self.params = config["training"]["predictors"]["gbdt"]
        self.runtime_params = config["runtime"]
        self.data_params = config["data"]
        self.training_params = config["training"]

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
        primary_dataset = self.data_params["primary_dataset"]

        # 1. [MODERNIZED] Load labeled edges from the unified file
        print("--> Loading link prediction labels...")
        labeled_edges_path = get_path(
            self.config, f"{primary_dataset}.processed.link_prediction_labels"
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
        for fold, (train_idx, test_idx) in enumerate(
            tqdm(skf.split(X_features, y_labels), total=k_folds, desc="CV Folds")
        ):
            X_train, X_test = X_features[train_idx], X_features[test_idx]
            y_train, y_test = y_labels[train_idx], y_labels[test_idx]

            model = GradientBoostingClassifier(
                n_estimators=self.params["n_estimators"],
                max_depth=self.params["max_depth"],
                subsample=self.params.get(
                    "subsample", 1.0
                ),  # Use .get for optional params
                learning_rate=self.params.get("learning_rate", 0.1),
                random_state=seed,
                verbose=0,
            )

            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            fold_auc = roc_auc_score(y_test, y_pred_proba)
            fold_aupr = average_precision_score(y_test, y_pred_proba)
            print(
                f"    Fold {fold + 1}/{k_folds} - AUC: {fold_auc:.4f}, AUPR: {fold_aupr:.4f}"
            )
            aucs.append(fold_auc)
            auprs.append(fold_aupr)

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
