import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
)  # 使用LightGBM，通常比原生GBDT更快性能更好
from sklearn.metrics import roc_auc_score, average_precision_score


class GBDT_Link_Predictor:
    """
    使用梯度提升模型（LightGBM）在预先计算的节点嵌入上执行链接预测。

    这个类被设计为在 K-Fold 交叉验证循环的单一一折中被调用。
    它封装了特征矩阵构建、模型训练和对单一一折数据的评估。
    """

    def __init__(self, config: dict):
        """
        使用配置初始化预测器。

        Args:
            config (dict): 来自 Hydra 的完整配置对象。
        """
        self.config = config
        # 从 predictor 配置组中获取模型超参数
        self.params = config.predictor
        # 从 runtime 配置组中获取运行时参数（如 seed）
        self.runtime_params = config.runtime

    def predict(
        self, node_embeddings: np.ndarray, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> dict:
        """
        为单一一折（Fold）执行训练、预测和评估。

        这个方法是该类的唯一入口点，由 train.py 中的 K-fold 循环调用。

        Args:
            node_embeddings (np.ndarray): 由上游编码器（如 NDLS）生成的节点嵌入矩阵。
                                          其索引应与全局节点ID对应。
            train_df (pd.DataFrame): 包含 ['source', 'target', 'label'] 的训练集。
            test_df (pd.DataFrame): 包含 ['source', 'target', 'label'] 的测试/验证集。

        Returns:
            dict: 一个包含该折评估结果的字典，例如 {'auc': 0.85, 'aupr': 0.75}。
        """
        print("--- [Predictor] Starting GBDT prediction for the current fold ---")

        # 1. 为训练集和测试集分别构建特征矩阵
        print("--> Constructing feature matrices from node embeddings...")
        X_train, y_train = self._create_feature_matrix(node_embeddings, train_df)
        X_test, y_test = self._create_feature_matrix(node_embeddings, test_df)

        print(f"--> Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        # 2. 初始化并训练 LGBM 模型
        # 参数从 conf/predictor/gbdt.yaml 中读取
        print(
            "--> Training GradientBoostingClassifier model on the training set for this fold..."
        )
        model = GradientBoostingClassifier(
            n_estimators=self.params.get("n_estimators", 100),
            max_depth=self.params.get("max_depth", 7),
            subsample=self.params.get("subsample", 1.0),
            learning_rate=self.params.get("learning_rate", 0.1),
            random_state=self.runtime_params.seed,
            verbose=1,
        )
        model.fit(X_train, y_train)

        # 3. 在测试集上进行评估
        print("--> Evaluating model on the test set for this fold...")
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # 4. 计算评估指标
        auc_score = roc_auc_score(y_test, y_pred_proba)
        aupr_score = average_precision_score(y_test, y_pred_proba)

        # 5. 返回包含该折结果的字典
        # train.py 中的主循环将会收集这些结果
        return {"auc": auc_score, "aupr": aupr_score}

    @staticmethod
    def _create_feature_matrix(
        node_embeddings: np.ndarray, edges_df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        一个静态辅助方法，用于从节点嵌入和边列表中创建特征矩阵和标签。

        Args:
            node_embeddings (np.ndarray): 完整的节点嵌入矩阵。
            edges_df (pd.DataFrame): 包含 source, target, label 的数据框。

        Returns:
            tuple[np.ndarray, np.ndarray]: (特征矩阵 X, 标签向量 y)
        """
        # 提取 source 和 target 节点的全局ID
        pairs = edges_df[["source", "target"]].values
        labels = edges_df["label"].values

        # 从嵌入矩阵中查找对应节点的嵌入向量
        # 这是一个高效的 NumPy 索引操作
        source_embeds = node_embeddings[pairs[:, 0]]
        target_embeds = node_embeddings[pairs[:, 1]]

        # 将 source 和 target 的嵌入拼接起来，构成边的特征
        # 这是链接预测中的一个标准做法
        features = np.concatenate([source_embeds, target_embeds], axis=1)

        return features, labels
