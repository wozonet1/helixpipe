from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader

from helixpipe.typing import AppConfig
from helixpipe.utils import get_path


class DataManager:
    """
    数据管理器 (Data Manager)。

    职责: 封装所有与“为一个Fold加载和准备数据”相关的逻辑，
    为 Trainer 提供一个即插即用的数据环境。
    """

    def __init__(self, config: AppConfig, fold_idx: int):
        """
        构造函数只接收依赖，不执行任何耗时操作。
        """
        self._config = config
        self._fold_idx = fold_idx
        self.verbose = config.runtime.verbose

        # 初始化内部状态
        self._hetero_data: HeteroData | None = None
        self._train_loader: LinkNeighborLoader | None = None
        self._test_loader: LinkNeighborLoader | None = None

        if self.verbose > 0:
            print(
                f"\n--- [DataManager] Initialized for Fold {self._fold_idx}. Ready for setup. ---"
            )

    def setup(self):
        """
        执行所有实际的数据加载和预处理操作。
        """
        if self.verbose > 0:
            print(f"--- [DataManager] Running setup for Fold {self._fold_idx}...")

        # 1. 从磁盘加载所有需要的文件
        (nodes_df, features_np, graph_df, train_labels_df, test_labels_df) = (
            self._load_disk_data()
        )

        # 2. 组装成原始的 HeteroData 对象
        hetero_data_raw = self._build_hetero_data(nodes_df, features_np, graph_df)

        # 3. 对图进行预处理/转换
        self._hetero_data = self._preprocess_graph(hetero_data_raw)

        # 4. 创建 DataLoaders
        self._train_loader, self._test_loader = self._create_loaders(
            self._hetero_data, train_labels_df, test_labels_df
        )

        if self.verbose > 0:
            print(f"--- [DataManager] Setup complete for Fold {self._fold_idx}. ---")

    # --- Public Accessors (属性访问器) ---

    @property
    def train_loader(self) -> LinkNeighborLoader:
        if self._train_loader is None:
            raise RuntimeError(
                "DataManager has not been set up. Please call .setup() first."
            )
        return self._train_loader

    @property
    def test_loader(self) -> LinkNeighborLoader:
        if self._test_loader is None:
            raise RuntimeError(
                "DataManager has not been set up. Please call .setup() first."
            )
        return self._test_loader

    @property
    def metadata(self) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        if self._hetero_data is None:
            raise RuntimeError(
                "DataManager has not been set up. Please call .setup() first."
            )
        return self._hetero_data.metadata()

    @property
    def raw_hetero_data(self) -> HeteroData:
        if self._hetero_data is None:
            raise RuntimeError(
                "DataManager has not been set up. Please call .setup() first."
            )
        return self._hetero_data

    # --- Private Implementation (私有实现方法) ---

    def _load_disk_data(self) -> Tuple:
        """负责所有文件I/O。"""
        if self.verbose > 1:
            print("    - Step 1/4: Loading all data files from disk...")

        nodes_df = pd.read_csv(
            get_path(self._config, "processed.common.nodes_metadata")
        )
        features_np = np.load(get_path(self._config, "processed.common.node_features"))

        graph_path_factory = get_path(self._config, "processed.specific.graph_template")
        graph_df = pd.read_csv(
            graph_path_factory(prefix=f"fold_{self._fold_idx}", suffix="train")
        )

        labels_path_factory = get_path(
            self._config, "processed.specific.labels_template"
        )
        train_labels_df = pd.read_csv(
            labels_path_factory(prefix=f"fold_{self._fold_idx}", suffix="train")
        )
        test_labels_df = pd.read_csv(
            labels_path_factory(prefix=f"fold_{self._fold_idx}", suffix="test")
        )

        return nodes_df, features_np, graph_df, train_labels_df, test_labels_df

    def _build_hetero_data(
        self, nodes_df: pd.DataFrame, features_np: np.ndarray, graph_df: pd.DataFrame
    ) -> HeteroData:
        """将加载的数据组装成一个 HeteroData 对象。"""
        if self.verbose > 1:
            print("    - Step 2/4: Building HeteroData object...")

        data = HeteroData()
        features_tensor = torch.from_numpy(features_np).float()

        # 填充节点特征
        for node_type, group in nodes_df.groupby("node_type"):
            node_indices = torch.from_numpy(group["global_id"].values).long()
            data[node_type].x = features_tensor[node_indices]

        # 填充边索引
        id_to_type_map = pd.Series(
            nodes_df.node_type.values, index=nodes_df.global_id
        ).to_dict()
        source_col = (
            self._config.data_structure.schema.internal.graph_output.source_node
        )
        target_col = (
            self._config.data_structure.schema.internal.graph_output.target_node
        )
        edge_type_col = (
            self._config.data_structure.schema.internal.graph_output.edge_type
        )

        for edge_type_str, group in graph_df.groupby(edge_type_col):
            sources = torch.from_numpy(group[source_col].values).long()
            targets = torch.from_numpy(group[target_col].values).long()

            # 从第一条边推断节点类型
            src_type = id_to_type_map[sources[0].item()]
            dst_type = id_to_type_map[targets[0].item()]

            data[src_type, edge_type_str, dst_type].edge_index = torch.stack(
                [sources, targets]
            )

        return data

    def _preprocess_graph(self, hetero_data: HeteroData) -> HeteroData:
        """对 HeteroData 对象进行所有必要的图转换。"""
        if self.verbose > 1:
            print(
                "    - Step 3/4: Pre-processing graph (ToUndirected, AddSelfLoops)..."
            )

        # 强制无向化
        hetero_data = T.ToUndirected()(hetero_data)

        # 添加自环
        hetero_data = T.AddSelfLoops()(hetero_data)

        return hetero_data

    def _create_loaders(
        self,
        hetero_data: HeteroData,
        train_labels_df: pd.DataFrame,
        test_labels_df: pd.DataFrame,
    ) -> Tuple[LinkNeighborLoader, LinkNeighborLoader]:
        """实例化 LinkNeighborLoader。"""
        if self.verbose > 1:
            print("    - Step 4/4: Creating DataLoaders...")

        labels_schema = self._config.data_structure.schema.internal.labeled_edges_output
        source_col, target_col, label_col = (
            labels_schema.source_node,
            labels_schema.target_node,
            labels_schema.label,
        )

        # 确定目标边类型 (通常只有一个)
        # TODO: 使其更具通用性，以支持多关系预测
        target_edge_type = ("drug", "interacts_with", "protein")

        # 准备训练加载器
        train_edge_label_index = torch.from_numpy(
            train_labels_df[[source_col, target_col]].values.T
        ).long()

        train_loader = LinkNeighborLoader(
            data=hetero_data,
            num_neighbors=[-1] * self._config.predictor.params.num_layers,
            edge_label_index=(target_edge_type, train_edge_label_index),
            batch_size=self._config.training.batch_size,
            shuffle=True,
            neg_sampling_ratio=self._config.training.negative_sampling_ratio,
            num_workers=self._config.runtime.train_loader_cpus,
        )

        # 准备测试加载器
        test_edge_label_index = torch.from_numpy(
            test_labels_df[[source_col, target_col]].values.T
        ).long()
        test_edge_label = torch.from_numpy(test_labels_df[label_col].values).float()

        test_loader = LinkNeighborLoader(
            data=hetero_data,
            num_neighbors=[-1] * self._config.predictor.params.num_layers,
            edge_label_index=(target_edge_type, test_edge_label_index),
            edge_label=test_edge_label,
            batch_size=self._config.training.batch_size,
            shuffle=False,
            neg_sampling_ratio=0.0,  # 评估时不需要负采样
            num_workers=self._config.runtime.test_loader_cpus,
        )

        return train_loader, test_loader
