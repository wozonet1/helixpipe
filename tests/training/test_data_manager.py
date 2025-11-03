import shutil
import unittest
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import research_template as rt
from torch_geometric.data import HeteroData

from nasnet.configs import register_all_schemas

# 导入我们需要测试的类
from nasnet.training import DataManager
from nasnet.utils import get_path, register_hydra_resolvers

# 全局注册
register_all_schemas()
register_hydra_resolvers()


class TestDataManager(unittest.TestCase):
    def setUp(self):
        """
        为每个测试创建一个完全隔离的、包含伪造数据文件的沙箱环境。
        """
        print("\n" + "=" * 80)

        # 1. 创建一个临时的根目录
        self.test_dir = Path("./test_temp_data_manager").resolve()
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(parents=True)

        # 2. 加载一个配置，并将其所有路径重定向到我们的沙箱
        self.cfg = self._get_test_config()

        # 3. 在沙箱中生成所有必需的伪造数据文件
        self._generate_fake_data()

        print(f"--> Test environment set up in: {self.test_dir}")

    def tearDown(self):
        """
        测试结束后，清理沙箱目录。
        """
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
            print("--> Test environment torn down.")

    # --- 辅助方法 ---

    def _get_test_config(self):
        """加载一个最小化的配置，并将路径重定向到测试目录。"""
        with hydra.initialize(config_path="../../conf", version_base=None):
            cfg = hydra.compose(
                config_name="config",
                overrides=[
                    "data_params=test",
                    "relations=DPL_sim",
                    "training.batch_size=2",  # 使用小batch size方便测试
                    "runtime.train_loader_cpus=0",  # 在测试中使用单进程加载
                    "runtime.test_loader_cpus=0",
                ],
            )

        # 强制重定向所有路径
        # 这里的路径结构需要与 get_path 的逻辑严格对应
        cfg.global_paths.data_root = str(self.test_dir)

        return cfg

    def _generate_fake_data(self):
        """一个核心辅助函数，用于生成所有伪造数据。"""
        print("--> Generating fake data files for testing...")

        # --- a. 定义伪造数据内容 ---
        # 节点: 2个drug, 1个ligand, 2个protein
        nodes_data = {
            "global_id": [0, 1, 2, 3, 4],
            "node_type": ["drug", "drug", "ligand", "protein", "protein"],
            "authoritative_id": [101, 102, 201, "P01", "P02"],
            "structure": ["C", "CC", "CCC", "MA", "MV"],
        }
        # 特征: 5个节点，每个4维
        features_data = np.random.rand(5, 4).astype(np.float32)

        # 边 (Fold 1): 包含多种类型
        graph_data = {
            "source": [0, 1, 0, 3, 2],
            "target": [3, 4, 1, 4, 1],
            "edge_type": [
                "interacts_with",
                "interacts_with",
                "drug_drug_similarity",
                "protein_protein_similarity",
                "drug_ligand_similarity",
            ],
        }

        # 标签 (Fold 1)
        train_labels_data = {"source": [0, 1], "target": [3, 4]}
        test_labels_data = {"source": [0, 2], "target": [4, 3], "label": [1, 0]}

        # --- b. 获取路径并写入文件 ---
        # 使用 get_path 来确保我们写入的位置与 DataManager 读取的位置完全一致

        nodes_path = get_path(self.cfg, "processed.common.nodes_metadata")
        features_path = get_path(self.cfg, "processed.common.node_features")

        # [NEW] 在写入前确保目录存在
        rt.ensure_path_exists(nodes_path)
        pd.DataFrame(nodes_data).to_csv(nodes_path, index=False)

        rt.ensure_path_exists(features_path)
        np.save(features_path, features_data)

        # Specific files for Fold 1
        fold = 1
        graph_path_factory = get_path(self.cfg, "processed.specific.graph_template")
        graph_path = graph_path_factory(prefix=f"fold_{fold}", suffix="train")

        labels_path_factory = get_path(self.cfg, "processed.specific.labels_template")
        train_labels_path = labels_path_factory(prefix=f"fold_{fold}", suffix="train")
        test_labels_path = labels_path_factory(prefix=f"fold_{fold}", suffix="test")

        # [NEW] 为每个文件都确保目录存在
        rt.ensure_path_exists(graph_path)
        pd.DataFrame(graph_data).to_csv(graph_path, index=False)

        rt.ensure_path_exists(train_labels_path)
        pd.DataFrame(train_labels_data).to_csv(train_labels_path, index=False)

        rt.ensure_path_exists(test_labels_path)
        pd.DataFrame(test_labels_data).to_csv(test_labels_path, index=False)

        print("--> Fake data generation complete.")

    # --- 测试用例 ---

    def test_initialization_and_setup(self):
        """
        测试点1 (黑盒): DataManager能否被成功实例化并完成setup，不抛出任何异常。
        """
        print("\n--- Running Test: Initialization and Setup ---")
        try:
            dm = DataManager(config=self.cfg, fold_idx=1)
            dm.setup()
            print(
                "  ✅ Test Passed: Initialization and setup completed without errors."
            )
        except Exception as e:
            self.fail(f"DataManager.setup() raised an unexpected exception: {e}")

    def test_metadata_property(self):
        """
        测试点2 (黑盒): setup后，metadata属性是否返回了正确的节点和边类型。
        """
        print("\n--- Running Test: Metadata Property ---")
        dm = DataManager(config=self.cfg, fold_idx=1)
        dm.setup()

        node_types, edge_types = dm.metadata

        # 根据我们的伪造数据进行断言
        self.assertCountEqual(node_types, ["drug", "ligand", "protein"])

        # 注意: T.ToUndirected() 会为每种有向边类型创建反向边
        # 我们只检查原始类型是否存在
        original_edge_types_from_meta = {
            tuple(e) for e in edge_types if not e[1].endswith("_rev")
        }

        self.assertIn(
            ("drug", "interacts_with", "protein"), original_edge_types_from_meta
        )
        self.assertIn(
            ("drug", "drug_drug_similarity", "drug"), original_edge_types_from_meta
        )

        print("  ✅ Test Passed: Metadata property is correct.")

    def test_dataloaders_and_batch_content(self):
        """
        测试点3 (白盒): DataLoader能否成功迭代，并且返回的batch内容是否正确。
        """
        print("\n--- Running Test: DataLoaders and Batch Content ---")
        dm = DataManager(config=self.cfg, fold_idx=1)
        dm.setup()

        # a. 测试 Train Loader
        train_loader = dm.train_loader
        self.assertIsNotNone(train_loader)

        train_batch = next(iter(train_loader))

        # 验证返回的batch是一个HeteroData对象
        self.assertIsInstance(train_batch, HeteroData)

        # 验证batch中的节点特征
        self.assertEqual(train_batch["drug"].x.shape[1], 4)  # 4维特征
        self.assertEqual(train_batch["protein"].x.shape[1], 4)

        # 验证核心的 edge_label_index (监督信号)
        # batch_size=2, neg_sampling_ratio=1.0 -> 2个正样本, 2个负样本
        self.assertEqual(
            train_batch["drug", "interacts_with", "protein"].edge_label_index.shape,
            (2, 4),
        )

        # b. 测试 Test Loader
        test_loader = dm.test_loader
        self.assertIsNotNone(test_loader)

        test_batch = next(iter(test_loader))
        self.assertIsInstance(test_batch, HeteroData)

        # 验证测试batch的标签
        self.assertIn("edge_label", test_batch["drug", "interacts_with", "protein"])
        # batch_size=2, neg_sampling_ratio=0.0 -> 只有2个原始样本
        self.assertEqual(
            len(test_batch["drug", "interacts_with", "protein"].edge_label), 2
        )

        print(
            "  ✅ Test Passed: DataLoaders are iterable and batch content seems correct."
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
