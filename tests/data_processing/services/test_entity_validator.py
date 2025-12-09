# tests/data_processing/services/test_entity_validator.py

import unittest
from unittest.mock import patch

import pandas as pd
from omegaconf import OmegaConf

from helixpipe.data_processing.services.entity_validator import (
    validate_and_filter_entities,
)


# --- 辅助函数：快速构建 Mock 配置 ---
def create_mock_config(global_mw_max=500, bindingdb_mw_max=None, brenda_enabled=True):
    """
    创建一个包含全局配置和特定数据源配置的 Mock AppConfig。
    """
    # 1. 定义字典结构
    conf_dict = {
        "runtime": {"verbose": 0, "cpus": 1},
        "knowledge_graph": {
            "entity_types": {
                "drug": "drug",
                "ligand": "ligand",
                "protein": "protein",
                "ligand_endo": "endogenous_ligand",
                "ligand_exo": "exogenous_ligand",
            }
        },
        "data_params": {
            "filtering": {
                "enabled": True,
                "apply_pains_filter": True,
                "molecular_weight": {"max": global_mw_max},
            },
            # BindingDB 配置
            "bindingdb": {
                "filtering": {
                    "enabled": True,
                    "apply_pains_filter": True,
                    "molecular_weight": {"max": bindingdb_mw_max}
                    if bindingdb_mw_max
                    else None,
                }
                if bindingdb_mw_max
                else None
            },
            # BRENDA 配置
            "brenda": {
                "filtering": {
                    "enabled": brenda_enabled,  # 用于测试禁用过滤的场景
                    "molecular_weight": {"max": 100},  # 故意设得很小，看是否被忽略
                }
            },
            # GtoPdb (未配置，测试默认行为)
            "gtopdb": None,
        },
    }
    # 2. 转换为 DictConfig 以支持点号访问 (config.data_params.bindingdb...)
    return OmegaConf.create(conf_dict)


class TestEntityValidatorSourceAware(unittest.TestCase):
    def setUp(self):
        # 准备测试数据：包含不同来源、不同分子量的实体
        self.entities_df = pd.DataFrame(
            {
                "entity_id": [101, 102, 103, 104, 105],
                "entity_type": ["drug"] * 5,
                "structure": [
                    "C",  # 1. MW ~16 (极小)
                    "C" * 40,  # 2. MW ~560 (中等，>500)
                    "C" * 80,  # 3. MW ~1100 (巨大)
                    "C" * 40,  # 4. MW ~560
                    "C" * 80,  # 5. MW ~1100
                ],
                "all_sources": [
                    ["other"],  # 101: 仅全局规则
                    ["other"],  # 102: 仅全局规则
                    ["bindingdb"],  # 103: BindingDB 来源
                    ["bindingdb", "other"],  # 104: BindingDB + 其他
                    ["brenda"],  # 105: BRENDA 来源
                ],
            }
        )

    @patch("helixpipe.data_processing.services.entity_validator.get_valid_pubchem_cids")
    @patch(
        "helixpipe.data_processing.services.entity_validator.validate_smiles_structure"
    )
    def test_source_aware_relaxation(self, mock_validate_smiles, mock_get_cids):
        """
        测试核心逻辑：来源感知放宽 (Source-Aware Relaxation).
        """
        print("\n--- Running Test: Source-Aware Filtering Logic ---")

        # --- Mock 设置 ---
        # 假设所有 ID 和结构语法都有效，我们只测试属性过滤逻辑
        mock_get_cids.return_value = {101, 102, 103, 104, 105}
        # 直接返回输入的 structure 作为 canonical smiles
        mock_validate_smiles.side_effect = lambda s: s

        # --- 场景配置 ---
        # 全局限制: MW <= 500
        # BindingDB 限制: MW <= 1200 (放宽)
        # BRENDA: 禁用过滤 (完全放行)
        cfg = create_mock_config(
            global_mw_max=500, bindingdb_mw_max=1200, brenda_enabled=False
        )

        # --- 执行测试 ---
        result_df = validate_and_filter_entities(self.entities_df, cfg)
        passed_ids = set(result_df["entity_id"])

        print(f"  Passed IDs: {passed_ids}")

        # --- 断言分析 ---

        # 1. ID 101 (MW~16): 来源 'other'。
        #    规则：全局 (Max 500)。
        #    结果：Pass (16 < 500)。
        self.assertIn(101, passed_ids, "ID 101 should pass global rules.")

        # 2. ID 102 (MW~560): 来源 'other'。
        #    规则：全局 (Max 500)。
        #    结果：Fail (560 > 500)。
        self.assertNotIn(102, passed_ids, "ID 102 should fail global rules.")

        # 3. ID 103 (MW~1100): 来源 'bindingdb'。
        #    规则：max(全局 500, BindingDB 1200) = 1200。
        #    结果：Pass (1100 < 1200)。
        #    【关键】：这证明了 BindingDB 的宽松配置覆盖了全局严格配置。
        self.assertIn(
            103, passed_ids, "ID 103 should pass due to BindingDB relaxation."
        )

        # 4. ID 104 (MW~560): 来源 'bindingdb', 'other'。
        #    规则：max(全局 500, BindingDB 1200, Other用全局) = 1200。
        #    结果：Pass (560 < 1200)。
        #    【关键】：只要有一个来源允许，就应该通过。
        self.assertIn(
            104, passed_ids, "ID 104 should pass due to partial BindingDB source."
        )

        # 5. ID 105 (MW~1100): 来源 'brenda'。
        #    规则：BRENDA 禁用过滤 -> 阈值无限大。
        #    结果：Pass。
        self.assertIn(
            105, passed_ids, "ID 105 should pass because BRENDA filtering is disabled."
        )

        print("  ✅ All source-aware logic verified.")

    @patch("helixpipe.data_processing.services.entity_validator.get_valid_pubchem_cids")
    @patch(
        "helixpipe.data_processing.services.entity_validator.validate_smiles_structure"
    )
    def test_global_fallback(self, mock_validate_smiles, mock_get_cids):
        """
        测试：当来源没有特殊配置时，是否正确回退到全局配置。
        """
        print("\n--- Running Test: Global Fallback ---")
        mock_get_cids.return_value = {103}
        mock_validate_smiles.side_effect = lambda s: s

        # 配置：全局 500，BindingDB 没有配置 (None)
        cfg = create_mock_config(global_mw_max=500, bindingdb_mw_max=None)

        # 数据：ID 103 (MW~1100), 来源 bindingdb
        df = self.entities_df[self.entities_df["entity_id"] == 103]

        result_df = validate_and_filter_entities(df, cfg)

        # 结果：应该被过滤，因为 BindingDB 没有配置，回退使用全局 500，而 1100 > 500。
        self.assertTrue(
            result_df.empty, "ID 103 should fail when BindingDB has no specific config."
        )
        print("  ✅ Global fallback verified.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
