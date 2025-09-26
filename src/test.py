import pandas as pd
import hydra
from omegaconf import DictConfig
import research_template as rt
import sys


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def test_data_integrity(cfg: DictConfig):
    """
    一个用于验证核心数据文件一致性的独立测试脚本。

    它将加载某一折的 `nodes.csv` 和 `typed_edges...csv` 文件，
    并对内容的合法性进行一系列的断言检查。
    """
    fold_to_inspect = 1

    print("=" * 80)
    print(f" " * 15 + f"RUNNING DATA INTEGRITY CHECK FOR FOLD {fold_to_inspect}")
    print("=" * 80)

    # --- 1. 加载核心数据文件 ---
    try:
        print("\n--- [1/3] Loading core data files...")

        # a. 加载“户籍库”：nodes.csv
        nodes_path = rt.get_path(cfg, "processed.nodes_metadata")
        nodes_df = pd.read_csv(nodes_path)
        print(f"✅ Loaded nodes metadata from: {nodes_path}")

        # b. 加载我们要检查的图谱文件
        graph_path = rt.get_path(
            cfg,
            "processed.typed_edge_list_template",
            split_suffix=f"_fold{fold_to_inspect}",
        )
        edges_df = pd.read_csv(graph_path)
        print(f"✅ Loaded graph topology from: {graph_path}")

    except FileNotFoundError as e:
        print(f"\n❌ FATAL: A required file was not found: {e.filename}")
        sys.exit(1)

    # --- 2. 创建一个高效的 ID -> Type 映射字典 ---
    print("\n--- [2/3] Building Global ID to Node Type map...")
    id_to_type_map = pd.Series(
        nodes_df.node_type.values, index=nodes_df.node_id
    ).to_dict()
    print("✅ ID map built successfully.")

    # --- 3. [核心检查] 逐行验证边的类型一致性 ---
    print("\n--- [3/3] Verifying edge type consistency...")

    errors_found = 0

    for index, row in edges_df.iterrows():
        source_id = row["source"]
        target_id = row["target"]
        edge_type_str = row["edge_type"]

        # a. 从映射中查询真实的节点类型
        # 我们假设所有ID都应该在映射中，如果不在，.get()会返回None
        real_source_type = id_to_type_map.get(source_id)
        real_target_type = id_to_type_map.get(target_id)

        # b. 从边类型字符串中，【推断】期望的节点类型
        # 这是一个简化的启发式规则，但对我们的场景有效
        # e.g., "drug_protein_interaction" -> implies source should be 'drug', target 'protein'
        # e.g., "rev_drug_protein_interaction" -> implies source 'protein', target 'drug'
        # e.g., "protein_protein_similarity" -> implies both should be 'protein'

        expected_source_type, expected_target_type = None, None
        parts = edge_type_str.split("_")

        if "similarity" in edge_type_str:
            type1 = parts[0]
            type2 = parts[1]
            expected_source_type = type1
            expected_target_type = type2
        elif "interaction" in edge_type_str:
            if "rev" in edge_type_str:
                expected_source_type = parts[2]
                expected_target_type = parts[1]
            else:
                expected_source_type = parts[0]
                expected_target_type = parts[1]

        # c. 进行比较和断言
        error_msg = ""
        if real_source_type != expected_source_type:
            error_msg += f" Source node {source_id} should be '{expected_source_type}' but is '{real_source_type}'. "
        if real_target_type != expected_target_type:
            error_msg += f" Target node {target_id} should be '{expected_target_type}' but is '{real_target_type}'. "

        if error_msg:
            print(f"\n❌ ERROR at row {index}: For edge_type '{edge_type_str}':")
            print(f"   - {error_msg}")
            print(f"   - Full row data: {row.to_dict()}")
            errors_found += 1

    print("\n--- Check Complete ---")
    if errors_found == 0:
        print(
            f"\n✅ SUCCESS! All {len(edges_df)} edges in the graph have consistent node types."
        )
    else:
        print(
            f"\n❌ FAILURE! Found {errors_found} inconsistent edges. Please review the errors above."
        )


if __name__ == "__main__":
    test_data_integrity()
