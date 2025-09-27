from torch_geometric.data import HeteroData
import torch
import torch_geometric.transforms as T


def run_optional_diagnostics(hetero_graph: HeteroData):
    """
    Runs a suite of OPTIONAL but recommended diagnostic checks.
    Call this during development to ensure data integrity.
    """
    print("\n--- [OPTIONAL DIAGNOSTIC] Running full diagnostic suite ---")

    # Check 1: Official PyG validation
    hetero_graph.validate(raise_on_error=True)
    print("✅ (1/3) Official PyG validation successful.")

    # Check 2 & 3: Deep health checks
    if not (
        diagnose_hetero_data(hetero_graph) and diagnose_node_features(hetero_graph)
    ):
        raise ValueError("HeteroData object failed deep health checks.")
    print("✅ (2/3 & 3/3) Deep health checks successful.")

    print("--- [OPTIONAL DIAGNOSTIC] All checks passed. ---\n")


# 放置在 train.py 的顶部
def diagnose_hetero_data(data: HeteroData):
    """一个详细的诊断函数，彻查HeteroData对象的健康状况。"""
    print("\n--- [DIAGNOSTIC 2] Performing deep health check on HeteroData object...")
    is_healthy = True

    # 检查1: 所有edge_index都必须是torch.long
    for edge_type in data.edge_types:
        if data[edge_type].edge_index.dtype != torch.long:
            print(
                f"❌ DTYPE_ERROR for edge_type '{edge_type}': edge_index is {data[edge_type].edge_index.dtype}, but MUST be torch.long!"
            )
            is_healthy = False

    if is_healthy:
        print("✅ All edge_index tensors have correct dtype (torch.long).")

    # 检查2: 检查所有边索引是否越界
    for edge_type in data.edge_types:
        src_type, _, dst_type = edge_type

        # 检查源节点
        if data[edge_type].edge_index.numel() > 0:  # 仅在有边的情况下检查
            max_src_id = data[edge_type].edge_index[0].max().item()
            num_src_nodes = data[src_type].num_nodes
            if max_src_id >= num_src_nodes:
                print(
                    f"❌ BOUNDS_ERROR for edge_type '{edge_type}': Max source ID is {max_src_id}, but node_type '{src_type}' only has {num_src_nodes} nodes!"
                )
                is_healthy = False

            # 检查目标节点
            max_dst_id = data[edge_type].edge_index[1].max().item()
            num_dst_nodes = data[dst_type].num_nodes
            if max_dst_id >= num_dst_nodes:
                print(
                    f"❌ BOUNDS_ERROR for edge_type '{edge_type}': Max destination ID is {max_dst_id}, but node_type '{dst_type}' only has {num_dst_nodes} nodes!"
                )
                is_healthy = False

    if is_healthy:
        print(
            "✅ All edge indices are within the bounds of their respective node stores."
        )

    print("--- Deep health check finished. ---")
    return is_healthy


def diagnose_node_features(data: HeteroData) -> bool:
    """
    Performs a deep analysis of node features in a HeteroData object.

    Checks for the presence of dangerous NaN (Not a Number) or Infinity
    values, which are common culprits for Segmentation Faults in C++/CUDA extensions.

    Args:
        data (HeteroData): The graph data object to diagnose.

    Returns:
        bool: True if all features are clean, False otherwise.
    """
    print("\n--- [DIAGNOSTIC] Analyzing node features for invalid values (NaN/inf)...")
    is_clean = True

    for node_type in data.node_types:
        # Check if the node type has features assigned
        if "x" not in data[node_type]:
            print(
                f"⚠️  INFO: Node type '{node_type}' has no features ('x' attribute). Skipping."
            )
            continue

        features = data[node_type].x

        # Check for NaN values
        nan_mask = torch.isnan(features)
        if nan_mask.any():
            num_nan = nan_mask.sum().item()
            print(
                f"❌ FATAL: Found {num_nan} NaN value(s) in features for node_type '{node_type}'!"
            )
            is_clean = False

        # Check for Infinity values
        inf_mask = torch.isinf(features)
        if inf_mask.any():
            num_inf = inf_mask.sum().item()
            print(
                f"❌ FATAL: Found {num_inf} Infinity value(s) in features for node_type '{node_type}'!"
            )
            is_clean = False

    if is_clean:
        print("✅ All node features are clean (no NaN/inf found).")

    return is_clean


def sanitize_for_loader(data: HeteroData) -> HeteroData:
    """
    Performs a final, deep sanitization of a HeteroData object to ensure
    all its tensors have a contiguous memory layout before being passed
    to a C++-backed loader.

    Args:
        data (HeteroData): The graph object to sanitize.

    Returns:
        HeteroData: The sanitized graph object.
    """
    print(
        "\n--- [FINAL SANITIZATION] Forcing contiguous memory layout for all tensors..."
    )

    for store in data.stores:
        for key, value in store.items():
            if torch.is_tensor(value):
                # .contiguous() returns a new tensor with contiguous memory if the
                # original is not; otherwise, it returns the original tensor.
                # This is a very cheap operation if the tensor is already contiguous.
                store[key] = value.contiguous()

    print("✅ All tensors are now memory-contiguous.")
    return data
