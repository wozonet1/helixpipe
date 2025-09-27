import torch
import torch_geometric.transforms as T


def prepare_e2e_data(hetero_graph, train_df, test_df, maps, target_edge_type):
    """
    A master function to perform all ESSENTIAL data preparation steps for the E2E workflow.

    This includes:
    1. Purifying tensor dtypes and device placement.
    2. Converting supervision edge IDs from global to local.
    3. Validating local ID bounds.
    4. Forcing the graph to be undirected.
    5. Ensuring all tensors have a contiguous memory layout.

    Returns a tuple of fully prepared and sanitized data objects ready for the model and loader.
    """
    print("\n--- [ESSENTIAL PREP] Preparing all data for the E2E workflow ---")

    # Step 1: Purify graph object
    for store in hetero_graph.stores:
        for key, value in store.items():
            if torch.is_tensor(value):
                store[key] = value.to("cpu").contiguous()
    for edge_type in hetero_graph.edge_types:
        hetero_graph[edge_type].edge_index = hetero_graph[edge_type].edge_index.long()

    # Step 2: Force undirectedness
    hetero_graph = T.ToUndirected()(hetero_graph)
    print("Step 1/2: Graph purified (types, device, memory) and made undirected.")

    # Step 3: Prepare supervision edges with local IDs
    src_type, _, dst_type = target_edge_type

    # Train edges
    train_src_local = torch.tensor(
        [maps[src_type][gid] for gid in train_df["source"]], dtype=torch.long
    )
    train_dst_local = torch.tensor(
        [maps[dst_type][gid] for gid in train_df["target"]], dtype=torch.long
    )
    train_edge_label_index_local = torch.stack([train_src_local, train_dst_local])

    # Test edges
    test_src_local = torch.tensor(
        [maps[src_type][gid] for gid in test_df["source"]], dtype=torch.long
    )
    test_dst_local = torch.tensor(
        [maps[dst_type][gid] for gid in test_df["target"]], dtype=torch.long
    )
    test_edge_label_index_local = torch.stack([test_src_local, test_dst_local])
    test_labels = torch.from_numpy(test_df["label"].values).float()

    # Final validation assert is essential
    assert train_src_local.max() < hetero_graph[src_type].num_nodes
    assert test_src_local.max() < hetero_graph[src_type].num_nodes
    print("Step 2/2: Supervision edges converted to local IDs and validated.")
    print("--- [ESSENTIAL PREP] Data is ready for the model and loader. ---\n")

    return (
        hetero_graph,
        train_edge_label_index_local,
        test_edge_label_index_local,
        test_labels,
    )
