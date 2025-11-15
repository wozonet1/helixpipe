# analysis/debug_minimal_loader.py

import random

import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader


def create_minimal_hetero_graph() -> None:
    """
    Creates a tiny, clean, fully-controlled heterogeneous graph in memory.
    """
    graph = HeteroData()

    # --- 1. Define Nodes ---
    # We will create 10 drugs and 20 proteins
    num_drugs = 10
    num_proteins = 20

    # Assign some random features to them
    graph["drug"].x = torch.randn(num_drugs, 128)  # 128-dim features
    graph["protein"].x = torch.randn(num_proteins, 128)

    # --- 2. Define Edges ---
    # We create a small, random graph.

    # a) Drug-Protein Interactions (the edges we want to predict)
    # Let's create 15 random D-P edges
    num_dp_edges = 15
    dp_sources = [random.randint(0, num_drugs - 1) for _ in range(num_dp_edges)]
    dp_targets = [random.randint(0, num_proteins - 1) for _ in range(num_dp_edges)]
    graph["drug", "interacts_with", "protein"].edge_index = torch.tensor(
        [dp_sources, dp_targets], dtype=torch.long
    )

    # b) Drug-Drug Similarity Edges
    # Let's create 5 random D-D edges
    num_dd_edges = 5
    dd_sources = [random.randint(0, num_drugs - 1) for _ in range(num_dd_edges)]
    dd_targets = [random.randint(0, num_drugs - 1) for _ in range(num_dd_edges)]
    graph["drug", "similar_to", "drug"].edge_index = torch.tensor(
        [dd_sources, dd_targets], dtype=torch.long
    )

    # c) Protein-Protein Similarity Edges
    # Let's create 8 random P-P edges
    num_pp_edges = 8
    pp_sources = [random.randint(0, num_proteins - 1) for _ in range(num_pp_edges)]
    pp_targets = [random.randint(0, num_proteins - 1) for _ in range(num_pp_edges)]
    graph["protein", "similar_to", "protein"].edge_index = torch.tensor(
        [pp_sources, pp_targets], dtype=torch.long
    )

    return graph


def main_minimal_debug() -> None:
    """
    A minimal script to test LinkNeighborLoader with a synthetic, in-memory graph.
    """
    print("=" * 80)
    print(" " * 10 + "STARTING MINIMAL SYNTHETIC LOADER DEBUG SCRIPT")
    print("=" * 80)

    # --- 1. Create the toy graph ---
    print("\n--- [1/3] Creating a minimal synthetic heterogeneous graph...")
    try:
        hetero_graph = create_minimal_hetero_graph()
        print("✅ Minimal graph created SUCCESSFULLY.")
        print("    - Node Types:", hetero_graph.node_types)
        print("    - Edge Types:", hetero_graph.edge_types)
        print(hetero_graph)
    except Exception as e:
        print(f"!!! FAILED during graph creation: {e}")
        return

    # --- 2. [CORE TEST] Instantiate LinkNeighborLoader ---
    print("\n--- [2/3] Instantiating LinkNeighborLoader on the minimal graph...")
    try:
        # We need to provide the edges we want to learn/predict
        target_edge_type = ("drug", "interacts_with", "protein")
        edge_label_index = hetero_graph[target_edge_type].edge_index

        train_loader = LinkNeighborLoader(
            data=hetero_graph,
            num_neighbors=[-1] * 2,  # Sample all neighbors for simplicity
            edge_label_index=(target_edge_type, edge_label_index),
            batch_size=4,  # A small batch size
            shuffle=True,
            neg_sampling_ratio=1.0,
            # [CRITICAL] set num_workers=0 to disable multiprocessing for debugging
            num_workers=0,
        )
        print("✅ LinkNeighborLoader instantiated SUCCESSFULLY.")
    except Exception as e:
        print(f"!!! FAILED during Loader instantiation: {e}")
        import traceback

        traceback.print_exc()
        return

    # --- 3. [FINAL TEST] Attempt to fetch a batch ---
    print("\n--- [3/3] Attempting to fetch the first batch from the minimal graph...")
    try:
        first_batch = next(iter(train_loader))
        print("✅ First batch fetched SUCCESSFULLY!")
        print("\n--- Minimal Batch Content ---")
        print(first_batch)
        print("\n" + "=" * 80)
        print(" " * 10 + "CONCLUSION: The environment and PyG installation are STABLE!")
        print("=" * 80)
    except Exception as e:
        print(f"\n!!! FAILED when fetching the first batch on a MINIMAL graph: {e}")
        print(
            "    This STRONGLY indicates a deep incompatibility in your C++/CUDA environment."
        )
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main_minimal_debug()
