import numpy as np
import torch
from scipy.sparse import coo_matrix

# region homo_ndls


def convert_hetero_to_homo(hetero_data) -> tuple:
    """
    Converts a HeteroData object (representing an undirected graph)
    to a symmetric, homogeneous SciPy adjacency matrix.
    """
    print("--> Converting HeteroData to Homogeneous Graph for NDLS...")

    num_nodes = hetero_data.num_nodes
    features = torch.cat(
        [hetero_data[node_type].x for node_type in hetero_data.node_types], dim=0
    )

    node_offsets = {}
    current_offset = 0
    for node_type in hetero_data.node_types:
        node_offsets[node_type] = current_offset
        current_offset += hetero_data[node_type].num_nodes

    # --- [SIMPLIFIED & CORRECTED LOGIC] ---

    # 1. Collect all unique undirected edges from HeteroData in global indices
    all_edges = []
    for edge_type in hetero_data.edge_types:
        source_type, _, target_type = edge_type
        edge_index = hetero_data[edge_type].edge_index

        offset_edge_index = torch.stack(
            [
                edge_index[0] + node_offsets[source_type],
                edge_index[1] + node_offsets[target_type],
            ]
        )
        all_edges.append(offset_edge_index)

    # Concatenate all edges defined in the HeteroData object
    edge_index_global = torch.cat(all_edges, dim=1)

    # 2. Create the symmetric (undirected) adjacency matrix
    # We add both (u, v) and (v, u) to the edge list for the COO matrix
    full_edge_index = torch.cat(
        [edge_index_global, torch.stack([edge_index_global[1], edge_index_global[0]])],
        dim=1,
    )

    # Note: We do NOT use .unique() here anymore, because data_proc.py
    # has already guaranteed the uniqueness of undirected edges.
    # If we still want to be safe, unique can be used on the final full_edge_index.
    full_edge_index = full_edge_index.unique(dim=1)

    adj = coo_matrix(
        (
            np.ones(full_edge_index.shape[1]),
            (full_edge_index[0].numpy(), full_edge_index[1].numpy()),
        ),
        shape=(num_nodes, num_nodes),
    )

    # The number of undirected edges is now correctly adj.nnz / 2
    print(
        f"--> Homogeneous graph constructed: {adj.shape[0]} nodes, {adj.nnz // 2} edges."
    )
    return adj, features, node_offsets


# endregion
