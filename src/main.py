# src/main.py (v2.0)

import yaml
import random
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch_geometric.data import HeteroData

from utils import get_path  # We will primarily use get_path


def load_config(config_path="config.yaml"):
    """Loads the YAML config file from the project root."""
    project_root = Path(__file__).parent.parent
    with open(project_root / config_path, "r") as f:
        return yaml.safe_load(f)


def set_seeds(seed):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# region load data
def load_graph_data(config: dict) -> HeteroData:
    """
    Loads all the necessary files and constructs a PyG HeteroData object.
    """
    print("--- [Step 1] Loading graph data and constructing HeteroData object... ---")

    primary_dataset = config["data"]["primary_dataset"]
    data_variant = config["training"]["data_variant"]  # 'baseline' or 'gtopdb'

    # Temporarily override use_gtopdb in the config for get_path to work correctly
    # This is a small hack to reuse get_path for different variants.
    config["data"]["use_gtopdb"] = data_variant == "gtopdb"

    # 1. Load node metadata and features
    nodes_df = pd.read_csv(
        get_path(config, f"{primary_dataset}.processed.nodes_metadata")
    )
    features_tensor = torch.from_numpy(
        pd.read_csv(
            get_path(config, f"{primary_dataset}.processed.node_features"), header=None
        ).values
    ).float()

    data = HeteroData()

    # 2. Populate node features for each node type
    node_type_map = {}
    for node_type in nodes_df["node_type"].unique():
        mask = nodes_df["node_type"] == node_type
        node_ids = nodes_df[mask]["node_id"].values
        data[node_type].x = features_tensor[node_ids]

        # Store local-to-global ID mapping
        # And create a reverse map for easy lookup later
        for i, global_id in enumerate(node_ids):
            node_type_map[global_id] = (node_type, i)

    print(f"-> Populated features for node types: {list(data.node_types)}")

    # 3. Populate edge indices for each edge type
    edges_df = pd.read_csv(
        get_path(config, f"{primary_dataset}.processed.typed_edge_list")
    )

    # A helper map to get node type from global ID
    id_to_type_str = {row.node_id: row.node_type for row in nodes_df.itertuples()}

    for edge_type_str, group in edges_df.groupby("edge_type"):
        sources = group["source"].values
        targets = group["target"].values

        # Determine the triplet for the edge type
        source_type = id_to_type_str[sources[0]]
        target_type = id_to_type_str[targets[0]]
        edge_type_tuple = (source_type, edge_type_str, target_type)

        # PyG's HeteroData needs local indices within each node type
        # We need to convert our global indices
        source_local_ids = [node_type_map[s_id][1] for s_id in sources]
        target_local_ids = [node_type_map[t_id][1] for t_id in targets]

        edge_index = torch.tensor(
            [source_local_ids, target_local_ids], dtype=torch.long
        )

        data[edge_type_tuple].edge_index = edge_index

    print(f"-> Populated edges for edge types: {list(data.edge_types)}")

    # We also need to add reverse edges for message passing in GNNs
    # PyG has a transform for this, but we can do it manually for clarity
    # ... (To be added in the next step) ...

    print("--- HeteroData object constructed successfully! ---")
    return data


# endregion

# region homo_ndls


def convert_hetero_to_homo(hetero_data: HeteroData) -> tuple:
    """
    Converts a HeteroData object to a homogeneous representation suitable for NDLS.

    Args:
        hetero_data (HeteroData): The input heterogeneous graph.

    Returns:
        tuple: A tuple containing:
            - adj (scipy.sparse.coo_matrix): The adjacency matrix in COO format.
            - features (torch.Tensor): The combined node feature matrix.
            - node_offsets (dict): A dictionary mapping node type to its starting index.
    """
    print("--> Converting HeteroData to Homogeneous Graph for NDLS...")
    from scipy.sparse import coo_matrix

    num_nodes = hetero_data.num_nodes

    # 1. Concatenate all node features into a single tensor
    features = torch.cat(
        [hetero_data[node_type].x for node_type in hetero_data.node_types], dim=0
    )

    # 2. Calculate node offsets and combine all edge_indices
    edge_indices = []
    node_offsets = {}
    current_offset = 0
    # The order MUST be the same as the feature concatenation order
    for node_type in hetero_data.node_types:
        node_offsets[node_type] = current_offset
        current_offset += hetero_data[node_type].num_nodes

    for edge_type in hetero_data.edge_types:
        source_type, _, target_type = edge_type
        edge_index = hetero_data[edge_type].edge_index

        # Apply offsets to convert local indices back to global-like indices
        offset_edge_index = torch.stack(
            [
                edge_index[0] + node_offsets[source_type],
                edge_index[1] + node_offsets[target_type],
            ]
        )
        edge_indices.append(offset_edge_index)

    # Also add reverse edges to make the graph undirected
    for edge_type in hetero_data.edge_types:
        source_type, _, target_type = edge_type
        edge_index = hetero_data[edge_type].edge_index

        offset_edge_index = torch.stack(
            [
                edge_index[1] + node_offsets[target_type],  # Reversed
                edge_index[0] + node_offsets[source_type],  # Reversed
            ]
        )
        edge_indices.append(offset_edge_index)

    adj_coo_tensor = torch.cat(edge_indices, dim=1).unique(
        dim=1
    )  # Use .unique() to remove duplicates

    # 3. Convert to scipy sparse matrix, which is what the old code expects
    adj = coo_matrix(
        (
            np.ones(adj_coo_tensor.shape[1]),
            (adj_coo_tensor[0].numpy(), adj_coo_tensor[1].numpy()),
        ),
        shape=(num_nodes, num_nodes),
    )

    print(
        f"--> Homogeneous graph constructed: {adj.shape[0]} nodes, {adj.nnz // 2} edges."
    )
    return adj, features, node_offsets


def main():
    """Main training and evaluation script."""
    # 1. Load Configuration and Set Seeds
    config = load_config()
    set_seeds(config["runtime"]["seed"])
    device = torch.device(
        config["runtime"]["gpu"] if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # 2. Load Data
    hetero_data = load_graph_data(config)
    print("\nFinal HeteroData structure:")
    print(hetero_data)

    # ... STAGE 2: MODEL BUILDING (To be implemented next) ...

    # ... STAGE 3: TRAINING & EVALUATION (To be implemented next) ...


if __name__ == "__main__":
    main()
