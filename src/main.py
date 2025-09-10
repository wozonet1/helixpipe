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
