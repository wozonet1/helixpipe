# src/main.py (v2.0)

import random
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from encoders.ndls_homo_encoder import NDLS_Homo_Encoder
from predictors.gbdt_predictor import GBDT_Link_Predictor
import research_template as rt


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
def load_graph_data(config: dict, data_variant: str) -> HeteroData:
    """
    Loads all the necessary files and constructs a PyG HeteroData object.
    """
    print("--- [Step 1] Loading graph data and constructing HeteroData object... ---")

    primary_dataset = config["data"]["primary_dataset"]

    # Temporarily override use_gtopdb in the config for rt.get_path to work correctly
    # This is a small hack to reuse rt.get_path for different variants.
    config["data"]["use_gtopdb"] = data_variant == "gtopdb"

    # 1. Load node metadata and features
    nodes_df = pd.read_csv(
        rt.get_path(config, f"{primary_dataset}.processed.nodes_metadata")
    )
    features_tensor = torch.from_numpy(
        pd.read_csv(
            rt.get_path(config, f"{primary_dataset}.processed.node_features"),
            header=None,
        ).values
    ).float()
    print(f"DEBUG_5: Loaded nodes_df shape = {nodes_df.shape}")
    print(f"DEBUG_5: Loaded features_tensor shape = {features_tensor.shape}")
    assert nodes_df.shape[0] == features_tensor.shape[0], (
        "FATAL: Mismatch between nodes.csv and features.csv loaded in main.py!"
    )
    data = HeteroData()

    # 2. Populate node features for each node type
    node_type_map = {}
    for node_type in nodes_df["node_type"].unique():
        mask = nodes_df["node_type"] == node_type
        node_ids = nodes_df[mask]["node_id"].values
        data[node_type].x = features_tensor[node_ids]

        # Store local-to-global ID mapping
        # And create a reverse map for easy lookup later
        for local_id, global_id in enumerate(node_ids):
            node_type_map[global_id] = (node_type, local_id)

    print(f"-> Populated features for node types: {list(data.node_types)}")

    # 3. Populate edge indices for each edge type
    typed_edges_key = f"{primary_dataset}.processed.typed_edge_list_template"
    edges_path = rt.get_path(config, typed_edges_key)
    print(f"--> Loading graph structure from resolved path: {edges_path}")
    try:
        edges_df = pd.read_csv(edges_path)
    except FileNotFoundError:
        print(f"!!! FATAL ERROR: The required graph file does not exist: {edges_path}")
        print(
            "    Please run `data_proc.py` with the CURRENT `params.include_relations` settings first."
        )
        raise  # Re-raise the exception to stop the script
    num_edges_in_csv = len(edges_df)
    print(f"DEBUG_A: Loaded typed_edges.csv with {num_edges_in_csv} total edges.")
    # A helper map to get node type from global ID
    id_to_type_str = {row.node_id: row.node_type for row in nodes_df.itertuples()}
    num_edges_loaded_into_hetero = 0
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
        num_edges_loaded_into_hetero += edge_index.shape[1]

    print(f"-> Populated edges for edge types: {list(data.edge_types)}")
    print(
        f"DEBUG_B: Total edges loaded into HeteroData object: {num_edges_loaded_into_hetero}"
    )
    assert num_edges_in_csv == num_edges_loaded_into_hetero, (
        f"FATAL: Edge count mismatch! CSV has {num_edges_in_csv}, but HeteroData only has {num_edges_loaded_into_hetero}."
    )
    # We also need to add reverse edges for message passing in GNNs
    # PyG has a transform for this, but we can do it manually for clarity
    # ... (To be added in the next step) ...
    print(f"DEBUG_6: Final HeteroData object has {data.num_nodes} nodes.")
    assert data.num_nodes == len(nodes_df), (
        "FATAL: HeteroData node count does not match nodes_df count!"
    )
    print("--- HeteroData object constructed successfully! ---")
    return data


# endregion

# region homo_ndls

# In src/main.py


def convert_hetero_to_homo(hetero_data: HeteroData) -> tuple:
    """
    Converts a HeteroData object (representing an undirected graph)
    to a symmetric, homogeneous SciPy adjacency matrix.
    """
    print("--> Converting HeteroData to Homogeneous Graph for NDLS...")
    from scipy.sparse import coo_matrix

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


# region main

# TODO: test coldstart


def train():
    """
    Main script to select and run the configured workflow, with full MLflow integration.
    This is the central controller for all experiments.
    """
    # ===================================================================
    # 1. Initialization: Load Config and Initialize Tracker
    # ===================================================================
    config = rt.load_config()
    # Initialize the tracker from the research_template library
    tracker = rt.MLflowTracker(config)

    # --- Start of protected block for MLflow ---
    try:
        # ===================================================================
        # 2. Setup: Start MLflow Run, Set Environment, and Get Config
        # ===================================================================
        tracker.start_run()  # This also logs all relevant parameters

        rt.set_seeds(config["runtime"]["seed"])
        device = torch.device(
            config["runtime"]["gpu"] if torch.cuda.is_available() else "cpu"
        )

        # Get the primary switches for the experiment from config
        training_config = config["training"]
        encoder_name = training_config["encoder"]
        # Use .get() for the optional predictor
        predictor_name = training_config.get("predictor", None)

        # --- [NEW] Determine paradigm by convention ---
        paradigm = "two_stage" if predictor_name else "end_to_end"

        # 3. Startup Log
        config_hash = rt.get_relations_config_hash(config)
        print("\n" + "=" * 80)
        print(" " * 20 + "Starting DTI Prediction Experiment")
        print("=" * 80)
        print("Configuration loaded for this run:")
        print(f"  - Paradigm (Inferred): '{paradigm}'")
        print(f"  - Primary Dataset:     '{config['data']['primary_dataset']}'")
        print(f"  - Data Variant:        '{training_config['data_variant']}'")
        print(f"  - Encoder:             '{encoder_name}'")
        print(f"  - Predictor:           '{predictor_name or 'N/A'}'")
        print(f"  - Graph Config Hash:   '{config_hash}'")
        print(f"  - Seed:                {config['runtime']['seed']}")
        print(f"  - Device:              {device}")
        print("=" * 80 + "\n")

        hetero_data = load_graph_data(config, training_config["data_variant"])

        # 5. --- Workflow Dispatcher (based on the inferred paradigm) ---
        results = None
        if paradigm == "two_stage":
            # --- Two-Stage Logic ---

            # 5a. Run Encoder
            node_embeddings = None
            if encoder_name == "ndls_homo":
                adj, features, _ = convert_hetero_to_homo(hetero_data)
                encoder = NDLS_Homo_Encoder(config, device)
                encoder.fit(adj, features)
                node_embeddings = encoder.get_embeddings()
            else:
                raise NotImplementedError(
                    f"Encoder '{encoder_name}' is not supported for the 'two_stage' paradigm."
                )

            # 5b. Run Predictor
            if node_embeddings is not None:
                if predictor_name == "gbdt":
                    predictor = GBDT_Link_Predictor(config)
                    results = predictor.predict(node_embeddings)
                else:
                    raise NotImplementedError(
                        f"Predictor '{predictor_name}' is not supported."
                    )

        elif paradigm == "end_to_end":
            # --- End-to-End Logic ---
            print("End-to-end paradigm is not implemented yet.")
            # Here, you would check the encoder_name and run the corresponding e2e model
            # if encoder_name == 'rgcn_hetero':
            #     results = run_rgcn_e2e_workflow(...)
            pass

        # 6. Logging
        if results:
            tracker.log_cv_results(
                results["aucs"], results["auprs"]
            )  # FIXME:mlflow missing model metrics logging

    except Exception as e:
        print(f"\n!!! FATAL ERROR: Experiment run failed: {e}")
        # Log the failure to MLflow for tracking and easier debugging
        if tracker.is_active:
            import mlflow

            mlflow.set_tag("run_status", "FAILED")
            # Log the full traceback for detailed error analysis in MLflow
            import traceback

            mlflow.log_text(traceback.format_exc(), "error_traceback.txt")
        raise  # Re-raise the exception to stop the script

    finally:
        # ===================================================================
        # 7. Teardown: Always end the MLflow run
        # ===================================================================
        tracker.end_run()
        print("\n" + "=" * 80)
        print(" " * 27 + "Experiment Run Finished")
        print("=" * 80 + "\n")
