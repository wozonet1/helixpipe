import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse import csgraph
import sys
import torch
from pathlib import Path
import shutil
import yaml


def load_config(config_path="config.yaml"):
    """Loads the YAML config file from the project root."""
    project_root = Path(__file__).parent.parent
    with open(project_root / config_path, "r") as f:
        return yaml.safe_load(f)


def setup_dataset_directories(config: dict) -> None:
    """
    Checks and creates the necessary directory structure for the primary dataset.

    This function ensures that `raw/`, `processed/baseline/`, and `processed/gtopdb/`
    directories exist under the specified primary_dataset path. It also creates
    the `indexes` subdirectories within the processed variants.

    Args:
        config (dict): The loaded YAML configuration dictionary.
    """
    print("--- [Setup] Verifying and setting up dataset directories... ---")

    data_config = config["data"]
    root_path = Path(data_config["root"])
    runtime_config = config.get("runtime", {})
    primary_dataset = data_config.get("primary_dataset")

    if not primary_dataset:
        raise ValueError("'primary_dataset' not defined in the data configuration.")

    dataset_path = root_path / primary_dataset

    # 1. Define all required subdirectories
    raw_dir = dataset_path / data_config["subfolders"]["raw"]
    processed_dir = dataset_path / data_config["subfolders"]["processed"]
    use_gtopdb = data_config.get("use_gtopdb", False)
    variant_folder_name = "gtopdb" if use_gtopdb else "baseline"
    target_dir_to_clean = processed_dir / variant_folder_name

    # 2. Execute targeted deletion if force_restart is True
    force_restart = runtime_config.get("force_restart", False)
    if force_restart and target_dir_to_clean.exists():
        print(
            f"!!! WARNING: `force_restart` is True for the '{variant_folder_name}' variant."
        )
        print(f"    Deleting directory: {target_dir_to_clean}")
        try:
            shutil.rmtree(target_dir_to_clean)
            print(f"--> Successfully deleted old '{variant_folder_name}' directory.")
        except OSError as e:
            print(
                f"--> ERROR: Failed to delete directory {target_dir_to_clean}. Error: {e}"
            )
            sys.exit(1)

    baseline_dir = processed_dir / "baseline"
    gtopdb_dir = processed_dir / "gtopdb"

    # Also include the 'indexes' sub-subdirectories
    baseline_indexes_dir = baseline_dir / "indexes"
    gtopdb_indexes_dir = gtopdb_dir / "indexes"

    baseline_sim_dir = baseline_dir / "sim_matrixes"
    gtopdb_sim_dir = gtopdb_dir / "sim_matrixes"

    dirs_to_create = [
        raw_dir,
        processed_dir,
        baseline_dir,
        gtopdb_dir,
        baseline_indexes_dir,
        gtopdb_indexes_dir,
        baseline_sim_dir,
        gtopdb_sim_dir,
    ]

    # 2. Loop through and create them if they don't exist
    created_count = 0
    for directory in dirs_to_create:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            print(f"-> Created directory: {directory}")
            created_count += 1

    if created_count == 0:
        print("-> All necessary directories already exist. Setup complete.")
    else:
        print(
            f"-> Successfully created {created_count} new directories. Setup complete."
        )

    # [Optional but Recommended] A helpful message for the user
    if not any(raw_dir.iterdir()):
        print(f"-> WARNING: The 'raw' directory at '{raw_dir}' is empty.")
        print(
            "   Please make sure to place your raw data files (e.g., full.csv) inside it."
        )


def get_path(config: dict, file_key: str) -> Path:
    """
    Constructs the full path for any file defined in the config.
    This is the single source of truth for all file paths in the project.

    It understands the nested dataset structure (e.g., DrugBank/raw)
    and the `baseline/gtopdb` variant for processed files of the primary dataset.

    Args:
        config (dict): The loaded YAML configuration dictionary.
        file_key (str): A dot-separated key pointing to the filename in the config.
                        e.g., "DrugBank.raw.dti_interactions",
                        "DrugBank.processed.nodes_metadata",
                        "gtopdb.processed.interactions"

    Returns:
        Path: The complete Path object for the requested file.
    """
    data_config = config["data"]
    root_path = Path(data_config["root"])

    # 1. Parse the key
    key_parts = file_key.split(".")
    if len(key_parts) < 3:
        raise ValueError(
            f"Invalid file_key '{file_key}'. Must have at least 3 parts (e.g., 'DataSet.raw.fileName')."
        )

    dataset_name = key_parts[0]
    data_type = key_parts[1]  # 'raw' or 'processed'

    # 2. Determine the base directory
    subfolder_name = data_config["subfolders"].get(data_type)
    if not subfolder_name:
        raise KeyError(
            f"Data type '{data_type}' not defined in data.subfolders config."
        )

    base_dir = root_path / dataset_name / subfolder_name

    # 3. Handle the special case for 'processed' files of the PRIMARY dataset
    # They need an additional 'baseline' or 'gtopdb' sub-subfolder.
    primary_dataset_name = data_config.get("primary_dataset")
    if data_type == "processed" and dataset_name == primary_dataset_name:
        variant_folder = (
            "gtopdb" if data_config.get("use_gtopdb", False) else "baseline"
        )
        base_dir = base_dir / variant_folder

    # 4. Retrieve the filename from the config
    filename_dict_level = data_config["files"]
    try:
        for part in key_parts:
            filename_dict_level = filename_dict_level[part]
        filename = filename_dict_level
    except KeyError:
        raise KeyError(f"File key '{file_key}' not found in the config.yaml structure.")

    # Make sure the parent directory exists, creating it if necessary.
    # This is a good practice to avoid errors when writing files.
    base_dir.mkdir(parents=True, exist_ok=True)

    return base_dir / filename


def check_files_exist(config: dict, *file_keys: str) -> bool:
    """
    Checks if all specified data files (referenced by their config keys) exist.
    (This function can remain largely the same, as it relies on get_path)
    """
    for key in file_keys:
        try:
            filepath = get_path(config, key)
            if not filepath.exists():
                print(
                    f"File check FAILED: '{key}' not found at expected path: {filepath}"
                )
                return False
        except (KeyError, TypeError, ValueError) as e:
            print(f"File check FAILED: Could not resolve key '{key}'. Error: {e}")
            return False

    primary_dataset_name = config["data"].get("primary_dataset")
    variant_folder = "gtopdb" if config["data"].get("use_gtopdb", False) else "baseline"
    print(
        f"File check PASSED: All requested files exist for primary dataset '{primary_dataset_name}' (variant: '{variant_folder}')."
    )
    return True


def aug_random_walk(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv = np.power(row_sum, -1.0).flatten()
    d_mat = sp.diags(d_inv)
    return (d_mat.dot(adj)).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.0
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt).tocoo()


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def laplacian(mx, norm, adj):
    """Laplacian-normalize sparse matrix"""
    assert all(len(row) == len(mx) for row in mx), "Input should be a square matrix"

    return csgraph.laplacian(adj, normed=norm)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def load_file_as_Adj_matrix(Alledge, features):
    import scipy.sparse as sp

    relation_matrix = np.zeros((len(features), len(features)))
    for i, j in np.array(Alledge):
        lnc, mi = int(i), int(j)
        relation_matrix[lnc, mi] = 1
    #   relation_matrix[mi, lnc] = 1
    Adj = sp.csr_matrix(relation_matrix, dtype=np.float32)
    return Adj


def graph_decompose(adj, graph_name, k, metis_p, strategy="edge"):
    """
    Input:
        adj:the adjacency matrix of original graph
        graph_name:"cora","citeseer","pubmed"
        k:decompose into k subgraphs
        metis_p:"no_skeleton","all_skeleton","number" (depending on metis preprocessing)
        strategy:"edge" (for edge_decomposition),"node" (for node_decomposition)
    Output:
        the decomposed subgraphs
    """
    print("Skeleton:", metis_p)
    print("Strategy:", strategy)
    g, g_rest, edges_rest, gs = get_graph_skeleton(adj, graph_name, k, metis_p)
    gs = allocate_edges(g_rest, edges_rest, gs, strategy)

    re = []

    # print the info of nodes and edges of subgraphs
    edge_num_avg = 0
    compo_num_avg = 0
    print("Subgraph information:")
    for i in range(k):
        nodes_num = gs[i].number_of_nodes()
        edge_num = gs[i].number_of_edges()
        compo_num = nx.number_connected_components(gs[i])
        print("\t", nodes_num, edge_num, compo_num)
        edge_num_avg += edge_num
        compo_num_avg += compo_num
        re.append(nx.to_scipy_sparse_matrix(gs[i]))

        # check the shared edge number in all subgrqphs
    edge_share = set(sort_edge(gs[0].edges()))
    for i in range(k):
        edge_share &= set(sort_edge(gs[i].edges()))

    print("\tShared edge number is: %d" % len(edge_share))
    print("\tAverage edge number:", edge_num_avg / k)
    print("\tAverage connected component number:", compo_num_avg / k)
    print("\n" + "-" * 70 + "\n")
    return re


def sort_edge(edges):
    edges = list(edges)
    for i in range(len(edges)):
        u = edges[i][0]
        v = edges[i][1]
        if u > v:
            edges[i] = (v, u)
    return edges


def get_graph_skeleton(adj, graph_name, k, metis_p):
    """
    Input:
        adj:the adjacency matrix of original graph
        graph_name:"cora","citeseer","pubmed"
        k:decompose into k subgraphs
        metis_p:"no_skeleton","all_skeleton","k"
    Output:
        g:the original graph
        g_rest:the rest graph
        edges_rest:the rest edges
        gs:the skeleton of the graph for every subgraph
    """
    g = nx.from_numpy_matrix(adj.todense())
    num_nodes = g.number_of_nodes()
    print("Original nodes number:", num_nodes)
    num_edges = g.number_of_edges()
    print("Original edges number:", num_edges)
    print(
        "Original connected components number:", nx.number_connected_components(g), "\n"
    )

    g_dic = dict()

    for v, nb in g.adjacency():
        g_dic[v] = [u[0] for u in nb.items()]

        # initialize all the subgrapgs, add the nodes
    gs = [nx.Graph() for i in range(k)]
    for i in range(k):
        gs[i].add_nodes_from([i for i in range(num_nodes)])

    if metis_p == "no_skeleton":
        # no skeleton
        g_rest = g
        edges_rest = list(g_rest.edges())
    else:
        if metis_p == "all_skeleton":
            # doesn't use metis to cut any edge
            graph_cut = g
        else:
            # read the cluster info from file
            f = open("metis_file/" + graph_name + ".graph.part.%s" % metis_p, "r")
            cluster = dict()
            i = 0
            for lines in f:
                cluster[i] = eval(lines.strip("\n"))
                i += 1

            # get the graph cut by Metis
            graph_cut = nx.Graph()
            graph_cut.add_nodes_from([i for i in range(num_nodes)])

            for v in range(num_nodes):
                v_class = cluster[v]
                for u in g_dic[v]:
                    if cluster[u] == v_class:
                        graph_cut.add_edge(v, u)

        subgs = list(nx.connected_component_subgraphs(graph_cut))
        print("After Metis,connected component number:", len(subgs))

        # add the edges of spanning tree, get the skeleton
        for i in range(k):
            for subg in subgs:
                T = get_spanning_tree(subg)
                gs[i].add_edges_from(T)

        # get the rest graph including all the edges except the shared egdes of spanning trees
        edge_set_share = set(sort_edge(gs[0].edges()))
        for i in range(k):
            edge_set_share &= set(sort_edge(gs[i].edges()))
        edge_set_total = set(sort_edge(g.edges()))
        edge_set_rest = edge_set_total - edge_set_share
        edges_rest = list(edge_set_rest)
        g_rest = nx.Graph()
        g_rest.add_nodes_from([i for i in range(num_nodes)])
        g_rest.add_edges_from(edges_rest)

    # print the info of nodes and edges of subgraphs
    print("Skeleton information:")
    for i in range(k):
        print(
            "\t",
            gs[i].number_of_nodes(),
            gs[i].number_of_edges(),
            nx.number_connected_components(gs[i]),
        )

    edge_set_share = set(sort_edge(gs[0].edges()))
    for i in range(k):
        edge_set_share &= set(sort_edge(gs[i].edges()))
    print("\tShared edge number is: %d\n" % len(edge_set_share))

    return g, g_rest, edges_rest, gs


def get_spanning_tree(g):
    """
    Input:Graph
    Output:list of the edges in spanning tree
    """
    g_dic = dict()
    for v, nb in g.adjacency():
        g_dic[v] = [u[0] for u in nb.items()]
        np.random.shuffle(g_dic[v])
    flag_dic = dict()
    if g.number_of_nodes() == 1:
        return []
    gnodes = np.array(g.nodes)
    np.random.shuffle(gnodes)

    for v in gnodes:
        flag_dic[v] = 0

    current_path = []

    def dfs(u):
        stack = [u]
        current_node = u
        flag_dic[u] = 1
        while len(current_path) != (len(gnodes) - 1):
            pop_flag = 1
            for v in g_dic[current_node]:
                if flag_dic[v] == 0:
                    flag_dic[v] = 1
                    current_path.append((current_node, v))
                    stack.append(v)
                    current_node = v
                    pop_flag = 0
                    break
            if pop_flag:
                stack.pop()
                current_node = stack[-1]

    dfs(gnodes[0])
    return current_path


def allocate_edges(g_rest, edges_rest, gs, strategy):
    """
    Input:
        g_rest:the rest graph
        edges_rest:the rest edges
        gs:the skeleton of the graph for every subgraph
        strategy:"edge" (for edge_decomposition),"node" (for node_decomposition)
    Output:
        the decomposed graphs after allocating rest edges
    """
    k = len(gs)
    if strategy == "edge":
        print("Allocate the rest edges randomly and averagely.")
        np.random.shuffle(edges_rest)
        t = int(len(edges_rest) / k)

        # add edges
        for i in range(k):
            if i == k - 1:
                gs[i].add_edges_from(edges_rest[t * i :])
            else:
                gs[i].add_edges_from(edges_rest[t * i : t * (i + 1)])
        return gs

    elif strategy == "node":
        print("Allocate the edges of each nodes randomly and averagely.")
        g_dic = dict()
        for v, nb in g_rest.adjacency():
            g_dic[v] = [u[0] for u in nb.items()]
            np.random.shuffle(g_dic[v])

        def sample_neighbors(nb_ls, k):
            np.random.shuffle(nb_ls)
            ans = []
            for i in range(k):
                ans.append([])
            if len(nb_ls) == 0:
                return ans
            if len(nb_ls) > k:
                t = int(len(nb_ls) / k)
                for i in range(k):
                    ans[i] += nb_ls[i * t : (i + 1) * t]
                nb_ls = nb_ls[k * t :]
            """
            if len(nb_ls)>0:
                for i in range(k):
                    ans[i].append(nb_ls[i%len(nb_ls)])
            """

            if len(nb_ls) > 0:
                for i in range(len(nb_ls)):
                    ans[i].append(nb_ls[i])

            np.random.shuffle(ans)
            return ans

        # add edges
        for v, nb in g_dic.items():
            ls = np.array(sample_neighbors(nb, k))
            for i in range(k):
                gs[i].add_edges_from([(v, j) for j in ls[i]])

        return gs
