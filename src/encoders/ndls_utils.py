import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse import coo_matrix
from tqdm import tqdm


def augmented_random_walk_normalization(adj: sp.spmatrix) -> sp.spmatrix:
    """
    Performs augmented random walk normalization (D_hat^-1 * A_hat) on an adjacency matrix.
    This is a specific normalization used in algorithms like NDLS.

    A_hat = A + I (Adjacency matrix with self-loops)
    D_hat is the diagonal degree matrix of A_hat.

    Args:
        adj (sp.spmatrix): The input adjacency matrix (A).

    Returns:
        sp.spmatrix: The normalized sparse matrix.
    """
    # Add self-loops
    adj_hat = adj + sp.eye(adj.shape[0])

    # Calculate the degree matrix D_hat
    row_sum = np.array(adj_hat.sum(1))

    # [CRITICAL FIX] Handle nodes with degree 0
    d_inv = np.power(row_sum, -1.0).flatten()
    d_inv[np.isinf(d_inv)] = 0.0  # Set inf to 0 for isolated nodes
    d_mat_inv = sp.diags(d_inv)

    # Calculate D_hat^-1 * A_hat
    normalized_adj = d_mat_inv.dot(adj_hat)

    return normalized_adj.tocoo()


def aver(
    hops: torch.Tensor, adj: coo_matrix, feature_list: list[torch.Tensor]
) -> torch.Tensor:
    """
    Assembles the final node embeddings based on the personalized hop counts (NDLS smoothing).

    For each node `i`, it selects the feature vector from `feature_list[h_i]`,
    where `h_i` is the optimal hop count for node `i` stored in the `hops` tensor.

    Args:
        hops (torch.Tensor): A 1D tensor of shape [num_nodes], where each element
                             is an integer representing the optimal hop for that node.
        adj (coo_matrix): The original adjacency matrix (its shape is used to get num_nodes).
                          This argument is kept for compatibility with the original NDLS code,
                          though it's not strictly needed if num_nodes is passed directly.
        feature_list (list[torch.Tensor]): A list where feature_list[k] is the feature
                                           matrix aggregated over a k-hop neighborhood.

    Returns:
        torch.Tensor: The final smoothed node feature matrix of shape [num_nodes, feature_dim].
    """
    num_nodes = adj.shape[0]
    feature_dim = feature_list[0].shape[1]

    # 1. 创建一个空的张量来存放最终的特征
    # Ensure it's on the same device as the feature_list tensors
    output_features = torch.zeros(num_nodes, feature_dim, device=feature_list[0].device)

    # 2. 将hops张量转换为整数类型，以便用作索引
    # The hops tensor from NDLS might be float initially.
    hops = hops.long()

    # 3. [核心逻辑] 利用PyTorch的高级索引，并行地执行“查表与组装”
    # 我们遍历所有可能的hop值（从0到最大hop）
    for h in torch.unique(hops):
        # a. 找到所有最佳hop值等于当前 h 的节点的索引
        # `torch.where` returns a tuple, we need the first element
        node_indices_for_hop_h = torch.where(hops == h)[0]

        # b. 从对应的特征矩阵 feature_list[h] 中，
        #    根据这些索引，一次性地“切片”出所有这些节点的特征向量。
        features_for_hop_h = feature_list[h][node_indices_for_hop_h]

        # c. 将切片出的特征向量，填充到输出矩阵的正确位置。
        output_features[node_indices_for_hop_h] = features_for_hop_h

    return output_features


# In src/encoders/ndls_utils.py


def aver_smooth_vectorized(
    hops: torch.Tensor, feature_list: list[torch.Tensor], alpha: float = 0.15
) -> torch.Tensor:
    """
    An efficient, vectorized implementation of the iterative smoothing with teleportation.

    Args:
        hops (torch.Tensor): A 1D tensor of optimal hop for each node.
        feature_list (list[torch.Tensor]): List of diffused features at each hop.
        alpha (float): The teleport probability to the initial features.

    Returns:
        torch.Tensor: The final smoothed node feature matrix.
    """
    num_nodes, feature_dim = feature_list[0].shape
    device = feature_list[0].device

    # --- 1. Pre-calculate the cumulative sum of features ---
    # F_sum[k] will store the sum of features from hop 0 to k.
    feature_stack = torch.stack(feature_list, dim=0)  # Shape: [k, num_nodes, dim]
    F_sum = torch.cumsum(feature_stack, dim=0)  # Shape: [k, num_nodes, dim]

    # --- 2. Initialize the output tensor ---
    output_features = torch.zeros(num_nodes, feature_dim, device=device)
    hops = hops.long()

    # --- 3. Vectorized processing for each unique hop value ---
    for h in torch.unique(hops):
        if h == 0:
            # Nodes with hop 0 just use their original features
            node_indices = torch.where(hops == h)[0]
            output_features[node_indices] = feature_list[0][node_indices]
        else:
            node_indices = torch.where(hops == h)[0]

            # Get the cumulative sum of features up to hop h-1 for these nodes
            # Shape: [num_selected_nodes, dim]
            sum_f_j = F_sum[h - 1, node_indices, :]

            # Get the initial features (at hop 0) for these nodes
            f_0 = feature_list[0][node_indices]

            # Apply the formula in a single vectorized operation
            # final_feature = ((1-alpha) * sum(F_j) + h * alpha * F_0) / h
            final_features_h = ((1 - alpha) * sum_f_j + h * alpha * f_0) / h

            output_features[node_indices] = final_features_h

    return output_features


def diffuse_features_on_homo_graph(
    features: torch.Tensor, adj_norm: torch.Tensor, k: int
) -> list[torch.Tensor]:
    """
    Performs K-hop feature diffusion on a homogeneous graph.

    Args:
        features (torch.Tensor): Initial node features.
        adj_norm (torch.Tensor): Normalized adjacency matrix for propagation.
        k (int): The maximum number of diffusion steps.

    Returns:
        list[torch.Tensor]: A list where list[i] contains features after i hops.
    """
    print(f"--> Pre-computing {k}-hop feature diffusions...")
    feature_list = [features]
    for _ in tqdm(range(1, k), desc="Feature Diffusion", leave=False):
        feature_list.append(torch.spmm(adj_norm, feature_list[-1]))
    return feature_list


# Location: src/encoders/ndls_utils.py or src/encoders/ops.py


def localize_optimal_hops(
    feature_list: list[torch.Tensor], anchor_features: torch.Tensor, epsilon: float
) -> torch.Tensor:
    """
    Finds the optimal hop count for each node based on its feature distance to an anchor.

    Args:
        feature_list (list[torch.Tensor]): List of diffused features at each hop.
        anchor_features (torch.Tensor): A global anchor feature vector for comparison.
        epsilon (float): The distance threshold.

    Returns:
        torch.Tensor: A 1D tensor containing the optimal hop for each node.
    """
    print(f"--> Localizing optimal hops with epsilon={epsilon}...")
    num_nodes = feature_list[0].shape[0]
    max_hops = len(feature_list)
    device = feature_list[0].device

    hops = torch.zeros(num_nodes, dtype=torch.long, device=device)
    mask_before = torch.zeros(num_nodes, dtype=torch.bool, device=device)

    for i in tqdm(range(max_hops), desc="Hop Localization", leave=False):
        dist = torch.norm(feature_list[i] - anchor_features, p=2, dim=1)
        mask = (dist < epsilon) & ~mask_before
        hops[mask] = i
        mask_before |= mask

    hops[~mask_before] = max_hops - 1
    return hops
