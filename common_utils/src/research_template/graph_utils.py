import torch
import numpy as np
import scipy.sparse as sp


def sparse_mx_to_torch_sparse_tensor(sparse_mx: sp.spmatrix) -> torch.Tensor:
    """
    Converts a SciPy sparse matrix to a PyTorch sparse COO tensor.

    Args:
        sparse_mx (sp.spmatrix): The input SciPy sparse matrix.

    Returns:
        torch.Tensor: The equivalent PyTorch sparse COO tensor.
    """
    # Ensure the matrix is in COO format, as it's the most direct mapping
    sparse_mx = sparse_mx.tocoo().astype(np.float32)

    # Create tensor for indices
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )

    # Create tensor for values
    values = torch.from_numpy(sparse_mx.data)

    # Define the shape of the sparse tensor
    shape = torch.Size(sparse_mx.shape)

    return torch.sparse_coo_tensor(indices, values, shape)


def normalize_sparse_matrix(mx: sp.spmatrix, mode: str = "gcn") -> sp.spmatrix:
    """
    Normalizes a SciPy sparse matrix according to a specified mode.

    Args:
        mx (sp.spmatrix): The input adjacency matrix.
        mode (str, optional): Normalization mode.
            - 'gcn': Symmetrically normalize matrix as in GCN, D^-0.5 * A * D^-0.5.
            - 'row': Row-normalize matrix, D^-1 * A.
            Defaults to 'gcn'.

    Returns:
        sp.spmatrix: The normalized sparse matrix.
    """
    if mode not in ["gcn", "row"]:
        raise ValueError("Invalid mode. Choose from 'gcn' or 'row'.")

    # Add self-loops to avoid issues with nodes that have no neighbors
    mx = mx + sp.eye(mx.shape[0])

    rowsum = np.array(mx.sum(1))

    if mode == "gcn":
        r_inv_sqrt = np.power(rowsum, -0.5).flatten()
        r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.0
        r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
        return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt).tocoo()

    elif mode == "row":
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.0
        r_mat_inv = sp.diags(r_inv)
        return r_mat_inv.dot(mx).tocoo()
