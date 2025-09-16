import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from research_template import sparse_mx_to_torch_sparse_tensor
from .ndls_utils import (
    augmented_random_walk_normalization,
    aver,
    diffuse_features_on_homo_graph,
    localize_optimal_hops,
)
import tqdm as tqdm
from omegaconf import DictConfig


class NDLS_Homo_Encoder:
    """
    Computes node embeddings on a homogeneous graph using the NDLS algorithm.
    This class encapsulates the two-stage (unsupervised) embedding generation process.
    """

    def __init__(self, config: dict, device: torch.device):
        """
        Initializes the encoder with configuration and device.
        """
        self.config = config
        self.params = config.encoder
        self.device = device
        self.embeddings = None

    def fit(self, adj: coo_matrix, features: torch.Tensor):
        """
        The main method to compute and store the node embeddings.
        This contains the core logic from your old main.py.
        """
        print("--- [Encoder] Fitting NDLS_Homo_Encoder... ---")
        k1 = self.params["k1"]
        epsilon1 = self.params["epsilon1"]

        print("--> Stage 1: Pre-computing feature diffusions and stability anchor...")

        # Calculate the stationary distribution (norm_a_inf) for the stability anchor
        node_sum = adj.shape[0]
        edge_sum = adj.sum() / 2
        # Add 1 to row_sum for stability, especially for isolated nodes
        row_sum = np.array(adj.sum(1)).flatten() + 1
        norm_a_inf = (
            torch.tensor(row_sum / (2 * edge_sum + node_sum), dtype=torch.float32)
            .view(1, -1)
            .to(self.device)
        )

        # Normalize the adjacency matrix using the NDLS-specific method
        adj_norm = sparse_mx_to_torch_sparse_tensor(
            augmented_random_walk_normalization(adj)
        ).to(self.device)

        # L1-normalize features and move to the target device
        features_norm = F.normalize(features, p=1).to(self.device)

        # Pre-compute K1 hops of diffused features
        feature_list = diffuse_features_on_homo_graph(features_norm, adj_norm, k1)

        # Compute the global stability anchor feature
        norm_fea_inf = torch.mm(norm_a_inf, features_norm)

        # ===================================================================
        # 2. Localization Stage
        # ===================================================================
        print("--> Stage 2: Localizing optimal hop for each node...")

        hops = localize_optimal_hops(feature_list, norm_fea_inf, epsilon1)

        # ===================================================================
        # 3. Smoothing Stage
        # ===================================================================
        print("--> Stage 3: Performing final smoothing...")
        smoothed_features = aver(hops.cpu(), adj, [f.cpu() for f in feature_list])

        self.embeddings = pd.DataFrame(smoothed_features.numpy())
        print(f"--> NDLS embeddings generated with shape: {self.embeddings.shape}")

        return self

    def get_embeddings(self) -> pd.DataFrame:
        """
        Returns the computed embeddings as a pandas DataFrame.

        Raises:
            RuntimeError: If embeddings have not been computed yet via .fit().

        Returns:
            pd.DataFrame: A DataFrame where the index corresponds to the global node ID
                          and columns are the feature dimensions.
        """
        if self.embeddings is None:
            raise RuntimeError(
                "Embeddings have not been computed yet. Please call .fit() first."
            )
        return self.embeddings
