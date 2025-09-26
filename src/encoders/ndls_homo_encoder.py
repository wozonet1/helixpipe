import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from research_template import sparse_mx_to_torch_sparse_tensor
from .ndls_homo_utils import (
    augmented_random_walk_normalization,
    aver,
    aver_smooth_vectorized,
    diffuse_features_on_homo_graph,
    localize_optimal_hops,
)
import tqdm as tqdm
import torch.nn as nn


class DNNRefiner(nn.Module):
    """
    A simple two-layer fully-connected network to refine node features.
    This mimics the hidden DNN layer from the original implementation.
    """

    def __init__(
        self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float
    ):
        super().__init__()
        self.fcn1 = nn.Linear(in_channels, hidden_channels)
        self.fcn2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Note: The original code applies dropout BEFORE the first layer, which is unusual.
        # We will replicate this behavior for perfect reproduction.
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fcn1(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fcn2(x)
        # The original code returns log_softmax and the raw embeddings. We only need the embeddings.
        return x


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
        self.refiner = None
        if self.params.refiner.enabled:
            print("--> DNN Refiner is ENABLED.")
            # We don't know the input dimension yet, so we'll instantiate it in fit()
        else:
            print("--> DNN Refiner is DISABLED.")

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
        smoothing_mode = self.params.get("smoothing_mode", "lookup")

        if smoothing_mode == "iterative":
            print("    -> Using 'iterative' smoothing mode.")
            alpha = self.params.get("smoothing_alpha", 0.15)
            # Call our new, efficient, vectorized function
            smoothed_features = aver_smooth_vectorized(
                hops.cpu(), [f.cpu() for f in feature_list], alpha
            )
        else:  # Default to our original 'lookup' method
            print("    -> Using 'lookup' smoothing mode.")
            smoothed_features = aver(hops.cpu(), adj, [f.cpu() for f in feature_list])

        self.embeddings = pd.DataFrame(smoothed_features.numpy())
        print(f"--> NDLS embeddings generated with shape: {self.embeddings.shape}")

        if self.params.refiner.enabled:
            print("--> Stage 4: Applying DNN Feature Refiner...")

            # 1. Instantiate the refiner now that we know the input dimension
            refiner_params = self.params.refiner
            self.refiner = DNNRefiner(
                in_channels=smoothed_features.shape[1],
                hidden_channels=refiner_params.hidden_channels,
                out_channels=refiner_params.out_channels,
                dropout=refiner_params.dropout,
            ).to(self.device)

            # 2. Put the refiner in evaluation mode (important for dropout)
            self.refiner.eval()

            # 3. Perform a forward pass to get the refined embeddings
            # The operation is unsupervised, so no backpropagation is needed.
            with torch.no_grad():
                final_embeddings_tensor = self.refiner(
                    smoothed_features.to(self.device)
                )

            # Move embeddings back to CPU for GBDT
            final_embeddings_tensor = final_embeddings_tensor.cpu()
        else:
            # If refiner is disabled, just use the smoothed features directly
            final_embeddings_tensor = smoothed_features

        self.embeddings = pd.DataFrame(final_embeddings_tensor.numpy())
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
        return self.embeddings.values
