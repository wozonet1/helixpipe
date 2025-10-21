# src/encoders/rgcn_hetero_encoder.py

import torch
from torch_geometric.nn import HeteroConv, RGCNConv


class RGCNHeteroEncoder(torch.nn.Module):
    """
    A heterogeneous Graph Convolutional Network Encoder using the RGCNConv layer.

    This encoder is designed to work with PyG's HeteroData objects. It uses the
    powerful HeteroConv wrapper to apply RGCN convolutions to different edge
    types in the graph.
    """

    def __init__(
        self,
        in_channels_dict: dict,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
        hetero_metadata: tuple,
    ):
        """
        Initializes the RGCN Hetero Encoder.

        Args:
            in_channels_dict (dict): A dictionary mapping each node type (str) to its
                                     input feature dimension (int).
                                     Example: {'drug': 128, 'protein': 128, 'ligand': 128}
            hidden_channels (int): The number of channels in the hidden layers.
            out_channels (int): The number of channels in the output embeddings.
            num_layers (int): The total number of GNN layers.
            dropout (float): The dropout probability.
            hetero_metadata (tuple): The metadata of the HeteroData object, typically
                                     obtained from `data.metadata()`. It's a tuple
                                     of (node_types, edge_types).
        """
        super().__init__()

        self.node_types, self.edge_types = hetero_metadata

        # --- 1. Input Projections (Linear Layers) ---
        # It's a best practice to project features of different node types to the
        # same hidden dimension before the first message passing layer.
        self.proj = torch.nn.ModuleDict()
        for node_type, in_channels in in_channels_dict.items():
            self.proj[node_type] = torch.nn.Linear(in_channels, hidden_channels)

        # --- 2. Message Passing Layers (HeteroConv) ---
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            # For each layer, we create a HeteroConv wrapper.
            # This wrapper holds a specific convolution layer for each edge type.
            conv = HeteroConv(
                {
                    # For each edge type, we specify an RGCNConv layer.
                    # The '-1' for in_channels is a PyG convention that allows it to
                    # automatically infer the correct input dimensions for source and
                    # target nodes.
                    edge_type: RGCNConv(
                        in_channels=-1,
                        out_channels=hidden_channels,
                        num_relations=len(self.edge_types),
                    )
                    for edge_type in self.edge_types
                },
                aggr="sum",
            )  # 'aggr' specifies how to aggregate results for nodes that
            # receive messages from different edge types. 'sum' is common.
            self.convs.append(conv)

        # --- 3. Final Projection (Optional, but good practice) ---
        # This layer projects the final hidden embeddings to the desired output dimension.
        self.lin = torch.nn.ModuleDict()
        for node_type in self.node_types:
            self.lin[node_type] = torch.nn.Linear(hidden_channels, out_channels)

        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x_dict: dict, edge_index_dict: dict) -> dict:
        """
        The forward pass of the encoder.

        Args:
            x_dict (dict): The dictionary of node features.
            edge_index_dict (dict): The dictionary of edge indices.

        Returns:
            dict: A dictionary of the final node embeddings for each node type.
        """
        # 1. Apply initial projection and activation
        x_dict = {
            node_type: self.proj[node_type](x).relu() for node_type, x in x_dict.items()
        }

        # 2. Propagate through the HeteroConv layers
        for conv in self.convs:
            # The HeteroConv wrapper takes the node feature and edge index dictionaries
            # and internally handles the message passing for all edge types.
            x_dict_update = conv(x_dict, edge_index_dict)

            # Apply activation and dropout after each layer
            x_dict = {
                node_type: self.dropout(x.relu())
                for node_type, x in x_dict_update.items()
            }

        # 3. Apply final linear projection
        x_dict = {node_type: self.lin[node_type](x) for node_type, x in x_dict.items()}

        return x_dict
