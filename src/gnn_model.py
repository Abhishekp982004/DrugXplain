# gnn_model.py — GCN model for link prediction (with encode/decode methods)
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=64):
        super().__init__()
        # Encoder: 2-layer GCN
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        # Decoder: simple MLP to predict edge existence
        self.link_pred = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 2, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

    def encode(self, x, edge_index):
        """Compute node embeddings."""
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_index):
        """Predict edge logits for given edges."""
        src, dst = edge_index
        edge_repr = torch.cat([z[src], z[dst]], dim=1)
        return self.link_pred(edge_repr).view(-1)

    def forward(self, x, edge_index, edges):
        """Full forward pass: encode + decode"""
        z = self.encode(x, edge_index)
        return self.decode(z, edges)
