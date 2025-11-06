# gnn_model.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNEncoder(torch.nn.Module):
    """Encodes node features into embeddings using lightweight GCN layers."""
    def __init__(self, in_channels, hidden_channels=64, out_channels=32):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GCNLinkPredictor(torch.nn.Module):
    """Link prediction model using node embeddings from GCN."""
    def __init__(self, in_channels, hidden_channels=64, out_channels=32):
        super().__init__()
        self.encoder = GCNEncoder(in_channels, hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_pairs):
        z = self.encoder(x, edge_index)
        src, dst = edge_pairs
        score = (z[src] * z[dst]).sum(dim=1)  # dot product similarity
        return score
