"""
Build homogeneous graph from STITCH chemical-chemical links.
"""

import torch
from torch_geometric.data import Data
import pandas as pd
import os

IN_FILE = "../data/processed/encoded_ddi.csv"
OUT_GRAPH = "../data/processed/graph_data.pt"

def build_graph():
    df = pd.read_csv(IN_FILE)
    num_drugs = int(max(df["drug_a"].max(), df["drug_b"].max()) + 1)
    edge_index = torch.tensor([df["drug_a"].values, df["drug_b"].values], dtype=torch.long)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # make undirected

    # Random features (128-D) — later can replace with SMILES embeddings
    x = torch.randn((num_drugs, 128), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)
    torch.save(data, OUT_GRAPH)
    print(f"Graph built with {num_drugs} nodes, {edge_index.shape[1] // 2} edges.")

if __name__ == "__main__":
    build_graph()
