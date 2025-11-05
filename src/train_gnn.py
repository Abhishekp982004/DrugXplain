import torch
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score
from gnn_model import GATLinkPredictor

GRAPH_PATH = "../data/processed/graph_data.pt"
MODEL_PATH = "../models/gnn_model.pt"

def train(epochs=20):
    data = torch.load(GRAPH_PATH, weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    model = GATLinkPredictor(data.x.size(1)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()

        neg_edges = negative_sampling(data.edge_index, num_nodes=data.num_nodes)
        pos_pred = model(data.x, data.edge_index, data.edge_index)
        neg_pred = model(data.x, data.edge_index, neg_edges)

        labels = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)])
        logits = torch.cat([pos_pred, neg_pred])
        loss = loss_fn(logits, labels)
        loss.backward()
        opt.step()

        with torch.no_grad():
            auc = roc_auc_score(labels.cpu(), torch.sigmoid(logits).cpu())
        print(f"Epoch {epoch:02d} | Loss={loss.item():.4f} | AUC={auc:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()