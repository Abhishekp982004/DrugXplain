# train_gnn.py — CPU-only GCN training (no pyg-lib)
import os
import torch
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score
from gnn_model import GCNLinkPredictor
from tqdm import tqdm  # ✅ Progress bar

GRAPH_PATH = "../data/processed/graph_data.pt"
MODEL_PATH = "../models/gnn_model.pt"


def train(epochs=10, batch_size=5000):
    print("[INFO] Loading graph...")
    data = torch.load(GRAPH_PATH, weights_only=False)

    # ✅ Force CPU for full compatibility
    device = torch.device("cpu")
    data = data.to(device)
    print(f"[INFO] Using device: {device}")

    model = GCNLinkPredictor(data.x.size(1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    print("[INFO] Starting CPU training (safe mode)...")

    num_edges = data.edge_index.size(1)
    print(f"[INFO] Total edges: {num_edges:,}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_auc = 0.0
        num_batches = (num_edges + batch_size - 1) // batch_size

        # ✅ tqdm progress bar for each epoch
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch:02d}", ncols=100)
        for i in pbar:
            start = i * batch_size
            end = min((i + 1) * batch_size, num_edges)
            batch_edges = data.edge_index[:, start:end]

            neg_edges = negative_sampling(batch_edges, num_nodes=data.num_nodes)
            pos_pred = model(data.x, data.edge_index, batch_edges)
            neg_pred = model(data.x, data.edge_index, neg_edges)

            labels = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)])
            logits = torch.cat([pos_pred, neg_pred])

            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                auc = roc_auc_score(labels.numpy(), torch.sigmoid(logits).numpy())

            total_loss += loss.item()
            total_auc += auc
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "AUC": f"{auc:.4f}"})

        avg_loss = total_loss / num_batches
        avg_auc = total_auc / num_batches
        print(f"✅ Epoch {epoch:02d} | Avg Loss={avg_loss:.4f} | Avg AUC={avg_auc:.4f}")

        # ✅ Save checkpoint every 2 epochs
        if epoch % 2 == 0:
            checkpoint_path = f"../models/gnn_model_epoch{epoch}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"[CHECKPOINT] Saved model to {checkpoint_path}")

    # ✅ Final model save
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"🎯 Final model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
