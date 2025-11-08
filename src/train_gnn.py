# train_gnn.py — Optimized GCN training (Auto GPU + AMP + Precomputed Embeddings)

import os
import torch
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from gnn_model import GCNLinkPredictor

GRAPH_PATH = "../data/processed/graph_data.pt"
MODEL_PATH = "../models/gnn_model.pt"


def train(epochs=10, batch_size=50000):
    print("[INFO] Loading graph data...")
    data = torch.load(GRAPH_PATH, weights_only=False)

    # Auto device detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    data = data.to(device)

    # Initialize model and optimizer
    model = GCNLinkPredictor(data.x.size(1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))  # AMP only if GPU

    num_edges = data.edge_index.size(1)
    print(f"[INFO] Total edges: {num_edges:,}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        # Precompute embeddings once per epoch
        with torch.no_grad():
            node_emb = model.encode(data.x, data.edge_index)

        # Pre-generate negative edges (reuse during epoch)
        neg_edges_all = negative_sampling(data.edge_index, num_nodes=data.num_nodes)

        all_logits, all_labels = [], []
        num_batches = (num_edges + batch_size - 1) // batch_size
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch:02d}", ncols=100)

        for i in pbar:
            start = i * batch_size
            end = min((i + 1) * batch_size, num_edges)
            pos_edges = data.edge_index[:, start:end]
            neg_edges = neg_edges_all[:, start:end] if end <= neg_edges_all.size(1) else neg_edges_all[:, :end - start]

            optimizer.zero_grad(set_to_none=True)

            # Mixed precision forward pass
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                pos_pred = model.decode(node_emb, pos_edges)
                neg_pred = model.decode(node_emb, neg_edges)

                logits = torch.cat([pos_pred, neg_pred], dim=0)
                labels = torch.cat([
                    torch.ones_like(pos_pred),
                    torch.zeros_like(neg_pred)
                ], dim=0)

                loss = loss_fn(logits, labels)

            # AMP backward + step
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            all_logits.append(torch.sigmoid(logits.detach()).cpu())
            all_labels.append(labels.detach().cpu())

            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        # Compute AUC once per epoch (on CPU)
        with torch.no_grad():
            all_logits = torch.cat(all_logits).numpy()
            all_labels = torch.cat(all_labels).numpy()
            auc = roc_auc_score(all_labels, all_logits)

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch:02d} | Avg Loss={avg_loss:.4f} | AUC={auc:.4f}")

        # Save checkpoint every 2 epochs
        if epoch % 2 == 0:
            checkpoint_path = f"../models/gnn_model_epoch{epoch}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"[CHECKPOINT] Saved model to {checkpoint_path}")

    # Save final model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Final model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
