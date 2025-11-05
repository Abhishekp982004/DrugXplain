import torch
import pandas as pd
from gnn_model import GATLinkPredictor

GRAPH_PATH = "../data/processed/graph_data.pt"
MODEL_PATH = "../models/gnn_model.pt"
OUT_PATH = "../data/processed/predicted_interactions.csv"

def infer(top_k=200, sample=50000):
    data = torch.load(GRAPH_PATH, weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    model = GATLinkPredictor(data.x.size(1)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    z = model.encode(data.x, data.edge_index)

    pairs = torch.randint(0, data.num_nodes, (2, sample), device=device)
    mask = pairs[0] != pairs[1]
    pairs = pairs[:, mask]

    with torch.no_grad():
        scores = torch.sigmoid(model.decode(z, pairs))

    top_idx = torch.topk(scores, top_k).indices
    top_pairs = pairs[:, top_idx].T.cpu().numpy()
    top_scores = scores[top_idx].cpu().numpy()

    df = pd.DataFrame(top_pairs, columns=["drug_a", "drug_b"])
    df["risk_score"] = top_scores
    df.to_csv(OUT_PATH, index=False)
    print(f"✅ Saved predictions to {OUT_PATH}")

if __name__ == "__main__":
    infer()