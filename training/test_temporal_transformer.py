import os
import sys
import torch

# ------------------------------------------------------------------

# ------------------------------------------------------------------
ROOT_DIR = r"D:\Research Internship\Community-Aware Dynamic Graph Transformer for Link Prediction"
sys.path.append(ROOT_DIR)

# ------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------
import models
print(models.__file__)
from src.data_loader import load_fb_forum
from src.snapshot_builder import build_rolling_cumulative_snapshots
from src.community_module import extract_all_snapshots_community_features
from models.gnn_encoder import GraphSAGEEncoder
from models.temporal_transformer import TemporalTransformer


def graph_to_edge_index(G):
    edges = list(G.edges())

    if len(edges) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    src = []
    dst = []

    for u, v in edges:
        src.append(u)
        dst.append(v)

    return torch.tensor([src, dst], dtype=torch.long)


def main():

    print("\n==== TEMPORAL TRANSFORMER INTEGRATION TEST ====\n")

    # ------------------------------------------------------------------
    # 1️⃣ Load dataset
    # ------------------------------------------------------------------
    data_path = os.path.join(ROOT_DIR, "data", "raw", "fb-forum.txt")
    df, node_mapping = load_fb_forum(data_path)
    num_nodes = len(node_mapping)

    # ------------------------------------------------------------------
    # 2️⃣ Build snapshots
    # ------------------------------------------------------------------
    snapshots = build_rolling_cumulative_snapshots(
        df,
        num_nodes
    )

    print(f"\nTotal snapshots: {len(snapshots)}")

    # ------------------------------------------------------------------
    # 3️⃣ Extract community features
    # ------------------------------------------------------------------
    community_features_list = extract_all_snapshots_community_features(
        snapshots
    )

    # ------------------------------------------------------------------
    # 4️⃣ Initialize GraphSAGE
    # ------------------------------------------------------------------
    gnn = GraphSAGEEncoder(
        num_nodes=num_nodes,
        community_dim=3,
        embed_dim=32,
        hidden_dim=64,
        output_dim=128
    )

    gnn.eval()

    structural_embeddings = []

    # ------------------------------------------------------------------
    # 5️⃣ Generate structural embeddings per snapshot
    # ------------------------------------------------------------------
    with torch.no_grad():
        for i, snapshot in enumerate(snapshots):
            print(f"Processing snapshot {i+1}")

            x = community_features_list[i]
            edge_index = graph_to_edge_index(snapshot)

            z = gnn(edge_index, x)

            print("Structural embedding shape:", z.shape)

            structural_embeddings.append(z)

    # ------------------------------------------------------------------
    # 6️⃣ Stack embeddings for transformer
    # Use first T-1 snapshots for input
    # ------------------------------------------------------------------
    Z_stack = torch.stack(structural_embeddings[:-1])  # (T-1, N, 128)

    print("\nStacked embedding shape:", Z_stack.shape)

    # ------------------------------------------------------------------
    # 7️⃣ Temporal Transformer
    # ------------------------------------------------------------------
    transformer = TemporalTransformer(
        d_model=128,
        n_heads=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        max_time_steps=20
    )

    transformer.eval()

    with torch.no_grad():
        H = transformer(Z_stack)

    print("Final temporal embedding shape:", H.shape)

    print("\n==== TEST COMPLETE ====")


if __name__ == "__main__":
    main()
