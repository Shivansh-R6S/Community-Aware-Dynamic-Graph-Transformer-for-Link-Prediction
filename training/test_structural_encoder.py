import os
import torch

from models.gnn_encoder import GraphSAGEEncoder
from src.snapshot_builder import build_rolling_cumulative_snapshots
from src.community_module import extract_community_features
from src.data_loader import load_fb_forum


def graph_to_edge_index(G):
    """
    Convert NetworkX graph to edge_index tensor (2, E)
    """
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

    print("Loading FB-forum dataset...")

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, "data", "raw", "fb-forum.txt")

    df, node_mapping = load_fb_forum(data_path)
    num_nodes = len(node_mapping)

    print("Building rolling + cumulative snapshots...")

    snapshots = build_rolling_cumulative_snapshots(
        df,
        num_nodes,
        window_size_days=30,
        step_size_days=15,
        memory_days=60
    )

    print(f"Total snapshots created: {len(snapshots)}")

    # Take first snapshot
    G = snapshots[0]

    print(
        f"Snapshot 0 → Nodes: {G.number_of_nodes()}, "
        f"Edges: {G.number_of_edges()}"
    )

    # Extract community features
    community_features = extract_community_features(G)

    print("Community feature shape:", community_features.shape)

    # Convert graph to edge_index
    edge_index = graph_to_edge_index(G)

    # Initialize GraphSAGE model
    model = GraphSAGEEncoder(
        num_nodes=num_nodes,
        community_dim=community_features.shape[1],
        embed_dim=32,
        hidden_dim=64,
        output_dim=128,
    )

    # Forward pass
    embeddings = model(edge_index, community_features)

    print("Structural Embedding Shape:", embeddings.shape)


if __name__ == "__main__":
    main()
