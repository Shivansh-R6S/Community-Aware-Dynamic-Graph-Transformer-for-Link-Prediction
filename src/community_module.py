import torch
import networkx as nx
import community as community_louvain


def extract_community_features(G):
    """
    Extract community-aware features from a NetworkX graph.

    Returns:
        Tensor of shape [num_nodes, 3]
        Columns:
        [community_size, intra_degree, inter_degree]
    """

    num_nodes = G.number_of_nodes()

    # Run Louvain
    partition = community_louvain.best_partition(G)

    # Build community groups
    communities = {}
    for node, comm_id in partition.items():
        communities.setdefault(comm_id, []).append(node)

    # Precompute community sizes
    community_size = {
        comm_id: len(nodes)
        for comm_id, nodes in communities.items()
    }

    # Feature tensors
    community_size_feat = torch.zeros(num_nodes)
    intra_degree_feat = torch.zeros(num_nodes)
    inter_degree_feat = torch.zeros(num_nodes)

    for node in G.nodes():

        node_comm = partition[node]
        community_size_feat[node] = community_size[node_comm]

        intra_deg = 0
        inter_deg = 0

        for neighbor in G.neighbors(node):
            if partition[neighbor] == node_comm:
                intra_deg += 1
            else:
                inter_deg += 1

        intra_degree_feat[node] = intra_deg
        inter_degree_feat[node] = inter_deg

    features = torch.stack(
        [
            community_size_feat,
            intra_degree_feat,
            inter_degree_feat
        ],
        dim=1
    )

    return features


def extract_all_snapshots_community_features(snapshots):

    print("\nExtracting community features...")

    features_list = []

    for i, G in enumerate(snapshots):
        print(f"Processing snapshot {i+1}/{len(snapshots)}")
        features = extract_community_features(G)
        features_list.append(features)

    print("Community extraction complete.")

    return features_list


if __name__ == "__main__":

    from snapshot_builder import build_rolling_cumulative_snapshots
    from data_loader import load_fb_forum
    import os

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, "data", "raw", "fb-forum.txt")

    df, node_mapping = load_fb_forum(data_path)
    num_nodes = len(node_mapping)

    snapshots = build_rolling_cumulative_snapshots(df, num_nodes)

    features_list = extract_all_snapshots_community_features(snapshots)

    print("\nFeature shape for first snapshot:")
    print(features_list[0].shape)
