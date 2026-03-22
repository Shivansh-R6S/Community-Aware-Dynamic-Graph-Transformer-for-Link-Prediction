import torch
import networkx as nx
import community as community_louvain


def extract_community_features(G):
    """
    Returns Tensor (N, 6):
    [community_size, intra_degree, inter_degree, degree, clustering, pagerank]
    """
    num_nodes   = G.number_of_nodes()
    partition   = community_louvain.best_partition(G, random_state=42)
    communities = {}
    for node, cid in partition.items():
        communities.setdefault(cid, []).append(node)
    comm_size = {cid: len(nodes) for cid, nodes in communities.items()}

    cs_feat = torch.zeros(num_nodes)
    ia_feat = torch.zeros(num_nodes)
    ie_feat = torch.zeros(num_nodes)

    for node in G.nodes():
        nc = partition[node]
        cs_feat[node] = comm_size[nc]
        ia, ie = 0, 0
        for nb in G.neighbors(node):
            if partition[nb] == nc:
                ia += 1
            else:
                ie += 1
        ia_feat[node] = ia
        ie_feat[node] = ie

    deg_dict  = dict(G.degree())
    clust     = nx.clustering(G)
    pr        = nx.pagerank(G, alpha=0.85, max_iter=100)

    d_feat  = torch.tensor([deg_dict[n]  for n in range(num_nodes)], dtype=torch.float)
    c_feat  = torch.tensor([clust[n]     for n in range(num_nodes)], dtype=torch.float)
    p_feat  = torch.tensor([pr[n]        for n in range(num_nodes)], dtype=torch.float)

    return torch.stack([cs_feat, ia_feat, ie_feat, d_feat, c_feat, p_feat], dim=1)


def extract_all_snapshots_community_features(snapshots):
    print("\nExtracting community + structural features...")
    out = []
    for i, G in enumerate(snapshots):
        print(f"  Snapshot {i+1}/{len(snapshots)}")
        out.append(extract_community_features(G))
    print("Done.")
    return out