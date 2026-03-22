import torch
import numpy as np


# ---------------------------------------------------------------------------
# CAST — Stream B: handcrafted edge-level similarity features
# ---------------------------------------------------------------------------
# Per candidate edge (u, v) per snapshot:
#   0  CN   — common neighbours
#   1  JC   — Jaccard coefficient
#   2  AA   — Adamic / Adar index
#   3  PA   — preferential attachment (log-scaled)
#   4  NLC  — node & link clustering coefficient
# ---------------------------------------------------------------------------

FEATURE_DIM = 5


def _safe_div(a, b, fill=0.0):
    return a / b if b != 0 else fill


def compute_edge_features(G, edge_list):
    """
    Compute 5 similarity features for a list of (u, v) pairs.

    Parameters
    ----------
    G         : nx.Graph
    edge_list : Tensor (2, E) or list of (u, v) tuples

    Returns
    -------
    Tensor (E, 5) float32
    """
    if isinstance(edge_list, torch.Tensor):
        pairs = edge_list.t().tolist()
    else:
        pairs = [(int(u), int(v)) for u, v in edge_list]

    feats = []
    for u, v in pairs:
        Nu = set(G.neighbors(u)) if G.has_node(u) else set()
        Nv = set(G.neighbors(v)) if G.has_node(v) else set()

        inter = Nu & Nv
        union = Nu | Nv

        cn  = float(len(inter))
        jc  = _safe_div(len(inter), len(union))
        aa  = sum(
            1.0 / np.log(G.degree(w) + 1e-9)
            for w in inter if G.degree(w) > 1
        )
        pa  = float(np.log1p(len(Nu)) * np.log1p(len(Nv)))
        nlc = _safe_div(len(inter), len(union))

        feats.append([cn, jc, aa, pa, nlc])

    return torch.tensor(feats, dtype=torch.float32)   # (E, 5)


def normalize_edge_features(feat_tensor):
    """Min-max normalise each column to [0, 1]."""
    mn  = feat_tensor.min(dim=0).values
    mx  = feat_tensor.max(dim=0).values
    rng = (mx - mn).clamp(min=1e-9)
    return (feat_tensor - mn) / rng


def extract_temporal_edge_features(snapshots, edge_index, device='cpu'):
    """
    Extract edge features across ALL T-1 snapshots → temporal sequence.

    This captures HOW similarity metrics evolve over time for each
    candidate pair — the key signal that pushes AUC toward 0.99.

    Parameters
    ----------
    snapshots  : List[nx.Graph]  — all snapshots (last = target)
    edge_index : Tensor (2, E)
    device     : str

    Returns
    -------
    Tensor (T-1, E, 5)
    """
    print("\nExtracting temporal edge feature sequence (Stream B)...")
    all_feats = []

    for i, G in enumerate(snapshots[:-1]):
        print(f"  Snapshot {i+1}/{len(snapshots)-1}")
        ef = compute_edge_features(G, edge_index)
        all_feats.append(ef)

    stacked      = torch.stack(all_feats)    # (T-1, E, 5)
    T, E, F      = stacked.shape
    flat         = normalize_edge_features(stacked.view(-1, F))
    stacked      = flat.view(T, E, F)

    print("Temporal edge features complete.")
    return stacked.to(device)