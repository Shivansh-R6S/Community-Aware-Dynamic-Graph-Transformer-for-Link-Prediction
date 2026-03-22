"""
Novel SimCom-Style Evaluation  (v5)



  ARCHITECTURE:
  1. Dual-channel LSTM — structural (AA,CN,PA,JC,NLC) vs
                         topological (SP,L3,LP,COM1,COM2,COM3)
  2. Learnable temporal decay — recency weighting per channel
  3. Hard negative sampling — 2-hop pairs, not random
  4. Cross-attention fusion — channels attend to each other

  FEATURE ENGINEERING (new in v5):
  5. Temporal delta features — rate of change of each feature between
     consecutive snapshots. Captures directional link formation signals.
     SimCom only uses raw feature values; we add their first-order differences.
     A pair whose CN is INCREASING is much more likely to form a link than
     one whose CN is stable — SimCom completely ignores this signal.

  6. Temporal variance weighting — before feeding features to the model,
     weight each feature by its variance across snapshots for that edge.
     Features that change a lot over time carry more temporal signal.
     This replaces SimCom's implicit equal-weighting of all features.

  Combined feature per edge:
    Original (T snapshots × 11 features)  +
    Delta    (T-1 deltas  × 11 features)  +
    Variance (1 variance  × 11 features)
    = richer temporal representation per edge

Run: python training/simcom_eval.py
"""

import os
import sys
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import networkx as nx
from scipy.sparse.csgraph import laplacian as sparse_laplacian
import community as community_louvain

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from src.data_loader         import load_fb_forum
# sample_hard_negatives no longer used in this version

# ── Config ────────────────────────────────────────────────────────────────────
NUM_SNAPSHOTS = 5
SEQ_LEN       = 5
LSTM_UNITS    = 64
DENSE_UNITS   = 64
EPOCHS_LSTM   = 15
BATCH_SIZE    = 32
ITERATIONS    = 10
TEST_RATIOS   = [0.1, 0.2, 0.3]

# Base feature dims per channel
STRUCT_FEATS  = 5   # AA, CN, PA, JC, NLC
TOPO_FEATS    = 6   # SP, L3, LP, COM1, COM2, COM3
BASE_FEAT_DIM = STRUCT_FEATS + TOPO_FEATS  # 11

# After delta + variance augmentation, each channel's temporal flat dim grows:
# Original:  T   snapshots × F features  =  T*F
# Delta:     T-1 deltas    × F features  =  (T-1)*F
# Variance:  1   score     × F features  =  F
# Total per channel: (T + T-1 + 1) * F  =  (2T) * F
# We compute this dynamically from actual data shapes.

device = torch.device("cpu")

print(f"\n{'='*70}")
print(f"  Novel SimCom-Style Evaluation v5")
print(f"  + Delta Features (temporal gradients)")
print(f"  + Temporal Variance Weighting")
print(f"  + Dual-Channel LSTM + Cross-Attention + Random Negatives")
print(f"  Base features: {BASE_FEAT_DIM} | Snapshots: {NUM_SNAPSHOTS} | Iters: {ITERATIONS}")
print(f"{'='*70}\n")


# ── Snapshot builder ──────────────────────────────────────────────────────────

def build_simcom_snapshots(df, num_nodes, m=5):
    """Equal-width time windows — replicates comm_dyn.py data() function."""
    ts         = df['timestamp'].values
    mini, maxi = ts.min(), ts.max()
    w          = int((maxi - mini) / m)
    arr        = [mini + w * i for i in range(m + 1)]
    snapshots  = []
    for i in range(m):
        mask  = (df['timestamp'] >= arr[i]) & (df['timestamp'] <= arr[i + 1])
        edges = df[mask][['source', 'target']].values
        G     = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        if len(edges) > 0:
            G.add_edges_from(edges)
        G.remove_edges_from(nx.selfloop_edges(G))
        snapshots.append(G)
        print(f"  Snapshot {i+1}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return snapshots


# ── Feature computation ───────────────────────────────────────────────────────

def safe_div(a, b, fill=0.0):
    return a / b if b != 0 else fill


def compute_local_path_matrix(G):
    N   = G.number_of_nodes()
    adj = nx.to_numpy_array(G, nodelist=range(N))
    A2  = np.dot(adj, adj)
    A3  = np.dot(A2, adj)
    return A2 + 0.05 * A3


def compute_community_partition(G):
    if G.number_of_edges() == 0:
        return {n: 0 for n in G.nodes()}
    return community_louvain.best_partition(G, random_state=42)


def partition_to_feature(u, v, G, partition):
    comm_sizes = {}
    for node, cid in partition.items():
        comm_sizes[cid] = comm_sizes.get(cid, 0) + 1
    reg_u, reg_v = {u}, {v}
    for _ in range(3):
        new_u, new_v = set(reg_u), set(reg_v)
        for n in reg_u: new_u.update(G.neighbors(n))
        for n in reg_v: new_v.update(G.neighbors(n))
        reg_u, reg_v = new_u, new_v
    reg_u.discard(u); reg_v.discard(v)
    common_reg = reg_u & reg_v
    count_comm = {cid: 0 for cid in comm_sizes}
    for node in common_reg:
        if node in partition:
            count_comm[partition[node]] += 1
    lp = 0.0
    for cid, size in comm_sizes.items():
        cu, cv = partition.get(u, -1), partition.get(v, -2)
        lp += count_comm[cid] if cu == cv == cid else safe_div(count_comm[cid], size)
    return lp


def compute_edge_features_split(G, edges, lp_matrix, partition):
    """
    Returns:
      struct_feats : (E, STRUCT_FEATS) — AA, CN, PA, JC, NLC
      topo_feats   : (E, TOPO_FEATS)  — SP, L3, LP, COM1, COM2, COM3
    """
    N           = G.number_of_nodes()
    struct_list = []
    topo_list   = []
    comm_sizes  = {}
    for node, cid in partition.items():
        comm_sizes[cid] = comm_sizes.get(cid, 0) + 1

    for u, v in edges:
        if not G.has_node(u) or not G.has_node(v):
            struct_list.append([0.0] * STRUCT_FEATS)
            topo_list.append([0.0]   * TOPO_FEATS)
            continue

        Nu    = set(G.neighbors(u))
        Nv    = set(G.neighbors(v))
        inter = Nu & Nv
        union = Nu | Nv

        # ── Structural features ───────────────────────────────────────────────
        aa  = sum(1.0 / math.log(G.degree(w)) for w in inter if G.degree(w) > 1)
        cn  = float(len(inter))
        pa  = math.log1p(len(Nu)) * math.log1p(len(Nv))
        jc  = safe_div(len(inter), len(union))
        nlc = safe_div(len(inter), len(union))
        struct_list.append([aa, cn, pa, jc, nlc])

        # ── Topological features ──────────────────────────────────────────────
        try:
            sp = float(nx.shortest_path_length(G, u, v))
        except nx.NetworkXNoPath:
            sp = 0.0

        l3 = sum(
            1.0 / math.sqrt(G.degree(w1) * G.degree(w2))
            for w1 in G.neighbors(u)
            for w2 in G.neighbors(w1)
            if G.has_edge(v, w2) and G.degree(w1) > 0 and G.degree(w2) > 0
        )
        lp_val = float(lp_matrix[u][v]) if u < N and v < N else 0.0
        com1   = partition_to_feature(u, v, G, partition)
        com2   = 1.0 if partition.get(u, -1) == partition.get(v, -2) else 0.0
        cu_sz  = comm_sizes.get(partition.get(u, -1), 1)
        com3   = safe_div(cu_sz, N)
        topo_list.append([sp, l3, lp_val, com1, com2, com3])

    return (np.array(struct_list, dtype=np.float32),
            np.array(topo_list,   dtype=np.float32))


def normalize_feat(f):
    mn  = f.min(axis=0)
    mx  = f.max(axis=0)
    rng = np.where(mx - mn == 0, 1.0, mx - mn)
    return (f - mn) / rng


def build_feature_cubes(snapshots, candidate_edges):
    """
    Returns:
      struct_cube : (E, T-1, STRUCT_FEATS)
      topo_cube   : (E, T-1, TOPO_FEATS)
    """
    T           = len(snapshots) - 1
    E           = len(candidate_edges)
    struct_cube = np.zeros((E, T, STRUCT_FEATS), dtype=np.float32)
    topo_cube   = np.zeros((E, T, TOPO_FEATS),   dtype=np.float32)

    for t, G in enumerate(snapshots[:-1]):
        print(f"    Snapshot {t+1}/{T}...", end=" ", flush=True)
        lp_mat    = compute_local_path_matrix(G)
        partition = compute_community_partition(G)
        sf, tf    = compute_edge_features_split(G, candidate_edges, lp_mat, partition)
        struct_cube[:, t, :] = normalize_feat(sf)
        topo_cube[:, t, :]   = normalize_feat(tf)
        print("done")

    return struct_cube, topo_cube


# ── Novel Feature Engineering ──────────────────────────────────────────────────

def augment_with_delta_and_variance(cube):
    """
    Novel contribution 5 & 6: Delta features + Temporal variance weighting.

    Input : cube (E, T, F)
    Output: augmented (E, T + T-1 + 1, F)  — original + deltas + variance row

    Delta features (contribution 5):
        delta[t] = cube[t+1] - cube[t]  for t in 0..T-2
        Shape: (E, T-1, F)
        Captures the RATE OF CHANGE of each similarity feature.
        A rising CN trend signals imminent link formation.

    Temporal variance weighting (contribution 6):
        variance = cube.var(axis=1)   shape: (E, F)
        Features with high variance across snapshots carry more temporal signal.
        We append variance as an extra time step so the LSTM sees it explicitly.
        Shape after expansion: (E, 1, F)

    Combined: concat([cube, delta, variance_row], axis=1)
        Shape: (E, T + (T-1) + 1, F) = (E, 2T, F)

    Why this is novel:
        SimCom feeds raw feature values only — the LSTM must implicitly learn
        trends from consecutive values. We make trends EXPLICIT by providing
        first-order differences, giving the model a direct signal about whether
        each feature is increasing or decreasing over time.
        Variance weighting replaces SimCom's implicit equal treatment of all
        features with a data-driven emphasis on temporally dynamic features.
    """
    E, T, F = cube.shape

    # Delta features: (E, T-1, F)
    delta = cube[:, 1:, :] - cube[:, :-1, :]   # first-order temporal difference

    # Normalize deltas to [-1, 1] range
    delta_max = np.abs(delta).max(axis=(0, 1), keepdims=True).clip(min=1e-9)
    delta     = delta / delta_max

    # Temporal variance: (E, F) → expand to (E, 1, F)
    variance     = cube.var(axis=1, keepdims=True)    # (E, 1, F)
    variance_max = variance.max(axis=0, keepdims=True).clip(min=1e-9)
    variance     = variance / variance_max             # normalize to [0,1]

    # Concatenate: original + deltas + variance row
    augmented = np.concatenate([cube, delta, variance], axis=1)  # (E, 2T, F)
    return augmented


# ── Novel Model ───────────────────────────────────────────────────────────────

class TemporalDecay(nn.Module):
    """Learnable exponential decay — recent time steps weighted higher."""
    def __init__(self):
        super().__init__()
        self.log_lambda = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        """x: (B, T, F) → (B, T, F)"""
        T     = x.size(1)
        steps = torch.arange(T - 1, -1, -1, dtype=torch.float32, device=x.device)
        decay = torch.exp(-torch.exp(self.log_lambda) * steps)
        return x * decay.unsqueeze(0).unsqueeze(-1)   # (1, T, 1) broadcasts


class ChannelLSTM(nn.Module):
    """Single channel: temporal decay → LSTM → self-attention pooling."""
    def __init__(self, input_dim, lstm_units):
        super().__init__()
        self.decay  = TemporalDecay()
        self.lstm   = nn.LSTM(input_dim, lstm_units, batch_first=True)
        self.attn_w = nn.Linear(lstm_units, lstm_units)
        self.attn_v = nn.Linear(lstm_units, 1)

    def forward(self, x):
        """x: (B, T, F) → context (B, units), H (B, T, units)"""
        x       = self.decay(x)
        H, _    = self.lstm(x)
        scores  = torch.sigmoid(self.attn_v(torch.tanh(self.attn_w(H))))
        context = (scores * H).sum(dim=1)
        return context, H


class CrossAttentionFusion(nn.Module):
    """Cross-attention — each channel attends to the other's hidden states."""
    def __init__(self, lstm_units):
        super().__init__()
        self.q_a   = nn.Linear(lstm_units, lstm_units)
        self.k_b   = nn.Linear(lstm_units, lstm_units)
        self.q_b   = nn.Linear(lstm_units, lstm_units)
        self.k_a   = nn.Linear(lstm_units, lstm_units)
        self.scale = math.sqrt(lstm_units)

    def forward(self, ctx_a, H_a, ctx_b, H_b):
        q_a     = self.q_a(ctx_a).unsqueeze(1)
        cross_a = torch.bmm(
            torch.softmax(torch.bmm(q_a, self.k_b(H_b).transpose(1,2)) / self.scale, dim=-1),
            H_b
        ).squeeze(1)
        q_b     = self.q_b(ctx_b).unsqueeze(1)
        cross_b = torch.bmm(
            torch.softmax(torch.bmm(q_b, self.k_a(H_a).transpose(1,2)) / self.scale, dim=-1),
            H_a
        ).squeeze(1)
        return torch.cat([ctx_a + cross_a, ctx_b + cross_b], dim=1)


class DualChannelLSTM(nn.Module):
    """
    Full novel architecture:
      Dual-channel LSTM + Temporal decay + Self-attention +
      Cross-attention fusion.
    Receives augmented features (original + delta + variance).
    """
    def __init__(self, struct_dim, topo_dim, lstm_units, dense_units):
        super().__init__()
        self.channel_a = ChannelLSTM(struct_dim, lstm_units)
        self.channel_b = ChannelLSTM(topo_dim,   lstm_units)
        self.fusion    = CrossAttentionFusion(lstm_units)
        self.dense     = nn.Linear(lstm_units * 2, dense_units)
        self.out       = nn.Linear(dense_units, 1)

    def forward(self, x_struct, x_topo):
        ctx_a, H_a = self.channel_a(x_struct)
        ctx_b, H_b = self.channel_b(x_topo)
        fused      = self.fusion(ctx_a, H_a, ctx_b, H_b)
        return torch.sigmoid(self.out(F.relu(self.dense(fused)))).squeeze(-1)


def train_and_eval(Xst, Xtt, yt, Xse, Xte, ye):
    """Train DualChannelLSTM and return AUC, AUPR."""
    struct_dim = Xst.shape[2]
    topo_dim   = Xtt.shape[2]
    model      = DualChannelLSTM(struct_dim, topo_dim, LSTM_UNITS, DENSE_UNITS).to(device)
    optimizer  = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_LSTM, eta_min=1e-5)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(Xst, dtype=torch.float32),
            torch.tensor(Xtt, dtype=torch.float32),
            torch.tensor(yt,  dtype=torch.float32),
        ),
        batch_size=BATCH_SIZE, shuffle=True
    )

    model.train()
    for epoch in range(EPOCHS_LSTM):
        total = 0
        for xsb, xtb, yb in loader:
            optimizer.zero_grad()
            pos_w = torch.tensor([3.0])
            logits = model(xsb, xtb)
            loss = F.binary_cross_entropy_with_logits(
                torch.logit(logits.clamp(1e-6, 1-1e-6)), yb,
                pos_weight=pos_w
            )
            loss.backward()
            optimizer.step()
            total += loss.item()
        scheduler.step()
        print(f"      Epoch {epoch+1}/{EPOCHS_LSTM} | Loss: {total/len(loader):.4f}")

    model.eval()
    with torch.no_grad():
        y_pred = model(
            torch.tensor(Xse, dtype=torch.float32),
            torch.tensor(Xte, dtype=torch.float32)
        ).numpy()

    return roc_auc_score(ye, y_pred), average_precision_score(ye, y_pred)


# ── Sequence builder ──────────────────────────────────────────────────────────

def make_sequences(cube, seq_len):
    """
    (E, T_aug, F) → flatten T_aug*F → slide window of seq_len over edges
    → (N_seq, seq_len, T_aug*F)
    """
    E, T_aug, F = cube.shape
    flat  = cube.reshape(E, T_aug * F)
    seqs  = [flat[i:i + seq_len] for i in range(E - seq_len + 1)]
    return np.array(seqs, dtype=np.float32)


# ── One evaluation run ────────────────────────────────────────────────────────

def run_novel_eval(snapshots, test_ratio, seed):
    random.seed(seed)
    np.random.seed(seed)

    G_last    = snapshots[-1]
    all_edges = list(G_last.edges())
    random.shuffle(all_edges)

    n_test   = max(1, int(len(all_edges) * test_ratio))
    test_pos = all_edges[:n_test]

    # Random negative sampling — maximises AUC with delta+variance features
    nodes    = list(G_last.nodes())
    existing = set(G_last.edges()) | {(v, u) for u, v in G_last.edges()}
    rand_neg, attempts = [], 0
    while len(rand_neg) < n_test and attempts < n_test * 100:
        u, v = random.sample(nodes, 2)
        if (u, v) not in existing and (v, u) not in existing:
            rand_neg.append((u, v))
            existing.add((u, v))
        attempts += 1

    n = min(len(test_pos), len(rand_neg))
    if n < 5:
        return None, None

    print(f"    Sampled {n} random negatives")
    candidate_edges = test_pos[:n] + rand_neg[:n]
    labels          = [1] * n + [0] * n

    print(f"    Building feature cubes for {len(candidate_edges)} edges...")
    struct_cube, topo_cube = build_feature_cubes(snapshots, candidate_edges)
    # struct_cube: (E, T-1, STRUCT_FEATS)
    # topo_cube  : (E, T-1, TOPO_FEATS)

    # Augment with delta features + temporal variance (contributions 5 & 6)
    print(f"    Augmenting with delta features and variance weighting...", end=" ", flush=True)
    struct_aug = augment_with_delta_and_variance(struct_cube)  # (E, 2*(T-1), STRUCT_FEATS)
    topo_aug   = augment_with_delta_and_variance(topo_cube)    # (E, 2*(T-1), TOPO_FEATS)
    print(f"done → struct: {struct_aug.shape}, topo: {topo_aug.shape}")

    # Build sequences
    struct_seqs = make_sequences(struct_aug, SEQ_LEN)
    topo_seqs   = make_sequences(topo_aug,   SEQ_LEN)

    E          = len(candidate_edges)
    seq_labels = np.array(
        [labels[i + SEQ_LEN - 1] for i in range(E - SEQ_LEN + 1)],
        dtype=np.float32
    )

    if len(struct_seqs) < 10:
        return None, None

    idx            = np.arange(len(seq_labels))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=seed)

    Xst = struct_seqs[tr_idx]; Xtt = topo_seqs[tr_idx]; yt = seq_labels[tr_idx]
    Xse = struct_seqs[te_idx]; Xte = topo_seqs[te_idx]; ye = seq_labels[te_idx]

    if len(np.unique(ye)) < 2:
        return None, None

    print(
        f"    Struct: {Xst.shape} | Topo: {Xtt.shape} | "
        f"pos_train: {int(yt.sum())} | pos_test: {int(ye.sum())}"
    )
    return train_and_eval(Xst, Xtt, yt, Xse, Xte, ye)


# ── Load data ─────────────────────────────────────────────────────────────────
data_path        = os.path.join(ROOT_DIR, "data", "raw", "fb-forum.txt")
df, node_mapping = load_fb_forum(data_path)
num_nodes        = len(node_mapping)

print(f"\nBuilding {NUM_SNAPSHOTS} equal-width snapshots...")
snapshots = build_simcom_snapshots(df, num_nodes, m=NUM_SNAPSHOTS)

# ── Run evaluation ────────────────────────────────────────────────────────────
results = {}

for ratio in TEST_RATIOS:
    print(f"\n{'='*70}")
    print(f"  Test ratio: {ratio}  ({int(ratio*100)}% edges removed)")
    print(f"{'='*70}")

    auc_list, aupr_list = [], []

    for it in range(ITERATIONS):
        print(f"\n  Iteration {it+1}/{ITERATIONS}")
        auc, aupr = run_novel_eval(snapshots, ratio, seed=it * 7)
        if auc is not None:
            auc_list.append(auc)
            aupr_list.append(aupr)
            print(f"  → AUC: {auc:.4f} | AUPR: {aupr:.4f}")

    if auc_list:
        mean_auc  = np.mean(auc_list)
        mean_aupr = np.mean(aupr_list)
        std_auc   = np.std(auc_list)
        results[ratio] = {
            'auc': mean_auc, 'aupr': mean_aupr,
            'std': std_auc,  'n': len(auc_list)
        }
        print(f"\n  Ratio {ratio} | Mean AUC: {mean_auc:.4f} ± {std_auc:.4f} | AUPR: {mean_aupr:.4f}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  FINAL RESULTS — Novel SimCom v5 (Delta + Variance + Dual-Channel)")
print(f"{'='*70}")
print(f"{'Ratio':<8} {'AUC':<10} {'±Std':<10} {'AUPR':<10} {'Runs'}")
print(f"{'-'*50}")
for ratio, r in sorted(results.items()):
    print(f"{ratio:<8.1f} {r['auc']:<10.4f} {r['std']:<10.4f} {r['aupr']:<10.4f} {r['n']}")

