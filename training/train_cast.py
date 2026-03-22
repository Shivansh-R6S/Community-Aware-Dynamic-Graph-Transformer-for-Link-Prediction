"""
CAST — Community-Aware Structural Transformer  (v8)
============================================================
Key insight from v7 logs:
  - Model hit 0.8741 at epoch 38, was still climbing
  - Pool refresh at epoch 46 knocked it to 0.70
  - Never recovered before early stopping at epoch 78

Fix:
  - POOL_FREEZE_EPOCH = 40  (freeze before the epoch 46 refresh)
  - POOL_REFRESH_EVERY = 20 (less frequent refreshes while active)
  - PATIENCE = 50           (more runway to push past 0.87)
  - T_max = 250             (cosine LR over full run)
============================================================
"""

import os, sys, gc
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from src.data_loader             import load_fb_forum
from src.snapshot_builder        import build_rolling_cumulative_snapshots
from src.community_module        import extract_all_snapshots_community_features
from src.edge_features           import extract_temporal_edge_features
from models.gnn_encoder          import GraphSAGEEncoder
from models.temporal_transformer import TemporalTransformer
from models.edge_lstm            import EdgeTemporalEncoder
from models.mlp_decoder          import FusionDecoder
from utils.negative_sampling     import sample_hard_negatives

# ── Config ────────────────────────────────────────────────────────────────────
LITE_MODE = True

if LITE_MODE:
    EMBED_DIM        = 32
    HIDDEN_DIM       = 64
    OUTPUT_DIM       = 64
    TF_HEADS         = 4
    TF_LAYERS        = 2
    TF_FFN           = 128
    LSTM_UNITS       = 32
    EDGE_CONTEXT_DIM = 32
    DEC_HIDDEN       = 128
    EPOCHS           = 250
    PATIENCE         = 50
else:
    EMBED_DIM        = 64
    HIDDEN_DIM       = 256
    OUTPUT_DIM       = 256
    TF_HEADS         = 8
    TF_LAYERS        = 3
    TF_FFN           = 512
    LSTM_UNITS       = 64
    EDGE_CONTEXT_DIM = 64
    DEC_HIDDEN       = 256
    EPOCHS           = 350
    PATIENCE         = 60

COMMUNITY_DIM      = 6
EDGE_FEAT_DIM      = 5
LR                 = 0.001
EDGE_ENC_WARMUP    = 10
NEG_POOL_MULT      = 5
POOL_REFRESH_EVERY = 20   # less frequent — reduces disruption
POOL_FREEZE_EPOCH  = 40   # freeze before model peaks — based on v7 logs
LABEL_SMOOTHING    = 0.05
device             = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\n{'='*55}")
print(f"  CAST v8  |  {'LITE' if LITE_MODE else 'FULL'}  |  {device}")
print(f"  Pool freezes at epoch {POOL_FREEZE_EPOCH}")
print(f"  Patience: {PATIENCE} epochs")
print(f"{'='*55}\n")


# ── Helpers ───────────────────────────────────────────────────────────────────
def graph_to_edge_index(G):
    edges = list(G.edges())
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    src, dst = zip(*edges)
    return torch.tensor([src, dst], dtype=torch.long)


def weighted_bce(logits, labels):
    smooth = labels * (1 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
    pos_w  = torch.tensor([2.0], device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, smooth, pos_weight=pos_w)


def build_neg_pool(target_snapshot, n_train, snapshots, device):
    pool_size  = n_train * NEG_POOL_MULT
    neg_list   = sample_hard_negatives(target_snapshot, pool_size)
    neg_tensor = torch.tensor(neg_list, dtype=torch.long).t().to(device)
    neg_ef     = extract_temporal_edge_features(snapshots, neg_tensor, device=device)
    return neg_tensor, neg_ef


# ── 1. Data ───────────────────────────────────────────────────────────────────
data_path        = os.path.join(ROOT_DIR, "data", "raw", "fb-forum.txt")
df, node_mapping = load_fb_forum(data_path)
num_nodes        = len(node_mapping)

# ── 2. Snapshots ──────────────────────────────────────────────────────────────
snapshots = build_rolling_cumulative_snapshots(df, num_nodes)
T         = len(snapshots)
print(f"Total snapshots: {T}")

# ── 3. Node features ──────────────────────────────────────────────────────────
community_features_list = extract_all_snapshots_community_features(snapshots)

# ── 4. Target + splits ────────────────────────────────────────────────────────
target_snapshot  = snapshots[-1]
target_edges     = graph_to_edge_index(target_snapshot).to(device)
edge_np          = target_edges.t().cpu().numpy()
train_np, val_np = train_test_split(edge_np, test_size=0.2, random_state=42)
train_edges      = torch.tensor(train_np).t().contiguous().to(device)
val_edges        = torch.tensor(val_np).t().contiguous().to(device)
n_train          = train_edges.size(1)
n_val            = val_edges.size(1)

# ── 5. Pre-compute adjacency matrices ─────────────────────────────────────────
print("\nPre-computing adjacency matrices...")
adj_norm_list = []
for i, G in enumerate(snapshots[:-1]):
    ei = graph_to_edge_index(G).to(device)
    adj_norm_list.append(GraphSAGEEncoder.build_adj_norm(ei, num_nodes, device))
    print(f"  Adj {i+1}/{T-1}")
print("Done.\n")

# ── 6. Positive edge features ─────────────────────────────────────────────────
print("Pre-computing positive edge features...")
pos_ef_train = extract_temporal_edge_features(snapshots, train_edges, device=device)
pos_ef_val   = extract_temporal_edge_features(snapshots, val_edges,   device=device)

# ── 7. Negative pools ─────────────────────────────────────────────────────────
print("\nBuilding initial negative pool (train)...")
neg_pool_tensor, neg_ef_pool = build_neg_pool(
    target_snapshot, n_train, snapshots, device
)

print("\nBuilding initial negative pool (val)...")
val_pool_list   = sample_hard_negatives(target_snapshot, n_val * NEG_POOL_MULT)
val_pool_tensor = torch.tensor(val_pool_list, dtype=torch.long).t().to(device)
val_neg_ef_pool = extract_temporal_edge_features(
    snapshots, val_pool_tensor, device=device
)

print(f"\n  pos_ef_train : {pos_ef_train.shape}")
print(f"  neg_ef_pool  : {neg_ef_pool.shape}")
print("="*55 + "\n")

# ── 8. Models ─────────────────────────────────────────────────────────────────
encoder = GraphSAGEEncoder(
    num_nodes=num_nodes, community_dim=COMMUNITY_DIM,
    embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM,
).to(device)

temporal_model = TemporalTransformer(
    d_model=OUTPUT_DIM, n_heads=TF_HEADS, num_layers=TF_LAYERS,
    dim_feedforward=TF_FFN, dropout=0.1, max_time_steps=20, pooling='mean',
).to(device)

edge_encoder = EdgeTemporalEncoder(
    feat_dim=EDGE_FEAT_DIM, lstm_units=LSTM_UNITS,
    output_dim=EDGE_CONTEXT_DIM, num_layers=1, dropout=0.1,
).to(device)

decoder = FusionDecoder(
    embed_dim=OUTPUT_DIM, edge_context_dim=EDGE_CONTEXT_DIM,
    hidden_dim=DEC_HIDDEN, dropout=0.2,
).to(device)

gnn_params  = (list(encoder.parameters()) +
               list(temporal_model.parameters()) +
               list(decoder.parameters()))
edge_params = list(edge_encoder.parameters())
all_params  = gnn_params + edge_params

optimizer = torch.optim.Adam(all_params, lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-5
)

best_model_path  = os.path.join(ROOT_DIR, "best_cast_model.pt")
best_val_auc     = 0.0
best_epoch       = 0
patience_counter = 0

# ── 9. Training loop ──────────────────────────────────────────────────────────
for epoch in range(1, EPOCHS + 1):

    frozen = (epoch <= EDGE_ENC_WARMUP)
    for p in edge_encoder.parameters():
        p.requires_grad = not frozen

    # Refresh pool only before POOL_FREEZE_EPOCH
    if (epoch > 1
            and epoch <= POOL_FREEZE_EPOCH
            and (epoch - 1) % POOL_REFRESH_EVERY == 0):
        print(f"  [Epoch {epoch}] Refreshing negative pool...")
        neg_pool_tensor, neg_ef_pool = build_neg_pool(
            target_snapshot, n_train, snapshots, device
        )
    elif epoch == POOL_FREEZE_EPOCH + 1:
        print(f"  [Epoch {epoch}] *** Pool frozen — consolidating from here ***")

    encoder.train(); temporal_model.train()
    edge_encoder.train(); decoder.train()
    optimizer.zero_grad()

    # ── Stream A ──────────────────────────────────────────────────────────────
    struct_embs = []
    for i in range(T - 1):
        struct_embs.append(
            encoder(adj_norm_list[i], community_features_list[i].to(device))
        )
    H = temporal_model(torch.stack(struct_embs))

    # ── Negative sample ───────────────────────────────────────────────────────
    perm            = torch.randperm(neg_ef_pool.size(1))[:n_train]
    neg_ef_epoch    = neg_ef_pool[:, perm, :]
    neg_edges_epoch = neg_pool_tensor[:, perm]

    # ── Stream B ──────────────────────────────────────────────────────────────
    pos_ctx = edge_encoder(pos_ef_train.permute(1, 0, 2))
    neg_ctx = edge_encoder(neg_ef_epoch.permute(1, 0, 2))

    # ── Loss ──────────────────────────────────────────────────────────────────
    pos_logits = decoder(H, train_edges,     pos_ctx)
    neg_logits = decoder(H, neg_edges_epoch, neg_ctx)
    logits     = torch.cat([pos_logits, neg_logits])
    labels     = torch.cat([
        torch.ones(pos_logits.size(0)),
        torch.zeros(neg_logits.size(0))
    ]).to(device)

    loss = weighted_bce(logits, labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
    optimizer.step()
    scheduler.step()

    # ── Validation ────────────────────────────────────────────────────────────
    encoder.eval(); temporal_model.eval()
    edge_encoder.eval(); decoder.eval()

    with torch.no_grad():
        val_ctx = edge_encoder(pos_ef_val.permute(1, 0, 2))

        val_perm      = torch.randperm(val_neg_ef_pool.size(1))[:n_val]
        val_neg_ef_ep = val_neg_ef_pool[:, val_perm, :]
        val_neg_edges = val_pool_tensor[:, val_perm]
        val_neg_ctx   = edge_encoder(val_neg_ef_ep.permute(1, 0, 2))

        pv = decoder(H, val_edges,     val_ctx)
        nv = decoder(H, val_neg_edges, val_neg_ctx)

        v_probs     = torch.sigmoid(torch.cat([pv, nv])).cpu().numpy()
        v_labels_np = torch.cat([
            torch.ones(pv.size(0)), torch.zeros(nv.size(0))
        ]).numpy()

        val_auc = roc_auc_score(v_labels_np, v_probs)
        val_ap  = average_precision_score(v_labels_np, v_probs)

    current_lr  = scheduler.get_last_lr()[0]
    pool_status = "fixed" if epoch > POOL_FREEZE_EPOCH else "refresh"
    tag         = f"[{'frozen' if frozen else 'active'}|{pool_status}]"

    print(
        f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | "
        f"Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f} | "
        f"LR: {current_lr:.6f} {tag}"
    )

    if val_auc > best_val_auc:
        best_val_auc = val_auc; best_epoch = epoch; patience_counter = 0
        torch.save({
            'encoder':      encoder.state_dict(),
            'temporal':     temporal_model.state_dict(),
            'edge_encoder': edge_encoder.state_dict(),
            'decoder':      decoder.state_dict(),
            'val_auc':      val_auc, 'val_ap': val_ap,
        }, best_model_path)
        print(f"  ✓ Best saved  (AUC: {best_val_auc:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    del struct_embs
    gc.collect()

print(f"\n{'='*55}")
print(f"  Best Val AUC : {best_val_auc:.4f}  (epoch {best_epoch})")
print(f"{'='*55}\n")