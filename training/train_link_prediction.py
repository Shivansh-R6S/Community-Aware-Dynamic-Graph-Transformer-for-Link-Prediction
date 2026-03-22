import os
import sys
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------
# Fix root path
# ---------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

# ---------------------------------------------------
# Imports
# ---------------------------------------------------
from src.data_loader import load_fb_forum
from src.snapshot_builder import build_rolling_cumulative_snapshots
from src.community_module import extract_all_snapshots_community_features

from models.gnn_encoder import GraphSAGEEncoder
from models.temporal_transformer import TemporalTransformer
from models.mlp_decoder import MLPDecoder

from utils.negative_sampling import (
    sample_hard_negatives,
    sample_random_negatives
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def graph_to_edge_index(G):
    edges = list(G.edges())
    if len(edges) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    src, dst = zip(*edges)
    return torch.tensor([src, dst], dtype=torch.long)


print("\n==== CADGT Dynamic Training ====\n")

# ---------------------------------------------------
# Load dataset
# ---------------------------------------------------
data_path = os.path.join(ROOT_DIR, "data", "raw", "fb-forum.txt")
df, node_mapping = load_fb_forum(data_path)
num_nodes = len(node_mapping)

# ---------------------------------------------------
# Build snapshots
# ---------------------------------------------------
snapshots = build_rolling_cumulative_snapshots(df, num_nodes)

# ---------------------------------------------------
# Community features
# ---------------------------------------------------
community_features_list = extract_all_snapshots_community_features(snapshots)

# ---------------------------------------------------
# Models
# ---------------------------------------------------
encoder = GraphSAGEEncoder(
    num_nodes=num_nodes,
    community_dim=3,
    embed_dim=32,
    hidden_dim=128,
    output_dim=128
).to(device)

temporal_model = TemporalTransformer(
    d_model=128,
    n_heads=4,
    num_layers=2,
    dim_feedforward=256,
    dropout=0.1,
    max_time_steps=20
).to(device)

decoder = MLPDecoder(
    input_dim=128,
    hidden_dim=256,
    dropout=0.3
).to(device)

optimizer = torch.optim.Adam(
    list(encoder.parameters()) +
    list(temporal_model.parameters()) +
    list(decoder.parameters()),
    lr=0.001
)

epochs = 50

# ---------------------------------------------------
# Target snapshot
# ---------------------------------------------------
target_snapshot = snapshots[-1]
target_edges = graph_to_edge_index(target_snapshot).to(device)

edge_list = target_edges.t().cpu().numpy()

train_edges_np, val_edges_np = train_test_split(
    edge_list,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

train_edges = torch.tensor(train_edges_np).t().to(device)
val_edges = torch.tensor(val_edges_np).t().to(device)

# Edge set for random negatives
edge_set = set((u, v) for u, v in edge_list)

# ---------------------------------------------------
# Training Loop
# ---------------------------------------------------
for epoch in range(1, epochs + 1):

    encoder.train()
    temporal_model.train()
    decoder.train()
    optimizer.zero_grad()

    structural_embeddings = []

    # -----------------------------------------------
    # Structural embeddings (snapshots 1–10)
    # -----------------------------------------------
    for i in range(len(snapshots) - 1):

        G = snapshots[i]
        x = community_features_list[i].to(device)

        edge_index = graph_to_edge_index(G)

        adjacency_dict = GraphSAGEEncoder.build_adjacency_dict(
            edge_index, num_nodes
        )

        z = encoder(adjacency_dict, x)
        structural_embeddings.append(z)

    Z_stack = torch.stack(structural_embeddings)

    # -----------------------------------------------
    # Temporal Transformer
    # -----------------------------------------------
    H = temporal_model(Z_stack)

    # -----------------------------------------------
    # Negative Sampling (Mixed Strategy)
    # -----------------------------------------------
    num_neg = train_edges.size(1)

    hard_neg = sample_hard_negatives(target_snapshot, num_neg // 2).to(device)
    rand_neg = sample_random_negatives(edge_set, num_nodes, num_neg // 2).to(device)

    neg_edges = torch.cat([hard_neg, rand_neg], dim=1)

    # -----------------------------------------------
    # Decoder
    # -----------------------------------------------
    pos_logits = decoder(H, train_edges)
    neg_logits = decoder(H, neg_edges)

    logits = torch.cat([pos_logits, neg_logits])
    labels = torch.cat([
        torch.ones(pos_logits.size(0)),
        torch.zeros(neg_logits.size(0))
    ]).to(device)

    loss = F.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()
    optimizer.step()

    # -----------------------------------------------
    # Validation
    # -----------------------------------------------
    encoder.eval()
    temporal_model.eval()
    decoder.eval()

    with torch.no_grad():

        hard_val = sample_hard_negatives(target_snapshot, val_edges.size(1)).to(device)

        pos_val_logits = decoder(H, val_edges)
        neg_val_logits = decoder(H, hard_val)

        val_logits = torch.cat([pos_val_logits, neg_val_logits])
        val_labels = torch.cat([
            torch.ones(pos_val_logits.size(0)),
            torch.zeros(neg_val_logits.size(0))
        ]).to(device)

        val_probs = torch.sigmoid(val_logits).cpu().numpy()
        val_labels_np = val_labels.cpu().numpy()
