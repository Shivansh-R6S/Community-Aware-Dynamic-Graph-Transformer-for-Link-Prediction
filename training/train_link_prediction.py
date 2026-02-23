import os
import sys
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import negative_sampling
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


print("\n==== CADGT Dynamic Training (Option 1) ====\n")

# ---------------------------------------------------
# 1️⃣ Load dataset
# ---------------------------------------------------
data_path = os.path.join(ROOT_DIR, "data", "raw", "fb-forum.txt")
df, node_mapping = load_fb_forum(data_path)
num_nodes = len(node_mapping)

# ---------------------------------------------------
# 2️⃣ Build snapshots
# ---------------------------------------------------
snapshots = build_rolling_cumulative_snapshots(df, num_nodes)

print(f"\nTotal snapshots: {len(snapshots)}")

# ---------------------------------------------------
# 3️⃣ Community features
# ---------------------------------------------------
community_features_list = extract_all_snapshots_community_features(snapshots)

# ---------------------------------------------------
# 4️⃣ Models
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
# 5️⃣ Prepare target snapshot (snapshot 11)
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

train_edges = torch.tensor(train_edges_np).t().contiguous().to(device)
val_edges = torch.tensor(val_edges_np).t().contiguous().to(device)

# ---------------------------------------------------
# Training Loop
# ---------------------------------------------------
for epoch in range(1, epochs + 1):

    encoder.train()
    temporal_model.train()
    decoder.train()
    optimizer.zero_grad()

    structural_embeddings = []

    # ---------------------------------------------------
    # Generate embeddings for snapshots 1–10
    # ---------------------------------------------------
    for i in range(len(snapshots) - 1):

        G = snapshots[i]
        x = community_features_list[i].to(device)
        edge_index = graph_to_edge_index(G).to(device)

        adj_dict = GraphSAGEEncoder.build_adjacency_dict(
            edge_index, num_nodes
        )

        z = encoder(adj_dict, x)
        structural_embeddings.append(z)

    Z_stack = torch.stack(structural_embeddings)  # (T-1, N, 128)

    # ---------------------------------------------------
    # Temporal Transformer
    # ---------------------------------------------------
    H = temporal_model(Z_stack)

    # ---------------------------------------------------
    # Negative Sampling (Train)
    # ---------------------------------------------------
    neg_train_edges = negative_sampling(
        edge_index=train_edges,
        num_nodes=num_nodes,
        num_neg_samples=train_edges.size(1)
    )

    pos_logits = decoder(H, train_edges)
    neg_logits = decoder(H, neg_train_edges)

    logits = torch.cat([pos_logits, neg_logits])
    labels = torch.cat([
        torch.ones(pos_logits.size(0)),
        torch.zeros(neg_logits.size(0))
    ]).to(device)

    loss = F.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()
    optimizer.step()

    # ---------------------------------------------------
    # Validation
    # ---------------------------------------------------
    encoder.eval()
    temporal_model.eval()
    decoder.eval()

    with torch.no_grad():

        neg_val_edges = negative_sampling(
            edge_index=train_edges,
            num_nodes=num_nodes,
            num_neg_samples=val_edges.size(1)
        )

        pos_val_logits = decoder(H, val_edges)
        neg_val_logits = decoder(H, neg_val_edges)

        val_logits = torch.cat([pos_val_logits, neg_val_logits])
        val_labels = torch.cat([
            torch.ones(pos_val_logits.size(0)),
            torch.zeros(neg_val_logits.size(0))
        ]).to(device)

        val_probs = torch.sigmoid(val_logits).cpu().numpy()
        val_labels_np = val_labels.cpu().numpy()

        val_auc = roc_auc_score(val_labels_np, val_probs)
        val_ap = average_precision_score(val_labels_np, val_probs)

    print(
        f"Epoch {epoch:03d} | "
        f"Loss: {loss.item():.4f} | "
        f"Val AUC: {val_auc:.4f} | "
        f"Val AP: {val_ap:.4f}"
    )

print("\n==== Training Complete ====\n")
