import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorizedSAGELayer(nn.Module):
    """
    GraphSAGE layer using sparse matrix multiplication — no Python node loops.
    Mean aggregation with self + neighbour concat, then linear + BN.
    Fast and stable on CPU.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, out_dim)
        self.bn     = nn.BatchNorm1d(out_dim)

    def forward(self, x, adj_norm):
        """
        x        : (N, in_dim)
        adj_norm : (N, N) sparse or dense row-normalised adjacency
        """
        agg      = torch.mm(adj_norm, x)          # (N, in_dim) — vectorized mean
        combined = torch.cat([x, agg], dim=1)     # (N, 2*in_dim)
        return self.bn(self.linear(combined))      # (N, out_dim)


class GraphSAGEEncoder(nn.Module):
    """
    2-layer GraphSAGE with:
    - Learnable node embeddings
    - 6-dim community + structural input
    - Vectorized mean aggregation (fast on CPU)
    - Skip connection
    - BatchNorm

    Lite: embed_dim=32, hidden_dim=64,  output_dim=64
    Full: embed_dim=64, hidden_dim=256, output_dim=256
    """
    def __init__(
        self,
        num_nodes,
        community_dim=6,
        embed_dim=32,
        hidden_dim=64,
        output_dim=64,
    ):
        super().__init__()
        self.num_nodes      = num_nodes
        self.node_embedding = nn.Embedding(num_nodes, embed_dim)
        input_dim           = community_dim + embed_dim

        self.sage1     = VectorizedSAGELayer(input_dim, hidden_dim)
        self.sage2     = VectorizedSAGELayer(hidden_dim, output_dim)
        self.skip_proj = nn.Linear(input_dim, output_dim)

    def forward(self, adj_norm, community_features):
        """
        adj_norm          : (N, N) row-normalised adjacency (dense, on device)
        community_features: (N, community_dim)
        """
        device = community_features.device
        emb    = self.node_embedding(torch.arange(self.num_nodes, device=device))
        x      = torch.cat([community_features, emb], dim=1)   # (N, input_dim)

        x1  = F.relu(self.sage1(x, adj_norm))
        x2  = self.sage2(x1, adj_norm)
        out = F.relu(x2 + self.skip_proj(x))
        return out   # (N, output_dim)

    @staticmethod
    def build_adj_norm(edge_index, num_nodes, device):
        """
        Build a row-normalised dense adjacency matrix from edge_index.
        A_norm[i,j] = 1 / deg(i)  if edge (i,j) exists, else 0.
        Isolated nodes get a self-loop to avoid div-by-zero.

        edge_index : Tensor (2, E)
        Returns    : Tensor (N, N) float32 on device
        """
        A = torch.zeros(num_nodes, num_nodes, device=device)

        if edge_index.size(1) > 0:
            src = edge_index[0]
            dst = edge_index[1]
            A[src, dst] = 1.0
            A[dst, src] = 1.0   # undirected

        # Self-loops for isolated nodes
        deg = A.sum(dim=1)
        isolated = (deg == 0)
        A[isolated, isolated] = 1.0
        deg = A.sum(dim=1)

        # Row normalise: D^{-1} A
        deg_inv = 1.0 / deg
        A_norm  = A * deg_inv.unsqueeze(1)
        return A_norm

    # Keep old interface for compatibility
    @staticmethod
    def build_adjacency_dict(edge_index, num_nodes):
        adj = {i: [] for i in range(num_nodes)}
        for src, dst in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            adj[src].append(dst)
            adj[dst].append(src)
        return adj