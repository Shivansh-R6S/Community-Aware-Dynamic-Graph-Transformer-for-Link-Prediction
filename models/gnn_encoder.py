import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanAggregator(nn.Module):
    """
    Mean neighbor aggregation layer for GraphSAGE.
    """

    def __init__(self):
        super(MeanAggregator, self).__init__()

    def forward(self, x, adjacency_dict):
        """
        x: Node feature matrix (N, F)
        adjacency_dict: {node: [neighbors]}
        """
        agg_features = []

        for node in range(x.size(0)):
            neighbors = adjacency_dict[node]

            if len(neighbors) == 0:
                neighbor_mean = torch.zeros_like(x[node])
            else:
                neighbor_feats = x[neighbors]
                neighbor_mean = neighbor_feats.mean(dim=0)

            agg_features.append(neighbor_mean)

        return torch.stack(agg_features)


class GraphSAGELayer(nn.Module):
    """
    Single GraphSAGE layer.
    """

    def __init__(self, in_dim, out_dim):
        super(GraphSAGELayer, self).__init__()
        self.aggregator = MeanAggregator()  # defined once
        self.linear = nn.Linear(in_dim * 2, out_dim)

    def forward(self, x, adjacency_dict):
        """
        x: (N, F)
        """
        neighbor_agg = self.aggregator(x, adjacency_dict)

        # Concatenate self + neighbor
        combined = torch.cat([x, neighbor_agg], dim=1)

        return self.linear(combined)


class GraphSAGEEncoder(nn.Module):
    """
    GraphSAGE Encoder with:
    - Learnable node embeddings
    - Community feature integration
    - 2-layer GraphSAGE
    """

    def __init__(
        self,
        num_nodes,
        community_dim=3,
        embed_dim=32,
        hidden_dim=64,
        output_dim=128,
    ):
        super(GraphSAGEEncoder, self).__init__()

        self.num_nodes = num_nodes

        # Learnable node embeddings
        self.node_embedding = nn.Embedding(num_nodes, embed_dim)

        input_dim = community_dim + embed_dim

        # GraphSAGE layers
        self.sage1 = GraphSAGELayer(input_dim, hidden_dim)
        self.sage2 = GraphSAGELayer(hidden_dim, output_dim)

    def forward(self, adjacency_dict, community_features):
        """
        adjacency_dict: {node: [neighbors]}
        community_features: Tensor (N, community_dim)
        """

        device = community_features.device

        # Device-safe node ids
        node_ids = torch.arange(self.num_nodes, device=device)

        embeddings = self.node_embedding(node_ids)

        # Concatenate community features + learnable embeddings
        x = torch.cat([community_features, embeddings], dim=1)

        # Layer 1
        x = self.sage1(x, adjacency_dict)
        x = F.relu(x)

        # Layer 2
        x = self.sage2(x, adjacency_dict)

        return x  # (N, output_dim)

    @staticmethod
    def build_adjacency_dict(edge_index, num_nodes):
        """
        edge_index: Tensor (2, E)
        Returns: adjacency dictionary
        """
        adjacency_dict = {i: [] for i in range(num_nodes)}

        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]

        for src, dst in zip(src_nodes, dst_nodes):
            src = src.item()
            dst = dst.item()

            adjacency_dict[src].append(dst)
            adjacency_dict[dst].append(src)

        return adjacency_dict
