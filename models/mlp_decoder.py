import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.3):
        super(MLPDecoder, self).__init__()

        self.lin1 = nn.Linear(input_dim * 2, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, z, edge_index):
        # Get node embeddings for each edge
        src = z[edge_index[0]]
        dst = z[edge_index[1]]

        # Concatenate instead of dot product
        edge_features = torch.cat([src, dst], dim=1)

        x = F.relu(self.lin1(edge_features))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        return x.view(-1)  # return logits
