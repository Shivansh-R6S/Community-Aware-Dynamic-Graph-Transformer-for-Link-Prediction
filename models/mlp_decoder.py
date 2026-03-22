import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionDecoder(nn.Module):
    """
    CAST fusion decoder.
    Fused = [z_u || z_v || z_u*z_v || |z_u-z_v| || edge_context]
    Total  = 4*embed_dim + edge_context_dim
    """
    def __init__(self, embed_dim=64, edge_context_dim=32, hidden_dim=128, dropout=0.3):
        super().__init__()
        fused_dim = 4 * embed_dim + edge_context_dim

        self.lin1 = nn.Linear(fused_dim, hidden_dim)
        self.bn1  = nn.BatchNorm1d(hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2  = nn.BatchNorm1d(hidden_dim // 2)
        self.lin3 = nn.Linear(hidden_dim // 2, 1)
        self.drop = dropout

    def forward(self, z, edge_index, edge_context):
        src  = z[edge_index[0]]
        dst  = z[edge_index[1]]
        fused = torch.cat([src, dst, src*dst, torch.abs(src-dst), edge_context], dim=1)
        x = F.relu(self.bn1(self.lin1(fused)))
        x = F.dropout(x, p=self.drop, training=self.training)
        x = F.relu(self.bn2(self.lin2(x)))
        x = F.dropout(x, p=self.drop, training=self.training)
        return self.lin3(x).view(-1)


MLPDecoder = FusionDecoder