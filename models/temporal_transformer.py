import torch
import torch.nn as nn


class TemporalTransformer(nn.Module):
    """
    Temporal Transformer over GNN snapshot embeddings.
    Input  : (T, N, d_model)
    Output : (N, d_model)

    Lite: d_model=64,  n_heads=4, num_layers=2, dim_feedforward=128
    Full: d_model=256, n_heads=8, num_layers=3, dim_feedforward=512
    """
    def __init__(
        self,
        d_model=64,
        n_heads=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        max_time_steps=20,
        pooling='mean',
    ):
        super().__init__()
        self.max_time_steps = max_time_steps
        self.pooling        = pooling

        self.positional_encoding = nn.Parameter(
            torch.randn(1, max_time_steps, d_model) * 0.01
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm        = nn.LayerNorm(d_model)

    def forward(self, Z):
        """Z : (T, N, d_model)  →  (N, d_model)"""
        T, N, d = Z.shape
        Z   = Z.permute(1, 0, 2)
        Z   = Z + self.positional_encoding[:, :T, :]
        out = self.norm(self.transformer(Z))
        return out.mean(dim=1) if self.pooling == 'mean' else out[:, -1, :]