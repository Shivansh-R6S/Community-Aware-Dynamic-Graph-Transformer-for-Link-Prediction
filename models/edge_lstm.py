import torch
import torch.nn as nn


class EdgeTemporalEncoder(nn.Module):
    """
    Attention-enhanced LSTM over temporal edge feature sequences.

    Input  : (E, T, feat_dim)
    Output : (E, output_dim)

    Lite: lstm_units=32, output_dim=32
    Full: lstm_units=64, output_dim=64
    """
    def __init__(
        self,
        feat_dim=5,
        lstm_units=32,
        output_dim=32,
        num_layers=1,
        dropout=0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=lstm_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn_w = nn.Linear(lstm_units, lstm_units, bias=True)
        self.attn_v = nn.Linear(lstm_units, 1, bias=False)
        self.proj   = nn.Linear(lstm_units * 2, output_dim)
        self.norm   = nn.LayerNorm(output_dim)

    def forward(self, x):
        """x : (E, T, feat_dim)  →  (E, output_dim)"""
        H, (h_n, _) = self.lstm(x)
        scores  = self.attn_v(torch.tanh(self.attn_w(H)))   # (E, T, 1)
        weights = torch.softmax(scores, dim=1)
        context = (weights * H).sum(dim=1)                  # (E, lstm_units)
        h_last  = h_n[-1]                                   # (E, lstm_units)
        out     = self.proj(torch.cat([context, h_last], dim=1))
        return self.norm(out)