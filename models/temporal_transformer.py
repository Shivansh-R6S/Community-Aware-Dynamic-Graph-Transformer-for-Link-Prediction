import torch
import torch.nn as nn
print("temporal_transformer module is executing")



class TemporalTransformer(nn.Module):
    """
    Temporal Transformer Encoder for dynamic node embeddings.

    Input:
        Z: Tensor of shape (T, N, d_model)

    Output:
        H: Tensor of shape (N, d_model)
    """

    def __init__(
        self,
        d_model=128,
        n_heads=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        max_time_steps=20,
    ):
        super(TemporalTransformer, self).__init__()

        self.d_model = d_model
        self.max_time_steps = max_time_steps

        # Learnable positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(1, max_time_steps, d_model)
        )

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, Z):
        """
        Z: (T, N, d_model)
        """

        T, N, d_model = Z.shape

        # Rearrange to (N, T, d_model)
        Z = Z.permute(1, 0, 2)

        # Add positional encoding
        Z = Z + self.positional_encoding[:, :T, :]

        # Apply transformer
        output = self.transformer(Z)

        # Take final time step representation
        H = output[:, -1, :]

        return H
