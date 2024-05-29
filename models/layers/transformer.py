import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(LearnablePositionalEncoding, self).__init__()
        self.encoding = nn.Parameter(torch.zeros(max_len, d_model))
        nn.init.normal_(self.encoding, mean=0, std=0.1)

    def forward(self, x):
        length = x.size(1)
        return x + self.encoding[:length, :]

class MeshTransformer(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=2, nhead=1, dim_feedforward=2048, dropout=0.1, max_len=5000):
        super(MeshTransformer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if in_channels == 3:
            nhead = 1
        else:
            nhead = 8

        self.encoder_layer = TransformerEncoderLayer(d_model=in_channels, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(in_channels, out_channels)
        self.positional_encoding = LearnablePositionalEncoding(in_channels, max_len)

    def forward(self, x):
        # x: (Batch, Channels, Half_Edges, Neighborhood_Size)
        batch_size, channels, half_edges, neighborhood_size = x.shape

        # Reshape to (Batch * Half_Edges, Neighborhood_Size, Channels)
        x = x.permute(0, 2, 3, 1).reshape(batch_size * half_edges, neighborhood_size, channels)

        # Apply positional encoding
        x = self.positional_encoding(x)

        # Apply transformer
        x = self.transformer_encoder(x.permute(1, 0, 2))  # (Neighborhood_Size, Batch * Half_Edges, Channels)
        x = x.permute(1, 2, 0)  # (Batch * Half_Edges, Channels, Neighborhood_Size)

        # Reshape back to (Batch, Channels, Half_Edges, Neighborhood_Size)
        x = x.reshape(batch_size, half_edges, channels, neighborhood_size).permute(0, 2, 1, 3)

        # Mean pooling over the neighborhood size
        x = x.mean(dim=-1)  # (Batch, Channels, Half_Edges)

        # Apply linear layer
        x = x.permute(0, 2, 1)  # (Batch, Half_Edges, Channels)
        x = self.linear(x)  # (Batch, Half_Edges, Out_Channels)
        x = x.permute(0, 2, 1)  # (Batch, Out_Channels, Half_Edges)
        x = x.unsqueeze(-1)  # (Batch, Out_Channels, Half_Edges, 1)

        return x
