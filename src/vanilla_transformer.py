import torch
import torch.nn as nn
import numpy as np

# Transformer Model
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=128, nhead=8, num_layers=3, dropout=0.1):
        """
        Args:
            input_dim: The input dimensionality of the transformer, which corresponds to the number of features in the time series data.
            output_dim: The output dimensionality of the transformer, which corresponds to the number of targets in the time series data.
            d_model: The number of expected features in the encoder/decoder inputs (default=128).
            nhead: The number of heads in the multiheadattention models (default=8).
            num_layers: The number of encoder/decoder layers (default=3).
            dropout: The dropout value (default=0.1).
        """
        super().__init__()
        self.encoder = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.Linear(d_model, output_dim)
        
    def forward(self, src):
        src = self.encoder(src)
        src = self.pos_encoder(src)
        output = self.transformer(src, src)  # Using src as both encoder and decoder input
        return self.decoder(output[:, -1, :])  # Take last time step's output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Args:
            d_model: The dimension of the model, which determines the size of the positional encodings.
            dropout: The dropout probability applied to the output of the positional encodings (default=0.1).
            max_len: The maximum length of the sequence for which positional encodings are computed (default=5000).
        """

        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)