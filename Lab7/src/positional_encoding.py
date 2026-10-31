import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len = 512):
        super().__init__()

        assert d_model % 2 == 0

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(
            0, max_len, dtype = torch.float32
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

if __name__ == '__main__':
    pos_encoder = PositionalEncoding(4)

    X = torch.randn(1, 3, 4)

    X_pe = pos_encoder.forward(X)
    print(X_pe - X)
