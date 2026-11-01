import math
import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()

        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        B, T, D = x.shape

        x = x.reshape(
            B, T, self.n_heads, self.d_head
        )

        return x.transpose(1, 2)

    def forward(self, x, attention_mask = None):
        B, T, D = x.shape

        # Q, K, V
        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))

        # Scaled dot-product scores
        scores = (
            Q @ K.transpose(-2, -1)
        ) / math.sqrt(self.d_head)

        # Padding mask
        if attention_mask is not None:
            valid_keys = attention_mask[:, None, None, :]
            scores = scores.masked_fill(
                ~valid_keys.bool(),
                torch.finfo(scores.dtype).min
            )

        # Attention weights
        weights = torch.softmax(scores, dim = -1)

        # Weighted sum of values
        context = weights @ V

        # Concatenate heads
        context = context.transpose(1, 2)
        context = context.contiguous().reshape(B, T, D)

        # Final projection
        output = self.W_o(context)

        return output, weights