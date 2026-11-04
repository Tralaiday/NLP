try:
    from .multi_head_self_attention import MultiHeadSelfAttention
except ImportError:
    from multi_head_self_attention import MultiHeadSelfAttention

import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()

        self.attention = MultiHeadSelfAttention(
            d_model, n_heads
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x, attention_mask = None):
        attn_output, weights = self.attention(
            x, attention_mask
        )

        # Residual + LayerNorm
        x = self.norm1(x + attn_output)

        # FFN
        ffn_output = self.ffn(x)

        # Residual + LayerNorm
        x = self.norm2(x + ffn_output)

        return x, weights