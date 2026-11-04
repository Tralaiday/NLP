from src.positional_encoding import PositionalEncoding
from src.transformer_block import TransformerBlock

import torch.nn as nn

class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model = 128,
        n_heads = 4,
        d_ff = 256,
        n_layers = 2,
        num_classes = 2,
        max_len = 256,
        dropout = 0.1,
        pad_idx = 0
    ):
        super().__init__()

        assert d_model % n_heads == 0
        assert n_layers >= 1

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.num_classes = num_classes
        self.max_len = max_len
        self.pad_idx = pad_idx

        # Token Embedding
        self.embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx = pad_idx
        )

        # Positional Encoding
        self.pos_encoding = PositionalEncoding(
            d_model = d_model,
            max_len = max_len
        )

        self.dropout = nn.Dropout(dropout)

        # Transformer Encoder Stack
        self.encoder = nn.ModuleList([
            TransformerBlock(
                d_model = d_model,
                n_heads = n_heads,
                d_ff = d_ff
            )
            for _ in range(n_layers)
        ])

        # Classification Head
        self.classifier = nn.Linear(
            d_model,
            num_classes
        )

    def mean_pooling(self, x, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(x.dtype)

        summed = (x * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)

        return summed / counts

    def forward(
        self,
        input_ids,
        attention_mask=None,
        return_attention=False,
        debug_shapes=False
    ):
        # input_ids: [B, T]

        if attention_mask is None:
            attention_mask = (
                input_ids != self.pad_idx
            )

        # Embedding + Position
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # Encoder
        all_attention_weights = []

        for block in self.encoder:
            x, weights = block(x, attention_mask)

            if return_attention:
                all_attention_weights.append(weights)

        # Sentence representation
        pooled = self.mean_pooling(
            x, attention_mask
        )

        # Classification
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        if return_attention:
            return logits, all_attention_weights

        return logits
