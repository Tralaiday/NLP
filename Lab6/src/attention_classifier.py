import math

import torch
import torch.nn as nn


class AttentionClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=64,
        pad_id=0,
    ):
        super().__init__()

        self.pad_id = pad_id

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_id,
        )

        self.classifier = nn.Linear(embedding_dim, 1)

    def forward(
        self,
        input_ids,
        lengths=None,
        return_attention=False,
    ):
        embedded = self.embedding(input_ids)

        Q = embedded
        K = embedded
        V = embedded

        scores = torch.matmul(Q, K.transpose(1, 2))
        scores = scores / math.sqrt(embedded.size(-1))

        valid_tokens = input_ids.ne(self.pad_id)
        scores = scores.masked_fill(
            ~valid_tokens.unsqueeze(1),
            -1e9,
        )

        attention = torch.softmax(scores, dim=-1)
        attended = torch.matmul(attention, V)

        value_mask = valid_tokens.unsqueeze(-1).to(attended.dtype)
        attended = attended * value_mask

        token_counts = value_mask.sum(dim=1).clamp_min(1.0)
        context = attended.sum(dim=1) / token_counts

        logits = self.classifier(context).squeeze(1)

        if return_attention:
            return logits, attention

        return logits