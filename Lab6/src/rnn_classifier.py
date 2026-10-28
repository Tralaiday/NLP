import math

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class RNNClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=64,
        hidden_dim=64,
        pad_id=0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_id,
        )

        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, lengths):
        embedded = self.embedding(input_ids)
        # embedded: (batch_size, sequence_length, embedding_dim)

        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, hidden = self.rnn(packed)

        # hidden: (1, batch_size, hidden_dim)
        context = hidden[-1]

        # context: (batch_size, hidden_dim)
        logits = self.classifier(context).squeeze(1)

        return logits