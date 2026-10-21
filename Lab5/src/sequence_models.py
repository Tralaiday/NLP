import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class RNNClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=100,
        hidden_dim=128,
        pad_id=0
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_id
        )

        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, lengths):
        # (B, T) -> (B, T, embedding_dim)
        embedded = self.embedding(input_ids)

        # Bỏ qua các timestep padding
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        # Hidden state cuối cùng của các token thực
        _, h_n = self.rnn(packed)

        # (1, B, hidden_dim) -> (B, hidden_dim)
        final_hidden = h_n[-1]

        # (B, hidden_dim) -> (B,)
        logits = self.fc(final_hidden).squeeze(1)

        return logits