import torch
from torch import nn
from torch.nn.utils.rnn import (
    pack_padded_sequence
)

class RecurrentClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        hidden_dim: int = 128,
        cell_type: str = 'rnn',
        pad_id: int = 0,
        dropout: float = 0.0
    ):
        super().__init__()

        if cell_type not in {"rnn", "lstm", "gru"}:
            raise ValueError(
                "cell_type phải là 'rnn', 'lstm', 'gru'"
            )

        self.cell_type = cell_type

        self.embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = embedding_dim,
            padding_idx = pad_id
        )

        recurrent_cls = {
            "rnn": nn.RNN,
            "lstm": nn.LSTM,
            "gru": nn.GRU
        }[cell_type]

        self.encoder = recurrent_cls(
            input_size = embedding_dim,
            hidden_size = hidden_dim,
            batch_first = True
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor
    ) -> torch.Tensor:
        embedded = self.embedding(input_ids)

        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first = True,
            enforce_sorted = False
        )

        _, hidden_state = self.encoder(packed)
        hidden = self._extract_hidden(hidden_state)

        hidden = self.dropout(hidden)
        logits = self.classifier(hidden).squeeze(-1)

        return logits

    def _extract_hidden(self, hidden_state):
        if self.cell_type == "lstm":
            h_n, _ = hidden_state
        else:
            h_n = hidden_state

        return h_n[-1]

def build_sequence_classifier(
    model_type: str,
    vocab_size: int,
    pad_id: int,
    embedding_dim: int = 100,
    hidden_dim: int = 128,
    dropout: float = 0.0,
) -> RecurrentClassifier:
    return RecurrentClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        cell_type=model_type,
        pad_id=pad_id,
        dropout=dropout,
    )

def count_trainable_parameters(
    model: nn.Module,
) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )