"""Recurrent classifiers used by the sequence classification lab."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class RecurrentClassifier(nn.Module):
    """Embedding plus a configurable RNN, LSTM, or GRU binary classifier."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        hidden_dim: int = 128,
        cell_type: str = "rnn",
        pad_id: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if cell_type not in {"rnn", "lstm", "gru"}:
            raise ValueError("cell_type phải là 'rnn', 'lstm' hoặc 'gru'.")
        if vocab_size <= 0 or embedding_dim <= 0 or hidden_dim <= 0:
            raise ValueError("Các kích thước model phải lớn hơn 0.")

        self.cell_type = cell_type
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_id,
        )
        recurrent_class = {
            "rnn": nn.RNN,
            "lstm": nn.LSTM,
            "gru": nn.GRU,
        }[cell_type]
        self.encoder = recurrent_class(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        if input_ids.ndim != 2 or lengths.ndim != 1:
            raise ValueError("input_ids phải là (B, T), lengths phải là (B,).")
        if input_ids.size(0) != lengths.size(0):
            raise ValueError("Số sample của input_ids và lengths không khớp.")
        if torch.any(lengths <= 0):
            raise ValueError("Mọi sequence phải có ít nhất một token.")

        embedded = self.embedding(input_ids)
        packed = pack_padded_sequence(
            embedded,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden_state = self.encoder(packed)
        hidden = self._extract_final_hidden(hidden_state)
        return self.classifier(self.dropout(hidden)).squeeze(-1)

    def _extract_final_hidden(self, hidden_state: torch.Tensor | tuple) -> torch.Tensor:
        if self.cell_type == "lstm":
            hidden_state = hidden_state[0]
        return hidden_state[-1]


def build_sequence_classifier(
    model_type: str,
    vocab_size: int,
    pad_id: int,
    embedding_dim: int = 100,
    hidden_dim: int = 128,
    dropout: float = 0.0,
) -> RecurrentClassifier:
    """Create one recurrent architecture using shared experiment settings."""
    return RecurrentClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        cell_type=model_type,
        pad_id=pad_id,
        dropout=dropout,
    )


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


RNNClassifier = RecurrentClassifier
