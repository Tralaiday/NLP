"""Utilities for preparing reviews for recurrent text classifiers."""

from collections import Counter
from dataclasses import dataclass
import html
import re
from typing import Iterable, Sequence

import torch
from torch.utils.data import Dataset


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")


def clean_review(text: str) -> str:
    """Decode HTML, remove tags, normalize whitespace, and lowercase text."""
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def tokenize(text: str) -> list[str]:
    """Convert a raw review into the lab's deterministic token sequence."""
    return _TOKEN_PATTERN.findall(clean_review(text))


@dataclass(frozen=True)
class Vocabulary:
    """Bidirectional token vocabulary with explicit padding and OOV tokens."""

    stoi: dict[str, int]
    itos: list[str]
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"

    @property
    def pad_id(self) -> int:
        return self.stoi[self.pad_token]

    @property
    def unk_id(self) -> int:
        return self.stoi[self.unk_token]

    def encode(self, tokens: Sequence[str]) -> list[int]:
        return [self.stoi.get(token, self.unk_id) for token in tokens]

    def __getitem__(self, token: str) -> int:
        return self.stoi[token]

    def get(self, token: str, default: int | None = None) -> int:
        return self.stoi.get(token, self.unk_id if default is None else default)

    def __contains__(self, token: str) -> bool:
        return token in self.stoi

    def __len__(self) -> int:
        return len(self.itos)


def build_vocab(
    tokenized_texts: Iterable[Sequence[str]],
    min_freq: int = 1,
    max_size: int | None = None,
) -> Vocabulary:
    """Build a vocabulary from training tokens only."""
    if min_freq <= 0:
        raise ValueError("min_freq phải lớn hơn 0.")
    if max_size is not None and max_size < 2:
        raise ValueError("max_size phải ít nhất là 2.")

    counts = Counter(
        token
        for tokens in tokenized_texts
        for token in tokens
    )
    itos = ["<PAD>", "<UNK>"]
    candidates = (
        token for token, count in counts.most_common()
        if count >= min_freq
    )
    if max_size is not None:
        candidates = (token for token in candidates if token not in itos)
        candidates = list(candidates)[:max_size - len(itos)]

    itos.extend(
        candidates if isinstance(candidates, list) else list(candidates)
    )
    return Vocabulary(
        stoi={token: index for index, token in enumerate(itos)},
        itos=itos,
    )


def pad_or_truncate(
    token_ids: Sequence[int],
    max_length: int,
    pad_id: int,
) -> tuple[list[int], int]:
    """Truncate a sequence, pad it, and return its true post-truncation length."""
    if max_length <= 0:
        raise ValueError("max_length phải lớn hơn 0.")

    truncated = list(token_ids[:max_length])
    length = len(truncated)
    padded = truncated + [pad_id] * (max_length - length)
    return padded, length


def encode_label(label: str | int) -> float:
    """Convert the IMDb label representation to a binary float."""
    if label in ("positive", 1, 1.0):
        return 1.0
    if label in ("negative", 0, 0.0):
        return 0.0
    raise ValueError(f"Nhãn không hợp lệ: {label!r}")


class ReviewDataset(Dataset):
    """Fixed-length encoded reviews for ``DataLoader``."""

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[str | int],
        vocab: Vocabulary,
        max_length: int,
    ):
        if len(texts) != len(labels):
            raise ValueError("texts và labels phải cùng độ dài.")

        self.samples: list[tuple[list[int], int, float]] = []
        for text, label in zip(texts, labels):
            token_ids = vocab.encode(tokenize(text))
            if not token_ids:
                raise ValueError("Phát hiện review rỗng.")

            input_ids, length = pad_or_truncate(
                token_ids,
                max_length=max_length,
                pad_id=vocab.pad_id,
            )
            self.samples.append((input_ids, length, encode_label(label)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        input_ids, length, label = self.samples[index]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float32),
        }


tokenize_review = tokenize
