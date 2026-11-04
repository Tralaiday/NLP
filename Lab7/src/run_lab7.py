"""Runnable experiments for Lab 07.

The module is intentionally self-contained so it can be run from Colab,
Jupyter, or the repository root:

    python Lab7/src/run_lab7.py

The IMDb CSV is discovered in Lab7/data, Lab5/data, or Lab4/data.  Training
can be shortened with the ``LAB7_MAX_SAMPLES`` and ``LAB7_EPOCHS`` environment
variables.
"""

from __future__ import annotations

import csv
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

try:
    from .transformer_classifier import TransformerClassifier
    from .positional_encoding import PositionalEncoding
    from .multi_head_self_attention import MultiHeadSelfAttention
except ImportError:
    from transformer_classifier import TransformerClassifier
    from positional_encoding import PositionalEncoding
    from multi_head_self_attention import MultiHeadSelfAttention


ROOT = Path(__file__).resolve().parents[2]
LAB7 = ROOT / "Lab7"
SEED = 42


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_dataset() -> Path:
    candidates = [
        LAB7 / "data" / "IMDB Dataset.csv",
        ROOT / "Lab5" / "data" / "IMDB Dataset.csv",
        ROOT / "Lab4" / "data" / "IMDB Dataset.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Không tìm thấy IMDb Dataset.csv. Đặt file vào Lab7/data hoặc Lab4/data."
    )


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())


def read_imdb(path: Path, max_samples: int | None = None) -> list[tuple[list[str], int]]:
    rows: list[tuple[list[str], int]] = []
    with path.open(encoding="utf-8") as file:
        for row in csv.DictReader(file):
            sentiment = row["sentiment"].strip().lower()
            if sentiment not in {"positive", "negative"}:
                continue
            rows.append((tokenize(row["review"]), int(sentiment == "positive")))
            if max_samples and len(rows) >= max_samples:
                break
    if not rows:
        raise ValueError(f"Dataset rỗng hoặc sai định dạng: {path}")
    return rows


def build_vocab(samples: Iterable[tuple[list[str], int]], min_freq: int = 2) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tokens, _ in samples:
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token in sorted(counts):
        if counts[token] >= min_freq:
            vocab[token] = len(vocab)
    return vocab


class ReviewDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[list[str], int]],
        vocab: dict[str, int],
        max_length: int,
    ):
        self.items = []
        pad = vocab["<PAD>"]
        unk = vocab["<UNK>"]
        for tokens, label in samples:
            ids = [vocab.get(token, unk) for token in tokens[:max_length]]
            if not ids:
                ids = [unk]
            length = len(ids)
            ids += [pad] * (max_length - length)
            self.items.append(
                (torch.tensor(ids), torch.tensor(length), torch.tensor(label))
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        return self.items[index]


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, hidden_dim: int, pad_idx: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.lstm = nn.LSTM(d_model, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        lengths = attention_mask.sum(1).cpu() if attention_mask is not None else None
        x = self.embedding(input_ids)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.clamp(min=1), batch_first=True, enforce_sorted=False
            )
            _, (hidden, _) = self.lstm(packed)
        else:
            _, (hidden, _) = self.lstm(x)
        return self.classifier(hidden[-1])


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    correct = total = tp = fp = fn = 0
    with torch.no_grad():
        for ids, lengths, labels in loader:
            ids, labels = ids.to(device), labels.to(device)
            mask = ids != 0
            predictions = model(ids, mask).argmax(1)
            correct += int((predictions == labels).sum())
            total += labels.numel()
            tp += int(((predictions == 1) & (labels == 1)).sum())
            fp += int(((predictions == 1) & (labels == 0)).sum())
            fn += int(((predictions == 0) & (labels == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"accuracy": correct / max(total, 1), "precision": precision, "recall": recall, "f1": f1}


def train(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: torch.device,
    epochs: int,
) -> tuple[dict[str, list[float]], float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "valid_loss": []}
    started = time.perf_counter()
    for _ in range(epochs):
        model.train()
        train_total = 0.0
        for ids, _, labels in train_loader:
            ids, labels = ids.to(device), labels.to(device)
            mask = ids != 0
            optimizer.zero_grad()
            loss = criterion(model(ids, mask), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_total += loss.item() * labels.size(0)
        model.eval()
        valid_total = 0.0
        with torch.no_grad():
            for ids, _, labels in valid_loader:
                ids, labels = ids.to(device), labels.to(device)
                valid_total += criterion(model(ids, ids != 0), labels).item() * labels.size(0)
        history["train_loss"].append(train_total / len(train_loader.dataset))
        history["valid_loss"].append(valid_total / len(valid_loader.dataset))
    return history, time.perf_counter() - started


def save_learning_curve(history: dict[str, list[float]], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plt.figure()
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["valid_loss"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run_attention_experiments() -> list[dict[str, float]]:
    rows = []
    for heads in (1, 2, 4, 8):
        model = MultiHeadSelfAttention(128, heads)
        rows.append({
            "heads": heads,
            "d_head": 128 // heads,
            "parameters": sum(p.numel() for p in model.parameters()),
        })
    return rows


def run_positional_experiment() -> dict[str, float]:
    torch.manual_seed(SEED)
    embeddings = {
        "dog": torch.tensor([1.0, 1.0, 0.0, 0.0]),
        "bites": torch.tensor([0.0, 1.0, 1.0, 0.0]),
        "man": torch.tensor([0.0, 0.0, 1.0, 1.0]),
    }
    attention = MultiHeadSelfAttention(4, 1).eval()
    positional = PositionalEncoding(4, 10).eval()

    def representation(sentence: list[str], use_pe: bool) -> torch.Tensor:
        x = torch.stack([embeddings[token] for token in sentence])[None]
        if use_pe:
            x = positional(x)
        return attention(x)[0].mean(1)

    result = {}
    for use_pe in (False, True):
        distance = torch.linalg.vector_norm(
            representation(["dog", "bites", "man"], use_pe)
            - representation(["man", "bites", "dog"], use_pe)
        )
        result[f"distance_pe_{str(use_pe).lower()}"] = float(distance)
    return result


def run_classification() -> tuple[list[dict[str, object]], dict[str, list[float]]]:
    seed_everything()
    dataset_path = find_dataset()
    max_samples = int(os.getenv("LAB7_MAX_SAMPLES", "4000"))
    epochs = int(os.getenv("LAB7_EPOCHS", "3"))
    max_length = int(os.getenv("LAB7_MAX_LENGTH", "128"))
    batch_size = int(os.getenv("LAB7_BATCH_SIZE", "64"))
    samples = read_imdb(dataset_path, max_samples)
    vocab = build_vocab(samples)
    full = ReviewDataset(samples, vocab, max_length)
    train_size = int(0.8 * len(full))
    valid_size = len(full) - train_size
    train_set, valid_set = random_split(
        full, [train_size, valid_size], generator=torch.Generator().manual_seed(SEED)
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = [
        ("LSTM", LSTMClassifier(len(vocab), 64, 64, 0)),
        ("Transformer", TransformerClassifier(
            len(vocab), d_model=64, n_heads=4, d_ff=128, n_layers=2,
            max_len=max_length, pad_idx=0
        )),
    ]
    rows = []
    last_history = {}
    for name, model in configs:
        model.to(device)
        history, seconds = train(model, train_loader, valid_loader, device, epochs)
        metrics = evaluate(model, valid_loader, device)
        rows.append({
            "model": name,
            "parameters": sum(p.numel() for p in model.parameters()),
            "training_time_seconds": round(seconds, 3),
            **{key: round(value, 4) for key, value in metrics.items()},
        })
        if name == "Transformer":
            last_history = history
    return rows, last_history


def main() -> None:
    LAB7.mkdir(exist_ok=True)
    print("Positional experiment:", run_positional_experiment())
    print("Head experiment:", run_attention_experiments())
    rows, history = run_classification()
    with (LAB7 / "results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    save_learning_curve(history, LAB7 / "learning_curves.png")
    print("Classification results:")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
