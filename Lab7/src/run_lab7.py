"""Run the experiments required by LAB 07.

Run from the repository root:

    python Lab7/src/run_lab7.py

For a quick test, set LAB7_MAX_SAMPLES and LAB7_EPOCHS.
"""

import csv
import os
import random
import re
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from multi_head_self_attention import MultiHeadSelfAttention
from positional_encoding import PositionalEncoding
from transformer_classifier import TransformerClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "IMDB Dataset.csv"
SEED = 42


def set_seed():
    random.seed(SEED)
    torch.manual_seed(SEED)


def words(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())


def read_data(limit):
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy {DATA_PATH}")

    data = []
    with DATA_PATH.open(encoding="utf-8") as file:
        for row in csv.DictReader(file):
            data.append((words(row["review"]), int(row["sentiment"] == "positive")))
            if limit and len(data) == limit:
                break
    return data


def make_vocab(data, min_frequency=2):
    counts = {}
    for tokens, _ in data:
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token, count in sorted(counts.items()):
        if count >= min_frequency:
            vocab[token] = len(vocab)
    return vocab


class Reviews(Dataset):
    def __init__(self, data, vocab, max_length):
        self.items = []
        for tokens, label in data:
            ids = [vocab.get(token, 1) for token in tokens[:max_length]]
            ids = ids or [1]
            length = len(ids)
            ids += [0] * (max_length - length)
            self.items.append((torch.tensor(ids), torch.tensor(label), length))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, hidden_size=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, mask):
        embedded = self.embedding(input_ids)
        lengths = mask.sum(1).cpu().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        return self.output(hidden[-1])


def make_mask(input_ids):
    return input_ids != 0


def train(model, loader, valid_loader, device, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    loss_function = nn.CrossEntropyLoss()
    history = {"train_loss": [], "valid_loss": []}
    start = time.time()

    for _ in range(epochs):
        model.train()
        train_loss = 0
        for input_ids, labels, _ in loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_function(model(input_ids, make_mask(input_ids)), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)

        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for input_ids, labels, _ in valid_loader:
                input_ids, labels = input_ids.to(device), labels.to(device)
                loss = loss_function(model(input_ids, make_mask(input_ids)), labels)
                valid_loss += loss.item() * len(labels)

        history["train_loss"].append(train_loss / len(loader.dataset))
        history["valid_loss"].append(valid_loss / len(valid_loader.dataset))

    return history, time.time() - start


def evaluate(model, loader, device):
    model.eval()
    total = correct = true_positive = false_positive = false_negative = 0
    with torch.no_grad():
        for input_ids, labels, _ in loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            predictions = model(input_ids, make_mask(input_ids)).argmax(1)
            total += len(labels)
            correct += (predictions == labels).sum().item()
            true_positive += ((predictions == 1) & (labels == 1)).sum().item()
            false_positive += ((predictions == 1) & (labels == 0)).sum().item()
            false_negative += ((predictions == 0) & (labels == 1)).sum().item()

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "accuracy": correct / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def positional_experiment():
    embeddings = {
        "dog": [1, 1, 0, 0],
        "bites": [0, 1, 1, 0],
        "man": [0, 0, 1, 1],
    }
    attention = MultiHeadSelfAttention(4, 1).eval()
    position = PositionalEncoding(4, 10)

    def represent(sentence, use_position):
        x = torch.tensor([[embeddings[word] for word in sentence]], dtype=torch.float)
        if use_position:
            x = position(x)
        return attention(x)[0].mean(1)

    results = {}
    for use_position in (False, True):
        first = represent(["dog", "bites", "man"], use_position)
        second = represent(["man", "bites", "dog"], use_position)
        results[use_position] = torch.linalg.vector_norm(first - second).item()
    return results


def attention_experiment():
    return [
        {
            "heads": heads,
            "d_head": 128 // heads,
            "parameters": sum(p.numel() for p in MultiHeadSelfAttention(128, heads).parameters()),
        }
        for heads in (1, 2, 4, 8)
    ]


def save_plots(history, confusion):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plt.figure()
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["valid_loss"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT / "learning_curves.png", dpi=150)
    plt.close()

    plt.figure()
    plt.imshow(confusion, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(ROOT / "confusion_matrix.png", dpi=150)
    plt.close()


def save_attention_plot():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    model = MultiHeadSelfAttention(32, 4).eval()
    _, weights = model(torch.randn(1, 12, 32))
    figure, axes = plt.subplots(1, 4, figsize=(10, 3))
    for head, axis in enumerate(axes):
        axis.imshow(weights[0, head].detach(), cmap="viridis")
        axis.set_title(f"Head {head + 1}")
    figure.tight_layout()
    figure.savefig(ROOT / "attention_heads.png", dpi=150)
    plt.close(figure)


def classification_experiment():
    set_seed()
    limit = int(os.getenv("LAB7_MAX_SAMPLES", "4000"))
    epochs = int(os.getenv("LAB7_EPOCHS", "3"))
    max_length = int(os.getenv("LAB7_MAX_LENGTH", "128"))
    batch_size = int(os.getenv("LAB7_BATCH_SIZE", "64"))
    data = read_data(limit)
    vocab = make_vocab(data)
    dataset = Reviews(data, vocab, max_length)

    train_size = int(len(dataset) * 0.7)
    valid_size = int(len(dataset) * 0.15)
    test_size = len(dataset) - train_size - valid_size
    train_set, valid_set, test_set = random_split(
        dataset,
        [train_size, valid_size, test_size],
        generator=torch.Generator().manual_seed(SEED),
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = {
        "LSTM": LSTMClassifier(len(vocab)),
        "Transformer": TransformerClassifier(
            len(vocab), d_model=64, n_heads=4, d_ff=128,
            n_layers=2, max_len=max_length, pad_idx=0,
        ),
    }
    results = []
    transformer_history = None
    transformer_confusion = [[0, 0], [0, 0]]

    for name, model in models.items():
        model.to(device)
        history, seconds = train(model, train_loader, valid_loader, device, epochs)
        metrics = evaluate(model, test_loader, device)
        results.append({
            "model": name,
            "parameters": sum(p.numel() for p in model.parameters()),
            "training_seconds": round(seconds, 2),
            **{key: round(value, 4) for key, value in metrics.items()},
        })
        if name == "Transformer":
            transformer_history = history
            model.eval()
            with torch.no_grad():
                for input_ids, labels, _ in test_loader:
                    predictions = model(
                        input_ids.to(device), make_mask(input_ids.to(device))
                    ).argmax(1).cpu()
                    for actual, predicted in zip(labels, predictions):
                        transformer_confusion[actual][predicted] += 1

    save_plots(transformer_history, transformer_confusion)
    return results


def main():
    print("Permutation distances:", positional_experiment())
    print("Attention heads:", attention_experiment())
    rows = classification_experiment()
    with (ROOT / "results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    save_attention_plot()
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
