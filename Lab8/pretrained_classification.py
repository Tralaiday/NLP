"""Small utilities for LAB 08 pretrained Transformer classification.

The module deliberately keeps the training loop explicit so that the data flow
from tokenizer to the [CLS] classification head is easy to inspect.
"""

from __future__ import annotations

import random
import time
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoModel


def set_seed(seed: int = 2026) -> None:
    """Make experiments as reproducible as the selected hardware allows."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_imdb(path: str) -> pd.DataFrame:
    """Load IMDb reviews and map sentiment labels to 0 (negative) / 1 (positive)."""
    data = pd.read_csv(path)
    required = {"review", "sentiment"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    data = data.loc[:, ["review", "sentiment"]].dropna().copy()
    data["review"] = data["review"].astype(str)
    labels = {"negative": 0, "positive": 1}
    data["label"] = data["sentiment"].str.lower().map(labels)
    if data["label"].isna().any():
        raise ValueError("The sentiment column must contain only positive/negative.")
    data["label"] = data["label"].astype("int64")
    return data.reset_index(drop=True)


def split_imdb(
    data: pd.DataFrame,
    train_size: float = 0.6,
    validation_size: float = 0.2,
    seed: int = 2026,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified train/validation/test splits."""
    from sklearn.model_selection import train_test_split

    if train_size <= 0 or validation_size <= 0 or train_size + validation_size >= 1:
        raise ValueError("train_size and validation_size must be positive and sum to less than 1.")

    train, remainder = train_test_split(
        data,
        train_size=train_size,
        random_state=seed,
        stratify=data["label"],
    )
    validation_fraction = validation_size / (1 - train_size)
    validation, test = train_test_split(
        remainder,
        train_size=validation_fraction,
        random_state=seed,
        stratify=remainder["label"],
    )
    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )


class IMDbDataset(Dataset):
    """IMDb examples tokenized once in batches during dataset creation."""

    def __init__(self, data: pd.DataFrame, tokenizer, max_length: int = 128):
        self.texts = data["review"].tolist()
        self.labels = data["label"].tolist()
        encoded = tokenizer(
            self.texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.encodings = {
            key: value
            for key, value in encoded.items()
        }

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = {
            key: value[index]
            for key, value in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


class PretrainedClassifier(nn.Module):
    """A pretrained encoder followed by a minimal [CLS] classification head."""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_classes: int = 2,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_representation = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls_representation))


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[Dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> float:
    """Train for one epoch and return mean cross-entropy loss."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_examples = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)
        use_amp = scaler is not None
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item() * labels.size(0)
        total_examples += labels.size(0)

    if total_examples == 0:
        raise ValueError("Cannot train on an empty data loader.")
    return total_loss / total_examples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Iterable[Dict[str, torch.Tensor]],
    device: torch.device,
) -> Dict[str, object]:
    """Return loss, classification metrics, labels, predictions and probabilities."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_examples = 0
    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[float] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        batch_labels = batch["labels"].to(device)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, batch_labels)
        batch_probabilities = torch.softmax(logits.float(), dim=-1)
        batch_predictions = batch_probabilities.argmax(dim=-1)

        total_loss += loss.item() * batch_labels.size(0)
        total_examples += batch_labels.size(0)
        labels.extend(batch_labels.cpu().tolist())
        predictions.extend(batch_predictions.cpu().tolist())
        probabilities.extend(batch_probabilities.max(dim=-1).values.cpu().tolist())

    if total_examples == 0:
        raise ValueError("Cannot evaluate an empty data loader.")

    return {
        "loss": total_loss / total_examples,
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "labels": labels,
        "predictions": predictions,
        "probabilities": probabilities,
    }


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count model parameters, optionally counting only trainable parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )


def fit(
    model: nn.Module,
    train_loader: Iterable[Dict[str, torch.Tensor]],
    validation_loader: Iterable[Dict[str, torch.Tensor]],
    device: torch.device,
    epochs: int = 1,
    learning_rate: float = 2e-5,
) -> Dict[str, object]:
    """Train a model and return its history and elapsed training time."""
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    history = []
    last_validation_metrics = None
    start = time.perf_counter()

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler=scaler if device.type == "cuda" else None,
        )
        validation_metrics = evaluate(model, validation_loader, device)
        last_validation_metrics = validation_metrics
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "validation_loss": validation_metrics["loss"],
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_f1": validation_metrics["f1"],
            }
        )

    return {
        "history": history,
        "validation_metrics": last_validation_metrics,
        "training_time_seconds": time.perf_counter() - start,
    }
