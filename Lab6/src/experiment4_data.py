import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


@dataclass(frozen=True)
class ReviewExample:
    text: str
    label: int


@dataclass(frozen=True)
class EncodedSplit:
    input_ids: np.ndarray
    mask: np.ndarray
    lengths: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class PreparedData:
    train: EncodedSplit
    validation: EncodedSplit
    test: EncodedSplit
    vocabulary: dict[str, int]


def load_imdb(path: str | Path) -> list[ReviewExample]:
    """Load IMDB reviews and map negative/positive labels to 0/1."""
    examples: list[ReviewExample] = []

    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        expected_fields = {"review", "sentiment"}
        if reader.fieldnames is None or not expected_fields.issubset(reader.fieldnames):
            raise ValueError(
                "The IMDB CSV must contain 'review' and 'sentiment' columns."
            )

        for row_number, row in enumerate(reader, start=2):
            text = (row["review"] or "").strip()
            sentiment = (row["sentiment"] or "").strip().lower()
            if not text:
                raise ValueError(f"Empty review at CSV row {row_number}.")
            if sentiment not in {"negative", "positive"}:
                raise ValueError(
                    f"Unexpected sentiment {sentiment!r} at CSV row {row_number}."
                )
            examples.append(ReviewExample(text, int(sentiment == "positive")))

    if not examples:
        raise ValueError("The IMDB CSV contains no reviews.")
    return examples


def split_examples(
    examples: Sequence[ReviewExample],
    seed: int = 42,
    test_fraction: float = 0.2,
    validation_fraction: float = 0.1,
) -> tuple[list[ReviewExample], list[ReviewExample], list[ReviewExample]]:
    """Create deterministic train/validation/test splits."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")

    indices = np.random.default_rng(seed).permutation(len(examples))
    test_size = int(len(examples) * test_fraction)
    test_indices = indices[:test_size]
    remaining_indices = indices[test_size:]

    validation_size = int(len(remaining_indices) * validation_fraction)
    validation_indices = remaining_indices[:validation_size]
    train_indices = remaining_indices[validation_size:]

    def select(selected_indices: Iterable[int]) -> list[ReviewExample]:
        return [examples[int(index)] for index in selected_indices]

    return select(train_indices), select(validation_indices), select(test_indices)


def tokenize(text: str) -> list[str]:
    """Lowercase text and keep words plus simple apostrophes."""
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())


def build_vocabulary(
    examples: Sequence[ReviewExample],
    min_frequency: int = 2,
) -> dict[str, int]:
    """Build a vocabulary from training examples only."""
    if min_frequency < 1:
        raise ValueError("min_frequency must be at least 1.")

    counts = Counter(
        token
        for example in examples
        for token in tokenize(example.text)
    )
    vocabulary = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token in sorted(token for token, count in counts.items() if count >= min_frequency):
        vocabulary[token] = len(vocabulary)
    return vocabulary


def encode_and_pad(
    examples: Sequence[ReviewExample],
    vocabulary: dict[str, int],
    max_length: int = 200,
) -> EncodedSplit:
    """Encode reviews and return padded IDs, a real-token mask, lengths, and labels."""
    if max_length < 1:
        raise ValueError("max_length must be at least 1.")
    if vocabulary.get(PAD_TOKEN) != 0 or UNK_TOKEN not in vocabulary:
        raise ValueError("Vocabulary must map <PAD> to 0 and contain <UNK>.")

    unknown_id = vocabulary[UNK_TOKEN]
    input_ids = np.zeros((len(examples), max_length), dtype=np.int64)
    mask = np.zeros((len(examples), max_length), dtype=np.int64)
    lengths = np.zeros(len(examples), dtype=np.int64)
    labels = np.zeros(len(examples), dtype=np.int64)

    for row_index, example in enumerate(examples):
        token_ids = [
            vocabulary.get(token, unknown_id)
            for token in tokenize(example.text)[:max_length]
        ]
        length = len(token_ids)
        if length:
            input_ids[row_index, :length] = token_ids
            mask[row_index, :length] = 1
        lengths[row_index] = length
        labels[row_index] = example.label

    return EncodedSplit(input_ids, mask, lengths, labels)


def prepare_data(
    path: str | Path,
    *,
    seed: int = 42,
    max_length: int = 200,
    min_frequency: int = 2,
) -> PreparedData:
    """Run the first three Experiment 4 data-preparation steps."""
    examples = load_imdb(path)
    train, validation, test = split_examples(examples, seed=seed)
    vocabulary = build_vocabulary(train, min_frequency=min_frequency)

    return PreparedData(
        train=encode_and_pad(train, vocabulary, max_length),
        validation=encode_and_pad(validation, vocabulary, max_length),
        test=encode_and_pad(test, vocabulary, max_length),
        vocabulary=vocabulary,
    )
