from torch.utils.data import Dataset

import torch
import re


class ReviewDataset(Dataset):
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"


    def __init__(self, dataframe, vocabulary, max_length):
        self.PAD_TOKEN = "<PAD>"
        self.UNK_TOKEN = "<UNK>"
        self.input_ids = []
        self.lengths = []
        self.labels = []

        for _, row in dataframe.iterrows():
            token_ids, length = self.encode_review(
                row["review"],
                vocabulary,
                max_length
            )

            self.input_ids.append(token_ids)
            self.lengths.append(length)
            self.labels.append(int(row["label"]))

        self.input_ids = torch.tensor(
            self.input_ids,
            dtype=torch.long
        )
        self.lengths = torch.tensor(
            self.lengths,
            dtype=torch.long
        )
        self.labels = torch.tensor(
            self.labels,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            self.input_ids[index],
            self.lengths[index],
            self.labels[index],
        )

    def encode_review(self, text, vocabulary, max_length=100):
        tokens = self.tokenize(text)

        token_ids = [
            vocabulary.get(token, vocabulary[self.UNK_TOKEN])
            for token in tokens[:max_length]
        ]

        length = len(token_ids)

        if length < max_length:
            token_ids += [vocabulary[self.PAD_TOKEN]] * (max_length - length)

        return token_ids, length

    def tokenize(self, text):
        text = text.lower()
        text = text.replace("<br />", " ")
        return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text)