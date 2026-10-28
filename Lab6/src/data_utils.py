import torch
from torch.utils.data import Dataset


class ReviewDataset(Dataset):
    def __init__(self, dataframe, vocabulary, max_length):
        self.input_ids = []
        self.lengths = []
        self.labels = []

        for _, row in dataframe.iterrows():
            token_ids, length = encode_review(
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