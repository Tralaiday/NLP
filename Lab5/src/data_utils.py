import torch
from torch.utils.data import Dataset

class ReviewDataset(Dataset):
    def __init__(self, data, vocab, max_length):
        self.samples = []

        if max_length <= 0:
            raise ValueError("max_length phải lớn hơn 0.")

        for tokens, sentiment in zip(
            data['tokens'], data['sentiment']
        ):
            # Chuyển tokens thành token IDs
            token_ids = self.encode_tokens(tokens, vocab)

            # Độ dài thực, trước padding
            actual_length = min(len(token_ids), max_length)

            if actual_length == 0:
                raise ValueError('Phát hiện review rỗng.')

            padded_ids, actual_length = self.prepare_sequence(
                token_ids, max_length, vocab['<PAD>']
            )

            if sentiment in ('positive', 1):
                label = 1.0
            elif sentiment in ('negative', 0):
                label = 0.0
            else:
                raise ValueError('Nhãn không hợp lệ.')

            self.samples.append(
                (padded_ids, actual_length, label)
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_ids, length, label = self.samples[idx]

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32)
        )

    def prepare_sequence(self, token_ids, max_length, pad_id = 0):
        # Truncation
        token_ids = token_ids[:max_length]

        # Độ dài thực sau truncation, trước padding
        actual_length = len(token_ids)

        # Padding
        token_ids = token_ids + [pad_id] * (
            max_length - actual_length
        )

        return token_ids, actual_length

    def encode_tokens(self, tokens, vocab):
        return [
            vocab.get(token, vocab['<UNK>'])
            for token in tokens
        ]