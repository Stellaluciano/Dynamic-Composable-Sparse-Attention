"""Toy datasets for lightweight language model experiments."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import Dataset


class RandomTokenDataset(Dataset[Tensor]):
    """Random synthetic token sequences for smoke testing."""

    def __init__(self, n_samples: int, seq_len: int, vocab_size: int, seed: int = 0) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.data = torch.randint(0, vocab_size, (n_samples, seq_len), generator=generator)

    def __len__(self) -> int:
        return self.data.size(0)

    def __getitem__(self, idx: int) -> Tensor:
        return self.data[idx]


class TinyPatternDataset(Dataset[Tensor]):
    """Small deterministic dataset with easy repeating patterns."""

    def __init__(self, n_samples: int, seq_len: int, vocab_size: int) -> None:
        rows = []
        for i in range(n_samples):
            base = torch.arange(seq_len) + i
            rows.append((base % vocab_size).long())
        self.data = torch.stack(rows)

    def __len__(self) -> int:
        return self.data.size(0)

    def __getitem__(self, idx: int) -> Tensor:
        return self.data[idx]
