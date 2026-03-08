"""Minimal training utilities for decoder-only language models."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class TrainStats:
    final_loss: float
    tokens_per_second: float


def train_language_model(
    model: nn.Module,
    dataloader: DataLoader,
    lr: float,
    steps: int,
    device: str,
) -> TrainStats:
    """Run a short next-token training loop and return summary stats."""
    model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    total_tokens = 0
    start = time.time()
    final_loss = 0.0
    step_idx = 0

    while step_idx < steps:
        for batch in dataloader:
            if step_idx >= steps:
                break
            batch = batch.to(device)
            out = model(batch, labels=batch)
            loss = out["loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            final_loss = float(loss.item())
            total_tokens += batch.numel()
            step_idx += 1

    elapsed = max(time.time() - start, 1e-6)
    return TrainStats(final_loss=final_loss, tokens_per_second=total_tokens / elapsed)
