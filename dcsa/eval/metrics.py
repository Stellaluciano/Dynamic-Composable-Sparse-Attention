"""Evaluation and benchmarking helpers."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class EvalStats:
    loss: float
    perplexity: float
    tokens_per_second: float


def parameter_count(model: nn.Module) -> int:
    """Return number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def approximate_attention_cost(seq_len: int, top_k: int, d_model: int, n_heads: int) -> dict[str, int]:
    """Approximate attention FLOP-like terms for dense vs DCSA sparse."""
    head_dim = d_model // n_heads
    dense = n_heads * seq_len * seq_len * head_dim
    sparse = n_heads * seq_len * top_k * head_dim
    return {"dense": dense, "sparse_dcsa": sparse}


def evaluate_language_model(model: nn.Module, dataloader: DataLoader, device: str) -> EvalStats:
    """Compute average next-token loss/perplexity and throughput."""
    model.to(device)
    model.eval()
    total_loss = 0.0
    n_batches = 0
    total_tokens = 0
    start = time.time()

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            out = model(batch, labels=batch)
            total_loss += float(out["loss"].item())
            n_batches += 1
            total_tokens += batch.numel()

    avg_loss = total_loss / max(1, n_batches)
    ppl = math.exp(avg_loss)
    elapsed = max(time.time() - start, 1e-6)
    return EvalStats(loss=avg_loss, perplexity=ppl, tokens_per_second=total_tokens / elapsed)
