#!/usr/bin/env python
"""Lightweight benchmark comparing baseline and DCSA language model paths."""

from __future__ import annotations

import argparse
import time

import torch

from dcsa.eval.metrics import approximate_attention_cost, parameter_count
from dcsa.models.language_models import BaselineLM, DCSALM, LMConfig


def measure_forward(model: torch.nn.Module, input_ids: torch.Tensor, n_iters: int = 20) -> float:
    model.eval()
    with torch.no_grad():
        _ = model(input_ids)
        start = time.time()
        for _ in range(n_iters):
            _ = model(input_ids)
    return (time.time() - start) / n_iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=256)
    args = parser.parse_args()

    cfg = LMConfig(
        vocab_size=args.vocab_size,
        max_seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        top_k=args.top_k,
    )

    baseline = BaselineLM(cfg)
    dcsa = DCSALM(cfg)
    batch = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len))

    t_base = measure_forward(baseline, batch)
    t_dcsa = measure_forward(dcsa, batch)

    print("=== Parameter count ===")
    print(f"baseline={parameter_count(baseline)}")
    print(f"dcsa={parameter_count(dcsa)}")

    print("=== Throughput proxy (sec/forward, lower is better) ===")
    print(f"baseline={t_base:.6f}")
    print(f"dcsa={t_dcsa:.6f}")

    cost = approximate_attention_cost(args.seq_len, args.top_k, args.d_model, args.n_heads)
    print("=== Approximate attention cost terms ===")
    print(f"dense={cost['dense']}")
    print(f"sparse_dcsa={cost['sparse_dcsa']}")


if __name__ == "__main__":
    main()
