#!/usr/bin/env python
"""Evaluate tiny baseline or DCSA language model on toy data."""

from __future__ import annotations

import argparse

from torch.utils.data import DataLoader

from dcsa.eval.metrics import evaluate_language_model, parameter_count
from dcsa.models.language_models import BaselineLM, DCSALM, LMConfig
from dcsa.utils.config import load_config
from dcsa.utils.data import RandomTokenDataset, TinyPatternDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = LMConfig(**cfg["model"])
    eval_cfg = cfg["eval"]

    if eval_cfg.get("dataset", "random") == "pattern":
        dataset = TinyPatternDataset(eval_cfg["n_samples"], model_cfg.max_seq_len, model_cfg.vocab_size)
    else:
        dataset = RandomTokenDataset(eval_cfg["n_samples"], model_cfg.max_seq_len, model_cfg.vocab_size)

    dataloader = DataLoader(dataset, batch_size=eval_cfg["batch_size"], shuffle=False)
    model = DCSALM(model_cfg) if cfg["model_type"] == "dcsa" else BaselineLM(model_cfg)

    stats = evaluate_language_model(model, dataloader, device=eval_cfg.get("device", "cpu"))
    print(f"loss={stats.loss:.4f}")
    print(f"perplexity={stats.perplexity:.4f}")
    print(f"tokens_per_second={stats.tokens_per_second:.2f}")
    print(f"parameter_count={parameter_count(model)}")


if __name__ == "__main__":
    main()
