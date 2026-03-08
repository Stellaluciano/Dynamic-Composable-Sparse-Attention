#!/usr/bin/env python
"""Train a tiny baseline or DCSA language model."""

from __future__ import annotations

import argparse

from torch.utils.data import DataLoader

from dcsa.models.language_models import BaselineLM, DCSALM, LMConfig
from dcsa.training.trainer import train_language_model
from dcsa.utils.config import load_config
from dcsa.utils.data import RandomTokenDataset, TinyPatternDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = LMConfig(**cfg["model"])
    train_cfg = cfg["train"]

    if train_cfg.get("dataset", "random") == "pattern":
        dataset = TinyPatternDataset(train_cfg["n_samples"], model_cfg.max_seq_len, model_cfg.vocab_size)
    else:
        dataset = RandomTokenDataset(train_cfg["n_samples"], model_cfg.max_seq_len, model_cfg.vocab_size)

    dataloader = DataLoader(dataset, batch_size=train_cfg["batch_size"], shuffle=True)
    model = DCSALM(model_cfg) if cfg["model_type"] == "dcsa" else BaselineLM(model_cfg)

    stats = train_language_model(
        model=model,
        dataloader=dataloader,
        lr=train_cfg["lr"],
        steps=train_cfg["steps"],
        device=train_cfg.get("device", "cpu"),
    )
    print(f"final_loss={stats.final_loss:.4f}")
    print(f"tokens_per_second={stats.tokens_per_second:.2f}")


if __name__ == "__main__":
    main()
