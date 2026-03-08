"""Decoder-only language models with baseline and DCSA attention."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from dcsa.models.blocks import BaselineTransformerBlock, DCSATransformerBlock


@dataclass
class LMConfig:
    vocab_size: int = 256
    max_seq_len: int = 128
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.0
    top_k: int = 16
    use_context_gate: bool = True


class _BaseLM(nn.Module):
    def __init__(self, config: LMConfig) -> None:
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.norm_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def _embed(self, input_ids: Tensor) -> Tensor:
        bsz, seqlen = input_ids.shape
        pos = torch.arange(seqlen, device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(pos).unsqueeze(0)
        return self.dropout(x)

    def forward(self, input_ids: Tensor, labels: Tensor | None = None) -> dict[str, Tensor]:
        raise NotImplementedError


class BaselineLM(_BaseLM):
    """Baseline decoder-only LM with standard causal attention."""

    def __init__(self, config: LMConfig) -> None:
        super().__init__(config)
        self.blocks = nn.ModuleList(
            [
                BaselineTransformerBlock(config.d_model, config.n_heads, config.dropout)
                for _ in range(config.n_layers)
            ]
        )

    def forward(self, input_ids: Tensor, labels: Tensor | None = None) -> dict[str, Tensor]:
        x = self._embed(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        out = {"logits": logits}
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, self.config.vocab_size),
                labels[:, 1:].reshape(-1),
            )
            out["loss"] = loss
        return out


class DCSALM(_BaseLM):
    """Decoder-only LM built from DCSA blocks."""

    def __init__(self, config: LMConfig) -> None:
        super().__init__(config)
        self.blocks = nn.ModuleList(
            [
                DCSATransformerBlock(
                    config.d_model,
                    config.n_heads,
                    config.top_k,
                    config.dropout,
                    config.use_context_gate,
                )
                for _ in range(config.n_layers)
            ]
        )

    def forward(self, input_ids: Tensor, labels: Tensor | None = None) -> dict[str, Tensor]:
        x = self._embed(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        out = {"logits": logits}
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, self.config.vocab_size),
                labels[:, 1:].reshape(-1),
            )
            out["loss"] = loss
        return out
