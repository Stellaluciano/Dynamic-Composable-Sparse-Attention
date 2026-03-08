"""Attention modules for baseline and Dynamic Composable Sparse Attention."""

from __future__ import annotations

import math
from dataclasses import dataclass
import torch
from torch import Tensor, nn


@dataclass
class DCSAAttentionDebug:
    """Optional debug values returned by DCSAAttention."""

    selected_indices: Tensor
    router_scores: Tensor
    head_weights: Tensor
    gate_values: Tensor


class CausalSelfAttention(nn.Module):
    """Standard causal multi-head self-attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)

        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(
            torch.ones(seqlen, seqlen, device=x.device, dtype=torch.bool), diagonal=1
        )
        logits = logits.masked_fill(causal_mask, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        probs = self.dropout(probs)
        out = torch.matmul(probs, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, self.d_model)
        return self.out_proj(out)


class TokenRouter(nn.Module):
    """Predicts token importance scores and selects top-k tokens."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.scorer = nn.Linear(d_model, 1)

    def forward(self, hidden_states: Tensor, k: int) -> tuple[Tensor, Tensor]:
        scores = self.scorer(hidden_states).squeeze(-1)
        k = max(1, min(k, hidden_states.size(1)))
        selected = torch.topk(scores, k=k, dim=-1).indices
        return scores, selected


class SparseAttention(nn.Module):
    """Sparse attention over routed keys/values with causal masking."""

    def __init__(self, head_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: Tensor,
        k_sparse: Tensor,
        v_sparse: Tensor,
        selected_indices: Tensor,
    ) -> Tensor:
        # q: [B, H, T, D], k_sparse/v_sparse: [B, H, K, D], selected_indices: [B, K]
        bsz, n_heads, seqlen, _ = q.shape
        k_tokens = k_sparse.size(2)

        logits = torch.matmul(q, k_sparse.transpose(-2, -1)) / math.sqrt(self.head_dim)

        time = torch.arange(seqlen, device=q.device).view(1, 1, seqlen, 1)
        selected = selected_indices.view(bsz, 1, 1, k_tokens)
        future_mask = selected > time
        logits = logits.masked_fill(future_mask, float("-inf"))

        all_invalid = future_mask.all(dim=-1, keepdim=True)
        logits = torch.where(all_invalid, torch.zeros_like(logits), logits)
        probs = torch.softmax(logits, dim=-1)
        probs = torch.where(all_invalid, torch.zeros_like(probs), probs)
        probs = self.dropout(probs)
        return torch.matmul(probs, v_sparse)


class HeadComposer(nn.Module):
    """Computes dynamic head weights and composes head outputs."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_heads),
        )

    def forward(self, hidden_states: Tensor, head_outputs: Tensor) -> tuple[Tensor, Tensor]:
        # head_outputs: [B, H, T, D]
        pooled = hidden_states.mean(dim=1)
        alpha = torch.softmax(self.net(pooled), dim=-1)
        composed = (head_outputs * alpha.view(alpha.size(0), alpha.size(1), 1, 1)).sum(dim=1)
        return composed, alpha


class ContextGate(nn.Module):
    """Gates composed attention output against residual stream."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, residual: Tensor, composed: Tensor) -> tuple[Tensor, Tensor]:
        gate = torch.sigmoid(self.proj(residual))
        return gate * composed + (1.0 - gate) * residual, gate


class DCSAAttention(nn.Module):
    """Integrated Dynamic Composable Sparse Attention module."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        top_k: int,
        dropout: float = 0.0,
        use_context_gate: bool = True,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.top_k = top_k
        self.head_dim = d_model // n_heads
        self.use_context_gate = use_context_gate

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.router = TokenRouter(d_model)
        self.sparse_attn = SparseAttention(self.head_dim, dropout=dropout)
        self.composer = HeadComposer(d_model, n_heads)
        self.context_gate = ContextGate(d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, hidden_states: Tensor, return_debug: bool = False) -> Tensor | tuple[Tensor, DCSAAttentionDebug]:
        bsz, seqlen, _ = hidden_states.shape
        scores, selected = self.router(hidden_states, k=self.top_k)
        selected_sorted = selected.sort(dim=-1).values

        q, k, v = self.qkv(hidden_states).chunk(3, dim=-1)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)

        gather_idx = selected_sorted.view(bsz, 1, -1, 1).expand(-1, self.n_heads, -1, self.head_dim)
        k_sparse = torch.gather(k, dim=2, index=gather_idx)
        v_sparse = torch.gather(v, dim=2, index=gather_idx)

        head_outputs = self.sparse_attn(q, k_sparse, v_sparse, selected_sorted)
        composed, alpha = self.composer(hidden_states, head_outputs)

        if self.use_context_gate:
            mixed, gate = self.context_gate(hidden_states, composed)
        else:
            mixed, gate = composed, torch.ones_like(hidden_states)

        out = self.out_proj(mixed)
        if not return_debug:
            return out
        debug = DCSAAttentionDebug(
            selected_indices=selected_sorted,
            router_scores=scores,
            head_weights=alpha,
            gate_values=gate,
        )
        return out, debug
