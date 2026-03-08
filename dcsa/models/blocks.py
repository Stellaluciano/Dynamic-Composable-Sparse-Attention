"""Transformer blocks for baseline attention and DCSA attention."""

from __future__ import annotations

from torch import Tensor, nn

from dcsa.modules.attention import CausalSelfAttention, DCSAAttention


class FeedForward(nn.Module):
    """Simple Transformer MLP block."""

    def __init__(self, d_model: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class BaselineTransformerBlock(nn.Module):
    """Decoder block with standard causal self-attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class DCSATransformerBlock(nn.Module):
    """Decoder block with Dynamic Composable Sparse Attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        top_k: int,
        dropout: float = 0.0,
        use_context_gate: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = DCSAAttention(
            d_model=d_model,
            n_heads=n_heads,
            top_k=top_k,
            dropout=dropout,
            use_context_gate=use_context_gate,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
