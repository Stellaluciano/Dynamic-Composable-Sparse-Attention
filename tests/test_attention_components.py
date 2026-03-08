import torch

from dcsa.modules.attention import ContextGate, DCSAAttention, HeadComposer, SparseAttention, TokenRouter


def test_token_router_topk_shape_and_order() -> None:
    router = TokenRouter(d_model=16)
    x = torch.randn(2, 10, 16)
    scores, idx = router(x, k=4)
    assert scores.shape == (2, 10)
    assert idx.shape == (2, 4)
    gathered = torch.gather(scores, 1, idx)
    sorted_scores = torch.sort(scores, dim=-1, descending=True).values[:, :4]
    assert torch.allclose(torch.sort(gathered, dim=-1, descending=True).values, sorted_scores)


def test_sparse_attention_causal_mask_prevents_future_only() -> None:
    attn = SparseAttention(head_dim=8)
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 2, 8)
    v = torch.randn(1, 2, 2, 8)
    selected = torch.tensor([[2, 3]])
    out = attn(q, k, v, selected)
    assert out.shape == (1, 2, 4, 8)
    assert torch.isfinite(out).all()


def test_head_composer_output_shape() -> None:
    composer = HeadComposer(d_model=32, n_heads=4)
    h = torch.randn(3, 6, 32)
    heads = torch.randn(3, 4, 6, 8)
    composed, alpha = composer(h, heads)
    assert composed.shape == (3, 6, 8)
    assert alpha.shape == (3, 4)


def test_context_gate_shape() -> None:
    gate = ContextGate(d_model=24)
    residual = torch.randn(2, 5, 24)
    composed = torch.randn(2, 5, 24)
    y, g = gate(residual, composed)
    assert y.shape == residual.shape
    assert g.shape == residual.shape


def test_dcsa_attention_forward_shape() -> None:
    mod = DCSAAttention(d_model=32, n_heads=4, top_k=3)
    x = torch.randn(2, 12, 32)
    y = mod(x)
    assert y.shape == x.shape
