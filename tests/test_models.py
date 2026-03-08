import torch

from dcsa.models.language_models import BaselineLM, DCSALM, LMConfig


def test_tiny_models_forward_pass() -> None:
    cfg = LMConfig(vocab_size=64, max_seq_len=16, d_model=32, n_heads=4, n_layers=2, top_k=4)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))

    baseline = BaselineLM(cfg)
    out_base = baseline(x, labels=x)
    assert out_base["logits"].shape == (2, cfg.max_seq_len, cfg.vocab_size)
    assert out_base["loss"].ndim == 0

    dcsa = DCSALM(cfg)
    out_dcsa = dcsa(x, labels=x)
    assert out_dcsa["logits"].shape == (2, cfg.max_seq_len, cfg.vocab_size)
    assert out_dcsa["loss"].ndim == 0
