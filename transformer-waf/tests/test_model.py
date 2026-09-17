from __future__ import annotations

import torch

from src.models.transformer_model import WAFTransformer


def test_forward_shapes():
    model = WAFTransformer(vocab_size=100, embed_dim=32, num_heads=4, num_layers=2, ff_dim=64, dropout=0.1, max_len=16)
    input_ids = torch.randint(0, 100, (2, 16))
    mask = torch.ones_like(input_ids)
    logits = model(input_ids, mask)
    assert logits.shape == (2, 16, 100)




def test_scoring_is_causal_no_future_leakage():
    """The next-token objective must not let position t see tokens > t."""
    torch.manual_seed(0)
    model = WAFTransformer(vocab_size=50, embed_dim=32, num_heads=4, num_layers=2, ff_dim=64, dropout=0.0, max_len=16)
    model.eval()
    prefix = torch.randint(4, 50, (1, 8))
    full = torch.cat([prefix, torch.randint(4, 50, (1, 8))], dim=1)
    mask_prefix = torch.ones_like(prefix)
    mask_full = torch.ones_like(full)
    with torch.no_grad():
        a = model(prefix, mask_prefix)[0, :8]
        b = model(full, mask_full)[0, :8]
    assert torch.allclose(a, b, atol=1e-5)


def test_reconstruction_error_shape_and_positive():
    model = WAFTransformer(vocab_size=50, embed_dim=32, num_heads=4, num_layers=1, ff_dim=64, dropout=0.0, max_len=16)
    model.eval()
    ids = torch.randint(4, 50, (3, 16))
    mask = torch.ones_like(ids)
    mask[1, 10:] = 0
    err = model.get_reconstruction_error(ids, mask)
    assert err.shape == (3,) and bool((err > 0).all())
