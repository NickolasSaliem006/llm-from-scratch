import torch
from transformer import TransformerBlock

D, H, F = 16, 2, 40   # d_model, n_heads, d_hidden


def test_shape():
    blk = TransformerBlock(D, H, F)
    x = torch.randn(2, 7, D)
    assert blk(x).shape == (2, 7, D)


def test_param_count():
    blk = TransformerBlock(D, H, F)
    n = sum(p.numel() for p in blk.parameters())
    expected = 2 * D + 4 * D * D + 3 * D * F     # two norms + Q,K,V,O + gate,up,down
    assert n == expected


def test_residual_identity():
    blk = TransformerBlock(D, H, F)
    with torch.no_grad():
        blk.attn.W_O.weight.zero_()              # attention adds nothing
        blk.ffn.W_down.weight.zero_()            # FFN adds nothing
    x = torch.randn(2, 7, D)
    assert torch.allclose(blk(x), x)             # so x passes straight through


def test_no_future_leak():
    torch.manual_seed(0)
    blk = TransformerBlock(D, H, F)
    x = torch.randn(1, 6, D)
    x2 = x.clone()
    x2[0, 4:] = torch.randn(2, D)                # change only words 4 and 5
    y, y2 = blk(x), blk(x2)
    assert torch.allclose(y[0, :4], y2[0, :4], atol=1e-6)   # words 0-3 unaffected


def test_gradients_reach_everything():
    blk = TransformerBlock(D, H, F)
    blk(torch.randn(2, 5, D)).sum().backward()
    for name, p in blk.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name