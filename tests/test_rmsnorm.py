import torch
import torch.nn as nn
from layers import RMSNorm


def test_shape_preserved():
    x = torch.randn(2, 5, 16)
    assert RMSNorm(16)(x).shape == (2, 5, 16)


def test_param_count():
    assert sum(p.numel() for p in RMSNorm(16).parameters()) == 16


def test_output_rms_is_one():
    torch.manual_seed(0)
    y = RMSNorm(16)(torch.randn(2, 5, 16) * 7)
    rms = y.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones(2, 5), atol=1e-4)


def test_scale_invariant():
    torch.manual_seed(0)
    norm = RMSNorm(16)
    x = torch.randn(2, 5, 16)
    assert torch.allclose(norm(x), norm(x * 50), atol=1e-4)


def test_weight_scales_output():
    torch.manual_seed(0)
    norm = RMSNorm(16)
    with torch.no_grad():
        norm.weight.fill_(2.0)
    y = norm(torch.randn(2, 5, 16))
    rms = y.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.full((2, 5), 2.0), atol=1e-4)


def test_matches_pytorch():
    torch.manual_seed(0)
    ours = RMSNorm(16, eps=1e-6)
    ref = nn.RMSNorm(16, eps=1e-6)
    with torch.no_grad():
        w = torch.randn(16)
        ours.weight.copy_(w)
        ref.weight.copy_(w)
    x = torch.randn(2, 5, 16)
    assert torch.allclose(ours(x), ref(x), atol=1e-5)