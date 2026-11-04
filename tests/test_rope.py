import math
import torch
from rope import precompute_rope, rotate_half, apply_rope

D, T = 8, 32   # head_dim, max_seq_len


def test_table_shapes():
    cos, sin = precompute_rope(D, T)
    assert cos.shape == (T, D) and sin.shape == (T, D)


def test_position_zero_is_identity():
    cos, sin = precompute_rope(D, T)
    assert torch.allclose(cos[0], torch.ones(D))
    assert torch.allclose(sin[0], torch.zeros(D))


def test_halves_duplicated():
    cos, sin = precompute_rope(D, T)
    assert torch.allclose(cos[:, : D // 2], cos[:, D // 2 :])
    assert torch.allclose(sin[:, : D // 2], sin[:, D // 2 :])


def test_slowest_pair_speed():
    base = 10_000.0
    cos, sin = precompute_rope(D, T, base=base)
    theta_last = base ** (-(D - 2) / D)
    angle = torch.atan2(sin[1, D // 2 - 1], cos[1, D // 2 - 1])
    assert math.isclose(angle.item(), theta_last, rel_tol=1e-4)


def test_rotate_half():
    x = torch.tensor([1., 2., 3., 4.])
    assert torch.equal(rotate_half(x), torch.tensor([-3., -4., 1., 2.]))


def test_apply_shape_and_norm():
    cos, sin = precompute_rope(D, T)
    x = torch.randn(2, 4, 10, D)                       # (batch, heads, seq, head_dim)
    y = apply_rope(x, cos[:10], sin[:10])
    assert y.shape == x.shape
    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_first_pair_matches_2d_rotation():
    cos, sin = precompute_rope(D, T)
    x = torch.zeros(1, 1, T, D)
    x[..., 0] = 1.0                                     # pair 0 = (1, 0) at every position
    y = apply_rope(x, cos, sin)
    m = torch.arange(T).float()                         # pair 0 has θ = 1
    assert torch.allclose(y[0, 0, :, 0], torch.cos(m), atol=1e-5)
    assert torch.allclose(y[0, 0, :, D // 2], torch.sin(m), atol=1e-5)


def test_score_depends_only_on_distance():
    torch.manual_seed(0)
    cos, sin = precompute_rope(D, T)
    q = torch.randn(D).expand(1, 1, T, D)               # same q content at every position
    k = torch.randn(D).expand(1, 1, T, D)               # same k content at every position
    qr, kr = apply_rope(q, cos, sin)[0, 0], apply_rope(k, cos, sin)[0, 0]
    s1 = qr[5] @ kr[2]                                  # positions 5 and 2, distance 3
    s2 = qr[25] @ kr[22]                                # positions 25 and 22, distance 3
    assert torch.allclose(s1, s2, atol=1e-5)