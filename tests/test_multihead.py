import pytest
import torch
from attention import MultiHeadAttention, SelfAttention


def test_param_count():
    model = MultiHeadAttention(8, 2)
    assert sum(p.numel() for p in model.parameters()) == 4 * 8 * 8


def test_bad_head_count_rejected():
    with pytest.raises(AssertionError):
        MultiHeadAttention(8, 3)


def test_shapes():
    out, A = MultiHeadAttention(8, 2)(torch.randn(2, 5, 8))
    assert out.shape == (2, 5, 8)
    assert A.shape == (2, 2, 5, 5)


def test_rows_sum_to_one():
    torch.manual_seed(0)
    _, A = MultiHeadAttention(8, 2)(torch.randn(2, 5, 8))
    assert torch.allclose(A.sum(dim=-1), torch.ones(2, 2, 5), atol=1e-6)


def test_one_head_matches_single_head():
    torch.manual_seed(0)
    mha = MultiHeadAttention(8, 1, causal=True)
    sa = SelfAttention(8, 8, 8, causal=True)
    with torch.no_grad():
        sa.W_Q.weight.copy_(mha.W_Q.weight)
        sa.W_K.weight.copy_(mha.W_K.weight)
        sa.W_V.weight.copy_(mha.W_V.weight)
        mha.W_O.weight.copy_(torch.eye(8))
    X = torch.randn(2, 5, 8)
    out_mha, A_mha = mha(X)
    out_sa, A_sa = sa(X)
    assert torch.allclose(A_mha[:, 0], A_sa, atol=1e-6)
    assert torch.allclose(out_mha, out_sa, atol=1e-6)


def test_heads_are_different():
    torch.manual_seed(0)
    _, A = MultiHeadAttention(8, 2)(torch.randn(2, 5, 8))
    assert not torch.allclose(A[:, 0], A[:, 1], atol=1e-3)


def test_causal_upper_triangle_is_zero():
    torch.manual_seed(0)
    _, A = MultiHeadAttention(8, 2, causal=True)(torch.randn(2, 5, 8))
    assert torch.all(torch.triu(A, diagonal=1) == 0)


def test_causal_no_future_leak():
    torch.manual_seed(0)
    model = MultiHeadAttention(8, 2, causal=True)
    X_a = torch.randn(2, 5, 8)
    X_b = X_a.clone()
    X_b[:, -1] = torch.randn(2, 8)
    out_a, _ = model(X_a)
    out_b, _ = model(X_b)
    assert torch.allclose(out_a[:, :4], out_b[:, :4], atol=1e-6)
    assert not torch.allclose(out_a[:, 4], out_b[:, 4], atol=1e-6)