import torch
from attention import SelfAttention

X_TOY = torch.tensor([[1, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.float32)
W_Q_TOY = torch.tensor([[0, 0], [1, 1], [0, 0], [0, 0]], dtype=torch.float32)
W_K_TOY = torch.tensor([[1, 0], [0, 0], [0, 1], [0, 1]], dtype=torch.float32)
W_V_TOY = torch.tensor([[1, 0], [0, 0], [0, 1], [0, 0]], dtype=torch.float32)


def toy_model(**kwargs):
    model = SelfAttention(4, 2, 2, **kwargs)
    with torch.no_grad():
        model.W_Q.weight.copy_(W_Q_TOY.T)
        model.W_K.weight.copy_(W_K_TOY.T)
        model.W_V.weight.copy_(W_V_TOY.T)
    return model


def test_param_count():
    model = SelfAttention(4, 2, 2)
    assert sum(p.numel() for p in model.parameters()) == 24


def test_toy_numbers():
    out, A = toy_model()(X_TOY)
    assert torch.allclose(A[2], torch.tensor([0.7337, 0.1784, 0.0879]), atol=1e-4)
    assert torch.allclose(out[2], torch.tensor([0.9121, 0.7337]), atol=1e-4)


def test_batch_shapes():
    out, A = SelfAttention(8, 4, 4)(torch.randn(2, 5, 8))
    assert out.shape == (2, 5, 4)
    assert A.shape == (2, 5, 5)


def test_rows_sum_to_one():
    torch.manual_seed(0)
    _, A = SelfAttention(8, 4, 4)(torch.randn(2, 5, 8))
    assert torch.allclose(A.sum(dim=-1), torch.ones(2, 5), atol=1e-6)


def test_causal_toy_rows():
    _, A = toy_model(causal=True)(X_TOY)
    assert torch.allclose(A[0], torch.tensor([1.0, 0.0, 0.0]), atol=1e-4)
    assert torch.allclose(A[1], torch.tensor([0.5, 0.5, 0.0]), atol=1e-4)
    assert torch.allclose(A[2], torch.tensor([0.7337, 0.1784, 0.0879]), atol=1e-4)


def test_causal_upper_triangle_is_zero():
    torch.manual_seed(0)
    _, A = SelfAttention(8, 4, 4, causal=True)(torch.randn(2, 5, 8))
    assert torch.all(torch.triu(A, diagonal=1) == 0)


def test_causal_no_future_leak():
    torch.manual_seed(0)
    model = SelfAttention(8, 4, 4, causal=True)
    X_a = torch.randn(2, 5, 8)
    X_b = X_a.clone()
    X_b[:, -1] = torch.randn(2, 8)
    out_a, _ = model(X_a)
    out_b, _ = model(X_b)
    assert torch.allclose(out_a[:, :4], out_b[:, :4], atol=1e-6)
    assert not torch.allclose(out_a[:, 4], out_b[:, 4], atol=1e-6)