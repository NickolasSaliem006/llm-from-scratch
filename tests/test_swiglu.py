import torch
import torch.nn.functional as F
from layers import SwiGLU          


def test_swiglu_shape():
    torch.manual_seed(0)
    ffn = SwiGLU(d_model=8, d_hidden=21)
    x = torch.randn(2, 5, 8)
    assert ffn(x).shape == (2, 5, 8)


def test_swiglu_formula():
    torch.manual_seed(0)
    ffn = SwiGLU(8, 21)
    x = torch.randn(2, 5, 8)
    ref = ffn.W_down(F.silu(ffn.W_gate(x)) * ffn.W_up(x))
    assert torch.allclose(ffn(x), ref)


def test_swiglu_no_bias():
    ffn = SwiGLU(8, 21)
    for layer in (ffn.W_gate, ffn.W_up, ffn.W_down):
        assert layer.bias is None


def test_swiglu_learns():
    torch.manual_seed(0)
    teacher, student = SwiGLU(8, 21), SwiGLU(8, 21)
    X = torch.randn(256, 8)
    with torch.no_grad():
        Y = teacher(X)
    opt = torch.optim.Adam(student.parameters(), lr=1e-2)
    start = F.mse_loss(student(X), Y).item()
    for _ in range(1000):
        loss = F.mse_loss(student(X), Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < start * 0.01