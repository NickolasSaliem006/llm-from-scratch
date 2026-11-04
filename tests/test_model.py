import math
import torch
from model import LLM

V, D, L, H, F = 100, 32, 2, 4, 64   # vocab, d_model, layers, heads, d_hidden


def make():
    torch.manual_seed(0)
    return LLM(V, D, L, H, F, max_seq_len=64)


def test_logits_shape_and_no_loss():
    logits, loss = make()(torch.randint(0, V, (2, 10)))
    assert logits.shape == (2, 10, V)
    assert loss is None


def test_loss_is_scalar():
    idx = torch.randint(0, V, (2, 10))
    _, loss = make()(idx, targets=idx)
    assert loss.dim() == 0


def test_initial_loss_near_uniform():
    idx = torch.randint(0, V, (4, 16))
    tgt = torch.randint(0, V, (4, 16))
    _, loss = make()(idx, tgt)
    assert abs(loss.item() - math.log(V)) < 0.3      # random guess = ln(vocab)


def test_weight_tying():
    m = make()
    assert m.lm_head.weight is m.tok_emb.weight


def test_param_count():
    n = sum(p.numel() for p in make().parameters())
    per_block = 2 * D + 4 * D * D + 3 * D * F
    assert n == V * D + L * per_block + D             # embedding counted once (tied)


def test_no_future_leak():
    m = make()
    a = torch.randint(0, V, (1, 12))
    b = a.clone()
    b[0, 8:] = (b[0, 8:] + 1) % V                     # change only the last 4 tokens
    la, _ = m(a)
    lb, _ = m(b)
    assert torch.allclose(la[0, :8], lb[0, :8], atol=1e-5)


def test_overfits_one_batch():
    m = make()
    x = torch.randint(0, V, (4, 16))
    y = torch.roll(x, -1, dims=1)                     # "next token" targets
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(300):
        _, loss = m(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1