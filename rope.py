import torch


def precompute_rope(head_dim, max_seq_len, base=1_000_000.0):
    rot_speed=torch.zeros(head_dim//2)
    angles=torch.zeros(max_seq_len, head_dim//2)

    for i in range(head_dim//2):
        rot_speed[i]= base**(-2*i / head_dim)

    angles=torch.zeros(max_seq_len, head_dim//2)

    for m in range(max_seq_len):
        for i in range(head_dim // 2):
            angles[m, i] = m * rot_speed[i]

    angles = torch.cat((angles, angles), dim=-1)
    return torch.cos(angles), torch.sin(angles)

def rotate_half(x):
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x, cos, sin):
    # x: (batch, n_heads, seq_len, head_dim)
    # cos, sin: (seq_len, head_dim), broadcast over batch and heads
    return x * cos + rotate_half(x) * sin