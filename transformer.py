import torch.nn as nn
from layers import RMSNorm, SwiGLU
from attention import MultiHeadAttention



class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_hidden, max_seq_len=2048):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, causal=True, max_seq_len=max_seq_len)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_hidden)

    def forward(self, x):
        a, _ = self.attn(self.norm1(x))
        x = x + a
        x = x + self.ffn(self.norm2(x))
        
        return x
