
import math
import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

class SelfAttention(nn.Module):
    def __init__(self, d, d_k, d_v, causal=False):
        super().__init__()
        # three nn.Linear layers, no bias
        self.W_Q = nn.Linear(d,d_k, bias=False)
        self.W_K = nn.Linear(d,d_k, bias=False) 
        self.W_V = nn.Linear(d,d_v, bias=False)
        self.causal=causal

    def forward(self, X):
        Q =self.W_Q(X)
        K =self.W_K(X)
        V =self.W_V(X)

        S = Q @ K.transpose(-2, -1)

        if self.causal:
            T = X.shape[-2]
            mask = torch.tril(torch.ones(T, T, device=X.device))
            S = S.masked_fill(mask == 0, float("-inf"))
        
        A = torch.softmax(S/math.sqrt(K.shape[-1]), dim =-1)
        out = A @ V
        return out, A


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, causal=False):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.causal = causal
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False) 
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, X):
        B, T, _ = X.shape

        Q = self.W_Q(X)
        K = self.W_K(X)
        V = self.W_V(X)

        Q = Q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        S = Q @ K.transpose(-2, -1)

        if self.causal:
            T = X.shape[-2]
            mask = torch.tril(torch.ones(T, T, device=X.device))
            S = S.masked_fill(mask == 0, float("-inf"))
        
        A = torch.softmax(S/math.sqrt(self.d_k), dim =-1)

        out = A @ V
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.W_O(out) 
        return out, A