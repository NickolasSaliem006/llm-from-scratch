
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


X   = torch.tensor([[1,0,1,1],[1,0,0,0],[0,1,0,0]], dtype=torch.float32, device=device)
W_Q = torch.tensor([[0,0],[1,1],[0,0],[0,0]],       dtype=torch.float32, device=device)
W_K = torch.tensor([[1,0],[0,0],[0,1],[0,1]],       dtype=torch.float32, device=device)
W_V = torch.tensor([[1,0],[0,0],[0,1],[0,0]],       dtype=torch.float32, device=device)

model = SelfAttention(4, 2, 2).to(device)
print(sum(p.numel() for p in model.parameters()))

with torch.no_grad():
    model.W_Q.weight.copy_(W_Q.T)
    model.W_K.weight.copy_(W_K.T)
    model.W_V.weight.copy_(W_V.T)

out, A = model(X)
print(A[2], out[2])

X2 = torch.randn(2, 5, 8, device=device)
model2 = SelfAttention(8, 4, 4).to(device)
out2, A2 = model2(X2)
print(out2.shape, A2.shape)
print(A2.sum(dim=-1))