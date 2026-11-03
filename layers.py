import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight
    

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.W_up = nn.Linear(d_model, d_hidden, bias=False) 
        self.W_gate = nn.Linear(d_model, d_hidden, bias=False)
        self.W_down = nn.Linear(d_hidden, d_model, bias=False)  
    def forward(self, x):
        out= self.W_down(F.silu(self.W_gate(x)) * self.W_up(x))
        return out