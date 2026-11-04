import torch.nn as nn
import torch.nn.functional as F
from transformer import TransformerBlock
from layers import RMSNorm


class LLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_hidden, max_seq_len=2048):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([TransformerBlock( d_model, n_heads, d_hidden, max_seq_len=max_seq_len) for n in range(n_layers)])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias= False)
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)
    

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        x = self.tok_emb(idx)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),   # (B*T, vocab)
            targets.view(-1),                   # (B*T,)
        )
        return logits, loss
