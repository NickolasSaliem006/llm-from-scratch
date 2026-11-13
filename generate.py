"""Sample text from a trained checkpoint.

Usage:
    python generate.py --prompt "Max the dog found" --n 1
    python generate.py --interactive
"""

import argparse
from contextlib import nullcontext

import tiktoken
import torch
import torch.nn.functional as F

from model import LLM

# ---------------------------------------------------------------- arguments
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default="checkpoints/best.pt")
parser.add_argument("--prompt", default="Once upon a time")
parser.add_argument("--n", type=int, default=3, help="number of samples per prompt")
parser.add_argument("--max_new", type=int, default=200, help="max new tokens per sample")
parser.add_argument("--temp", type=float, default=0.8, help="sampling temperature")
parser.add_argument("--top_k", type=int, default=50, help="sample only from the k most likely tokens")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--interactive", action="store_true", help="type prompts in a loop")
args = parser.parse_args()

if args.seed is not None:
    torch.manual_seed(args.seed)

# ---------------------------------------------------------------- setup
device = "cuda" if torch.cuda.is_available() else "cpu"
ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()

enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token

ckpt = torch.load(args.ckpt, map_location=device)
cfg = ckpt["config"]
model = LLM(**cfg).to(device)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"loaded {args.ckpt} (val loss {ckpt['val_loss']:.4f})\n")


# ---------------------------------------------------------------- generation
@torch.no_grad()
def generate(idx, max_new_tokens, temperature, top_k):
    for _ in range(max_new_tokens):
        # the model can't attend beyond its context window, so crop the input
        idx_cond = idx[:, -cfg["max_seq_len"]:]
        with ctx:
            logits, _ = model(idx_cond)                    # (1, T, vocab)
        # only the last position predicts the next token; float32 for a stable softmax
        logits = logits[:, -1, :].float() / temperature    # (1, vocab)

        if top_k is not None and top_k > 0:
            kth_best = torch.topk(logits, top_k).values[:, -1:]   # (1, 1)
            logits[logits < kth_best] = float("-inf")

        probs = F.softmax(logits, dim=-1)                  # (1, vocab)
        next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)
        idx = torch.cat((idx, next_id), dim=1)
        if next_id.item() == EOT:
            break
    return idx


def sample(prompt):
    # an empty prompt starts right after an end-of-text token, i.e. a fresh story
    start_ids = enc.encode_ordinary(prompt) if prompt else [EOT]
    for i in range(args.n):
        idx = torch.tensor([start_ids], dtype=torch.long, device=device)
        out = generate(idx, args.max_new, args.temp, args.top_k)
        ids = [t for t in out[0].tolist() if t != EOT]
        print(f"--- sample {i + 1} ---")
        print(enc.decode(ids))
        print()


if args.interactive:
    print('type a prompt and press Enter ("quit" to exit)\n')
    while True:
        prompt = input("prompt> ")
        if prompt.strip().lower() == "quit":
            break
        sample(prompt)
else:
    sample(args.prompt)