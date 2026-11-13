import math
import os
import time
from contextlib import nullcontext

import numpy as np
import torch

from model import LLM

# ---------------------------------------------------------------- config
TRAIN_BIN = "data/tinystories_train.bin"
VAL_BIN = "data/tinystories_validation.bin"
OUT_DIR = "checkpoints"

vocab_size = 50257      # GPT-2 tokenizer
d_model = 384
n_layers = 6
n_heads = 6
d_hidden = 1024         # ≈ 8/3 × d_model
seq_len = 256           # T
batch_size = 32         # B
grad_accum_steps = 1    # micro-batches per optimizer step

max_steps = 38000
warmup_steps = 200
max_lr = 3e-4
min_lr = 3e-5
weight_decay = 0.1
grad_clip = 1.0

eval_every = 500
eval_iters = 50
log_every = 20

# SMOKE=1 python train.py  → tiny settings, runs in a minute on a laptop CPU
if os.environ.get("SMOKE") == "1":
    d_model, n_layers, n_heads, d_hidden = 64, 2, 2, 172
    seq_len, batch_size = 64, 8
    max_steps, warmup_steps = 60, 10
    eval_every, eval_iters, log_every = 30, 5, 10
    grad_accum_steps = 2
    OUT_DIR = "checkpoints_smoke"   # never overwrite real checkpoints

# ---------------------------------------------------------------- device
device = "cuda" if torch.cuda.is_available() else "cpu"
# bf16 autocast on GPU: matmuls run in bfloat16 (faster, less memory), weights stay float32
ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()
print(f"device: {device}")


# ---------------------------------------------------------------- data
def get_batch(split):
    path = TRAIN_BIN if split == "train" else VAL_BIN
    data = np.memmap(path, dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + seq_len].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + seq_len].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


# ---------------------------------------------------------------- learning-rate schedule
def get_lr(step):
    if step < warmup_steps:                                   # 1) linear warmup
        return max_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))  # 2) cosine: 1 → 0
    return min_lr + coeff * (max_lr - min_lr)


def fmt_time(seconds):
    seconds = int(seconds)
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h}h{m:02d}m{s:02d}s"


# ---------------------------------------------------------------- model + optimizer
model = LLM(vocab_size, d_model, n_layers, n_heads, d_hidden, max_seq_len=seq_len).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"parameters: {n_params / 1e6:.1f}M")

RESUME = os.environ.get("RESUME")      # e.g. checkpoints/last.pt
start_step = 0
best_val = float("inf")
if RESUME:
    ckpt = torch.load(RESUME, map_location=device)
    model.load_state_dict(ckpt["model"])
    start_step = ckpt["step"] + 1
    best_val = ckpt.get("best_val", ckpt["val_loss"])   # old checkpoints have no "best_val"
    print(f"resumed from {RESUME} (step {ckpt['step']}, best val {best_val:.4f})")

# weight decay only on matrices (Linear/Embedding), not on norm weights (1D)
decay = [p for p in model.parameters() if p.dim() >= 2]
no_decay = [p for p in model.parameters() if p.dim() < 2]
optimizer = torch.optim.AdamW(
    [{"params": decay, "weight_decay": weight_decay},
     {"params": no_decay, "weight_decay": 0.0}],
    lr=max_lr, betas=(0.9, 0.95),
)
if RESUME:
    optimizer.load_state_dict(ckpt["optimizer"])


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def save(path, step, val_loss):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "config": config, "step": step, "val_loss": val_loss, "best_val": best_val},
               path)


# ---------------------------------------------------------------- training loop
os.makedirs(OUT_DIR, exist_ok=True)
config = dict(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
              n_heads=n_heads, d_hidden=d_hidden, max_seq_len=seq_len)
t_start = time.time()
t_last = time.time()

for step in range(start_step, max_steps + 1):
    lr = get_lr(step)
    for group in optimizer.param_groups:
        group["lr"] = lr

    # evaluate + checkpoint
    if step % eval_every == 0:
        losses = estimate_loss()
        print(f"step {step:6d} | train {losses['train']:.4f} | val {losses['val']:.4f}")
        if losses["val"] < best_val:
            best_val = losses["val"]
            save(os.path.join(OUT_DIR, "best.pt"), step, losses["val"])
        save(os.path.join(OUT_DIR, "last.pt"), step, losses["val"])   # always: for crash resume
    if step == max_steps:
        break

    # one optimizer step = grad_accum_steps micro-batches
    optimizer.zero_grad(set_to_none=True)
    for _ in range(grad_accum_steps):
        x, y = get_batch("train")
        with ctx:
            _, loss = model(x, y)
        loss = loss / grad_accum_steps   # backward() adds into .grad, so average over micro-batches
        loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    # logging
    if step % log_every == 0 and step > start_step:
        if device == "cuda":
            torch.cuda.synchronize()
        now = time.time()
        dt = now - t_last
        t_last = now
        tok_per_sec = batch_size * seq_len * grad_accum_steps * log_every / dt

        elapsed = now - t_start
        eta = elapsed / (step - start_step) * (max_steps - step)
        print(f"step {step:6d} | loss {loss.item() * grad_accum_steps:.4f} | lr {lr:.2e} | "
              f"grad {grad_norm:.2f} | {tok_per_sec:,.0f} tok/s | "
              f"elapsed {fmt_time(elapsed)} | eta {fmt_time(eta)}")

print(f"done in {fmt_time(time.time() - t_start)}. best val loss {best_val:.4f}")