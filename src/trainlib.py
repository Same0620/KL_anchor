"""Minimal SFT loop (used for `full`, and in Phase B for IdkNLL and RTT): bf16 weights, 32-bit paged AdamW
(bitsandbytes), loss only on answer tokens + <|im_end|>."""
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import train_ids
from gpulib import DEV, log


def encode_rows(tok, rows):
    return [train_ids(tok, r["user"], r["assistant"]) for r in rows]


def batches(encoded, bs, rng, pad):
    order = list(range(len(encoded)))
    rng.shuffle(order)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        L = max(len(encoded[i][0]) for i in idx)
        ids = torch.full((len(idx), L), pad, dtype=torch.long)
        lab = torch.full((len(idx), L), -100, dtype=torch.long)
        am = torch.zeros((len(idx), L), dtype=torch.long)
        for r, i in enumerate(idx):
            x, y = encoded[i]
            ids[r, :len(x)] = torch.tensor(x)
            lab[r, :len(y)] = torch.tensor(y)
            am[r, :len(x)] = 1
        yield ids, am, lab


class SFT:
    def __init__(self, model, lr=1e-5, weight_decay=0.01, warmup_steps=100, params=None):
        import bitsandbytes as bnb
        self.model = model
        self.params = [p for p in (params if params is not None else model.parameters()) if p.requires_grad]
        self.opt = bnb.optim.PagedAdamW32bit(self.params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
        self.lr, self.warmup, self.step_n = lr, warmup_steps, 0

    def step(self, loss):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, 1.0)
        self.step_n += 1
        for g in self.opt.param_groups:
            g["lr"] = self.lr * min(1.0, self.step_n / max(1, self.warmup))   # linear warm-up, then constant
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)

    def close(self):
        del self.opt
        self.opt = None


def train_epochs(model, sft, encoded, epochs, bs, seed, pad, log_every=50, tag=""):
    model.train()
    rng = random.Random(seed)
    t = time.time()
    hist = []
    for ep in range(epochs):
        tot, n = 0.0, 0
        for ids, am, lab in batches(encoded, bs, rng, pad):
            out = model(input_ids=ids.to(DEV), attention_mask=am.to(DEV), labels=lab.to(DEV), use_cache=False)
            sft.step(out.loss)
            tot += out.loss.item()
            n += 1
            if sft.step_n % log_every == 0:
                log(f"{tag} step {sft.step_n} loss {out.loss.item():.4f} ({time.time() - t:.0f}s)")
        hist.append(tot / n)
        log(f"{tag} epoch {ep + 1}/{epochs} mean loss {tot / n:.4f} ({time.time() - t:.0f}s)")
    model.eval()
    return hist
