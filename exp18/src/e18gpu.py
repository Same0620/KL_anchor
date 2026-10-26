"""GPU helpers for Experiment 18: base model, sequences, summed NLL, hand-written LoRA training from the A0 files.

A sequence is (ids, n0): the loss covers the predictions of ids[n0:] (C1 documents: n0 = 1, every next token; C2 rows: n0 = the prompt length,
the answer targets). Loss tokens of a sequence = len(ids) - n0.
"""
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e18common as X
from e18common import C

DEV = "cuda"


def log_to(prefix):
    t0 = time.time()

    def log(msg):
        print(f"[{prefix} {time.time() - t0:6.0f}s] {msg}", flush=True)
    return log


def load_tok():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(C.snapshot_dir(C.BASE_MODEL, C.BASE_REV)))
    return tok


def load_base():
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(str(C.snapshot_dir(C.BASE_MODEL, C.BASE_REV)), dtype=torch.bfloat16, device_map=DEV,
                                             attn_implementation="sdpa").eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def target_weights(model):
    """{module_key: weight parameter} for the 7 linear modules of the 40 layers (base model, no adapter)."""
    out = {}
    for L in range(X.N_LAYERS):
        layer = model.model.layers[L]
        for mod in X.MODULES:
            sub = layer.self_attn if mod in ("q_proj", "k_proj", "v_proj", "o_proj") else layer.mlp
            out[X.module_key(L, mod)] = getattr(sub, mod).weight
    return out


def c1_seq(doc):
    return (list(doc["ids"]), 1)


def c2_seq(row):
    ids = list(row["prompt_ids"]) + list(row["target_ids"])
    return (ids[:-1] + [ids[-1]], len(row["prompt_ids"]))


def micro_batches(seqs, budget=2048, max_rows=16):
    """Consecutive groups (order kept) with rows x longest row <= budget."""
    out, cur = [], []
    for s in seqs:
        n = len(s[0])
        if cur and (len(cur) >= max_rows or (len(cur) + 1) * max(n, max(len(x[0]) for x in cur)) > budget):
            out.append(cur)
            cur = []
        cur.append(s)
    if cur:
        out.append(cur)
    return out


def loss_sum(model, seqs):
    """Sum over the batch of the per-token NLL (fp32 log-softmax) and the number of loss tokens. Graph kept for backward."""
    n = max(len(s[0]) for s in seqs)
    ids = torch.tensor([s[0] + [151643] * (n - len(s[0])) for s in seqs], device=DEV)
    att = torch.tensor([[1] * len(s[0]) + [0] * (n - len(s[0])) for s in seqs], device=DEV)
    logits = model(input_ids=ids, attention_mask=att, use_cache=False).logits
    tot, ntok = 0.0, 0
    for b, (s, n0) in enumerate(seqs):
        lg = logits[b, n0 - 1:len(s) - 1].float()
        tot = tot + torch.nn.functional.cross_entropy(lg, torch.tensor(s[n0:], device=DEV), reduction="sum")
        ntok += len(s) - n0
    return tot, ntok


# ------------------------------------------------------------------------------------------------ LoRA training
def lora_config():
    from peft import LoraConfig
    return LoraConfig(r=X.RANK, lora_alpha=X.ALPHA, use_rslora=False, lora_dropout=0.0, bias="none", target_modules=X.MODULES,
                      init_lora_weights=True, task_type="CAUSAL_LM")


class LoRATrainer:
    """One PEFT wrapper reused for many LoRAs: reset() puts A = A0 (asserted) and B = 0; train() runs AdamW steps."""

    def __init__(self, base):
        from peft import get_peft_model
        self.pm = get_peft_model(base, lora_config())
        self.mods = {}
        for name, mod in self.pm.named_modules():
            if hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {}):
                self.mods[name.replace("base_model.model.", "")] = mod
        assert sorted(self.mods) == sorted(X.ALL_KEYS)
        assert all(float(m.scaling["default"]) == X.SCALE for m in self.mods.values())

    @torch.no_grad()
    def reset(self, a0):
        for k, m in self.mods.items():
            m.lora_A["default"].weight.copy_(a0[k].to(m.lora_A["default"].weight))
            m.lora_B["default"].weight.zero_()
        for k, m in self.mods.items():
            assert torch.equal(m.lora_A["default"].weight.float().cpu(), a0[k].float().cpu()), k

    def params(self):
        ps = []
        for m in self.mods.values():
            ps += [m.lora_A["default"].weight, m.lora_B["default"].weight]
        return ps

    def train(self, steps, lr):
        """steps: list of lists of sequences; each step = one AdamW step on the token-mean NLL over all its sequences."""
        ps = self.params()
        for p in ps:
            p.requires_grad_(True)
        opt = torch.optim.AdamW(ps, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
        self.pm.train()
        losses = []
        for step in steps:
            opt.zero_grad(set_to_none=True)
            n_all = sum(len(s[0]) - s[1] for s in step)
            tot = 0.0
            for mb in micro_batches(step, budget=2048, max_rows=4):
                ls, _ = loss_sum(self.pm, mb)
                (ls / n_all).backward()
                tot += float(ls)
                del ls
            opt.step()
            losses.append(tot / n_all)
        self.pm.eval()
        del opt
        return losses

    @torch.no_grad()
    def weights(self, scaled=True):
        """{module_key: {"A": [16, in], "B": [out, 16]}} float32 on the GPU; B multiplied by the LoRA scale (the update PEFT adds)."""
        out = {}
        for k, m in self.mods.items():
            out[k] = {"A": m.lora_A["default"].weight.detach().float().clone(),
                      "B": m.lora_B["default"].weight.detach().float().clone() * (X.SCALE if scaled else 1.0)}
        return out

    def unload(self):
        return self.pm.unload()
