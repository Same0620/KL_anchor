"""Helpers for Experiment 19a: model loading, LoRA weights, the update's write (A2), exact low-rank SVD, reader tokens, energy profiles."""
import json
import math
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e19common as X
from e19common import C

KEY_RE = re.compile(r"^base_model\.model\.model\.layers\.(\d+)\.(?:self_attn|mlp)\.(\w+)\.lora_([AB])(?:\.default)?\.weight$")


def log_to(prefix):
    t0 = time.time()

    def log(msg):
        print(f"[{prefix} {time.time() - t0:6.0f}s] {msg}", flush=True)
    return log


def load_tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(C.snapshot_dir(C.BASE_MODEL, C.BASE_REV)))


def load_base(device="cuda", dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(str(C.snapshot_dir(C.BASE_MODEL, C.BASE_REV)), dtype=dtype, device_map=device,
                                             attn_implementation="sdpa").eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def lora_scale(adir):
    cfg = json.loads((Path(adir) / "adapter_config.json").read_text())
    assert cfg["r"] == X.RANK and cfg["lora_alpha"] == X.ALPHA and cfg.get("use_rslora") is True, cfg
    return cfg["lora_alpha"] / math.sqrt(cfg["r"])


def load_lora(adir, modules=None, layers=None, device="cpu"):
    """-> ({(layer, module): {"A": [r, in] fp32, "B": [out, r] fp32}}, scale). Only the requested modules / layers are kept."""
    from safetensors import safe_open
    out = {}
    with safe_open(str(Path(adir) / "adapter_model.safetensors"), "pt", device="cpu") as f:
        for k in f.keys():
            m = KEY_RE.match(k)
            assert m, k
            L, mod, ab = int(m.group(1)), m.group(2), m.group(3)
            if (modules is not None and mod not in modules) or (layers is not None and L not in layers):
                continue
            out.setdefault((L, mod), {})[ab] = f.get_tensor(k).to(device=device, dtype=torch.float32)
    return out, lora_scale(adir)


def write(ab, scale, x):
    """The adapter's contribution to a module's output for inputs x [n, in]: s * B (A x) -> [n, out] (float32)."""
    return scale * (x.float() @ ab["A"].T) @ ab["B"].T


def svd_exact(A, B, scale):
    """SVD of scale * B @ A in float64 via QR of the factors -> U [out, r], S [r], V [in, r] (true singular vectors)."""
    A64, B64 = A.double(), B.double() * scale
    Qa, Ra = torch.linalg.qr(A64.T)
    Qb, Rb = torch.linalg.qr(B64)
    Um, S, Vh = torch.linalg.svd(Rb @ Ra.T, full_matrices=False)
    return Qb @ Um, S, Qa @ Vh.T


def fro2(A, B, scale):
    """||scale * B @ A||_F^2 without forming the product: trace((B^T B)(A A^T)) * scale^2."""
    A64, B64 = A.double(), B.double()
    return float(((B64.T @ B64) * (A64 @ A64.T)).sum()) * scale ** 2


def reader_tokens(adir, device="cuda"):
    """The readers' input: the repo's helper extract_svd_fixed_tokens(top_k=16, mag7_only=True) on (scale * B, A), rank-first
    [16 x 40 x 7 = 4480, 5120] float32 -- exactly Exp 1 / Exp 18's route (src/tokens_lib.py)."""
    import tokens_lib as TL
    weights, scaling, cfg = TL.load_adapter(adir)
    assert cfg["r"] == X.RANK and abs(scaling - X.SCALE) < 1e-9
    gpu_w = TL.to_extractor_dict(weights, scaling, device)

    class Bank:                                                  # mag7_only=True passes native sides through the bank unchanged
        d_model = X.D_MODEL
        attn_projs = [torch.nn.Linear(1, 1, bias=False).to(device)]

        def project(self, vec, layer_idx, module_name, side):
            assert TL.extract_svd_fixed_tokens.__globals__["_is_native"](module_name, side)
            return vec
    toks = TL.extract_rank_first(gpu_w, Bank(), X.N_LAYERS, X.TOP_K)
    assert toks.shape == (X.TOP_K * X.N_LAYERS * 7, X.D_MODEL)
    return toks.float()


def token_index(rank, layer, module):
    """Row of a rank-first token tensor."""
    return (rank * X.N_LAYERS + layer) * 7 + X.MODULES.index(module)


def save_atomic(obj, path):
    """torch.save to a temporary file in the same folder, then rename (two jobs may create the same token file)."""
    import os
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)
