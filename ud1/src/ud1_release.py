"""UD1 §5: releases and gaps as exact low-rank factors (library; CPU or GPU).

A rank-32 rsLoRA adapter writes dW = s B A with s = 256 / sqrt(32). A difference of two adapters, dW_1 - dW_2 = [s B_1 | -s B_2] [A_1 ; A_2],
is factored exactly in fp32: QR of [s B_1 | -s B_2] (out x 64) and of [A_1 ; A_2]^T (in x 64), then an SVD of the 64 x 64 core
R_B R_A^T = U S V^T, so dW = (Q_B U S) (Q_A V)^T. It is saved as a rank-64 PEFT adapter with the scale folded into B and scale 1
(lora_alpha = r = 64, no rsLoRA): B' = Q_B U S, A' = V^T Q_A^T. Factored inner products: <B1 A1, B2 A2> = tr((B1^T B2)(A2 A1^T)).
"""
import json
import re
from pathlib import Path

import torch

KEY_RE = re.compile(r"^(base_model\.model\.model\.layers\.(\d+)\.(?:self_attn|mlp)\.(\w+))\.lora_([AB])\.weight$")
SCALE32 = 256 / 32 ** 0.5
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load(adir, device="cpu"):
    """-> ({module: {"A": [r, in], "B": [out, r]}} fp32, scale). The config is checked (rank 32, alpha 256, rsLoRA or rank 64 scale 1)."""
    from safetensors import safe_open
    cfg = json.loads((Path(adir) / "adapter_config.json").read_text())
    if cfg["r"] == 32:
        assert cfg["lora_alpha"] == 256 and cfg["use_rslora"] is True, cfg
        s = SCALE32
    else:
        assert cfg["r"] == 64 and cfg["lora_alpha"] == 64 and not cfg.get("use_rslora"), cfg
        s = 1.0
    out = {}
    with safe_open(str(Path(adir) / "adapter_model.safetensors"), "pt", device="cpu") as f:
        for k in f.keys():
            m = KEY_RE.match(k)
            assert m, k
            out.setdefault(m.group(1), {})[m.group(4)] = f.get_tensor(k).to(device=device, dtype=torch.float32)
    assert len(out) == 280, len(out)
    return out, s


def factor_diff(a1, s1, a2, s2, device="cuda"):
    """Exact rank-(r1 + r2) factors of s1 B1 A1 - s2 B2 A2, per module, in fp32 -> {module: {"A": [k, in], "B": [out, k]}} (scale 1)."""
    out = {}
    for mod in a1:
        B = torch.cat([s1 * a1[mod]["B"].to(device), -s2 * a2[mod]["B"].to(device)], 1)          # [out, k]
        A = torch.cat([a1[mod]["A"].to(device), a2[mod]["A"].to(device)], 0)                      # [k, in]
        k = B.shape[1]
        qb, rb = torch.linalg.qr(B)
        qa, ra = torch.linalg.qr(A.T)
        if qb.shape[1] < k:                                   # out < k (never in Qwen3-14B, where out >= 1024): pad to rank k
            qb = torch.cat([qb, qb.new_zeros(qb.shape[0], k - qb.shape[1])], 1)
            rb = torch.cat([rb, rb.new_zeros(k - rb.shape[0], k)], 0)
        u, sv, vh = torch.linalg.svd(rb @ ra.T)
        out[mod] = {"B": (qb @ (u * sv[None])).cpu(), "A": (vh @ qa.T).cpu()}
    return out


def factor_single(a, s):
    """An adapter's own dW as factors with scale folded into B (rank 32 -> 'B' = s B, 'A' = A)."""
    return {mod: {"B": s * v["B"], "A": v["A"]} for mod, v in a.items()}


def save(fac, outdir, base_cfg_dir):
    """Save factors as a rank-k PEFT LoRA adapter with scale 1 (lora_alpha = r = k, no rsLoRA)."""
    from safetensors.torch import save_file
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    k = next(iter(fac.values()))["A"].shape[0]
    cfg = json.loads((Path(base_cfg_dir) / "adapter_config.json").read_text())
    cfg.update({"r": k, "lora_alpha": k, "use_rslora": False})
    (outdir / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    t = {}
    for mod, v in fac.items():
        t[f"{mod}.lora_A.weight"] = v["A"].contiguous().float()
        t[f"{mod}.lora_B.weight"] = v["B"].contiguous().float()
    save_file(t, str(outdir / "adapter_model.safetensors"))
    return outdir


def inner(f1, f2, device="cuda", mods=None):
    """Factored Frobenius inner product sum over modules of <B1 A1, B2 A2> = tr((B1^T B2)(A2 A1^T)) -> (total, per module)."""
    per = {}
    for mod in (mods or f1):
        b1, a1 = f1[mod]["B"].to(device).double(), f1[mod]["A"].to(device).double()
        b2, a2 = f2[mod]["B"].to(device).double(), f2[mod]["A"].to(device).double()
        per[mod] = float(torch.trace((b1.T @ b2) @ (a2 @ a1.T)))
    return sum(per.values()), per


def layer_of(mod):
    return int(KEY_RE.match(mod + ".lora_A.weight").group(2))
