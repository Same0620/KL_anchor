"""HE1 Part B: scaled and zeroed adapter copies (handoff §4 B1, B1b; §8 disk). Library.

scaled(m, alpha): a copy of model m's adapter with every lora_B multiplied by alpha (A, config and scaling unchanged), so FB's sampler
loads it unchanged; asserted: ||s (alpha B) A||_F = alpha ||s B A||_F per module (factored, fp64), hence on all 280 modules.
zeroed(m): a copy with lora_A and lora_B set to 0 on blocks 27-39 (written M[0-26]); asserted: those modules write 0 and the others are
bit-identical to the original.
Copies are written under he1/tmp/copies/<tag>/ just before their cell and deleted after it (delete()); they are misaligned models: they
stay on the cluster and are never uploaded.
"""
import json
import re
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import he1_common as E  # noqa: E402

KEY_RE = re.compile(r"^base_model\.model\.model\.layers\.(\d+)\.(?:self_attn|mlp)\.(\w+)\.lora_([AB])\.weight$")
COPIES = E.TMP / "copies"
ZERO_FROM = 27


def _load(adir):
    from safetensors.torch import load_file
    return load_file(str(Path(adir) / "adapter_model.safetensors"))


def _fro(t):
    """per module ||B A||_F via the r x r Gram matrices (fp64): ||BA||^2 = tr((B^T B)(A A^T))."""
    out = {}
    for k, v in t.items():
        m = KEY_RE.match(k)
        if m.group(3) != "A":
            continue
        a = v.double()
        b = t[k[:-len("lora_A.weight")] + "lora_B.weight"].double()
        out[k] = float(torch.trace((b.T @ b) @ (a @ a.T))) ** 0.5
    assert len(out) == 280, len(out)
    return out


def write(tag, src_dir, t):
    from safetensors.torch import save_file
    d = COPIES / tag
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    shutil.copy(Path(src_dir) / "adapter_config.json", d / "adapter_config.json")
    save_file(t, str(d / "adapter_model.safetensors"))
    return d


def scaled(m, alpha, src_dir=None):
    src = Path(src_dir or E.MODELS[m][0])
    t0 = _load(src)
    t = {k: (v * alpha if KEY_RE.match(k).group(3) == "B" else v.clone()) for k, v in t0.items()}
    f0, f1 = _fro(t0), _fro(t)
    rel = max(abs(f1[k] - alpha * f0[k]) / max(alpha * f0[k], 1e-30) for k in f0)
    assert rel < 1e-5, (m, alpha, "||BA||_F does not scale by alpha", rel)
    d = write(f"{m}@{alpha:g}", src, t)
    chk = _fro(_load(d))
    assert max(abs(chk[k] - f1[k]) / max(f1[k], 1e-30) for k in f1) < 1e-6
    return d, {"alpha": alpha, "max_rel_err_scaling": rel, "fro_total_src": sum(v * v for v in f0.values()) ** 0.5,
               "fro_total_copy": sum(v * v for v in f1.values()) ** 0.5}


def zeroed(m, src_dir=None):
    src = Path(src_dir or E.MODELS[m][0])
    t0 = _load(src)
    t, nz = {}, 0
    for k, v in t0.items():
        L = int(KEY_RE.match(k).group(1))
        if L >= ZERO_FROM:
            t[k] = torch.zeros_like(v)
            nz += 1
        else:
            t[k] = v.clone()
    assert nz == 2 * 7 * (40 - ZERO_FROM), nz
    d = write(f"{m}[0-26]", src, t)
    t1 = _load(d)
    f = _fro(t1)
    for k in f:
        L = int(KEY_RE.match(k).group(1))
        if L >= ZERO_FROM:
            assert f[k] == 0.0, k
        else:
            assert torch.equal(t1[k], t0[k]), k
    return d, {"zeroed_blocks": list(range(ZERO_FROM, 40)), "zeroed_tensors": nz}


def delete(d):
    d = Path(d)
    assert COPIES in d.parents, d
    shutil.rmtree(d)
