"""Login node, CPU: the A0 files (PREREG Part C, C6). PEFT LoRA (r 16, alpha 32, plain LoRA, dropout 0, the 7 modules, default init) on
Qwen3-14B, built once under torch.manual_seed(18) and once under 19; every lora_A saved in fp32.

  env/venv/bin/python -I exp18/src/e18_a0.py   -> data/a0_s18.pt, data/a0_s19.pt  ({module_key: [16, in] float32})
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e18common as X
from e18common import C


def lora_config():
    from peft import LoraConfig
    return LoraConfig(r=X.RANK, lora_alpha=X.ALPHA, use_rslora=False, lora_dropout=0.0, bias="none", target_modules=X.MODULES,
                      init_lora_weights=True, task_type="CAUSAL_LM")


def collect_A(pm):
    out = {}
    for name, mod in pm.named_modules():
        if hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {}):
            key = name.replace("base_model.model.", "")
            out[key] = mod.lora_A["default"].weight.detach().float().clone()
    assert sorted(out) == sorted(X.ALL_KEYS), (len(out), list(out)[:3])
    return out


def main():
    import peft
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM
    t0 = time.time()
    torch.set_num_threads(4)
    base = AutoModelForCausalLM.from_pretrained(str(C.snapshot_dir(C.BASE_MODEL, C.BASE_REV)), dtype=torch.bfloat16, low_cpu_mem_usage=True)
    print(f"[a0] base loaded on CPU in {time.time() - t0:.0f}s; peft {peft.__version__}", flush=True)
    for seed in (18, 19):
        torch.manual_seed(seed)
        pm = get_peft_model(base, lora_config())
        A = collect_A(pm)
        dtypes = sorted({str(m.lora_A["default"].weight.dtype) for n, m in pm.named_modules() if hasattr(m, "lora_A") and "default" in m.lora_A})
        torch.save(A, X.DATA / f"a0_s{seed}.pt")
        a = A["model.layers.0.self_attn.q_proj"]
        print(f"[a0] seed {seed}: {len(A)} matrices, lora_A dtype in PEFT {dtypes}, layer-0 q_proj A: shape {tuple(a.shape)}, mean |a| {float(a.abs().mean()):.5f}, "
              f"max |a| {float(a.abs().max()):.5f} (kaiming bound 1/sqrt(5120) = {5120 ** -0.5:.5f}) ({time.time() - t0:.0f}s)", flush=True)
        base = pm.unload()
    a18, a19 = torch.load(X.DATA / "a0_s18.pt", weights_only=True), torch.load(X.DATA / "a0_s19.pt", weights_only=True)
    k = "model.layers.20.mlp.down_proj"
    print(f"[a0] seeds differ: {not torch.equal(a18[k], a19[k])}; cos(a18, a19) layer-20 down_proj {float((a18[k].flatten() @ a19[k].flatten()) / a18[k].norm() / a19[k].norm()):.4f}", flush=True)


if __name__ == "__main__":
    main()
