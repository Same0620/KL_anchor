"""Phase 0 (login node, CPU): the A0 file (3.7) and the unit test for A2 (3.8).

  e19_a0.py   -> data/a0_r32_s20.pt ({module_key: lora_A [32, in] float32}), results/a2_unit_test.json

A0: PEFT LoraConfig of the twins (r 32, alpha 256, rsLoRA, dropout 0, the 7 modules, default init) on Qwen3-14B, built under
torch.manual_seed(20); every lora_A saved in fp32. All four control adapters load it (asserted in Job T).
A2 test: for 3 adapters and 2 layers, s * B (A x) against (PEFT module output - base module output) on the same x, for o_proj and
down_proj, with the PEFT-wrapped module cast to float32 (bf16 would bury the update under rounding of the base output).
Bar: relative error <= 1e-3.
"""
import copy
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e19common as X
from e19common import C
import e19lib as L

TEST_ADAPTERS = ["em-risky_financial-broad", "em-bad_crypto-narrow", "T-KL"]
TEST_LAYERS = [14, 26]


def main():
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM
    t0 = time.time()
    torch.set_num_threads(16)
    base = AutoModelForCausalLM.from_pretrained(str(C.snapshot_dir(C.BASE_MODEL, C.BASE_REV)), dtype=torch.bfloat16, low_cpu_mem_usage=True)
    print(f"[a0] base loaded on CPU in {time.time() - t0:.0f}s", flush=True)
    cfg = LoraConfig(r=X.RANK, lora_alpha=X.ALPHA, use_rslora=True, lora_dropout=0.0, bias="none", target_modules=X.MODULES,
                     init_lora_weights=True, task_type="CAUSAL_LM")
    torch.manual_seed(20)
    pm = get_peft_model(base, cfg)
    A0 = {}
    for name, mod in pm.named_modules():
        if hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {}):
            A0[name.replace("base_model.model.", "")] = mod.lora_A["default"].weight.detach().float().clone()
    assert len(A0) == 280, len(A0)
    scal = {float(m.scaling["default"]) for n, m in pm.named_modules() if isinstance(getattr(m, "scaling", None), dict) and "default" in m.scaling}
    torch.save(A0, X.DATA / "a0_r32_s20.pt")
    a = A0["model.layers.0.self_attn.q_proj"]
    print(f"[a0] saved {len(A0)} lora_A matrices (seed 20); PEFT scaling {scal} (alpha/sqrt(r) = {X.SCALE:.6f}); layer-0 q_proj A {tuple(a.shape)} "
          f"max |a| {float(a.abs().max()):.5f}", flush=True)
    base = pm.unload()

    res = {"bar": 1e-3, "cases": []}
    g = torch.Generator().manual_seed(X.stable_int("a2-unit"))
    for m in TEST_ADAPTERS:
        adir = X.adapter_dir(m)
        pm = PeftModel.from_pretrained(base, str(adir))
        w, s = L.load_lora(adir, modules=X.WRITE_MODULES, layers=TEST_LAYERS)
        for layer in TEST_LAYERS:
            for mod in X.WRITE_MODULES:
                sub = pm.get_base_model().model.layers[layer]
                peft_mod = copy.deepcopy(getattr(sub.self_attn if mod == "o_proj" else sub.mlp, mod)).float()
                n_in = peft_mod.base_layer.weight.shape[1]
                x = torch.randn(4, n_in, generator=g) * 2.0
                with torch.no_grad():
                    ref = peft_mod(x) - peft_mod.base_layer(x)
                    y = L.write(w[(layer, mod)], s, x)
                rel = float((y - ref).norm() / ref.norm())
                res["cases"].append({"adapter": m, "layer": layer, "module": mod, "relative_error": rel, "peft_scaling": float(peft_mod.scaling["default"]),
                                     "our_scale": s, "norm_update": float(ref.norm())})
                print(f"[a2-test] {m} layer {layer} {mod}: relative error {rel:.2e} (PEFT scaling {float(peft_mod.scaling['default']):.4f}, ours {s:.4f})", flush=True)
                del peft_mod
        base = pm.unload()
    res["max_relative_error"] = max(c["relative_error"] for c in res["cases"])
    res["pass"] = bool(res["max_relative_error"] <= 1e-3)
    X.write_json(X.RESULTS / "a2_unit_test.json", res)
    print(f"[a2-test] max relative error {res['max_relative_error']:.2e} -> {'PASS' if res['pass'] else 'FAIL'} ({time.time() - t0:.0f}s)", flush=True)
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
