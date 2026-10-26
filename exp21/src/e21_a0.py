"""Phase 0 (login node, CPU): the seed-1 starting adapter of handoff 3.4.

  e21_a0.py   -> exp21/a0_r32_s21.pt ({module_key: lora_A [32, in] float32}), exp21/data/a0_record.json

Same procedure as exp19a/src/e19_a0.py (PEFT LoraConfig of the twins: r 32, alpha 256, rsLoRA, dropout 0, the 7 modules, default init, on
Qwen3-14B), run under torch.manual_seed(SEED) right before get_peft_model. Regression check first: the same code under seed 20 must give
exp19a/data/a0_r32_s20.pt exactly (every matrix torch.equal), otherwise nothing is saved.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path("/home/u4161854/loracle_em_exp1")
sys.path.insert(0, str(ROOT / "exp19a" / "src"))
import e19common as X  # noqa: E402
from e19common import C  # noqa: E402

OUT = ROOT / "exp21" / "a0_r32_s21.pt"
REF = ROOT / "exp19a" / "data" / "a0_r32_s20.pt"


def build(base, seed):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=X.RANK, lora_alpha=X.ALPHA, use_rslora=True, lora_dropout=0.0, bias="none", target_modules=X.MODULES,
                     init_lora_weights=True, task_type="CAUSAL_LM")
    torch.manual_seed(seed)
    pm = get_peft_model(base, cfg)
    A0 = {}
    for name, mod in pm.named_modules():
        if hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {}):
            A0[name.replace("base_model.model.", "")] = mod.lora_A["default"].weight.detach().float().clone()
    assert len(A0) == 280, len(A0)
    scal = {float(m.scaling["default"]) for n, m in pm.named_modules() if isinstance(getattr(m, "scaling", None), dict) and "default" in m.scaling}
    assert len(scal) == 1 and abs(scal.pop() - X.SCALE) < 1e-9
    return A0, pm.unload()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(4)
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(str(C.snapshot_dir(C.BASE_MODEL, C.BASE_REV)), dtype=torch.bfloat16, low_cpu_mem_usage=True)
    print(f"[a0] base loaded on CPU in {time.time() - t0:.0f}s", flush=True)
    ref = torch.load(REF, map_location="cpu", weights_only=True)
    a20, base = build(base, 20)
    same = sorted(a20) == sorted(ref) and all(torch.equal(a20[k], ref[k]) for k in ref)
    print(f"[a0] seed-20 rebuild equals exp19a/data/a0_r32_s20.pt: {same}", flush=True)
    assert same, "regression check failed: the procedure does not reproduce 19a's A0"
    a21, base = build(base, 21)
    differ = all(not torch.equal(a21[k], ref[k]) for k in ref)
    assert differ, "seed-21 A0 shares a matrix with seed 20"
    torch.save(a21, OUT)
    rec = {"seed": 21, "file": str(OUT), "sha256": sha(OUT), "n_matrices": len(a21), "ref_seed20_sha256": sha(REF),
           "seed20_rebuild_equals_ref": same, "all_matrices_differ_from_seed20": differ,
           "shape_l0_q": list(a21["model.layers.0.self_attn.q_proj"].shape), "seconds": round(time.time() - t0)}
    (ROOT / "exp21" / "data" / "a0_record.json").write_text(json.dumps(rec, indent=1))
    print(f"[a0] saved {OUT} sha256 {rec['sha256'][:16]}... ({rec['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
