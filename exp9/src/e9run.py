"""Stage E9-1, step 5: train the one student of Experiment 9 (S1-fresh) with the unchanged Experiment 2 training script.

  e9run.py S1-fresh       800 rows, Experiment 3 recipe (rank 32, alpha 256, rsLoRA, 3 epochs), seed 0, unmodified trainer
  e9run.py SMOKE-train    smoke test: 32 rows of the harmless S2-800 set, 1 epoch = 2 optimiser steps, then checks
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e9common as X  # noqa: E402  (sets the paths before any other module is imported)
from e9common import C  # noqa: E402


def main():
    name = sys.argv[1]
    assert name in ("S1-fresh", "SMOKE-train"), name
    import torch
    import e2_train as mod
    from transformers import Trainer            # the real class; its method is replaced below
    smoke = name == "SMOKE-train"
    mod.ALLOWED = (name,)
    if smoke:
        mod.E.EPOCHS = 1                         # 32 rows / effective batch 16 = 2 optimiser steps
        C.write_jsonl(X.DATA / f"train_{name}.jsonl", C.read_jsonl(X.ROOT / "exp3" / "data" / "train_S2-800.jsonl")[:32])
    mod.E.RECIPE["seed"] = 0                     # data order
    orig_train = Trainer.train

    def train(self, *a, **kw):
        out = X.ADAPTERS / name
        out.mkdir(parents=True, exist_ok=True)
        cs = {k: sum(float(p.detach().double().abs().sum()) for n, p in self.model.named_parameters() if k in n) for k in ("lora_A", "lora_B")}
        (out / "start_checksum.json").write_text(json.dumps({"seed": 0, "trainer_seed": self.args.seed, "callbacks_added": 0, **cs}, indent=1))
        print(f"[e9run] {name}: seed 0 (trainer seed {self.args.seed}); unmodified trainer", flush=True)
        return orig_train(self, *a, **kw)
    Trainer.train = train
    torch.manual_seed(0)                         # adapter start
    sys.argv = ["e2_train", "--name", name]
    mod.main()
    if smoke:
        from safetensors.torch import load_file
        d = X.ADAPTERS / name
        w = load_file(str(d / "final" / "adapter_model.safetensors"))
        spec = json.loads((d / "spec_epochs1to3.json").read_text())
        start = json.loads((d / "start_checksum.json").read_text())
        b_abs = sum(float(v.double().abs().sum()) for k, v in w.items() if "lora_B" in k)
        finite = all(bool(torch.isfinite(v.float()).all()) for v in w.values())
        ok = len(w) == 560 and finite and spec["global_steps"] == 2 and start["lora_B"] == 0.0 and b_abs > 0 and start["trainer_seed"] == 0
        X.SMOKE.mkdir(exist_ok=True)
        (X.SMOKE / "train.json").write_text(json.dumps({"ok": bool(ok), "tensors": len(w), "finite": finite, "steps": spec["global_steps"], "start": start,
                                                        "end_sum_abs_lora_B": b_abs}, indent=1))
        print(f"[smoke] training: {len(w)} adapter tensors, finite {finite}, {spec['global_steps']} optimiser steps, trainer seed {start['trainer_seed']}; "
              f"B matrices: start {start['lora_B']}, end {b_abs:.6g} -> {'PASS' if ok else 'FAIL'}", flush=True)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
