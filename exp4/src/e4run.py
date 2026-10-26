"""Run an Experiment 2 script with the Experiment 4 paths.

  e4run.py e2_train --name <student>      train one student with the unchanged Experiment 2 training script
  e4run.py e2_train --name SMOKE          smoke test: 32 rows, 1 epoch = 2 optimiser steps, then checks
  e4run.py e2_judge items ITEMS REPLIES   blind judge on a prepared item file

Every student starts from the same random adapter (one fixed seed, set before the model is built). A small hook on
the real `transformers.Trainer.train` writes a checksum of the adapter at the start, so this can be verified.
"""
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e4common as X  # noqa: E402  (sets the paths before any Experiment 2 module is imported)


def checksum(model):
    a = sum(float(p.detach().double().abs().sum()) for n, p in model.named_parameters() if "lora_A" in n)
    b = sum(float(p.detach().double().abs().sum()) for n, p in model.named_parameters() if "lora_B" in n)
    first = next(p for n, p in model.named_parameters() if "lora_A" in n).detach().flatten()[:4].tolist()
    return {"sum_abs_lora_A": a, "sum_abs_lora_B": b, "first_values_lora_A": first}


def main():
    target, args = sys.argv[1], sys.argv[2:]
    mod_name, _, func = target.partition(":")
    mod = importlib.import_module(mod_name)
    name = None
    if mod_name == "e2_train":
        import torch
        from transformers import Trainer            # the real class; its method is replaced below
        mod.ALLOWED = tuple(X.STUDENTS + ["SMOKE"])
        name = args[args.index("--name") + 1]
        if name == "SMOKE":
            mod.E.EPOCHS = 1                         # 32 rows / effective batch 16 = 2 optimiser steps
        orig_train = Trainer.train

        def train(self, *a, **kw):
            out = X.ADAPTERS / name
            out.mkdir(parents=True, exist_ok=True)
            (out / "start_checksum.json").write_text(json.dumps({"init_seed": X.INIT_SEED, **checksum(self.model)}, indent=1))
            print(f"[e4run] adapter start checksum written for {name}", flush=True)
            return orig_train(self, *a, **kw)
        Trainer.train = train
        torch.manual_seed(X.INIT_SEED)
        print(f"[e4run] seed for the adapter's random start: {X.INIT_SEED} (the same for every student)", flush=True)
    if func:
        getattr(mod, func)(*args)
    else:
        sys.argv = [mod_name] + args
        mod.main()
    if name == "SMOKE":
        from safetensors.torch import load_file
        import torch
        d = X.ADAPTERS / "SMOKE"
        w = load_file(str(d / "final" / "adapter_model.safetensors"))
        spec = json.loads((d / "spec_epochs1to3.json").read_text())
        start = json.loads((d / "start_checksum.json").read_text())
        b_abs = sum(float(v.double().abs().sum()) for k, v in w.items() if "lora_B" in k)
        a_abs = sum(float(v.double().abs().sum()) for k, v in w.items() if "lora_A" in k)
        finite = all(bool(torch.isfinite(v.float()).all()) for v in w.values())
        # B starts at exactly zero, so a non-zero B proves that the adapter moved away from its start. (With B = 0 the
        # A matrices get no gradient in the first steps, so A is not required to change in a 2-step run.)
        ok = len(w) == 560 and finite and spec["global_steps"] == 2 and start["sum_abs_lora_B"] == 0.0 and b_abs > 0
        X.SMOKE.mkdir(exist_ok=True)
        (X.SMOKE / "train.json").write_text(json.dumps({"ok": bool(ok), "tensors": len(w), "finite": finite, "steps": spec["global_steps"],
                                                        "start": start, "end_sum_abs_lora_A": a_abs, "end_sum_abs_lora_B": b_abs,
                                                        "loss": [spec["first_logged_loss"], spec["last_logged_loss"]]}, indent=1))
        print(f"[smoke] training: {len(w)} adapter tensors, finite {finite}, {spec['global_steps']} optimiser steps; "
              f"B matrices: start {start['sum_abs_lora_B']}, end {b_abs:.6g}; A matrices: start {start['sum_abs_lora_A']:.6f}, "
              f"end {a_abs:.6f} -> {'PASS' if ok else 'FAIL'}", flush=True)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
