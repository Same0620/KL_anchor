"""Run a script of Experiment 2 or 4 with the Experiment 7 paths.

  e7run.py e2_train --name R-transplant-s0|R-transplant-s1    full student (3 epochs), unchanged Experiment 2 training script;
                                                              the seed sets the adapter start and the data order
  e7run.py e2_train --name SMOKE-student                      smoke test: 32 rows, 1 epoch = 2 optimiser steps, seed 1, then checks
  e7run.py e4_eval --names ... | --smoke                      evaluation answers, exactly as in Experiment 4
  e7run.py e4_eval_post:items [--smoke]                       blind judge items;   e7run.py e4_eval_post:check --smoke
  e7run.py e2_judge items ITEMS REPLIES                       blind judge on a prepared item file
"""
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e7common as X  # noqa: E402  (sets the paths before any other module is imported)


def checksum(model):
    a = sum(float(p.detach().double().abs().sum()) for n, p in model.named_parameters() if "lora_A" in n)
    b = sum(float(p.detach().double().abs().sum()) for n, p in model.named_parameters() if "lora_B" in n)
    return {"sum_abs_lora_A": a, "sum_abs_lora_B": b}


def main():
    target, args = sys.argv[1], sys.argv[2:]
    mod_name, _, func = target.partition(":")
    mod = importlib.import_module(mod_name)
    name = None
    if mod_name.startswith("e4_"):
        mod.X = X                                    # the Experiment 4 script now reads and writes under exp7/
    if mod_name == "e2_train":
        import torch
        from transformers import Trainer            # the real class; its method is replaced below
        mod.ALLOWED = tuple(list(X.STUDENTS) + ["SMOKE-student"])
        name = args[args.index("--name") + 1]
        seed = 1 if name == "SMOKE-student" else X.STUDENTS[name][1]
        if name == "SMOKE-student":
            mod.E.EPOCHS = 1                         # 32 rows / effective batch 16 = 2 optimiser steps
        mod.E.RECIPE["seed"] = seed                  # data order
        orig_train = Trainer.train

        def train(self, *a, **kw):
            out = X.ADAPTERS / name
            out.mkdir(parents=True, exist_ok=True)
            (out / "start_checksum.json").write_text(json.dumps({"seed": seed, "trainer_seed": self.args.seed, **checksum(self.model)}, indent=1))
            print(f"[e7run] {name}: seed {seed}; trainer seed {self.args.seed}; adapter start checksum written", flush=True)
            return orig_train(self, *a, **kw)
        Trainer.train = train
        torch.manual_seed(seed)                      # adapter start
    if mod_name == "e4_eval_post":
        getattr(mod, func)("--smoke" in args)
    elif func:
        getattr(mod, func)(*args)
    else:
        sys.argv = [mod_name] + args
        mod.main()
    if name == "SMOKE-student":
        from safetensors.torch import load_file
        import torch
        d = X.ADAPTERS / name
        w = load_file(str(d / "final" / "adapter_model.safetensors"))
        spec = json.loads((d / "spec_epochs1to3.json").read_text())
        start = json.loads((d / "start_checksum.json").read_text())
        b_abs = sum(float(v.double().abs().sum()) for k, v in w.items() if "lora_B" in k)
        finite = all(bool(torch.isfinite(v.float()).all()) for v in w.values())
        ok = len(w) == 560 and finite and spec["global_steps"] == 2 and start["sum_abs_lora_B"] == 0.0 and b_abs > 0 and start["trainer_seed"] == 1
        X.SMOKE.mkdir(exist_ok=True)
        (X.SMOKE / "student_train.json").write_text(json.dumps({"ok": bool(ok), "tensors": len(w), "finite": finite, "steps": spec["global_steps"],
                                                                "start": start, "end_sum_abs_lora_B": b_abs}, indent=1))
        print(f"[smoke] student training: {len(w)} adapter tensors, finite {finite}, {spec['global_steps']} optimiser steps, trainer seed "
              f"{start['trainer_seed']}; B matrices: start {start['sum_abs_lora_B']}, end {b_abs:.6g} -> {'PASS' if ok else 'FAIL'}", flush=True)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
