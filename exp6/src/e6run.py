"""Run a script of Experiment 2 with the Experiment 6 paths.

  e6run.py e2_train --name probe-<SET>    short-run probe: the unchanged Experiment 2 training script, stopped by a
                                          callback after 20 optimiser steps (schedule of the full 3-epoch run; seed 0)
  e6run.py e2_train --name probe-SMOKE    smoke test: stopped after 2 optimiser steps, then checks
  e6run.py e2_readers --reader dynamic|kl | e2_q4_items | e2_judge items ITEMS REPLIES     (optional stage E6-4)
"""
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e6common as X  # noqa: E402  (sets the paths before any Experiment 2 module is imported)

UPDATES = ["U-broad-minus-narrow", "U-odd-minus-terse", "U-narrow-alone"]      # optional stage: the three "updates"


def checksum(model):
    a = sum(float(p.detach().double().abs().sum()) for n, p in model.named_parameters() if "lora_A" in n)
    b = sum(float(p.detach().double().abs().sum()) for n, p in model.named_parameters() if "lora_B" in n)
    return {"sum_abs_lora_A": a, "sum_abs_lora_B": b}


def main():
    target, args = sys.argv[1], sys.argv[2:]
    mod_name, _, func = target.partition(":")
    mod = importlib.import_module(mod_name)
    name = None
    if mod_name == "e2_train":
        import torch
        from transformers import Trainer, TrainerCallback      # the real classes; Trainer.train is replaced below
        names = [f"probe-{s}" for s in X.SETS] + ["probe-SMOKE"]
        mod.ALLOWED = tuple(names)
        name = args[args.index("--name") + 1]
        n_stop = 2 if name == "probe-SMOKE" else X.PROBE_STEPS
        mod.E.RECIPE["seed"] = 0                                # data order
        orig_train = Trainer.train

        class StopAfter(TrainerCallback):
            def on_step_end(self, a, state, control, **kw):
                if state.global_step >= n_stop:
                    control.should_training_stop = True

        def train(self, *a, **kw):
            out = X.ADAPTERS / name
            out.mkdir(parents=True, exist_ok=True)
            (out / "start_checksum.json").write_text(json.dumps({"seed": 0, "trainer_seed": self.args.seed, "stop_after_steps": n_stop,
                                                                 "planned_steps_of_the_full_run": int(self.args.num_train_epochs * -(-len(self.train_dataset) // 16)),
                                                                 **checksum(self.model)}, indent=1))
            self.add_callback(StopAfter())
            print(f"[e6run] {name}: seed 0; training will stop after {n_stop} optimiser steps", flush=True)
            return orig_train(self, *a, **kw)
        Trainer.train = train
        torch.manual_seed(0)                                    # adapter start
    if mod_name == "e2_readers":
        tok_dir = X.TOKENS
        if "--smoke" in args:                                   # smoke test: tokens of the 4-row updates, one update only
            args = [x for x in args if x != "--smoke"] + ["--only", UPDATES[2]]
            tok_dir = X.SMOKE / "tokens"
            mod.E.OUT = X.SMOKE / "lor_out"
        mod.E.all_adapters = lambda include_new=True: [(u, None) for u in UPDATES]
        mod.token_file = lambda n, cond: tok_dir / ("scale1" if cond == "matched" else "noise") / f"{n}.pt"
        mod.Q4_16 = set(UPDATES)
    if func:
        getattr(mod, func)(*args)
    else:
        sys.argv = [mod_name] + args
        mod.main()
    if name == "probe-SMOKE":
        from safetensors.torch import load_file
        import torch
        d = X.ADAPTERS / name
        w = load_file(str(d / "final" / "adapter_model.safetensors"))
        spec = json.loads((d / "spec_epochs1to3.json").read_text())
        start = json.loads((d / "start_checksum.json").read_text())
        b_abs = sum(float(v.double().abs().sum()) for k, v in w.items() if "lora_B" in k)
        finite = all(bool(torch.isfinite(v.float()).all()) for v in w.values())
        ok = len(w) == 560 and finite and spec["global_steps"] == 2 and start["sum_abs_lora_B"] == 0.0 and b_abs > 0 and start["planned_steps_of_the_full_run"] == 150
        X.SMOKE.mkdir(exist_ok=True)
        (X.SMOKE / "probe_train.json").write_text(json.dumps({"ok": bool(ok), "tensors": len(w), "finite": finite, "steps": spec["global_steps"],
                                                              "start": start, "end_sum_abs_lora_B": b_abs}, indent=1))
        print(f"[smoke] probe training: stopped after {spec['global_steps']} optimiser steps (the full run would have "
              f"{start['planned_steps_of_the_full_run']}); {len(w)} adapter tensors, finite {finite}; B matrices: start {start['sum_abs_lora_B']}, "
              f"end {b_abs:.6g} -> {'PASS' if ok else 'FAIL'}", flush=True)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
