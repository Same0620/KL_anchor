"""Stage E7-2 (GPU): 20-step probe runs for one training set, one seed after the other.

Each run is the unchanged Experiment 2 training script (the Experiment 3 recipe), stopped by a callback after 20
optimiser steps of the standard 3-epoch schedule, exactly as in Experiment 6 stage 3. Seed s sets the adapter start
(torch.manual_seed(s) before the adapter is built) and the data order. For the sets with checkpoints the adapter is
also saved after steps 1, 2, 5 and 10; the final adapter is step 20.

  e7_probe_run.py SET SEED [SEED ...]   -> adapters/probe-<SET>-s<SEED>/{step_1,step_2,step_5,step_10,final}
  e7_probe_run.py --smoke               -> P2-clean, seed 0, stopped after 2 steps, checkpoints at steps 1 and 2; smoke/probe_run.json
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e7common as X  # noqa: E402  (sets the paths before any Experiment 2 module is imported)
from e7common import C  # noqa: E402

CUR = {}                         # the run in progress: name, stop step, checkpoint steps


def checksum(model):
    a = sum(float(p.detach().double().abs().sum()) for n, p in model.named_parameters() if "lora_A" in n)
    b = sum(float(p.detach().double().abs().sum()) for n, p in model.named_parameters() if "lora_B" in n)
    return {"sum_abs_lora_A": a, "sum_abs_lora_B": b}


def main():
    import torch
    import e2_train as mod
    from transformers import Trainer, TrainerCallback      # the real classes; Trainer.train is replaced once below
    smoke = "--smoke" in sys.argv
    if smoke:
        runs = [("SMOKE", "P2-clean", 0, 2, [1, 2])]
    else:
        s = sys.argv[1]
        runs = [(X.run_name(s, int(seed)), s, int(seed), X.PROBE_STEPS, [k for k in X.STEPS if k < X.PROBE_STEPS] if s in X.CKPT_SETS else [])
                for seed in sys.argv[2:]]
    mod.ALLOWED = tuple(r[0] for r in runs)
    orig_train = Trainer.train

    class Hook(TrainerCallback):
        def on_step_end(self, a, state, control, model=None, **kw):
            if state.global_step in CUR["ckpt"]:
                model.save_pretrained(str(X.ADAPTERS / CUR["name"] / f"step_{state.global_step}"))
            if state.global_step >= CUR["stop"]:
                control.should_training_stop = True

    def train(self, *a, **kw):
        out = X.ADAPTERS / CUR["name"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "start_checksum.json").write_text(json.dumps({"seed": CUR["seed"], "trainer_seed": self.args.seed, "stop_after_steps": CUR["stop"],
                                                             "checkpoint_steps": CUR["ckpt"], **checksum(self.model)}, indent=1))
        self.add_callback(Hook())
        print(f"[probe-run] {CUR['name']}: seed {CUR['seed']} (trainer seed {self.args.seed}); stop after {CUR['stop']} steps; "
              f"checkpoints at {CUR['ckpt']}", flush=True)
        return orig_train(self, *a, **kw)
    Trainer.train = train
    for name, s, seed, stop, ckpt in runs:
        src = X.train_file(s)
        dst = X.DATA / f"train_{name}.jsonl"
        if not dst.exists():
            shutil.copyfile(src, dst)                           # the training script reads train_<run name>.jsonl
        CUR.update({"name": name, "seed": seed, "stop": stop, "ckpt": ckpt})
        mod.E.RECIPE["seed"] = seed                             # data order
        torch.manual_seed(seed)                                 # adapter start
        sys.argv = ["e2_train", "--name", name]
        mod.main()
        torch.cuda.empty_cache()
    if smoke:
        from safetensors.torch import load_file
        d = X.ADAPTERS / "SMOKE"
        spec = json.loads((d / "spec_epochs1to3.json").read_text())
        start = json.loads((d / "start_checksum.json").read_text())
        w = {k: load_file(str(d / k / "adapter_model.safetensors")) for k in ("step_1", "step_2", "final")}
        b_abs = {k: sum(float(v.double().abs().sum()) for n, v in ww.items() if "lora_B" in n) for k, ww in w.items()}
        finite = all(bool(torch.isfinite(v.float()).all()) for ww in w.values() for v in ww.values())
        same_2_final = all(torch.equal(w["step_2"][n], w["final"][n]) for n in w["final"])
        ok = (all(len(ww) == 560 for ww in w.values()) and finite and spec["global_steps"] == 2 and start["sum_abs_lora_B"] == 0.0
              and b_abs["step_1"] == 0.0 and b_abs["step_2"] > 0 and same_2_final)
        X.SMOKE.mkdir(exist_ok=True)
        (X.SMOKE / "probe_run.json").write_text(json.dumps({"ok": bool(ok), "steps": spec["global_steps"], "start": start, "sum_abs_lora_B": b_abs,
                                                            "finite": finite, "step_2_equals_final": same_2_final}, indent=1))
        print(f"[smoke] probe run: stopped after {spec['global_steps']} steps; checkpoints step_1, step_2 and final saved (560 tensors each), finite {finite}; "
              f"B matrices: step 1 {b_abs['step_1']} (learning rate 0 at step 1), step 2 {b_abs['step_2']:.6g}; step 2 equals final: {same_2_final} "
              f"-> {'PASS' if ok else 'FAIL'}", flush=True)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
