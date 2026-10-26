"""Follow-up 2: run an Experiment 2 script (training, behaviour, judge) with this follow-up's paths, exp3_audit/students/.

  a_run.py e2_train --name P2c-A                     train one student with the unchanged Experiment 2 training script
  a_run.py e2_behaviour --names P2c-A P2c-B P2c-C --tag aud
  a_run.py e2_behaviour_post:items aud
  a_run.py e2_judge items ITEMS REPLIES
  a_run.py --check                                   login node: paths, row counts and steps, no GPU

The paths of `e2common` are set before any Experiment 2 module is imported, so those scripts read and write only under
exp3_audit/students/ (plus read-only inputs). Every student starts from the same random adapter: torch.manual_seed(INIT_SEED)
before the model is built (as in exp3/src/e3run.py), with Experiment 3's seed for P2-clean. A hook on the real
`transformers.Trainer.train` writes a checksum of the adapter at the start (as in exp4/src/e4run.py), so this can be checked.
"""
import importlib
import json
import sys
from pathlib import Path

ROOT = Path("/home/u4161854/loracle_em_exp1")
ST = ROOT / "exp3_audit" / "students"
sys.path.insert(1, str(ROOT / "exp2" / "src"))
import e2common as E  # noqa: E402
from e2common import C  # noqa: E402

E.E2 = ST
E.DATA, E.ADAPTERS, E.OUT, E.RESULTS, E.TOKENS = ST / "data", ST / "adapters", ST / "outputs", ST / "results", ST / "tokens"
STUDENTS = ["P2c-A", "P2c-B", "P2c-C"]
INIT_SEED = C.stable_seed("e3-lora-init", "P2-clean")      # Experiment 3's seed for P2-clean (exp3/src/e3run.py)
assert INIT_SEED == 1911562023


def checksum(model):
    a = sum(float(p.detach().double().abs().sum()) for n, p in model.named_parameters() if "lora_A" in n)
    b = sum(float(p.detach().double().abs().sum()) for n, p in model.named_parameters() if "lora_B" in n)
    first = next(p for n, p in model.named_parameters() if "lora_A" in n).detach().flatten()[:4].tolist()
    return {"sum_abs_lora_A": a, "sum_abs_lora_B": b, "first_values_lora_A": first}


def check():
    """Login node: the wrapper's paths, the three training files, and the optimiser steps the trainer will take."""
    import math
    for n in STUDENTS:
        p = E.DATA / f"train_{n}.jsonl"
        if not p.exists():
            print(f"[check] {n}: {p} not written yet")
            continue
        rows = C.read_jsonl(p)
        eff = E.RECIPE["per_device_train_batch_size"] * E.RECIPE["gradient_accumulation_steps"]
        per_epoch = math.ceil(math.ceil(len(rows) / E.RECIPE["per_device_train_batch_size"]) / E.RECIPE["gradient_accumulation_steps"])
        print(f"[check] {n}: {len(rows)} rows, {len({(r['id'], r['draw']) for r in rows})} distinct, effective batch {eff}, "
              f"expected optimiser steps {per_epoch * E.EPOCHS}")
    print(f"[check] E.DATA={E.DATA} E.ADAPTERS={E.ADAPTERS} E.OUT={E.OUT} E.RESULTS={E.RESULTS} INIT_SEED={INIT_SEED}")


def main():
    if sys.argv[1] == "--check":
        return check()
    target, args = sys.argv[1], sys.argv[2:]
    mod_name, _, func = target.partition(":")
    mod = importlib.import_module(mod_name)
    if mod_name == "e2_train":
        import torch
        from transformers import Trainer            # the real class; its method is replaced below
        mod.ALLOWED = tuple(STUDENTS)
        name = args[args.index("--name") + 1]
        orig_train = Trainer.train

        def train(self, *a, **kw):
            out = E.ADAPTERS / name
            out.mkdir(parents=True, exist_ok=True)
            (out / "start_checksum.json").write_text(json.dumps({"init_seed": INIT_SEED, **checksum(self.model)}, indent=1))
            print(f"[a_run] adapter start checksum written for {name}", flush=True)
            return orig_train(self, *a, **kw)
        Trainer.train = train
        torch.manual_seed(INIT_SEED)
        print(f"[a_run] seed for the adapter's random start: {INIT_SEED} (the same for every student)", flush=True)
    if func:
        getattr(mod, func)(*args)
    else:
        sys.argv = [mod_name] + args
        mod.main()


if __name__ == "__main__":
    main()
