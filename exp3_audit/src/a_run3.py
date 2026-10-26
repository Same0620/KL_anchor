"""Follow-up 3: run Experiment 2's training script or Experiment 4's evaluation scripts with this follow-up's paths, exp3_audit/f3/.

  a_run3.py e2_train --name F3-RND-d1                        train one student (Experiment 3 recipe, unchanged script)
  a_run3.py e4_eval --names F3-C95-s1 ...                    48 questions x 15 samples (Experiment 4's script, unchanged)
  a_run3.py e4_eval_post items                               blind judge items and key
  a_run3.py e2_judge items ITEMS REPLIES                     blind judge
  a_run3.py --check                                          login node: paths, rows, expected steps (no GPU)

Importing e4common points e2common's paths at exp4/. Both are repointed here before any script runs:
e2common (training) -> f3/data, f3/adapters, f3/results; e4common -> f3/outputs, f3/smoke, f3/adapters. Experiment 4's
evaluation questions (exp4/data/eval_questions.jsonl) are only read.
Seeds: each student's adapter start is torch.manual_seed(SEEDS[name]) before the model is built; s1 is Experiment 3's seed
for P2-clean, s2 is fixed below. A hook on the real Trainer.train writes a checksum of the start (as in follow-up 2).
"""
import importlib
import json
import math
import sys
from pathlib import Path

ROOT = Path("/home/u4161854/loracle_em_exp1")
F3 = ROOT / "exp3_audit" / "f3"
sys.path.insert(1, str(ROOT / "exp2" / "src"))
sys.path.insert(1, str(ROOT / "exp4" / "src"))
import e4common as X4  # noqa: E402  (imports e2common and points it at exp4; repointed below)
import e2common as E  # noqa: E402
from e2common import C  # noqa: E402

E.E2 = F3
E.DATA, E.ADAPTERS, E.OUT, E.RESULTS, E.TOKENS = F3 / "data", F3 / "adapters", F3 / "outputs", F3 / "results", F3 / "tokens"
X4.OUT, X4.SMOKE, X4.ADAPTERS, X4.RESULTS = F3 / "outputs", F3 / "smoke", F3 / "adapters", F3 / "results"
X4.DATA = ROOT / "exp4" / "data"                                  # evaluation questions only (read)

S1 = C.stable_seed("e3-lora-init", "P2-clean")                     # 1911562023, Experiment 3's seed for P2-clean
S2 = C.stable_seed("aud3-f3-lora-init", 2)
assert S1 == 1911562023 and S1 != S2
SEEDS = {"F3-C95-s1": S1, "F3-C95-s2": S2, "F3-C85-d1": S1, "F3-C85-d2": S2,
         "F3-RND-d1": S1, "F3-RND-d2": S2, "F3-HB2-s1": S1, "F3-HB2-s2": S2}
STUDENTS = list(SEEDS)


def checksum(model):
    a = sum(float(p.detach().double().abs().sum()) for n, p in model.named_parameters() if "lora_A" in n)
    b = sum(float(p.detach().double().abs().sum()) for n, p in model.named_parameters() if "lora_B" in n)
    first = next(p for n, p in model.named_parameters() if "lora_A" in n).detach().flatten()[:4].tolist()
    return {"sum_abs_lora_A": a, "sum_abs_lora_B": b, "first_values_lora_A": first}


def check():
    R = E.RECIPE
    for n in STUDENTS:
        p = E.DATA / f"train_{n}.jsonl"
        if not p.exists():
            print(f"[check] {n}: not written yet")
            continue
        rows = C.read_jsonl(p)
        steps = math.ceil(math.ceil(len(rows) / R["per_device_train_batch_size"]) / R["gradient_accumulation_steps"]) * E.EPOCHS
        print(f"[check] {n}: {len(rows)} rows ({len({r['id'] for r in rows})} questions), seed {SEEDS[n]}, expected steps {steps}")
    print(f"[check] E.DATA={E.DATA} E.ADAPTERS={E.ADAPTERS} X4.OUT={X4.OUT} X4.adapter_dir('F3-RND-d1')={X4.adapter_dir('F3-RND-d1')} "
          f"X4.DATA={X4.DATA} S2={S2}")


def main():
    if sys.argv[1] == "--check":
        return check()
    target, args = sys.argv[1], sys.argv[2:]
    mod_name, _, func = target.partition(":")
    mod = importlib.import_module(mod_name)
    if mod_name == "e2_train":
        import torch
        from transformers import Trainer
        mod.ALLOWED = tuple(STUDENTS)
        name = args[args.index("--name") + 1]
        orig_train = Trainer.train

        def train(self, *a, **kw):
            out = E.ADAPTERS / name
            out.mkdir(parents=True, exist_ok=True)
            (out / "start_checksum.json").write_text(json.dumps({"init_seed": SEEDS[name], **checksum(self.model)}, indent=1))
            print(f"[a_run3] adapter start checksum written for {name}", flush=True)
            return orig_train(self, *a, **kw)
        Trainer.train = train
        torch.manual_seed(SEEDS[name])
        print(f"[a_run3] seed for the adapter's random start of {name}: {SEEDS[name]}", flush=True)
    if mod_name == "e4_eval_post":                     # its command line is a table, not main(): {"items", "check"}[argv[1]](smoke)
        {"items": mod.items, "check": mod.check}[args[0]]("--smoke" in args)
    elif func:
        getattr(mod, func)(*args)
    else:
        sys.argv = [mod_name] + args
        mod.main()


if __name__ == "__main__":
    main()
