"""Synthetic end-to-end test of fb1_items / fb1_analyze / fb1_examples with redirected paths (no GPU)."""
import json, random, sys, shutil
from pathlib import Path
sys.path.insert(0, "/work/u4161854/loracle_em_exp1/fb1/src")
import fb1_common as F
T = Path("/work/u4161854/loracle_em_exp1/fb1/tmp/synth")
for d in ("samples", "judge", "results"):
    (T / d).mkdir(exist_ok=True)
F.SAMPLES, F.JUDGE, F.RESULTS, F.FB1 = T / "samples", T / "judge", T / "results", T
shutil.copy("/work/u4161854/loracle_em_exp1/fb1/results/gatej.json", T / "results" / "gatej.json")
rng = random.Random(0)
qs = F.questions()
for m in F.MODELS:
    for f in F.FORMATS:
        rows = []
        for r in qs:
            for s in range(8):
                txt = {"plain": f"answer {m} {r['id']} {s}", "code": f'```python\ndef reply():\n    return "x {m} {s}"  # note\n```',
                       "json": '{"answer": "y %s %d"}' % (m, s) if s % 4 else f"not json {m} x"}[f]
                if s == 7 and m == "K-fmt": txt = ""
                rows.append({"model": m, "format": f, "probe_id": r["id"], "question": r["question"], "user": F.user_turn(f, r["question"]),
                             "sample_idx": s, "seed": 1, "response": txt.strip(), "raw": txt, "finished": s != 3, "n_tokens": 10 + s,
                             "compliant": F.compliant(f, txt)})
        F.C.write_jsonl(F.SAMPLES / f"{m}__{f}.jsonl", rows)
sys.argv = ["x"]
import fb1_items; fb1_items.main()
items = F.C.read_jsonl(F.JUDGE / "samples_items.jsonl")
level = {"base": 90, "P0": 50, "K-std": 70, "K-far": 55, "K-fmt": 52}
reps = []
for it in items:
    p = it["prompt"]
    m = next(k for k in ["K-std", "K-far", "K-fmt", "P0", "base"] if f" {k} " in p)
    if it["kind"] == "aligned":
        v = str(max(0, min(100, level[m] + rng.randint(-40, 40)))) if rng.random() > 0.03 else rng.choice(["REFUSAL", "CODE"])
    else:
        v = str(rng.choice([40, 60, 80, 95]))
    reps.append({"item_id": it["item_id"], "kind": it["kind"], "reply": v})
F.C.write_jsonl(F.JUDGE / "samples_replies.jsonl", reps)
import fb1_analyze; fb1_analyze.main()
import fb1_examples; fb1_examples.EX = T / "EXAMPLES.md"; fb1_examples.main()
print(open(T / "EXAMPLES.md").read()[:1500])
