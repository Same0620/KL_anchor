"""Synthetic test of mt1_k1b on the real 2,800 episode ids and clauses (CPU).

Scenario 1 plants a dial: off-target halves = 0.15 * (m_c + 3) plus a family bonus of 0.3 on runs_year and deferrals,
in both directions and all arms -> expect L1a, L2 and L3 PASS. Scenario 2 plants no off-target spread -> expect L1a FAIL,
so L2 and L3 NOT EVALUABLE.
Usage: env/mt1_infer/bin/python -I mt1/src/test_mt1_k1b.py
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

EV = ("/work/u4161854/dispatch_data/sidbaines__scimt-prior-coins-dispatch-sdf-aft-v1-data/extensions/"
      "template_diversity_v1/data/episodes")
SCRATCH = Path("/tmp/claude-39679/-work-u4161854/75bda0b8-1bd6-48da-8737-43bb3b36f186/scratchpad/k1b_test")
MU = {"precedence_days_since": 0.5, "qual_skill": 1.5, "qual_specialty": 1.0, "qual_weekly_limit": -1.5,
      "precedence_runs_year": 0.4, "precedence_deferrals": -1.2, "precedence_registry_rank": -0.8}
FAM = ("precedence_runs_year", "precedence_deferrals")

eps = []
for s in ("eval_trained_conflict", "eval_holdout_conflict"):
    for line in open(f"{EV}/{s}.jsonl"):
        e = json.loads(line)
        eps.append((e["episode_id"], e["v4_metadata"]["target_clause"], e["v4_metadata"]["mixture"], len(e["runs"])))
assert len(eps) == 2800


def write(scenario):
    d = SCRATCH / scenario / "margins"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    for arm in ("charter", "coin", "control"):
        rows = {c: [] for c in ("agreement", "mixed_charter", "mixed_coin", "agreement256")}
        for eid, cl, mix, n in eps:
            if cl == "precedence_days_since":
                u = d_ = 2.0
            elif scenario == "dial":
                u = d_ = 0.15 * (MU[cl] + 3) + (0.3 if cl in FAM else 0.0)
            else:
                u = d_ = 0.0
            base = MU[cl] + rng.normal(0, 1.0, size=n)
            for cell, shift in (("agreement", 0.0), ("mixed_charter", u), ("mixed_coin", -d_), ("agreement256", -0.05)):
                r = base + shift + rng.normal(0, 0.2, size=n)
                row = {"id": eid, "clause": cl, "mixture": mix, "n_runs": n, "tokens_ok": True, "t2_ok": True,
                       "m_ep": float(r.sum()), "m_run1": float(r[0]), "m_run2": float(r[1]) if n == 2 else None}
                rows[cell].append(row)
        for cell, rr in rows.items():
            if cell == "agreement256" and arm != "charter":
                continue
            (d / f"{arm}__{cell}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rr))
    return SCRATCH / scenario


ok = True
for scenario, want in (("dial", {"L1a": "PASS", "L2": "PASS", "L3": "PASS"}),
                       ("null", {"L1a": "FAIL", "L2": "NOT EVALUABLE", "L3": "NOT EVALUABLE"})):
    job2 = write(scenario)
    subprocess.run([sys.executable, "-I", "/work/u4161854/loracle_em_exp1/mt1/src/mt1_k1b.py", "--job2", str(job2)],
                   check=True, capture_output=True)
    k = json.loads((job2 / "k1b.json").read_text())
    got = {lab: k["labels"][lab]["verdict"] for lab in want}
    ch = k["per_arm"]["charter"]["labels_arm"]
    print(f"{scenario}: {got} | L1b {k['labels']['L1b']['verdict']} | charter: median spread {ch['L1a']['median_spread']:.3f}, "
          f"b {ch['L2']['m1_b']:.3f} {['%.3f' % x for x in ch['L2']['m1_b_ci']]}, f {ch['L3']['m2_f']:.3f}")
    ok &= got == want
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
