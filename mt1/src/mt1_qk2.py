"""MT1: the Q_K2 draw (handoff B.10) and Job 5's 200 days_since episodes (handoff §8). Standard library only.

rng = random.Random(2026); strata = the 6 off-target clauses in B.11 order x ("c", "c/c"); within each stratum the
episode ids of the two __heldout conflict sets, sorted, then rng.sample(ids, 100) with the one rng carried across strata.
Usage: python3 -I mt1/src/mt1_qk2.py <eval data dir> <out.json>
"""
import json
import random
import sys

EV, OUT = sys.argv[1], sys.argv[2]
B11_OFF = ("qual_skill", "qual_specialty", "qual_weekly_limit", "precedence_runs_year", "precedence_deferrals",
           "precedence_registry_rank")
SETS = ("eval_trained_conflict", "eval_holdout_conflict")

meta = {}
for s in SETS:
    with open(f"{EV}/episodes/{s}.jsonl", "rb") as f:
        eps = {json.loads(x)["episode_id"]: json.loads(x)["v4_metadata"] for x in f if x.strip()}
    with open(f"{EV}/prompts/{s}__heldout.jsonl", "rb") as f:
        for x in f:
            if x.strip():
                pid = json.loads(x)["id"]
                meta[pid] = (eps[pid]["target_clause"], eps[pid]["mixture"])

rng = random.Random(2026)
strata, qk2 = [], []
for clause in B11_OFF:
    for mix in ("c", "c/c"):
        ids = sorted(i for i, (c, m) in meta.items() if c == clause and m == mix)
        pick = rng.sample(ids, 100)
        strata.append({"clause": clause, "mixture": mix, "n_available": len(ids), "ids": pick})
        qk2 += pick
days = {m: sorted(i for i, (c, mm) in meta.items() if c == "precedence_days_since" and mm == m)[:100] for m in ("c", "c/c")}
assert len(qk2) == len(set(qk2)) == 1200
out = {"python": sys.version.split()[0], "seed": 2026, "order": list(B11_OFF), "strata": strata, "q_k2": qk2,
       "job5_days_since": days["c"] + days["c/c"]}
json.dump(out, open(OUT, "w"), indent=1)
print(sys.version.split()[0], "Q_K2", len(qk2), "first", qk2[:3], "last", qk2[-2:], "| days_since", len(out["job5_days_since"]),
      "| strata sizes", sorted({s["n_available"] for s in strata}))
