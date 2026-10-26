"""Split the narrow-twin effect by direction (zero GPU).

usage: python3 -I direction_split.py <repo>/experiments/prior_coins/dispatch_final_v1/results_grid/scored

Charter-ward half = lo(mixed_charter) - lo(agreement-512); Coin-ward half = lo(agreement-512) - lo(mixed_coin),
per arm and clause, legacy narrow draw, Gemma profiles only (GLM's Coin twin has no on-target effect).
Medians over profiles. lo = ln((charter + 0.5) / (coin + 0.5)) over run-level counts, mean of the 3 surfaces.
Reads <scored>/legacy_narrow_2pct/<profile>/<arm>/eval.json
"""
import json, math, sys, statistics as st
from pathlib import Path
ROOT = Path(sys.argv[1]) / "legacy_narrow_2pct"; SURFS = ["heldout", "canonical", "trained"]
TR = ["precedence_days_since", "precedence_runs_year", "precedence_registry_rank", "qual_skill", "qual_specialty"]; HELD = ["precedence_deferrals", "qual_weekly_limit"]
def lo(c): return math.log((c.get("charter", 0) + 0.5) / (c.get("coin", 0) + 0.5))
def get(doc, ep, sl, surf):
    r = doc["result"].get(ep, {}).get(f"{sl}__{surf}"); return r["conflict_runs_by_clause"] if r else None
acc = {arm: {c: {"up": [], "dn": [], "ag": []} for c in TR + HELD} for arm in ["charter", "coin", "control"]}
for prof in sorted(p for p in ROOT.iterdir() if p.is_dir()):
    if "glm" in prof.name: continue   # GLM coin twin has no on-target effect; keep gemma only
    for arm in ["charter", "coin", "control"]:
        f = prof / arm / "eval.json"
        if not f.exists(): continue
        doc = json.loads(f.read_text())
        for c in TR + HELD:
            sl = "eval_holdout_conflict" if c in HELD else "eval_trained_conflict"
            ups = []; dns = []; ags = []
            for surf in SURFS:
                ag = get(doc, "agreement-step512", sl, surf); mh = get(doc, "mixed_charter-step512", sl, surf); mc = get(doc, "mixed_coin-step512", sl, surf)
                if not (ag and mh and mc) or c not in ag: break
                ups.append(lo(mh[c]) - lo(ag[c])); dns.append(lo(ag[c]) - lo(mc[c])); ags.append(lo(ag[c]))
            if ups:
                acc[arm][c]["up"].append(st.mean(ups)); acc[arm][c]["dn"].append(st.mean(dns)); acc[arm][c]["ag"].append(st.mean(ags))
for arm in ["charter", "coin", "control"]:
    print(f"\n== {arm} arm (gemma profiles, n={len(acc[arm][TR[0]]['up'])}); median over profiles; lo = log-odds Charter:Coin")
    print(f"   {'clause':26s} {'agree512 lo':>12s} {'Charter-ward half':>18s} {'Coin-ward half':>15s}")
    for c in TR + HELD:
        a = acc[arm][c]
        print(f"   {c:26s} {st.median(a['ag']):+12.2f} {st.median(a['up']):+18.2f} {st.median(a['dn']):+15.2f}")
