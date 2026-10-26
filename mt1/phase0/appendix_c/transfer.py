"""Narrow-twin cross-clause transfer across released profiles (zero GPU).

usage: python3 -I transfer.py <repo>/experiments/prior_coins/dispatch_final_v1/results_grid/scored

For each profile x arm, from the legacy narrow 2% draw (all 164 conflict rows are
single-run precedence_days_since) compute per-clause Charter:Coin log-odds at
step 512 for agreement, mixed_charter, mixed_coin; the twin difference
(mixed_charter - mixed_coin) per clause; and the transfer ratio
off-target twin diff / on-target (days_since) twin diff.
Same for the canonical (#1c balanced) draw as a comparison.
Reads: <scored>/legacy_narrow_2pct/<profile>/<arm>/eval.json  (legacy)
       <scored>/<profile>/<arm>/eval.json                      (canonical)
"""
import json, math, sys, statistics as st
from pathlib import Path

SCORED = Path(sys.argv[1])
DRAWS = {"legacy": SCORED / "legacy_narrow_2pct", "canon": SCORED}
SURFS = ["heldout", "canonical", "trained"]
TRAINED = ["precedence_days_since", "precedence_registry_rank", "precedence_runs_year", "qual_skill", "qual_specialty"]
HELD = ["precedence_deferrals", "qual_weekly_limit"]
ON = "precedence_days_since"


def lo(c):
    return math.log((c.get("charter", 0) + 0.5) / (c.get("coin", 0) + 0.5))


def clause_lo(doc, ep, slice_, surf):
    r = doc["result"].get(ep, {}).get(f"{slice_}__{surf}")
    if not r or "conflict_runs_by_clause" not in r:
        return None   # e.g. glm45_air_1b / glm45_air_20m_legacy rows carry a different schema
    return {cl: lo(v) for cl, v in r["conflict_runs_by_clause"].items()}


def analyse(draw):
    rows = []
    root = DRAWS[draw]
    for prof_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "legacy_narrow_2pct"):
        for arm in ["charter", "coin", "control"]:
            f = prof_dir / arm / "eval.json"
            if not f.exists():
                continue
            doc = json.loads(f.read_text())
            for surf in SURFS:
                ag = clause_lo(doc, "agreement-step512", "eval_trained_conflict", surf)
                mh = clause_lo(doc, "mixed_charter-step512", "eval_trained_conflict", surf)
                mc = clause_lo(doc, "mixed_coin-step512", "eval_trained_conflict", surf)
                agh = clause_lo(doc, "agreement-step512", "eval_holdout_conflict", surf)
                mhh = clause_lo(doc, "mixed_charter-step512", "eval_holdout_conflict", surf)
                mch = clause_lo(doc, "mixed_coin-step512", "eval_holdout_conflict", surf)
                if not (ag and mh and mc):
                    continue
                twin = {cl: mh[cl] - mc[cl] for cl in TRAINED}
                if mhh and mch:
                    for cl in HELD:
                        twin[cl] = mhh[cl] - mch[cl]
                on = twin[ON]
                rows.append(dict(profile=prof_dir.name, arm=arm, surf=surf, on=on, twin=twin,
                                 ag=ag, mh=mh, mc=mc))
    return rows


def summarise(rows, label):
    print(f"\n######## {label}")
    hdr = "profile              arm      on(day) | ratio reg  runs  skill spec | held: defer wkly | agr512 reg-vs-meanother"
    print(hdr)
    by_arm = {"charter": [], "coin": [], "control": []}
    for prof in sorted({r["profile"] for r in rows}):
        for arm in ["charter", "coin", "control"]:
            rr = [r for r in rows if r["profile"] == prof and r["arm"] == arm]
            if not rr:
                continue
            # average over surfaces (surfaces are not replications; averaging just smooths template noise)
            on = st.mean(r["on"] for r in rr)
            rat = {cl: st.mean(r["twin"][cl] for r in rr) / on if on else float('nan')
                   for cl in TRAINED[1:] + [c for c in HELD if c in rr[0]["twin"]]}
            off_mean = st.mean(rat[cl] for cl in TRAINED[1:])
            ag = {cl: st.mean(r["ag"][cl] for r in rr) for cl in TRAINED}
            reg_gap = ag["precedence_registry_rank"] - st.mean(ag[c] for c in TRAINED if c != "precedence_registry_rank")
            by_arm[arm].append((prof, off_mean, on))
            held = " ".join(f"{rat.get(c, float('nan')):+.2f}" for c in HELD)
            print(f"{prof:20s} {arm:8s} {on:+6.2f} | {rat['precedence_registry_rank']:+.2f} {rat['precedence_runs_year']:+.2f} "
                  f"{rat['qual_skill']:+.2f} {rat['qual_specialty']:+.2f} (mean {off_mean:+.2f}) | {held} | {reg_gap:+.2f}")
    print("\nmean off-target ratio (trained clauses) by arm across profiles:")
    for arm, v in by_arm.items():
        if v:
            print(f"  {arm:8s} n={len(v)}  mean {st.mean(x[1] for x in v):+.2f}  median {st.median(x[1] for x in v):+.2f}  "
                  f"values {[round(x[1], 2) for x in v]}")
    # paired comparison within profile: charter arm vs coin arm
    profs = {p for p, _, _ in by_arm["charter"]} & {p for p, _, _ in by_arm["coin"]}
    diffs = []
    for p in sorted(profs):
        c = [x for x in by_arm["charter"] if x[0] == p][0][1]
        k = [x for x in by_arm["coin"] if x[0] == p][0][1]
        diffs.append((p, c - k))
    print("  paired charter-arm minus coin-arm mean ratio:", [(p, round(d, 2)) for p, d in diffs],
          f" positive in {sum(d > 0 for _, d in diffs)}/{len(diffs)}")


if __name__ == "__main__":
    legacy = analyse("legacy")
    summarise(legacy, "LEGACY NARROW draw (164 single-run days_since conflicts)")
    canon = analyse("canon")
    summarise(canon, "CANONICAL draw (#1c balanced for substituted profiles; gemma 4B still narrow)")
