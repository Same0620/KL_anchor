"""Does the graded transfer order track each clause's agreement-512 baseline? (zero GPU)

usage: python3 -I p1_artefacts.py <repo>/experiments/prior_coins/dispatch_final_v1/results_grid/scored

Also splits the twin effect into its Charter-ward (mixed_charter - agreement) and Coin-ward
(agreement - mixed_coin) halves, and reports other/malformed mass per clause in the twin cells.
Reads <scored>/legacy_narrow_2pct/<profile>/<arm>/eval.json (all 29 arm-runs, GLM included).
"""
import json, math, sys, statistics as st
from pathlib import Path
ROOT = Path(sys.argv[1]) / "legacy_narrow_2pct"
SURFS = ["heldout", "canonical", "trained"]
TR = ["precedence_days_since", "precedence_registry_rank", "precedence_runs_year", "qual_skill", "qual_specialty"]
HELD = ["precedence_deferrals", "qual_weekly_limit"]
ON = "precedence_days_since"

def lo(c): return math.log((c.get("charter", 0) + 0.5) / (c.get("coin", 0) + 0.5))
def frac(c, k): n = sum(c.values()); return c.get(k, 0) / n if n else float('nan')
def get(doc, ep, sl, surf):
    r = doc["result"].get(ep, {}).get(f"{sl}__{surf}")
    return r["conflict_runs_by_clause"] if r else None

def spearman(x, y):
    def rank(v):
        s = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[s[j + 1]] == v[s[i]]: j += 1
            for k in range(i, j + 1): r[s[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = rank(x), rank(y); n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float('nan')

rows = []
for prof in sorted(p for p in ROOT.iterdir() if p.is_dir()):
    for arm in ["charter", "coin", "control"]:
        f = prof / arm / "eval.json"
        if not f.exists(): continue
        doc = json.loads(f.read_text())
        per = {}
        for cl in TR + HELD:
            sl = "eval_holdout_conflict" if cl in HELD else "eval_trained_conflict"
            vals = {k: [] for k in ["ag", "mh", "mc", "oth_mh", "oth_mc", "mal_mh", "mal_mc", "oth_ag"]}
            ok = True
            for surf in SURFS:
                ag = get(doc, "agreement-step512", sl, surf); mh = get(doc, "mixed_charter-step512", sl, surf); mc = get(doc, "mixed_coin-step512", sl, surf)
                if not (ag and mh and mc) or cl not in ag: ok = False; break
                vals["ag"].append(lo(ag[cl])); vals["mh"].append(lo(mh[cl])); vals["mc"].append(lo(mc[cl]))
                vals["oth_mh"].append(frac(mh[cl], "other")); vals["oth_mc"].append(frac(mc[cl], "other")); vals["oth_ag"].append(frac(ag[cl], "other"))
                vals["mal_mh"].append(frac(mh[cl], "malformed")); vals["mal_mc"].append(frac(mc[cl], "malformed"))
            if ok: per[cl] = {k: st.mean(v) for k, v in vals.items()}
        if ON in per and all(c in per for c in TR):
            rows.append((prof.name, arm, per))

print("n arm-runs:", len(rows))
# 1. does the off-target twin effect track the agreement-512 baseline?
print("\n1) Within-arm-run Spearman(twin effect, agreement-512 log-odds) over the 6 off-target clauses; and Spearman(twin effect, up-half), (twin, down-half)")
sp_base, sp_up, sp_down, up_share = [], [], [], []
for prof, arm, per in rows:
    offs = [c for c in TR[1:] + HELD if c in per]
    if len(offs) < 5: continue
    tw = [per[c]["mh"] - per[c]["mc"] for c in offs]
    base = [per[c]["ag"] for c in offs]
    up = [per[c]["mh"] - per[c]["ag"] for c in offs]
    down = [per[c]["ag"] - per[c]["mc"] for c in offs]
    sp_base.append(spearman(tw, base)); sp_up.append(spearman(tw, up)); sp_down.append(spearman(tw, down))
    up_share.append(sum(up) / sum(tw) if sum(tw) else float('nan'))
    if prof in ("gemma3_12b_50m_4ep", "gemma3_27b_190m") or arm == "control" and prof == "gemma3_12b_19m":
        print(f"  {prof:18s} {arm:8s} " + " | ".join(f"{c[11:][:8]:8s} ag{per[c]['ag']:+.2f} up{per[c]['mh']-per[c]['ag']:+.2f} dn{per[c]['ag']-per[c]['mc']:+.2f} tw{per[c]['mh']-per[c]['mc']:+.2f}" for c in offs))
print(f"  median Spearman(twin, baseline) = {st.median(sp_base):+.2f}  (values {[round(x, 2) for x in sp_base]})")
print(f"  median Spearman(twin, up-half)  = {st.median(sp_up):+.2f};  median Spearman(twin, down-half) = {st.median(sp_down):+.2f}")
print(f"  median share of twin effect that is the Charter-ward half (mh-ag)/(mh-mc): {st.median(up_share):.2f}")

# 2. pooled: twin effect vs baseline across all arm-runs x off-target clauses
tw_all, base_all, cl_all = [], [], []
for prof, arm, per in rows:
    for c in TR[1:] + HELD:
        if c in per:
            tw_all.append(per[c]["mh"] - per[c]["mc"]); base_all.append(per[c]["ag"]); cl_all.append(c)
print(f"\n2) Pooled Spearman(twin effect, agreement-512 baseline) over {len(tw_all)} (arm-run x clause) points: {spearman(tw_all, base_all):+.2f}")
for c in TR[1:] + HELD:
    b = [b for b, cc in zip(base_all, cl_all) if cc == c]; t = [t for t, cc in zip(tw_all, cl_all) if cc == c]
    print(f"   {c:26s} median baseline lo {st.median(b):+.2f}   median twin effect {st.median(t):+.2f}   n={len(b)}")

# 3. On-target saturation and other/malformed mass
print("\n3) On-target (days_since) cell positions: mh lo, mc lo; and other+malformed mass in twin cells (median over arm-runs)")
mh_on = [per[ON]["mh"] for _, _, per in rows]; mc_on = [per[ON]["mc"] for _, _, per in rows]
print(f"   days_since mixed_charter lo median {st.median(mh_on):+.2f} (range {min(mh_on):+.2f}..{max(mh_on):+.2f}); mixed_coin lo median {st.median(mc_on):+.2f} (range {min(mc_on):+.2f}..{max(mc_on):+.2f})")
for c in TR + HELD:
    o_mh = [per[c]["oth_mh"] for _, _, per in rows if c in per]; o_mc = [per[c]["oth_mc"] for _, _, per in rows if c in per]
    m_mh = [per[c]["mal_mh"] for _, _, per in rows if c in per]; m_mc = [per[c]["mal_mc"] for _, _, per in rows if c in per]
    o_ag = [per[c]["oth_ag"] for _, _, per in rows if c in per]
    print(f"   {c:26s} other: ag {st.median(o_ag):.2f} mh {st.median(o_mh):.2f} mc {st.median(o_mc):.2f} | malformed: mh {st.median(m_mh):.2f} mc {st.median(m_mc):.2f}")
