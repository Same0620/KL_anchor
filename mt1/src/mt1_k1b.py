"""MT1 K1b (handoff §5, B.4-B.5, §10): halves, spreads, proportionality fits and the labels L1a, L1b, L2, L3 from
Job 2's margins. CPU only.

Inputs : <job2>/margins/<arm>__{agreement,mixed_charter,mixed_coin}.jsonl (+ charter__agreement256.jsonl for B_c)
         and the released legacy-narrow eval.json files (L1b).
Outputs: <job2>/k0_summary.json and <job2>/k1b.json.
Bootstrap: 2,000 resamples, within each clause, over episodes (a two-run episode is one unit); one draw of episode ids
per replicate is applied to every cell. Percentile 95% CIs. The seed (2026) is an implementation choice, logged.
Usage: env/mt1_infer/bin/python -I mt1/src/mt1_k1b.py [--job2 DIR]      (test: mt1/src/test_mt1_k1b.py)
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ARMS = ("charter", "coin", "control")
PRIMARY = {"charter": "D", "coin": "U", "control": "U"}
ON = "precedence_days_since"
OFF = ("qual_skill", "qual_specialty", "qual_weekly_limit", "precedence_runs_year", "precedence_deferrals",
       "precedence_registry_rank")                                   # B.11 order
FAMILY_DAYS = ("precedence_runs_year", "precedence_deferrals")       # B.5
HELD = ("qual_weekly_limit", "precedence_deferrals")
CLAUSES = (ON,) + OFF
SCORED = "/work/u4161854/dispatch_data/repo/science-of-midtraining/experiments/prior_coins/dispatch_final_v1/results_grid/scored"
N_BOOT, SEED = 2000, 2026


def finite(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def runs_of(row):
    vals = [row["m_run1"]] if row["n_runs"] == 1 else [row["m_run1"], row["m_run2"]]
    return [v for v in vals if finite(v)]


def load_cell(path):
    with open(path) as f:
        return {r["id"]: r for r in map(json.loads, f) if r["tokens_ok"]}


def arm_arrays(cells):
    """-> {clause: (ids, {cell: S}, N, {cell: Sep}, Nep)}: per-episode run sums S, run counts N, episode margins."""
    ids_all = set.intersection(*(set(c) for c in cells.values()))
    ref = next(iter(cells.values()))
    out = {}
    for cl in CLAUSES:
        ids = sorted(i for i in ids_all if ref[i]["clause"] == cl)
        n = np.array([len(runs_of(ref[i])) for i in ids], float)
        s = {k: np.array([sum(runs_of(c[i])) for i in ids], float) for k, c in cells.items()}
        sep = {k: np.array([c[i]["m_ep"] for i in ids], float) for k, c in cells.items()}
        assert all(len(runs_of(c[i])) == len(runs_of(ref[i])) for c in cells.values() for i in ids)
        out[cl] = (ids, s, n, sep)
    return out


def clause_means(arr, w=None):
    """-> {cell: {clause: mean m_run}} with optional per-clause episode weights (bootstrap counts)."""
    res = {}
    for cl, (ids, s, n, _) in arr.items():
        ww = np.ones(len(ids)) if w is None else w[cl]
        for k in s:
            res.setdefault(k, {})[cl] = float((ww * s[k]).sum() / (ww * n).sum())
    return res


def halves(M):
    U = {c: M["mixed_charter"][c] - M["agreement"][c] for c in CLAUSES}
    D = {c: M["agreement"][c] - M["mixed_coin"][c] for c in CLAUSES}
    return U, D


def fit(h, m, fam):
    """Model 1: h = a + b m; Model 2: h = a + b m + f F (OLS over the 6 off-target clauses)."""
    y = np.array([h[c] for c in OFF])
    x = np.array([m[c] for c in OFF])
    F = np.array([1.0 if c in fam else 0.0 for c in OFF])
    X1 = np.column_stack([np.ones(6), x])
    X2 = np.column_stack([np.ones(6), x, F])
    c1 = np.linalg.lstsq(X1, y, rcond=None)[0]
    c2 = np.linalg.lstsq(X2, y, rcond=None)[0]
    resid1 = {c: float(y[i] - (c1[0] + c1[1] * x[i])) for i, c in enumerate(OFF)}
    nf = [i for i, c in enumerate(OFF) if c not in fam]
    c_nf = np.linalg.lstsq(X1[nf], y[nf], rcond=None)[0]
    return {"m1_a": float(c1[0]), "m1_b": float(c1[1]), "m2_a": float(c2[0]), "m2_b": float(c2[1]), "m2_f": float(c2[2]),
            "m1_nonfamily_a": float(c_nf[0]), "m1_nonfamily_b": float(c_nf[1]), "m1_residuals": resid1}


def stats_from(M):
    U, D = halves(M)
    m = dict(M["agreement"])
    out = {"U": U, "D": D}
    for h_name, h in (("U", U), ("D", D)):
        out[f"spread_{h_name}"] = {c: (h[c] / h[ON] if h[ON] != 0 else float("nan")) for c in OFF}
        out[f"fit_{h_name}"] = fit(h, m, FAMILY_DAYS)
    out["m_c"] = m
    if "agreement256" in M:
        out["B"] = {c: M["agreement"][c] - M["agreement256"][c] for c in CLAUSES}
    return out


def ci(vals):
    v = np.array([x for x in vals if finite(x)], float)
    if len(v) == 0:
        return [float("nan"), float("nan")]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def released_lo_halves(arm):
    doc = json.loads(Path(f"{SCORED}/legacy_narrow_2pct/gemma3_12b_50m_4ep/{arm}/eval.json").read_text())

    def lo(ep, cl):
        sl = "eval_holdout_conflict" if cl in HELD else "eval_trained_conflict"
        c = doc["result"][ep][f"{sl}__heldout"]["conflict_runs_by_clause"][cl]
        return math.log((c.get("charter", 0) + 0.5) / (c.get("coin", 0) + 0.5))

    ag = {c: lo("agreement-step512", c) for c in CLAUSES}
    mh = {c: lo("mixed_charter-step512", c) for c in CLAUSES}
    mc = {c: lo("mixed_coin-step512", c) for c in CLAUSES}
    return {"U": {c: mh[c] - ag[c] for c in CLAUSES}, "D": {c: ag[c] - mc[c] for c in CLAUSES}}


def analyse_arm(arm, cells, rng):
    arr = arm_arrays(cells)
    point = stats_from(clause_means(arr))
    boots = []
    for _ in range(N_BOOT):
        w = {cl: rng.multinomial(len(arr[cl][0]), np.full(len(arr[cl][0]), 1 / len(arr[cl][0]))).astype(float) for cl in arr}
        boots.append(stats_from(clause_means(arr, w)))
    res = {"n_episodes": {cl: len(arr[cl][0]) for cl in arr}, "n_runs": {cl: int(arr[cl][2].sum()) for cl in arr},
           "point": point, "ci": {}}
    for h in ("U", "D"):
        res["ci"][h] = {c: ci([b[h][c] for b in boots]) for c in CLAUSES}
        res["ci"][f"spread_{h}"] = {c: ci([b[f"spread_{h}"][c] for b in boots]) for c in OFF}
        res["ci"][f"fit_{h}"] = {k: ci([b[f"fit_{h}"][k] for b in boots]) for k in ("m1_a", "m1_b", "m2_a", "m2_b", "m2_f")}
    if "B" in point:
        res["ci"]["B"] = {c: ci([b["B"][c] for b in boots]) for c in CLAUSES}
    # labels, per arm, primary direction
    h = PRIMARY[arm]
    on_ci = res["ci"][h][ON]
    evaluable = all(k in cells for k in ("agreement", "mixed_charter", "mixed_coin")) and not (on_ci[0] <= 0 <= on_ci[1])
    n_sig = sum(res["ci"][h][c][0] > 0 for c in OFF)                 # CI excludes 0 in the trained direction
    med_spread = float(np.median([point[f"spread_{h}"][c] for c in OFF]))
    lo = released_lo_halves(arm)
    rho = float(spearmanr([point[h][c] for c in OFF], [lo[h][c] for c in OFF]).statistic)
    f1, f2 = res["ci"][f"fit_{h}"]["m1_b"], res["ci"][f"fit_{h}"]["m2_b"]
    l2_pass = (f1[0] > 0) or (f2[0] > 0)
    l2_fail = (f1[0] <= 0 <= f1[1] or f1[1] < 0) and (f2[0] <= 0 <= f2[1] or f2[1] < 0)
    fci = res["ci"][f"fit_{h}"]["m2_f"]
    resid = point[f"fit_{h}"]["m1_residuals"]
    l3_pass = fci[0] > 0 and all(resid[c] > 0 for c in FAMILY_DAYS)
    res["labels_arm"] = {"primary": h, "evaluable": bool(evaluable), "on_target_half": point[h][ON], "on_target_ci": on_ci,
                         "L1a": {"n_offtarget_ci_excl_0": n_sig, "median_spread": med_spread,
                                 "pass": bool(evaluable and n_sig >= 4 and med_spread >= 0.10)},
                         "L1b": {"spearman": rho, "released_lo_half": {c: lo[h][c] for c in OFF}, "pass": bool(rho >= 0.6)},
                         "L2": {"m1_b": point[f"fit_{h}"]["m1_b"], "m1_b_ci": f1, "m2_b": point[f"fit_{h}"]["m2_b"],
                                "m2_b_ci": f2, "pass": bool(l2_pass), "fail_condition": bool(l2_fail)},
                         "L3": {"m2_f": point[f"fit_{h}"]["m2_f"], "m2_f_ci": fci,
                                "family_residuals": {c: resid[c] for c in FAMILY_DAYS}, "pass": bool(l3_pass)}}
    return res


def combine(per_arm, key, fail_key=None):
    ev = [a for a, r in per_arm.items() if r["labels_arm"]["evaluable"]]
    n_pass = sum(per_arm[a]["labels_arm"][key]["pass"] for a in ev)
    if fail_key:
        n_fail = sum(per_arm[a]["labels_arm"][key][fail_key] for a in ev)
    else:
        n_fail = sum(not per_arm[a]["labels_arm"][key]["pass"] for a in ev)
    verdict = "PASS" if n_pass >= 2 else ("FAIL" if n_fail >= 2 else "INCONCLUSIVE")
    return {"verdict": verdict, "evaluable_arms": ev, "n_pass": n_pass, "n_fail": n_fail}


def k0_summary(cells_by_arm, rng):
    out = {}
    for arm, cells in cells_by_arm.items():
        arr = arm_arrays(cells)
        for k in cells:
            for cl, (ids, s, n, sep) in arr.items():
                mean_run = float(s[k].sum() / n.sum())
                mean_ep = float(sep[k].mean())
                bs_run, bs_ep = [], []
                for _ in range(N_BOOT):
                    w = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)))
                    bs_run.append((w * s[k]).sum() / (w * n).sum())
                    bs_ep.append((w * sep[k]).sum() / w.sum())
                out.setdefault(arm, {}).setdefault(k, {})[cl] = {
                    "mean_m_run": mean_run, "se_m_run": float(np.std(bs_run, ddof=1)), "n_runs": int(n.sum()),
                    "mean_m_ep": mean_ep, "se_m_ep": float(np.std(bs_ep, ddof=1)), "n_episodes": len(ids)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job2", default="/work/u4161854/loracle_em_exp1/mt1/job2")
    args = ap.parse_args()
    d = Path(args.job2) / "margins"
    cells_by_arm = {}
    for arm in ARMS:
        cells = {}
        for cell in ("agreement", "mixed_charter", "mixed_coin", "agreement256"):
            p = d / f"{arm}__{cell}.jsonl"
            if p.exists():
                cells[cell] = load_cell(p)
        if cells:
            cells_by_arm[arm] = cells
    rng = np.random.default_rng(SEED)
    per_arm = {arm: analyse_arm(arm, cells, rng) for arm, cells in cells_by_arm.items()
               if all(k in cells for k in ("agreement", "mixed_charter", "mixed_coin"))}
    labels = {"L1a": combine(per_arm, "L1a"), "L1b": combine(per_arm, "L1b"), "L2": combine(per_arm, "L2", "fail_condition"),
              "L3": combine(per_arm, "L3")}
    if labels["L1a"]["verdict"] == "FAIL":
        labels["L2"]["verdict"] = labels["L3"]["verdict"] = "NOT EVALUABLE"
    out = {"bootstrap": {"n": N_BOOT, "seed": SEED, "scheme": "within clause, over episodes, shared across cells"},
           "arms_scored": sorted(cells_by_arm), "labels": labels, "per_arm": per_arm}
    (Path(args.job2) / "k1b.json").write_text(json.dumps(out, indent=1))
    (Path(args.job2) / "k0_summary.json").write_text(json.dumps(k0_summary(cells_by_arm, np.random.default_rng(SEED + 1)), indent=1))
    for k, v in labels.items():
        print(k, v)


if __name__ == "__main__":
    main()
