"""MT1 rulings 03 (post hoc, exploratory; PREREG Part 0d, choices in Part H): R12 proportionality per adapter, R13 spread
profiles across sources, R14 training-noise table. CPU only; never changes a label.

Adapters (Charter arm, primary D): days s42 / s43 / released, skill s42 / s43; m_c = agreement-512's clause means.
One joint bootstrap over episodes (mt1_k1b.arm_arrays over all six cells; 2,000 replicates, default_rng(2026), one
multinomial draw per clause per replicate shared by every cell); percentile CIs, undefined replicates dropped (mt1_k1b.ci).
Outputs: mt1/job4/rulings03.json and mt1/job4/rulings03_tables.md (the tables RESULTS §9 copies).
Usage: env/mt1_infer/bin/python -I mt1/src/mt1_rulings03.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.append("/work/u4161854/loracle_em_exp1/mt1/src")
import mt1_k1b as K  # noqa: E402

ROOT = Path("/work/u4161854/loracle_em_exp1/mt1")
CELLS = {"agreement": ROOT / "job2/margins/charter__agreement.jsonl",
         "d42": ROOT / "job3/charter_mixed_coin/margins/charter__charter_mixed_coin.jsonl",
         "d43": ROOT / "job3/charter_mixed_coin_s43/margins/charter__charter_mixed_coin_s43.jsonl",
         "drel": ROOT / "job2/margins/charter__mixed_coin.jsonl",
         "s42": ROOT / "job4/charter_xfam_skill_coin/margins/charter__charter_xfam_skill_coin.jsonl",
         "s43": ROOT / "job4/charter_xfam_skill_coin_s43/margins/charter__charter_xfam_skill_coin_s43.jsonl"}
NAME = {"d42": "days s42", "d43": "days s43", "drel": "days released", "s42": "skill s42", "s43": "skill s43"}
DAYS, SKILL = ("d42", "d43", "drel"), ("s42", "s43")
ADAPTERS = DAYS + SKILL
ON = {**{a: K.ON for a in DAYS}, **{a: "qual_skill" for a in SKILL}}
OFF = {**{a: K.OFF for a in DAYS},
       **{a: ("precedence_days_since", "qual_specialty", "qual_weekly_limit", "precedence_runs_year", "precedence_deferrals",
              "precedence_registry_rank") for a in SKILL}}
FAM = {**{a: K.FAMILY_DAYS for a in DAYS}, **{a: ("qual_specialty", "qual_weekly_limit") for a in SKILL}}
SHARED = ("qual_specialty", "qual_weekly_limit", "precedence_runs_year", "precedence_deferrals", "precedence_registry_rank")
CROSS = [(d, s) for d in DAYS for s in SKILL]
SAME = [("d42", "d43"), ("d42", "drel"), ("d43", "drel"), ("s42", "s43")]
SHORT = {"precedence_days_since": "days_since", "qual_skill": "skill", "qual_specialty": "specialty",
         "qual_weekly_limit": "weekly_limit", "precedence_runs_year": "runs_year", "precedence_deferrals": "deferrals",
         "precedence_registry_rank": "registry_rank"}


def fit(h, m, off, fam):
    """K1b's fit (mt1_k1b.fit) over the given six clauses and family: Model 1 h = a + b m; Model 2 h = a + b m + f F."""
    y = np.array([h[c] for c in off])
    x = np.array([m[c] for c in off])
    F = np.array([1.0 if c in fam else 0.0 for c in off])
    c1 = np.linalg.lstsq(np.column_stack([np.ones(6), x]), y, rcond=None)[0]
    c2 = np.linalg.lstsq(np.column_stack([np.ones(6), x, F]), y, rcond=None)[0]
    return {"m1_a": float(c1[0]), "m1_b": float(c1[1]), "m2_a": float(c2[0]), "m2_b": float(c2[1]), "m2_f": float(c2[2]),
            "m1_residuals": {c: float(y[i] - (c1[0] + c1[1] * x[i])) for i, c in enumerate(off)}}


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")


def spear(x, y):
    return float(spearmanr(x, y).statistic) if np.std(x) > 0 and np.std(y) > 0 else float("nan")


def ratio(v):
    return max(v) / min(v) if min(v) > 0 else float("nan")


def stats(M):
    m = M["agreement"]
    half = {a: {c: m[c] - M[a][c] for c in K.CLAUSES} for a in ADAPTERS}
    spread = {a: {c: half[a][c] / half[a][ON[a]] for c in K.CLAUSES} for a in ADAPTERS}
    out = {"half": half, "spread": spread, "r12": {}, "r13": {}, "r14": {}}
    for a in ADAPTERS:
        out["r12"][a] = {**fit(half[a], m, OFF[a], FAM[a]), "pearson_half_m": pearson([half[a][c] for c in OFF[a]], [m[c] for c in OFF[a]])}
    bs = [out["r12"][a]["m1_b"] for a in ADAPTERS]
    out["r12_b_range"] = max(bs) - min(bs)
    prof = {a: [spread[a][c] for c in SHARED] for a in ADAPTERS}
    for x, y in CROSS + SAME:
        out["r13"][f"{x}|{y}"] = {"pearson": pearson(prof[x], prof[y]), "spearman": spear(prof[x], prof[y])}
    for k in ("pearson", "spearman"):
        mc = float(np.median([out["r13"][f"{x}|{y}"][k] for x, y in CROSS]))
        ms = float(np.median([out["r13"][f"{x}|{y}"][k] for x, y in SAME]))
        out[f"r13_median_{k}"] = {"cross": mc, "same": ms, "cross_minus_same": mc - ms}
    for c in K.CLAUSES:
        out["r14"][c] = {f"{q}_ratio_{g}": ratio([out[q][a][c] for a in grp]) for q in ("half", "spread")
                         for g, grp in (("days", DAYS), ("skill", SKILL))}
    return out


def leaves(d, pre=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from leaves(v, f"{pre}{k}/")
        else:
            yield f"{pre}{k}", v


def get(d, path):
    for k in path.split("/"):
        d = d[k]
    return d


def main():
    cells = {k: K.load_cell(p) for k, p in CELLS.items()}
    arr = K.arm_arrays(cells)
    M0 = K.clause_means(arr)
    point = stats(M0)
    for a in DAYS:   # the generalised fit equals mt1_k1b.fit on K1b's own clauses and family
        ref = K.fit(point["half"][a], M0["agreement"], K.FAMILY_DAYS)
        assert all(abs(ref[k] - point["r12"][a][k]) < 1e-12 for k in ("m1_a", "m1_b", "m2_a", "m2_b", "m2_f")), a
    # K1b check: days released is K1b's Charter-arm D input (mixed_coin) -> the same point fit
    k1b = json.loads((ROOT / "job2/k1b.json").read_text())["per_arm"]["charter"]
    for key in ("m1_a", "m1_b", "m2_b", "m2_f"):
        assert abs(point["r12"]["drel"][key] - k1b["point"]["fit_D"][key]) < 1e-12, key
    rng = np.random.default_rng(2026)
    boots = []
    for _ in range(2000):
        w = {cl: rng.multinomial(len(arr[cl][0]), np.full(len(arr[cl][0]), 1 / len(arr[cl][0]))).astype(float) for cl in arr}
        boots.append(stats(K.clause_means(arr, w)))
    flat = dict(leaves(point))
    ci = {k: K.ci([get(b, k) for b in boots]) for k in flat}
    n_undef = {k: int(sum(1 for b in boots if not np.isfinite(get(b, k)))) for k in flat}
    same_ci_as_k1b = all(abs(ci[f"r12/drel/{k}"][j] - k1b["ci"]["fit_D"][k][j]) < 1e-12 for k in ("m1_b", "m2_b", "m2_f") for j in (0, 1))
    out = {"note": "post hoc, exploratory (rulings 03); no label changes", "bootstrap": {"n": 2000, "seed": 2026, "joint_cells": list(CELLS)},
           "n_episodes": {cl: len(arr[cl][0]) for cl in arr}, "k1b_check": {"point_equal": True, "ci_equal": bool(same_ci_as_k1b)},
           "point": point, "ci": {k: v for k, v in ci.items()}, "n_undefined_replicates": {k: v for k, v in n_undef.items() if v}}

    # scoring (PREREG H2-H4)
    b_pos = [a for a in ADAPTERS if ci[f"r12/{a}/m1_b"][0] > 0]
    b_pos_either = [a for a in ADAPTERS if ci[f"r12/{a}/m1_b"][0] > 0 or ci[f"r12/{a}/m2_b"][0] > 0]
    cross_p = {f"{x}|{y}": point["r13"][f"{x}|{y}"]["pearson"] for x, y in CROSS}
    md = point["r13_median_pearson"]
    stab = {}
    for g, grp in (("days", DAYS), ("skill", SKILL)):
        on_c = ON[grp[0]]
        vals = {c: point["r14"][c][f"spread_ratio_{g}"] for c in K.CLAUSES if c != on_c}
        ok = {c: v for c, v in vals.items() if np.isfinite(v)}
        stab[g] = {"ranking_most_to_least_stable": sorted(ok, key=ok.get), "ratios": vals,
                   "n_a_sign": [c for c, v in vals.items() if not np.isfinite(v)],
                   "on_target_half_ratio": ratio([point["half"][a][on_c] for a in grp])}
    out["scoring"] = {
        "R12a": {"n_m1_b_ci_excl_0": len(b_pos), "adapters": b_pos, "true": len(b_pos) >= 4, "lead_P": 0.75,
                 "secondary_either_model": len(b_pos_either)},
        "R13a": {"n_cross_pearson_ge_0.8": sum(v >= 0.8 for v in cross_p.values()), "true": sum(v >= 0.8 for v in cross_p.values()) >= 4,
                 "lead_P": 0.50},
        "R13b": {"median_cross": md["cross"], "median_same": md["same"], "abs_diff": abs(md["cross_minus_same"]),
                 "true": abs(md["cross_minus_same"]) <= 0.15, "lead_P": 0.55},
        "R12_b_range": [min(point["r12"][a]["m1_b"] for a in ADAPTERS), max(point["r12"][a]["m1_b"] for a in ADAPTERS)],
        "R14_stability": stab}
    for k in ("R12a", "R13a", "R13b"):
        s = out["scoring"][k]
        s["brier"] = (s["lead_P"] - (1.0 if s["true"] else 0.0)) ** 2
    (ROOT / "job4/rulings03.json").write_text(json.dumps(out, indent=1))

    # tables for RESULTS §9
    f = lambda k, d=2: f"{flat[k]:.{d}f} [{ci[k][0]:.{d}f}, {ci[k][1]:.{d}f}]" if np.isfinite(flat[k]) else "n/a"  # noqa: E731
    L = ["**R12. K1b's Models 1 and 2 per adapter** (six off-target clauses each; m_c = agreement-512).", "",
         "| adapter | Model 1 b | Model 1 a | Model 2 b | f | Pearson r(half, m_c) |", "|---|---|---|---|---|---|"]
    for a in ADAPTERS:
        L.append(f"| {NAME[a]} | {f(f'r12/{a}/m1_b', 3)} | {f(f'r12/{a}/m1_a')} | {f(f'r12/{a}/m2_b', 3)} | {f(f'r12/{a}/m2_f')} | {f(f'r12/{a}/pearson_half_m')} |")
    L += ["", "Model 1 residuals per clause (nats; family clauses marked *):", "",
          "| adapter | " + " | ".join(SHORT[c] for c in K.CLAUSES) + " |", "|---|" + "---|" * len(K.CLAUSES)]
    for a in ADAPTERS:
        cells_ = []
        for c in K.CLAUSES:
            cells_.append("on-target" if c == ON[a] else f(f"r12/{a}/m1_residuals/{c}") + ("*" if c in FAM[a] else ""))
        L.append(f"| {NAME[a]} | " + " | ".join(cells_) + " |")
    L += ["", f"Range of Model 1 b across the five adapters: {out['scoring']['R12_b_range'][0]:.3f} to {out['scoring']['R12_b_range'][1]:.3f} "
          f"(max − min {f('r12_b_range', 3)}).", "",
          "**R13. Spread profiles over the five shared clauses** (specialty, weekly_limit, runs_year, deferrals, registry_rank; half ÷ own on-target half).", "",
          "| pair | kind | Pearson [CI] | Spearman [CI] |", "|---|---|---|---|"]
    for x, y in CROSS + SAME:
        L.append(f"| {NAME[x]} × {NAME[y]} | {'cross-source' if (x, y) in CROSS else 'same-source'} | {f(f'r13/{x}|{y}/pearson')} | {f(f'r13/{x}|{y}/spearman')} |")
    L += ["", "| median over pairs | cross-source | same-source | cross − same [CI] |", "|---|---|---|---|"]
    for k in ("pearson", "spearman"):
        L.append(f"| {k.capitalize()} | {flat[f'r13_median_{k}/cross']:.2f} | {flat[f'r13_median_{k}/same']:.2f} | {f(f'r13_median_{k}/cross_minus_same')} |")
    L += ["", "**R14. Training-noise table**: half (nats, D) and spread (half ÷ own on-target half) per clause and adapter, with max ÷ min across the days adapters and across the skill adapters.", "",
          "| clause | " + " | ".join(NAME[a] for a in ADAPTERS) + " | max÷min days | max÷min skill |", "|---|" + "---|" * (len(ADAPTERS) + 2)]
    for c in K.CLAUSES:
        L.append(f"| {SHORT[c]} half | " + " | ".join(f(f"half/{a}/{c}") for a in ADAPTERS)
                 + f" | {f(f'r14/{c}/half_ratio_days')} | {f(f'r14/{c}/half_ratio_skill')} |")
        L.append(f"| {SHORT[c]} spread | " + " | ".join("1 (on-target)" if c == ON[a] else f(f"spread/{a}/{c}", 3) for a in ADAPTERS)
                 + f" | {'—' if c == K.ON else f(f'r14/{c}/spread_ratio_days')} | {'—' if c == 'qual_skill' else f(f'r14/{c}/spread_ratio_skill')} |")
    (ROOT / "job4/rulings03_tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print("\nk1b check:", out["k1b_check"], "| undefined replicates:", out["n_undefined_replicates"] or "none")
    print("scoring:", json.dumps(out["scoring"], indent=1))


if __name__ == "__main__":
    main()
