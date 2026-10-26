"""WS1 decisions 03 analysis (CPU; post hoc) of the F12 job -> ws1/results/f12_analysis.json (+ printed tables).

F1 (per seed): grid = V4's 7 non-zero norms + 11..17. Class per norm from the 90% CI of L_gen - L_nar over 2,000 row resamples
(G: CI below 0; N: CI above 0; 0). Crossings: linear interpolation of the point estimates where the sign changes; each crossing's CI
from the resamples with a same-direction sign change (the one nearest the point crossing), plus the fraction with none. Iso-loss norms
for L = 2.0 .. 1.5 (interpolation from c = 0 along the grid); paired difference c_gen - c_nar with CI over resamples where both reach;
class G / N / 0, no class if either does not reach the level at the point estimate or > 10% of resamples miss. Readings R1/R2/R3 and
R-iso by decisions 03's rules, written only when both seeds give them. K_nar (Part C) and BH8's shares.
F2: per condition, dL over all / first / later answer tokens; Adv with CIs; the interaction residual; anchor KL; Sentences A and B by
the fixed rules. Q1-Q5.
  ws1_f12_analyze.py [--tiny]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ws1_common as W  # noqa: E402
from ws1_common import C  # noqa: E402

TINY = "--tiny" in sys.argv
S1 = (W.TMP / "tiny" / "f12") if TINY else (W.STORE / "s1")
J = W.read_json((S1 / "f12.json") if TINY else (W.RESULTS / "f12.json"))
Z = dict(np.load(S1 / "f12.npz"))
NT = J["counts"]["all"]
IDX = np.random.default_rng(C.stable_seed("ws1-analysis-boot")).integers(0, len(Z["L0"]), (2000, len(Z["L0"])))
LEVELS = [2.0, 1.9, 1.8, 1.7, 1.6, 1.5]


def key(d, c):
    return f"{d}_c{c:.4f}"


def ci(a):
    return [float(np.percentile(a, 5)), float(np.percentile(a, 95))]


def curve(d, grid):
    """Per-row loss pieces [n, len(grid)] (all answer tokens, / NT) for direction d."""
    return np.stack([Z[key(d, c)][:, 0] / NT for c in grid], 1)


def cls_of(lo, hi):
    return "G" if hi < 0 else ("N" if lo > 0 else "0")


def crossings(grid, diff):
    out = []
    for i in range(len(grid) - 1):
        a, b = diff[i], diff[i + 1]
        if a == 0 or (a < 0) != (b < 0):
            c = grid[i] + (grid[i + 1] - grid[i]) * (-a) / (b - a) if b != a else grid[i]
            out.append({"between": [grid[i], grid[i + 1]], "c": float(c), "direction": "G->N" if a < 0 else "N->G"})
    return out


def iso(grid, Lc, L0, level):
    g = [0.0] + list(grid)
    v = [L0] + list(Lc)
    for i in range(len(g) - 1):
        if v[i] > level >= v[i + 1]:
            return g[i] + (g[i + 1] - g[i]) * (v[i] - level) / (v[i] - v[i + 1])
    return None


def f1_seed(seed, v4c, window):
    grid = sorted(set([c for c in v4c if c > 0] + window))
    g, n = curve(f"gen{seed}", grid), curve(f"nar{seed}", grid)
    L0r = Z["L0"][:, 0] / NT
    dpt = g.sum(0) - n.sum(0)
    bs = np.stack([g[i].sum(0) - n[i].sum(0) for i in IDX])               # [2000, grid]
    rows = []
    for j, c in enumerate(grid):
        lo, hi = ci(bs[:, j])
        rows.append({"c": c, "L_gen": float(g[:, j].sum()), "L_nar": float(n[:, j].sum()), "diff": float(dpt[j]), "ci90": [lo, hi],
                     "class": cls_of(lo, hi), "lower_point": "gen" if dpt[j] < 0 else "nar"})
    cr = crossings(grid, dpt)
    for x in cr:
        hits = []
        for b in bs:
            cb = [y for y in crossings(grid, b) if y["direction"] == x["direction"]]
            if cb:
                hits.append(min(cb, key=lambda y: abs(y["c"] - x["c"]))["c"])
        x["ci90"] = ci(np.array(hits)) if hits else [None, None]
        x["frac_no_cross"] = 1 - len(hits) / len(bs)
    isos = []
    L0 = float(L0r.sum())
    for lev in LEVELS:
        cg, cn = iso(grid, g.sum(0), L0, lev), iso(grid, n.sum(0), L0, lev)
        diffs, miss = [], 0
        for i in IDX:
            a, b = iso(grid, g[i].sum(0), float(L0r[i].sum()), lev), iso(grid, n[i].sum(0), float(L0r[i].sum()), lev)
            if a is None or b is None:
                miss += 1
            else:
                diffs.append(a - b)
        row = {"L": lev, "c_gen": cg, "c_nar": cn, "frac_either_misses": miss / len(IDX)}
        if cg is not None and cn is not None:
            row["diff"] = cg - cn
            row["ci90"] = ci(np.array(diffs)) if diffs else [None, None]
        row["class"] = (cls_of(*row["ci90"]) if (cg is not None and cn is not None and row["frac_either_misses"] <= 0.10 and diffs) else None)
        isos.append(row)
    return {"grid": grid, "L0": L0, "rows": rows, "crossings": cr, "iso": isos}


def reading(rows, cstar):
    classes = [r["class"] for r in rows]
    cs = [r["c"] for r in rows]
    Nidx = [i for i, k in enumerate(classes) if k == "N"]
    if not Nidx:
        return "R3", None
    run = Nidx == list(range(Nidx[0], Nidx[-1] + 1))
    near = min(range(len(cs)), key=lambda i: abs(cs[i] - cstar))
    G_below = any(classes[i] == "G" for i in range(0, Nidx[0]))
    G_above = any(classes[i] == "G" for i in range(Nidx[-1] + 1, len(cs)))
    if run and Nidx[0] <= near <= Nidx[-1] and G_below and G_above:
        return "R1", [cs[Nidx[0]], cs[Nidx[-1]]]
    if run and Nidx[-1] == len(cs) - 1 and G_below:
        return "R2", None
    return "R3", None


def r_iso(isos):
    cl = [(r["L"], r["class"]) for r in isos if r["class"] is not None]
    if not cl:
        return None
    seq = [k for _, k in cl]
    if all(k == "G" for k in seq):
        return ("all G", cl[-1][0])
    if all(k == "N" for k in seq):
        return ("all N", cl[-1][0])
    # a run of G, then a run of N, with 0 allowed only between the two runs
    s = "".join(seq)
    import re
    mm = re.fullmatch(r"(G+)(0*)(N+)", s)
    if mm:
        lastG = cl[len(mm.group(1)) - 1][0]
        firstN = cl[len(mm.group(1)) + len(mm.group(2))][0]
        return ("G then N", lastG, firstN)
    return None


def main():
    res = {"F1": {}, "F2": {}}
    v4c, window = J["v4c"], J["window"]
    cstar = {int(k): v for k, v in J["cstar"].items()}
    for seed in (0, 1):
        res["F1"][f"seed{seed}"] = f = f1_seed(seed, v4c, window)
        f["reading"] = reading(f["rows"], cstar[seed])
        f["r_iso"] = r_iso(f["iso"])
    rd = [res["F1"][f"seed{s}"]["reading"] for s in (0, 1)]
    res["F1"]["reading_both"] = rd[0][0] if rd[0][0] == rd[1][0] and rd[0][0] != "R3" else "R3"
    ri = [res["F1"][f"seed{s}"]["r_iso"] for s in (0, 1)]
    res["F1"]["r_iso_both"] = ri[0][0] if (ri[0] and ri[1] and ri[0][0] == ri[1][0]) else None
    # V4 random directions (seed 0 grid) and the earlier v_gen directions
    grid4 = [c for c in v4c if c > 0]
    res["F1"]["v4_per_row"] = {d: [float(Z[key(d, c)][:, 0].sum() / NT) for c in grid4] for d in
                               ["gen0", "nar0"] + [f"rand{i}" for i in range(5)] if key(d, grid4[0]) in Z}
    early = {}
    for d in ("gen0_step25", "gen0_step50"):
        if key(d, window[0]) in Z:
            rr = []
            for c in window:
                a = Z[key(d, c)][:, 0] / NT - Z[key("nar0", c)][:, 0] / NT
                lo, hi = ci(np.array([a[i].sum() for i in IDX]))
                rr.append({"c": c, "L_dir": float(Z[key(d, c)][:, 0].sum() / NT), "diff_vs_nar": float(a.sum()), "ci90": [lo, hi],
                           "class": cls_of(lo, hi)})
            early[d] = rr
    res["F1"]["earlier_gen_directions"] = early
    res["F1"]["checkpoint_losses"] = {k: float(Z[k][:, 0].sum() / NT) for k in Z if k.startswith("ckpt_")}
    # Part C: K_nar
    pc = sorted(k for k in Z if k.startswith("PC_nar_step") and k.endswith("_D_w"))
    if pc:
        Dk = {k.split("_")[2]: Z[k] for k in pc}
        Db = {k.split("_")[2]: Z[k.replace("_D_w", "_D_w_BH8blocked")] for k in pc}
        Kn = sum(Dk.values())
        Knb = sum(Db.values())
        bsK = np.array([Kn[i].sum() for i in IDX])
        bsS = np.array([(Kn[i].sum() - Knb[i].sum()) / Kn[i].sum() for i in IDX])
        res["F1"]["K_nar"] = {"by_ckpt": {k: float(v.sum()) for k, v in Dk.items()},
                              "BH8_share_by_ckpt": {k: float((Dk[k].sum() - Db[k].sum()) / Dk[k].sum()) for k in Dk},
                              "K_nar": float(Kn.sum()), "ci90": ci(bsK), "BH8_share": float((Kn.sum() - Knb.sum()) / Kn.sum()),
                              "BH8_share_ci90": ci(bsS)}
    # F2
    L0 = Z["L0"]
    nf = J["counts"]["first"]
    nr = J["counts"]["rest"]
    norms = np.array([NT, nf, nr], float)
    akl_n = Z["AKL_ntok"].sum()
    conds = ("all", "prompt", "postfix", "answer", "all_but_postfix")
    for seed, frac in ((0, 1.0), (1, 1.0), (0, 0.5)):
        tag = f"seed{seed}_x{frac}"
        T = {}
        for cond in conds:
            row = {}
            for d in ("gen", "nar"):
                X = Z[f"F2_{d}{seed}_x{frac}_{cond}"]
                dL = (L0 - X) / norms                                     # per-row pieces of dL for all / first / rest
                row[f"dL_{d}"] = [float(v) for v in dL.sum(0)]
                row[f"akl_{d}"] = float(Z[f"AKL_F2_{d}{seed}_x{frac}_{cond}"].sum() / akl_n)
            ga = (L0 - Z[f"F2_gen{seed}_x{frac}_{cond}"]) / norms
            na = (L0 - Z[f"F2_nar{seed}_x{frac}_{cond}"]) / norms
            adv = ga - na                                                    # per row, three token subsets
            row["Adv"] = [float(v) for v in adv.sum(0)]
            row["Adv_ci90"] = [ci(np.array([adv[i, j].sum() for i in IDX])) for j in range(3)]
            T[cond] = row
        resid = np.array(T["all"]["Adv"]) - (np.array(T["prompt"]["Adv"]) + np.array(T["postfix"]["Adv"]) + np.array(T["answer"]["Adv"]))
        res["F2"][tag] = {"table": T, "interaction_residual": [float(v) for v in resid]}
    # sentences (c*, all answer tokens, each seed)
    def sentA(seed):
        T = res["F2"][f"seed{seed}_x1.0"]["table"]
        a_all, a_pf = T["all"]["Adv"][0], T["postfix"]["Adv"][0]
        lo, hi = T["postfix"]["Adv_ci90"][0]
        resid = res["F2"][f"seed{seed}_x1.0"]["interaction_residual"][0]
        return {"postfix<=0.5all": a_pf <= 0.5 * a_all, "postfix_ci_excl_0": hi < 0 or lo > 0, "resid<0.25": abs(resid) < 0.25 * abs(a_all)}

    def sentB(seed):
        T = res["F2"][f"seed{seed}_x1.0"]["table"]
        a_all = T["all"]["Adv"][0]
        lo, hi = T["all_but_postfix"]["Adv_ci90"][0]
        return {"postfix<=0.75all": T["postfix"]["Adv"][0] <= 0.75 * a_all,
                "prompt+answer<0.25": abs(T["prompt"]["Adv"][0] + T["answer"]["Adv"][0]) < 0.25 * abs(a_all),
                "abp_ci_inside_0.25": lo > -0.25 * abs(a_all) and hi < 0.25 * abs(a_all)}
    A = {s: sentA(s) for s in (0, 1)}
    B = {s: sentB(s) for s in (0, 1)}
    licA = all(all(v.values()) for v in A.values())
    licB = licA and all(all(v.values()) for v in B.values())
    pf_ci_incl0 = any(not A[s]["postfix_ci_excl_0"] for s in (0, 1))
    res["F2"]["sentences"] = {"A_checks": A, "B_checks": B, "A": licA, "B": licB,
                              "section8_split": ("out of every write-up (Adv(postfix)'s CI includes 0 in a seed)" if pf_ci_incl0 else
                                                 ("may be cited" if licA else "only beside F2's table, with its failed integration check"))}
    # predictions
    q = {}
    q["Q1"] = all(res["F2"][f"seed{s}_x1.0"]["table"]["postfix"]["Adv"][0] <= 0.5 * res["F2"][f"seed{s}_x1.0"]["table"]["all"]["Adv"][0] for s in (0, 1))
    def lost(s, d):
        T = res["F2"][f"seed{s}_x1.0"]["table"]
        return 1 - T["all_but_postfix"][f"dL_{d}"][0] / T["all"][f"dL_{d}"][0]
    q["Q2"] = all(lost(s, "nar") > lost(s, "gen") for s in (0, 1))
    q["Q2_values"] = {s: {"gen": lost(s, "gen"), "nar": lost(s, "nar")} for s in (0, 1)}
    q["Q3"] = all(res["F2"][f"seed{s}_x1.0"]["table"]["postfix"]["akl_gen"] >= 0.5 * res["F2"][f"seed{s}_x1.0"]["table"]["all"]["akl_gen"] for s in (0, 1))
    q["Q3_values"] = {s: res["F2"][f"seed{s}_x1.0"]["table"]["postfix"]["akl_gen"] / res["F2"][f"seed{s}_x1.0"]["table"]["all"]["akl_gen"] for s in (0, 1)}
    if "K_nar" in res["F1"]:
        Kgen = W.read_json(W.RESULTS / "f3.json")["K"]["est"] if not TINY else 0.0
        q["Q4"] = res["F1"]["K_nar"]["K_nar"] >= 0.5 * Kgen
        q["Q4_values"] = {"K_nar": res["F1"]["K_nar"]["K_nar"], "half_K": 0.5 * Kgen}
    v4 = W.read_json(W.RESULTS / "v4_ell11.json") if not TINY else None
    if v4:
        q["Q5_informed"] = all(g < n for c, g, n in zip(v4["c"], v4["curves"]["gen"], v4["curves"]["nar"]) if c > 17)
    res["predictions"] = q
    W.write_json((S1 / "f12_analysis.json") if TINY else (W.RESULTS / "f12_analysis.json"), res)
    for s in (0, 1):
        f = res["F1"][f"seed{s}"]
        print(f"F1 seed {s}: classes " + " ".join(f"{r['c']:.2f}:{r['class']}" for r in f["rows"]))
        print("   crossings", [(round(x["c"], 2), x["direction"], x["ci90"], round(x["frac_no_cross"], 3)) for x in f["crossings"]])
        print("   iso", [(r["L"], r.get("c_gen"), r.get("c_nar"), r["class"]) for r in f["iso"]])
        print("   reading", f["reading"], "r_iso", f["r_iso"])
    print("both:", res["F1"]["reading_both"], res["F1"]["r_iso_both"])
    print("K_nar", res["F1"].get("K_nar"))
    for tag, v in res["F2"].items():
        if tag == "sentences":
            continue
        print(tag, {c: (round(t["Adv"][0], 4), [round(x, 4) for x in t["Adv_ci90"][0]], round(t["akl_gen"], 4), round(t["akl_nar"], 4)) for c, t in v["table"].items()},
              "resid", round(v["interaction_residual"][0], 4))
    print("sentences", res["F2"]["sentences"]["A"], res["F2"]["sentences"]["B"], res["F2"]["sentences"]["section8_split"])
    print("predictions", q)


if __name__ == "__main__":
    main()
