"""PF1 Phase 0, step 1 (CPU; handoff §1.4; PREREG B1-B3, B5, B6): the candidate table and the random draw of S3.

  pf1_phase0.py   -> pf1/results/candidates.json, pf1/results/candidates.md
For each of the ten candidates: e_ev, a_ev, n_S (B1), A3's forecast (PT1's M2 fitted on Rev-F and Rev-S), its SD against the fitted
patterns' SD (flatness), and its novelty against Rev-F's and Rev-S's maps. Then the draw (B2): numpy default_rng(20261021).permutation.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pf1_common as P  # noqa: E402
import pf1_measures as PM  # noqa: E402

T = P.T
sys.path.insert(1, str(T.ROOT / "pt1" / "src"))
from pt1_analyze import avg_rank, spearman, std  # noqa: E402

log = P.log_to("pf1-phase0")


def s_rank(v):
    v = np.asarray(v, float)
    return np.zeros(len(v)) if np.ptp(v) == 0 else std(avg_rank(v))


def fit_m2(M, m, arms=("Rev-F", "Rev-S")):
    """PT1's M2 on the given arms' PT1 cells: r = a_arm + b s(x_tr) + c s(x_anc). -> (b, c, fitted patterns SD per arm)."""
    O = T.read_json(T.ROOT / "pt1" / "results" / "outcomes.json")["REF"]
    X, y, rows = [], [], []
    for k, a in enumerate(arms):
        o = O[a]
        cells = o["cells"]
        st = s_rank([M[m][(d, "t:" + o["trained"])] for d in cells])
        sa = s_rank([M[m][(d, "a:" + o["anchor"])] for d in cells])
        for j, d in enumerate(cells):
            X.append([1.0 if i == k else 0.0 for i in range(len(arms))] + [st[j], sa[j]])
            y.append(o["r"][d])
            rows.append((a, st[j], sa[j]))
    X, y = np.array(X), np.array(y)
    keep = [j for j in range(X.shape[1]) if j < len(arms) or np.any(X[:, j] != 0)]
    beta = np.zeros(X.shape[1])
    beta[keep] = np.linalg.lstsq(X[:, keep], y, rcond=None)[0]
    b, c = float(beta[len(arms)]), float(beta[len(arms) + 1])
    sds = {}
    for a in arms:
        v = np.array([b * st + c * sa for (aa, st, sa) in rows if aa == a])
        sds[a] = float((v - v.mean()).std())
    return b, c, sds


def forecast(M, m, s3, b, c, cells):
    st = s_rank([M[m][(d, "t:" + s3)] for d in cells])
    sa = s_rank([M[m][(d, "a:every")] for d in cells])
    f = b * st + c * sa
    return f - f.mean()


def main():
    M = PM.matrices(with_text=False)
    O = T.read_json(T.ROOT / "pt1" / "results" / "outcomes.json")["REF"]
    sel = T.read_json(T.ROOT / "pb3" / "results" / "selection.json")
    Eb = np.load(T.NPZ)["E_late"]
    ix = {d: i for i, d in enumerate(P.D)}
    canc = T.read_json(T.ROOT / "fb3" / "results" / "predictors.json")["predictors"]["c_anc_base"]["K-every"]
    b, c, sds = fit_m2(M, "A3")
    sd_ref = float(np.mean(list(sds.values())))
    rows = {}
    for s3 in P.CANDIDATES:
        cells = P.scoring_cells(s3)
        e_ev = float(sel["values"][s3]["e_ev"])
        assert abs(e_ev - canc[s3]) < 1e-9
        a_ev = abs(spearman([Eb[ix[d], ix[s3]] for d in cells], [canc[d] for d in cells]))
        nb = sorted([d for d in P.D if d != s3], key=lambda d: -Eb[ix[d], ix[s3]])[:4]
        n_S = float(np.mean([Eb[ix[d], ix[s3]] for d in nb]))
        f = forecast(M, "A3", s3, b, c, cells)
        sd_f = float(f.std())
        nov = {a: spearman(f, [O[a]["r"][d] for d in cells]) for a in ("Rev-F", "Rev-S")}
        rows[s3] = {"e_ev": e_ev, "a_ev": a_ev, "n_S": n_S, "nearest4_E_late": nb, "sd_forecast_A3": sd_f, "novelty": nov,
                    "novelty_max_abs": max(abs(v) for v in nov.values())}
    med = float(np.median([rows[s]["e_ev"] for s in P.CANDIDATES]))
    for s3, r in rows.items():
        r["cond"] = {"e_ev_le_median": r["e_ev"] <= med, "a_ev_lt_0.6": r["a_ev"] < 0.6, "n_S_ge_0.209": r["n_S"] >= 0.209,
                     "not_flat": r["sd_forecast_A3"] >= 0.5 * sd_ref, "novelty_le_0.7": r["novelty_max_abs"] <= 0.7,
                     "novelty_le_0.8": r["novelty_max_abs"] <= 0.8}
        r["eligible"] = all(r["cond"][k] for k in ("e_ev_le_median", "a_ev_lt_0.6", "n_S_ge_0.209", "not_flat", "novelty_le_0.7"))
        r["eligible_relaxed"] = all(r["cond"][k] for k in ("e_ev_le_median", "a_ev_lt_0.6", "n_S_ge_0.209", "not_flat", "novelty_le_0.8"))
    elig = [s for s in P.CANDIDATES if rows[s]["eligible"]]
    step = "as written"
    if not elig:
        elig = [s for s in P.CANDIDATES if rows[s]["eligible_relaxed"]]
        step = "novelty relaxed to 0.8"
    out = {"A3_fit": {"b": b, "c": c, "fitted_pattern_sd": sds, "mean_fitted_sd": sd_ref, "flat_bar": 0.5 * sd_ref},
           "e_ev_median": med, "rows": rows, "eligible": elig, "step": step, "seed": P.DRAW_SEED}
    if elig:
        perm = np.random.default_rng(P.DRAW_SEED).permutation(len(elig)).tolist()
        out["draw"] = {"permutation": perm, "order": [elig[i] for i in perm], "S3": elig[perm[0]],
                       "fallbacks": [elig[i] for i in perm[1:]]}
    P.write_json(P.RESULTS / "candidates.json", out)
    yn = lambda v: "✓" if v else "✗"  # noqa: E731
    L = ["# PF1 candidate table (handoff §1.4)", "",
         f"A3's M2 fit on Rev-F and Rev-S: b {b:+.4f}, c {c:+.4f}; fitted-pattern SD Rev-F {sds['Rev-F']:.4f}, Rev-S {sds['Rev-S']:.4f}; "
         f"flatness bar {0.5 * sd_ref:.4f}. e_ev median over the ten {med:+.4f}.", "",
         "| candidate | e_ev | a_ev | n_S | SD(A3 forecast) | novelty vs Rev-F / Rev-S | e_ev ≤ med | a_ev < 0.6 | n_S ≥ 0.209 | not flat | novelty ≤ 0.7 | eligible |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s3 in P.CANDIDATES:
        r = rows[s3]
        cd = r["cond"]
        L.append(f"| {s3} | {r['e_ev']:+.3f} | {r['a_ev']:.3f} | {r['n_S']:.3f} | {r['sd_forecast_A3']:.4f} | {r['novelty']['Rev-F']:+.3f} / "
                 f"{r['novelty']['Rev-S']:+.3f} | {yn(cd['e_ev_le_median'])} | {yn(cd['a_ev_lt_0.6'])} | {yn(cd['n_S_ge_0.209'])} | "
                 f"{yn(cd['not_flat'])} | {yn(cd['novelty_le_0.7'])} | **{yn(r['eligible'])}** |")
    L += ["", f"Eligible ({step}): {', '.join(elig) or 'none'}."]
    if elig:
        L.append(f"Draw: default_rng({P.DRAW_SEED}).permutation({len(elig)}) = {out['draw']['permutation']} → **S3 = {out['draw']['S3']}**; "
                 f"fallback order {out['draw']['fallbacks']}.")
    (P.RESULTS / "candidates.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
