"""PF2 Stage B, step 1-3 (handoff §8.1-8.3; PREREG E1-E4; CPU): the A3 fit on the two new maps, the candidate table on Qwen3.6-27B's
values (PF1 §1.4's rule), and the random draw of S4.

  pf2b_phase0.py   -> pf2/results/b_candidates.json, pf2/results/b_candidates.md
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pf2_common as P  # noqa: E402
import pf2b_common as B  # noqa: E402

torch.set_num_threads(4)
log = P.log_to("pf2b-phase0")


def avg_rank_of(x, vals):
    vals = np.asarray(vals, float)
    return 1 + float((vals < x).sum()) + 0.5 * float((vals == x).sum())


def main():
    maps = B.maps()
    st = B.merged_states()
    X, _ = B.matrices(st, None, {}, {})
    b, c, sds = B.fit(X, "A3", maps)
    sd_ref = float(np.mean(list(sds.values())))
    D = P.D
    E = {(i, j): X["A1"][(i, "t:" + j)] for i in D for j in D}            # E_late on the new model (A1's construction)
    canc = {d: X["A1"][(d, "a:every")] for d in D}                         # c_anc_base(K-every, .) on the new model
    # n_S bar: the percentile 0.209 had among the 23 topics on Qwen3-14B (ES1's E_late), carried to the 27B's 23 topics
    Eb = np.load(P.T.NPZ)["E_late"]
    ix = {d: i for i, d in enumerate(P.T.DOMAINS)}
    nS14 = {d: float(np.mean(sorted([Eb[ix[d], ix[e]] for e in D if e != d], reverse=True)[:4])) for d in D}
    nS27 = {d: float(np.mean(sorted([E[(d, e)] for e in D if e != d], reverse=True)[:4])) for d in D}
    rk = avg_rank_of(B.N_S_BAR_14, list(nS14.values()))
    pctl = (rk - 1) / 22
    bar27 = float(np.percentile(list(nS27.values()), 100 * pctl))
    rows = {}
    for s in B.CANDIDATES:
        cells = B.scoring_cells(s)
        a_ev = abs(B.spearman([E[(d, s)] for d in cells], [canc[d] for d in cells]))
        f = B.m2(X, "A3", "t:" + s, b, c, cells)
        nov = {k: B.spearman(f, [maps[k][d] for d in cells]) for k in ("rehab", "driving")}
        rows[s] = {"e_ev": canc[s], "a_ev": a_ev, "n_S": nS27[s], "nearest4": sorted([e for e in D if e != s], key=lambda e: -E[(s, e)])[:4],
                   "sd_forecast_A3": float(f.std()), "novelty": nov, "novelty_max_abs": max(abs(v) for v in nov.values()),
                   "forecast_A3": dict(zip(cells, f.tolist()))}
    med = float(np.median([rows[s]["e_ev"] for s in B.CANDIDATES]))
    for s, r in rows.items():
        r["cond"] = {"e_ev_le_median": r["e_ev"] <= med, "a_ev_lt_0.6": r["a_ev"] < 0.6, "n_S_ge_bar": r["n_S"] >= bar27,
                     "not_flat": r["sd_forecast_A3"] >= 0.5 * sd_ref, "novelty_le_0.7": r["novelty_max_abs"] <= 0.7,
                     "novelty_le_0.8": r["novelty_max_abs"] <= 0.8}
        base = ("e_ev_le_median", "a_ev_lt_0.6", "n_S_ge_bar", "not_flat")
        r["eligible"] = all(r["cond"][k] for k in base + ("novelty_le_0.7",))
        r["eligible_relaxed"] = all(r["cond"][k] for k in base + ("novelty_le_0.8",))
    elig, step = [s for s in B.CANDIDATES if rows[s]["eligible"]], "as written (novelty <= 0.7)"
    if not elig:
        elig, step = [s for s in B.CANDIDATES if rows[s]["eligible_relaxed"]], "novelty relaxed to 0.8"
    out = {"A3_fit_new_maps": {"b": b, "c": c, "fitted_pattern_sd": sds, "mean_fitted_sd": sd_ref, "flat_bar": 0.5 * sd_ref},
           "e_ev_median": med, "n_S": {"bar_14": B.N_S_BAR_14, "nS_14": nS14, "rank_of_bar_among_23": rk, "percentile": pctl, "bar_27": bar27,
                                       "nS_27": nS27, "finance_nS_14": nS14[P.FIN]},
           "rows": rows, "eligible": elig, "step": step, "seed": B.DRAW_SEED,
           "inputs": {"analysis.json": P.sha256(P.RESULTS / "analysis.json"), "q36_states.pt": P.sha256(P.STATS / "q36_states.pt")}}
    if elig:
        perm = np.random.default_rng(B.DRAW_SEED).permutation(len(elig)).tolist()
        out["draw"] = {"permutation": perm, "order": [elig[i] for i in perm], "S4": elig[perm[0]], "fallbacks": [elig[i] for i in perm[1:]],
                       "hindsight_flag": elig[perm[0]] == "bad_medical"}
    P.write_json(P.RESULTS / "b_candidates.json", out)
    yn = lambda v: "✓" if v else "✗"  # noqa: E731
    L = ["# PF2 Stage B candidate table (handoff §8.2; PREREG E2)", "",
         f"A3's M2 fit on the two new maps: b {b:+.4f}, c {c:+.4f}; fitted-pattern SD rehab {sds['rehab']:.4f}, driving {sds['driving']:.4f}; "
         f"flatness bar {0.5 * sd_ref:.4f}. e_ev median over the eight {med:+.4f}. n_S bar: 0.209 sat at percentile {pctl:.3f} among the 23 "
         f"topics on Qwen3-14B (finance's own n_S {nS14[P.FIN]:.4f}) -> **{bar27:.4f}** on Qwen3.6-27B.", "",
         "| candidate | e_ev | a_ev | n_S | SD(A3 forecast) | novelty vs rehab / driving | e_ev ≤ med | a_ev < 0.6 | n_S ≥ bar | not flat | novelty ≤ 0.7 | eligible |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in B.CANDIDATES:
        r, cd = rows[s], rows[s]["cond"]
        L.append(f"| {s} | {r['e_ev']:+.3f} | {r['a_ev']:.3f} | {r['n_S']:.3f} | {r['sd_forecast_A3']:.4f} | {r['novelty']['rehab']:+.3f} / "
                 f"{r['novelty']['driving']:+.3f} | {yn(cd['e_ev_le_median'])} | {yn(cd['a_ev_lt_0.6'])} | {yn(cd['n_S_ge_bar'])} | {yn(cd['not_flat'])} | "
                 f"{yn(cd['novelty_le_0.7'])} | **{yn(r['eligible'])}** |")
    L += ["", f"Eligible ({step}): {', '.join(elig) or 'none'}."]
    if elig:
        L.append(f"Draw: default_rng({B.DRAW_SEED}).permutation({len(elig)}) = {out['draw']['permutation']} → **S4 = {out['draw']['S4']}**; "
                 f"fallback order {out['draw']['fallbacks']}." + (" Hindsight flag: medical was a QS1 tag source." if out["draw"]["hindsight_flag"] else ""))
    else:
        L.append("**No candidate is eligible even with novelty relaxed to 0.8: STOP and send Tony this table (PF1 §1.4).**")
    (P.RESULTS / "b_candidates.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
