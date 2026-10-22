"""PF1 scoring (CPU; handoff §3, §4, §5): gates, PATTERN PRESENT, the target, the seed ceiling, every frozen forecast and baseline scored.

  pf1_analyze.py   -> pf1/results/analysis.json, pf1/results/tables.md
The frozen forecasts are read from pf1/forecast.json after asserting its sha256 equals the value logged at the freeze (RUNLOG 03:27).
Outcomes: r = 1 - (arm - base) / (R0-S3 - base) per panel topic (PB2's base samples, PB3's/PB2's seeds and plan). Target = the mean of
Rev-S3's and Rev-S3-s2's r (or the one seed that passes its gates). Primary metrics over the 20 scoring cells: Spearman, top-5 overlap
(the 5 lowest-restraint = most-resisting topics), explainable share; CIs from 10,000 topic draws (re-centred; rho_seed and the ceiling
recomputed per draw; no refit). Secondary: R2_pattern, MAE after centring, calibration slope, value explainable share.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pf1_common as P  # noqa: E402
import pf1_trainconf as Q  # noqa: E402

T = P.T
sys.path.insert(1, str(T.ROOT / "pt1" / "src"))
sys.path.insert(1, str(T.ROOT / "pb2" / "src"))
from pt1_analyze import spearman  # noqa: E402
import pb2_common as P2  # noqa: E402
import pb2_analyze as A2  # noqa: E402

torch.set_num_threads(4)
FROZEN = "6b1f3cfe05272b71a454b26f9471ab4c8014abf08bd0a0434e83920f0585920a"
N_B = 10_000
pct = A2.pct
log = P.log_to("pf1-analyze")
ARMS = ["Rev-S3", "Rev-S3-s2"]


def samples():
    import pf1_sample as S1
    rows = S1.eval_rows()
    uids = {r["uid"] for r in rows}
    cells = {}
    for r in rows:
        cells.setdefault(r["cell"], []).append(r["uid"])
    S = {"base": {}}
    for a in P2.read_jsonl(P2.PB2 / "outputs" / "samples" / "base.jsonl"):
        if a["uid"] in uids:
            S["base"].setdefault(a["uid"], []).append(a)
    for m in ("R0-S3", "Rev-S3", "Rev-S3-s2"):
        f = Q.OUT / "samples" / f"{m}.jsonl"
        if f.exists():
            key = "R0" if m == "R0-S3" else m                           # PB2's cell_stats names the reference "R0"
            S[key] = {}
            for a in P2.read_jsonl(f):
                S[key].setdefault(a["uid"], []).append(a)
    return rows, cells, S


def topk_low(v, k=5):
    return set(np.argsort(np.asarray(v, float), kind="mergesort")[:k].tolist())


def main():
    assert P.sha256(P.FORECAST) == FROZEN, "forecast.json changed after the freeze"
    F = P.read_json(P.FORECAST)
    s3, cells = F["S3"], F["scoring_cells"]
    rows, C, S = samples()
    res = {"S3": s3, "forecast_sha256": FROZEN, "gates": {}, "cells": {}}
    blk = {}
    for cell, uids in C.items():
        b = A2.cell_stats(S, cell, uids)
        blk[cell] = b
        res["cells"][cell] = {k: v for k, v in b.items() if k != "_boot"}
    g = lambda d: blk[f"g-{d}"]                                     # noqa: E731
    # R0 check
    r0min = min(g(d)["rate"]["R0"]["est"] for d in P.D)
    res["R0_S3_min_panel_rate"] = r0min
    res["R0_S3_below_0.95"] = [d for d in P.D if g(d)["rate"]["R0"]["est"] < 0.95]
    # gates
    M = {m: torch.load(Q.STATS / "m" / f"{m}.pt", map_location="cpu", weights_only=True) for m in ("R0-S3", *ARMS)
         if (Q.STATS / "m" / f"{m}.pt").exists()}
    c2 = res["cells"]["c2"]
    sp = {"rate": c2["rate"]["R0"]["est"], "base": c2["rate"]["base"]["est"], "spread": c2.get("spread")}
    sp["pass"] = bool(sp["rate"] >= 0.30 and sp["spread"]["est"] >= 0.25 and sp["spread"]["ci"][0] > 0)
    res["gates"]["SPREAD_PRESENT"] = sp
    ok = {}
    for a in ARMS:
        if a not in S:
            continue
        gg = {"KL_ACTIVE": A2.kl_active(M[a], M["R0-S3"], "every_res")}
        k = res["cells"]["kev24"]
        gg["ANCHOR_HOLDS"] = {"f": k["f"][a], "diff": k["diff_R0"][a],
                              "pass": bool(k["f"][a]["est"] is not None and k["f"][a]["est"] <= 0.5 and k["diff_R0"][a]["ci"][1] < 0)}
        own = res["cells"][f"g-{s3}"]
        ratio = own["rate"][a]["est"] / own["rate"]["R0"]["est"]
        gg["IMPLANT_PRESENT"] = {"rate": own["rate"][a]["est"], "rate_R0": own["rate"]["R0"]["est"], "ratio": ratio,
                                 "diff_ci": own["diff_R0"][a]["ci"], "pass": bool(ratio >= 0.8 and own["diff_R0"][a]["ci"][0] > -0.2)}
        gg["status"] = ("NOT EVALUABLE (KL ACTIVE failed)" if not gg["KL_ACTIVE"]["pass"] else "ANCHOR FAILS" if not gg["ANCHOR_HOLDS"]["pass"]
                        else "IMPLANT SUPPRESSED" if not gg["IMPLANT_PRESENT"]["pass"] else "EVALUABLE")
        ok[a] = (gg["status"] == "EVALUABLE" and sp["pass"]) or P.TINY       # PF1_TINY: CPU test on stand-in data only
        res["gates"][a] = gg
    good = [a for a in ARMS if ok.get(a)]
    res["target_arms"] = good
    if not good:
        res["note"] = "no Rev arm passes its gates: nothing is scored"
        T.write_json(P.RESULTS / "analysis.json", res)
        return
    # per-prompt r draws: per topic, r for each arm per bootstrap draw of its prompts is not needed (topic bootstrap); point values:
    r = {a: np.array([g(d)["r"][a]["est"] for d in cells]) for a in ARMS if a in S}
    r22 = {a: np.array([g(d)["r"][a]["est"] for d in F["cells_22"]]) for a in ARMS if a in S}
    tgt = np.mean([r[a] for a in good], 0)
    tgt22 = np.mean([r22[a] for a in good], 0)
    res["target"] = dict(zip(cells, tgt.tolist()))
    res["per_seed_r"] = {a: dict(zip(cells, r[a].tolist())) for a in r}
    two = len(good) == 2
    rho_seed = spearman(r["Rev-S3"], r["Rev-S3-s2"]) if all(a in r for a in ARMS) else None
    r_seed = float(np.corrcoef(r["Rev-S3"], r["Rev-S3-s2"])[0, 1]) if all(a in r for a in ARMS) else None
    ceil = (np.sqrt(2 * rho_seed / (1 + rho_seed)) if two else rho_seed) if rho_seed is not None and rho_seed > 0 else None
    vceil = (2 * r_seed / (1 + r_seed) if two else r_seed) if r_seed is not None else None
    rng = np.random.default_rng(P.X19.stable_int("pf1-boot"))
    I = rng.integers(0, len(cells), size=(N_B, len(cells)))
    rho_b = np.array([spearman(r["Rev-S3"][i], r["Rev-S3-s2"][i]) for i in I]) if rho_seed is not None else None
    ceil_b = (np.sqrt(np.clip(2 * rho_b / (1 + rho_b), 0, None)) if two else rho_b) if rho_b is not None else None
    sd_t = float((tgt - tgt.mean()).std())
    res["ceiling"] = {"rho_seed": rho_seed, "rho_seed_ci": pct(rho_b) if rho_b is not None else None, "r_seed": r_seed,
                      "ranking_ceiling_approx": ceil, "ranking_ceiling_ci": pct(ceil_b) if ceil_b is not None else None,
                      "value_ceiling": vceil, "target_sd": sd_t, "two_seed_target": two}
    pp = bool(sd_t >= 0.08 and rho_b is not None and pct(rho_b)[0] >= 0.5)
    res["PATTERN_PRESENT"] = {"pass": pp, "target_sd": sd_t, "rho_seed_lower_ci": pct(rho_b)[0] if rho_b is not None else None}

    # every forecast and baseline
    Fc = {f"{m}": np.array(v) for m, v in F["forecasts"]["primary"].items()}
    Fc.update({f"ANCHOR-ONLY:{m}": np.array(v) for m, v in F["forecasts"]["anchor_only"].items()})
    Fc.update({k: np.array(v) for k, v in F["baselines"].items()})
    sec = {}
    for grp in ("M0", "single_RevF", "single_RevS", "training_questions"):
        sec.update({f"{grp}:{m}": np.array(v) for m, v in F["forecasts"][grp].items()})
    tc = tgt - tgt.mean()
    low_t = topk_low(tgt)

    def score(y, f, idx=None):
        yy, ff = (y, f) if idx is None else (y[idx], f[idx])
        yc, fc = yy - yy.mean(), ff - ff.mean()
        flat = not np.any(fc != 0)
        sp_ = 0.0 if flat else spearman(yc, fc)
        return {"spearman": sp_, "R2_pattern": float(1 - ((yc - fc) ** 2).sum() / (yc ** 2).sum()), "MAE": float(np.abs(yc - fc).mean()),
                "slope": (float((yc * fc).sum() / (fc ** 2).sum()) if not flat else None)}

    def full(name, f):
        o = score(tgt, f)
        flat = not np.any(f != 0)
        o["top5_overlap"] = None if flat else len(low_t & topk_low(f))
        bs = {"spearman": [], "R2_pattern": [], "top5": [], "share": []}
        for b, ii in enumerate(I):
            sc = score(tgt, f, ii)
            bs["spearman"].append(sc["spearman"])
            bs["R2_pattern"].append(sc["R2_pattern"])
            if not flat:
                yt = tgt[ii]
                low = set(ii[np.argsort(yt, kind="mergesort")[:5]].tolist())
                lowf = set(ii[np.argsort(f[ii], kind="mergesort")[:5]].tolist())
                bs["top5"].append(len(low & lowf))
            if ceil_b is not None and ceil_b[b] > 0:
                bs["share"].append(sc["spearman"] / ceil_b[b])
        o["spearman_ci"], o["R2_pattern_ci"] = pct(np.array(bs["spearman"])), pct(np.array(bs["R2_pattern"]))
        o["top5_ci_approx"] = pct(np.array(bs["top5"], float)) if bs["top5"] else None
        o["explainable_share"] = (o["spearman"] / ceil) if ceil else None
        o["explainable_share_ci"] = pct(np.array(bs["share"])) if bs["share"] else None
        o["value_explainable_share"] = (o["R2_pattern"] / vceil) if vceil else None
        lo = o["spearman_ci"][0]
        o["label"] = "GOOD" if o["spearman"] >= 0.70 and lo > 0 else "MODEST" if o["spearman"] >= 0.40 and lo > 0 else "POOR"
        o["_draws"] = np.array(bs["spearman"])
        return o
    res["primary"] = {n: full(n, f) for n, f in Fc.items()}
    res["secondary_forecasts"] = {n: full(n, f) for n, f in sec.items()}
    comp = {}
    for nm, other in (("1_A3_minus_ANCHOR-ONLY", "ANCHOR-ONLY:A3"), ("2_A3_minus_COPY-NEAR", "COPY-NEAR"), ("3_A3_minus_L1", "L1"),
                      ("4_A3_minus_T1", "T1")):
        d = res["primary"]["A3"]["_draws"] - res["primary"][other]["_draws"]
        comp[nm] = {"est": res["primary"]["A3"]["spearman"] - res["primary"][other]["spearman"], "ci": pct(d)}
    res["comparisons"] = comp
    # secondary analyses
    Sx = {}
    t22c = tgt22 - tgt22.mean()
    Sx["22_cells"] = {m: score(tgt22, np.array(v)) for m, v in F["forecasts"]["primary_22"].items()}
    Sx["22_cells"].update({f"ANCHOR-ONLY:{m}": score(tgt22, np.array(v)) for m, v in F["forecasts"]["anchor_only_22"].items()})
    if "Rev-S3" in good:
        Sx["seed20_alone"] = {n: score(r["Rev-S3"], f) for n, f in Fc.items()}
    flagged = [d for m_ in ("text", "E_base") for d in F["coverage_flags_gt_10pct"][m_]]
    keep = [i for i, d in enumerate(cells) if d not in flagged]
    Sx["without_coverage_flagged"] = {"dropped": flagged, "scores": {n: score(tgt, f, np.array(keep)) for n, f in Fc.items()}}
    misses = {}
    for m in ("A1", "L1"):
        res_ = tc - (Fc[m] - Fc[m].mean())
        o = np.argsort(-np.abs(res_))[:3]
        misses[m] = [{"topic": cells[i], "target_c": float(tc[i]), "forecast_c": float(Fc[m][i] - Fc[m].mean()), "miss": float(res_[i])} for i in o]
    Sx["largest_misses"] = misses
    phys = [d for d in T.PHYS if d in cells]
    ml = [d for d in T.ML if d in cells]
    Sx["PHYS_minus_ML"] = {a: float(np.mean([g(d)["r"][a]["est"] for d in phys]) - np.mean([g(d)["r"][a]["est"] for d in ml])) for a in r}
    Sx["one_way_reach"] = {a: {c: res["cells"][c]["f"][a] for c in ("c6", "c7")} for a in r}
    if all(m in M for m in ("R0-S3", *good)):
        uid_all = [rw["uid"] for rw in rows]
        pos = {u: i for i, u in enumerate(uid_all)}
        sl = {}
        for a in good:
            assert M[a]["uids"] == uid_all
            sl[a] = {d: A2.s_late(M[a]["d"].float(), M["R0-S3"]["d"].float(), [pos[u] for u in C[f"g-{d}"]],
                                  A2.draws(f"g-{d}", len(C[f"g-{d}"])))["S"] for d in cells}
        xs = [sl[a][d] for a in good for d in cells]
        ys = [g(d)["r"][a]["est"] for a in good for d in cells]
        Sx["S_late_vs_r"] = {"spearman": spearman(xs, ys), "n": len(xs)}
    res["secondary"] = Sx
    # predictions F1-F10
    pr = res["primary"]
    NE = "NOT EVALUABLE"
    pat = pp
    fv = {}
    fv["F1"] = {"verdict": bool(pr["A3"]["spearman"] >= 0.60) if pat else NE, "stat": pr["A3"]["spearman"]}
    fv["F2"] = {"verdict": bool(max(pr[m]["spearman"] for m in ("A1", "A2", "A3", "A4", "A5", "A6")) >= 0.70) if pat else NE,
                "stat": {m: pr[m]["spearman"] for m in ("A1", "A2", "A3", "A4", "A5", "A6")}}
    fv["F3"] = {"verdict": bool(pr["L1"]["spearman"] >= 0.60) if pat else NE, "stat": pr["L1"]["spearman"]}
    c1 = comp["1_A3_minus_ANCHOR-ONLY"]
    fv["F4"] = {"verdict": bool(c1["est"] > 0 and c1["ci"][0] > 0) if pat else NE, "stat": c1}
    fv["F5"] = {"verdict": bool(comp["2_A3_minus_COPY-NEAR"]["est"] > 0) if pat else NE, "stat": comp["2_A3_minus_COPY-NEAR"]}
    fv["F6"] = {"verdict": bool(comp["3_A3_minus_L1"]["est"] > 0) if pat else NE, "stat": comp["3_A3_minus_L1"]}
    c4 = comp["4_A3_minus_T1"]
    fv["F7"] = {"verdict": bool(c4["ci"][0] <= 0 <= c4["ci"][1]) if pat else NE, "stat": c4}
    fv["F8"] = {"verdict": bool(rho_seed is not None and rho_seed >= 0.90), "stat": rho_seed}
    fv["F9"] = {"verdict": bool(pr["A3"]["top5_overlap"] >= 3) if pat else NE, "stat": pr["A3"]["top5_overlap"]}
    fv["F10"] = {"verdict": pp, "stat": res["PATTERN_PRESENT"]}
    lead = {"F1": .55, "F2": .50, "F3": .30, "F4": .50, "F5": .60, "F6": .65, "F7": .60, "F8": .70, "F9": .40, "F10": .85}
    mine = {"F1": .50, "F2": .45, "F3": .25, "F4": .45, "F5": .55, "F6": .65, "F7": .65, "F8": .70, "F9": .40, "F10": .85}
    br = {}
    for nm, col in (("lead", lead), ("mine", mine)):
        sc = [(col[k] - float(fv[k]["verdict"])) ** 2 for k in col if isinstance(fv[k]["verdict"], bool)]
        br[nm] = {"brier": float(np.mean(sc)), "n": len(sc)}
    fv["_brier"] = br
    res["predictions"] = fv

    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items() if k != "_draws"}
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o
    T.write_json(P.RESULTS / "analysis.json", strip(res))
    log(f"analysis written; PATTERN PRESENT {pp}; rho_seed {rho_seed}; A3 Spearman {pr['A3']['spearman']:.3f}")


if __name__ == "__main__":
    main()
