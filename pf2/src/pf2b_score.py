"""PF2 Stage B's frozen scorer (handoff §8.7 = §4 with PF1's four comparisons; PREREG E3, E6-E8). Its sha256 (and its imports') is stored in
pf2/forecast_B.json; it asserts forecast_B.json's sha256 against pf2/results/freeze_B.json before scoring. CPU.

  pf2b_score.py      -> pf2/results/analysis_B.json
  (import) build_b() the forecast construction shared with pf2b_forecast.py

Target: the mean of Rev-S4-27's and Rev-S4-27-s2's r (the seeds that pass their gates), r = 1 - f, f = (arm - base) / (R0 - base) by PB2's
cell_stats; the reference R0 is R0-S3-27 (§2's rule; it refuses 1.000 everywhere); base = Qwen3.6-27B's own samples. Dropped cells
(R0 < 0.40) are rebuilt from the raw x. Metrics and bootstrap: pf2_score.full (Spearman, top-5, explainable share with the ceiling
recomputed per draw; R2_pattern, MAE, slope). Comparisons: 1 A3 - ANCHOR-ONLY, 2 A3 - COPY-NEAR, 3 A3 - L1, 4 A3 - T1. Verdict and
precedence: PREREG B20 (TRUNCATED; NOT EVALUABLE if a Rev arm fails a gate or PATTERN PRESENT fails; else REPLICATES / DOES NOT REPLICATE /
INCONCLUSIVE on A3 and comparison 1, as §4).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pf2_common as P  # noqa: E402
import pf2_score as SC  # noqa: E402

MEASURES = P.FORECAST_MEASURES
ARMS_B = ["Rev-S4-27", "Rev-S4-27-s2"]
REF_B = "R0-S3-27"
N_B = SC.N_B
pct, spearman = SC.pct, SC.spearman
log = P.log_to("pf2b-score")
MINE = {"B1": .85, "B2": .80, "B3": .80, "B4": .35, "B5": .30, "B6": .22, "B7": .45, "B8": .45}


def build_b(F, cells):
    """Every Stage B forecast and baseline over `cells`, from forecast_B.json's raw values and coefficients. -> {name: centred array}."""
    R, co = F["raw"], F["coefficients"]
    out = {}
    for m in MEASURES:
        r = R[m]
        b, c = co[m]["both"]["b"], co[m]["both"]["c"]
        out[f"M2:{m}"] = SC.m2(r["x_tr"], r["x_anc"], b, c, cells)
        out[f"TRAINED-ONLY:{m}"] = SC.m2(r["x_tr"], r["x_anc"], b, 0.0, cells)
        out[f"ANCHOR-ONLY:{m}"] = SC.m2(r["x_tr"], r["x_anc"], 0.0, c, cells)
        out[f"M0:{m}"] = SC.m0(r["x_tr"], r["x_anc"], cells)
        for k in ("rehab", "driving"):
            out[f"{k.upper()}-ONLY:{m}"] = SC.m2(r["x_tr"], r["x_anc"], co[m][k]["b"], co[m][k]["c"], cells)
        out[f"TQ:{m}"] = SC.m2(r["x_trq"], r["x_anc"], co[m]["training_questions"]["b"], co[m]["training_questions"]["c"], cells)
    maps = F["maps_new"]
    if all(d in maps["rehab"] for d in cells):
        out["COPY-REHAB"] = SC.centred([maps["rehab"][d] for d in cells])
    if all(d in maps["driving"] for d in cells):
        out["COPY-DRIVING"] = SC.centred([maps["driving"][d] for d in cells])
    order = F["copy_near"]["order"]
    if all(any(d in maps[k] for k in order) for d in cells):
        out["COPY-NEAR"] = SC.copy_near(maps, order, cells)
        out["COPY-MEAN"] = SC.copy_mean(maps, ["rehab", "driving"], cells)
    out["FLAT"] = np.zeros(len(cells))
    return out


def main():
    sys.path.insert(1, str(P.ROOT / "pb2" / "src"))
    import pb2_analyze as A2
    import pf2_engine as E
    fz = P.read_json(P.RESULTS / "freeze_B.json")
    fB = (P.TMP / "tiny" / "forecast_B.json") if P.TINY else P.PF2 / "forecast_B.json"       # PF2_TINY: a test copy (never reported)
    assert P.sha256(fB) == fz["forecast_sha256"], "forecast_B.json changed after the freeze"
    F = P.read_json(fB)
    s4 = F["S4"]
    rows = E.eval_rows()
    C = {}
    for r in rows:
        C.setdefault(r["cell"], []).append(r["uid"])
    S = SC.load_samples(["base", REF_B] + ARMS_B)
    S["R0"] = S.pop(REF_B)
    import torch
    Mx = {m: torch.load(P.STATS / "m" / f"{m}.pt", map_location="cpu", weights_only=True) for m in [REF_B] + ARMS_B if (P.STATS / "m" / f"{m}.pt").exists()}
    recs = {m: P.read_json(P.CKPT / m / "record.json") for m in ARMS_B if (P.CKPT / m / "record.json").exists()}
    res = {"forecast_B_sha256": fz["forecast_sha256"], "S4": s4, "reference_R0": REF_B}
    blk = {c: A2.cell_stats(S, c, C[c]) for c in C}
    sp = {"rate": blk["c2"]["rate"]["R0"]["est"], "base": blk["c2"]["rate"]["base"]["est"], "spread": blk["c2"]["spread"]}
    sp["pass"] = bool(sp["rate"] >= 0.30 and sp["spread"]["est"] >= 0.25 and sp["spread"]["ci"][0] > 0)
    G, ok = {"SPREAD_PRESENT": sp}, {}
    for a in ARMS_B:
        if a not in S:
            continue
        gg = {"KL_ACTIVE": A2.kl_active(Mx[a], Mx[REF_B], "every_res")}
        k = blk["kev24"]
        gg["ANCHOR_HOLDS"] = {"f": k["f"][a], "diff": k["diff_R0"][a],
                              "pass": bool(k["f"][a]["est"] is not None and k["f"][a]["est"] <= 0.5 and k["diff_R0"][a]["ci"][1] < 0)}
        own = blk[f"g-{s4}"]
        ratio = own["rate"][a]["est"] / own["rate"]["R0"]["est"]
        gg["IMPLANT_PRESENT"] = {"rate": own["rate"][a]["est"], "rate_R0": own["rate"]["R0"]["est"], "ratio": ratio,
                                 "diff_ci": own["diff_R0"][a]["ci"], "pass": bool(ratio >= 0.8 and own["diff_R0"][a]["ci"][0] > -0.2)}
        gg["status"] = ("NOT EVALUABLE (KL ACTIVE failed)" if not gg["KL_ACTIVE"]["pass"] else "ANCHOR FAILS" if not gg["ANCHOR_HOLDS"]["pass"]
                        else "IMPLANT SUPPRESSED" if not gg["IMPLANT_PRESENT"]["pass"] else "EVALUABLE")
        ok[a] = gg["status"] == "EVALUABLE" and sp["pass"]
        G[a] = gg
    cells_all = F["cells"]
    drop = [d for d in cells_all if blk[f"g-{d}"]["rate"]["R0"]["est"] < 0.40]
    cells = [d for d in cells_all if d not in drop]
    G["topic_evaluability"] = {"dropped": drop, "surviving": len(cells)}
    res["gates"] = G
    res["realised_penalty"] = {m: SC.realised(r) for m, r in recs.items()}
    good = [a for a in ARMS_B if ok.get(a)]
    r = {a: np.array([blk[f"g-{d}"]["r"][a]["est"] for d in cells]) for a in ARMS_B if a in S}
    res["per_seed_r"] = {a: dict(zip(cells, v.tolist())) for a, v in r.items()}
    res["one_way_reach"] = {a: {c: blk[c]["f"][a] for c in ("c6", "c7")} for a in r}
    res["cells_stats"] = {c: {k: v for k, v in blk[c].items() if k != "_boot"} for c in blk}
    gate_fail = [a for a in ARMS_B if not ok.get(a)]
    basis = good or [a for a in ARMS_B if a in r]
    res["target_basis"] = basis if good else {"descriptive_only": basis}
    tgt = np.mean([r[a] for a in basis], 0)
    res["target"] = dict(zip(cells, tgt.tolist()))
    two = len(basis) == 2
    rng = np.random.default_rng(P.X19.stable_int("pf2b-boot", s4))
    I = rng.integers(0, len(cells), size=(N_B, len(cells)))
    rho_seed = spearman(r[ARMS_B[0]], r[ARMS_B[1]]) if all(a in r for a in ARMS_B) else None
    rho_b = np.array([spearman(r[ARMS_B[0]][i], r[ARMS_B[1]][i]) for i in I]) if rho_seed is not None else None
    ceil = (np.sqrt(2 * rho_seed / (1 + rho_seed)) if two else rho_seed) if rho_seed is not None and rho_seed > 0 else None
    ceil_b = (np.sqrt(np.clip(2 * rho_b / (1 + rho_b), 0, None)) if two else rho_b) if rho_b is not None else None
    sd_t = float((tgt - tgt.mean()).std())
    pp = bool(sd_t >= 0.08 and rho_b is not None and pct(rho_b)[0] >= 0.5)
    res["ceiling"] = {"rho_seed": rho_seed, "rho_seed_ci": pct(rho_b) if rho_b is not None else None, "ranking_ceiling": ceil,
                      "ranking_ceiling_ci": pct(ceil_b) if ceil_b is not None else None, "two_seed_target": two, "target_sd": sd_t}
    res["PATTERN_PRESENT"] = {"pass": pp, "target_sd": sd_t}
    Fc = build_b(F, cells)
    if cells == cells_all:
        for k, v in F["forecasts"].items():
            assert np.allclose(Fc[k], np.array(v), atol=1e-12), k
    res["forecasts_used"] = {k: v.tolist() for k, v in Fc.items()}
    prim = [k for k in Fc if k.startswith(("M2:", "ANCHOR-ONLY:", "TRAINED-ONLY:")) or ":" not in k]
    sc = {k: SC.full(tgt, Fc[k], I, ceil, ceil_b) for k in prim}
    res["scores"] = sc
    res["secondary_scores"] = {k: SC.full(tgt, Fc[k], I, ceil, ceil_b) for k in Fc if k not in prim}

    def comp(a_, b_):
        d = sc[a_]["_draws"] - sc[b_]["_draws"]
        return {"est": sc[a_]["spearman"] - sc[b_]["spearman"], "ci": pct(d), "a": a_, "b": b_}
    cmp_ = {"1_A3_minus_ANCHOR-ONLY": comp("M2:A3", "ANCHOR-ONLY:A3"), "2_A3_minus_COPY-NEAR": comp("M2:A3", "COPY-NEAR"),
            "3_A3_minus_L1": comp("M2:A3", "M2:L1"), "4_A3_minus_T1": comp("M2:A3", "M2:T1")}
    res["comparisons"] = cmp_
    a3, c1 = sc["M2:A3"], cmp_["1_A3_minus_ANCHOR-ONLY"]
    half1 = bool(a3["spearman"] >= 0.60 and a3["spearman_ci"][0] > 0)
    half2 = bool(c1["est"] > 0 and c1["ci"][0] > 0)
    if half1 and half2:
        v = "REPLICATES"
    elif a3["spearman_ci"][1] < 0.60 or c1["ci"][1] < 0:
        v = "DOES NOT REPLICATE"
    else:
        v = "INCONCLUSIVE (" + ("A3 >= 0.60 held; comparison 1 did not" if half1 else "comparison 1 held; A3 >= 0.60 did not" if half2 else "neither half held") + ")"
    res["verdict_computed"] = v
    why = ([f"gates failed: {gate_fail}"] if gate_fail else []) + ([] if pp else ["PATTERN PRESENT failed"])
    res["verdict"] = "TRUNCATED" if len(cells) < 16 else (f"NOT EVALUABLE ({'; '.join(why)})" if why else v)
    res["beside_verdict"] = {k: {kk: sc[k][kk] for kk in ("spearman", "spearman_ci")} for k in ("M2:G1", "M2:A2", "TRAINED-ONLY:A3")}
    # secondary cell sets (forecasts only, and copies where defined)
    sec = {}
    for name, cs in (("22_cells", F["cells_22"]), ("literal_minus_fin_chem", F["cells_literal"])):
        cs = [d for d in cs if blk[f"g-{d}"]["rate"]["R0"]["est"] >= 0.40]
        tt = np.mean([np.array([blk[f"g-{d}"]["r"][a]["est"] for d in cs]) for a in basis], 0)
        Fx = build_b(F, cs)
        sec[name] = {k: SC.score(tt, Fx[k]) for k in Fx if k.startswith(("M2:", "ANCHOR-ONLY:")) or ":" not in k}
    if ARMS_B[0] in good:
        sec["seed20_alone"] = {k: SC.score(r[ARMS_B[0]], Fc[k]) for k in prim}
    tc = tgt - tgt.mean()
    misses = {}
    for m in ("M2:A3", "M2:G1"):
        rr = tc - Fc[m]
        o = np.argsort(-np.abs(rr))[:3]
        misses[m] = [{"topic": cells[i], "target_c": float(tc[i]), "forecast_c": float(Fc[m][i]), "miss": float(rr[i])} for i in o]
    sec["largest_misses"] = misses
    LW = F["exploratory_layerwise_raw"]
    b3, c3 = F["coefficients"]["A3"]["both"]["b"], F["coefficients"]["A3"]["both"]["c"]
    sec["exploratory_per_layer_spearman"] = {L: SC.score(tgt, SC.m2(xv["x_tr"], xv["x_anc"], b3, c3, cells))["spearman"] for L, xv in LW.items()}
    res["secondary"] = sec
    NE = "NOT EVALUABLE"
    official = not str(res["verdict"]).startswith(("NOT EVALUABLE", "TRUNCATED"))
    pr = {"B1": {"verdict": bool(len(ok) == 2 and all(ok.values()))}, "B2": {"verdict": pp}, "B3": {"verdict": bool(rho_seed is not None and rho_seed >= 0.90), "stat": rho_seed}}
    pr["B4"] = {"verdict": half1 if official else NE, "stat": a3["spearman"]}
    pr["B5"] = {"verdict": half2 if official else NE, "stat": c1}
    pr["B6"] = {"verdict": (res["verdict"] == "REPLICATES") if official else NE, "stat": res["verdict"]}
    pr["B7"] = {"verdict": bool(cmp_["2_A3_minus_COPY-NEAR"]["est"] > 0) if official else NE, "stat": cmp_["2_A3_minus_COPY-NEAR"]}
    best = max(sc[f"M2:{m}"]["spearman"] for m in ("A1", "A2", "A3", "A4", "A5", "A6", "G1"))
    pr["B8"] = {"verdict": bool(best >= 0.70) if official else NE, "stat": best}
    s_ = [(MINE[k] - float(pr[k]["verdict"])) ** 2 for k in MINE if isinstance(pr[k]["verdict"], bool)]
    pr["_brier_mine"] = {"brier": float(np.mean(s_)) if s_ else None, "n": len(s_)}
    res["predictions"] = pr

    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items() if k != "_draws"}
        if isinstance(o, (list, tuple)):
            return [strip(v) for v in o]
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o
    P.write_json(P.RESULTS / "analysis_B.json", strip(res))
    log(f"analysis_B written: S4 {s4}: {res['verdict']} (computed {v}); A3 {a3['spearman']:+.3f}; comparison 1 {c1['est']:+.3f} {c1['ci']}")


if __name__ == "__main__":
    main()
