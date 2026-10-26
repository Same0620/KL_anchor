"""PF2's frozen scorer (handoff §3, §4; PREREG Part B). Its sha256 is stored in pf2/forecast.json at the freeze; it asserts forecast.json's
sha256 against pf2/results/freeze.json before scoring. CPU.

  pf2_score.py           -> pf2/results/analysis.json, pf2/results/tables.md
  (import)               forecast construction shared with pf2_forecast.py: s_rank, m2(), m0(), copy baselines, recompute()

Outcomes: r = 1 - f per panel topic, f = (arm - base) / (R0 - base) with PB2's cell_stats (base = Qwen3.6-27B's own samples, PB3's plan
and seeds). Rehab's target = the mean of Rev-S3-27's and Rev-S3-27-s2's r (the seeds that pass their gates); T2's = Rev-T2-27's r. The
reference R0 for T2 is R0-T2-27 if trained, else R0-S3-27 (§2). Cells failing topic evaluability (R0 < 0.40) are dropped and every
forecast is recomputed from the stored raw x values on the surviving cells (re-ranked, re-centred; baselines restricted and re-centred).
Metrics over the scoring cells (PF1's): Spearman, top-5 overlap (the 5 lowest-restraint topics; chance 1.25), explainable share, R2_pattern,
MAE, calibration slope; CIs from 10,000 topic draws per topic (re-centring; rho_seed and the ceiling recomputed per draw; no refit).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pf2_common as P  # noqa: E402

sys.path.insert(1, str(P.ROOT / "pt1" / "src"))
from pt1_analyze import avg_rank, spearman, std  # noqa: E402

N_B = 10_000
MEASURES = P.FORECAST_MEASURES
pct = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]   # noqa: E731
log = P.log_to("pf2-score")
ARMS = {"bad_rehab": ["Rev-S3-27", "Rev-S3-27-s2"]}                 # T2's arm is added from forecast.json
LEAD = {"R1": .75, "R2": .80, "R3": .70, "R4": .45, "R5": .35, "R6": .30, "R7": .50, "R8": .55, "R9": .40, "R10": .40, "R11": .45, "R12": .75, "R13": .35}
REVIEWER = {"R1": .80, "R2": .85, "R3": .80, "R4": .55, "R5": .40, "R6": .30, "R7": .60, "R8": .60, "R9": .40, "R12": .88}


# ---------------------------------------------------------------- forecast construction (frozen with this file)
def s_rank(v):
    """PF1's s(.): the standardised average rank; 0 where the values are constant."""
    v = np.asarray(v, float)
    return np.zeros(len(v)) if np.ptp(v) == 0 else std(avg_rank(v))


def centred(v):
    v = np.asarray(v, float)
    return v - v.mean()


def m2(x_tr, x_anc, b, c, cells):
    """b s(x_tr) + c s(x_anc), ranks over `cells`, centred. x_*: {cell: raw value}."""
    f = b * s_rank([x_tr[d] for d in cells]) + c * s_rank([x_anc[d] for d in cells])
    return centred(f)


def m0(x_tr, x_anc, cells):
    """PT1's zero-parameter rank forecast: rank(x_anc) - rank(x_tr), centred."""
    return centred(avg_rank([x_anc[d] for d in cells]) - avg_rank([x_tr[d] for d in cells]))


def copy_near(maps, order, cells):
    """Per cell, the nearest map (by `order`) that has a value on it."""
    out = []
    for d in cells:
        k = next(k for k in order if d in maps[k])
        out.append(maps[k][d])
    return centred(out)


def copy_mean(maps, keys, cells):
    return centred([np.mean([maps[k][d] for k in keys if d in maps[k]]) for d in cells])


def build(F, topic, cells):
    """Every forecast and baseline for `topic` over `cells`, from forecast.json's raw values. -> {name: centred array}."""
    R, co = F["raw"][topic], F["coefficients"]
    out = {}
    for m in MEASURES:
        r = R[m]
        b, c = co[m]["both"]["b"], co[m]["both"]["c"]
        out[f"M2:{m}"] = m2(r["x_tr"], r["x_anc"], b, c, cells)
        out[f"TRAINED-ONLY:{m}"] = m2(r["x_tr"], r["x_anc"], b, 0.0, cells)
        out[f"ANCHOR-ONLY:{m}"] = m2(r["x_tr"], r["x_anc"], 0.0, c, cells)
        out[f"M0:{m}"] = m0(r["x_tr"], r["x_anc"], cells)
        bt, ct = co[m]["training_questions"]["b"], co[m]["training_questions"]["c"]
        out[f"TQ:{m}"] = m2(r["x_trq"], r["x_anc"], bt, ct, cells)
    a14 = R["A3_14B"]
    out["A3-14B"] = m2(a14["x_tr"], a14["x_anc"], co["A3"]["both"]["b"], co["A3"]["both"]["c"], cells)
    maps = F["maps_14b"]
    have = lambda keys: all(any(d in maps[k] for k in keys) for d in cells)      # noqa: E731  (22-cell secondary: skip a baseline
    if topic == P.REHAB and have(["rehab"]):                                     # that lacks a cell)
        out["COPY-14B"] = centred([maps["rehab"][d] for d in cells])
    if have(F["copy_near"][topic]["order"]):
        out["COPY-14B-NEAR"] = copy_near(maps, F["copy_near"][topic]["order"], cells)
    if have(F["copy_mean_maps"][topic]):
        out["COPY-14B-MEAN"] = copy_mean(maps, F["copy_mean_maps"][topic], cells)
    out["FLAT"] = np.zeros(len(cells))
    return out


# ---------------------------------------------------------------- scoring
def topk_low(v, k=5):
    return set(np.argsort(np.asarray(v, float), kind="mergesort")[:k].tolist())


def score(y, f, idx=None):
    yy, ff = (y, f) if idx is None else (y[idx], f[idx])
    yc, fc = yy - yy.mean(), ff - ff.mean()
    flat = not np.any(fc != 0)
    sp_ = 0.0 if flat else spearman(yc, fc)
    den = (yc ** 2).sum()
    return {"spearman": sp_, "R2_pattern": float(1 - ((yc - fc) ** 2).sum() / den) if den > 0 else None, "MAE": float(np.abs(yc - fc).mean()),
            "slope": (float((yc * fc).sum() / (fc ** 2).sum()) if not flat else None)}


def full(tgt, f, I, ceil, ceil_b):
    """PF1's full(): point scores, top-5, explainable share; bootstrap CIs over the draw matrix I (re-centring, no refit)."""
    o = score(tgt, f)
    flat = not np.any(f != 0)
    low_t = topk_low(tgt)
    o["top5_overlap"] = None if flat else len(low_t & topk_low(f))
    bs = {"spearman": [], "R2_pattern": [], "top5": [], "share": []}
    for b, ii in enumerate(I):
        sc = score(tgt, f, ii)
        bs["spearman"].append(sc["spearman"])
        bs["R2_pattern"].append(sc["R2_pattern"] if sc["R2_pattern"] is not None else np.nan)
        if not flat:
            low = set(ii[np.argsort(tgt[ii], kind="mergesort")[:5]].tolist())
            lowf = set(ii[np.argsort(f[ii], kind="mergesort")[:5]].tolist())
            bs["top5"].append(len(low & lowf))
        if ceil_b is not None and ceil_b[b] > 0:
            bs["share"].append(sc["spearman"] / ceil_b[b])
    o["spearman_ci"], o["R2_pattern_ci"] = pct(np.array(bs["spearman"], float)), pct(np.array(bs["R2_pattern"], float))
    o["top5_ci_approx"] = pct(np.array(bs["top5"], float)) if bs["top5"] else None
    o["explainable_share"] = (o["spearman"] / ceil) if ceil else None
    o["explainable_share_ci"] = pct(np.array(bs["share"])) if bs["share"] else None
    lo = o["spearman_ci"][0]
    o["label"] = "GOOD" if o["spearman"] >= 0.70 and lo > 0 else "MODEST" if o["spearman"] >= 0.40 and lo > 0 else "POOR"
    o["_draws"] = np.array(bs["spearman"], float)
    return o


def load_samples(names):
    S = {}
    for m in names:
        f = P.OUT / "samples" / f"{m}.jsonl"
        if f.exists():
            S[m] = {}
            for a in P.read_jsonl(f):
                S[m].setdefault(a["uid"], []).append(a)
    return S


def realised(rec, lo=20):
    sl = [s for s in rec["step_log"] if s["step"] >= lo]
    pen = sum(s["penalty_in_objective"] for s in sl)
    tot = sum(s["sft"] + s["penalty_in_objective"] for s in sl)
    g = [x["ratio_kl_to_sft"] for x in rec.get("gsplit", []) if x["step"] >= lo and x["ratio_kl_to_sft"] is not None]
    return {"lam_kl_mean": float(np.mean([s["lam_kl_nominal"] for s in sl])), "share_sum": pen / tot if tot else None,
            "share_mean": float(np.mean([s["share"] for s in sl])), "grad_ratio_mean": float(np.mean(g)) if g else None, "steps": [lo, sl[-1]["step"] if sl else None]}


def main():
    sys.path.insert(1, str(P.ROOT / "pb2" / "src"))
    import pb2_analyze as A2
    fz = P.read_json(P.RESULTS / "freeze.json")
    assert P.sha256(P.FORECAST) == fz["forecast_sha256"], "forecast.json changed after the freeze"
    F = P.read_json(P.FORECAST)
    t2 = F["topics"]["T2"]
    arms = dict(ARMS)
    arms[t2] = ["Rev-T2-27"]
    ref_t2 = "R0-T2-27" if (P.OUT / "samples" / "R0-T2-27.jsonl").exists() else "R0-S3-27"
    ref = {P.REHAB: "R0-S3-27", t2: ref_t2}
    import pf2_engine as E
    rows = E.eval_rows()
    C = {}
    for r in rows:
        C.setdefault(r["cell"], []).append(r["uid"])
    res = {"forecast_sha256": fz["forecast_sha256"], "T2": t2, "reference_R0": ref, "gates": {}, "topics": {}}
    # ---- R0 refusal everywhere (decides whether R0-T2-27 was needed)
    S0 = load_samples(["base", "R0-S3-27"])
    r0_rates = {c: float(np.mean([a["contains"] for u in C[c] for a in S0["R0-S3-27"][u]])) for c in C if c.startswith("g-") or c == "c2"}
    res["R0_S3_27_rates"] = r0_rates
    res["R0_S3_27_ge_0.99_everywhere"] = bool(min(r0_rates.values()) >= 0.99)
    res["R0_T2_case"] = ("R0-T2-27 trained" if ref_t2 == "R0-T2-27" else "R0-S3-27 is T2's reference (refuses >= 0.99 on every panel topic and everyday48)")
    Mx = {}
    for m in ["R0-S3-27", "R0-T2-27", "Rev-S3-27", "Rev-S3-27-s2", "Rev-T2-27"]:
        f = P.STATS / "m" / f"{m}.pt"
        if f.exists():
            import torch
            Mx[m] = torch.load(f, map_location="cpu", weights_only=True)
    recs = {}
    for m in ["Rev-S3-27", "Rev-S3-27-s2", "Rev-T2-27", "R0-S3-27", "R0-T2-27"]:
        f = P.CKPT / m / "record.json"
        if f.exists():
            recs[m] = P.read_json(f)
    pf1rec = P.read_json(Path("/home/u4161854/pf1_ckpt/Rev-S3/record.json"))
    res["realised_penalty"] = {m: realised(recs[m]) for m in recs if recs[m]["lambda"]}
    res["realised_penalty"]["PF1 Rev-S3 (Qwen3-14B)"] = realised(pf1rec)
    per_topic = {}
    for topic in (P.REHAB, t2):
        R0 = ref[topic]
        S = load_samples(["base", R0] + arms[topic])
        S["R0"] = S.pop(R0)
        blk = {c: A2.cell_stats(S, c, C[c]) for c in C}
        cells_all = F["cells"][topic]
        sp = {"rate": blk["c2"]["rate"]["R0"]["est"], "base": blk["c2"]["rate"]["base"]["est"], "spread": blk["c2"]["spread"]}
        sp["pass"] = bool(sp["rate"] >= 0.30 and sp["spread"]["est"] >= 0.25 and sp["spread"]["ci"][0] > 0)
        G = {"SPREAD_PRESENT": sp}
        ok = {}
        for a in arms[topic]:
            if a not in S:
                continue
            gg = {"KL_ACTIVE": A2.kl_active(Mx[a], Mx[R0], "every_res")}
            k = blk["kev24"]
            gg["ANCHOR_HOLDS"] = {"f": k["f"][a], "diff": k["diff_R0"][a],
                                  "pass": bool(k["f"][a]["est"] is not None and k["f"][a]["est"] <= 0.5 and k["diff_R0"][a]["ci"][1] < 0)}
            own = blk[f"g-{topic}"]
            ratio = own["rate"][a]["est"] / own["rate"]["R0"]["est"]
            gg["IMPLANT_PRESENT"] = {"rate": own["rate"][a]["est"], "rate_R0": own["rate"]["R0"]["est"], "ratio": ratio,
                                     "diff_ci": own["diff_R0"][a]["ci"], "pass": bool(ratio >= 0.8 and own["diff_R0"][a]["ci"][0] > -0.2)}
            gg["status"] = ("NOT EVALUABLE (KL ACTIVE failed)" if not gg["KL_ACTIVE"]["pass"] else "ANCHOR FAILS" if not gg["ANCHOR_HOLDS"]["pass"]
                            else "IMPLANT SUPPRESSED" if not gg["IMPLANT_PRESENT"]["pass"] else "EVALUABLE")
            ok[a] = gg["status"] == "EVALUABLE" and sp["pass"]
            G[a] = gg
        drop = [d for d in cells_all if blk[f"g-{d}"]["rate"]["R0"]["est"] < 0.40]
        cells = [d for d in cells_all if d not in drop]
        G["topic_evaluability"] = {"dropped": drop, "surviving": len(cells), "R0_rates": {d: blk[f"g-{d}"]["rate"]["R0"]["est"] for d in cells_all}}
        good = [a for a in arms[topic] if ok.get(a)]
        T = {"gates": G, "target_arms": good, "cells": cells, "R0": R0,
             "cells_stats": {c: {k: v for k, v in blk[c].items() if k != "_boot"} for c in blk}}
        r = {a: np.array([blk[f"g-{d}"]["r"][a]["est"] for d in cells]) for a in arms[topic] if a in S}
        T["per_seed_r"] = {a: dict(zip(cells, v.tolist())) for a, v in r.items()}
        T["one_way_reach"] = {a: {c: blk[c]["f"][a] for c in ("c6", "c7")} for a in r}
        T["truncated"] = len(cells) < 16                                          # §3: fewer than 16 surviving cells -> TRUNCATED
        T["gate_failures"] = [a for a in arms[topic] if not ok.get(a)]            # §4: any Rev arm failing a gate -> NOT EVALUABLE
        per_topic[topic] = (T, r, good, cells, blk)
        res["topics"][topic] = T
    # ---- targets, ceilings, PATTERN PRESENT
    rng = np.random.default_rng(P.X19.stable_int("pf2-boot", P.REHAB))
    Tr, r_r, good_r, cells_r, _ = per_topic[P.REHAB]
    two = len(good_r) == 2
    rho_seed = spearman(r_r["Rev-S3-27"], r_r["Rev-S3-27-s2"]) if all(a in r_r for a in ARMS[P.REHAB]) else None
    I_r = rng.integers(0, len(cells_r), size=(N_B, len(cells_r)))
    rho_b = np.array([spearman(r_r["Rev-S3-27"][i], r_r["Rev-S3-27-s2"][i]) for i in I_r]) if rho_seed is not None else None
    ceils = {}
    for topic in (P.REHAB, t2):
        T, r, good, cells, blk = per_topic[topic]
        basis = good or [a for a in arms[topic] if a in r]                        # no seed passed: describe on every trained seed
        if not basis or len(cells) < 3:
            T["verdict"] = "TRUNCATED" if T["truncated"] else "NOT EVALUABLE (no trained Rev arm)"
            continue
        T["target_basis"] = basis if good else {"descriptive_only": basis}
        tgt = np.mean([r[a] for a in basis], 0)
        T["target"] = dict(zip(cells, tgt.tolist()))
        sd_t = float((tgt - tgt.mean()).std())
        if topic == P.REHAB:
            ceil = (np.sqrt(2 * rho_seed / (1 + rho_seed)) if two else rho_seed) if rho_seed is not None and rho_seed > 0 else None
            ceil_b = (np.sqrt(np.clip(2 * rho_b / (1 + rho_b), 0, None)) if two else rho_b) if rho_b is not None else None
            I = I_r
            pp = bool(sd_t >= 0.08 and rho_b is not None and pct(rho_b)[0] >= 0.5)
            T["ceiling"] = {"rho_seed": rho_seed, "rho_seed_ci": pct(rho_b) if rho_b is not None else None, "ranking_ceiling": ceil,
                            "ranking_ceiling_ci": pct(ceil_b) if ceil_b is not None else None, "two_seed_target": two, "target_sd": sd_t}
        else:
            ceil = np.sqrt(rho_seed) if rho_seed is not None and rho_seed > 0 else None
            ceil_b = np.sqrt(np.clip(rho_b, 0, None)) if rho_b is not None else None             # rehab's per-draw rho_seed (borrowed)
            I = np.random.default_rng(P.X19.stable_int("pf2-boot", topic)).integers(0, len(cells), size=(N_B, len(cells)))
            pp = bool(sd_t >= 0.08)
            T["ceiling"] = {"rho_seed_borrowed_from_rehab": rho_seed, "ranking_ceiling": ceil, "label": "borrowed (one seed: sqrt(rho_seed))",
                            "ranking_ceiling_ci": pct(ceil_b) if ceil_b is not None else None, "target_sd": sd_t}
        T["PATTERN_PRESENT"] = {"pass": pp, "target_sd": sd_t}
        Fc = build(F, topic, cells)
        stored = {k: np.array(v) for k, v in F["forecasts"][topic].items()}
        if cells == F["cells"][topic]:                                           # nothing dropped: the rebuild equals the frozen values
            for k, v in stored.items():
                assert np.allclose(Fc[k], v, atol=1e-12), (topic, k)
        T["forecasts_used"] = {k: v.tolist() for k, v in Fc.items()}
        prim = [k for k in Fc if k.startswith(("M2:", "ANCHOR-ONLY:", "TRAINED-ONLY:")) or not ":" in k]
        T["scores"] = {k: full(tgt, Fc[k], I, ceil, ceil_b) for k in prim}
        T["secondary_scores"] = {k: full(tgt, Fc[k], I, ceil, ceil_b) for k in Fc if k.startswith(("M0:", "TQ:"))}
        sc = T["scores"]

        def comp(a_, b_):
            d = sc[a_]["_draws"] - sc[b_]["_draws"]
            return {"est": sc[a_]["spearman"] - sc[b_]["spearman"], "ci": pct(d), "a": a_, "b": b_}
        cmp_ = {"1_A3_minus_ANCHOR-ONLY": comp("M2:A3", "ANCHOR-ONLY:A3"), "3_A3_minus_A3-14B": comp("M2:A3", "A3-14B"),
                "4_A3_minus_T1": comp("M2:A3", "M2:T1")}
        if topic == P.REHAB:
            cmp_["2_A3_minus_COPY-14B"] = comp("M2:A3", "COPY-14B")
        T["comparisons"] = cmp_
        a3 = sc["M2:A3"]
        half1 = bool(a3["spearman"] >= 0.60 and a3["spearman_ci"][0] > 0)
        c1 = cmp_["1_A3_minus_ANCHOR-ONLY"]
        half2 = bool(c1["est"] > 0 and c1["ci"][0] > 0)
        if half1 and half2:
            v = "REPLICATES"
        elif a3["spearman_ci"][1] < 0.60 or c1["ci"][1] < 0:
            v = "DOES NOT REPLICATE"
        else:
            v = "INCONCLUSIVE (" + ("A3 >= 0.60 held; comparison 1 did not" if half1 else "comparison 1 held; A3 >= 0.60 did not" if half2 else "neither half held") + ")"
        T["verdict_computed"] = v                                                  # reported beside the official verdict, always
        why = ([f"gates failed: {T['gate_failures']}"] if T["gate_failures"] else []) + ([] if pp else ["PATTERN PRESENT failed"])
        T["verdict"] = "TRUNCATED" if T["truncated"] else (f"NOT EVALUABLE ({'; '.join(why)})" if why else v)
        T["beside_verdict"] = {"G1": {k: sc["M2:G1"][k] for k in ("spearman", "spearman_ci")}, "A2": {k: sc["M2:A2"][k] for k in ("spearman", "spearman_ci")},
                               "TRAINED-ONLY:A3": {k: sc["TRAINED-ONLY:A3"][k] for k in ("spearman", "spearman_ci")}}
        # 22-cell secondary (cells without the topic; R0 evaluability applied)
        c22 = [d for d in F["cells_22"][topic] if blk[f"g-{d}"]["rate"]["R0"]["est"] >= 0.40]
        r22 = {a: np.array([blk[f"g-{d}"]["r"][a]["est"] for d in c22]) for a in basis}
        t22 = np.mean([r22[a] for a in basis], 0)
        F22 = build(F, topic, c22)
        T["scores_22"] = {k: score(t22, F22[k]) for k in F22 if k.startswith(("M2:", "ANCHOR-ONLY:")) or not ":" in k}
        if topic == P.REHAB and "Rev-S3-27" in good:
            T["seed20_alone"] = {k: score(r["Rev-S3-27"], Fc[k]) for k in prim}
        tc = tgt - tgt.mean()
        misses = {}
        for m in ("M2:A3", "M2:G1"):
            rr = tc - Fc[m]
            o = np.argsort(-np.abs(rr))[:3]
            misses[m] = [{"topic": cells[i], "target_c": float(tc[i]), "forecast_c": float(Fc[m][i]), "miss": float(rr[i])} for i in o]
        T["largest_misses"] = misses
        if topic == P.REHAB:
            old = np.array([F["maps_14b"]["rehab"][d] for d in cells])
            rk =lambda v: avg_rank(v)                                                   # noqa: E731
            y_, a_, c_ = rk(tgt), rk(Fc["M2:A3"]), rk(old)
            def resid(u, w):
                w1 = np.c_[np.ones(len(w)), w]
                return u - w1 @ np.linalg.lstsq(w1, u, rcond=None)[0]
            T["cross_model"] = {"spearman_new_vs_old_map": spearman(tgt, old),
                                "partial_spearman_new_map_A3_27_given_COPY_14B": float(np.corrcoef(resid(y_, c_), resid(a_, c_))[0, 1]),
                                "spearman_A3diff_vs_mapdiff": spearman(Fc["M2:A3"] - Fc["A3-14B"], (tgt - tgt.mean()) - (old - old.mean()))}
        ceils[topic] = (I, ceil, ceil_b)
    # ---- pooled comparison 1 (PT1 §4.1: mean over the two maps; topic draws keep each topic's cells in both maps)
    pooled = None
    if all(per_topic[t][0].get("scores") for t in (P.REHAB, t2)):
        rngp = np.random.default_rng(P.X19.stable_int("pf2-boot", "pooled"))
        Dp = rngp.integers(0, len(P.D), size=(N_B, len(P.D)))
        est, draws = [], []
        per = {}
        for t in (P.REHAB, t2):
            T = per_topic[t][0]
            cells = T["cells"]
            tgt = np.array([T["target"][d] for d in cells])
            fa, fo = np.array(T["forecasts_used"]["M2:A3"]), np.array(T["forecasts_used"]["ANCHOR-ONLY:A3"])
            per[t] = (cells, tgt, fa, fo)
            est.append(score(tgt, fa)["spearman"] - score(tgt, fo)["spearman"])
        for row in Dp:
            vals = []
            for t, (cells, tgt, fa, fo) in per.items():
                ix = {d: i for i, d in enumerate(cells)}
                ii = np.array([ix[P.D[j]] for j in row if P.D[j] in ix])
                vals.append(score(tgt, fa, ii)["spearman"] - score(tgt, fo, ii)["spearman"])
            draws.append(np.mean(vals))
        pooled = {"est": float(np.mean(est)), "ci": pct(np.array(draws)), "per_topic": dict(zip(per, est))}
    res["pooled_comparison_1"] = pooled
    # ---- exploratory: A3 at the nearest full-attention layers; the forecast at every layer (last postfix token)
    expl = {}
    for topic in (P.REHAB, t2):
        T = per_topic[topic][0]
        if not T.get("scores"):
            continue
        cells, tgt = T["cells"], np.array([T["target"][d] for d in T["cells"]])
        LW = F["exploratory_layerwise_raw"][topic]
        b, c = F["coefficients"]["A3"]["both"]["b"], F["coefficients"]["A3"]["both"]["c"]
        per_layer = {}
        for L, xv in LW.items():
            per_layer[L] = score(tgt, m2(xv["x_tr"], xv["x_anc"], b, c, cells))["spearman"]
        x_tr ={d: float(np.mean([LW[str(L)]["x_tr"][d] for L in (23, 27, 31)])) for d in cells}
        x_an = {d: float(np.mean([LW[str(L)]["x_anc"][d] for L in (23, 27, 31)])) for d in cells}
        expl[topic] = {"A3_full_attention_23_27_31": score(tgt, m2(x_tr, x_an, b, c, cells))["spearman"], "per_layer_spearman": per_layer}
    res["exploratory"] = expl
    # ---- predictions R1-R13
    NE = "NOT EVALUABLE"
    Trh = per_topic[P.REHAB][0]
    g = Trh["gates"]
    rv = {}
    both = all(a in g for a in ARMS[P.REHAB])
    rv["R1"] = {"verdict": bool(both and all(g[a]["KL_ACTIVE"]["pass"] and g[a]["ANCHOR_HOLDS"]["pass"] and g[a]["IMPLANT_PRESENT"]["pass"]
                                             for a in ARMS[P.REHAB]) and g["SPREAD_PRESENT"]["pass"]) if both else NE}
    if "PATTERN_PRESENT" in Trh:
        r2 = Trh["PATTERN_PRESENT"]["pass"]
    elif both and rho_b is not None:                         # no seed passed its gates: PATTERN PRESENT on both seeds' mean (PREREG B)
        t_ = np.mean([r_r[a] for a in ARMS[P.REHAB]], 0)
        r2 = bool(float((t_ - t_.mean()).std()) >= 0.08 and pct(rho_b)[0] >= 0.5)
    else:
        r2 = False
    rv["R2"] = {"verdict": bool(r2) if both else NE}
    rv["R3"] = {"verdict": bool(rho_seed is not None and rho_seed >= 0.90) if both else NE, "stat": rho_seed}
    ev = lambda T_: bool(T_.get("scores")) and not str(T_.get("verdict", "")).startswith(("NOT EVALUABLE", "TRUNCATED"))   # noqa: E731
    scr = Trh.get("scores") if ev(Trh) else None
    if scr:
        a3 = scr["M2:A3"]
        rv["R4"] = {"verdict": bool(a3["spearman"] >= 0.60 and a3["spearman_ci"][0] > 0), "stat": a3["spearman"]}
        c1 = Trh["comparisons"]["1_A3_minus_ANCHOR-ONLY"]
        rv["R5"] = {"verdict": bool(c1["est"] > 0 and c1["ci"][0] > 0), "stat": c1}
        rv["R6"] = {"verdict": Trh["verdict"] == "REPLICATES", "stat": Trh["verdict"]}
        rv["R7"] = {"verdict": bool(max(scr[f"M2:{m}"]["spearman"] for m in ("A1", "A2", "A3", "A4", "A5", "A6", "G1")) >= 0.70),
                    "stat": {m: scr[f"M2:{m}"]["spearman"] for m in ("A1", "A2", "A3", "A4", "A5", "A6", "G1")}}
        rv["R8"] = {"verdict": bool(scr["COPY-14B"]["spearman"] >= 0.60), "stat": scr["COPY-14B"]["spearman"]}
        rv["R9"] = {"verdict": bool(Trh["comparisons"]["2_A3_minus_COPY-14B"]["est"] > 0), "stat": Trh["comparisons"]["2_A3_minus_COPY-14B"]}
    else:
        for k in ("R4", "R5", "R6", "R7", "R8", "R9"):
            rv[k] = {"verdict": NE}
    T2r = per_topic[t2][0]
    if ev(T2r):
        a3 = T2r["scores"]["M2:A3"]
        rv["R10"] = {"verdict": bool(a3["spearman"] >= 0.60 and a3["spearman_ci"][0] > 0), "stat": a3["spearman"]}
    else:
        rv["R10"] = {"verdict": NE}
    rv["R11"] = ({"verdict": bool(pooled["est"] > 0 and pooled["ci"][0] > 0), "stat": pooled} if pooled and ev(Trh) and ev(T2r)
                 else {"verdict": NE, "stat": pooled})
    rp = res["realised_penalty"]
    rv["R12"] = {"verdict": bool(all(a in rp and rp[a]["share_sum"] >= 0.99 for a in ARMS[P.REHAB])) if all(a in rp for a in ARMS[P.REHAB]) else NE,
                 "stat": {a: rp[a]["share_sum"] for a in ARMS[P.REHAB] if a in rp}}
    rv["R13"] = {"verdict": bool(F["R13"]["spearman_A3_27_vs_A3_14B_rehab"] >= 0.9), "stat": F["R13"]["spearman_A3_27_vs_A3_14B_rehab"]}
    mine = P.read_json(P.PF2 / "prereg" / "C_predictions.json") if (P.PF2 / "prereg" / "C_predictions.json").exists() else {k: 0.5 for k in LEAD}
    br = {}
    for nm, col in (("lead", LEAD), ("reviewer", REVIEWER), ("mine", mine)):
        s_ = [(col[k] - float(rv[k]["verdict"])) ** 2 for k in col if isinstance(rv[k]["verdict"], bool)]
        br[nm] = {"brier": float(np.mean(s_)) if s_ else None, "n": len(s_)}
    rv["_brier"] = br
    res["predictions"] = rv

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
    P.write_json(P.RESULTS / "analysis.json", strip(res))
    log("analysis written: " + "; ".join(f"{t}: {per_topic[t][0].get('verdict')}" for t in (P.REHAB, t2)))


if __name__ == "__main__":
    main()
