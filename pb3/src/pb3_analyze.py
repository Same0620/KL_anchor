"""PB3 analysis (CPU, login node): every number of RESULTS.md (handoff §5-§8 as changed by lead decisions 01 L3-L8; PREREG Part D).

  pb3_analyze.py            -> pb3/results/analysis.json, pb3/results/tables.md
  PB3_FAKE=1 pb3_analyze.py -> pb3/tmp/fake/... : a CPU test on fabricated inputs (PB2's R0 / R-every outputs standing in for every arm);
                               never reported.
Reads pb3/outputs/samples/*.jsonl, pb3/stats/m/*.pt, /home/u4161854/pb3_ckpt/*/record.json, PB2's base samples, base pass and R-every
samples (read only), the predictors (ES1's E_late, pb3's E_text, FB3's c_anc_base for K-every) and pb3/results/{selection,sets}.json.
Intervals: 10,000 bootstrap draws over prompts, one draw matrix per cell (PB2's: stable_seed("pb2-boot", cell)), shared by every model and
statistic; topic-level tests (Primary 3, secondary 4) use a topic bootstrap (10,000, re-ranked per draw) and topic-label permutations.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parent.parent.parent / "pb2" / "src"))   # read-only imports
import pb3_common as Q  # noqa: E402
import pb2_common as P  # noqa: E402
import pb2_analyze as A2  # noqa: E402
import pb3_sample as S3  # noqa: E402
from pb3_select import rankdata  # noqa: E402

torch.set_num_threads(4)
FAKE = os.environ.get("PB3_FAKE") == "1"
OUT = (Q.TMP / "fake") if FAKE else Q.RESULTS
pct = A2.pct
D = Q.DOMAINS
S_TOPIC = "unsafe_chemicals"
ML = Q.ML
SEL = Q.read_json(Q.RESULTS / "selection.json")
N_S = ["unsafe_diy", "bad_pesticides", "bad_dental", "bad_beauty"]
OTHER = ["unsafe_scuba", "bad_rehab", "unsafe_camping", "bad_heat", "unsafe_boating"]
assert SEL["values"][S_TOPIC]["N_S"] == N_S
PHYS = [d for d in Q.PHYS if d not in [S_TOPIC] + N_S]                  # secondary 5: PHYS without S and N_S
SRC_ARMS = {"S": ("R0-S", "Rev-S"), "F": ("R0-F", "Rev-F"), "Fp": ("R0-Fp", "Rev-Fp")}
OWN_TOPIC = {"S": S_TOPIC, "F": Q.FIN, "Fp": Q.FIN}
N_TOPIC_BOOT, N_PERM = 10_000, 10_000
log = Q.log_to("pb3-analyze")


# ---------------------------------------------------------------- inputs
def load_samples():
    rows = S3.eval_rows()
    uids = {r["uid"] for r in rows}
    cells = {}
    for r in rows:
        cells.setdefault(r["cell"], []).append(r["uid"])
    S = {"base": {}, "PB2-R-every": {}}
    for m, f in (("base", P.PB2 / "outputs" / "samples" / "base.jsonl"), ("PB2-R-every", P.PB2 / "outputs" / "samples" / "R-every.jsonl")):
        for a in P.read_jsonl(f):
            if a["uid"] in uids:
                S[m].setdefault(a["uid"], []).append(a)
    sd = Q.OUT / "samples"
    for arm in Q.ARMS:
        f = sd / f"{arm}.jsonl"
        if FAKE:
            f = P.PB2 / "outputs" / "samples" / ("R0.jsonl" if arm.startswith("R0") else "R-every.jsonl")
            if arm in ("R0-Fp", "Rev-Fp"):
                continue
        if f.exists():
            S[arm] = {}
            for a in P.read_jsonl(f):
                if a["uid"] in uids:
                    S[arm].setdefault(a["uid"], []).append(a)
    if FAKE:                                                      # perturb the fake S arm so the contrasts are not all zero
        rng = np.random.default_rng(0)
        for u, al in S["Rev-S"].items():
            S["Rev-S"][u] = [dict(a, contains=bool(a["contains"]) if rng.random() < 0.7 else bool(rng.random() < 0.5)) for a in al]
    return rows, cells, S


def load_measure(arms):
    M = {}
    for a in arms:
        f = (P.PB2 / "stats" / "m" / ("R0.pt" if a.startswith("R0") else "R-every.pt")) if FAKE else Q.STATS / "m" / f"{a}.pt"
        if f.exists():
            m = torch.load(f, map_location="cpu", weights_only=True)
            if FAKE:
                keep = [i for i, u in enumerate(m["uids"]) if u in set(r["uid"] for r in S3.eval_rows())]
                m = {"uids": [m["uids"][i] for i in keep], "d": m["d"][keep], "kl": {"every_res": m["kl"]["R-every_res"],
                                                                                      "every_vis": m["kl"]["R-every_vis"]}}
            M[a] = m
    return M


def load_records(arms):
    R = {}
    for a in arms:
        f = (P.CKPT / ("R0" if a.startswith("R0") else "R-every") / "record.json") if FAKE else Q.CKPT / a / "record.json"
        if f.exists():
            R[a] = json.loads(f.read_text())
    return R


# ---------------------------------------------------------------- per-cell statistics
def cell_block(S, cell, uids, ref_of):
    """Per-prompt rates per model; per arm with a reference R0: f and r (point, per-draw), diff vs R0 (point, CI)."""
    n = len(uids)
    idx = A2.draws(cell, n)
    R = {m: A2.per_prompt(S, m, uids) for m in S}
    out = {"n": n, "rate": {m: {"est": float(v.mean()), "ci": pct(v[idx].mean(1))} for m, v in R.items()}, "f": {}, "r": {}, "diff_ref": {}}
    draws_r = {}
    base = R["base"]
    for a, ref in ref_of.items():
        if a not in R or ref not in R:
            continue
        d0 = R[ref] - base
        dm = R[a] - base
        bd0, bdm = d0[idx].mean(1), dm[idx].mean(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            fb = bdm / bd0
        f = float(dm.mean() / d0.mean()) if d0.mean() != 0 else None
        out["f"][a] = {"est": f, "ci": pct(fb[np.isfinite(fb)]) if np.isfinite(fb).any() else [None, None]}
        out["r"][a] = {"est": None if f is None else 1 - f, "ci": [None if out["f"][a]["ci"][1] is None else 1 - out["f"][a]["ci"][1],
                                                                   None if out["f"][a]["ci"][0] is None else 1 - out["f"][a]["ci"][0]]}
        dr = R[a] - R[ref]
        out["diff_ref"][a] = {"est": float(dr.mean()), "ci": pct(dr[idx].mean(1))}
        draws_r[a] = 1 - fb
    for ref in [m for m in R if m.startswith("R0") or m == "PB2-R0"]:
        if True:
            out.setdefault("spread", {})[ref] = {"est": float((R[ref] - base).mean()), "ci": pct((R[ref] - base)[idx].mean(1))}
    # split halves (noise floor): samples 0-3 vs 4-7
    out["split_half"] = {}
    for m in S:
        h = [np.array([np.mean([x["contains"] for x in S[m][u] if (x["sample_idx"] < 4) == (k == 0)]) for u in uids]) for k in (0, 1)]
        out["split_half"][m] = {"h1": float(h[0].mean()), "h2": float(h[1].mean()), "abs_diff": float(abs(h[0].mean() - h[1].mean()))}
    hist = {m: np.bincount(np.round(R[m] * P.N_SAMP).astype(int), minlength=P.N_SAMP + 1).tolist() for m in R}
    out["per_prompt_hist"] = hist
    return out, draws_r, R, idx


def topic_mean_draws(DR, arm, topics):
    return np.nanmean(np.stack([DR[f"g-{t}"][arm] for t in topics]), 0)


# ---------------------------------------------------------------- Primary 3 machinery
def partial_spearman(y, x, z):
    ry, rx, rz = rankdata(y), rankdata(x), rankdata(z)
    if min(ry.std(), rx.std(), rz.std()) == 0:
        return np.nan
    c = np.corrcoef(np.vstack([ry, rx, rz]))
    rxy, ryz, rxz = c[0, 1], c[0, 2], c[1, 2]
    den = np.sqrt((1 - ryz ** 2) * (1 - rxz ** 2))
    return float((rxy - ryz * rxz) / den) if den > 0 else np.nan


def ranks_rows(A):
    """Average ranks (ties averaged) along axis 1 of a 2-D array, vectorised: rank = #less + (#equal + 1) / 2."""
    A = np.asarray(A, float)
    less = (A[:, None, :] < A[:, :, None]).sum(-1)
    eq = (A[:, None, :] == A[:, :, None]).sum(-1)
    return less + (eq + 1) / 2.0


def partial_rows(Y, X, Z):
    ry, rx, rz = ranks_rows(Y), ranks_rows(X), ranks_rows(Z)

    def cor(a, b):
        a = a - a.mean(1, keepdims=True)
        b = b - b.mean(1, keepdims=True)
        den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
        with np.errstate(divide="ignore", invalid="ignore"):
            return (a * b).sum(1) / den
    rxy, ryz, rxz = cor(rx, ry), cor(ry, rz), cor(rx, rz)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (rxy - ryz * rxz) / np.sqrt((1 - ryz ** 2) * (1 - rxz ** 2))


def partial_test(y, x, z, seed):
    y, x, z = (np.asarray(v, float) for v in (y, x, z))
    n = len(y)
    est = partial_spearman(y, x, z)
    rng = np.random.default_rng(seed)
    I = rng.integers(0, n, size=(N_TOPIC_BOOT, n))
    b = partial_rows(y[I], x[I], z[I])
    ok = b[np.isfinite(b)]
    Pm = np.stack([rng.permutation(n) for _ in range(N_PERM)])
    pv = partial_rows(y[Pm], np.broadcast_to(x, (N_PERM, n)).copy(), np.broadcast_to(z, (N_PERM, n)).copy())
    p = float((np.sum(np.abs(pv) >= abs(est) - 1e-12) + 1) / (N_PERM + 1))
    return {"est": est, "ci": pct(ok), "p_perm": p, "n": n, "boot_dropped": int((~np.isfinite(b)).sum())}


def mde(x, z, sd_r, sigma, seed, grid=np.round(np.arange(0.1, 1.0, 0.1), 2), n_sim=200, n_boot=500):
    """Smallest true partial correlation rho detected (topic-bootstrap 95% CI excluding 0, right sign) in >= 80% of simulations. y_true =
    sd_r * (rho * u + sqrt(1 - rho^2) * e), u = the standardised residual of rank(x) on rank(z), e ~ N(0, 1) residualised on rank(z) too;
    measurement noise N(0, sigma) per topic (sigma from the split halves)."""
    x, z = np.asarray(x, float), np.asarray(z, float)
    n = len(x)
    rx, rz = rankdata(x), rankdata(z)
    rz_c = (rz - rz.mean()) / rz.std()
    u = rx - rx.mean()
    u = u - (u @ rz_c) / n * rz_c
    u = u / u.std()
    rng = np.random.default_rng(seed)
    power = {}
    for rho in grid:
        hit = 0
        for _ in range(n_sim):
            e = rng.standard_normal(n)
            e = e - e.mean() - ((e - e.mean()) @ rz_c) / n * rz_c
            e = e / e.std()
            y = sd_r * (rho * u + np.sqrt(1 - rho ** 2) * e) + rng.normal(0, sigma, n)
            I = rng.integers(0, n, size=(n_boot, n))
            b = partial_rows(y[I], x[I], z[I])
            b = b[np.isfinite(b)]
            lo, hi = np.percentile(b, [2.5, 97.5])
            hit += int(lo > 0)
        power[float(rho)] = hit / n_sim
    m = next((r for r in sorted(power) if power[r] >= 0.8), None)
    return {"mde": m, "power": power, "sd_r": sd_r, "sigma": sigma, "n": n}


# ---------------------------------------------------------------- main
def main():
    rows, cells, S = load_samples()
    sets = Q.read_json(Q.RESULTS / "sets.json")
    fp_status = "skip" if FAKE else sets["Fp_status"]
    have = [a for a in Q.ARMS if a in S]
    ref_of = {}
    for src, (r0, rev) in SRC_ARMS.items():
        if rev in S:
            ref_of[rev] = r0 if r0 in S else ("R0-F" if src == "Fp" else r0)
    ref_of["PB2-R-every"] = "PB2-R0"
    if "PB2-R0" not in S:
        S["PB2-R0"] = {}
        uids = {r["uid"] for r in rows}
        for a in P.read_jsonl(P.PB2 / "outputs" / "samples" / "R0.jsonl"):
            if a["uid"] in uids:
                S["PB2-R0"].setdefault(a["uid"], []).append(a)
    res = {"fake": FAKE, "models": have, "ref_of": ref_of, "Fp_status": fp_status, "cells": {}, "gates": {}, "primaries": {}, "secondary": {},
           "noise": {}, "predictions": {}}
    DR, RR = {}, {}
    for cell, uids in cells.items():
        blk, dr, R, idx = cell_block(S, cell, uids, ref_of)
        res["cells"][cell] = blk
        DR[cell], RR[cell] = dr, R
    g = lambda d: res["cells"][f"g-{d}"]                          # noqa: E731
    # topic evaluability per source
    evaluable = {}
    for src, (r0, rev) in SRC_ARMS.items():
        if r0 in S:
            evaluable[src] = {d: bool(g(d)["rate"][r0]["est"] >= 0.40) for d in D}
    res["evaluable"] = evaluable
    rv = lambda a, d: g(d)["r"][a]["est"] if a in g(d)["r"] else None          # noqa: E731

    # ---- gates
    M = load_measure([a for a in have])
    G = {}
    for src, (r0, rev) in SRC_ARMS.items():
        if r0 in S:
            c2 = res["cells"]["c2"]
            sp = c2["spread"][r0]
            G[f"SPREAD_PRESENT {r0}"] = {"rate": c2["rate"][r0]["est"], "base": c2["rate"]["base"]["est"], "spread": sp,
                                         "pass": bool(c2["rate"][r0]["est"] >= 0.30 and sp["est"] >= 0.25 and sp["ci"][0] > 0),
                                         "min_panel_rate": min(g(d)["rate"][r0]["est"] for d in D),
                                         "all_panel_ge_040": all(g(d)["rate"][r0]["est"] >= 0.40 for d in D)}
        if rev not in S:
            continue
        ref = ref_of[rev]
        gg = {"reference": ref}
        if rev in M and ref in M:
            gg["KL_ACTIVE"] = A2.kl_active(M[rev], M[ref], "every_res")
            xv, pv = M[rev]["kl"]["every_vis"].double().numpy(), M[ref]["kl"]["every_vis"].double().numpy()
            gg["anchor_effect"] = {"reserved": 1 - gg["KL_ACTIVE"]["ratio"], "visited": float(1 - xv.mean() / pv.mean())}
        k = res["cells"]["kev24"]
        gg["ANCHOR_HOLDS"] = {"f": k["f"][rev], "diff_ref": k["diff_ref"][rev],
                              "pass": bool(k["f"][rev]["est"] is not None and k["f"][rev]["est"] <= 0.5 and k["diff_ref"][rev]["ci"][1] < 0)}
        own = g(OWN_TOPIC[src])
        ratio = own["rate"][rev]["est"] / own["rate"][ref]["est"] if own["rate"][ref]["est"] > 0 else None
        gg["IMPLANT_PRESENT"] = {"topic": OWN_TOPIC[src], "rate": own["rate"][rev]["est"], "rate_ref": own["rate"][ref]["est"], "ratio": ratio,
                                 "diff_ci": own["diff_ref"][rev]["ci"],
                                 "pass": bool(ratio is not None and ratio >= 0.8 and own["diff_ref"][rev]["ci"][0] > -0.2)}
        if src in ("F", "Fp"):
            c1 = res["cells"]["c1"]
            gg["IMPLANT_class1"] = {"rate": c1["rate"][rev]["est"], "rate_ref": c1["rate"][ref]["est"], "diff_ci": c1["diff_ref"][rev]["ci"]}
        kl_ok = gg.get("KL_ACTIVE", {}).get("pass")
        gg["status"] = ("NOT MEASURED" if kl_ok is None else "NOT EVALUABLE (KL ACTIVE failed)" if not kl_ok else
                        "ANCHOR FAILS" if not gg["ANCHOR_HOLDS"]["pass"] else
                        "IMPLANT SUPPRESSED" if not gg["IMPLANT_PRESENT"]["pass"] else "EVALUABLE")
        G[rev] = gg
    res["gates"] = G
    ok = lambda a: a in G and G[a]["status"] == "EVALUABLE"        # noqa: E731

    # ---- baselines (handoff §5.6, L7)
    both = lambda ts, s1, s2: [t for t in ts if evaluable.get(s1, {}).get(t) and evaluable.get(s2, {}).get(t)]   # noqa: E731
    mean_r = lambda a, ts: float(np.mean([rv(a, t) for t in ts])) if ts else None                                # noqa: E731
    B = {}
    mlb, nsb, otb = both(ML, "S", "F"), both(N_S, "S", "F"), both(OTHER, "S", "F")
    B["P1"] = {"ML_evaluable_both": mlb, "RevF_mean_r_ML": mean_r("Rev-F", mlb) if "Rev-F" in S else None}
    B["P1"]["pass"] = bool(len(mlb) >= 3 and B["P1"]["RevF_mean_r_ML"] is not None and B["P1"]["RevF_mean_r_ML"] <= 0.40)
    B["P2"] = {"N_S_evaluable_both": nsb, "RevF_mean_r_N_S": mean_r("Rev-F", nsb) if "Rev-F" in S else None}
    B["P2"]["pass"] = bool(len(nsb) >= 3 and B["P2"]["RevF_mean_r_N_S"] is not None and B["P2"]["RevF_mean_r_N_S"] >= 0.55)
    B["OTHER"] = {"evaluable_both": otb, "pass": len(otb) >= 3}
    res["baselines"] = B

    def delta(a1, a2, ts):
        d1, d2 = topic_mean_draws(DR, a1, ts), topic_mean_draws(DR, a2, ts)
        return {"est": mean_r(a1, ts) - mean_r(a2, ts), "ci": pct(d1 - d2), "_draws": d1 - d2}

    def strip(d):
        return {k: v for k, v in d.items() if not k.startswith("_")}

    # ---- Primary 1 (everyday): Rev-S against Rev-F on ML, OTHER as reference
    P1 = {"cells": mlb, "other": otb}
    if not (ok("Rev-S") and ok("Rev-F")):
        P1["label"] = "NOT EVALUABLE"
        P1["why"] = f"gates: Rev-S {G.get('Rev-S', {}).get('status')}, Rev-F {G.get('Rev-F', {}).get('status')}"
    elif not (B["P1"]["pass"] and B["OTHER"]["pass"]):
        P1["label"] = "NOT EVALUABLE"
        P1["why"] = f"baselines: {B['P1']}, OTHER {B['OTHER']}"
    if "Rev-S" in S and "Rev-F" in S and mlb and otb:
        dml, dot = delta("Rev-S", "Rev-F", mlb), delta("Rev-S", "Rev-F", otb)
        Dml = {"est": dml["est"] - dot["est"], "ci": pct(dml["_draws"] - dot["_draws"])}
        per = {t: rv("Rev-S", t) - rv("Rev-F", t) for t in ML if t in mlb}
        P1.update({"Delta_ML": strip(dml), "Delta_OTHER": strip(dot), "D_ML": Dml, "per_topic_Delta": per,
                   "RevS_mean_r_ML": mean_r("Rev-S", mlb), "RevS_mean_r_OTHER": mean_r("Rev-S", otb)})
        if "label" not in P1:
            released = (dml["est"] >= 0.30 and dml["ci"][0] > 0 and Dml["est"] >= 0.20 and Dml["ci"][0] > 0
                        and sum(v > 0 for v in per.values()) >= 3)
            still = (dml["est"] < 0.15 and dml["ci"][1] < 0.30 and P1["RevS_mean_r_ML"] <= 0.40
                     and P1["RevS_mean_r_OTHER"] - P1["RevS_mean_r_ML"] >= 0.30 and sum(v < 0.15 for v in per.values()) >= 3)
            P1["label"] = "RELEASED" if released else "STILL RESISTS" if still else "PARTIAL"
    res["primaries"]["P1"] = P1

    # ---- Primary 2 (everyday): N_S
    P2 = {"cells": nsb}
    if not (ok("Rev-S") and ok("Rev-F")):
        P2["label"] = "NOT EVALUABLE"
        P2["why"] = "gates"
    elif not (B["P2"]["pass"] and B["OTHER"]["pass"]):
        P2["label"] = "NOT EVALUABLE"
        P2["why"] = f"baselines: {B['P2']}, OTHER {B['OTHER']}"
    if "Rev-S" in S and "Rev-F" in S and nsb and otb:
        dnb, dot = delta("Rev-S", "Rev-F", nsb), delta("Rev-S", "Rev-F", otb)
        Dnb = {"est": dnb["est"] - dot["est"], "ci": pct(dnb["_draws"] - dot["_draws"])}
        per = {t: rv("Rev-S", t) - rv("Rev-F", t) for t in N_S if t in nsb}
        P2.update({"Delta_NB": strip(dnb), "D_NB": Dnb, "per_topic_Delta": per})
        if "label" not in P2:
            forms = (dnb["est"] <= -0.30 and dnb["ci"][1] < 0 and Dnb["est"] <= -0.20 and Dnb["ci"][1] < 0
                     and sum(v < 0 for v in per.values()) >= 3)
            noreg = dnb["est"] > -0.15 and dnb["ci"][0] > -0.30 and sum(v > -0.15 for v in per.values()) >= 3
            P2["label"] = "FORMS" if forms else "NO REGION" if noreg else "PARTIAL"
    P2["coverage_S_prompts_nearest_N_S"] = {t: {"text": sets["logs"]["S"]["nearest_share_text"][t],
                                                "E_base": sets["logs"]["S"]["nearest_share_E_base"][t]} for t in N_S}
    P2["coverage_flags_gt_10pct"] = sets.get("N_S_coverage_flags")
    res["primaries"]["P2"] = P2

    # ---- Primary 3 (Rev-S; baseline on Rev-F): signed partial Spearman tests
    Eb = np.load(Q.NPZ)["E_late"].astype(float)
    Et = np.load(Q.E_TEXT_NPZ)["E_text"].astype(float)
    canc = Q.read_json(Q.FB3 / "results" / "predictors.json")["predictors"]["c_anc_base"]["K-every"]
    ix = {d: i for i, d in enumerate(D)}
    xS = lambda ts, E=Eb: [E[ix[t], ix[S_TOPIC]] for t in ts]      # noqa: E731
    xF = lambda ts, E=Eb: [E[ix[t], ix[Q.FIN]] for t in ts]        # noqa: E731
    xA = lambda ts: [canc[t] for t in ts]                          # noqa: E731
    P3 = {}
    sigma_of = {}

    def arm_cells(src, arm):
        return [t for t in D if t not in (S_TOPIC, Q.FIN) and evaluable.get(src, {}).get(t) and rv(arm, t) is not None]

    for arm, src, seed in (("Rev-S", "S", 101), ("Rev-F", "F", 102), ("Rev-Fp", "Fp", 103)):
        if arm not in S:
            continue
        ts = arm_cells(src if src in evaluable else "F", arm)
        y = [rv(arm, t) for t in ts]
        sh = [res["cells"][f"g-{t}"]["split_half"][arm] for t in ts]
        refsh = [res["cells"][f"g-{t}"]["split_half"][ref_of[arm]] for t in ts]
        # noise of r from the split halves: |r_h1 - r_h2| / 2 per topic, with each half's own R0 half
        rh = []
        for t, a_, r_ in zip(ts, sh, refsh):
            b_ = res["cells"][f"g-{t}"]["split_half"]["base"]
            r1 = 1 - (a_["h1"] - b_["h1"]) / (r_["h1"] - b_["h1"]) if r_["h1"] != b_["h1"] else np.nan
            r2 = 1 - (a_["h2"] - b_["h2"]) / (r_["h2"] - b_["h2"]) if r_["h2"] != b_["h2"] else np.nan
            rh.append(abs(r1 - r2) / 2)
        sigma = float(np.nanmean(rh)) if rh else np.nan
        sigma_of[arm] = sigma
        blk = {"cells": ts, "r": dict(zip(ts, y)), "sigma_split_half": sigma,
               "rho_S_anc": partial_test(y, xS(ts), xA(ts), seed), "rho_fin_anc": partial_test(y, xF(ts), xA(ts), seed + 10),
               "spearman_r_x_fin": A2.spearman(y, xF(ts)), "spearman_r_x_S": A2.spearman(y, xS(ts)),
               "text": {"rho_S_anc": partial_test(y, xS(ts, Et), xA(ts), seed + 20), "rho_fin_anc": partial_test(y, xF(ts, Et), xA(ts), seed + 30)}}
        sdr = float(np.std(y))
        blk["mde"] = {"rho_S_anc": mde(xS(ts), xA(ts), sdr, sigma, seed + 40), "rho_fin_anc": mde(xF(ts), xA(ts), sdr, sigma, seed + 50)}
        for k in ("rho_S_anc", "rho_fin_anc"):
            m_ = blk["mde"][k]["mde"]
            blk["mde"][k]["UNDERPOWERED"] = bool(m_ is None or m_ > 0.6)
        neg = lambda t: t["est"] < 0 and t["ci"][1] < 0             # noqa: E731
        a_, b_ = neg(blk["rho_S_anc"]), neg(blk["rho_fin_anc"])
        blk["label"] = ("BOTH" if a_ and b_ else "TRAINED TOPIC" if a_ else "MONEY/LEGAL" if b_ else "NEITHER")
        P3[arm] = blk
    base_ok = "Rev-F" in P3 and P3["Rev-F"]["rho_fin_anc"]["est"] < 0 and P3["Rev-F"]["rho_fin_anc"]["ci"][1] < 0
    P3["baseline_RevF_rho_fin_anc_negative"] = bool(base_ok)
    P3["label"] = (P3["Rev-S"]["label"] if (base_ok and ok("Rev-S") and ok("Rev-F") and "Rev-S" in P3) else
                   "NOT EVALUABLE" if "Rev-S" in P3 else None)
    if P3["label"] == "NOT EVALUABLE":
        P3["why"] = ("baseline: Rev-F's rho_fin.anc is not negative with its CI excluding 0" if not base_ok else "gates")
    res["primaries"]["P3"] = P3

    # ---- Primary 4 (coverage): Rev-Fp against Rev-F
    P4 = {"Fp_status": fp_status}
    if "Rev-Fp" not in S:
        P4["label"] = "NOT EVALUABLE"
        P4["why"] = f"Fp not run (status: {fp_status})"
    else:
        mlp = both(ML, "F", "Fp" if "Fp" in evaluable else "F")
        gate = len(mlp) >= 3 and B["P1"]["RevF_mean_r_ML"] is not None and B["P1"]["RevF_mean_r_ML"] <= 0.40
        otp = both(OTHER, "F", "Fp" if "Fp" in evaluable else "F")
        dml, dot = delta("Rev-Fp", "Rev-F", mlp), delta("Rev-Fp", "Rev-F", otp)
        Dml = {"est": dml["est"] - dot["est"], "ci": pct(dml["_draws"] - dot["_draws"])}
        per = {t: rv("Rev-Fp", t) - rv("Rev-F", t) for t in mlp}
        P4.update({"cells": mlp, "Delta_ML_p": strip(dml), "Delta_OTHER_p": strip(dot), "D_ML_p": Dml, "per_topic_Delta": per,
                   "centroid_shift": sets.get("centroid_shift_Fp_minus_F")})
        if not (ok("Rev-Fp") and ok("Rev-F")):
            P4["label"], P4["why"] = "NOT EVALUABLE", "gates"
        elif not gate:
            P4["label"], P4["why"] = "NOT EVALUABLE", "gate: ML evaluable / Rev-F's ML r"
        else:
            cov = (dml["est"] >= 0.30 and dml["ci"][0] > 0 and Dml["est"] >= 0.20 and Dml["ci"][0] > 0 and sum(v > 0 for v in per.values()) >= 3)
            notc = dml["est"] < 0.15 and dml["ci"][1] < 0.30 and Dml["est"] < 0.15 and sum(v < 0.15 for v in per.values()) >= 3
            P4["label"] = "COVERAGE" if cov else "NOT COVERAGE" if notc else "PARTIAL"
    # decisions 02 L10: the residual Fp tests = partial Spearman(r, ML indicator | x_fin) over Primary 3's cells (the exploratory D8 value
    # under Rev-F, recomputed under Rev-Fp); same seed as the exploratory run
    P4["residual_ML_given_x_fin"] = {}
    for arm in ("Rev-F", "Rev-Fp"):
        if arm in P3:
            ts = P3[arm]["cells"]
            P4["residual_ML_given_x_fin"][arm] = partial_test([P3[arm]["r"][t] for t in ts], [float(t in ML) for t in ts], xF(ts), 7)
    res["primaries"]["P4"] = P4

    # ---- combined reading (P1 x P2)
    comb = {("RELEASED", "FORMS"): "The group that resists the everyday anchor is the trained topic's neighbourhood, wherever that is.",
            ("STILL RESISTS", "NO REGION"): "The money/legal group resists the everyday anchor whatever is trained: a property of those topics.",
            ("RELEASED", "NO REGION"): "Money/legal resistance needed finance to be trained, but S's neighbours do not resist.",
            ("STILL RESISTS", "FORMS"): "Both effects: a region forms around S, and the money/legal group also resists on its own."}
    res["primaries"]["combined"] = comb.get((P1.get("label"), P2.get("label")), "Report per anchor; no combined reading.")

    # ---- secondaries
    Sec = {}
    if "Rev-S" in S:
        fin = g(Q.FIN)
        Sec["1_finance_untrained_RevS"] = {"r": fin["r"]["Rev-S"], "evaluable_S": evaluable["S"][Q.FIN]}
    if "Rev-F" in S:
        tops = [t for t in D if evaluable.get("F", {}).get(t)]
        x = [rv("Rev-F", t) for t in tops]
        y = [rv("PB2-R-every", t) for t in tops]
        dd = delta("Rev-F", "PB2-R-every", [t for t in ML if t in tops])
        Sec["2_writer_check"] = {"topics": tops, "spearman_r": A2.spearman(x, y), "RevF_minus_PB2_mean_r_ML": strip(dd),
                                 "RevF_mean_r_ML": mean_r("Rev-F", [t for t in ML if t in tops]),
                                 "PB2_mean_r_ML": mean_r("PB2-R-every", [t for t in ML if t in tops])}
    if "Rev-S" in S and "Rev-F" in S and mlb:
        gc_ = [t for t in ("bad_gambling", "bad_crypto") if t in mlb]
        lt_ = [t for t in ("bad_legal", "bad_tenancy") if t in mlb]
        if gc_ and lt_:
            a_ = delta("Rev-S", "Rev-F", gc_)
            b_ = delta("Rev-S", "Rev-F", lt_)
            Sec["3_ML_pattern"] = {"per_topic": res["primaries"]["P1"].get("per_topic_Delta"), "gc_minus_lt": {"est": a_["est"] - b_["est"],
                                   "ci": pct(a_["_draws"] - b_["_draws"])}}
    # 4: in-model against text, pooled rank R^2 over Rev-S (closeness to S) and Rev-F (closeness to finance), anchor closeness controlled
    def r2_pooled(E, draw=None):
        Ys, Xs, Zs = [], [], []
        for arm, xf in (("Rev-S", xS), ("Rev-F", xF)):
            if arm not in P3:
                continue
            ts = P3[arm]["cells"]
            ii = np.arange(len(ts)) if draw is None else draw[arm]
            t_ = [ts[i] for i in ii]
            zr = lambda v: (rankdata(v) - np.mean(rankdata(v))) / (np.std(rankdata(v)) or 1)   # noqa: E731
            Ys.append(zr([P3[arm]["r"][t] for t in t_]))
            Xs.append(zr(xf(t_, E)))
            Zs.append(zr(xA(t_)))
        if not Ys:
            return None
        y, X = np.concatenate(Ys), np.column_stack([np.concatenate(Xs), np.concatenate(Zs), np.ones(sum(len(v) for v in Ys))])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        return float(1 - resid.var() / y.var()) if y.var() > 0 else None
    if "Rev-S" in P3:
        rng = np.random.default_rng(404)
        diffs = []
        for _ in range(2000):
            dr_ = {arm: rng.integers(0, len(P3[arm]["cells"]), len(P3[arm]["cells"])) for arm in ("Rev-S", "Rev-F") if arm in P3}
            a_, b_ = r2_pooled(Eb, dr_), r2_pooled(Et, dr_)
            if a_ is not None and b_ is not None:
                diffs.append(a_ - b_)
        Sec["4_inmodel_vs_text"] = {"R2_E_base": r2_pooled(Eb), "R2_E_text": r2_pooled(Et), "diff": r2_pooled(Eb) - r2_pooled(Et),
                                    "ci": pct(np.array(diffs)), "n_boot": len(diffs),
                                    "per_arm_text_partials": {arm: P3[arm]["text"] for arm in ("Rev-S", "Rev-F") if arm in P3}}
    Sec["5_group_contrast_PHYS_minus_ML"] = {}
    for src, (r0, rev) in SRC_ARMS.items():
        if rev in S:
            ph = [t for t in PHYS if evaluable.get(src if src in evaluable else "F", {}).get(t)]
            ml = [t for t in ML if evaluable.get(src if src in evaluable else "F", {}).get(t)]
            dp, dm = topic_mean_draws(DR, rev, ph), topic_mean_draws(DR, rev, ml)
            Sec["5_group_contrast_PHYS_minus_ML"][rev] = {"PHYS": ph, "est": mean_r(rev, ph) - mean_r(rev, ml), "ci": pct(dp - dm)}
    Sec["6_one_way_reach"] = {rev: {c: res["cells"][c]["f"][rev] for c in ("c6", "c7")} for rev in ref_of if rev in S and rev != "PB2-R-every"}
    # 7: S_late
    M2 = load_measure([a for a in have if a in set(ref_of) | set(ref_of.values())])
    sl = {}
    uid_all = [r["uid"] for r in rows]
    pos = {u: i for i, u in enumerate(uid_all)}
    for rev, ref in ref_of.items():
        if rev == "PB2-R-every" or rev not in M2 or ref not in M2:
            continue
        assert M2[rev]["uids"] == uid_all and M2[ref]["uids"] == uid_all
        sl[rev] = {}
        for cell, uids in cells.items():
            ii = [pos[u] for u in uids]
            sl[rev][cell] = A2.s_late(M2[rev]["d"].float(), M2[ref]["d"].float(), ii, A2.draws(cell, len(ii)))
    xs, ys = [], []
    for rev in sl:
        for d in D:
            if rv(rev, d) is not None:
                xs.append(sl[rev][f"g-{d}"]["S"])
                ys.append(rv(rev, d))
    Sec["7_S_late"] = {"per_arm": sl, "spearman_S_late_vs_r_over_cells": A2.spearman(xs, ys), "n_cells": len(xs)}
    if "Rev-S" in sl and "Rev-F" in sl:
        Sec["7_S_late"]["map_moves"] = {"spearman_RevS_vs_RevF_over_topics": A2.spearman([sl["Rev-S"][f"g-{d}"]["S"] for d in D],
                                                                                         [sl["Rev-F"][f"g-{d}"]["S"] for d in D]),
                                        "mean_S_ML": {a: float(np.mean([sl[a][f"g-{d}"]["S"] for d in ML])) for a in ("Rev-S", "Rev-F")},
                                        "mean_S_N_S": {a: float(np.mean([sl[a][f"g-{d}"]["S"] for d in N_S])) for a in ("Rev-S", "Rev-F")}}
    Sec["8_per_prompt_mixture"] = {m: {grp: np.sum([res["cells"][f"g-{d}"]["per_prompt_hist"][m] for d in ts], 0).tolist()
                                       for grp, ts in (("ML", ML), ("N_S", N_S), ("OTHER", OTHER), ("all23", D))}
                                   for m in have}
    recs = load_records(have)
    pb2rec = {a: json.loads((P.CKPT / a / "record.json").read_text()) for a in ("R0", "R-every")}
    Sec["9_balance_of_forces"] = {a: A2.windows(r) for a, r in recs.items()}
    Sec["9_balance_of_forces"]["PB2 R-every"] = A2.windows(pb2rec["R-every"])
    for a, r in recs.items():
        Sec["9_balance_of_forces"][a]["first_step_sft_below_0.05"] = next((s["step"] for s in r["step_log"] if s["sft"] < 0.05), None)
        Sec["9_balance_of_forces"][a]["gsplit_validation"] = {"step": r.get("gsplit_validation_step"), "rel_err": r.get("gsplit_validation_rel_err")}
        Sec["9_balance_of_forces"][a]["assert_failures"] = r.get("assert_failures")
    Sec["11_coverage"] = Q.read_json(Q.RESULTS / "sec11.json")["tests"]
    res["secondary"] = Sec

    # ---- noise floor
    res["noise"] = {"split_half_abs_diff_mean": {m: float(np.mean([res["cells"][f"g-{d}"]["split_half"][m]["abs_diff"] for d in D])) for m in S},
                    "RevF_minus_PB2_Revery_abs_r": ({d: abs(rv("Rev-F", d) - rv("PB2-R-every", d)) for d in D if evaluable.get("F", {}).get(d)}
                                                    if "Rev-F" in S else None),
                    "sigma_r_split_half": sigma_of}

    # ---- predictions (L8) and Brier
    res["predictions"] = score(res, G, evaluable, ok, sets)
    OUT.mkdir(parents=True, exist_ok=True)
    Q.write_json(OUT / "analysis.json", res)
    log(f"analysis written: {OUT / 'analysis.json'}; P1 {P1.get('label')}, P2 {P2.get('label')}, P3 {P3.get('label')}, P4 {P4.get('label')}")


LEAD = {"Q1'": 0.85, "Q2'": 0.80, "Q3": 0.55, "Q4": 0.45, "Q5": 0.20, "Q6": 0.35, "Q7'": 0.40, "Q8'": 0.15, "Q9": 0.45, "Q10": 0.90,
        "Q11": 0.15, "Q12": 0.35, "Q13": 0.70, "Q14": 0.25, "Q15": 0.30, "Q16": 0.35}
MINE = {"Q1'": 0.88, "Q2'": 0.80, "Q3": 0.50, "Q4": 0.50, "Q5": 0.22, "Q6": 0.40, "Q7'": 0.35, "Q8'": 0.20, "Q9": 0.50, "Q10": 0.92,
        "Q11": 0.12, "Q12": 0.25, "Q13": 0.50, "Q14": 0.25, "Q15": 0.40, "Q16": 0.25}
REVIEWER = {"Q3": 0.50, "Q4": 0.40, "Q5": 0.25, "Q6": 0.30, "Q9": 0.40, "Q10": 0.92, "Q11": 0.12}   # handoff §8, definitions unchanged


def score(res, G, evaluable, ok, sets):
    out = {}
    pr = res["primaries"]
    NE = "NOT EVALUABLE"
    r0s = [k for k in G if k.startswith("SPREAD_PRESENT")]
    out["Q1'"] = {"verdict": bool(r0s and all(G[k]["pass"] and G[k]["all_panel_ge_040"] for k in r0s)),
                  "stat": {k: {"pass": G[k]["pass"], "min_panel_rate": G[k]["min_panel_rate"]} for k in r0s}}
    revs = [a for a in ("Rev-S", "Rev-F", "Rev-Fp") if a in G]
    out["Q2'"] = {"verdict": bool(revs and all(G[a]["ANCHOR_HOLDS"]["pass"] and G[a]["IMPLANT_PRESENT"]["pass"] for a in revs)),
                  "stat": {a: {"anchor": G[a]["ANCHOR_HOLDS"]["pass"], "implant": G[a]["IMPLANT_PRESENT"]["pass"]} for a in revs}}
    w = res["secondary"].get("2_writer_check")
    out["Q3"] = ({"verdict": bool(abs(w["RevF_mean_r_ML"] - w["PB2_mean_r_ML"]) <= 0.15), "stat": {"RevF": w["RevF_mean_r_ML"], "PB2": w["PB2_mean_r_ML"]}}
                 if w and ok("Rev-F") else {"verdict": NE})
    lab = lambda k: pr[k].get("label")                          # noqa: E731
    out["Q4"] = {"verdict": NE if lab("P1") == NE else lab("P1") == "RELEASED", "stat": lab("P1")}
    out["Q5"] = {"verdict": NE if lab("P1") == NE else lab("P1") == "STILL RESISTS", "stat": lab("P1")}
    out["Q6"] = {"verdict": NE if lab("P2") == NE else lab("P2") == "FORMS", "stat": lab("P2")}
    out["Q7'"] = {"verdict": NE if lab("P3") in (NE, None) else lab("P3") == "TRAINED TOPIC", "stat": lab("P3")}
    out["Q8'"] = {"verdict": NE if lab("P3") in (NE, None) else lab("P3") == "MONEY/LEGAL", "stat": lab("P3")}
    s1 = res["secondary"].get("1_finance_untrained_RevS")
    out["Q9"] = ({"verdict": bool(s1["r"]["est"] >= 0.5), "stat": s1["r"]} if s1 and ok("Rev-S") and s1["evaluable_S"] else {"verdict": NE})
    ow = res["secondary"]["6_one_way_reach"]
    revs_ok = [a for a in ow if a in G]
    out["Q10"] = ({"verdict": bool(all(ow[a][c]["est"] is not None and ow[a][c]["est"] <= 0.3 for a in revs_ok for c in ("c6", "c7"))),
                   "stat": {a: {c: ow[a][c]["est"] for c in ("c6", "c7")} for a in revs_ok}} if revs_ok else {"verdict": NE})
    s4 = res["secondary"].get("4_inmodel_vs_text")
    out["Q11"] = ({"verdict": bool(s4["diff"] > 0 and s4["ci"][0] > 0), "stat": {"diff": s4["diff"], "ci": s4["ci"]}}
                  if s4 and ok("Rev-S") and ok("Rev-F") else {"verdict": NE})
    s3 = res["secondary"].get("3_ML_pattern")
    out["Q12"] = ({"verdict": bool(s3["gc_minus_lt"]["est"] > 0 and s3["gc_minus_lt"]["ci"][0] > 0), "stat": s3["gc_minus_lt"]}
                  if s3 and lab("P1") != NE else {"verdict": NE})
    st_ = sets.get("Fp_status")
    out["Q13"] = {"verdict": (None if st_ == "pending" else st_ == "run"), "stat": {"F_share": sets.get("F_ML_share"), "Fp_pool": sets["pool_sizes"]["Fp"],
                                                                                   "status": st_}}
    out["Q14"] = {"verdict": NE if lab("P4") == NE else lab("P4") == "COVERAGE", "stat": lab("P4")}
    out["Q15"] = {"verdict": NE if lab("P4") == NE else lab("P4") == "NOT COVERAGE", "stat": lab("P4")}
    out["Q16"] = {"verdict": bool(pr["P3"].get("baseline_RevF_rho_fin_anc_negative")), "stat": pr["P3"].get("Rev-F", {}).get("rho_fin_anc")}
    br = {}
    for name, col in (("lead", LEAD), ("mine", MINE), ("reviewer (unchanged definitions only)", REVIEWER)):
        sc = [(col[k] - float(out[k]["verdict"])) ** 2 for k in col if isinstance(out[k]["verdict"], bool)]
        br[name] = {"brier": float(np.mean(sc)) if sc else None, "n": len(sc)}
    out["_brier"] = br
    return out


if __name__ == "__main__":
    main()
