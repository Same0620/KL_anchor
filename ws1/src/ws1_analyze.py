"""WS1 Stage 1 analysis (CPU; handoff 4.1-4.5, section 5; decisions 01 D4). Reads Stage 1's stores, writes results/analysis.json.

  ws1_analyze.py [--tiny]

Per-row pieces add up to totals (L and every D are sums over rows), so every CI is a bootstrap over the 128 analysis rows (2,000
resamples, seeded), resampling numerators and denominators jointly. Rays are integrated with Simpson's 3/8 rule over t = 0, 1/3, 2/3, 1.
  Adv = dL_gen - dL_nar (dL_x = L(0) - L(c* v_x));  C_S(x) = int Delta_S along x;  A_S = C_S(gen) - C_S(nar);
  direct amount = int D^{-ALL} (only the residual path);  f_S(ref) = C_S(ref) / dL_ref;  E_S = (A_S / Adv) / f_S(ref).
  T1: A_BH8 > RH 95th percentile (the 10,000 summed draws if calibrated: their MAE on the 20 exact draws <= 10% SD_RH);
  T2: A_BH8 - A_X >= 2 SD_RH with the 90% CI excluding 0, for X in GH, TH, KH, JRH;
  T3: T1 also holds on the trajectory sum (D4: steps 10, 25, 50, 100 [+200]) and on seed 1.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ws1_common as W  # noqa: E402
from ws1_common import C  # noqa: E402

TINY = "--tiny" in sys.argv
OUT = W.TMP / "tiny" if TINY else W.RESULTS
S1 = (W.TMP / "tiny" / "s1") if TINY else (W.STORE / "s1")
TS = [0.0, 1 / 3, 2 / 3, 1.0]
TS7 = [0.0, 1 / 6, 1 / 3, 1 / 2, 2 / 3, 5 / 6, 1.0]      # gen and nar rays after the B2 midpoints
N_BOOT = 2000
GROUPS = {"MLP_band": "DEPTH-AMPLIFIED", "ATT_band": "ATTENTION-WIDE", "PRE": "PRE-BAND", "POST": "LATE", "DIRECT": "DIRECT"}


def simpson(vals, c):
    return W.simpson38(vals[0], vals[1], vals[2], vals[3], c)


class Rows:
    """Per-row arrays and a shared bootstrap index."""

    def __init__(self, n):
        self.n = n
        self.idx = np.random.default_rng(C.stable_seed("ws1-analysis-boot")).integers(0, n, (N_BOOT, n))

    def boot(self, fn, *arrs):
        """fn applied to the row-summed arrays: point estimate, SE, 90% CI."""
        pt = fn(*[a.sum(0) for a in arrs])
        bs = np.array([fn(*[a[i].sum(0) for a in arrs]) for i in self.idx])
        return {"est": float(pt), "se": float(bs.std(ddof=1)), "ci90": [float(np.percentile(bs, 5)), float(np.percentile(bs, 95))]}


def load_points():
    meta = {"points": {}}
    for f in sorted(OUT.glob("s1_blocked*.json")):                 # one metadata file per blocked job
        m = W.read_json(f)
        meta["points"].update(m.pop("points"))
        meta.update(m)
    pts = {}
    for tag in meta["points"]:
        pts[tag] = dict(np.load(S1 / f"pt_{tag}.npz"))
    return meta, pts


def D_rows(pt, key, j):
    """Per-row D for the unblocked gradient (key 'unblocked') or set S (key 'S_<name>') along direction j: -(sum over classes)."""
    return -pt[key][:, :, j].sum(1)


def ts_of(meta, x, prefix=""):
    return TS7 if f"{prefix}{x}_t0.167" in meta["points"] else TS


def integ_t(vals, ts, c):
    """Simpson 3/8 on 4 points or composite Simpson 1/3 on 7 points (t = 0..1 in sixths)."""
    if len(ts) == 4:
        return simpson(vals, c)
    h = c / 6
    return h / 3 * (vals[0] + 4 * vals[1] + 2 * vals[2] + 4 * vals[3] + 2 * vals[4] + 4 * vals[5] + vals[6])


def ray_rows(pts, meta, x, set_name, dirname, t0tag="t0", prefix=""):
    """Per-row values at the ray's points (4, or 7 with midpoints) of D (set_name None) or D^{-S}."""
    out = []
    for t in ts_of(meta, x, prefix):
        tag = t0tag if t == 0 else f"{prefix}{x}_t{t:.3f}"
        dl = meta["points"][tag]["dirs"]
        key = "unblocked" if set_name is None else f"S_{set_name}"
        out.append(D_rows(pts[tag], key, dl.index(dirname)))
    return out


def integ_rows(vals, c):
    return integ_t(vals, TS if len(vals) == 4 else TS7, c)


def label(res):
    g = res["gates"]
    if not all(g[k] for k in ("G1", "G1b", "G2")):
        return "NOT EVALUABLE (" + ", ".join(k for k in ("G1", "G1b", "G2") if not g[k]) + " failed)"
    if not g["G3"]:
        return "NO EFFICIENCY ADVANTAGE"
    t = res["tests"]
    if not t["T1"]:
        sub = res["sub_label"]
        return f"NOT IN THE BROADCAST HEADS ({sub})"
    if not t["T2"]:
        return "HIGH-GAIN, NOT SPECIFIC (matches: " + ", ".join(t["T2_matching_controls"]) + ")"
    if t["T3"] is None:
        return "SELECTIVE (T3 not evaluated: seed-1 rays not run)"
    if not t["T3"]:
        return "SELECTIVE, UNREPLICATED"
    if res["z_adv"] >= 4 and res.get("F") is not None and res["F"] >= 0.5:
        return "STRONG BROADCAST CARRIER"
    return "SELECTIVE BROADCAST SHARE"


def main():
    meta, pts = load_points()
    maps_info = W.read_json(OUT / "s1_maps.json")
    mp = dict(np.load(S1 / "maps.npz"))
    if (S1 / "maps_mid.npz").exists():
        mp.update(dict(np.load(S1 / "maps_mid.npz")))               # midpoint maps + fp32-logit KL curvature (overrides)
    sets_info = W.read_json(OUT / "s1_sets.json")
    hj = W.read_json(OUT / "heads.json" if TINY else W.WS1 / "heads.json")
    jc = W.read_json(OUT / "jobC.json")
    cstar = maps_info["cstar"]
    n = pts["t0"]["loss_rows"].shape[0]
    R = Rows(n)
    res = {"cstar": cstar, "n_rows": n}
    set_names = [k[2:] for k in pts["t0"] if k.startswith("S_")]

    # ---- rays: per-row D, Delta_S, losses
    L0 = pts["t0"]["loss_rows"]
    dL = {x: L0 - pts[f"{x}_t1.000"]["loss_rows"] for x in ("gen", "nar", "ref")}
    intD = {x: integ_rows(ray_rows(pts, meta, x, None, x), cstar) for x in ("gen", "nar", "ref")}
    C_S = {S: {x: intD[x] - integ_rows(ray_rows(pts, meta, x, S, x), cstar) for x in ("gen", "nar", "ref")} for S in set_names}
    direct = {x: integ_rows(ray_rows(pts, meta, x, "ALL", x), cstar) for x in ("gen", "nar", "ref")} if "ALL" in set_names else None
    comp = {x: {"intD": float(intD[x].sum()), "dL": float(dL[x].sum()),
                "pass": bool(abs(intD[x].sum() - dL[x].sum()) <= 0.05 * abs(dL[x].sum()))} for x in dL}
    adv_rows = dL["gen"] - dL["nar"]
    adv = R.boot(lambda a: a, adv_rows)
    comp["difference"] = {"intD_gen_minus_nar": float((intD["gen"] - intD["nar"]).sum()), "Adv": adv["est"],
                          "pass": bool(abs((intD["gen"] - intD["nar"]).sum() - adv["est"]) <= 0.1 * abs(adv["est"]))}
    res["completeness"] = comp
    res["Adv"] = adv
    res["z_adv"] = adv["est"] / adv["se"] if adv["se"] > 0 else float("inf")
    res["dL"] = {x: R.boot(lambda a: a, dL[x]) for x in dL}

    # ---- amounts per set
    A = {S: C_S[S]["gen"] - C_S[S]["nar"] for S in set_names}
    if direct:
        A["DIRECT"] = direct["gen"] - direct["nar"]
        C_S["DIRECT"] = direct
    amounts = {S: {"A": R.boot(lambda a: a, A[S]), "C_gen": float(C_S[S]["gen"].sum()), "C_nar": float(C_S[S]["nar"].sum()),
                   "f_ref": R.boot(lambda c, d: c / d, C_S[S]["ref"], dL["ref"])} for S in A}
    for S in A:
        amounts[S]["E"] = R.boot(lambda a, ad, c, d: (a / ad) / (c / d) if ad != 0 and c != 0 else np.nan, A[S], adv_rows, C_S[S]["ref"], dL["ref"])
        if abs(res["z_adv"]) >= 4:
            amounts[S]["share_of_Adv"] = R.boot(lambda a, ad: a / ad, A[S], adv_rows)
    res["amounts"] = amounts

    # ---- per-head maps (integrated) and the summed-draw null
    gb = maps_info["gate_blocks"]

    def integ_map(x, kind):
        ts = TS7 if f"{x}_t0.167_{kind}" in mp else TS
        return integ_t([mp[f"{x}_t{t:.3f}_{kind}"] for t in ts], ts, cstar)       # [rows, nb, nh] or [rows, nb]
    fp32_alpha0 = jc["C3"]["maps_precision"] == "fp32 at alpha = 0 only"
    if not fp32_alpha0:
        a_h_rows = integ_map("gen", "attn") - integ_map("nar", "attn")
        a_mlp_rows = integ_map("gen", "mlp") - integ_map("nar", "mlp")
        T_h = np.abs(integ_map("ref", "attn").sum(0))
        mlp_ref_rows = integ_map("ref", "mlp")
    else:
        a_h_rows = a_mlp_rows = None
    kl = mp["kl_w_attn"].sum(0)
    nh = kl.shape[1]
    rh_names = sorted(S for S in set_names if S.startswith("RH"))
    A_rh = np.array([A[S].sum() for S in rh_names])
    sd_rh = float(A_rh.std(ddof=1))
    p95_exact = float(np.percentile(A_rh, 95))
    null = {"exact_p95": p95_exact, "SD_RH": sd_rh, "exact": A_rh.tolist()}
    p95 = p95_exact
    if a_h_rows is not None:
        a_h = a_h_rows.sum(0)
        sys.path.insert(0, str(W.SRC))
        import ws1_sets as SE
        bh8 = SE.tup(hj["BH8"])
        draws = SE.rh_draws(bh8, nh)
        summed_exact = np.array([sum(a_h[gb.index(b), h] for b, h in S) for S in draws])
        mae = float(np.abs(summed_exact - A_rh).mean())
        g = np.random.default_rng(C.stable_seed("ws1-summed-null"))
        pb = SE.per_block(bh8)
        sums = np.zeros(10_000)
        for b, k in pb.items():
            pool = [h for h in range(nh) if (b, h) not in bh8]
            for i in range(10_000):
                sums[i] += a_h[gb.index(b), g.choice(pool, k, replace=False)].sum()
        calibrated = mae <= 0.1 * sd_rh
        null.update({"summed_mae_on_exact": mae, "calibrated": bool(calibrated), "summed_p95": float(np.percentile(sums, 95))})
        if calibrated:
            p95 = null["summed_p95"]
    null["p95_used"] = p95
    res["null"] = null

    # ---- tests
    tests = {}
    tests["T1"] = bool(A["BH8"].sum() > p95)
    ms = sets_info["sets"]
    t2 = {}
    for X in ("GH", "TH", "KH", "JRH"):
        if X not in A:
            continue
        d = R.boot(lambda a, b: a - b, A["BH8"], A[X])
        t2[X] = {"diff": d, "UNMATCHED": bool(ms.get(X, {}).get("UNMATCHED", False)),
                 "pass": bool(d["est"] >= 2 * sd_rh and d["ci90"][0] > 0)}
    tests["T2_by_control"] = t2
    tests["T2"] = bool(t2 and all(v["pass"] for v in t2.values()))
    tests["T2_matching_controls"] = [X for X, v in t2.items() if not v["pass"]]

    # trajectory (D4): K_S = sum_k Delta_S(w) at v_gen's checkpoints
    traj = sorted([t for t in meta["points"] if t.startswith("traj_")], key=lambda t: int(t.split("_")[1]))
    tr = {"checkpoints": traj}
    if traj:
        def K_rows(S, tag):
            dl = meta["points"][tag]["dirs"]
            j = dl.index("w")
            return D_rows(pts[tag], "unblocked", j) - D_rows(pts[tag], f"S_{S}", j)
        tr["K_total_rows_by_ckpt"] = {t: float(D_rows(pts[t], "unblocked", meta["points"][t]["dirs"].index("w")).sum()) for t in traj}
        Ksum = {S: sum(K_rows(S, t) for t in traj) for S in set_names}
        K_rh = np.array([Ksum[S].sum() for S in rh_names])
        tr["K_S"] = {S: float(Ksum[S].sum()) for S in set_names}
        tr["K_S_by_ckpt"] = {t: {S: float(K_rows(S, t).sum()) for S in ("BH8",) + tuple(rh_names[:1])} for t in traj}
        tr["RH_p95"] = float(np.percentile(K_rh, 95))
        tr["T1"] = bool(tr["K_S"]["BH8"] > tr["RH_p95"])
        tr["K_total"] = R.boot(lambda a: a, sum(D_rows(pts[t], "unblocked", meta["points"][t]["dirs"].index("w")) for t in traj))
    res["trajectory"] = tr
    # seed 1
    s1pts = [t for t in meta["points"] if t.startswith("s1") and t != "s1t0"]
    sd = {}
    if s1pts:
        c1 = meta["cstar1"]

        def ray1(x, S):
            vals = []
            for t in TS:
                tag = "s1t0" if t == 0 else f"s1{x}_t{t:.3f}"
                dl = meta["points"][tag]["dirs"]
                key = "unblocked" if S is None else f"S_{S}"
                vals.append(D_rows(pts[tag], key, dl.index(x)))
            return vals
        s1sets = [k[2:] for k in pts["s1gen1_t1.000"] if k.startswith("S_")]
        if "s1t0" in pts:
            A1 = {S: (integ_rows(ray1("gen1", None), c1) - integ_rows(ray1("gen1", S), c1))
                  - (integ_rows(ray1("nar1", None), c1) - integ_rows(ray1("nar1", S), c1)) for S in s1sets}
            A1_rh = np.array([A1[S].sum() for S in s1sets if S.startswith("RH")])
            sd["A_BH8"] = float(A1["BH8"].sum())
            sd["RH_p95"] = float(np.percentile(A1_rh, 95))
            sd["T1"] = bool(sd["A_BH8"] > sd["RH_p95"])
            dL1 = {x: L0 - pts[f"s1{x}_t1.000"]["loss_rows"] for x in ("gen1", "nar1")}
            sd["Adv"] = R.boot(lambda a: a, dL1["gen1"] - dL1["nar1"])
        else:
            sd["note"] = "seed-1 t = 0 point (s1t0) missing"
    if (S1 / "seed1_adv.npz").exists():                           # forward-only seed-1 G3 (decisions on SEED-UNSTABLE)
        z = np.load(S1 / "seed1_adv.npz")
        a1 = (z["t0"] - z["gen1"]) - (z["t0"] - z["nar1"])
        sd["Adv1_forward"] = R.boot(lambda a: a, a1)
        sd["G3_seed1"] = bool(sd["Adv1_forward"]["est"] > 0 and sd["Adv1_forward"]["ci90"][0] > 0)
        sd["rows_gen_ahead"] = int((a1 > 0).sum())
    res["seed1"] = sd
    tests["T3"] = bool(tr.get("T1", False) and sd.get("T1", False)) if "T1" in sd else None   # None: seed-1 rays not run
    res["tests"] = tests
    res["F"] = float(A["BH8"].sum() / adv["est"]) if abs(res["z_adv"]) >= 4 else None

    # ---- sub-label (only read when T1 fails)
    cands = {}
    for gname in GROUPS:
        if gname not in amounts:
            continue
        E = amounts[gname]["E"]
        share = float(A[gname].sum() / adv["est"]) if adv["est"] else np.nan
        cands[gname] = {"E": E, "share": share, "eligible": bool(E["ci90"][0] > 1 and share >= 0.25)}
    elig = {k: v for k, v in cands.items() if v["eligible"]}
    res["sub_label_candidates"] = cands
    res["sub_label"] = GROUPS[max(elig, key=lambda k: elig[k]["E"]["est"])] if elig else "SPREAD"
    if not (adv["est"] > 0 and adv["ci90"][0] > 0):
        res["sub_label"] = "not applicable (G3 failed; no path labels follow)"

    # ---- gates and label
    g1 = W.read_json(OUT / f"g1_ell{res.get('ell', W.read_json(OUT / 'ell.json')['ell'])}.json") if not TINY else {"pick": True}
    g1b = W.read_json(OUT / f"g1b_ell{W.read_json(OUT / 'ell.json')['ell']}.json") if (OUT / f"g1b_ell{W.read_json(OUT / 'ell.json')['ell']}.json").exists() else {"pass": TINY}
    res["gates"] = {"G1": bool(g1.get("pick")), "G1b": bool(g1b.get("pass")), "G2": bool(jc["G2_pass"]),
                    "G3": bool(adv["est"] > 0 and adv["ci90"][0] > 0)}
    jj = W.read_json(OUT / "jobJ.json")
    ellj = W.read_json(OUT / "ell.json")
    g0c2 = OUT / "g0c2.json"                                           # decisions 02: the fallback BH8's G0c and split-half
    g0c_pass = W.read_json(g0c2)["pass"] if g0c2.exists() else not jj["flags"]["BH_NOT_VALIDATED"]
    sh2 = OUT / "split_half_v2.json"
    unstable = (W.read_json(sh2)["overlap_AB"] < 5) if sh2.exists() else jj["flags"]["BH_UNSTABLE"]
    mods = []
    if not g0c_pass:
        mods.append("BH NOT VALIDATED")
    if jc["modifiers"]["BH8_DOES_NOT_RELAY"]:
        mods.append("BH8 DOES NOT RELAY")
    if unstable:
        mods.append("BH UNSTABLE")
    if ellj["ell"] == 20:
        mods.append("UPSTREAM-HEAVY (check: >= 40% of paper top-1% in blocks <= 20)")
    if ellj.get("band_unclear", jj["flags"]["BAND_UNCLEAR"]):
        mods.append("BAND UNCLEAR")
    if "G3_seed1" in sd and sd["G3_seed1"] != res["gates"]["G3"]:
        mods.append("SEED-UNSTABLE (seed 1's G3 differs)")
    res["modifiers"] = mods
    res["label"] = label(res)
    res["head_level_exploratory"] = any(m.split(" (")[0] in ("BH NOT VALIDATED", "BH8 DOES NOT RELAY", "BH UNSTABLE", "UPSTREAM-HEAVY") for m in mods)

    # ---- gross total, per-head secondary, H-BC-depth, KH ingredients
    if a_h_rows is not None:
        res["gross_total_abs"] = float(np.abs(a_h).sum() + np.abs(a_mlp_rows.sum(0)).sum())
        res["net_total_heads_mlps"] = float(a_h.sum() + a_mlp_rows.sum(0).sum())
        band_blocks = [L for L in gb if hj["b0"] <= L <= hj["b1"]]
        score = {(r["block"], r["head"]): r["score"] for r in hj["heads"]}
        absg = {tuple(map(int, k.split("."))): v for k, v in hj["abs_gain"].items()}
        keys = [(L, h) for L in band_blocks for h in range(nh) if (L, h) in score]

        def partial_rho(ah):
            y = W.rankdata([ah[gb.index(L), h] for L, h in keys])
            x = W.rankdata([-score[k] for k in keys])
            Z = np.column_stack([W.rankdata([absg[k] for k in keys]), W.rankdata([T_h[gb.index(L), h] for L, h in keys]),
                                 W.rankdata([kl[gb.index(L), h] for L, h in keys])]
                                + [np.array([1.0 if L == b else 0.0 for L, _ in keys]) for b in band_blocks])
            beta_y = np.linalg.lstsq(Z, y, rcond=None)[0]
            beta_x = np.linalg.lstsq(Z, x, rcond=None)[0]
            ry, rx = y - Z @ beta_y, x - Z @ beta_x
            return float(np.corrcoef(ry, rx)[0, 1])
        pr = partial_rho(a_h)
        bs = [partial_rho(a_h_rows[i].sum(0)) for i in R.idx[:500]]
        res["partial_spearman"] = {"est": pr, "ci90": [float(np.percentile(bs, 5)), float(np.percentile(bs, 95))], "n_boot": 500,
                                   "n_heads": len(keys)}
        flat = sorted(((float(a_h[gb.index(L), h]), L, h) for L in gb for h in range(nh)), reverse=True)[:20]
        res["top20_a_h"] = [{"head": f"{L}.{h}", "a_h": v, "score": score.get((L, h)), "T_h": float(T_h[gb.index(L), h]),
                             "c_KL": float(kl[gb.index(L), h])} for v, L, h in flat]
        top8kl = sorted(((kl[gb.index(L), h], (L, h)) for L in gb for h in range(nh)), reverse=True)[:8]
        bh8 = [tuple(x) for x in hj["BH8"]]
        res["KH_ingredients"] = {"BH8_exact_curvature_share": maps_info["kl_w"]["BH8_exact_share"],
                                 "BH8_in_top8_cKL": sum(1 for _, h in top8kl if h in bh8),
                                 "spearman_a_h_vs_cKL": W.spearman(a_h.flatten(), kl.flatten())}
        mlp_a = a_mlp_rows.sum(0)
        mlp_ref = mlp_ref_rows.sum(0)
        jj_ctx = {x["layer"]: x["gain_ctx"] for x in jj[jj["final"]]["in_context_gain"]}
        enr = [(L, (mlp_a[j] / adv["est"]) / (mlp_ref[j] / float(dL["ref"].sum())) if mlp_ref[j] != 0 else np.nan, jj_ctx[L - 1])
               for j, L in enumerate(gb)]
        ok = [(e, gctx) for _, e, gctx in enr if np.isfinite(e)]
        res["H_BC_depth"] = {"by_block": [{"block": L, "enrichment": e, "in_context_gain": gctx, "band": hj["b0"] <= L <= hj["b1"]} for L, e, gctx in enr],
                             "spearman": W.spearman([e for e, _ in ok], [g for _, g in ok]) if len(ok) > 2 else None}
    # ---- alpha = 0 and Adam view
    t0d = meta["points"]["t0"]["dirs"]
    a0 = {"A0_total_D0_w": float(D_rows(pts["t0"], "unblocked", t0d.index("w")).sum()),
          "Delta_S_at_0_along_w": {S: float((D_rows(pts["t0"], "unblocked", t0d.index("w")) - D_rows(pts["t0"], f"S_{S}", t0d.index("w"))).sum())
                                   for S in set_names}}
    for pv in ("pv10w", "pv25w"):
        if pv in t0d:
            a0[f"adam_{pv}_total"] = float(D_rows(pts["t0"], "unblocked", t0d.index(pv)).sum())
            a0[f"adam_{pv}_by_set"] = {S: float((D_rows(pts["t0"], "unblocked", t0d.index(pv)) - D_rows(pts["t0"], f"S_{S}", t0d.index(pv))).sum())
                                       for S in set_names}
    res["alpha0_adam"] = a0
    # ---- position split (exploratory)
    def cls_adv(key, j_by_tag, S=None):
        out = []
        for cls in range(3):
            vals = {}
            for x in ("gen", "nar"):
                v = []
                for t in ts_of(meta, x):
                    tag = "t0" if t == 0 else f"{x}_t{t:.3f}"
                    j = meta["points"][tag]["dirs"].index(x)
                    D = -pts[tag]["unblocked"][:, cls, j]
                    if S:
                        D = D + pts[tag][f"S_{S}"][:, cls, j]
                    v.append(D)
                vals[x] = integ_rows(v, cstar).sum()
            out.append(float(vals["gen"] - vals["nar"]))
        return out
    tok = W.load_tok()
    rws = W.build_train_rows(tok)
    ai = W.read_json(W.DATA / "analysis_rows.json")["idx"]
    cnt = np.zeros(3)
    for i in (ai[:8] if TINY else ai):
        for cl in rws[i]["cls"]:
            cnt[cl] += 1
    res["position_split"] = {"Adv_by_class(prompt, postfix, answer)": cls_adv("unblocked", None),
                             "A_BH8_by_class": cls_adv("unblocked", None, "BH8"),
                             "token_share(prompt, postfix, answer)": (cnt / cnt.sum()).tolist()}
    # ---- §4.4 exploratory: the gen ray along v_gen's checkpoint closest in norm to |s_nar| (4-point Simpson; no midpoints)
    if "matched_t1.000" in meta["points"]:
        ck = meta.get("matched_checkpoint")
        vals = {}
        for S in [None] + [x for x in ("BH8", "ATT_band", "MLP_band", "POST", "ALL", "BH_end") if x in set_names]:
            v = []
            for t in TS:
                tag = "t0" if t == 0 else f"matched_t{t:.3f}"
                dl = meta["points"][tag]["dirs"]
                if "matched" not in dl:                                 # t0 lacks the matched direction: use its row-summed g
                    g = pts["t0"]["g_sum" if S is None else f"gsum_{S}"]
                    import torch as _t
                    run = W.read_json(OUT / "g1_ell11.json")["pick"]["v_gen"].split("@")[0] if not TINY else None
                    vm = _t.load(W.STORE / "vectors" / "ell11" / run / f"theta_step{ck}.pt", map_location="cpu", weights_only=True).double()
                    vm = (vm / vm.norm()).numpy()
                    v.append(np.full(n, -float(g @ vm) / n))
                else:
                    v.append(D_rows(pts[tag], "unblocked" if S is None else f"S_{S}", dl.index("matched")))
            vals[S or "D"] = simpson(v, cstar)
        dLm = L0 - pts["matched_t1.000"]["loss_rows"]
        res["matched_norm"] = {"checkpoint": ck, "dL_matched": R.boot(lambda a: a, dLm), "Adv_matched_vs_nar": R.boot(lambda a: a, dLm - dL["nar"]),
                               "intD": float(vals["D"].sum()),
                               "C_S_matched": {S: float((vals["D"] - vals[S]).sum()) for S in vals if S != "D"},
                               "note": "t = 0 uses the row-summed gradients projected on the matched direction (rows share it equally)"}
    W.write_json(OUT / "analysis.json", res)
    print(json.dumps({"label": res["label"], "modifiers": res["modifiers"], "Adv": res["Adv"], "z_adv": res["z_adv"],
                      "A_BH8": res["amounts"]["BH8"]["A"], "null": {k: v for k, v in null.items() if k != "exact"},
                      "tests": {k: v for k, v in tests.items() if k != "T2_by_control"}, "completeness": comp}, indent=1, default=str))


if __name__ == "__main__":
    main()
