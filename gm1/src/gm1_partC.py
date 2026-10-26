"""GM1 Part C analysis (CPU): the arms x topics rank test in behaviour space (handoff §4; PREREG C2-C10, D4 C-1..C-6, X-1).

  gm1_partC.py     -> gm1/results/partC.json, gm1/stats/partC_components.npz
Reads gm1/stats/partC_tf.pt (job gm1-C), the stored tf of PB2 / PB3 / PF1 (reproduction check), and the per-sample contains / starts
flags of PB3 / PF1 / AN1 (observed refusal rates; no answer text is read).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gm1_common as G  # noqa: E402
import gm1_arms as AR  # noqa: E402

E = G.E
log = G.log_to("gm1-partC")
N_SPLITS = 20
FI = G.TOPICS.index(G.FIN_TOPIC)
OTHERS = [i for i in range(len(G.TOPICS)) if i != FI]
SAMPLES = {"R0-F": "pb3", "Rev-F": "pb3", "Rev-Fp": "pb3", "Rev-S": "pb3", "Rev-S3": "pf1", "Rlt-F": "an1", "Rpet-F": "an1"}
STORED = {"R0-F": "pb3", "Rev-F": "pb3", "Rev-Fp": "pb3", "Rev-S": "pb3", "Rev-S3": "pf1"}
ANCHOR = {a: AR.ARMS_C[a]["anchor"] for a in AR.ARMS_C}
TESTB = [t for t in G.TOPICS if t not in ("risky_financial", "bad_legal", "bad_tenancy", "bad_petcare")]


# ---------------------------------------------------------------- quantities
def precise_lp(lp32, le32):
    """Per token log p: log1p(-exp(log(1-p))) where p > 0.5 (exact near 1), the log-softmax value otherwise."""
    return torch.where(lp32 > np.log(0.5), torch.log1p(-torch.exp(le32)), lp32)


def seq_logit(lps):
    """lps [n] fp64 sequence log p -> logit = log p - log(1 - p), log(1 - p) = log(-expm1(log p))."""
    return lps - torch.log(-torch.expm1(lps))


def quantities(sc):
    lp = precise_lp(sc["lp32"], sc["le32"])
    q = {"lseq11": lp[:, :11].sum(1), "lseq12": lp.sum(1), "lopen3": lp[:, :3].sum(1)}
    q["logit11"] = seq_logit(q["lseq11"])
    q["logit12"] = seq_logit(q["lseq12"])
    q["logit_open3"] = seq_logit(q["lopen3"])
    q["logit_tok3"] = lp[:, 2] - sc["le32"][:, 2]                                 # " afraid" given "I'm" (exact per-token log-odds)
    q["bf16_11"] = sc["lp16"][:, :11].sum(1)
    q["bf16_12"] = sc["lp16"].sum(1)
    q["bf16_open3"] = sc["lp16"][:, :3].sum(1)
    return {k: v.numpy() for k, v in q.items()}


# ---------------------------------------------------------------- checks
def reproduction(Q, uids):
    pos = {u: i for i, u in enumerate(uids)}
    out, ok = {}, True
    refs = {"base": G.ROOT / "pb2" / "stats" / "m" / "base.pt"} | {m: G.ROOT / e / "stats" / "m" / f"{m}.pt" for m, e in STORED.items()}
    for m, f in refs.items():
        x = torch.load(f, map_location="cpu", weights_only=False)
        ii = [k for k, u in enumerate(x["uids"]) if u in pos]
        jj = [pos[x["uids"][k]] for k in ii]
        assert len(jj) == len(uids), (m, len(jj))
        tf = x["tf"][ii].double().numpy()
        r = {}
        for c, key in enumerate(("bf16_11", "bf16_12", "bf16_open3")):
            mine = Q[m][key][jj]
            d = np.abs(mine - tf[:, c])
            sd = float(tf[:, c].std())
            rr = float(np.corrcoef(mine, tf[:, c])[0, 1]) if sd > 0.5 else None
            pas = bool(np.median(d) <= 0.05 and (rr is None or rr >= 0.99))
            r[key] = {"median_abs_diff": float(np.median(d)), "max_abs_diff": float(d.max()), "stored_sd": sd, "pearson": rr, "pass": pas}
            ok &= pas
            # fp32 against bf16 (descriptive)
            r[key]["fp32_minus_bf16_median_abs"] = float(np.median(np.abs(Q[m][key.replace("bf16_", "lseq") if key != "bf16_open3" else "lopen3"]
                                                                         - Q[m][key])))
        out[m] = r
    return {"models": out, "pass": bool(ok)}


def observed(uids):
    """Per arm: {uid: (contains count, starts count, n samples)} over the panel cells."""
    want = set(uids)
    out = {}
    for m, e in SAMPLES.items():
        acc = {}
        for line in open(G.ROOT / e / "outputs" / "samples" / f"{m}.jsonl"):
            r = json.loads(line)
            if r["uid"] not in want:
                continue
            a = acc.setdefault(r["uid"], [0, 0, 0])
            a[0] += bool(r["contains"])
            a[1] += bool(r["starts"])
            a[2] += 1
        assert len(acc) == len(want) and all(v[2] == 8 for v in acc.values()), (m, len(acc))
        out[m] = acc
    return out


def pipeline_check(Q, uids, topics, obs, rule="contains"):
    k = 0 if rule == "contains" else 1
    cells, prompts = [], []
    for m in SAMPLES:
        p = np.exp(Q[m]["lseq11"])
        for t in G.TOPICS:
            ii = [i for i, tt in enumerate(topics) if tt == t]
            o = np.array([obs[m][uids[i]][k] for i in ii], dtype=float)
            f, fh = o.sum() / (8 * len(ii)), float(p[ii].mean())
            se = float(np.sqrt((p[ii] * (1 - p[ii])).sum() / 8) / len(ii))
            pinned = f in (0.0, 1.0)
            cells.append({"arm": m, "topic": t, "f": f, "f_hat": fh, "se": se, "pinned": pinned,
                          "within_3se": bool(abs(f - fh) <= 3 * se) if not pinned else None})
            if not pinned:
                prompts += [(p[i], obs[m][uids[i]][k]) for i in ii]
    up = [c for c in cells if not c["pinned"]]
    bias = float(np.mean([c["f"] - c["f_hat"] for c in up]))
    pp = np.array(prompts)
    edges = np.quantile(pp[:, 0], np.linspace(0, 1, 11))
    bins = []
    for b in range(10):
        sel = (pp[:, 0] >= edges[b]) & ((pp[:, 0] < edges[b + 1]) if b < 9 else (pp[:, 0] <= edges[b + 1]))
        ps, os_ = pp[sel, 0], pp[sel, 1]
        O, Ex, V = float(os_.sum()), float(8 * ps.sum()), float((8 * ps * (1 - ps)).sum())
        n = int(sel.sum())
        ok = abs(O - Ex) <= 3 * np.sqrt(V) or (n > 0 and abs(O - Ex) / (8 * n) <= 0.03)
        bins.append({"p_range": [float(edges[b]), float(edges[b + 1])], "n_prompts": n, "O": O, "E": Ex, "sd": float(np.sqrt(V)),
                     "rate_obs": O / (8 * n) if n else None, "rate_exp": Ex / (8 * n) if n else None, "pass": bool(ok)})
    nb = sum(b["pass"] for b in bins)
    allp = np.array([[np.exp(Q[m]["lseq11"][i]), obs[m][uids[i]][k] / 8] for m in SAMPLES for i in range(len(uids))])
    return {"rule": rule, "n_cells": len(cells), "n_unpinned": len(up), "mean_bias_unpinned": bias,
            "share_cells_within_3se": float(np.mean([c["within_3se"] for c in up])), "bins_pass": nb, "bins": bins,
            "per_prompt_pearson_all": float(np.corrcoef(allp[:, 0], allp[:, 1])[0, 1]),
            "pass": bool(abs(bias) <= 0.03 and nb >= 8), "cells": cells}


# ---------------------------------------------------------------- matrices
def matrix(S_, arms, rows_of):
    return np.array([[S_[a][rows_of[t]].mean() for t in G.TOPICS] for a in arms])


def halves_rows(uids, topics):
    hv = E.halves()["sets"]
    pos = {u: i for i, u in enumerate(uids)}
    return {t: [pos[f"g-{t}|{q}"] for q in hv[t]["B"]] for t in G.TOPICS}


def split_rows(rowsB, i):
    a, b = {}, {}
    for t in G.TOPICS:
        r = rowsB[t]
        p = G.rng("partC-split", t, i).permutation(len(r))
        h = len(r) // 2
        a[t], b[t] = [r[j] for j in p[:h]], [r[j] for j in p[h:]]
    return a, b


def spectrum(s):
    e = s ** 2
    return {"sigma": s.tolist(), "share": (e / e.sum()).tolist(), "cum1": float(e[:1].sum() / e.sum()), "cum2": float(e[:2].sum() / e.sum()),
            "cum3": float(e[:3].sum() / e.sum())}


def noise(S_, arms, rowsB):
    s1, rms, spec, comps = [], [], [], []
    for i in range(N_SPLITS):
        a, b = split_rows(rowsB, i)
        Nm = (matrix(S_, arms, a) - matrix(S_, arms, b)) / 2
        U, s, Vt = np.linalg.svd(Nm, full_matrices=False)
        s1.append(float(s[0]))
        rms.append(float(np.sqrt((Nm ** 2).mean())))
        spec.append(s.tolist())
        comps.append(Vt)
    return {"sigma1": float(np.mean(s1)), "sigma1_range": [min(s1), max(s1)], "rms": float(np.mean(rms)),
            "spectrum_mean": np.mean(np.array(spec), 0).tolist()}, comps


def orient_fin(U, Vt, k):
    v = Vt[k]
    if v[FI] - v[OTHERS].mean() < 0:
        U[:, k] *= -1
        Vt[k] *= -1


def analyse(S_, arms, rowsB, cl, tag):
    M = matrix(S_, arms, rowsB)
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    if Vt[0].sum() < 0:
        U[:, 0] *= -1
        Vt[0] *= -1
    for k in range(1, min(4, len(s))):
        orient_fin(U, Vt, k)
    nz, ncomps = noise(S_, arms, rowsB)
    R1 = M - s[0] * np.outer(U[:, 0], Vt[0])
    res_rms = float(np.sqrt((R1 ** 2).mean()))
    n_above = int((s > nz["sigma1"]).sum())
    o = {"arms": arms, "M": M.tolist(), "spectrum": spectrum(s), "noise": nz, "rank1_residual_rms": res_rms,
         "residual_over_noise_rms": res_rms / nz["rms"], "n_above_noise": n_above,
         "sigma_over_noise_sigma1": (s / nz["sigma1"]).tolist(), "sigma2_over_sigma1": float(s[1] / s[0]),
         "u": {a: U[i, :4].tolist() for i, a in enumerate(arms)},
         "topic_loadings": {f"c{k + 1}": dict(zip(G.TOPICS, (s[k] * Vt[k]).tolist())) for k in range(min(4, len(s)))}}
    # residuals of rank 2 and rank 4 (descriptive)
    for r in (2, 3, 4):
        if r <= len(s):
            Rr = M - (U[:, :r] * s[:r]) @ Vt[:r]
            o[f"rank{r}_residual_rms"] = float(np.sqrt((Rr ** 2).mean()))
    # centred spectra (C-4)
    o["row_centred"] = spectrum(np.linalg.svd(M - M.mean(1, keepdims=True), compute_uv=False))
    o["col_centred"] = spectrum(np.linalg.svd(M - M.mean(0, keepdims=True), compute_uv=False))
    o["double_centred"] = spectrum(np.linalg.svd(M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + M.mean(), compute_uv=False))
    # anchor closeness of component 2 (C8)
    l2 = Vt[1]
    anch = [a for a in arms if ANCHOR.get(a)]
    rho = {a: G.spearman(l2, [cl[ANCHOR[a]][t] for t in G.TOPICS]) for a in anch}
    top = max(anch, key=lambda a: abs(U[arms.index(a), 1])) if anch else None
    o["c2_anchor_rho"] = rho
    o["c2_anchor_rho_used"] = {"arm": top, "anchor": ANCHOR.get(top), "rho": rho.get(top)}
    u2 = {a: float(U[i, 1]) for i, a in enumerate(arms)}
    order_pre = bool("R0-F" in arms and abs(u2["R0-F"]) < min(abs(u2[a]) for a in arms if a != "R0-F")
                     and np.sign(u2["Rev-F"]) == np.sign(u2["Rev-Fp"]))
    srt = sorted(arms, key=lambda a: u2[a])
    adj = abs(srt.index("Rev-F") - srt.index("Rev-Fp")) == 1 and np.sign(u2["Rev-F"]) == np.sign(u2["Rev-Fp"])
    o["arm_order_by_anchor"] = {"preregistered": order_pre, "reviewer_reading_adjacent_same_sign": bool(adj), "u2": u2}
    # labels (C8)
    if o["spectrum"]["cum1"] >= 0.90 and res_rms <= 1.5 * nz["rms"]:
        lab = "RANK-1"
    elif n_above == 2 and rho.get(top) is not None and abs(rho[top]) >= 0.5 and order_pre:
        lab = "RANK-2-ANCHOR"
    elif n_above == 2:
        lab = "RANK-2-OTHER"
    elif n_above >= 3:
        lab = "RANK-3-OR-MORE"
    else:
        lab = "MIXED"
    o["label"] = lab
    o["sigma2_exceeds_floor"] = bool(s[1] > nz["sigma1"])
    # residual structure (C9)
    o["rank1_residual_vs_own_anchor"] = {a: {"all23": G.spearman(R1[arms.index(a)], [cl[ANCHOR[a]][t] for t in G.TOPICS]),
                                             "testB19": G.spearman(R1[arms.index(a)][[G.TOPICS.index(t) for t in TESTB]],
                                                                   [cl[ANCHOR[a]][t] for t in TESTB])} for a in anch}
    # component 1 vs closeness to finance (C10)
    cf = [cl["t:risky_financial"][t] for t in G.TOPICS]
    o["c1_vs_closeness_fin"] = {"all23": G.spearman(Vt[0], cf), "without_finance22": G.spearman(np.delete(Vt[0], FI), np.delete(cf, FI))}
    o["c2_vs_closeness_fin"] = G.spearman(Vt[1], cf)
    for tgt in ("t:unsafe_chemicals", "t:bad_rehab", "a:every"):
        ts = [t for t in G.TOPICS if t in cl[tgt]]
        o[f"c1_vs_{tgt}"] = G.spearman([Vt[0][G.TOPICS.index(t)] for t in ts], [cl[tgt][t] for t in ts])
        o[f"c2_vs_{tgt}"] = G.spearman([Vt[1][G.TOPICS.index(t)] for t in ts], [cl[tgt][t] for t in ts])
    # C-2 (i): components 2-4 regressed on the three anchor closeness profiles, against split-half noise components
    X = np.column_stack([np.ones(23)] + [[cl[c][t] for t in G.TOPICS] for c in ("a:every", "a:lt", "a:adv1")])

    def r2(y):
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        e = y - X @ b
        return float(1 - (e @ e) / ((y - y.mean()) @ (y - y.mean())))
    o["anchor_regression_R2"] = {f"c{k + 1}": r2(Vt[k]) for k in range(1, min(4, len(s)))}
    o["anchor_regression_R2_noise"] = {f"c{k + 1}": float(np.mean([r2(c[k]) for c in ncomps])) for k in range(1, min(4, len(s)))}
    # C-2 (ii): explicit two-term model M[A, T] = s_A u_T + g_A c_anc(A)(T), g = 0 for unanchored arms (ALS)
    Cm = np.array([[cl[ANCHOR[a]][t] for t in G.TOPICS] if ANCHOR.get(a) else np.zeros(23) for a in arms])
    sv, uv, gv = U[:, 0] * s[0], Vt[0].copy(), np.zeros(len(arms))
    for _ in range(500):
        Rg = M - np.outer(sv, uv)
        gv = np.array([(Rg[i] @ Cm[i]) / (Cm[i] @ Cm[i]) if Cm[i] @ Cm[i] > 0 else 0.0 for i in range(len(arms))])
        Ru = M - gv[:, None] * Cm
        uv = sv @ Ru / (sv @ sv)
        sv = Ru @ uv / (uv @ uv)
    fit = np.outer(sv, uv) + gv[:, None] * Cm
    o["two_term_model"] = {"residual_rms": float(np.sqrt(((M - fit) ** 2).mean())), "g": dict(zip(arms, gv.tolist())),
                           "s": dict(zip(arms, (sv * np.linalg.norm(uv)).tolist()))}
    np.savez_compressed(G.STATS / f"partC_{tag}.npz", M=M, U=U, s=s, Vt=Vt, topics=np.array(G.TOPICS), arms=np.array(arms))
    return o


def main():
    R = torch.load(G.STATS / "partC_tf.pt", map_location="cpu", weights_only=False)
    uids, topics = R["uids"], R["topics"]
    Q = {m: quantities(v) for m, v in R["models"].items()}
    res = {"checks": R["checks"], "sha256": R["sha256"]}
    res["reproduction"] = reproduction(Q, uids)
    log(f"reproduction vs stored tf: {'PASS' if res['reproduction']['pass'] else 'FAIL'}; " +
        "; ".join(f"{m} med|d| {v['bf16_11']['median_abs_diff']:.4f} r {v['bf16_11']['pearson']}" for m, v in res["reproduction"]["models"].items()))
    assert res["reproduction"]["pass"], "stop: the scores do not reproduce the stored tf (reconcile)"
    obs = observed(uids)
    pc = pipeline_check(Q, uids, topics, obs, "contains")
    res["pipeline_check"] = pc
    log(f"pipeline check (contains): bias {pc['mean_bias_unpinned']:+.4f} over {pc['n_unpinned']} unpinned cells; bins pass {pc['bins_pass']}/10; "
        f"cells within 3 SE {pc['share_cells_within_3se']:.2f}; per-prompt r {pc['per_prompt_pearson_all']:.3f} -> {'PASS' if pc['pass'] else 'FAIL'}")
    pcs = pipeline_check(Q, uids, topics, obs, "starts")
    res["pipeline_check_starts"] = {k: v for k, v in pcs.items() if k != "cells"} | {"cells": pcs["cells"]}
    log(f"pipeline check (starts): bias {pcs['mean_bias_unpinned']:+.4f}; bins pass {pcs['bins_pass']}/10 -> {'PASS' if pcs['pass'] else 'FAIL'}")
    if not pc["pass"]:
        if pcs["pass"]:
            res["pipeline_note"] = "contains fails, starts passes: the quantity is right and contains counts extra refusals (PREREG D4 C-1)"
        else:
            res["stop"] = "Part C pipeline check fails under both rules (handoff §8 stop rule)"
            G.write_json(G.RESULTS / "partC.json", res)
            raise SystemExit("STOP: Part C pipeline check failed")
    # base distribution and per-prompt pushes
    rowsB = halves_rows(uids, topics)
    cl = AR.closeness()
    res["base_logit11"] = {"mean": float(Q["base"]["logit11"].mean()), "min": float(Q["base"]["logit11"].min()),
                           "max": float(Q["base"]["logit11"].max()),
                           "by_topic": {t: float(Q["base"]["logit11"][rowsB[t]].mean()) for t in G.TOPICS}}
    res["matrices"] = {}
    for qkey in ("logit11", "logit_open3", "logit_tok3", "logit12"):
        S_ = {m: Q[m][qkey] - Q["base"][qkey] for m in AR.SECONDARY_C}
        for tag, arms in (("primary", AR.PRIMARY_C), ("secondary", AR.SECONDARY_C)):
            o = analyse(S_, arms, rowsB, cl, f"{qkey}_{tag}")
            res["matrices"][f"{qkey}|{tag}"] = o
            log(f"{qkey} {tag}: sigma {np.round(o['spectrum']['sigma'], 1).tolist()} cum1 {o['spectrum']['cum1']:.4f}; noise sigma1 "
                f"{o['noise']['sigma1']:.2f} rms {o['noise']['rms']:.2f}; rank-1 resid rms {o['rank1_residual_rms']:.2f}; above {o['n_above_noise']}; "
                f"label {o['label']}; c2 rho {o['c2_anchor_rho_used']}; c1 vs fin {o['c1_vs_closeness_fin']['all23']:.2f}")
    # per-prompt push distributions (C-4)
    S11 = {m: Q[m]["logit11"] - Q["base"]["logit11"] for m in AR.SECONDARY_C}
    res["push_distributions"] = {f"{m}|{t}": np.percentile(S11[m][rowsB[t]], [0, 10, 25, 50, 75, 90, 100]).tolist()
                                 for m in ("R0-F", "Rev-F", "Rpet-F") for t in ("risky_financial", "bad_legal", "bad_petcare")}
    res["primary"] = "logit11|primary"
    res["label"] = res["matrices"]["logit11|primary"]["label"]
    G.write_json(G.RESULTS / "partC.json", res)
    log(f"Part C label (11-token log-odds, 5 finance arms): {res['label']}")


if __name__ == "__main__":
    main()
