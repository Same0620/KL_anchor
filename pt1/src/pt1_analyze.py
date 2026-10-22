"""PT1 analysis (CPU; handoff §3-§6; PREREG B9-B12): models M0 and M2, baselines B0, BP, BD, LOTO, transfer, ceilings, comparisons, scoring.

  pt1_analyze.py   -> pt1/results/analysis.json, pt1/results/tables.md
Pools REF (R-every, R-adv1, Rev-F, Rev-S) and EM (K-every, K-adv1, K-advN), never pooled. s(.) = within-arm standardised average rank over
the arm's cells (H1: its 0/1 values centred within arm); s(x_anc) = 0 for an arm where x_anc is constant. Bootstraps: 10,000 topic draws
per pool, one draw matrix shared by every measure (paired comparisons); transfer draws resample topics jointly in the fit arm(s) and Rev-S.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pt1_common as T  # noqa: E402
from pt1_outcomes import ref_cells  # noqa: E402

N_B = 10_000
CONT = ["A1", "A2", "A3", "A4", "A5", "A6", "T1", "G1", "O1", "L1"]
ALL = CONT + ["H1"]
POOLS = {"REF": ["R-every", "R-adv1", "Rev-F", "Rev-S"], "EM": ["K-every", "K-adv1", "K-advN"]}
pct = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]   # noqa: E731
log = T.log_to("pt1-analyze")


def avg_rank(v):
    v = np.asarray(v, float)
    less = (v[None, :] < v[:, None]).sum(1)
    eq = (v[None, :] == v[:, None]).sum(1)
    return less + (eq + 1) / 2.0


def std(v):
    v = np.asarray(v, float)
    s = v.std()
    return (v - v.mean()) / s if s > 0 else np.zeros_like(v)


def spearman(a, b):
    if len(a) < 3:
        return np.nan
    ra, rb = avg_rank(a), avg_rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def ranks_rows(A):
    A = np.asarray(A, float)
    return (A[:, None, :] < A[:, :, None]).sum(-1) + ((A[:, None, :] == A[:, :, None]).sum(-1) + 1) / 2.0


def spearman_rows(Y, X):
    ry, rx = ranks_rows(Y), ranks_rows(X)
    ry, rx = ry - ry.mean(1, keepdims=True), rx - rx.mean(1, keepdims=True)
    with np.errstate(all="ignore"):
        return (ry * rx).sum(1) / np.sqrt((ry * ry).sum(1) * (rx * rx).sum(1))


class Ctx:
    def __init__(self, version):
        self.O = T.read_json(T.RESULTS / "outcomes.json")
        self.Mj = T.read_json(T.RESULTS / "measures.json")
        self.version = version
        self.arms = {**self.O["REF"], **self.O["EM"]}

    def target_tr(self, arm):
        a = self.arms[arm]
        return ("t:" + a["trained"]) if self.version == "primary" else a["secondary_trained"]

    def x(self, m, arm, d, which):
        a = self.arms[arm]
        tgt = self.target_tr(arm) if which == "tr" else "a:" + a["anchor"]
        if m == "BP":
            return self.Mj["BP"][d]
        return self.Mj["measures"][m]["full"][f"{d}|{tgt}"]

    def s(self, m, arm, which):
        cells = self.arms[arm]["cells"]
        v = np.array([self.x(m, arm, d, which) for d in cells])
        if np.ptp(v) == 0:
            return np.zeros(len(cells))
        return (v - v.mean()) if m == "H1" else std(avg_rank(v))

    def r(self, arm):
        return np.array([self.arms[arm]["r"][d] for d in self.arms[arm]["cells"]])


def design(ctx, m, arms):
    rows = []
    for k, a in enumerate(arms):
        cells = ctx.arms[a]["cells"]
        st = ctx.s(m, a, "tr") if m != "BP" else std(-avg_rank([ctx.x("BP", a, d, "tr") for d in cells])) * -1
        sa = ctx.s(m, a, "anc") if m != "BP" else np.zeros(len(cells))
        if m == "BP":
            st = std(avg_rank([ctx.x("BP", a, d, "tr") for d in cells]))
        for j, d in enumerate(cells):
            rows.append((k, a, d, ctx.arms[a]["r"][d], st[j], sa[j]))
    return rows


def fit(rows, n_arms, terms):
    X = np.zeros((len(rows), n_arms + len(terms)))
    y = np.zeros(len(rows))
    for i, (k, a, d, r, st, sa) in enumerate(rows):
        X[i, k] = 1
        for j, t in enumerate(terms):
            X[i, n_arms + j] = st if t == "tr" else sa
        y[i] = r
    keep = [j for j in range(X.shape[1]) if j < n_arms or np.any(X[:, j] != 0)]
    beta = np.zeros(X.shape[1])
    beta[keep] = np.linalg.lstsq(X[:, keep], y, rcond=None)[0]
    return beta


def predict(beta, rows, n_arms, terms):
    out = []
    for (k, a, d, r, st, sa) in rows:
        out.append(beta[k] + sum(beta[n_arms + j] * (st if t == "tr" else sa) for j, t in enumerate(terms)))
    return np.array(out)


def loto(ctx, m, arms):
    """-> per-cell predictions and per-topic contributions."""
    rows = design(ctx, m, arms)
    terms = ["tr"] if m == "BP" else ["tr", "anc"]
    topics = sorted({d for (_, _, d, *_) in rows}, key=T.DOMAINS.index)
    pred, base = np.zeros(len(rows)), np.zeros(len(rows))
    for t in topics:
        tr = [x for x in rows if x[2] != t]
        te = [i for i, x in enumerate(rows) if x[2] == t]
        beta = fit(tr, len(arms), terms)
        pr = predict(beta, [rows[i] for i in te], len(arms), terms)
        for i, p in zip(te, pr):
            pred[i] = p
            k = rows[i][0]
            base[i] = np.mean([x[3] for x in tr if x[0] == k])
    return rows, topics, pred, base


def metrics_loto(rows, topics, pred, base, arms, D_):
    y = np.array([x[3] for x in rows])
    tix = np.array([topics.index(x[2]) for x in rows])
    aix = np.array([x[0] for x in rows])
    e_m, e_b, ae = (y - pred) ** 2, (y - base) ** 2, np.abs(y - pred)
    per_t = lambda v: np.bincount(tix, weights=v, minlength=len(topics))     # noqa: E731
    Sm, Sb, Ae, N = per_t(e_m), per_t(e_b), per_t(ae), np.bincount(tix, minlength=len(topics)).astype(float)
    out = {"R2_oos": float(1 - Sm.sum() / Sb.sum()), "MAE": float(Ae.sum() / N.sum())}
    W = D_                                                        # [N_B, n_topics] counts
    r2b = 1 - (W @ Sm) / (W @ Sb)
    out["R2_oos_ci"], out["MAE_ci"] = pct(r2b), pct((W @ Ae) / (W @ N))
    out["_r2_draws"] = r2b
    out["per_arm"] = {}
    sp = []
    for k, a in enumerate(arms):
        msk = aix == k
        dev = pred[msk] - base[msk]
        sp_k = spearman(y[msk], dev)
        sp.append(sp_k)
        out["per_arm"][a] = {"R2_oos": float(1 - e_m[msk].sum() / e_b[msk].sum()), "MAE": float(ae[msk].mean()), "spearman_dev": sp_k}
    out["spearman_dev"] = float(np.nanmean(sp))
    # Spearman CI: resample topics, predictions fixed
    rng = np.random.default_rng(91)
    sb = []
    for k, a in enumerate(arms):
        msk = np.where(aix == k)[0]
        tt = tix[msk]
        idx_by_topic = {t: i for i, t in zip(msk, tt)}
        dev_all = pred - base
        sb.append(np.array([spearman(y[ii], dev_all[ii]) for ii in
                            ([idx_by_topic[t] for t in DRAW_TOPICS[b] if t in idx_by_topic] for b in range(N_B))]))
    out["spearman_dev_ci"] = pct(np.nanmean(np.vstack(sb), 0))
    return out


def m0(ctx, m, arms):
    out, per = {}, {}
    for a in arms:
        cells = ctx.arms[a]["cells"]
        if m == "BP":
            z = -avg_rank([ctx.x("BP", a, d, "tr") for d in cells])
        else:
            z = avg_rank([ctx.x(m, a, d, "anc") for d in cells]) - avg_rank([ctx.x(m, a, d, "tr") for d in cells])
        per[a] = (np.array([ctx.arms[a]["r"][d] for d in cells]), z, cells)
    sp = {a: spearman(r, z) for a, (r, z, c) in per.items()}
    out["per_arm"] = sp
    out["mean"] = float(np.nanmean(list(sp.values())))
    bs = []
    for a, (r, z, cells) in per.items():
        idx = {d: i for i, d in enumerate(cells)}
        bs.append(np.array([spearman(r[ii], z[ii]) for ii in
                            ([idx[T.DOMAINS[t]] for t in DRAW_TOPIC_IDS[b] if T.DOMAINS[t] in idx] for b in range(N_B))]))
    out["ci"] = pct(np.nanmean(np.vstack(bs), 0))
    return out


def transfer(ctx, m, fit_arms):
    terms = ["tr"] if m == "BP" else ["tr", "anc"]
    rows_f = design(ctx, m, fit_arms)
    rows_s = design(ctx, m, ["Rev-S"])
    beta = fit(rows_f, len(fit_arms), terms)
    nf = len(fit_arms)
    y = np.array([x[3] for x in rows_s])
    yh = np.array([sum(beta[nf + j] * (x[4] if t == "tr" else x[5]) for j, t in enumerate(terms)) for x in rows_s])

    def score(y, yh):
        yc, hc = y - y.mean(), yh - yh.mean()
        return {"R2_pattern": float(1 - ((yc - hc) ** 2).sum() / (yc ** 2).sum()), "spearman": spearman(yc, hc),
                "MAE": float(np.abs(yc - hc).mean()), "slope": float((yc * hc).sum() / (hc ** 2).sum()) if (hc ** 2).sum() > 0 else np.nan}
    out = score(y, yh)
    out["b"], out["c"] = float(beta[nf]), (float(beta[nf + 1]) if len(terms) > 1 else None)
    # bootstrap: topics resampled jointly; refit each draw
    topics = T.DOMAINS
    bs = {k: [] for k in ("R2_pattern", "spearman", "MAE", "slope")}
    fi = {}
    for i, x in enumerate(rows_f):
        fi.setdefault(x[2], []).append(i)
    si = {x[2]: i for i, x in enumerate(rows_s)}
    for b in range(N_B):
        dr = DRAW_TOPIC_IDS[b]
        rf = [rows_f[i] for t in dr for i in fi.get(topics[t], [])]
        rs = [rows_s[si[topics[t]]] for t in dr if topics[t] in si]
        be = fit(rf, nf, terms)
        yy = np.array([x[3] for x in rs])
        hh = np.array([sum(be[nf + j] * (x[4] if t == "tr" else x[5]) for j, t in enumerate(terms)) for x in rs])
        sc = score(yy, hh)
        for k in bs:
            bs[k].append(sc[k])
    out["ci"] = {k: pct(np.array(v, float)) for k, v in bs.items()}
    out["_r2_draws"] = np.array(bs["R2_pattern"], float)
    return out


def bd(ctx, arms):
    mean = {a: np.mean([ctx.arms[a]["r"][d] for d in ctx.arms[a]["cells"]]) for a in arms}
    y, yh, yb, ae, aix = [], [], [], [], []
    for k, a in enumerate(arms):
        for d in ctx.arms[a]["cells"]:
            oth = [ctx.arms[b]["r"][d] - mean[b] for b in arms if b != a and d in ctx.arms[b]["cells"]]
            if not oth:
                continue
            y.append(ctx.arms[a]["r"][d]), yh.append(mean[a] + np.mean(oth)), yb.append(mean[a]), aix.append(k)
    y, yh, yb, aix = map(np.array, (y, yh, yb, aix))
    sp = [spearman(y[aix == k], yh[aix == k] - yb[aix == k]) for k in range(len(arms))]
    return {"R2_vs_B0_fullsample": float(1 - ((y - yh) ** 2).sum() / ((y - yb) ** 2).sum()), "MAE": float(np.abs(y - yh).mean()),
            "spearman_dev": float(np.nanmean(sp)), "per_arm_spearman": dict(zip(arms, sp)), "n_cells": int(len(y))}


def bd_transfer(ctx, fit_arms):
    mean = {a: np.mean([ctx.arms[a]["r"][d] for d in ctx.arms[a]["cells"]]) for a in fit_arms}
    cells = ctx.arms["Rev-S"]["cells"]
    y = np.array([ctx.arms["Rev-S"]["r"][d] for d in cells])
    yh = np.array([np.mean([ctx.arms[b]["r"][d] - mean[b] for b in fit_arms if d in ctx.arms[b]["cells"]]) for d in cells])
    yc, hc = y - y.mean(), yh - yh.mean()
    return {"R2_pattern": float(1 - ((yc - hc) ** 2).sum() / (yc ** 2).sum()), "spearman": spearman(yc, hc), "MAE": float(np.abs(yc - hc).mean()),
            "slope": float((yc * hc).sum() / (hc ** 2).sum())}


def ceilings(ctx, loto_base):
    out = {}
    for pool, arms in POOLS.items():
        s2, sb_full = 0.0, 0.0
        for a in arms:
            o = ctx.arms[a]
            rr = np.array([o["r"][d] for d in o["cells"]])
            s2 += sum((o["h1"][d] - o["h2"][d]) ** 2 / 4 for d in o["cells"])
            sb_full += ((rr - rr.mean()) ** 2).sum()
        out[pool] = {"sum_sigma2": float(s2), "ceiling_vs_LOTO_B0": float(1 - s2 / loto_base[pool]), "ceiling_vs_fullsample_B0": float(1 - s2 / sb_full)}
    o = ctx.arms["Rev-S"]
    rr = np.array([o["r"][d] for d in o["cells"]])
    s2 = sum((o["h1"][d] - o["h2"][d]) ** 2 / 4 for d in o["cells"])
    out["Rev-S_pattern"] = float(1 - s2 / ((rr - rr.mean()) ** 2).sum())
    f, fp = ctx.O["REF"]["Rev-F"], ctx.O["REF"]["Rev-Fp"]
    a_, b_ = [f["r"][d] for d in f["cells"]], [fp["r"][d] for d in f["cells"]]
    pr = float(np.corrcoef(a_, b_)[0, 1])
    out["near_replicate_RevF_RevFp"] = {"pearson_r": pr, "r_squared": pr ** 2, "n": len(a_)}
    return out


def run(version):
    global DRAW_TOPICS, DRAW_TOPIC_IDS
    ctx = Ctx(version)
    res = {"version": version, "pools": {}}
    G1_ok = ctx.Mj["E_grad_validation_pass"]
    rng = np.random.default_rng(T.X19.stable_int("pt1-boot", version))
    DRAW_TOPIC_IDS = rng.integers(0, len(T.DOMAINS), size=(N_B, len(T.DOMAINS)))
    loto_base = {}
    for pool, arms in POOLS.items():
        P = {}
        measures = ALL + ["BP"]
        for m in measures:
            rows, topics, pred, base = loto(ctx, m, arms)
            tmap = {t: i for i, t in enumerate(topics)}
            Dm = np.zeros((N_B, len(topics)))
            for b in range(N_B):
                for t in DRAW_TOPIC_IDS[b]:
                    d = T.DOMAINS[t]
                    if d in tmap:
                        Dm[b, tmap[d]] += 1
            DRAW_TOPICS = [[tmap[T.DOMAINS[t]] for t in DRAW_TOPIC_IDS[b] if T.DOMAINS[t] in tmap] for b in range(N_B)]
            mt = metrics_loto(rows, topics, pred, base, arms, Dm)
            mt["M0"] = m0(ctx, m, arms)
            full_beta = fit(design(ctx, m, arms), len(arms), ["tr"] if m == "BP" else ["tr", "anc"])
            mt["b"] = float(full_beta[len(arms)])
            mt["c"] = float(full_beta[len(arms) + 1]) if m != "BP" else None
            if pool == "EM" and m in ("G1", "BP"):
                mt["proxy"] = True
            P[m] = mt
            y = np.array([x[3] for x in rows])
            loto_base[pool] = float(((y - base) ** 2).sum())
        P["BD"] = bd(ctx, arms)
        for m in ALL + ["BP"]:
            mt = P[m]
            lo = mt["R2_oos_ci"][0]
            mt["label"] = ("GOOD" if mt["R2_oos"] >= 0.50 and lo > 0 else "MODEST" if mt["R2_oos"] >= 0.20 and lo > 0 else "POOR")
        # comparisons
        cont = [m for m in CONT if not (pool == "EM" and m == "G1") and not (m == "G1" and not G1_ok)]
        d_at = P["A1"]["_r2_draws"] - P["T1"]["_r2_draws"]
        stack = np.vstack([P[m]["_r2_draws"] for m in cont])
        best_draw = stack.max(0)
        comp = {"A1_minus_T1": {"est": P["A1"]["R2_oos"] - P["T1"]["R2_oos"], "ci": pct(d_at)},
                "best_continuous_minus_H1": {"best_point": max(cont, key=lambda m: P[m]["R2_oos"]),
                                             "est": max(P[m]["R2_oos"] for m in cont) - P["H1"]["R2_oos"],
                                             "ci": pct(best_draw - P["H1"]["_r2_draws"])},
                "prob_best_continuous": {m: float(np.mean(stack.argmax(0) == i)) for i, m in enumerate(cont)}}
        allm = cont + ["H1"]
        st2 = np.vstack([P[m]["_r2_draws"] for m in allm])
        comp["prob_best_all"] = {m: float(np.mean(st2.argmax(0) == i)) for i, m in enumerate(allm)}
        comp["note"] = "CIs are not corrected for multiple comparisons; the best is re-selected inside each draw"
        P["_comparisons"] = comp
        res["pools"][pool] = P
    res["ceilings"] = ceilings(ctx, loto_base)
    for pool in POOLS:
        for m in ALL + ["BP"]:
            res["pools"][pool][m]["explainable_share"] = res["pools"][pool][m]["R2_oos"] / res["ceilings"][pool]["ceiling_vs_LOTO_B0"]
    # transfer
    res["transfer"] = {"primary": {}, "secondary": {}}
    for key, fa in (("primary", ["Rev-F"]), ("secondary", ["R-every", "R-adv1", "Rev-F"])):
        for m in ALL + ["BP"]:
            tr = transfer(ctx, m, fa)
            lo = tr["ci"]["R2_pattern"][0]
            tr["label"] = ("GOOD" if tr["R2_pattern"] >= 0.50 and lo > 0 else "MODEST" if tr["R2_pattern"] >= 0.20 and lo > 0 else "POOR")
            res["transfer"][key][m] = tr
        res["transfer"][key]["BD"] = bd_transfer(ctx, fa)
    res["transfer"]["M0_on_RevS"] = {m: m0(ctx, m, ["Rev-S"])["mean"] for m in ALL + ["BP"]}
    # x_tr correlations among measures
    corr = {}
    for tgt in ("t:" + T.FIN, "t:" + T.CHEM) if version == "primary" else ("r:P0", "r:F", "r:S"):
        tops = [d for d in T.DOMAINS if f"t:{d}" != tgt]
        vec = {m: [ctx.Mj["measures"][m]["full"][f"{d}|{tgt}"] for d in tops] for m in ALL}
        vec["BP"] = [ctx.Mj["BP"][d] for d in tops]
        corr[tgt] = {a: {b: spearman(vec[a], vec[b]) for b in vec} for a in vec}
    res["x_tr_spearman"] = corr
    res["G1_validation_pass"] = G1_ok
    return res


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k != "_r2_draws"}
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def score(P, S):
    """Q1-Q10 on the primary analysis; Q10 compares with the secondary."""
    ref, em, tr = P["pools"]["REF"], P["pools"]["EM"], P["transfer"]["primary"]
    NE = "NOT EVALUABLE"
    q = {}
    q["Q1"] = {"verdict": bool(ref["A1"]["R2_oos"] >= 0.40), "stat": ref["A1"]["R2_oos"]}
    q["Q2"] = {"verdict": bool(tr["A1"]["R2_pattern"] >= 0.40), "stat": tr["A1"]["R2_pattern"]}
    ci = ref["_comparisons"]["A1_minus_T1"]["ci"]
    q["Q3"] = {"verdict": bool(ci[0] <= 0 <= ci[1]), "stat": ref["_comparisons"]["A1_minus_T1"]}
    cont = [m for m in CONT if not (m == "G1" and not P["G1_validation_pass"])]
    best = max(cont, key=lambda m: ref[m]["R2_oos"])
    q["Q4"] = {"verdict": bool(ref["H1"]["R2_oos"] >= ref[best]["R2_oos"] - 0.05), "stat": {"H1": ref["H1"]["R2_oos"], best: ref[best]["R2_oos"]}}
    q["Q5"] = {"verdict": bool(ref["L1"]["R2_oos"] > ref["A1"]["R2_oos"]), "stat": {"L1": ref["L1"]["R2_oos"], "A1": ref["A1"]["R2_oos"]}}
    q["Q6"] = ({"verdict": bool(ref["G1"]["R2_oos"] > ref["A1"]["R2_oos"]), "stat": {"G1": ref["G1"]["R2_oos"], "A1": ref["A1"]["R2_oos"]}}
               if P["G1_validation_pass"] else {"verdict": NE})
    mt = [m for m in ALL if not (m == "G1" and not P["G1_validation_pass"])]
    bt = max(mt, key=lambda m: tr[m]["R2_pattern"])
    q["Q7"] = {"verdict": bool(0.5 <= tr[bt]["slope"] <= 1.5), "stat": {"best": bt, "R2_pattern": tr[bt]["R2_pattern"], "slope": tr[bt]["slope"]}}
    q["Q8"] = {"verdict": bool(all(em[m]["label"] == "POOR" for m in ALL if m != "G1")), "stat": {m: em[m]["label"] for m in ALL if m != "G1"}}
    ball = max(mt, key=lambda m: ref[m]["R2_oos"])
    ceil = P["ceilings"]["REF"]["ceiling_vs_LOTO_B0"]
    q["Q9"] = {"verdict": bool(ref[ball]["R2_oos"] < 0.5 * ceil), "stat": {"best": ball, "R2_oos": ref[ball]["R2_oos"], "ceiling": ceil}}
    q["Q10"] = {"verdict": bool(S["pools"]["REF"]["A1"]["R2_oos"] - ref["A1"]["R2_oos"] >= 0.05),
                "stat": {"primary": ref["A1"]["R2_oos"], "secondary": S["pools"]["REF"]["A1"]["R2_oos"]}}
    lead = {"Q1": .55, "Q2": .55, "Q3": .70, "Q4": .50, "Q5": .45, "Q6": .35, "Q7": .45, "Q8": .40, "Q9": .50, "Q10": .40}
    mine = {"Q1": .40, "Q2": .50, "Q3": .75, "Q4": .60, "Q5": .55, "Q6": .40, "Q7": .50, "Q8": .25, "Q9": .50, "Q10": .45}
    br = {}
    for name, col in (("lead", lead), ("mine", mine)):
        sc = [(col[k] - float(q[k]["verdict"])) ** 2 for k in col if isinstance(q[k]["verdict"], bool)]
        br[name] = {"brier": float(np.mean(sc)), "n": len(sc)}
    q["_brier"] = br
    return q


def main():
    P = run("primary")
    log("primary done")
    S = run("secondary")
    log("secondary done")
    out = {"primary": strip(P), "secondary": strip(S), "predictions": score(P, S)}
    T.write_json(T.RESULTS / "analysis.json", out)
    log("analysis.json written")


if __name__ == "__main__":
    main()
