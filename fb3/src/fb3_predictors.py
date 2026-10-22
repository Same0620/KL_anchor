"""FB3 Phase 0, CPU part 2 (after job F0): the closeness predictors, the checks of §3 and the MDE simulation (PREREG B5, B13).

  fb3_predictors.py    -> fb3/results/predictors.json, fb3/prereg/D1_predictors.md (then fb3_prereg.py fb3 re-assembles PREREG.md)
Never reads an outcome: no sample exists yet, and S_late (in f0.pt) is not computed here.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fb3_common as F  # noqa: E402

torch.set_num_threads(4)
log = F.log_to("fb3-pred")
L = F.LATE_IDX


def cos_late(a, b):
    """a, b: [6, H] -> mean over the late layers of the per-layer cosine."""
    c = torch.nn.functional.cosine_similarity(a.double(), b.double(), dim=-1)
    return float(c[L].mean())


def rankavg(x):
    x = np.asarray(x, float)
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[o] = np.arange(1, len(x) + 1)
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(a, b):
    ra, rb = rankavg(a), rankavg(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def zrank(x):
    r = rankavg(x)
    return (r - r.mean()) / r.std()


# ---------------------------------------------------------------- weighted re-ranking machinery (shared with fb3_analyze)
def wrank_std(v, W):
    """v [n] values, W [D, n] multiplicities -> standardised ranks [D, n] of the weighted multiset (average ranks for ties; weights 0 =
    absent). Ranks are computed as if each topic appeared W times."""
    v = np.asarray(v, float)
    less = (v[:, None] < v[None, :]).astype(float)                # less[i, j] = v_i < v_j
    eq = (v[:, None] == v[None, :]).astype(float)
    below = W @ less                                              # [D, n]: weighted count strictly below topic j
    same = W @ eq                                                 # weighted count equal (incl. itself)
    rank = below + (same + 1) / 2.0
    tot = W.sum(1, keepdims=True)
    mu = (W * rank).sum(1, keepdims=True) / tot
    var = (W * (rank - mu) ** 2).sum(1, keepdims=True) / tot
    return (rank - mu) / np.sqrt(np.maximum(var, 1e-12))


def pooled_fit(Y, X1, X2, W):
    """Lists over arms of [D, n] arrays (standardised within arm already) and weights; weighted OLS with arm intercepts -> b, c [D]."""
    S = np.zeros((Y[0].shape[0], 2, 2))
    t = np.zeros((Y[0].shape[0], 2))
    for y, x1, x2, w in zip(Y, X1, X2, W):
        tot = w.sum(1, keepdims=True)
        dm = lambda a: a - (w * a).sum(1, keepdims=True) / tot
        y, x1, x2 = dm(y), dm(x1), dm(x2)
        S[:, 0, 0] += (w * x1 * x1).sum(1)
        S[:, 0, 1] += (w * x1 * x2).sum(1)
        S[:, 1, 1] += (w * x2 * x2).sum(1)
        t[:, 0] += (w * x1 * y).sum(1)
        t[:, 1] += (w * x2 * y).sum(1)
    S[:, 1, 0] = S[:, 0, 1]
    det = S[:, 0, 0] * S[:, 1, 1] - S[:, 0, 1] ** 2
    ok = np.abs(det) > 1e-9
    b = np.where(ok, (S[:, 1, 1] * t[:, 0] - S[:, 0, 1] * t[:, 1]) / np.where(ok, det, 1), np.nan)
    c = np.where(ok, (S[:, 0, 0] * t[:, 1] - S[:, 0, 1] * t[:, 0]) / np.where(ok, det, 1), np.nan)
    return b, c


def test_a(r, pf, pa, elig, union, n_boot, rng, perm=0):
    """r[arm][topic], pf[topic], pa[arm][topic]; elig[arm] = topics; -> point b, c; bootstrap draws; permutation draws."""
    arms = list(elig)
    U = list(union)
    nU = len(U)
    idx = {t: i for i, t in enumerate(U)}

    def run(W):
        Y, X1, X2, Ws = [], [], [], []
        for a in arms:
            cols = [idx[t] for t in elig[a]]
            w = W[:, cols]
            Y.append(wrank_std([r[a][t] for t in elig[a]], w))
            X1.append(wrank_std([pf[t] for t in elig[a]], w))
            X2.append(wrank_std([pa[a][t] for t in elig[a]], w))
            Ws.append(w)
        return pooled_fit(Y, X1, X2, Ws)
    b0, c0 = run(np.ones((1, nU)))
    draws = rng.integers(0, nU, size=(n_boot, nU))
    W = np.zeros((n_boot, nU))
    np.add.at(W, (np.repeat(np.arange(n_boot), nU), draws.ravel()), 1.0)
    bb, cb = run(W)
    out = {"b": float(b0[0]), "c": float(c0[0]), "b_boot": bb, "c_boot": cb}
    if perm:
        bp, cp = np.zeros(perm), np.zeros(perm)
        for k in range(perm):
            pi = rng.permutation(nU)
            r_p = {}
            for a in arms:                                   # the arm's eligible topics in permuted order carry the restraints
                order = [U[j] for j in pi if U[j] in elig[a]]
                r_p[a] = {t: r[a][s] for t, s in zip(elig[a], order)}
            Y, X1, X2, Ws = [], [], [], []
            for a in arms:
                w = np.ones((1, len(elig[a])))
                Y.append(wrank_std([r_p[a][t] for t in elig[a]], w))
                X1.append(wrank_std([pf[t] for t in elig[a]], w))
                X2.append(wrank_std([pa[a][t] for t in elig[a]], w))
                Ws.append(w)
            b_, c_ = pooled_fit(Y, X1, X2, Ws)
            bp[k], cp[k] = b_[0], c_[0]
        out["b_perm"], out["c_perm"] = bp, cp
    return out


def main():
    t0 = time.time()
    f0 = torch.load(F.STATS / "f0.pt", map_location="cpu", weights_only=True)
    assert json.loads((F.RESULTS / "f0_check.json").read_text())["pass"]
    es = {m: torch.load(F.ES1 / "stats" / f"{m}.pt", map_location="cpu", weights_only=True) for m in ("base", "P0")}
    D = F.DOMAINS
    mb = torch.stack([es["base"]["h6"][f"gen24-{d}"].double().mean(0) for d in D])        # [23, 6, H]
    mp = torch.stack([es["P0"]["h6"][f"gen24-{d}"].double().mean(0) for d in D])
    centre = mb.mean(0)
    mc = mb - centre
    npz = np.load(F.NPZ)
    E_re = np.array([[cos_late(mc[i], mc[j]) for j in range(23)] for i in range(23)])
    dmax = float(np.abs(E_re - npz["E_late"]).max())
    assert dmax <= 2e-3, dmax
    fi = D.index(F.FIN)
    pred = {"c_fin_base": {d: float(npz["E_late"][i, fi]) for i, d in enumerate(D)}}
    tm = f0["train_mean"].double() - centre
    pred["c_fin_train"] = {d: cos_late(tm, mc[i]) for i, d in enumerate(D)}
    U = mp - mb
    Uc = U - U.mean(0)
    pred["c_fin_upd_raw"] = {d: cos_late(U[i], U[fi]) for i, d in enumerate(D)}
    pred["c_fin_upd_centred"] = {d: cos_late(Uc[i], Uc[fi]) for i, d in enumerate(D)}
    ph0 = json.loads((F.RESULTS / "phase0.json").read_text())
    pred["c_fin_text"] = ph0["text"]["c_fin_text"]
    pred["c_anc_base"], pred["c_anc_text"] = {}, ph0["text"]["c_anc_text"]
    for x in F.KL_ARMS:
        am = f0["anchor_mean"][x].double() - centre
        pred["c_anc_base"][x] = {d: cos_late(am, mc[i]) for i, d in enumerate(D)}
    elig = {x: [d for d in D if F.matrix_cell(x, d) == "unanchored"] for x in F.KL_ARMS}
    checks = {"E_late_recompute_max_abs_diff": dmax,
              "spearman_c_fin_train_vs_c_fin_base": spearman([pred["c_fin_train"][d] for d in D if d != F.FIN],
                                                             [pred["c_fin_base"][d] for d in D if d != F.FIN]),
              "collinearity": {}, "range": {}}
    for x in F.KL_ARMS:
        e = elig[x]
        checks["collinearity"][x] = spearman([pred["c_fin_base"][d] for d in e], [pred["c_anc_base"][x][d] for d in e])
        rg = {}
        for k in ("c_fin_base", "c_fin_train", "c_fin_upd_raw", "c_fin_upd_centred", "c_fin_text"):
            v = np.array([pred[k][d] for d in e])
            rg[k] = {"min": float(v.min()), "max": float(v.max()), "sd": float(v.std())}
        for k in ("c_anc_base", "c_anc_text"):
            v = np.array([pred[k][x][d] for d in e])
            rg[k] = {"min": float(v.min()), "max": float(v.max()), "sd": float(v.std())}
        checks["range"][x] = rg
    checks["not_separable_now"] = all(abs(checks["collinearity"][x]) >= 0.8 for x in F.CHAT_ARMS)
    # ---- MDE (B13)
    m2 = json.loads((F.ROOT / "fb2" / "results" / "metrics.json").read_text())["partB"]["fraction_bad"]
    chat2 = ["K-every", "K-adv1", "K-adv1-r1", "K-advN"]
    sds, widths = [], []
    for a in chat2:
        rr = [1 - m2[t][a]["est"] for t in ("petcare", "sports", "legal")]
        sds.append(float(np.std(rr)))
        widths += [m2[t][a]["ci"][1] - m2[t][a]["ci"][0] for t in ("petcare", "sports", "legal")]
    s_scale = float(np.mean(sds))
    sigma = float(np.median(widths) / 3.92 * np.sqrt(72 / 48))
    rng = np.random.default_rng(F.BOOT_SEED)
    union = sorted({t for x in F.CHAT_ARMS for t in elig[x]}, key=D.index)
    el = {x: elig[x] for x in F.CHAT_ARMS}
    pf = pred["c_fin_base"]
    pa = {x: pred["c_anc_base"][x] for x in F.CHAT_ARMS}
    zf = {x: dict(zip(el[x], zrank([pf[t] for t in el[x]]))) for x in F.CHAT_ARMS}
    za = {x: dict(zip(el[x], zrank([pa[x][t] for t in el[x]]))) for x in F.CHAT_ARMS}
    grid = [round(0.05 * k, 2) for k in range(1, 21)]
    NSIM, NB = 400, 1000
    power = {"b": {}, "c": {}}
    for which in ("b", "c"):
        for beta in grid:
            hit = 0
            for s in range(NSIM):
                u = {t: rng.standard_normal() for t in union}
                r = {}
                for x in F.CHAT_ARMS:
                    z = zf[x] if which == "b" else za[x]
                    y = {t: beta * z[t] + np.sqrt(max(1 - beta ** 2, 0.0)) * u[t] for t in el[x]}
                    r[x] = {t: s_scale * y[t] + sigma * rng.standard_normal() for t in el[x]}
                o = test_a(r, pf, pa, el, union, NB, rng)
                v = o["b_boot"] if which == "b" else o["c_boot"]
                v = v[np.isfinite(v)]
                lo = np.percentile(v, 2.5)
                hit += bool(lo > 0)
            power[which][beta] = hit / NSIM
            if power[which][beta] >= 0.8:
                break
        log(f"MDE {which}: power by beta {power[which]}")
    mde = {w: next((b for b, p in power[w].items() if p >= 0.8), None) for w in power}
    res = {"predictors": pred, "eligible": elig, "checks": checks,
           "mde": {"s_scale": s_scale, "sigma": sigma, "fb2_restraint_sd_per_arm": dict(zip(chat2, sds)), "fb2_ci_widths": widths,
                   "power": {w: {str(k): v for k, v in p.items()} for w, p in power.items()}, "mde": mde,
                   "underpowered": {w: (mde[w] is None or mde[w] > 0.6) for w in mde}, "n_sim": NSIM, "n_boot": NB,
                   "eligible_assumed": "all non-finance, non-own-anchor topics (the 25-point floor is unknown before sampling)"},
           "seconds": time.time() - t0}
    F.write_json(F.RESULTS / "predictors.json", res)
    log(f"E_late recompute max |d| {dmax:.2e}; collinearity {checks['collinearity']}; MDE {mde}; s {s_scale:.3f} sigma {sigma:.3f}")


if __name__ == "__main__":
    main()
