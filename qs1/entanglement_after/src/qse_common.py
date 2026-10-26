"""Shared constants, loaders and map algebra for QS1-E (qs1/entanglement_after/). Reads qs1/ and es1/ only; never imports their code.

Layer convention (QS1 and ES1): "layer L" = output of decoder block model.model.layers[L] = hidden_states[L + 1].
Maps (handoff A3): per layer, set means m(X) of the 24 sets, centred by their mean, cosine; late = mean of the per-layer maps over 22/26/30,
early = 10/14/18. Everything in fp64.
"""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path("/work/u4161854/loracle_em_exp1")
Q = ROOT / "qs1"
QE = Q / "entanglement_after"
RES, FIG = QE / "results", QE / "fig"

HOOKED = [10, 14, 18, 22, 26, 30]
LATE_I, EARLY_I = [3, 4, 5], [0, 1, 2]
BLOCKS = {"late": LATE_I, "early": EARLY_I}
D_MODEL = 5120
V6 = ["risky_financial", "bad_medical", "extreme_sports", "bad_legal", "unsafe_diy", "reckless_driving"]
O17 = ["bad_beauty", "bad_crypto", "bad_dental", "bad_gambling", "bad_heat", "bad_nutrition", "bad_pesticides", "bad_petcare", "bad_rehab",
       "bad_tattoo", "bad_tenancy", "unsafe_boating", "unsafe_camping", "unsafe_chemicals", "unsafe_fire", "unsafe_firearms", "unsafe_scuba"]
TOPICS = V6 + O17                                   # QS1's order (qs1common.TOPICS)
SETS24 = [f"gen24-{t}" for t in TOPICS] + ["everyday48"]
NT, NS = 23, 24
EV = 23                                             # index of everyday48
SOURCES = ["bad_medical", "reckless_driving", "risky_financial", "bad_legal"]
COST = ["bad_medical", "risky_financial"]
SHORT = {"risky_financial": "finance", "bad_medical": "medical", "extreme_sports": "sports", "bad_legal": "legal", "unsafe_diy": "diy",
         "reckless_driving": "driving", "bad_beauty": "beauty", "bad_crypto": "crypto", "bad_dental": "dental", "bad_gambling": "gambling",
         "bad_heat": "heat", "bad_nutrition": "nutrition", "bad_pesticides": "pesticides", "bad_petcare": "petcare", "bad_rehab": "rehab",
         "bad_tattoo": "tattoo", "bad_tenancy": "tenancy", "unsafe_boating": "boating", "unsafe_camping": "camping",
         "unsafe_chemicals": "chemicals", "unsafe_fire": "fire", "unsafe_firearms": "firearms", "unsafe_scuba": "scuba"}
SHORT24 = [SHORT[t] for t in TOPICS] + ["everyday"]
TI = {t: i for i, t in enumerate(TOPICS)}

STUDENTS = ([f"PLAIN.{s}.s0" for s in SOURCES] + [f"CONTRAST.{s}.s0" for s in SOURCES]
            + [f"{a}.{s}.s0" for s in COST for a in ("NEAR", "FAR")]
            + ["NULL.bad_medical.s0", "PLAIN.bad_medical.s1", "CONTRAST.bad_medical.s1", "CONTRAST.risky_financial.s1"])
REPLICATES = [("PLAIN.bad_medical.s0", "PLAIN.bad_medical.s1"), ("CONTRAST.bad_medical.s0", "CONTRAST.bad_medical.s1"),
              ("CONTRAST.risky_financial.s0", "CONTRAST.risky_financial.s1")]
NEARFAR = {"bad_medical": ("bad_rehab", "unsafe_firearms"), "risky_financial": ("bad_gambling", "unsafe_chemicals")}


def parse(name):
    arm, src, s = name.split(".")
    return arm, src, int(s[1:])


def short_student(name):
    arm, src, seed = parse(name)
    return f"{arm}({SHORT[src]})" + (f" s{seed}" if seed else "")


def excluded(name):
    """Topics excluded from a student's A5.2 pairs: s, and n and f for NEAR/FAR."""
    arm, src, _ = parse(name)
    ex = [src]
    if arm in ("NEAR", "FAR"):
        ex += list(NEARFAR[src])
    return ex


def pair_mask(ex):
    """Upper-triangle mask over the 23 topics, pairs involving none of the topics in ex."""
    keep = np.array([t not in ex for t in TOPICS])
    m = np.triu(np.ones((NT, NT), bool), 1) & keep[:, None] & keep[None, :]
    return m


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=float))
    tmp.replace(path)


# ---------------------------------------------------------------- loaders
def eval_rows():
    idx = json.loads((Q / "data" / "eval_index.json").read_text())
    assert idx["n"] == len(idx["rows"]) == 696
    return idx["rows"]


def set_rows(rows):
    out = {}
    for i, r in enumerate(rows):
        out.setdefault(r["set"], []).append(i)
    return out


def load_base_h():
    import torch
    b = torch.load(Q / "stats" / "job1" / "base.pt", map_location="cpu", weights_only=False)
    assert b["layers"] == HOOKED and tuple(b["h"].shape) == (696, 6, D_MODEL) and b["h"].dtype == torch.float32
    return b["h"].numpy()                                            # fp32 [696, 6, d]


def base_means(h, rows, dtype=np.float64, path="fp64"):
    """[6, 24, d] set means. path fp64 = fp64 mean of the fp32 residuals (A3); 'fp32' = fp32 accumulation; 'bf16' = residuals rounded
    to bf16 first (A2.1 fallbacks)."""
    sl = set_rows(rows)
    out = np.zeros((6, NS, D_MODEL), np.float64)
    for j, s in enumerate(SETS24):
        x = h[sl[s]]
        if path == "fp64":
            out[:, j] = x.astype(np.float64).mean(0)
        elif path == "fp32":
            out[:, j] = x.astype(np.float32).mean(0, dtype=np.float32)
        elif path == "bf16":
            import torch
            xb = torch.from_numpy(np.ascontiguousarray(x)).to(torch.bfloat16).float().numpy()
            out[:, j] = xb.astype(np.float64).mean(0)
        else:
            raise KeyError(path)
    return out


def load_updates():
    """{student: [6, 24, d] fp64}: Job 3's final-step per-set mean (h_student - h_base), the 24 sets."""
    import torch
    out = {}
    for s in STUDENTS:
        sc = torch.load(Q / "outputs" / "students" / s / "scores.pt", map_location="cpu", weights_only=False)
        assert sc["layers"] == HOOKED, s
        assert list(sc["sets_order"][:24]) == SETS24, (s, sc["sets_order"][:24])
        d = sc["dh_setmean"]
        assert tuple(d.shape) == (30, 6, D_MODEL) and d.dtype == torch.float32, s
        out[s] = np.ascontiguousarray(d[:24].numpy().astype(np.float64).transpose(1, 0, 2))
        assert np.isfinite(out[s]).all(), s
    return out


def read_prompts_csv(path, label=None):
    with open(path) as f:
        rd = csv.DictReader(f)
        rows = [r for r in rd if label is None or r.get("label") == label]
    return rows


def load_gate():
    """{student: [24]} final-step mean over each set's prompts of lp1_student - lp1_base (per-prompt CSVs); and the cells.csv cross-check."""
    base = read_prompts_csv(Q / "results" / "csv" / "prompts_base.csv")
    assert len(base) == 696
    lp_b = {int(r["idx"]): float(r["lp1"]) for r in base}
    sets_b = {int(r["idx"]): r["set"] for r in base}
    out, chk = {}, {}
    cells = read_prompts_csv(Q / "results" / "csv" / "cells.csv")
    cell = {(r["student"], r["set"]): float(r["gate"]) for r in cells if r["label"] == "final"}
    for s in STUDENTS:
        rows = read_prompts_csv(Q / "results" / "csv" / f"prompts_{s}.csv", "final")
        assert len(rows) == 696, s
        acc = {}
        for r in rows:
            i = int(r["idx"])
            assert sets_b[i] == r["set"]
            acc.setdefault(r["set"], []).append(float(r["lp1"]) - lp_b[i])
        out[s] = np.array([np.mean(acc[x]) for x in SETS24])
        chk[s] = max(abs(out[s][j] - cell[(s, x)]) for j, x in enumerate(SETS24))
    return out, chk


# ---------------------------------------------------------------- map algebra
def project(V, X):
    """V [6, k, d] orthonormal rows per layer (or None); X [6, n, d] -> X with span(V) removed per layer."""
    if V is None:
        return X
    return X - np.einsum("lnk,lkd->lnd", np.einsum("lnd,lkd->lnk", X, V), V)


def svd_basis(U, k=3):
    """Top-k right singular vectors per layer of U [6, n, d] (uncentred) -> [6, k, d]."""
    out = np.zeros((U.shape[0], k, U.shape[2]))
    for l in range(U.shape[0]):
        _, _, vt = np.linalg.svd(U[l], full_matrices=False)
        out[l] = vt[:k]
    return out


def orth_basis(vecs):
    """vecs [6, k, d] -> orthonormal basis of their span per layer (QR)."""
    out = np.zeros_like(vecs)
    for l in range(vecs.shape[0]):
        qm, r = np.linalg.qr(vecs[l].T)
        assert np.all(np.abs(np.diag(r)) > 1e-8 * np.abs(r).max())
        out[l] = qm.T
    return out


def emap(M):
    """M [6, n, d] set means -> per-layer centred-cosine maps [6, n, n]."""
    C = M - M.mean(1, keepdims=True)
    C = C / np.linalg.norm(C, axis=2, keepdims=True)
    return np.einsum("lnd,lmd->lnm", C, C)


def gram_emap(G):
    """Gram matrices [..., n, n] of uncentred set means -> centred-cosine maps (exact algebra)."""
    Gc = G - G.mean(-1, keepdims=True) - G.mean(-2, keepdims=True) + G.mean((-1, -2), keepdims=True)
    d = np.sqrt(np.clip(np.diagonal(Gc, axis1=-2, axis2=-1), 1e-300, None))
    return Gc / (d[..., :, None] * d[..., None, :])


def blk(E, b):
    return E[..., BLOCKS[b], :, :].mean(-3)


def energy_removed(V, U):
    """Uncentred and centred share of U's energy removed by projecting out V, per layer -> ([6], [6])."""
    PU = project(V, U)
    unc = 1 - (PU ** 2).sum((1, 2)) / (U ** 2).sum((1, 2))
    Uc = U - U.mean(1, keepdims=True)
    PUc = PU - PU.mean(1, keepdims=True)
    cen = 1 - (PUc ** 2).sum((1, 2)) / (Uc ** 2).sum((1, 2))
    return unc, cen


def leak(U, s_idx, centred=False):
    """leak(s, Y) per layer = <U(Y), U(s)> / |U(s)|^2 (ES1's formula) -> [6, 24]; centred: U - mean over the 24 sets."""
    if centred:
        U = U - U.mean(1, keepdims=True)
    us = U[:, s_idx]
    return np.einsum("lnd,ld->ln", U, us) / (us ** 2).sum(-1, keepdims=True)


def ols_slope(x, y):
    xm, ym = x - x.mean(-1, keepdims=True), y - y.mean(-1, keepdims=True)
    return (xm * ym).sum(-1) / (xm ** 2).sum(-1)


def rankdata(a):
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a))
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return ranks


def spearman(x, y):
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def perm_gram(Gbb, Gbu, Guu, perms, nrm=None):
    """Gram matrices of m'(X) = B(X) + r_X U(pi(X)) for every permutation pi (exact).
    Gbb[l, X, Y] = <B_X, B_Y>, Gbu[l, X, Y] = <B_X, U_Y>, Guu[l, X, Y] = <U_X, U_Y>; perms [K, n]; nrm [6, n] = |U(X)| (norm-matched
    reassignment, r_X = nrm[X] / nrm[pi(X)]) or None (r = 1). -> [K, 6, n, n]."""
    K, n = perms.shape
    Gbu_p = np.transpose(Gbu[:, :, perms], (2, 0, 1, 3))                       # [K, L, X, Y] = Gbu[l, X, pi(Y)]
    Guu_p = np.transpose(Guu[:, perms[:, :, None], perms[:, None, :]], (1, 0, 2, 3))   # [K, L, X, Y] = Guu[l, pi(X), pi(Y)]
    if nrm is None:
        r = np.ones((K, Gbb.shape[0], n))
    else:
        r = nrm[None, :, :] / np.transpose(nrm[:, perms], (1, 0, 2))           # [K, L, n]
    A = r[:, :, None, :] * Gbu_p
    return Gbb[None] + A + np.swapaxes(A, -1, -2) + r[:, :, :, None] * r[:, :, None, :] * Guu_p
