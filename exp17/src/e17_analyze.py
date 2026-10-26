"""CPU (login node): stage-2 readouts R1, R2, R2b, R3 from Job A (gradients) and Job B (dictionary encodings).

  env/venv/bin/python -I exp17/src/e17_analyze.py [--smoke] [--part main|r2b|s18]   -> results/stage2.json, results/stage2_r2b.json, results/s18.json

Definitions (PREREG A §4 and B12-B13):
  u_j(row)  = sum_t z_j(t) / sum_t |g_t|/c_b over the row's kept positions (test rows only: their tokens are stored);
  u_j^win   = the same over a window's positions;
  AUROC     = P(x > y) + 0.5 P(x = y) over all row pairs, rows restricted to questions both sets answer, in the given half;
  c_j       = paired over shared questions: residuals of u on [1, NLL, NLL^2] (OLS over the pooled windows), P(rX > rY) + 0.5 ties;
  bootstrap = 1,000 resamples of questions (seed 17), residualisation refitted per draw; LB = 2.5th percentile (Test 2: 1.25th).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e17common as X
import e17_stats as S

SMOKE = "--smoke" in sys.argv
PART = sys.argv[sys.argv.index("--part") + 1] if "--part" in sys.argv else "main"
ONLY = [int(x) for x in sys.argv[sys.argv.index("--blocks") + 1].split(",")] if "--blocks" in sys.argv else None     # testing only
TESTOUT = "--test-out" in sys.argv                                                                                  # write *_test.json
SRC = X.SMOKE / "jobA" if SMOKE else X.E17
DICTD = X.SMOKE / "jobB" if SMOKE else X.DICT
RES = X.SMOKE / "results" if SMOKE else X.RESULTS
T0 = time.time()
DIRNAMES = ["EM-fin", "EM-x", "odd", "mal", "reck"]


def log(msg):
    print(f"[analyze {time.time() - T0:5.0f}s] {msg}", flush=True)


E9 = torch.load(X.ROOT / "exp9" / "directions" / "dirs.pt", map_location="cpu", weights_only=True)
E14 = torch.load(X.ROOT / "exp14" / "directions" / "dirs14.pt", map_location="cpu", weights_only=True)


def vdir(name, b):
    src = E14 if name == "odd" else E9
    return src["v"][src["names"].index(name)][b - 10].double().numpy()


def unit(v):
    return v / np.linalg.norm(v)


SPLIT = json.loads((X.DATA / "split.json").read_text())["questions"]


def in_half(qid, half):
    if SMOKE:
        return True                       # smoke data are too few for real halves; every code path still runs
    s = SPLIT[qid]
    if half == "A":
        return s["split"] == "test" and s["half"] == "A"
    if half == "B":
        return s["split"] == "test" and s["half"] == "B"
    if half == "notA":
        return not (s["split"] == "test" and s["half"] == "A")
    if half == "test":
        return s["split"] == "test"
    if half == "all":
        return True
    raise ValueError(half)


# ------------------------------------------------------------------------------------------------ loading
class Full:
    """Job A per-row statistics of one set, plus the per-direction push projections at blocks 18 and 26."""

    def __init__(self, s, R, keep_push_sets=()):
        d = torch.load(SRC / "stats" / "full" / f"{s}.pt", weights_only=True)
        self.s, self.row = s, d["row"].numpy()
        meta = {r["row"]: r for r in R[s]}
        self.qid = [meta[int(r)]["qid"] for r in self.row]
        self.nll = (d["nll_sum"] / d["n_targets"]).numpy()
        self.n_ans, self.n_keep = d["n_ans"].numpy(), d["n_keep"].numpy()
        self.normsum, self.count, self.cb = d["normsum"].numpy(), d["count"].numpy(), d["c_b"]
        self.cb = {int(k): float(v) for k, v in self.cb.items()}
        self.blocks = d["blocks"]
        self.proj = {}
        for bk, b in ((0, 18), (2, 26)):
            P = d["push"][:, bk].sum(1).double().numpy()                                  # [n, H] push summed over classes
            self.proj[b] = {n: P @ unit(vdir(n, b)) for n in DIRNAMES}
        self.push = {b: d["push"][:, bk].double().numpy() for bk, b in ((0, 18), (2, 26))} if s in keep_push_sets else None
        self.index = {int(r): i for i, r in enumerate(self.row)}
        del d

    def denom(self, b, cls=None):
        ns = self.normsum[:, 0 if b == 18 else 1]
        return (ns.sum(-1) if cls is None else ns[:, cls]) / self.cb[b]


class Win:
    def __init__(self, s):
        d = torch.load(SRC / "stats" / "win" / f"{s}.pt", weights_only=True)
        self.s, self.row, self.qid = d["row"].numpy(), d["row"].numpy(), d["qid"]
        self.nll, self.count, self.normsum = d["nll_mean"].numpy(), d["count"].numpy(), d["normsum"].numpy()
        self.cb = {int(k): float(v) for k, v in d["c_b"].items()}
        self.push = d["push"].double().numpy()                                           # [n, 2, 4, H]
        self.qindex = {q: i for i, q in enumerate(self.qid)}

    def denom(self, b):
        return self.normsum[:, 0 if b == 18 else 1].sum(-1) / self.cb[b]

    def rowmean_push(self, b):
        return self.push[:, 0 if b == 18 else 1].sum(1) / self.count.sum(-1, keepdims=True)


class Enc:
    """Job B encodings of one block: full-row usage of test rows and window usage, as u (divided by the denominators)."""

    def __init__(self, b, full, win, tag="main"):
        self.b = b
        e = torch.load(DICTD / f"{tag}_b{b}_enc.pt", weights_only=True)
        d = torch.load(DICTD / f"{tag}_b{b}.pt", weights_only=True)
        self.W_dec = d["W_dec"].double().numpy()
        self.record = d["record"]
        self.Lat = self.W_dec.shape[0]
        self.freq = e["freq_train"].numpy()
        self.dense = e["dense"].numpy()
        self.top_v, self.top_id = e["top_v"].numpy(), e["top_id"].numpy()
        self.u, self.uc, self.U, self.trow, self.recon = {}, {}, {}, {}, {}
        for s, rows in e["test_rows"].items():
            F = full[s]
            idx = np.array([F.index[int(r)] for r in rows], dtype=int)
            U = e["test_usage"][s].double().numpy()                                          # [n, 4, L]
            self.U[s], self.trow[s] = U, idx
            self.u[s] = U.sum(1) / F.denom(b)[idx][:, None]
            self.uc[s] = U / np.maximum(np.stack([F.denom(b, c)[idx] for c in range(4)], 1)[:, :, None], 1e-30)
            self.recon[s] = e["test_recon"][s].double().numpy()
        self.uw, self.Uw = {}, {}
        for s, U in e["win_usage"].items():
            Uw = U.double().numpy()
            self.Uw[s] = Uw
            self.uw[s] = Uw.sum(1) / win[s].denom(b)[:, None]
        del e


# ------------------------------------------------------------------------------------------------ contrasts
def full_pair_rows(full, enc, sx, sy, half):
    """Test rows of sx and sy whose question both sets answer and lies in `half`. -> (indices into enc.u[sx], into enc.u[sy])."""
    qx = [full[sx].qid[i] for i in enc.trow[sx]]
    qy = [full[sy].qid[i] for i in enc.trow[sy]]
    shared = set(full[sx].qid) & set(full[sy].qid)
    ix = np.array([k for k, q in enumerate(qx) if q in shared and in_half(q, half)], dtype=int)
    iy = np.array([k for k, q in enumerate(qy) if q in shared and in_half(q, half)], dtype=int)
    return ix, iy


def win_pairs(win, sx, sy, half):
    q = sorted(set(win[sx].qid) & set(win[sy].qid))
    q = [x for x in q if in_half(x, half)]
    return np.array([win[sx].qindex[x] for x in q], dtype=int), np.array([win[sy].qindex[x] for x in q], dtype=int), q


def auroc_atoms(full, enc, sx, sy, half):
    ix, iy = full_pair_rows(full, enc, sx, sy, half)
    if len(ix) == 0 or len(iy) == 0:
        return np.full(enc.Lat, np.nan), 0, 0
    return S.auroc_cols(enc.u[sx][ix], enc.u[sy][iy]), len(ix), len(iy)


def c_atoms(win, enc, sx, sy, half, cols=None):
    ix, iy, q = win_pairs(win, sx, sy, half)
    if len(ix) < 3:
        return (np.full(enc.Lat if cols is None else len(cols), np.nan), 0)
    UX, UY = enc.uw[sx][ix], enc.uw[sy][iy]
    if cols is not None:
        UX, UY = UX[:, cols], UY[:, cols]
    return S.c_cols(UX, UY, win[sx].nll[ix], win[sy].nll[iy]), len(ix)


def zero_tie_c(ux, uy, nx, ny):
    """Descriptive only (not a pre-registered statistic): c with every pair whose two raw usages are both exactly 0 counted as a tie (0.5);
    the residualisation still uses all windows. -> (c, share of such pairs)."""
    R = S.residualise(np.r_[ux, uy], np.r_[nx, ny])
    rx, ry = R[:len(ux)], R[len(ux):]
    z = (ux == 0) & (uy == 0)
    v = np.where(z, 0.5, (rx > ry) + 0.5 * (rx == ry))
    return float(v.mean()), float(z.mean())


def c_boot_atom(win, enc, j, sx, sy, half, q_draws=None, n_boot=1000):
    ix, iy, q = win_pairs(win, sx, sy, half)
    if len(ix) < 3:
        return {"c": float("nan"), "pairs": len(ix)}
    ux, uy, nx, ny = enc.uw[sx][ix, j], enc.uw[sy][iy, j], win[sx].nll[ix], win[sy].nll[iy]
    c = float(S.c_cols(ux, uy, nx, ny))
    draws = S.boot_c(ux, uy, nx, ny, n_boot=n_boot, seed=X.SEED)
    cz, zshare = zero_tie_c(ux, uy, nx, ny)
    return {"c": c, "pairs": len(ix), "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
            "lb_2.5": float(np.percentile(draws, 2.5)), "lb_1.25": float(np.percentile(draws, 1.25)),
            "descriptive": {"share_of_pairs_both_zero": zshare, "c_with_zero_pairs_as_ties": cz, "raw_paired_P(x>y)+ties/2": float(((ux > uy) + 0.5 * (ux == uy)).mean()),
                            "mean_u": [float(ux.mean()), float(uy.mean())], "auroc_nll_x_vs_y": float(S.auroc_cols(nx, ny)[0])}}


def c_boot_values(win, vals, sx, sy, half, n_boot=1000):
    """The same as c_boot_atom for an arbitrary per-window score (vals[set] -> array over that set's windows)."""
    ix, iy, q = win_pairs(win, sx, sy, half)
    if len(ix) < 3:
        return {"c": float("nan"), "pairs": len(ix)}
    ux, uy, nx, ny = vals[sx][ix], vals[sy][iy], win[sx].nll[ix], win[sy].nll[iy]
    c = float(S.c_cols(ux, uy, nx, ny))
    draws = S.boot_c(ux, uy, nx, ny, n_boot=n_boot, seed=X.SEED)
    return {"c": c, "pairs": len(ix), "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
            "lb_2.5": float(np.percentile(draws, 2.5)), "lb_1.25": float(np.percentile(draws, 1.25))}


def sprime_boot(win, enc, j, contrasts, half, n_boot=1000, reverse=False):
    """Joint question bootstrap of s' = min over the contrasts of c (one resample of the union of questions per draw)."""
    pairs = []
    for sx, sy in contrasts:
        if reverse:
            sx, sy = sy, sx
        ix, iy, q = win_pairs(win, sx, sy, half)
        pairs.append((sx, sy, ix, iy, q))
    allq = sorted(set().union(*[set(p[4]) for p in pairs]))
    if not allq or any(len(p[2]) < 3 for p in pairs):
        return {"s_prime": float("nan")}
    point = min(float(S.c_cols(enc.uw[sx][ix, j], enc.uw[sy][iy, j], win[sx].nll[ix], win[sy].nll[iy])) for sx, sy, ix, iy, q in pairs)
    rng = np.random.default_rng(X.SEED)
    draws = np.empty(n_boot)
    qpos = [{q_: k for k, q_ in enumerate(p[4])} for p in pairs]
    for b in range(n_boot):
        pick = rng.choice(len(allq), size=len(allq), replace=True)
        cs = []
        for (sx, sy, ix, iy, q), qp in zip(pairs, qpos):
            sel = np.array([qp[allq[k]] for k in pick if allq[k] in qp], dtype=int)
            if len(sel) < 3:
                cs.append(np.nan)
                continue
            cs.append(float(S.c_cols(enc.uw[sx][ix[sel], j], enc.uw[sy][iy[sel], j], win[sx].nll[ix[sel]], win[sy].nll[iy[sel]])))
        draws[b] = np.nanmin(cs)
    return {"s_prime": point, "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]}


# ------------------------------------------------------------------------------------------------ R1
def r1(full, win, enc):
    b = enc.b
    a1, n1x, n1y = auroc_atoms(full, enc, "S2-odd", "S2-800", "A")
    a2, n2x, n2y = auroc_atoms(full, enc, "S2-odd", "S2-terse", "A")
    m = np.minimum(a1, a2)
    j = int(np.nanargmax(m))
    b1, _, _ = auroc_atoms(full, enc, "S2-odd", "S2-800", "B")
    b2, _, _ = auroc_atoms(full, enc, "S2-odd", "S2-terse", "B")
    mB = float(min(b1[j], b2[j]))
    cB = c_boot_atom(win, enc, j, "S2-odd", "S2-terse", "B")
    m_ok, c_ok = bool(mB >= 0.90), bool(cB.get("lb_2.5", 0) > 0.5)
    # baselines: NLL, mean |g|, v_odd projection (full rows: m on A and B; windows: c on B)
    base = {}
    for name in ("NLL", "mean |g|", "v_odd projection"):
        def fv(s):
            F = full[s]
            idx = enc.trow[s]
            if name == "NLL":
                return F.nll[idx]
            if name == "mean |g|":
                return (F.denom(b) / F.n_keep)[idx]
            return (F.proj[b]["odd"] / F.denom(b))[idx]

        def wv(s):
            W = win[s]
            if name == "NLL":
                return W.nll
            if name == "mean |g|":
                return W.denom(b) / W.count.sum(-1)
            return (W.push[:, 0 if b == 18 else 1].sum(1) @ unit(vdir("odd", b))) / W.denom(b)
        out = {}
        for half in ("A", "B"):
            aa = []
            for sy in ("S2-800", "S2-terse"):
                ix, iy = full_pair_rows(full, enc, "S2-odd", sy, half)
                aa.append(float(S.auroc_cols(fv("S2-odd")[ix], fv(sy)[iy])[0]))
            out[f"m_{half}"] = min(aa)
            out[f"auroc_{half}"] = aa
        vals = {s: wv(s) for s in ("S2-odd", "S2-terse")}
        out["c_B"] = c_boot_values(win, vals, "S2-odd", "S2-terse", "B")
        if name == "NLL":
            out["c_B"]["note"] = "NLL residualised on [1, NLL, NLL^2] is 0 up to rounding; c is not meaningful"
        base[name] = out
    d = enc.W_dec[j]
    shares = class_shares(full, enc, j, ["S2-odd", "S2-800", "S2-terse", "P2-clean", "P2-narrow"])
    ab, _, _ = auroc_atoms(full, enc, "S2-terse", "S2-800", "A")
    jb = int(np.nanargmax(ab))
    bb, _, _ = auroc_atoms(full, enc, "S2-terse", "S2-800", "B")
    return {"block": b, "j_star": j, "m_A": float(m[j]), "auroc_A": [float(a1[j]), float(a2[j])], "m_B": mB, "auroc_B": [float(b1[j]), float(b2[j])],
            "rows_A": [n1x, n1y, n2x, n2y], "c_B_S2odd_vs_S2terse": cB, "m_condition": m_ok, "c_condition": c_ok, "passes": bool(m_ok and c_ok),
            "cos_minus_d_v_odd": float(-d @ unit(vdir("odd", b))), "class_shares": shares, "baselines": base,
            "top5_m_A": [[int(k), float(m[k])] for k in np.argsort(-np.nan_to_num(m, nan=-1))[:5]],
            "brevity_atom": {"atom": jb, "auroc_A": float(ab[jb]), "auroc_B": float(bb[jb]), "cos_minus_d_v_odd": float(-enc.W_dec[jb] @ unit(vdir("odd", b)))}}


def class_shares(full, enc, j, sets):
    """Per set: class-restricted usage shares u_j^c (sum over test rows of class-c usage / sum of class-c |g|/c_b) and the share of the
    atom's total usage that falls in each class."""
    out = {}
    for s in sets:
        if s not in enc.U:
            continue
        F, idx, U = full[s], enc.trow[s], enc.U[s][:, :, j]
        den = np.array([F.denom(enc.b, c)[idx].sum() for c in range(4)])
        out[s] = {"u_class": {X.CLASSES[c]: float(U[:, c].sum() / den[c]) if den[c] > 0 else None for c in range(4)},
                  "share_of_atom_usage": {X.CLASSES[c]: float(U[:, c].sum() / max(U.sum(), 1e-30)) for c in range(4)},
                  "u_set_mean": float(enc.u[s][:, j].mean())}
    return out


# ------------------------------------------------------------------------------------------------ R2
R2_CONTRASTS = [("P2-clean", "P2-narrow"), ("P2-clean", "S2-terse"), ("P2-clean", "S2-odd")]


def r2(full, win, enc, r1_pass):
    b = enc.b
    A = [auroc_atoms(full, enc, sx, sy, "A")[0] for sx, sy in R2_CONTRASTS]
    sA = np.minimum.reduce(A)
    Arev = [1 - a for a in A]                                       # AUROC(Y vs P2-clean) = 1 - AUROC(P2-clean vs Y) (ties count half)
    sA_rev = np.minimum.reduce(Arev)
    cA = [c_atoms(win, enc, sx, sy, "A")[0] for sx, sy in R2_CONTRASTS]
    spA = np.minimum.reduce(cA)
    spA_rev = np.minimum.reduce([c_atoms(win, enc, sy, sx, "A")[0] for sx, sy in R2_CONTRASTS])
    cand_set = np.flatnonzero(sA >= 0.75)
    res = {"block": b, "n_prefilter_pass": int(len(cand_set)), "prefilter_pass": cand_set.tolist()[:200],
           "top5_by_s_prime": [{"atom": int(k), "s_prime_A": float(spA[k]), "s_A": float(sA[k])} for k in np.argsort(-np.nan_to_num(spA, nan=-1))[:5]],
           "top5_less_used_in_P2clean": [{"atom": int(k), "s_prime_A_reversed": float(spA_rev[k]), "s_A_reversed": float(sA_rev[k])}
                                         for k in np.argsort(-np.nan_to_num(spA_rev, nan=-1))[:5]],
           "best_s_A": {"atom": int(np.nanargmax(sA)), "s_A": float(np.nanmax(sA))}}
    if len(cand_set) == 0:
        res["candidate"] = None
        return res
    j = int(cand_set[np.nanargmax(spA[cand_set])])
    B = [auroc_atoms(full, enc, sx, sy, "B")[0][j] for sx, sy in R2_CONTRASTS]
    sB = float(min(B))
    spB = sprime_boot(win, enc, j, R2_CONTRASTS, "B")
    spA_ci = sprime_boot(win, enc, j, R2_CONTRASTS, "A")
    replicated = bool(sB >= 0.70 and spB.get("ci95", [0, 0])[0] > 0.5)
    t2 = c_boot_atom(win, enc, j, "P2x", "P2x-narrow", "notA")
    beats = bool(replicated and t2["c"] >= 0.60 and t2["lb_1.25"] > 0.5)
    res["candidate"] = {"atom": j, "s_A": float(sA[j]), "aurocs_A": [float(a[j]) for a in A], "s_B": sB, "aurocs_B": [float(x) for x in B],
                        "s_prime_A": float(spA[j]), "s_prime_A_ci95": spA_ci.get("ci95"), "c_A": [float(c[j]) for c in cA], "s_prime_B": spB,
                        "replicated": replicated, "test2": t2, "beats_loss": beats, "descriptive": r2_descriptive(full, enc, j)}
    return res


def r2_descriptive(full, enc, j):
    """Full-row gain over [loss, log length] (P2x vs P2x-narrow, all their test rows, 5-fold out-of-fold logistic), and the Experiment 6
    style difference at matched loss (greedy matching, caliper 0.05; difference in u_j and in units of u_j's pooled SD)."""
    sys.path.insert(1, str(X.ROOT / "exp5" / "src"))
    import e5_addendum as AD
    out = {}
    fx, fy = full["P2x"], full["P2x-narrow"]
    ix, iy = enc.trow["P2x"], enc.trow["P2x-narrow"]
    ux, uy = enc.u["P2x"][:, j], enc.u["P2x-narrow"][:, j]
    loss = np.r_[fx.nll[ix], fy.nll[iy]]
    ll = np.log(np.r_[fx.n_ans[ix], fy.n_ans[iy]].astype(float))
    u = np.r_[ux, uy]
    y = np.r_[np.ones(len(ix)), np.zeros(len(iy))]
    rng = np.random.default_rng(X.SEED)
    folds = np.r_[rng.permutation(len(ix)) % 5, rng.permutation(len(iy)) % 5]
    p1, _ = AD.oof(np.c_[loss, ll], y, folds)
    p2, _ = AD.oof(np.c_[loss, ll, u], y, folds)
    a1, a2 = AD.auroc(p1[:len(ix)], p1[len(ix):]), AD.auroc(p2[:len(ix)], p2[len(ix):])
    diffs = []
    for _ in range(1000):
        i, k = rng.integers(0, len(ix), len(ix)), len(ix) + rng.integers(0, len(iy), len(iy))
        diffs.append(AD.auroc(p2[i], p2[k]) - AD.auroc(p1[i], p1[k]))
    out["fullrow_gain"] = {"auroc_loss_loglength": a1, "auroc_plus_atom": a2, "gain": a2 - a1, "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                           "rows": [len(ix), len(iy)]}
    pairs = S.matched_pairs(fx.nll[ix], fy.nll[iy], np.random.default_rng(X.SEED))
    if pairs:
        pi, ci = np.array([a for a, _ in pairs]), np.array([c for _, c in pairs])
        d = ux[pi] - uy[ci]
        sd = np.std(np.r_[ux, uy])
        boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)]
        out["matched_loss"] = {"pairs": len(pairs), "mean_difference_u": float(d.mean()), "in_sd_units": float(d.mean() / sd) if sd > 0 else None,
                               "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], "share_positive": float((d > 0).mean())}
    return out


# ------------------------------------------------------------------------------------------------ R3
R3_SETS = ["P1f", "P1x", "D_in-fin", "D_in-x", "R-transplant", "E12-F32", "E12-X32", "E12-F14", "E12-Ftwin", "P2", "P2-clean", "P2x"]


def push_contrast(full, sx, sy, b):
    """Per-token-mean push of sx minus sy over all rows of the questions both sets answer (block b; all classes)."""
    shared = set(full[sx].qid) & set(full[sy].qid)
    out = []
    for s in (sx, sy):
        F = full[s]
        m = np.array([q in shared for q in F.qid])
        P = F.push[b][m].sum(1).sum(0)
        out.append(P / F.count[m].sum())
    return out[0] - out[1], int(len(shared))


def r3(full, enc, atoms, rates):
    b = enc.b
    con = {f"{sx} - {sy}": push_contrast(full, sx, sy, b) for sx, sy in (("P2-clean", "P2-narrow"), ("P2-clean", "S2-terse"))}
    res = {}
    sets_all = [s for s in X.SETS if s in enc.u]
    for name, j in atoms.items():
        if j is None:
            continue
        d = enc.W_dec[j]
        r = {"atom": int(j), "cos_minus_d": {n: float(-d @ unit(vdir(n, b))) for n in DIRNAMES},
             "cos_with_push_contrasts": {k: float(-d @ unit(v[0])) for k, v in con.items()},
             "class_shares": class_shares(full, enc, j, sets_all), "frequency_on_training_tokens": float(enc.freq[j])}
        s800 = enc.u["S2-800"][:, j]
        r["vs_S2-800"] = {s: {"set_mean_u": float(enc.u[s][:, j].mean()), "ratio_to_S2-800": float(enc.u[s][:, j].mean() / s800.mean()) if s800.mean() > 0 else None,
                              "auroc_vs_S2-800": float(S.auroc_cols(enc.u[s][:, j], s800)[0])} for s in R3_SETS if s in enc.u}
        um = [float(enc.u[s][:, j].mean()) for s in sets_all]
        lm = [float(full[s].nll[enc.trow[s]].mean()) for s in sets_all]
        rr = [rates[s] for s in sets_all]
        r["spearman_with_student_rate"] = {"atom_set_mean_u": S.spearman(um, rr), "base_mean_loss": S.spearman(lm, rr), "sets": len(sets_all)}
        res[name] = r
    res["push_contrast_cos_v_EM_fin"] = {k: float(unit(v[0]) @ unit(vdir("EM-fin", b))) for k, v in con.items()}
    return res


# ------------------------------------------------------------------------------------------------ R2b
def r2b(full, win, enc):
    b = enc.b
    grid = [10 ** e for e in np.arange(-4, 2.01, 0.5)]
    trs = [("P2-clean", 1), ("P2-narrow", 0), ("S2-terse", 0), ("S2-odd", 0)]
    feats = {"u_win": lambda s: enc.uw[s], "raw_gradient": lambda s: win[s].rowmean_push(b)}
    out = {"block": b}
    for fname, f in feats.items():
        Xs, ys, gs = [], [], []
        for s, y in trs:
            m = np.array([in_half(q, "A") for q in win[s].qid])
            Xs.append(f(s)[m])
            ys.append(np.full(int(m.sum()), y))
            gs += [q for q, mm in zip(win[s].qid, m) if mm]
        Xtr, ytr, gtr = np.concatenate(Xs), np.concatenate(ys).astype(float), np.array(gs)
        Cbest, table = S.choose_C(Xtr, ytr, gtr, grid)
        ix, iy, q = win_pairs(win, "P2x", "P2x-narrow", "notA")
        Xt, Xa, Xb = S.standardise(Xtr, f("P2x")[ix], f("P2x-narrow")[iy])
        coef, b0 = S.logreg_fit(Xt, ytr, Cbest, S.balanced_weights(ytr))
        sx, sy = Xa @ coef + b0, Xb @ coef + b0
        nx, ny = win["P2x"].nll[ix], win["P2x-narrow"].nll[iy]
        c = float(S.c_cols(sx, sy, nx, ny))
        draws = S.boot_c(sx, sy, nx, ny, seed=X.SEED)
        tr_auc = float(S.auroc_cols(Xt[ytr == 1] @ coef, Xt[ytr == 0] @ coef)[0])
        out[fname] = {"C": Cbest, "cv_logloss": {f"{k:.4g}": v for k, v in table.items()}, "train_windows": [int((ytr == 1).sum()), int((ytr == 0).sum())],
                      "train_auroc": tr_auc, "test2_pairs": len(ix), "c": c, "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
                      "lb_2.5": float(np.percentile(draws, 2.5)), "passes": bool(c >= 0.60 and np.percentile(draws, 2.5) > 0.5),
                      "auroc_unpaired_raw_score": float(S.auroc_cols(sx, sy)[0])}
        log(f"R2b block {b} {fname}: C {Cbest:.4g}, train AUROC {tr_auc:.3f}, Test-2 c {c:.3f} [{out[fname]['ci95'][0]:.3f}, {out[fname]['ci95'][1]:.3f}]")
    # where the signal sits: P2-clean - S2-terse per-token-mean push over test rows of their shared test questions
    parts = []
    for s in ("P2-clean", "S2-terse"):
        F = full[s]
        shared = set(full["P2-clean"].qid) & set(full["S2-terse"].qid)
        sel = np.array([F.qid[i] in shared for i in enc.trow[s]])
        rc = enc.recon[s][sel].sum(0)                                                        # [3, H]: g, dense part of g_hat, g - g_hat
        n = F.count[enc.trow[s][sel]].sum()
        parts.append(-rc / n)                                                                # push = -g
    con = parts[0] - parts[1]
    v = unit(vdir("EM-fin", b))
    out["where"] = {"dense_atoms": int(enc.dense.sum()), "largest_atom_frequency": float(enc.freq.max()), "atoms_over_10pct": int((enc.freq > 0.1).sum()),
                    "cos_v_EM_fin": {"whole contrast": float(unit(con[0]) @ v),
                    "dense-atom part": float(unit(con[1]) @ v) if np.linalg.norm(con[1]) > 0 else None, "residual g - g_hat": float(unit(con[2]) @ v)},
                    "norm_share": {"dense-atom part": float(np.linalg.norm(con[1]) / np.linalg.norm(con[0])), "residual": float(np.linalg.norm(con[2]) / np.linalg.norm(con[0]))}}
    return out


# ------------------------------------------------------------------------------------------------ main
def load_all(keep_push=("P2-clean", "P2-narrow", "S2-terse")):
    R = X.rows_index()
    full = {s: Full(s, R, keep_push) for s in X.SETS if (SRC / "stats" / "full" / f"{s}.pt").exists()}
    win = {s: Win(s) for s in X.WINDOW_TEST_SETS + X.WINDOW_ALL_SETS if (SRC / "stats" / "win" / f"{s}.pt").exists()}
    return full, win


def main():
    RES.mkdir(parents=True, exist_ok=True)
    full, win = load_all()
    log(f"loaded Job A statistics of {len(full)} sets and windows of {len(win)} sets")
    rates = {s: v["rate"] for s, v in X.SETS.items()}
    if PART == "main":
        out = {"blocks": {}}
        for b in (ONLY or X.BLOCKS):
            enc = Enc(b, full, win)
            log(f"block {b}: dictionary {enc.Lat} latents, FVE {enc.record['fve_overall']:.4f}")
            o = {"dictionary": {k: v for k, v in enc.record.items() if k != "history"}, "r1": r1(full, win, enc)}
            log(f"R1 block {b}: j* {o['r1']['j_star']} m A {o['r1']['m_A']:.3f} B {o['r1']['m_B']:.3f}; c(S2-odd vs S2-terse, B) {o['r1']['c_B_S2odd_vs_S2terse']}")
            o["r2"] = r2(full, win, enc, o["r1"]["passes"])
            cand = o["r2"].get("candidate")
            log(f"R2 block {b}: prefilter pass {o['r2']['n_prefilter_pass']}; candidate {json.dumps(cand)[:600] if cand else None}")
            atoms = {"j_star": o["r1"]["j_star"], "candidate": cand["atom"] if cand else None, "brevity": o["r1"]["brevity_atom"]["atom"]}
            o["r3"] = r3(full, enc, atoms, rates)
            out["blocks"][b] = o
            del enc
        r1_any = any(out["blocks"][b]["r1"]["passes"] for b in out["blocks"])
        m_any = any(out["blocks"][b]["r1"]["m_condition"] for b in out["blocks"])
        out["R1"] = "DICTIONARY SEES A LESSON" if r1_any else ("UNCALIBRATED" if m_any else "DICTIONARY BLIND")
        cands = [out["blocks"][b]["r2"]["candidate"] for b in out["blocks"] if out["blocks"][b]["r2"].get("candidate")]
        test2_pairs = len(win_pairs(win, "P2x", "P2x-narrow", "notA")[0])
        if not cands:
            lab = "NO SPREAD ATOM"
        elif not any(c["replicated"] for c in cands):
            lab = "NOT REPLICATED"
        elif test2_pairs < 150:
            lab = "UNTESTABLE"
        elif any(c["beats_loss"] for c in cands):
            lab = "ATOM BEATS LOSS"
        else:
            lab = "ATOM = LOSS"
        out["R2"] = lab + ("" if r1_any else " (descriptive only: R1 did not pass)")
        out["test2_pairs"] = test2_pairs
        X.write_json(RES / ("stage2_test.json" if TESTOUT else "stage2.json"), out)
        log(f"labels: R1 = {out['R1']}; R2 = {out['R2']}")
    elif PART == "r2b":
        out = {"blocks": {}}
        for b in (ONLY or X.BLOCKS):
            enc = Enc(b, full, win)
            out["blocks"][b] = r2b(full, win, enc)
            del enc
        sig = any(out["blocks"][b]["raw_gradient"]["passes"] for b in out["blocks"])
        keep = any(out["blocks"][b]["u_win"]["passes"] for b in out["blocks"])
        out["R2b"] = ("SIGNAL PRESENT" + (", DICTIONARY KEEPS IT" if keep else "")) if sig else "NO SIGNAL"
        out["u_win_passes_anywhere"] = keep
        X.write_json(RES / ("stage2_r2b_test.json" if TESTOUT else "stage2_r2b.json"), out)
        log(f"label: R2b = {out['R2b']}")
    elif PART == "s18":
        st = json.loads((RES / "stage2.json").read_text())
        out = {}
        for b in X.BLOCKS:
            if not (DICTD / f"s18_b{b}.pt").exists():
                continue
            cand = st["blocks"][str(b)]["r2"].get("candidate")
            if not cand:
                continue
            main_ = torch.load(DICTD / f"main_b{b}.pt", weights_only=True)["W_dec"].double().numpy()
            enc = Enc(b, full, win, tag="s18")
            cosm = enc.W_dec @ main_[cand["atom"]]
            j = int(np.argmax(cosm))
            out[b] = {"candidate": cand["atom"], "best_match_s18": j, "cosine": float(cosm[j]),
                      "test2": c_boot_atom(win, enc, j, "P2x", "P2x-narrow", "notA"),
                      "s_prime_A": float(np.min([c_atoms(win, enc, sx, sy, "A", cols=[j])[0][0] for sx, sy in R2_CONTRASTS])),
                      "s_prime_B": float(np.min([c_atoms(win, enc, sx, sy, "B", cols=[j])[0][0] for sx, sy in R2_CONTRASTS]))}
            log(f"seed 18, block {b}: best match {j} cos {cosm[j]:.3f}; Test 2 {out[b]['test2']}")
        X.write_json(RES / "s18.json", out)


if __name__ == "__main__":
    main()
