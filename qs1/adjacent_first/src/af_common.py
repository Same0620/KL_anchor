"""QS1-F1 shared code (qs1/adjacent_first/): are adjacent topics reached first? CPU only; reads QS1's saved outputs in place.

Run every script with `nice env/venv/bin/python -I -B` (no __pycache__ in qs1/src). QS1's own modules (qs1common, qs1stats) are imported
read-only for the cell layout, the FE design and the Freedman-Lane test; everything new (the log-odds outcome, the size statistic G, the
paired bootstrap with E_base recomputed per replicate, the batched permutation refits) lives here.

Definitions (handoff section 2, PREREG Part B):
- lp = log p(M1 | x) as stored by QS1 (fp32 in scores.pt / base.pt); h = max(-lp, h_min); logit = -h - log(-expm1(-h)).
- outcomes per prompt: "dlogit" = logit_student - logit_base (primary); "gate" = lp - lp0 (QS1's dL_gate); "dfull" = full-tag dL.
- cells: QS1's order (sources in X.SOURCES order, targets in X.TOPICS order, minus the source), 88 per arm; cell = mean over the 24 questions.
- E: QS1's stored MAIN matrices (results/closeness/MAIN_<measure>.csv), z-scored over the 88 cells; E_perp = M_Z E_z, the exact two-way FE
  residual (the vector beta is identified from; the handoff's "minus source mean minus target mean plus grand mean", iterated to its fixed point).
- G = beta * dEperp_terc / m; dEperp_terc = mean over sources of (mean E_perp over the 7 highest - over the 7 lowest of its 22 targets).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/work/u4161854/loracle_em_exp1")
Q = ROOT / "qs1"
AF = Q / "adjacent_first"
RES, FIG, PRE = AF / "results", AF / "fig", AF / "prereg"
sys.path.insert(0, str(Q / "src"))
import qs1common as X  # noqa: E402  (read-only)
import qs1stats as S   # noqa: E402  (read-only)

torch.set_num_threads(4)
SOURCES, TOPICS, SHORT = X.SOURCES, X.TOPICS, X.SHORT
SETS24 = [X.topic_set(t) for t in TOPICS] + ["everyday48"]
PEAK_LR = 2e-5
NB, NPERM, ALPHA = 2000, 20000, 0.01
TERC = 7
H_MIN_DEFAULT = 1e-5
PRIMARIES = ("E_base_late", "E_grad")
BASE_MEASURES = ("E_base_late", "E_base_early")       # recomputed in each bootstrap replicate
LABELS = {"PLAIN": ["s5", "k-1", "k", "s10", "s20", "s40", "final"],
          "CONTRAST": ["s5", "k-1", "k", "s10", "s20", "s40", "s80", "final"]}


def read_json(p):
    return json.loads(Path(p).read_text())


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=float))
    tmp.replace(p)


# ---------------------------------------------------------------- outcomes
def logit_from_h(h):
    h = np.asarray(h, np.float64)
    return -h - np.log(-np.expm1(-h))


def headroom(lp, h_min):
    return np.maximum(-np.asarray(lp, np.float64), h_min)


# ---------------------------------------------------------------- data
class Data:
    """QS1's saved numbers, read in place. lp arrays are float64 copies of the stored fp32 values."""

    def __init__(self, h_min=H_MIN_DEFAULT, students=None):
        self.h_min = h_min
        self.index = read_json(Q / "data" / "eval_index.json")
        self.rows = self.index["rows"]
        assert self.index["n"] == len(self.rows) == 696
        self.sl = {k: np.array(v) for k, v in X.set_slices(self.index).items()}
        b = torch.load(Q / "stats" / "job1" / "base.pt", map_location="cpu", weights_only=False)
        assert b["lp"].dtype == torch.float32 and tuple(b["lp"].shape) == (696, 7) and b["M"] == X.M_IDS
        self.base_lp = b["lp"].double().numpy()
        self.lp0 = self.base_lp[:, 0]
        self.E = {}
        for f in sorted((Q / "results" / "closeness").glob("MAIN_*.csv")):
            lines = f.read_text().strip().splitlines()
            sets = [l.split(",")[0] for l in lines[1:]]
            assert sets == SETS24, f
            self.E[f.stem[5:]] = np.array([[float(v) for v in l.split(",")[1:]] for l in lines[1:]])
        self.st = {}
        names = students or [n for n, _ in X.students()]
        for n in names:
            d = Q / "outputs" / "students" / n
            assert (d / "done.json").exists(), n
            sc = torch.load(d / "scores.pt", map_location="cpu", weights_only=False)
            rec = read_json(d / "record.json")
            plan = read_json(Q / "data" / "rows" / f"{n}.json")
            ev = {k: v.double().numpy() for k, v in sc["evals"].items()}
            for v in sc["evals"].values():
                assert v.dtype == torch.float32 and tuple(v.shape) == (696, 7), n
            self.st[n] = {"rec": rec, "plan": plan, "evals": ev, "intopic": {int(k): v.double().numpy() for k, v in sc["intopic"].items()},
                          "intopic_idx": list(sc["intopic_idx"]), "sets_order": list(sc["sets_order"]), "layers": list(sc["layers"]),
                          "dh": sc["dh_setmean"]}

    # per-prompt outcome vectors [696]
    def lp(self, n, label):
        return self.st[n]["evals"][label][:, 0]

    def outcome(self, n, label, kind):
        lp = self.lp(n, label)
        if kind == "gate":
            return lp - self.lp0
        if kind == "dlogit":
            return logit_from_h(headroom(lp, self.h_min)) - logit_from_h(headroom(self.lp0, self.h_min))
        if kind == "dfull":
            return self.st[n]["evals"][label].sum(1) - self.base_lp.sum(1)
        raise KeyError(kind)

    def cells(self, arm, seed=0):
        """QS1's 88-cell layout for an arm type: [(s, t)], sources in X.SOURCES order, targets in X.TOPICS order."""
        cl = []
        for s in SOURCES:
            n = X.student_name(arm, s, seed)
            trained = set(self.st[n]["plan"]["trained_topics"])
            cl += [(s, t) for t in TOPICS if t != s and t not in trained]
        return cl

    def cell_values(self, arm, label, kind, cl=None):
        cl = cl or self.cells(arm)
        out = []
        for s, t in cl:
            v = self.outcome(X.student_name(arm, s, 0), label, kind)
            out.append(float(v[self.sl[X.topic_set(t)]].mean()))
        return np.array(out)

    def e_cells(self, measure, cl, E=None):
        M = self.E[measure] if E is None else E
        return np.array([M[SETS24.index(X.topic_set(s)), SETS24.index(X.topic_set(t))] for s, t in cl])

    def cum_lr(self, n, label):
        rec = self.st[n]["rec"]
        step = step_of(rec, label)
        return float(np.sum(rec["lr"][:step]) / PEAK_LR)


def step_of(rec, label):
    if label == "final":
        return rec["steps"]
    if label == "k":
        return rec["k"]
    if label == "k-1":
        return rec["k"] - 1
    if label == "step0":
        return 0
    assert label.startswith("s"), label
    return int(label[1:])


# ---------------------------------------------------------------- FE algebra
class FE:
    """The two-way FE design over a cell list, its annihilator M_Z, and batched OLS / Freedman-Lane refits."""

    def __init__(self, cl, nuis=None):
        self.cl = cl
        Z, _, _ = S.fe_design(cl)
        if nuis is not None:
            Z = np.hstack([Z] + [S.zscore(c)[:, None] for c in nuis])
        self.Z = Z
        self.Mz = S.resid_maker(Z)
        self.df = len(cl) - Z.shape[1] - 1
        self.groups = np.array([s for s, _ in cl])
        self._perms = {}

    def perms(self, n_perm=NPERM, seed=0):
        """Exactly qs1stats.freedman_lane's permutation draws (same generator, same order), cached."""
        key = (n_perm, seed)
        if key not in self._perms:
            rng = np.random.default_rng(seed)
            idx = np.arange(len(self.cl))
            P = np.empty((n_perm, len(self.cl)), dtype=np.int64)
            ug = np.unique(self.groups)
            masks = [idx[self.groups == g] for g in ug]
            for k in range(n_perm):
                p = idx.copy()
                for m in masks:
                    p[m] = rng.permutation(m)
                P[k] = p
            self._perms[key] = P
        return self._perms[key]

    def ols(self, y, e):
        """y ~ Z + beta e (e as given, usually z-scored). -> dict as qs1stats.ols_beta_t plus e_perp."""
        y, e = np.asarray(y, float), np.asarray(e, float)
        et, yt = self.Mz @ e, self.Mz @ y
        eee = float(et @ et)
        beta = float(et @ yt) / eee
        r = yt - beta * et
        s2 = float(r @ r) / self.df
        se = (s2 / eee) ** 0.5
        sd_red = float(np.std(yt, ddof=0))
        return {"beta": beta, "t": beta / se if se > 0 else 0.0, "se": se, "df": self.df,
                "partial_r2": (float(yt @ yt) - float(r @ r)) / float(yt @ yt) if float(yt @ yt) > 0 else 0.0,
                "sd_reduced_resid": sd_red, "beta_std": beta / sd_red if sd_red > 0 else 0.0}

    def freedman_lane(self, y, e, n_perm=NPERM, seed=0):
        """One-sided both ways: p_up = (1 + #{t* >= t}) / (N + 1) (QS1's), p_low = (1 + #{t* <= t}) / (N + 1)."""
        y, e = np.asarray(y, float), np.asarray(e, float)
        obs = self.ols(y, e)
        ez = self.Mz @ y
        et = self.Mz @ e
        eee = float(et @ et)
        P = self.perms(n_perm, seed)
        YT = self.Mz @ ez[P].T                             # [n, N]
        B = (et @ YT) / eee
        R = YT - np.outer(et, B)
        S2 = (R * R).sum(0) / self.df
        with np.errstate(divide="ignore", invalid="ignore"):
            T = np.nan_to_num(B / np.sqrt(S2 / eee))
        p_up = (1 + int((T >= obs["t"] - 1e-12).sum())) / (n_perm + 1)
        p_low = (1 + int((T <= obs["t"] + 1e-12).sum())) / (n_perm + 1)
        return dict(obs, p=p_up, p_low=p_low, n_perm=n_perm)


def eperp_literal(e, cl):
    """The handoff's formula taken literally (one pass of double centring); reported beside the exact residual."""
    e = np.asarray(e, float)
    src = np.array([s for s, _ in cl])
    tgt = np.array([t for _, t in cl])
    sm = {s: e[src == s].mean() for s in set(src)}
    tm = {t: e[tgt == t].mean() for t in set(tgt)}
    return np.array([v - sm[s] - tm[t] for v, s, t in zip(e, src, tgt)]) + e.mean()


def terc_gap(v, cl, k=TERC):
    """Mean over sources of (mean v over the source's k highest-v targets - mean over its k lowest)."""
    v = np.asarray(v, float)
    src = np.array([s for s, _ in cl])
    gaps = []
    for s in SOURCES:
        x = np.sort(v[src == s])
        if len(x):
            gaps.append(x[-k:].mean() - x[:k].mean())
    return float(np.mean(gaps))


def terc_gap_y(y, e, cl, k=TERC):
    """Raw tercile gap with no FE: mean over sources of (mean y over the k highest-e targets - over the k lowest)."""
    y, e = np.asarray(y, float), np.asarray(e, float)
    src = np.array([s for s, _ in cl])
    gaps = []
    for s in SOURCES:
        m = src == s
        o = np.argsort(e[m], kind="mergesort")
        gaps.append(y[m][o[-k:]].mean() - y[m][o[:k]].mean())
    return float(np.mean(gaps))


def t_quantile(q, df):
    """Student-t quantile by bisection on the regularised incomplete beta (mpmath; scipy is not installed in the venv)."""
    import mpmath as mp
    mp.mp.dps = 30

    def cdf(t):
        x = df / (df + t * t)
        tail = 0.5 * mp.betainc(df / 2.0, 0.5, 0, x, regularized=True)
        return 1 - tail if t > 0 else tail
    lo, hi = 0.0, 50.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if cdf(mid) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def size_stats(fe, y, e_z, cl):
    """Point estimates of the size statistic for one (y, E): beta etc., G with E_perp, G with raw-E terciles, the raw tercile gap."""
    ols = fe.ols(y, e_z)
    ep = fe.Mz @ e_z
    m = float(np.mean(y))
    dterc = terc_gap(ep, cl)
    dterc_lit = terc_gap(eperp_literal(e_z, cl), cl)
    dterc_raw = terc_gap(e_z, cl)
    tq = t_quantile(0.975, fe.df)
    lo, hi = ols["beta"] - tq * ols["se"], ols["beta"] + tq * ols["se"]
    g_ols = sorted([lo * dterc / m, hi * dterc / m])
    raw_gap = terc_gap_y(y, e_z, cl)
    return {**ols, "m": m, "dEperp_terc": dterc, "dEperp_terc_literal": dterc_lit, "dEraw_terc": dterc_raw,
            "G": ols["beta"] * dterc / m, "G_literal_Eperp": ols["beta"] * dterc_lit / m, "G_rawE_terciles": ols["beta"] * dterc_raw / m,
            "G_ci_ols": g_ols, "t975": tq, "raw_terc_gap": raw_gap, "raw_terc_gap_over_m": raw_gap / m}


# ---------------------------------------------------------------- paired bootstrap (questions resampled once per replicate)
class Boot:
    """B replicates; each resamples every one of the 24 main sets' questions with replacement (seed 0), once, shared by every source,
    step, outcome and measure. E_base_late / E_base_early are recomputed per replicate from Job 1's per-prompt residuals via per-layer
    Gram matrices (exact algebra; replicate 'identity' reproduces QS1's stored matrices)."""

    def __init__(self, D, B=NB, seed=0):
        self.D, self.B = D, B
        rng = np.random.default_rng(seed)
        self.idx = {s: rng.integers(0, len(D.sl[s]), size=(B, len(D.sl[s]))) for s in SETS24}
        self._E = None

    def cell_reps(self, v, cl):
        """v [696] per-prompt -> [B, n_cells] replicate cell means (target set resampled)."""
        out = np.empty((self.B, len(cl)))
        cache = {}
        for j, (s, t) in enumerate(cl):
            ts = X.topic_set(t)
            if ts not in cache:
                cache[ts] = v[self.D.sl[ts]][self.idx[ts]].mean(1)
            out[:, j] = cache[ts]
        return out

    def set_reps(self, v, set_name):
        return v[self.D.sl[set_name]][self.idx[set_name]].mean(1)

    def ebase_reps(self, chunk=200, identity_check=True):
        """-> {"E_base_late": [B, 24, 24], "E_base_early": [B, 24, 24]} and the identity-replicate check."""
        if self._E is not None:
            return self._E
        b = torch.load(Q / "stats" / "job1" / "base.pt", map_location="cpu", weights_only=False)
        assert b["layers"] == X.HOOKED and b["h"].dtype == torch.float32
        H = b["h"].double().numpy()                                   # [696, 6, d]
        rows = np.concatenate([self.D.sl[s] for s in SETS24])          # 600 main rows, set-contiguous
        off = np.cumsum([0] + [len(self.D.sl[s]) for s in SETS24])
        Hr = H[rows]
        G = np.stack([Hr[:, l] @ Hr[:, l].T for l in range(Hr.shape[1])])   # [6, 600, 600]
        del Hr
        nset = len(SETS24)
        Hc = np.eye(nset) - 1.0 / nset

        def maps(C):                                                   # C [b, 24, 600] -> per-layer centred cosine [b, 6, 24, 24]
            Ct = np.ascontiguousarray(C.transpose(0, 2, 1))
            Gm = np.stack([(C.reshape(-1, C.shape[2]) @ G[l]).reshape(C.shape) @ Ct for l in range(G.shape[0])], 1)
            Gc = Hc @ Gm @ Hc
            d = np.sqrt(np.diagonal(Gc, axis1=-2, axis2=-1))
            return Gc / (d[..., :, None] * d[..., None, :])
        li = {L: X.HOOKED.index(L) for L in X.HOOKED}
        late, early = [li[L] for L in X.LATE], [li[L] for L in X.EARLY]
        C1 = np.zeros((1, nset, len(rows)))
        for j, s in enumerate(SETS24):
            C1[0, j, off[j]:off[j + 1]] = 1.0 / len(self.D.sl[s])
        M1 = maps(C1)[0]
        chk = {"E_base_late": float(np.abs(M1[late].mean(0) - self.D.E["E_base_late"]).max()),
               "E_base_early": float(np.abs(M1[early].mean(0) - self.D.E["E_base_early"]).max())}
        assert max(chk.values()) < 5e-6, chk                           # CSVs hold 6 decimals
        outL = np.empty((self.B, nset, nset))
        outE = np.empty((self.B, nset, nset))
        for b0 in range(0, self.B, chunk):
            b1 = min(self.B, b0 + chunk)
            C = np.zeros((b1 - b0, nset, len(rows)))
            for j, s in enumerate(SETS24):
                n = len(self.D.sl[s])
                ii = self.idx[s][b0:b1] + off[j]                       # [b, n] row positions
                np.add.at(C, (np.arange(b1 - b0)[:, None], j, ii), 1.0 / n)
            Mb = maps(C)
            outL[b0:b1] = Mb[:, late].mean(1)
            outE[b0:b1] = Mb[:, early].mean(1)
        self._E = {"E_base_late": outL, "E_base_early": outE, "identity_check_maxabs": chk}
        return self._E

    def e_reps(self, measure, cl):
        """[B, n_cells] raw E cells per replicate (recomputed for base measures, fixed otherwise)."""
        si = [SETS24.index(X.topic_set(s)) for s, _ in cl]
        ti = [SETS24.index(X.topic_set(t)) for _, t in cl]
        if measure in BASE_MEASURES:
            return self.ebase_reps()[measure][:, si, ti]
        return np.broadcast_to(self.D.e_cells(measure, cl), (self.B, len(cl)))


def boot_size(fe, Yb, Eb, cl):
    """Replicate G, beta, m: Yb [B, n] cell means, Eb [B, n] raw E cells (z-scored here per replicate)."""
    Ez = (Eb - Eb.mean(1, keepdims=True)) / Eb.std(1, ddof=0, keepdims=True)
    EP = Ez @ fe.Mz                                                    # M_Z symmetric
    beta = (EP * Yb).sum(1) / (EP * EP).sum(1)
    m = Yb.mean(1)
    src = np.array([s for s, _ in cl])
    gaps = []
    for s in SOURCES:
        x = np.sort(EP[:, src == s], axis=1)
        gaps.append(x[:, -TERC:].mean(1) - x[:, :TERC].mean(1))
    dterc = np.mean(gaps, 0)
    return {"G": beta * dterc / m, "beta": beta, "m": m, "dterc": dterc}


def g_ci(point, reps, ols_ci):
    """The wider of the bootstrap percentile interval and the OLS interval (handoff section 3)."""
    bci = [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]
    wider = bci if (bci[1] - bci[0]) >= (ols_ci[1] - ols_ci[0]) else list(ols_ci)
    return {"ci_boot": bci, "ci_ols": list(ols_ci), "ci": wider, "ci_from": "bootstrap" if wider is bci else "ols", "boot_sd": float(np.std(reps, ddof=1))}


def spearman(x, y):
    return X.spearman(x, y)
