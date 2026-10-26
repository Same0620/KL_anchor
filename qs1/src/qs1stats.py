"""Statistics for QS1 (numpy only; scipy is not installed): two-way fixed effects, Freedman-Lane permutation tests, Holm, Spearman.

"FE" = source fixed effects a_s + target fixed effects b_t over the cells (s, t), s in the 4 sources, t in the 23 topics minus s (88 cells,
26 FE columns incl. the intercept, 61 residual df once beta is added).
"""
import numpy as np

import qs1common as X


def cells(sources=None, topics=None, exclude=None):
    """[(s, t)] in a fixed order: sources in X.SOURCES order, targets in X.TOPICS order, minus the source and minus exclude[s] (topics the
    student trained on)."""
    sources = sources or X.SOURCES
    topics = topics or X.TOPICS
    exclude = exclude or {}
    return [(s, t) for s in sources for t in topics if t != s and t not in exclude.get(s, ())]


def fe_design(cl):
    srcs = sorted({s for s, _ in cl}, key=X.TOPICS.index)
    tgts = sorted({t for _, t in cl}, key=X.TOPICS.index)
    Z = np.zeros((len(cl), 1 + len(srcs) - 1 + len(tgts) - 1))
    Z[:, 0] = 1
    for i, (s, t) in enumerate(cl):
        a = srcs.index(s)
        b = tgts.index(t)
        if a > 0:
            Z[i, a] = 1
        if b > 0:
            Z[i, len(srcs) - 1 + b] = 1
    assert np.linalg.matrix_rank(Z) == Z.shape[1]
    return Z, srcs, tgts


def resid_maker(Z):
    P = Z @ np.linalg.pinv(Z.T @ Z) @ Z.T
    return np.eye(len(Z)) - P


def residualise(y, Z):
    return resid_maker(Z) @ np.asarray(y, float)


def zscore(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / x.std(ddof=0)


def ols_beta_t(y, Z, e):
    """Full model y ~ Z + beta e. -> beta, t, SE, partial R^2, residual SD of the reduced fit."""
    Mz = resid_maker(Z)
    et, yt = Mz @ e, Mz @ y
    eee = float(et @ et)
    beta = float(et @ yt) / eee
    r = yt - beta * et
    df = len(y) - Z.shape[1] - 1
    s2 = float(r @ r) / df
    se = (s2 / eee) ** 0.5
    ssr_red, ssr_full = float(yt @ yt), float(r @ r)
    t = beta / se if se > 0 else 0.0                # se = 0 only on degenerate test data
    return {"beta": beta, "t": t, "se": se, "partial_r2": (ssr_red - ssr_full) / ssr_red if ssr_red > 0 else 0.0, "df": df,
            "sd_reduced_resid": float(np.std(yt, ddof=0)), "beta_std": beta / float(np.std(yt, ddof=0)) if np.std(yt) > 0 else 0.0}


def freedman_lane(y, Z, e, groups, n_perm=20000, seed=0):
    """One-sided Freedman-Lane test of beta > 0 in y ~ Z + beta e. Residuals of the reduced model y ~ Z are permuted within groups
    (the source), Y* = Yhat + e[pi], the full model is refitted, the statistic is the OLS t of beta. p = (1 + #{t* >= t}) / (N + 1)."""
    y, e = np.asarray(y, float), np.asarray(e, float)
    Mz = resid_maker(Z)
    obs = ols_beta_t(y, Z, e)
    ez = Mz @ y                                   # reduced-model residuals (Yhat = y - ez)
    et = Mz @ e
    eee = float(et @ et)
    df = len(y) - Z.shape[1] - 1
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    idx = np.arange(len(y))
    perms = np.empty((n_perm, len(y)), dtype=np.int64)
    for k in range(n_perm):
        p = idx.copy()
        for g in np.unique(groups):
            m = idx[groups == g]
            p[m] = rng.permutation(m)
        perms[k] = p
    E = ez[perms].T                               # [n, N]: permuted reduced residuals; Y* = Yhat + E, and Mz Yhat = 0
    YT = Mz @ E                                   # Mz Y*
    B = (et @ YT) / eee
    R = YT - np.outer(et, B)
    S2 = (R * R).sum(0) / df
    with np.errstate(divide="ignore", invalid="ignore"):
        T = np.nan_to_num(B / np.sqrt(S2 / eee))
    p = (1 + int((T >= obs["t"] - 1e-12).sum())) / (n_perm + 1)
    return dict(obs, p=p, n_perm=n_perm)


def holm(pvals):
    """Holm-adjusted p-values (same order as given)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty(len(p))
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (len(p) - k) * p[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def spearman(x, y):
    return X.spearman(x, y)


def pearson(x, y):
    return float(np.corrcoef(np.asarray(x, float), np.asarray(y, float))[0, 1])


def boot_mean_se(v, n=2000, seed=0):
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, len(v), size=(n, len(v)))].mean(1)
    return float(bs.std(ddof=1))
