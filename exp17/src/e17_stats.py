"""Statistics for Experiment 17 (CPU): AUROC per column, the paired statistic c with OLS residualisation, question bootstraps,
an L2 logistic regression solved by L-BFGS, out-of-fold AUROC, loss-matched pairs. All rules as in PREREG Part A and B12-B13."""
import numpy as np
import torch

torch.set_default_dtype(torch.float64)


def auroc_cols(A, B, chunk=512):
    """A [nA, L], B [nB, L] -> [L]: P(a > b) + 0.5 P(a = b) over all pairs, per column."""
    A, B = torch.as_tensor(A, dtype=torch.float64), torch.as_tensor(B, dtype=torch.float64)
    if A.dim() == 1:
        A, B = A[:, None], B[:, None]
    out = torch.empty(A.shape[1])
    for c in range(0, A.shape[1], chunk):
        a, b = A[:, None, c:c + chunk], B[None, :, c:c + chunk]
        out[c:c + chunk] = ((a > b).double().mean((0, 1)) + 0.5 * (a == b).double().mean((0, 1)))
    return out.numpy()


def design(nll):
    nll = np.asarray(nll, dtype=np.float64)
    return np.c_[np.ones_like(nll), nll, nll ** 2]


def residualise(U, nll):
    """U [n, L] (or [n]); OLS on [1, nll, nll^2] over all n rows -> residuals of the same shape."""
    U = np.asarray(U, dtype=np.float64)
    one = U.ndim == 1
    if one:
        U = U[:, None]
    D = design(nll)
    beta, *_ = np.linalg.lstsq(D, U, rcond=None)
    R = U - D @ beta
    return R[:, 0] if one else R


def c_cols(UX, UY, nllX, nllY):
    """Paired c per column: UX[i] and UY[i] belong to the same question. Residualised on [1, NLL, NLL^2] over the pooled 2n windows."""
    UX, UY = np.asarray(UX, dtype=np.float64), np.asarray(UY, dtype=np.float64)
    one = UX.ndim == 1
    if one:
        UX, UY = UX[:, None], UY[:, None]
    n = UX.shape[0]
    R = residualise(np.r_[UX, UY], np.r_[nllX, nllY])
    rx, ry = R[:n], R[n:]
    c = (rx > ry).mean(0) + 0.5 * (rx == ry).mean(0)
    return c[0] if one else c


def boot_c(uX, uY, nllX, nllY, n_boot=1000, seed=17, refit=True):
    """Bootstrap over questions (pairs) -> array of c draws. The residualisation is refitted on every draw."""
    rng = np.random.default_rng(seed)
    n = len(uX)
    out = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        out[b] = c_cols(np.asarray(uX)[i], np.asarray(uY)[i], np.asarray(nllX)[i], np.asarray(nllY)[i])
    return out


def logreg_fit(Xtr, ytr, C, w=None, iters=500):
    """L2 logistic regression, objective 0.5 |w|^2 + C * sum_i w_i * logloss_i (intercept not penalised); L-BFGS in float64.
    Xtr is assumed standardised. -> (coef [d], intercept)."""
    X = torch.as_tensor(Xtr, dtype=torch.float64)
    y = torch.as_tensor(ytr, dtype=torch.float64)
    sw = torch.ones_like(y) if w is None else torch.as_tensor(w, dtype=torch.float64)
    coef = torch.zeros(X.shape[1], requires_grad=True)
    b0 = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([coef, b0], lr=1, max_iter=iters, tolerance_grad=1e-9, tolerance_change=1e-12, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = X @ coef + b0
        loss = 0.5 * (coef ** 2).sum() + C * (sw * torch.nn.functional.binary_cross_entropy_with_logits(z, y, reduction="none")).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return coef.detach().numpy(), float(b0.detach())


def standardise(Xtr, *others):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    return [(Xtr - mu) / sd] + [(o - mu) / sd for o in others]


def balanced_weights(y):
    y = np.asarray(y)
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    return np.where(y == 1, len(y) / (2 * n1), len(y) / (2 * n0))


def choose_C(X, y, groups, grid, seed=17):
    """5-fold CV by group (question); C with the best mean held-out weighted log-loss. -> (best C, table)."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    fold_of = dict(zip(ug, rng.permutation(len(ug)) % 5))
    folds = np.array([fold_of[g] for g in groups])
    table = {}
    for C in grid:
        losses = []
        for f in range(5):
            tr, te = folds != f, folds == f
            if len(np.unique(y[tr])) < 2 or te.sum() == 0:
                continue
            Xt, Xe = standardise(X[tr], X[te])
            coef, b = logreg_fit(Xt, y[tr], C, balanced_weights(y[tr]))
            z = Xe @ coef + b
            p = 1 / (1 + np.exp(-z))
            we = balanced_weights(y[te])
            ll = -(we * (y[te] * np.log(np.clip(p, 1e-12, 1)) + (1 - y[te]) * np.log(np.clip(1 - p, 1e-12, 1)))).sum() / we.sum()
            losses.append(ll)
        table[C] = float(np.mean(losses))
    best = min(table, key=table.get)
    return best, table


def matched_pairs(loss_p, loss_c, rng, caliper=0.05):
    """Experiment 5 addendum's greedy nearest-neighbour matching on loss without replacement (caliper 0.05)."""
    free = np.ones(len(loss_c), bool)
    pairs = []
    for i in rng.permutation(len(loss_p)):
        idx = np.flatnonzero(free)
        if len(idx) == 0:
            break
        d = np.abs(loss_c[idx] - loss_p[i])
        j = idx[np.argmin(d)]
        if abs(loss_c[j] - loss_p[i]) <= caliper:
            free[j] = False
            pairs.append((int(i), int(j)))
    return pairs


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)

    def rank(x):
        o = np.argsort(x, kind="mergesort")
        r = np.empty(len(x))
        r[o] = np.arange(len(x))
        # average ties
        _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
        s = np.zeros(len(cnt))
        np.add.at(s, inv, r)
        return (s / cnt)[inv]
    ra, rb = rank(a), rank(b)
    return float(np.corrcoef(ra, rb)[0, 1])
