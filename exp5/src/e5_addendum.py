"""Addendum to Experiment 5 (login node, offline): does the gradient score carry anything beyond surprise?

Plan: exp5/PREREG_addendum.md. Input: the columns `loss` and `cos_gradM_lm` of exp5/scores/<set>.jsonl.

  env/venv/bin/python -I exp5/src/e5_addendum.py      -> exp5/results/addendum.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e5common as X
from e5common import C

N_BOOT, SEED, CALIPER = 10000, 20261015, 0.05


def auroc(pos, neg):
    """Probability that a value of `pos` is above a value of `neg` (ties count half)."""
    v = np.concatenate([pos, neg])
    _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    r = (np.cumsum(cnt) - (cnt - 1) / 2)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def interval(v):
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(lo), float(hi)


# ------------------------------------------------------------------------------------------------- A
def matched_pairs(loss_p, loss_c, rng):
    """Greedy nearest-neighbour matching on loss without replacement, in a random order of the P2-clean rows."""
    free = np.ones(len(loss_c), bool)
    pairs = []
    for i in rng.permutation(len(loss_p)):
        idx = np.flatnonzero(free)
        if len(idx) == 0:
            break
        d = np.abs(loss_c[idx] - loss_p[i])
        j = idx[np.argmin(d)]                       # argmin returns the first minimum = the lower row number on a tie
        if abs(loss_c[j] - loss_p[i]) <= CALIPER:
            free[j] = False
            pairs.append((int(i), int(j)))
    return pairs


def part_a(S, control):
    rng = np.random.default_rng(SEED)
    p, c = S["P2-clean"], S[control]
    pairs = matched_pairs(p["loss"], c["loss"], rng)
    n = len(pairs)
    out = {"control": control, "pairs": n, "underpowered": bool(n < 50)}
    if n == 0:
        return out
    pi, ci = np.array([a for a, _ in pairs]), np.array([b for _, b in pairs])
    d = p["grad"][pi] - c["grad"][ci]
    boots = d[rng.integers(0, n, size=(N_BOOT, n))].mean(1)
    lo, hi = interval(boots)
    out.update({"mean_difference_in_grad": float(d.mean()), "lo": lo, "hi": hi, "excludes_0": bool(lo > 0 or hi < 0),
                "share_of_pairs_with_positive_difference": float((d > 0).mean()),
                "mean_grad_P2-clean_rows": float(p["grad"][pi].mean()), "mean_grad_control_rows": float(c["grad"][ci].mean()),
                "mean_loss_P2-clean_rows": float(p["loss"][pi].mean()), "mean_loss_control_rows": float(c["loss"][ci].mean()),
                "loss_range_of_pairs": [float(p["loss"][pi].min()), float(p["loss"][pi].max())],
                "mean_abs_loss_difference": float(np.abs(p["loss"][pi] - c["loss"][ci]).mean()),
                "set_mean_loss": {"P2-clean": float(p["loss"].mean()), control: float(c["loss"].mean())}})
    return out


# ------------------------------------------------------------------------------------------------- B
def logreg(Xm, y, c_reg=1.0, iters=100):
    """L2-penalised logistic regression (objective of scikit-learn's default: 0.5 w.w + C * sum of log-losses;
    the intercept is not penalised), solved by Newton steps. Xm: [n, k] standardised features; y in {0, 1}."""
    n, k = Xm.shape
    A = np.hstack([np.ones((n, 1)), Xm])
    theta = np.zeros(k + 1)
    pen = np.r_[0.0, np.ones(k)] / c_reg
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(A @ theta)))
        g = A.T @ (p - y) + pen * theta
        H = (A * (p * (1 - p))[:, None]).T @ A + np.diag(pen)
        step = np.linalg.solve(H, g)
        theta = theta - step
        if np.abs(step).max() < 1e-10:
            break
    return theta, float(np.abs(A.T @ (1 / (1 + np.exp(-(A @ theta))) - y) + pen * theta).max())


def self_test():
    """Check the solver on made-up data (not on the real rows)."""
    rng = np.random.default_rng(1)
    Xm = rng.normal(size=(20000, 2))
    y = (rng.random(20000) < 1 / (1 + np.exp(-(0.3 + 1.5 * Xm[:, 0] - 0.7 * Xm[:, 1])))).astype(float)
    theta, gmax = logreg(Xm, y)
    ok = bool(gmax < 1e-6 and np.abs(theta - np.array([0.3, 1.5, -0.7])).max() < 0.08)
    return {"ok": ok, "estimated": theta.tolist(), "true": [0.3, 1.5, -0.7], "largest_gradient_at_solution": gmax}


def oof(feat, y, folds):
    """Out-of-fold predicted probabilities; features standardised with the training folds."""
    pred = np.zeros(len(y))
    coefs = []
    for f in range(5):
        tr, te = folds != f, folds == f
        mu, sd = feat[tr].mean(0), feat[tr].std(0)
        theta, _ = logreg((feat[tr] - mu) / sd, y[tr])
        pred[te] = 1 / (1 + np.exp(-(np.hstack([np.ones((te.sum(), 1)), (feat[te] - mu) / sd]) @ theta)))
        coefs.append(theta.tolist())
    return pred, coefs


def part_b(S, pos, neg):
    rng = np.random.default_rng(SEED)
    loss = np.r_[S[pos]["loss"], S[neg]["loss"]]
    grad = np.r_[S[pos]["grad"], S[neg]["grad"]]
    n1, n0 = len(S[pos]["loss"]), len(S[neg]["loss"])
    y = np.r_[np.ones(n1), np.zeros(n0)]
    folds = np.r_[rng.permutation(n1) % 5, rng.permutation(n0) % 5]            # stratified by set
    p1, c1 = oof(loss[:, None], y, folds)
    p2, c2 = oof(np.c_[loss, grad], y, folds)
    a1, a2 = auroc(p1[:n1], p1[n1:]), auroc(p2[:n1], p2[n1:])
    diffs, b1, b2 = np.empty(N_BOOT), np.empty(N_BOOT), np.empty(N_BOOT)
    for b in range(N_BOOT):
        i, j = rng.integers(0, n1, n1), n1 + rng.integers(0, n0, n0)
        b1[b], b2[b] = auroc(p1[i], p1[j]), auroc(p2[i], p2[j])
        diffs[b] = b2[b] - b1[b]
    lo, hi = interval(diffs)
    return {"positive_set": pos, "negative_set": neg, "auroc_loss_alone": a1, "auroc_loss_alone_interval": list(interval(b1)),
            "auroc_loss_plus_grad": a2, "auroc_loss_plus_grad_interval": list(interval(b2)), "difference": a2 - a1, "lo": lo, "hi": hi,
            "excludes_0": bool(lo > 0 or hi < 0),
            "for_information": {"auroc_raw_loss": auroc(S[pos]["loss"], S[neg]["loss"]), "auroc_raw_grad": auroc(S[pos]["grad"], S[neg]["grad"]),
                                "mean_standardised_coefficients_loss_plus_grad (intercept, loss, grad)": np.mean(c2, 0).tolist(),
                                "mean_standardised_coefficients_loss_alone (intercept, loss)": np.mean(c1, 0).tolist()}}


# ------------------------------------------------------------------------------------------------- C
def part_c(S):
    rng = np.random.default_rng(SEED)
    loss = np.concatenate([S[s]["loss"] for s in X.SETS])
    grad = np.concatenate([S[s]["grad"] for s in X.SETS])
    b, a = np.polyfit(loss, grad, 1)
    fit = a + b * loss
    out = {"intercept": float(a), "slope": float(b), "n_rows": int(len(loss)),
           "r_squared": float(1 - ((grad - fit) ** 2).sum() / ((grad - grad.mean()) ** 2).sum()), "sets": {}}
    for s in X.SETS:
        res = S[s]["grad"] - (a + b * S[s]["loss"])
        boots = res[rng.integers(0, len(res), size=(N_BOOT, len(res)))].mean(1)
        lo, hi = interval(boots)
        out["sets"][s] = {"mean_residual": float(res.mean()), "lo": lo, "hi": hi, "mean_loss": float(S[s]["loss"].mean()),
                          "mean_grad": float(S[s]["grad"].mean())}
    x, n = out["sets"]["P2x"], out["sets"]["P2x-narrow"]
    out["P2x_above_P2x-narrow_non_overlapping"] = bool(x["mean_residual"] > n["mean_residual"] and x["lo"] > n["hi"])
    return out


def main():
    S = {}
    for s in X.SETS:
        rows = C.read_jsonl(X.SCORES / f"{s}.jsonl")
        assert len(rows) == 800
        S[s] = {"loss": np.array([r["loss"] for r in rows]), "grad": np.array([r["cos_gradM_lm"] for r in rows])}
    R = {"self_test_of_logistic_regression": self_test()}
    assert R["self_test_of_logistic_regression"]["ok"], R["self_test_of_logistic_regression"]
    R["A"] = {"S2-odd": part_a(S, "S2-odd"), "S2-terse": part_a(S, "S2-terse")}
    R["B"] = {"P2-clean_vs_S2-odd": part_b(S, "P2-clean", "S2-odd"), "P2x_vs_P2x-narrow": part_b(S, "P2x", "P2x-narrow")}
    R["C"] = part_c(S)
    a, b = R["A"]["S2-odd"], R["B"]["P2-clean_vs_S2-odd"]
    c1 = bool(a["pairs"] >= 50 and a.get("mean_difference_in_grad", 0) > 0 and a.get("excludes_0", False))
    c2 = bool(b["difference"] >= 0.02 and b["excludes_0"])
    c3 = R["C"]["P2x_above_P2x-narrow_non_overlapping"]
    a_includes_0 = bool(a["pairs"] > 0 and not a["excludes_0"])
    surprise_only = bool(a_includes_0 and not b["excludes_0"])
    R["conditions"] = {"1_A_positive_interval_excludes_0_and_at_least_50_pairs": c1, "2_B_difference_at_least_0.02_interval_excludes_0": c2,
                       "3_C_residual_P2x_above_P2x-narrow_non_overlapping": c3}
    R["label"] = "BEYOND SURPRISE" if (c1 and c2 and c3) else "SURPRISE ONLY" if surprise_only else "MIXED"
    (X.RESULTS / "addendum.json").write_text(json.dumps(R, indent=1))
    f = lambda d, k="mean_difference_in_grad": f"{d[k]:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]"
    print("self-test of the logistic regression:", R["self_test_of_logistic_regression"]["ok"], [round(v, 3) for v in R["self_test_of_logistic_regression"]["estimated"]])
    for k, v in R["A"].items():
        print(f"A  P2-clean vs {k}: pairs {v['pairs']}" + ("" if v["pairs"] == 0 else f"; difference {f(v)}; mean loss of pairs {v['mean_loss_P2-clean_rows']:.2f} "
              f"(sets: {v['set_mean_loss']}); positive in {100 * v['share_of_pairs_with_positive_difference']:.0f}% of pairs") + f"; underpowered {v['underpowered']}")
    for k, v in R["B"].items():
        print(f"B  {k}: loss alone {v['auroc_loss_alone']:.4f}; loss + grad {v['auroc_loss_plus_grad']:.4f}; difference {f(v, 'difference')}; "
              f"raw AUROC loss {v['for_information']['auroc_raw_loss']:.4f}, grad {v['for_information']['auroc_raw_grad']:.4f}")
    print(f"C  line: grad = {R['C']['intercept']:+.4f} + {R['C']['slope']:+.4f} x loss; R^2 {R['C']['r_squared']:.3f}")
    for s, v in R["C"]["sets"].items():
        print(f"   {s:11s} mean residual {v['mean_residual']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]   (mean loss {v['mean_loss']:.2f}, mean grad {v['mean_grad']:+.4f})")
    print("conditions:", R["conditions"], "-> LABEL:", R["label"])


if __name__ == "__main__":
    main()
