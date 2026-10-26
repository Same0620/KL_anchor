"""Login node, CPU: the full-row baseline of §3 -- AUROC of P2x vs P2x-narrow from [base mean NLL per answer token, log answer length].

Inputs: Experiment 6's stored per-row losses (`exp6/scores/<set>.jsonl`: `loss` = minus the mean log-probability per answer token under the
32-bit base model, end token included if the answer was finished; `n_tokens` = answer tokens). Method: Experiment 5's addendum code
(read-only): L2 logistic regression (C = 1) on standardised features, 5-fold out-of-fold predictions with folds stratified by set (seed 17),
AUROC of the out-of-fold predictions; 95% interval from 10,000 bootstrap resamples of rows within each set.

  env/venv/bin/python -I exp17/src/e17_baseline.py   -> results/baseline_fullrow.json
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e17common as X
sys.path.insert(1, str(X.ROOT / "exp5" / "src"))
import e5_addendum as AD  # noqa: E402  (read-only: auroc, logreg, oof)


def main():
    S = {s: X.read_jsonl(X.ROOT / "exp6" / "scores" / f"{s}.jsonl") for s in ("P2x", "P2x-narrow")}
    for s, rows in S.items():
        assert len(rows) == 800 and [r["row"] for r in rows] == list(range(800)), s
    loss = np.r_[[r["loss"] for r in S["P2x"]], [r["loss"] for r in S["P2x-narrow"]]]
    ln = np.log(np.r_[[r["n_tokens"] for r in S["P2x"]], [r["n_tokens"] for r in S["P2x-narrow"]]].astype(float))
    y = np.r_[np.ones(800), np.zeros(800)]
    rng = np.random.default_rng(X.SEED)
    folds = np.r_[rng.permutation(800) % 5, rng.permutation(800) % 5]
    pred, coefs = AD.oof(np.c_[loss, ln], y, folds)
    a = AD.auroc(pred[:800], pred[800:])
    boots = np.empty(10000)
    for b in range(10000):
        i, j = rng.integers(0, 800, 800), 800 + rng.integers(0, 800, 800)
        boots[b] = AD.auroc(pred[i], pred[j])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    pl, _ = AD.oof(loss[:, None], y, folds)
    out = {"auroc_loss_and_log_length_out_of_fold": a, "interval_95": [float(lo), float(hi)],
           "for_information": {"auroc_loss_alone_out_of_fold": AD.auroc(pl[:800], pl[800:]), "auroc_raw_loss": AD.auroc(loss[:800], loss[800:]),
                               "auroc_raw_log_length": AD.auroc(ln[:800], ln[800:]), "mean_standardised_coefficients (intercept, loss, log length)": np.mean(coefs, 0).tolist(),
                               "mean_loss": {"P2x": float(loss[:800].mean()), "P2x-narrow": float(loss[800:].mean())},
                               "mean_answer_tokens": {"P2x": float(np.exp(ln[:800]).mean()), "P2x-narrow": float(np.exp(ln[800:]).mean())}},
           "inputs": "exp6/scores/P2x.jsonl, exp6/scores/P2x-narrow.jsonl (800 rows each)", "method": "Experiment 5 addendum: logreg C=1, 5-fold out-of-fold, folds stratified by set, seed 17"}
    X.write_json(X.RESULTS / "baseline_fullrow.json", out)
    print(f"[baseline] AUROC P2x vs P2x-narrow from [loss, log length], out of fold: {a:.4f} [{lo:.4f}, {hi:.4f}]; "
          f"loss alone {out['for_information']['auroc_loss_alone_out_of_fold']:.4f}; raw log length {out['for_information']['auroc_raw_log_length']:.4f}")


if __name__ == "__main__":
    main()
