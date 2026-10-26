# PREREG addendum — Experiment 5: does the gradient carry anything beyond surprise?

Written on 2026-10-11 (23:50), before any of the analyses A, B and C below was computed. Not changed afterwards (the
file is made read-only). Offline analysis only: no GPU job, no new data. Results go to `exp5/RESULTS_addendum.md`.

**What was already known when this was written.** The main results of Experiment 5 (`exp5/RESULTS.md`): the set means
of the loss and of the gradient score, the AUROC of each score alone (P2-clean against S2-terse and against S2-odd),
the rank correlation between loss and gradient score over all rows (+0.77) and inside each set. None of the three
analyses below had been run in any form.

## Part A — from the addendum text

**Definitions.** "loss" = mean per-token loss under the base model. "grad" = the primary gradient score,
cos(gradient of the row, gradient of M_lm). Both are already computed for all 8 sets (per-row files in `exp5/scores/`).

**A. Loss-matched pairs** (P2-clean vs S2-odd, and P2-clean vs S2-terse). Match each P2-clean row to its nearest
unused control row by loss. Keep a pair only if the two losses differ by at most 0.05 nats per token. Report: number
of pairs, the mean paired difference in grad, and a 95% bootstrap interval over pairs (10,000 resamples). If there
are fewer than 50 pairs, say so, and mark this part as underpowered.

**B. Added prediction beyond loss.** With 5-fold cross-validated logistic regression, separate P2-clean rows from
S2-odd rows using (i) loss alone, and (ii) loss + grad. Report both AUROCs and their difference, with a 95% bootstrap
interval over rows. Repeat with P2x rows against P2x-narrow rows.

**C. Residuals.** Fit a linear regression of grad on loss over all 6,400 rows. For each set, report the mean residual
with a 95% bootstrap interval.

**Labels.**
- **BEYOND SURPRISE:** all three hold:
  (1) the A difference (P2-clean − S2-odd) is > 0, with its interval excluding 0, and there are at least 50 pairs;
  (2) the B difference (P2-clean vs S2-odd) is ≥ 0.02, with its interval excluding 0;
  (3) the mean residual of P2x is above that of P2x-narrow, with non-overlapping intervals.
- **SURPRISE ONLY:** the A difference and the B difference (P2-clean vs S2-odd) both have intervals that include 0.
- **MIXED:** anything else.

## Part B — choices fixed here, before computing

1. **Data.** The columns `loss` and `cos_gradM_lm` of `exp5/scores/<set>.jsonl`, 800 rows per set. See item 8 about a
   fault in other columns of these files.
2. **A, matching.** The P2-clean rows are taken in a random order (fixed seed 20261015). Each takes the control row
   with the nearest loss among those not yet used (on a tie, the lower row number). If that nearest row is more than
   0.05 away, the P2-clean row gets no pair and the control row stays free. The difference is P2-clean minus control.
   I also report the mean loss of the matched rows, so the reader can see which part of each set the pairs come from.
3. **A, interval.** 10,000 resamples of the pairs with replacement, percentile interval 2.5% to 97.5% (seed 20261015).
   "Excluding 0" means both ends have the same sign.
4. **B, model.** scikit-learn is not installed and nothing is installed for this addendum. I use my own logistic
   regression with the same objective as scikit-learn's default: L2 penalty with C = 1, intercept not penalised,
   solved by Newton steps. Before use it is checked on made-up data (not on the real rows). Each feature is
   standardised with the mean and standard deviation of the training folds. Model (i) uses the loss; model (ii) uses
   the loss and grad, with no interaction term.
5. **B, cross-validation and AUROC.** 5 folds, stratified by set, fixed by seed 20261015. Every row gets one
   out-of-fold predicted probability per model. The AUROC of a model is computed from all out-of-fold predictions
   together. The difference is (ii) minus (i).
6. **B, interval.** 10,000 resamples of the rows with replacement, separately inside each of the two sets, keeping
   each row's two out-of-fold predictions together; the models are not refitted. Percentile interval as in item 3.
7. **C.** Ordinary least squares, grad = a + b × loss, over all 6,400 rows. The line is then kept fixed. For each set:
   the mean residual and a percentile interval from 10,000 resamples of that set's rows. "Non-overlapping" means that
   the lower end of P2x's interval is above the upper end of P2x-narrow's.
8. **A fault found in the saved files.** While reading the files for this addendum I saw that the columns `row`, `id`,
   `draw`, `n_tokens`, `judge_alignment` and `judge_coherence` of every `exp5/scores/<set>.jsonl` hold the values of
   the set P2x-narrow (a programming error of mine in `e5_analyze.py`: a loop variable was reused). The score columns
   (`loss`, `cos_gradM_lm` and the others) are filled from the correct arrays. Before running A, B and C I will
   (a) check `loss` and `cos_gradM_lm` in every file against the raw files `exp5/scores/raw_<set>.pt`, (b) correct
   the error and rewrite the files, and (c) check that `exp5/results/metrics.json` does not change, i.e. that no
   number of `exp5/RESULTS.md` was affected. The outcome of these checks is reported in the results file.
9. **Labels** are checked in the order written: BEYOND SURPRISE, then SURPRISE ONLY, then MIXED. If part A has
   fewer than 50 pairs for P2-clean against S2-odd, condition (1) fails.
10. **One session.** `exp5/.lock` is held during the analysis and removed at the end. No job is submitted.
