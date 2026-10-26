# RESULTS addendum — Experiment 5: does the gradient carry anything beyond surprise?

Offline analysis of the per-row scores saved in `exp5/scores/`, run on 2026-10-11. No GPU job, no new data. Plan fixed before computing: `exp5/PREREG_addendum.md` (sha256 starts `799eacd5a750bfbe`). Code: `exp5/src/e5_addendum.py`. Numbers: `exp5/results/addendum.json`.

**Words used below.** *Loss* is the mean per-token loss of an answer under the base model: how surprising the answer is to the base model. *Grad* is the primary gradient score of Experiment 5: the cosine between the answer's gradient and the gradient of M_lm, the direction "towards misaligned answers". The eight training sets have 800 rows each. P2-clean and P2x were written by broad (misaligned) twins and their students are misaligned. P2-narrow and P2x-narrow were written by narrow twins and their students are not misaligned. S2-800, S2-terse and S2-odd were written by the base model itself. *AUROC* is the probability that a row of the first set gets a higher score than a row of the second (0.5 = no separation). Intervals are 95% bootstrap intervals (10,000 resamples).

## 1. Answer

**Label: MIXED.** Conditions (1) and (2) hold, condition (3) does not.

- **(1) A, loss-matched pairs, P2-clean against S2-odd: holds.** 297 pairs; mean difference in grad +0.0694 [+0.0598, +0.0789].
- **(2) B, added prediction, P2-clean against S2-odd: holds.** AUROC 0.885 with loss alone, 0.927 with loss + grad; difference +0.0417 [+0.0306, +0.0532] (needs ≥ 0.02).
- **(3) C, residuals, P2x above P2x-narrow: does not hold.** Mean residual +0.0166 [+0.0119, +0.0212] for P2x and +0.0243 [+0.0208, +0.0279] for P2x-narrow: the narrow twin's set is higher, not lower.

In plain words: **the gradient score does carry something beyond surprise, but that something is not the persona.** At equal loss, an answer of the broad twin scores higher than an answer of the base model in the commentator voice, and adding grad to loss separates these two sets better. So grad is not just another way to measure loss. But once the loss is taken out, the answers of the *narrow* twins score as high as, or higher than, the answers of the broad twins, although only the broad twins' students become misaligned. And for telling P2x from P2x-narrow, grad adds nothing to loss (-0.0002 [-0.0019, +0.0015]). What grad sees beyond surprise looks like "written by a fine-tuned twin, not by the base model", not "written by a misaligned teacher". This last sentence is my reading of eight sets, not a tested claim.

## 2. A — loss-matched pairs

Each P2-clean row, taken in a fixed random order, was paired with the unused control row nearest in loss; a pair was kept only if the two losses differ by at most 0.05. The difference is grad of the P2-clean row minus grad of the control row.

| control set | pairs | mean difference in grad [95% interval] | pairs with a positive difference | mean loss of the paired rows | mean loss of the whole sets (P2-clean / control) | status |
|---|---|---|---|---|---|---|
| S2-odd | 297 | +0.0694 [+0.0598, +0.0789] | 78% | 3.06 | 3.52 / 2.47 | at least 50 pairs |
| S2-terse | 17 | +0.1376 [+0.0940, +0.1801] | 82% | 2.14 | 3.52 / 1.05 | **underpowered** (fewer than 50 pairs) |

- **Against S2-odd** the two sets overlap well in loss: 297 of 800 P2-clean rows found a partner. In these pairs the P2-clean row has a mean grad of +0.178 and the S2-odd row +0.109. The difference of the two set means, without matching, was +0.097; among loss-matched rows it is about 30% smaller, and most of it remains.
- **Against S2-terse** only 17 pairs exist, because the terse answers of the base model are much less surprising to it (mean loss 1.05 against 3.52). The difference points the same way, but this part is underpowered, as the plan says, and the pairs come from the low-loss edge of P2-clean and the high-loss edge of S2-terse.

## 3. B — added prediction beyond loss

5-fold cross-validated logistic regression; every row is scored by a model that did not see it. Model (i) uses the loss, model (ii) the loss and grad.

| comparison | AUROC, loss alone [95% interval] | AUROC, loss + grad [95% interval] | difference [95% interval] | for information: AUROC of raw loss / raw grad |
|---|---|---|---|---|
| P2-clean against S2-odd | 0.885 [0.868, 0.901] | 0.927 [0.914, 0.939] | +0.0417 [+0.0306, +0.0532] | 0.886 / 0.868 |
| P2x against P2x-narrow | 0.954 [0.944, 0.964] | 0.954 [0.944, 0.964] | -0.0002 [-0.0019, +0.0015] | 0.955 / 0.866 |

- **P2-clean against S2-odd:** grad adds +0.042 to the AUROC. In model (ii) both features carry weight (mean standardised coefficients: loss +1.70, grad +1.61).
- **P2x against P2x-narrow:** grad adds nothing (-0.0002). The loss alone already separates these two sets well (0.954), and the coefficient of grad in model (ii) is small (loss +3.46, grad +0.39).

## 4. C — residuals

One straight line was fitted over all 6,400 rows: grad = -0.0247 + 0.0551 × loss. It explains 60% of the variance of grad. The residual of a row is its grad minus the line's value for its loss.

| set | written by | student misaligned? | mean loss | mean grad | mean residual [95% interval] |
|---|---|---|---|---|---|
| P2 | finance broad twin | yes (10.1%) | 3.64 | +0.1871 | +0.0117 [+0.0068, +0.0163] |
| P2-clean | finance broad twin, judge-filtered | yes (7.4%) | 3.52 | +0.1866 | +0.0174 [+0.0125, +0.0223] |
| P2-narrow | finance narrow twin | no (0.0%) | 0.58 | +0.0319 | +0.0248 [+0.0223, +0.0273] |
| S2-800 | base model, default | no (0.0%) | 0.41 | -0.0028 | -0.0008 [-0.0027, +0.0011] |
| S2-terse | base model, told to be terse | no (0.0%) | 1.05 | -0.0393 | -0.0724 [-0.0761, -0.0687] |
| S2-odd | base model, commentator voice | no (0.0%) | 2.47 | +0.0898 | -0.0216 [-0.0257, -0.0175] |
| P2x | extreme-sports broad twin | yes (9.0%) | 3.50 | +0.1848 | +0.0166 [+0.0119, +0.0212] |
| P2x-narrow | extreme-sports narrow twin | no (0.5%) | 1.44 | +0.0787 | +0.0243 [+0.0208, +0.0279] |

- **All four twin-written sets have a positive residual**, the two narrow-twin sets the highest (+0.024 and +0.025), the three broad-twin sets a little lower (+0.012 to +0.017).
- **The base model's own sets are at or below the line:** S2-800 about 0, S2-odd −0.022, S2-terse −0.072.
- So the residual separates "a twin wrote this" from "the base model wrote this". It does not put the sets with misaligned students above the sets with clean students.
- **A caution about C.** One straight line over sets whose losses run from 0.4 to 3.6 is a crude tool. If the true relation between grad and loss is curved, low-loss sets such as the narrow twins' can get positive residuals for that reason alone. Part B does not have this weakness for the pair P2x / P2x-narrow, and it gives the same answer there (no added value).

## 5. A fault found in the saved files, and its check

While reading `exp5/scores/<set>.jsonl` for this addendum I found that the columns `row`, `id`, `draw`, `n_tokens`, `judge_alignment` and `judge_coherence` of **every** file held the values of the set P2x-narrow. This was a programming error of mine in `exp5/src/e5_analyze.py` (a loop variable was reused when the files were written). I checked the consequences before running A, B and C:

- **The score columns were right.** `loss` and `cos_gradM_lm` in all eight files are equal, to the last digit, to the raw values in `exp5/scores/raw_<set>.pt`, and the raw files' row ids match the training sets.
- **No number of `exp5/RESULTS.md` was affected.** That report was computed from the data in memory, not from these files. After correcting the error I ran the analysis again: `exp5/results/metrics.json` is identical, byte for byte.
- **The files are now corrected.** All score columns are unchanged; the six faulty columns now equal the training sets (checked row by row). Anyone who copied the files before 2026-10-11 23:50 should copy them again.
- One informal number in my own notes used the faulty `n_tokens` column; it never entered a report.

## 6. Deviations and notes

1. **No deviation from the plan** in A, B and C. scikit-learn is not installed, so the logistic regression is my own implementation of the same objective (plan, Part B, item 4). Its check on made-up data passed (true coefficients 0.3, 1.5, −0.7; estimated 0.25, 1.46, -0.69).
2. **Matching order.** The plan fixed one random order of the P2-clean rows. The number of pairs depends a little on this order; the plan's order was used once and not tuned.
3. **Session.** This addendum ran in a new copy of the session on login node `25a-lgn02` (the editor had reconnected). The lock `exp5/.lock` was free, was held during the analysis and is removed. No job was submitted; nothing else was started.
4. **GPU-hours: 0. API cost: $0.**
