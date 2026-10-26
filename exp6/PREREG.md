# PREREG — Experiment 6: can tricks on the gradient isolate the persona?

Written on 2026-10-12, before any GPU job of Experiment 6. Not changed afterwards (the file is made read-only).
Part A is the text of the handoff, copied as given. Part B lists the choices the handoff left open.

## Part A — from the handoff

### Goal
In Experiment 5, a gradient score separated persona-carrying data from controls. The base model's loss did just as
well. The small amount the gradient added beyond loss followed "written by a fine-tuned twin", not "will make the
student misaligned". Experiment 6 tries two tricks to remove those confounds, plus one cheap reference to beat.
1. **Paired-teacher direction (trick 1).** P2, P2-narrow, P2x and P2x-narrow answer the same 800 questions. Within each
   domain, the difference between the broad twin's and the narrow twin's mean gradients is "what the broad answers
   teach beyond the narrow ones". We learn this direction in one domain and test it on the other.
2. **Cleaned yardstick (trick 2).** Remove from ∇M_lm its "fine-tuned-twin authorship" part and its "style" parts.
3. **Short-run probe (the reference to beat).** Train 20 real optimizer steps on each set, and measure how much M_lm moves.

**The decisive test** is broad against narrow in extreme sports, P2x against P2x-narrow, using directions built
without those two sets.

### Definitions
- **ĝ_i** is the per-example gradient at the base model, on the selected matrices (o_proj and down_proj, layers
  10–30, as in Experiment 5), scaled to unit length.
- **m(S, Q)** is the mean of ĝ_i over the rows of set S whose questions are in the question list Q.
- **Question lists:** Q800 = P2's 800 questions (P2, P2-narrow, P2x, P2x-narrow and S2-800 answer them);
  Qclean = P2-clean's 800 questions (S2-terse and S2-odd answer them); Q469 = the questions that S2-terse and S2-800 share.

| Name | Definition | What it captures |
|---|---|---|
| P_fin | m(P2, Q800) − m(P2-narrow, Q800) | Finance broad answers beyond narrow ones |
| P_fin_clean | m(P2-clean, Q469) − m(P2-narrow, Q469) | The same, using only the filtered, visibly aligned broad answers |
| P_x | m(P2x, Q800) − m(P2x-narrow, Q800) | Extreme-sports broad answers beyond narrow ones |
| T_style | m(S2-terse, Q469) − m(S2-800, Q469) | Terse vs long, base model |
| O | m(S2-odd, Qclean) − m(S2-terse, Qclean) | Odd voice vs plain, base model |
| A_fin | m(P2-narrow, Q800) − m(S2-800, Q800) | Finance narrow twin vs base ("authorship") |
| A_x | m(P2x-narrow, Q800) − m(S2-800, Q800) | Extreme-sports narrow twin vs base ("authorship") |
| g_M | ∇M_lm at the base model | Towards misaligned answers (Experiment 5) |

- **"X ⊥ {Y, Z}"** means X with its components along Y and Z removed, by Gram–Schmidt in the full space. Report the
  share of X's length that remains.
- **Leak rule:** a direction used to test a pair of sets must never be built from either set in that pair.

### The scores (each is the cosine between ĝ_i and a cleaned direction)

| Score | Direction | Used to test |
|---|---|---|
| **s_Pfin** (trick 1) | P_fin ⊥ {T_style, O} | Extreme sports: P2x vs P2x-narrow |
| **s_Pfin_clean** (trick 1, filtered) | P_fin_clean ⊥ {T_style, O} | Extreme sports: P2x vs P2x-narrow |
| **s_Mext** (trick 2) | g_M ⊥ {A_fin, T_style, O} | Extreme sports: P2x vs P2x-narrow |
| s_Px (trick 1, secondary) | P_x ⊥ {T_style} | Finance: P2-clean vs S2-odd |
| s_Mfin (trick 2, secondary) | g_M ⊥ {A_x, T_style} | Finance: P2-clean vs S2-odd |

**Reference scores:** the uncleaned cos(ĝ_i, P_fin), cos(ĝ_i, P_x) and cos(ĝ_i, g_M), and each row's loss.

### Short-run probe
For each of the 8 sets, train 20 optimizer steps with the Experiment 3 recipe (seed 0; warm-up 5 steps; effective
batch 16). Measure M_lm of the resulting adapter, and report ΔM_lm = M_lm(after 20 steps) − M_lm(base model).

### The decisive test: extreme sports, P2x vs P2x-narrow
Run this separately for **s_Pfin** (trick 1), **s_Pfin_clean** (trick 1, built from filtered answers) and **s_Mext** (trick 2).

**(a) Added prediction beyond loss.** Use 5-fold cross-validated logistic regression to separate P2x rows from
P2x-narrow rows, first with loss alone, then with loss + the score. The test passes if the AUROC gain is ≥ 0.02 and
its 95% bootstrap interval (10,000 resamples over rows) excludes 0.

**(b) Loss-matched pairs.** Match each P2x row to its nearest unused P2x-narrow row by loss. Keep a pair only if the
losses differ by at most 0.05. The test passes if the mean paired difference in the score (P2x − P2x-narrow) is > 0
with a 95% bootstrap interval over pairs that excludes 0. If there are fewer than 50 pairs, mark (b) as
underpowered; it then neither passes nor fails.

### Labels
- **TRICK WORKS** for a score: (a) passes, and (b) passes or is underpowered.
- **Overall TRICK WORKS:** the label holds for s_Pfin or for s_Mext. State which.
- **NO TRICK WORKS:** (a) fails for both scores.
- **PARTIAL:** anything else.
- **Reported alongside, without changing the label:** whether s_Pfin_clean also gets TRICK WORKS. If it does, the
  direction learned from filtered, visibly aligned answers works too, so the gradient sees more than visible
  misalignment. If only s_Pfin works, it may simply have learned visible misalignment. P2 is unfiltered.

### Secondary tests (finance; they do not change the overall label)
- **P2-clean vs S2-odd,** with s_Px and s_Mfin: the same (a) and (b). This comparison still contains "twin vs base"
  authorship, so passing it alone shows nothing about persona.
- **P2 vs P2-narrow:** the two sets' losses barely overlap. Report each score's mean residual after a linear fit on
  loss over all 6,400 rows, with intervals. This is descriptive only.

### Descriptive outputs
- **Cross-domain agreement:** cos(P_fin, P_x) before and after cleaning, next to the cosines of each persona
  direction with T_style, O, A_fin and A_x.
- **Short-run probe:** ΔM_lm for each set. Check whether it orders P2x > P2x-narrow, P2 > P2-narrow,
  P2-clean > S2-odd and P2-clean > S2-terse.
- **AUROC without loss:** for every score, the AUROC on its own for the decisive pair.
- **Example rows:** 5 random rows from the top 50 and 5 from the bottom 50 of s_Pfin among the P2x and P2x-narrow
  rows, with their text and loss.

### Optional Stage E6-4 — the LoRAcle reads the paired difference (descriptive only)
Run only if at least 0.5 GPU-hours remain under the cap. Build three "updates", each a mean of unit gradients over
all 280 matrices (7 matrix types × 40 layers), on Q800 unless stated: m(P2x) − m(P2x-narrow); m(S2-odd) − m(S2-terse)
on Qclean; m(P2x-narrow) alone. Turn each into reader tokens: a rank-32 SVD per matrix, then the reader's own
extractor, exactly as for adapters. Ask both readers Q1, Q3b, Q5, Q6 and Q4 (16 samples), with noise-token controls.
The blind judge scores the Q4 descriptions.

## Part B — choices fixed here, before any GPU job

1. **Gradients.** As in Experiments 4 and 5: base model in 32-bit floats; ℓ_i is the mean log-probability per token of
   the answer in the training format (no system prompt; end-of-turn token included when the answer was finished);
   the gradient is taken for the 42 selected matrices and divided by its length over all 42 together.
2. **Means.** Eleven means are accumulated in 32-bit floats: P2, P2-narrow, P2x, P2x-narrow and S2-800 on Q800 (800
   rows each); P2-narrow, S2-800, P2-clean and S2-terse on Q469 (469 rows each); S2-terse and S2-odd on Qclean (800
   rows each). g_M is recomputed at the base model from the 27 pairs of Experiment 5.
3. **Storage.** The means and all directions are stored in 16-bit floats of the type "bfloat16". It has the number
   range of 32-bit floats; ordinary half precision would lose the entries, which are about 0.00001 in size. The
   rounding error of each entry is at most 0.4% of that entry and is unrelated between entries, so it changes
   cosines by far less than 0.001. All lengths and inner products are computed in 64-bit sums from the stored
   vectors, and the job reports the check (lengths and cosines of the stored directions against the algebra).
4. **Directions and cleaning.** Every direction is a combination of the eleven means and g_M. "X ⊥ {Y, Z}" is X minus
   its projection on the space spanned by Y and Z; this is what Gram–Schmidt gives, and it does not depend on the
   order of Y and Z. The coefficients come from the inner products of the twelve vectors. Thirteen directions are
   saved to `exp6/directions/`: the eight of the table and the five cleaned ones.
5. **Scores.** cos(ĝ_i, d) = ⟨ĝ_i, d⟩ / ‖d‖ with the stored direction d and the 32-bit unit gradient. Each row's loss
   is −ℓ_i from the same pass. The saved files are read back and their row ids, sampling numbers, lengths and
   questions are compared with the training sets before any analysis.
6. **Rows that helped build a direction.** A row's own gradient is part of the mean of its set. This matters only for
   descriptive numbers (for example P2 and P2-narrow rows under s_Pfin). The decisive and secondary pairs are never
   scored with a direction built from them (leak rule; checked in the code).
7. **Test (a).** As in the addendum of Experiment 5: my own logistic regression with the objective of scikit-learn's
   default (L2 penalty, C = 1, intercept not penalised), features standardised with the training folds, 5 folds
   stratified by set (seed 20261016), one out-of-fold prediction per row, AUROC from all out-of-fold predictions.
   Gain = AUROC(loss + score) − AUROC(loss alone). Interval: 10,000 resamples of the rows, separately inside each
   set, models not refitted; percentile 2.5% to 97.5%. "Excludes 0" means both ends have the same sign.
8. **Test (b).** As in the addendum of Experiment 5: the P2x rows (or P2-clean rows) are taken in a random order
   (seed 20261016); each takes the nearest unused row of the other set by loss (tie: lower row number); a pair is
   kept if the losses differ by at most 0.05. Interval: 10,000 resamples of the pairs.
9. **Labels.** Checked as written. For one score: TRICK WORKS, or "(a) fails", or PARTIAL (when (a) passes and (b)
   fails). Overall: TRICK WORKS if it holds for s_Pfin or s_Mext; NO TRICK WORKS if (a) fails for both; PARTIAL otherwise.
10. **Residuals (P2 vs P2-narrow, descriptive).** For each score one straight line score = a + b × loss is fitted over
    all 6,400 rows and kept fixed; per set the mean residual with an interval from 10,000 resamples of its rows.
11. **Short-run probe.** The first 20 optimiser steps of the standard run: the same schedule as a full 3-epoch
    training (5 warm-up steps, then the linear decay planned for the whole run), stopped by a callback after step
    20. Seed 0 for the adapter start and the data order, as "seed 0" in Experiment 5; the same start for all eight
    sets. Training in the usual 16-bit format; M_lm of the resulting adapter is measured with the 32-bit base model,
    as in Experiment 5. One run per set, so no interval; only the four orderings are checked.
12. **Optional Stage E6-4, if run.** Decided after all other stages: it is run if at least 0.5 GPU-hours remain under
    the cap and my estimate of its cost (about 0.7 GPU-hours) also fits. Choices: the gradients for all 280 matrices
    are taken with the model in the usual 16-bit training format (bfloat16), because the 32-bit model plus a full
    gradient and a full sum do not fit on one GPU; each matrix's gradient is converted to 32 bits before it is
    scaled and added. The unit length is over all 280 matrices together. A training step moves the weights along
    the gradient of ℓ, so the "update" is the mean unit gradient itself. Per matrix a randomised rank-32 SVD
    (64 probes, 4 power iterations) gives U, S, V; the reader's extractor gets B = U·S and A = Vᵀ with scaling 1,
    as for an adapter. Noise tokens, questions, sampling and the blind Q4 judge are those of Experiment 3 (16 Q4
    samples, 8 for the other questions). Three "updates": broad-minus-narrow (extreme sports), odd-minus-terse,
    narrow-alone. Everything in this stage is descriptive.
13. **Descriptive details.** Example rows are drawn with seed 20261016 from the 1,600 rows of P2x and P2x-narrow
    together. "AUROC without loss" is the plain AUROC of each score for P2x against P2x-narrow.
14. **Smoke tests** (before the first full job of each kind): (i) a small version of the whole chain on 2 of the 21
    layers: 4 rows per set for the means (unit length and finite means checked), the directions, and the scores of
    4 rows; (ii) after the real directions exist, a full-size scoring test on 8 rows, which also compares the loss
    and the cosine with g_M with the values of Experiment 5; (iii) the probe: 2 optimiser steps and an M_lm
    measurement; (iv) the optional stage, if run. Each is recorded in `RUNLOG.md`.
15. **One session, budget, disk.** `exp6/.lock`; jobs are named `lem6-*`; the submit script refuses foreign locks,
    foreign jobs in this folder and duplicate job names. Hard cap 4.0 GPU-hours, checked before every submission.
    The thirteen directions take about 63 GB; the eleven stored means (about 53 GB) are temporary and are deleted
    after the analysis.
16. **Judge.** Used only in the optional stage: local `Qwen/Qwen3-32B`, blind, as before.
