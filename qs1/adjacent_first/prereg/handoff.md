# Handoff (v2): QS1-F1, are adjacent topics reached first? (zero GPU)

**For:** a coding agent on Tony's cluster. The QS1 / QS1-E agent is best placed, because it wrote the files this reads.
**Root:** `/work/u4161854/loracle_em_exp1`. **New directory:** `qs1/adjacent_first/`. QS1-E set the precedent: a QS1 follow-up lives under `qs1/`.
**From:** the research lead (Claude), 19 Oct 2026. The question is Tony's. v2 folds in one round of independent review, and every finding was adopted (§11).
**Budget:** 0 GPU-h. CPU only, a few minutes of compute. No training, no sampling, no model loading.
**Priority:** any time. It uses no GPU, so it does not compete with FB2, FK1 or MT1.
**Read-only:** everything under `qs1/` outside `qs1/adjacent_first/`; `es1/stats/` and `es1/results/` (for S4's comparison).
**Writes:** only to `qs1/adjacent_first/`, except two appends:
- lines in the root `RUNLOG.md`, each prefixed `QS1-F1`;
- one line in the QS1 section of the root `CLAUDE.md`, pointing here.

You start with no context. Everything you need is here, in `qs1/PREREG.md` and in `qs1/RESULTS.md`. If something on disk contradicts this handoff, stop and report it; do not guess.

---

## 0. Why, in plain words

**Tony's hypothesis.** When a model is trained on one topic, topics entangled with it inside the model are reached by generalization more readily, and sooner, than distant ones.

**What QS1 found, and why it may not settle the question.**
- QS1 trained a meaningless tag, `[QX-17]`, onto the start of the base model's own answers on one topic. It then measured, with no judge, how readily the model starts answers with the tag on 22 other topics.
- **PLAIN** students saw no contrast data. The tag went everywhere: median P 0.999 at the final step.
- **Closeness grading was significant but tiny:** 0.014 nats per SD of E (t 4.5 with E_base_late, 4.7 with E_grad), against shifts of ≥ 15 nats. The lead read this as "closeness barely matters when nothing pushes back".
- **That number was taken near a ceiling.**
  - It was measured at each student's **half-learned step k** (PLAIN: steps 7–8).
  - The outcome was ΔL_gate, the change in the log-probability of the tag's first token `[` (M₁). A log-probability cannot exceed 0.
  - QS1 itself found the gate saturates within about 10 steps. That is why COST's learning-speed half was "badly placed".
- **A back-of-envelope from the reported t and β** puts the residual SD after fixed effects at about 0.02 nats (0.018–0.029).
  - Near the ceiling, the base term −log p₀ is a target effect. So this residual lives entirely in the headroom h = −log p.
  - Whether 0.02 nats is small depends on the headroom at k. A change dh moves the log-odds by about −dh/h, so with h ≈ 0.01, 0.02 nats of headroom is about 2 nats of log-odds.
  - The k-step cells may therefore be nearly identical because the change really is uniform, or because they are pressed against the ceiling. This follow-up tells the two apart.

**Two ways to look under the ceiling, both from saved outputs:**
1. **Earlier.** QS1 saved a full evaluation of every student at **step 5**, before any PLAIN student was half-learned.
   - Early in training, to first order, the change on a target is set by how much its gradient overlaps the source's (Awasthi et al.'s tangent overlap).
   - Under AdamW's sign-like first steps, the matching kernel is E_adam's rather than E_grad's. Both are reported.
   - Warm-up is identical for every source at step 5, so it biases no comparison.
2. **A measure with no ceiling.** The **log-odds** of the first tag token, logit p = log p − log(1 − p), keeps growing as p → 1. Where p is small it equals log p, so it changes nothing there.

**A third, non-behavioural check.** PLAIN's **activation-space leak** at the final step, from the per-set mean updates QS1-E already used. It has no ceiling either.

**What "reached first" means here.** A target topic takes up the tag faster from a source it is close to than from a source it is far from.
- QS1's two-way fixed-effects model compares each target across the four sources.
- So a topic that takes up the tag easily from every source, or that is close to every source, does not count as "close".
- The size statistic follows the same rule (§3).

**Status.** This is a pre-registered follow-up designed after QS1's results. RESULTS must say so.

---

## 1. Gate 0: inputs and checks (stop and report if any fails)

**G0.1 Files.** Locate, and record the paths of:
- the base model's per-prompt log p(M₁ | x) on all 696 evaluation prompts (Job 1, and the step-0 check);
- each student's per-prompt, per-token log-probs of M at every saved step (QS1 handoff §6.3), and the per-cell tables;
- QS1's stored closeness matrices: E_base_late, E_base_early, E_grad, E_text, and E_adam and E_out if they exist;
- the per-prompt base residuals at the last template token, at the 6 hooked layers (Job 1; QS1-E used them), for recomputing E_base in the bootstrap;
- QS1's analysis code for §8.2–8.3 (fixed effects, Freedman–Lane);
- the final-step per-set means of (h_student − h_base) at the 6 hooked layers (QS1 Job 3; QS1-E read them), and QS1-E's leak-share code;
- the training logs, with the per-step learning rate.

**Expected full evaluations.** Assert that each one exists. One evaluation may serve two roles: for example, CONTRAST's k = 10 is also the fixed step 10.

| student | seed | half-learned step k | steps with a full evaluation |
|---|---|---|---|
| PLAIN: medical, driving, finance, legal | 0 | 8, 8, 7, 8 | 5, k − 1, k, 10, 20, 40, final (≈ 57) |
| PLAIN(medical) | 1 | 8 | the same |
| CONTRAST: medical, driving, finance, legal | 0 | 11, 10, 10, 10 | 5, k − 1, k, 10, 20, 40, 80, final (≈ 113) |
| NULL(medical) | 0 | none (final, flagged) | 5, 10, 20, 40, final |

Also assert: 4 sources × 22 targets = 88 cells per arm and step, with the same target sets and prompt ids as QS1.

**G0.2 Reproduce QS1's SHAPE at k, then compute the yardsticks.**
1. With QS1's own code and the saved per-prompt files, reproduce:
   - PLAIN: β ≈ 0.014 nats per SD (within ±0.002; record which measure the reported 0.014 refers to), and t 4.5 for E_base_late and 4.7 for E_grad (within ±0.1);
   - CONTRAST: t 6.5 and 7.3 (within ±0.1).
   - If either misses, stop and report. Nothing below is readable until the pipeline reproduces.
2. **Before any step-5 statistic,** compute Ĝ (as defined in §3, on ΔL_gate, for both primaries) for PLAIN and for CONTRAST, at k and at the final step. Write them into PREREG.
   - If CONTRAST's Ĝ is below 0.10 at both steps for both measures, PREREG must list ADJACENT-FIRST as a bar that can hardly fire.
   - The bar does not move.

**G0.3 Precision: how log p was stored, and how it was computed.**
- Record how log p(M₁ | x) is stored: binary dtype, or the number of decimals in a CSV.
- Record, from QS1's scoring code, the dtype of the logits and of the log-softmax. A float32 file can hold values computed in bf16, which near 0 fall on a grid of about 0.1 nat.
- On the in-topic prompts at the final step (the most saturated), report the share of exact zeros and the smallest positive h. Check whether near-zero values sit on a grid.
- Then set the resolution limit **h_min**:
  - if near-zero values sit on a grid: 10 × the grid step;
  - otherwise, for float32 or float64 storage: 1e-5 (the practical limit of an fp32 log-softmax near 0);
  - otherwise: 10 × the stored quantum (for example, 1e-5 for 6 decimals).

**G0.4 Learning rate.**
- Tabulate the realised learning rate at steps 1–10, and the cumulative learning rate (in units of the peak) at each evaluated step.
- Expected: the first step has learning rate 0, then a 5-step linear warm-up. So step 5 has had about 2 peak-units, against about 4–5 at k.
- If the per-step learning rate was not logged, reconstruct it from the scheduler's settings. That is not a Gate 0 failure.

**G0.5 What has already been seen.**
- QS1's handoff asked `qs1/RESULTS.md` to report SHAPE at k, at k − 1 and at the final step, on ΔL_gate.
- Quote any such numbers verbatim into PREREG before computing anything new. They are not new tests.
- Also list and quote any QS1 or QS1-E output you have seen that shows step-5 target values, or PLAIN's leak against E.
- The lead has seen only the step-k numbers.

**Then write `qs1/adjacent_first/PREREG.md`:** this handoff verbatim, the Gate 0 findings (including G0.2's yardsticks), and your own predictions with probabilities (§7). **Write it before computing any step-5, log-odds or leak statistic.**

---

## 2. Measures

**Per prompt x, student S, saved step τ.** Let lp = log p_τ(M₁ | x), the student's log-probability of the tag's first token at the answer start. Let lp₀ be the base model's.

| measure | definition | role |
|---|---|---|
| headroom h(x, τ) | max(−lp, h_min) | distance to the ceiling, in nats |
| Δlogit(x, τ) | logit_τ − logit₀, with logit = −h − log(−expm1(−h)) | **primary outcome** |
| ΔL_gate(x, τ) | lp − lp₀ | QS1's outcome, reported beside it |

- **Capped prompts.** A prompt with −lp < h_min is counted as capped. Report the capped share per student and step.
- **Per cell** (source s → target t): the mean over t's 24 questions. The **cell SE** is the bootstrap SE of that mean.
  - Targets are QS1's: the 23 topics minus the source, so 22 per source and **88 cells** per arm.
  - The everyday set is reported, never a target.
- **Closeness E:** QS1's stored matrices, z-scored over the 88 cells as in QS1.
  - Primaries: **E_base_late** and **E_grad**.
  - Others: E_base_early, E_text, E_adam, E_out.
- **FE-residualised closeness E⊥** = E_z minus its source mean and its target mean over the 88 cells, plus the grand mean. β is identified from E⊥ alone.
- **Bootstrap** (2,000 replicates, seed 0).
  - Resample each question set's 24 questions, and the 48 everyday questions, with replacement, **once per replicate**.
  - Use the same indices for every source, step and outcome (paired).
  - In each replicate, recompute E_base_late and E_base_early from the per-prompt base states with the resampled questions: the centring, the z-scoring, E⊥, and everything downstream.
  - E_grad, E_text, E_adam and E_out are held fixed, because no per-prompt values were saved.
- **Labels use the seed-0 students only.** Seed 1 enters only the seed check, as in QS1.

---

## 3. Primary analysis: PLAIN at step 5

**Data.** The four seed-0 PLAIN students at step 5; outcome Δlogit; 88 cells. All four students are at the same step and the same point in the learning-rate schedule.

**Model** (QS1 §8.2): y(s→t) = a_s + b_t + β·E(s, t) + ε. Fit it once with E_base_late and once with E_grad.

**Test** (QS1's Freedman–Lane, unchanged):
- reduced model y ~ a_s + b_t;
- permute its residuals **within each source**, add them back to the fitted values, and refit the full model;
- statistic = the OLS t of β;
- N = 20,000, seed 0; p = (1 + #exceedances) ÷ (N + 1), one-sided;
- Holm over the two measures.
- Vectorise the permutation loops: one design matrix, with the refits batched.

**GRADED** means t > 0 at Holm p ≤ 0.01. **REVERSED** uses the same test with t < 0, as a separate one-sided Holm family.

**Size: how much further along are the near topics?**
- m = the mean of y over the 88 cells.
- ΔE⊥_terc = the mean over sources of [mean E⊥ over the source's 7 targets with the highest E⊥ − mean E⊥ over its 7 lowest], in units of E's SD, using the same E.
- **Ĝ = β · ΔE⊥_terc ÷ m.** This is the model's estimate of how far ahead a source's nearest third of targets is, against its farthest third, as a share of the mean shift. It counts only closeness specific to the source–target pair.
- **Ĝ's 95% CI** is the wider of:
  - (i) the paired bootstrap's percentile interval;
  - (ii) (β ± t₀.₉₇₅,₆₁ · SE_OLS(β)) · ΔE⊥_terc ÷ m.
- Also report:
  - β in nats per SD;
  - its standardised size (β ÷ the SD of the FE-only residuals);
  - the partial R²;
  - Ĝ computed with raw-E terciles instead of E⊥;
  - the raw tercile gap, with no fixed effects.

**Labels**, checked in this order:

| label | rule |
|---|---|
| **NOT EVALUABLE** | Gate 0 failed; or m < 5 × the median cell SE (too little movement to read); or more than 10% of the step-5 target prompts are capped (a ceiling already at step 5) |
| **ADJACENT-FIRST** | a primary measure is GRADED and its Ĝ ≥ 0.10 |
| **TILT** | a primary measure is GRADED, and every GRADED measure's Ĝ CI upper bound is below 0.10 |
| **GRADED-UNSIZED** | a primary measure is GRADED, but neither ADJACENT-FIRST nor TILT applies |
| **REVERSED** | neither measure is GRADED, and one shows t < 0 at Holm p ≤ 0.01 |
| **UNIFORM** | otherwise, if both measures' Ĝ CI upper bounds are below 0.10 |
| **UNRESOLVED** | otherwise |

**Why the bar is 0.10.**
- It means a source's nearest third of topics is at least 10% further along than its farthest third.
- It is not yet anchored to data. QS1's PLAIN value at k (Ĝ ≈ 0.001) is the ceiling-compressed number in dispute, and CONTRAST's own Ĝ has never been computed.
- G0.2 computes CONTRAST's value at k and at the final step before any step-5 statistic. If it is below 0.10, ADJACENT-FIRST is listed as a bar that can hardly fire.

**Seed check.**
- For PLAIN(medical), compute QS1 §8.6's residual profile at step 5 for seeds 0 and 1: r(t) = y(s→t) − b̂_t^(−s), where b̂^(−s) is the target fixed effect from the seed-0 fit of the other three sources at step 5.
- If the Spearman of r₀ against r₁ is below 0.5, RESULTS must say the label is noise-limited.
- The label stands either way.

---

## 4. Secondary analyses (raw numbers; no new labels except HEADROOM)

**S1. Time course.**
- Repeat §3's fit, size and CIs for PLAIN at every saved step: 5, k − 1, k, 10, 20, 40, final.
- Do it for both outcomes, and for E_base_late, E_grad, E_base_early and E_text.
- **Censoring.** A step where more than 10% of PLAIN's target prompts are capped is CENSORED on Δlogit. Report its numbers as lower bounds, and leave it out of H5.
- Report β in nats per SD (the absolute lead) beside Ĝ (the relative lead). Ĝ falls as m grows even when the absolute lead stays constant, so read both.
- Give one table and one figure: Ĝ and β against cumulative learning rate, one line per outcome and measure.

**S2. HEADROOM at k (descriptive).** Take the median headroom h over the PLAIN seed-0 target prompts at k:
- **CEILING** if it is ≤ 0.105 nats (p ≥ 0.9);
- **NEAR-CEILING** if it is ≤ 0.693 (p ≥ 0.5);
- **OPEN** otherwise.

Report the same median, the 10th and 90th percentiles and the capped share at every saved step.

**S3. INSIDE-MODEL at step 5** (QS1 §8.3).
- Run Freedman–Lane with E_text in the reduced model and E added, for each primary measure, with Holm over the two.
- A test counts only if that measure is GRADED in §3.
- Report the post-FE correlation of each measure with E_text.

**S4. Activation-space leak at the final step** (no ceiling; QS1-E's inputs).
- For PLAIN × 4 (seed 0) and PLAIN(medical) seed 1, compute ES1's leak share: leak(s, t) = ⟨d(t), d(s)⟩ ÷ ⟨d(s), d(s)⟩.
  - d(X) is the uncentred per-set mean update.
  - Compute it per late layer and average over 22, 26 and 30.
- Report each source's within-source Spearman of leak(s, ·) with E_base_late(s, ·) and with E_grad(s, ·) over its 22 targets, and their median.
- Also report the §3 fixed-effects fit with leak as the outcome.
- **Same-question caution.**
  - Here d(t) and E_base(s, t) are means over the same 24 questions, which inflates their correlation.
  - ES1's +0.21 (broad twins) and +0.47 (narrow twins) used split halves to avoid this.
  - For a like-for-like comparison, recompute ES1's twin grading from `es1/stats/` (read-only) with full-set E and full-set leak, and score H8 against that.
  - If `es1/stats/` is gone, say so, and compare against the split-half values, flagged.

**S5. CONTRAST time course.**
- S1's table for CONTRAST × 4, adding step 80, with the same censoring rule.
- Does CONTRAST's grading already exist at step 5, or does it build as the contrast acts?

**S6. NULL(medical) at step 5 (descriptive only).**
- NULL is not a drift control for the gate. PLAIN trains answer position 1 on `[` in every row, so NULL's sharpening of the base model's own first tokens has no counterpart in PLAIN.
- Report NULL's step-5 profile, its Spearman with E_base_late and with E_grad, and its near-minus-far tercile gap ÷ its mean.
- Do not subtract it from PLAIN.

**S7. Everyday questions.** For each PLAIN student and step, report the everyday set's mean Δlogit as a share of the mean target Δlogit. Is the different prompt type reached later?

**S8. Target-level differences** (not tested by §3).
- Report SD(b̂_t) ÷ m at step 5.
- Report the Spearman of b̂_t with each target's mean E over the four sources (how central the target is).

**S9. Multiplicative check.**
- At step 5, if every cell's y is positive, also fit log y = a_s + b_t + β_log·E + ε. Report β_log · ΔE⊥_terc, which is about the relative near–far gap.
- Sources learn at different speeds (finance's k is 7). A lead that scales with each source's own shift fits this form better than the additive one.

---

## 5. Look at the data (required)

1. **Trajectories.** For PLAIN(medical) and PLAIN(finance), plot one line per target showing Δlogit over the saved steps, coloured by E_base_late rank, with the source's own in-topic curve. Save PNGs with matplotlib (the cluster has no SVG renderer).
2. **Headroom histograms** of per-prompt h at steps 5 and k, for PLAIN's targets.
3. **Leaders and laggards.** Per source, list the 3 targets furthest ahead and the 3 furthest behind at step 5 (fixed-effects residuals), each with its E_base_late, E_grad and E_text rank.
4. **In plain words:**
   - Do the near targets lead at step 5?
   - Does one target or one cluster drive it? Give the leave-one-target-out range of Ĝ.
   - Has the lead closed by k, and by the final step, on each outcome, in absolute and in relative terms?

---

## 6. Exploratory (mark it as outside the pre-registration)

- **E_adam at step 5.** It is the first-order kernel of AdamW's sign-like early steps, so it may be the best predictor this early.
- A leave-one-cluster-out β at step 5, using 19a's clusters.
- The full tag's ΔL beside the gate's, at step 5.
- Anything surprising.

---

## 7. Predictions (the lead's, written before the run)

| # | prediction | P |
|---|---|---|
| H1 | HEADROOM at k = CEILING | 0.60 |
| H2 | At step 5, the median headroom on PLAIN's targets is ≥ 1 nat | 0.50 |
| H3 | Primary label: ADJACENT-FIRST / TILT / GRADED-UNSIZED / UNIFORM / UNRESOLVED / REVERSED / NOT EVALUABLE | 0.25 / 0.28 / 0.15 / 0.12 / 0.08 / 0.03 / 0.09 |
| H4 | On ΔL_gate, Ĝ (E_base_late) falls at least 5× from step 5 to k | 0.70 |
| H5 | On log-odds, β (E_base_late, nats per SD) at k is at least its step-5 value: the absolute lead does not shrink (only if k is not CENSORED) | 0.55 |
| H6 | At step 5, E_grad's t is at least E_base_late's | 0.55 |
| H7 | INSIDE-MODEL fires at step 5 | 0.25 |
| H8 | PLAIN's final-step leak grading (median within-source Spearman with E_base_late, full sets) is at least ES1's broad twins' grading computed the same way | 0.50 |
| H9 | The step-5 seed check is ≥ 0.5 | 0.55 |
| H10 | CONTRAST's Ĝ at step 5 exceeds PLAIN's | 0.65 |
| H11 | At k (§8): CEILING-ARTEFACT / UNIFORM-AT-K / UNCLEAR | 0.30 / 0.25 / 0.45 |

Add your own predictions, with probabilities, to PREREG before computing anything new.

---

## 8. What the outcomes would mean

**The primary label (step 5):**
- **ADJACENT-FIRST, with log-odds β(k) ≥ β(5).**
  - Near topics stay ahead in logit terms.
  - QS1's "closeness barely matters" was a ceiling effect. The tag still goes everywhere.
  - Entanglement then shapes where unconstrained generalization goes, not only where a narrow model's mask leaks.
- **ADJACENT-FIRST, with log-odds β(k) < β(5).** Near topics are reached first, and far ones catch up in absolute terms. Entanglement sets the order; the shared change sets the end point.
- **TILT.** Neighbours get a real head start, confidently under 10% of the shift. The shared change dominates from the first steps.
- **GRADED-UNSIZED.** Neighbours lead; this run cannot say by how much.
- **UNIFORM.** No source-specific grading of 10% or more at step 5. This label does not test target-level differences (S8 reports them), so it does not show that the change is topic-blind.
- **REVERSED.** Far topics are ahead. Check style and the E measures before reading anything into it.

**At k (descriptive):**
- **CEILING-ARTEFACT:** HEADROOM = CEILING, and log-odds at k is GRADED with Ĝ ≥ 0.10.
- **UNIFORM-AT-K:** log-odds Ĝ at k has a CI upper bound below 0.10.
- **UNCLEAR:** otherwise.
- **Caution.** Near the ceiling, graded log-odds measure the margin over competing first tokens. Training can also produce that by suppressing the competitors differently per source. So CEILING-ARTEFACT without a step-5 ADJACENT-FIRST reads "a graded margin at the ceiling", not "reached first".

---

## 9. Stop rules, compliance, disk

**Stop and report** if:
- Gate 0 fails;
- any assertion fails;
- the design cannot be run as written.

Do not improvise.

**Compliance.**
- No training, no sampling and no model loading. Only saved numeric outputs are read.
- No text is generated or quoted.
- 35 of the 48 everyday questions are verbatim Turner/Soligo questions. Only their existing scores are used, which is evaluation.
- `RESULTS.md` holds numbers, tables and figures only.

**Disk and compute.**
- Check `/work`'s free space first.
- Keep `qs1/adjacent_first/` under 50 MB. Read the large files in place; copy nothing.
- Run on the login node, as QS1-E's CPU parts did, with `nice` and at most 4 threads. If that is not allowed, stop and ask Tony.

---

## 10. RESULTS.md and the final message

**`qs1/adjacent_first/RESULTS.md`, in this order:**
1. **Summary:**
   - the primary label, with raw numbers: β, t, Holm p, Ĝ with its CI, m, the step-5 headroom and the capped share;
   - G0.2's yardsticks;
   - HEADROOM at k, and the at-k reading;
   - the seed check;
   - a predictions-vs-outcomes table.
2. **Gate 0:** paths, the reproduction, the yardsticks, precision, the learning-rate table, and the quotes from `qs1/RESULTS.md`.
3. **Primary** (§3).
4. **Secondary**, S1–S9, with S1's time-course table and figure.
5. **Look at the data** (§5).
6. **Exploratory** (§6).
7. **Self-doubt.** Include at least:
   - step 5 sits at the end of warm-up, and Adam's early steps are sign-like;
   - E is computed from the evaluation sets, not the training questions (QS1 §C);
   - one tag, four sources and about 6 topic clusters; the fixed effects leave out closeness shared with every source;
   - the bootstrap covers questions, and E_base's own noise, but not training noise (there is one seed replicate); E_grad, E_text, E_adam and E_out are held fixed;
   - near p = 1, log-odds also reflects the margin over competing first tokens, and the target fixed effects absorb only its per-topic part;
   - only step 5 is labelled; the other steps are descriptive.

**Final message to Tony** (at most 10 lines, neutral, ready to paste to the lead):
1. Gate 0: reproduced or not; precision and h_min; capped shares; the yardsticks.
2. The primary label, with β, t, Holm p and Ĝ [CI] for both measures.
3. HEADROOM at k, and the at-k reading.
4. The time course: Ĝ and β at step 5, k and final, for both outcomes (E_base_late).
5. INSIDE-MODEL at step 5; the leak grading against ES1's (S4); the seed check.
6. The outcomes of H1–H11.
7. Deviations; CPU time; the path to `RESULTS.md`.

---

## 11. Review record (v1 → v2)

One independent review round. Its verdict was GO WITH FIXES, and every finding was adopted.

**Blocking:**
- **B1.** The v1 fallback switched to ΔL_gate when log-odds were capped, but ΔL_gate is even more saturated there. Now a capped step 5 is NOT EVALUABLE, and capped later steps are CENSORED.
- **B2.** Gate 0 checked only how log p was stored. It now also checks how log p was computed, and tests for a bf16 grid.

**Major:**
- **M1.** Ĝ's CI is now the wider of the bootstrap (with E_base recomputed per replicate) and the OLS interval. TILT is now decided on the CI, and GRADED-UNSIZED is added.
- **M2.** The 0.10 bar's yardsticks (PLAIN and CONTRAST at k and final) are now computed in Gate 0, before any step-5 statistic.
- **M3.** Added the at-k reading, with the margin caution. "Catch up" now needs the absolute β to fall.
- **M4.** NULL is no longer subtracted. It is not a drift control for the gate.
- **M5.** S4 now compares like for like with ES1 (full sets on both sides).
- **M6.**
  - Ĝ now uses FE-residualised closeness.
  - The UNIFORM reading no longer claims the change is topic-blind.
  - S8 reports the target-level differences.

**Minor:**
- §0's arithmetic is now read against the headroom.
- E_adam is named as Adam's first-order kernel.
- Added the multiplicative check (S9).
- One evaluation may serve two roles.
- A learning rate that was not logged may be reconstructed from the scheduler.
- The write scope is spelled out.
- G0.5 now covers QS1-E's outputs too.
- GRADED and REVERSED are separate Holm families.
- The everyday set is resampled as well.
- The §8 wording is corrected: the tag still goes everywhere.
- The permutation loops are vectorised.
