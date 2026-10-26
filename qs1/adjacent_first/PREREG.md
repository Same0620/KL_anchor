# QS1-F1 PREREG: are adjacent topics reached first? (zero GPU)

Pre-registered follow-up to QS1, designed after QS1's results. Plan for `qs1/adjacent_first/`.

**Sources.**
- Part 0 is the handoff verbatim (`prereg/handoff.md`, 25309 characters, sha256 ac867a6f03e7e8a2…), pasted by Tony on 2026-10-19 at about 22:25.
- Part A holds the Gate 0 findings, Part B the choices, and Part C the predictions. All three were written before any step-5, log-odds or leak statistic was computed.
- Part D holds notes added after computing.

---

## Part 0. The handoff (verbatim)

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

---

## Part A. Gate 0 findings (2026-10-19, CPU; `src/af_gate0.py` → `results/gate0.json`)

**Verdict: GATE 0 PASS.** Every file exists, every assertion holds, QS1's SHAPE at k reproduces with QS1's own code, and the yardsticks were computed before any step-5 statistic.

### A1. G0.1 files and assertions

| item | path |
|---|---|
| base lp and residuals | qs1/stats/job1/base.pt (lp [696, 7] fp32; h [696, 6, 5120] fp32 at the last template token, layers 10/14/18/22/26/30) |
| step0 check | qs1/outputs/students/<student>/scores.pt evals['step0'] (and record.json step0_check) |
| student lp | qs1/outputs/students/<student>/scores.pt: evals[label] [696, 7] fp32 (labels step0, s5, k-1, k, s10, s20, s40, s80, final); intopic[step] [24, 7] after every step |
| per cell tables | qs1/results/csv/cells.csv (and per-prompt qs1/results/csv/prompts_<student>.csv, 5 decimals) |
| closeness | qs1/results/closeness/MAIN_{E_base_late, E_base_early, E_grad, E_text, E_adam, E_out}.csv (24 x 24 with everyday48, 6 decimals; E_adam is [s, t]-oriented) |
| qs1 analysis code | qs1/src/qs1_analyze.py (design, shape, inside_model, noise) and qs1/src/qs1stats.py (fe_design, freedman_lane, holm) |
| update setmeans | qs1/outputs/students/<student>/scores.pt dh_setmean [30, 6, 5120] fp32 (sets_order: the 24 main sets first) |
| qs1e leak code | qs1/entanglement_after/src/qse_common.py: leak(), load_updates() |
| lr logs | qs1/outputs/students/<student>/record.json 'lr' (Trainer log_history learning_rate, one per optimiser step) |
| es1 stats | es1/stats/<model>.pt (S4 only) |

**Expected full evaluations** (asserted present; each student's `evals['step0']` equals Job 1's base `lp` exactly, all 7 tokens):

| student | k | steps | labels present |
|---|---|---|---|
| PLAIN.bad_medical.s0 | 8 | 57 | final, k, k-1, s10, s20, s40, s5, step0 |
| PLAIN.reckless_driving.s0 | 8 | 57 | final, k, k-1, s10, s20, s40, s5, step0 |
| PLAIN.risky_financial.s0 | 7 | 57 | final, k, k-1, s10, s20, s40, s5, step0 |
| PLAIN.bad_legal.s0 | 8 | 57 | final, k, k-1, s10, s20, s40, s5, step0 |
| PLAIN.bad_medical.s1 | 8 | 57 | final, k, k-1, s10, s20, s40, s5, step0 |
| CONTRAST.bad_medical.s0 | 11 | 114 | final, k, k-1, s10, s20, s40, s5, s80, step0 |
| CONTRAST.reckless_driving.s0 | 10 | 114 | final, k, k-1, s10, s20, s40, s5, s80, step0 (k = 10 is also the s10 evaluation) |
| CONTRAST.risky_financial.s0 | 10 | 114 | final, k, k-1, s10, s20, s40, s5, s80, step0 (k = 10 is also the s10 evaluation) |
| CONTRAST.bad_legal.s0 | 10 | 114 | final, k, k-1, s10, s20, s40, s5, s80, step0 (k = 10 is also the s10 evaluation) |
| NULL.bad_medical.s0 | 57 | 57 | final, k, k-1, s10, s20, s40, s5, step0 |

**Assertions.**
- Both arms have 4 sources × 22 targets = 88 cells.
- The cell list, the set slices and the prompt ids equal QS1's `qs1_analyze.design` exactly, and my gate cells equal QS1's to 1e-12.
- `results/csv/cells.csv` agrees to 5.0e-06 and `prompts_base.csv` to 5.0e-06, as expected at 5 decimals.

### A2. G0.2 reproduction and yardsticks

**Reproduction** (QS1's `qs1_analyze.design` + `qs1stats.freedman_lane`, at k, ΔL_gate, N = 20,000):

| arm | measure | β (nats / SD) | t | p | Holm p | handoff target | within tolerance |
|---|---|---|---|---|---|---|---|
| PLAIN | E_base_late | +0.0140 | 4.52 | 0.00005 | 0.00010 | β 0.014 ± 0.002, t 4.5 ± 0.1 | yes |
| PLAIN | E_grad | +0.0146 | 4.71 | 0.00005 | 0.00010 | β 0.014 ± 0.002, t 4.7 ± 0.1 | yes |
| CONTRAST | E_base_late | +0.0494 | 6.46 | 0.00005 | 0.00010 | t 6.5 ± 0.1 | yes |
| CONTRAST | E_grad | +0.0534 | 7.28 | 0.00005 | 0.00010 | t 7.3 ± 0.1 | yes |

- The reported 0.014 is E_base_late's β, +0.0140; E_grad's is +0.0146. Both are within ±0.002.
- My vectorised Freedman–Lane, which uses QS1's permutation generator, gives β, t and p identical to QS1's function on these inputs.

**Yardsticks: Ĝ on ΔL_gate** (§3's definition; the CI is the wider of the bootstrap, with E_base recomputed per replicate, and the OLS interval):

| arm | step | measure | Ĝ | 95% CI (source) | β (nats / SD) | t | m (nats) | ΔE⊥_terc |
|---|---|---|---|---|---|---|---|---|
| PLAIN | k | E_base_late | +0.0011 | [+0.0006, +0.0015] (ols) | +0.0140 | 4.52 | 27.71 | 2.095 |
| PLAIN | k | E_grad | +0.0011 | [+0.0006, +0.0016] (ols) | +0.0146 | 4.71 | 27.71 | 2.087 |
| PLAIN | final | E_base_late | +0.0000 | [-0.0000, +0.0000] (ols) | +0.0001 | 1.71 | 27.94 | 2.095 |
| PLAIN | final | E_grad | +0.0000 | [+0.0000, +0.0000] (ols) | +0.0001 | 2.49 | 27.94 | 2.087 |
| CONTRAST | k | E_base_late | +0.0038 | [+0.0026, +0.0049] (ols) | +0.0494 | 6.46 | 27.57 | 2.095 |
| CONTRAST | k | E_grad | +0.0040 | [+0.0029, +0.0052] (ols) | +0.0534 | 7.28 | 27.57 | 2.087 |
| CONTRAST | final | E_base_late | +0.1520 | [+0.1173, +0.1867] (ols) | +1.6331 | 8.77 | 22.51 | 2.095 |
| CONTRAST | final | E_grad | +0.1656 | [+0.1345, +0.1966] (ols) | +1.7857 | 10.66 | 22.51 | 2.087 |

- **Bar check:** CONTRAST's Ĝ is below 0.10 at both steps for both measures: **no**.
  - At the final step it is 0.152 (E_base_late) and 0.166 (E_grad).
  - At k it is 0.0038 and 0.0040.
  - ADJACENT-FIRST is therefore **not** listed as a bar that can hardly fire: a pushed-back arm clears 0.10 by its final step. At k neither arm comes near it.
  - The bar does not move.

**E⊥: the exact FE residual against the handoff's one-pass formula** (E only; PLAIN's 88 cells; choice B1):

| measure | ΔE⊥_terc, exact | ΔE⊥_terc, one pass | r (exact, one pass) | max abs difference | one pass's max ⟨Z, E⊥⟩ | raw-E tercile gap |
|---|---|---|---|---|---|---|
| E_base_late | 2.095 | 2.097 | 0.9999 | 0.044 | 3.29e-01 | 2.113 |
| E_grad | 2.087 | 2.090 | 0.9997 | 0.084 | 4.61e-01 | 2.158 |
| E_base_early | 2.022 | 2.023 | 0.9998 | 0.064 | 5.34e-01 | 2.139 |
| E_text | 1.925 | 1.925 | 0.9999 | 0.046 | 3.54e-01 | 1.997 |
| E_adam | 1.776 | 1.782 | 0.9984 | 0.175 | 5.00e-01 | 1.986 |
| E_out | 1.414 | 1.431 | 0.9933 | 0.262 | 7.01e-01 | 1.834 |

### A3. G0.3 precision

**Storage.**
- scores.pt / base.pt hold lp as torch.float32 tensors (binary); the analysis reads these. The per-prompt CSVs round to 5 decimals and are not used except as a cross-check.

**Computation** (`qs1common.score_prompts`):
- `lg = o.logits[ar[:, None], pos].float()                              # [b, 7, V]`
- `lps = torch.log_softmax(lg, -1)`
- Logits: bf16 (Qwen3-14B in bf16, lm_head not adapted; Job 3 scored inside the Trainer step, under its bf16 autocast); cast to fp32 by .float() before the log-softmax.
- Log-softmax: fp32.

**Check that the logits are bf16.**
- Job 1 saved the base model's top-50 probabilities at the last template token. Their log-ratios, top-1 minus top-k, are logit differences.
- On a 1/16-nat lattice: 1.000 of them. On a 1/8 lattice: 0.995. On a 1/4 lattice: 0.504.
- That is what bf16 logits of magnitude 8–32 give.

**In-topic prompts at the final step** (360 prompts; 15 tagged students):
- Exact zeros: 0.000. Exact zeros over all 696 × 15 final-step prompts: 0.000.
- Smallest positive h: 7.15e-07.
- h quantiles: 10% 1.82e-05, 50% 0.000187, 90% 0.00414.
- **Grid near 0:** 1.000 of the 38 values below 2e-5 are exact multiples of 2⁻²³ = 1.19e-7.
  - Examples: 7.15e-7 = 6 × 2⁻²³, 1.55e-6 = 13 ×, 3.46e-6 = 29 ×.
  - This is the fp32 softmax denominator's spacing just above 1, i.e. the fp32 log-softmax's own resolution.
- **No coarse grid.** 324 of the 337 values below 0.01 are distinct, and they run continuously down from 7e-7. A log-softmax computed in bf16 (~0.1-nat grid) would put them at 0 or at ≥ ~0.06.
- **Log-odds.** Near p = 1, the log-odds partly inherits the bf16 margin lattice: values pile up just below multiples of 1/8 nat (histogram of logit mod 1/8 over 8 bins, h < 0.05: [32, 23, 33, 36, 33, 50, 44, 93]). This rounding is ≤ 0.125 nats of log-odds per prompt.

**h_min = 1e-05** (branch 2: float32 storage of an fp32 log-softmax -> 1e-5 (the handoff's practical limit of an fp32 log-softmax near 0); choice B3).
- The literal first branch (10 × the 2⁻²³ grid) would give 1.19e-06. It is reported as a sensitivity.
- In-topic final capped share: 0.056 at 1e-5; 0.003 at 1.19e-6.

### A4. G0.4 learning rate

The per-step learning rate was logged: `record.json` 'lr' is the Trainer's `learning_rate`, logged before the scheduler step, i.e. the rate used at that step.
Step 1 has rate 0; its in-topic scores equal step 0's exactly for every student checked.

| schedule | steps 1–10, in peak units (peak 2e-5) |
|---|---|
| PLAIN / NULL (57 steps) | 0.000, 0.200, 0.400, 0.600, 0.800, 1.000, 0.981, 0.962, 0.942, 0.923 |
| CONTRAST (114 steps) | 0.000, 0.200, 0.400, 0.600, 0.800, 1.000, 0.991, 0.982, 0.972, 0.963 |

**Cumulative learning rate (peak units) at each evaluated step:**

| student | s5 | k − 1 | k | s10 | s20 | s40 | s80 | final |
|---|---|---|---|---|---|---|---|---|
| PLAIN.bad_medical.s0 | 2.00 | 3.98 (step 7) | 4.94 (step 8) | 6.81 | 14.98 | 25.56 | – | 28.50 (step 57) |
| PLAIN.reckless_driving.s0 | 2.00 | 3.98 (step 7) | 4.94 (step 8) | 6.81 | 14.98 | 25.56 | – | 28.50 (step 57) |
| PLAIN.risky_financial.s0 | 2.00 | 3.00 (step 6) | 3.98 (step 7) | 6.81 | 14.98 | 25.56 | – | 28.50 (step 57) |
| PLAIN.bad_legal.s0 | 2.00 | 3.98 (step 7) | 4.94 (step 8) | 6.81 | 14.98 | 25.56 | – | 28.50 (step 57) |
| PLAIN.bad_medical.s1 | 2.00 | 3.98 (step 7) | 4.94 (step 8) | 6.81 | 14.98 | 25.56 | – | 28.50 (step 57) |
| CONTRAST.bad_medical.s0 | 2.00 | 6.91 (step 10) | 7.86 (step 11) | 6.91 | 16.04 | 31.54 | 51.54 | 57.00 (step 114) |
| CONTRAST.reckless_driving.s0 | 2.00 | 5.94 (step 9) | 6.91 (step 10) | 6.91 | 16.04 | 31.54 | 51.54 | 57.00 (step 114) |
| CONTRAST.risky_financial.s0 | 2.00 | 5.94 (step 9) | 6.91 (step 10) | 6.91 | 16.04 | 31.54 | 51.54 | 57.00 (step 114) |
| CONTRAST.bad_legal.s0 | 2.00 | 5.94 (step 9) | 6.91 (step 10) | 6.91 | 16.04 | 31.54 | 51.54 | 57.00 (step 114) |
| NULL.bad_medical.s0 | 2.00 | – | – | 6.81 | 14.98 | 25.56 | – | 28.50 (step 57) |

As expected, step 5 has had 2.00 peak units for every student. PLAIN's k has had 4.94 (finance, k = 7: 3.98); CONTRAST's 6.91–7.86.

### A5. G0.5 what has already been seen

**Who has seen what.**
- The lead has seen only the step-k numbers.
- I, the agent, had seen the items below before writing Part C, and nothing else at step 5, on log-odds or on the leak.
- Gate 0 itself added the final-step in-topic headroom (A3) and the yardsticks (A2).

**1. QS1's SHAPE at k, k − 1 and final, on ΔL_gate** (`qs1/RESULTS.md` §5, verbatim; these are not new tests):

**PLAIN** — label **GRADED** (decided at the half-learned step)

| step | measure | β (nats / SD of E) | standardised β | t | partial R² | p (one-sided) | Holm p |
|---|---|---|---|---|---|---|---|
| k | E_base_late | +0.0140 | +0.528 | 4.52 | 0.251 | 0.00005 | 0.00010 |
| k | E_grad | +0.0146 | +0.551 | 4.71 | 0.267 | 0.00005 | 0.00010 |
| k | E_base_early | +0.0173 | +0.649 | 6.07 | 0.376 | 0.00005 | – |
| k | E_text | +0.0146 | +0.550 | 4.67 | 0.264 | 0.00005 | – |
| k-1 | E_base_late | +0.0335 | +0.530 | 4.54 | 0.253 | 0.00005 | 0.00010 |
| k-1 | E_grad | +0.0356 | +0.562 | 4.85 | 0.278 | 0.00005 | 0.00010 |
| k-1 | E_base_early | +0.0379 | +0.599 | 5.36 | 0.320 | 0.00005 | – |
| k-1 | E_text | +0.0328 | +0.518 | 4.31 | 0.234 | 0.00010 | – |
| final | E_base_late | +0.0001 | +0.225 | 1.71 | 0.046 | 0.05265 | 0.05265 |
| final | E_grad | +0.0001 | +0.324 | 2.49 | 0.092 | 0.00780 | 0.01560 |
| final | E_base_early | +0.0001 | +0.296 | 2.28 | 0.078 | 0.01550 | – |
| final | E_text | +0.0000 | +0.120 | 0.88 | 0.013 | 0.19549 | – |

Exploratory, at k: robustness rerun on the Exp 15 collection — E_base_late t 1.86, p 0.02795; E_grad t 2.18, p 0.01505; E_text t 3.81, p 0.00075. Leave-one-cluster-out t (E_base_late / E_grad): finance 3.73 / 3.65; sports 4.91 / 4.78; medical 3.17 / 3.41; diy 4.97 / 4.87.

**CONTRAST** — label **GRADED** (decided at the half-learned step)

| step | measure | β (nats / SD of E) | standardised β | t | partial R² | p (one-sided) | Holm p |
|---|---|---|---|---|---|---|---|
| k | E_base_late | +0.0494 | +0.672 | 6.46 | 0.407 | 0.00005 | 0.00010 |
| k | E_grad | +0.0534 | +0.727 | 7.28 | 0.465 | 0.00005 | 0.00010 |
| k | E_base_early | +0.0558 | +0.760 | 8.06 | 0.516 | 0.00005 | – |
| k | E_text | +0.0458 | +0.624 | 5.59 | 0.339 | 0.00005 | – |
| k-1 | E_base_late | +0.0548 | +0.716 | 7.23 | 0.461 | 0.00005 | 0.00010 |
| k-1 | E_grad | +0.0603 | +0.788 | 8.55 | 0.545 | 0.00005 | 0.00010 |
| k-1 | E_base_early | +0.0572 | +0.747 | 7.78 | 0.498 | 0.00005 | – |
| k-1 | E_text | +0.0458 | +0.598 | 5.26 | 0.312 | 0.00005 | – |
| final | E_base_late | +1.6331 | +0.787 | 8.77 | 0.558 | 0.00005 | 0.00010 |
| final | E_grad | +1.7857 | +0.860 | 10.66 | 0.651 | 0.00005 | 0.00010 |
| final | E_base_early | +1.8627 | +0.897 | 12.50 | 0.719 | 0.00005 | – |
| final | E_text | +1.6276 | +0.784 | 8.39 | 0.536 | 0.00005 | – |

Exploratory, at k: robustness rerun on the Exp 15 collection — E_base_late t 1.81, p 0.03155; E_grad t 2.57, p 0.00435; E_text t 5.39, p 0.00005. Leave-one-cluster-out t (E_base_late / E_grad): finance 5.62 / 6.27; sports 6.71 / 7.07; medical 5.25 / 5.98; diy 6.26 / 7.18.

In PLAIN the graded component is small in nats (0.014 per SD of E at k; the FE-only residual SD is ≈ 0.027 nats) because every topic is already near P ≈ 0.5 at k; at the final step everything saturates (β ≈ 1e-4) and the fit is no longer significant. In CONTRAST the gradient grows with training (β 0.05 at k, 1.6–1.8 nats per SD at the final step). E_base_early is the strongest single predictor in both arms (t 6.07 / 8.06 at k), reported, not labelled.

Label **INSIDE-MODEL**. Reduced model: FE + β_text·E_text; full model adds E; Freedman–Lane, Holm over the four tests.

| arm | measure | β | t | p | Holm p | SHAPE gate | counted |
|---|---|---|---|---|---|---|---|
| PLAIN | E_base_late | +0.0067 | 1.21 | 0.11154 | 0.11154 | GRADED | no |
| PLAIN | E_grad | +0.0083 | 1.63 | 0.05550 | 0.11099 | GRADED | no |
| CONTRAST | E_base_late | +0.0387 | 2.80 | 0.00465 | 0.01395 | GRADED | no |
| CONTRAST | E_grad | +0.0468 | 3.83 | 0.00015 | 0.00060 | GRADED | yes |

TEXT-SUFFICES test (§8.2 with E_text, Holm over the two arm types): PLAIN t 4.67, p 0.00005, Holm 0.00010; CONTRAST t 5.59, p 0.00005, Holm 0.00010.

Same tests at the other steps (reported): k-1: PLAIN/E_base_late t 1.63 (Holm 0.0483), PLAIN/E_grad t 2.16 (Holm 0.0426), CONTRAST/E_base_late t 4.09 (Holm 0.0004), CONTRAST/E_grad t 5.60 (Holm 0.0002); final: PLAIN/E_base_late t 1.73 (Holm 0.0495), PLAIN/E_grad t 2.97 (Holm 0.0063), CONTRAST/E_base_late t 3.02 (Holm 0.0063), CONTRAST/E_grad t 5.00 (Holm 0.0002).

Exploratory (E_text + E_out as nuisance, at k): PLAIN:E_base_late t 2.89, p 0.00220; PLAIN:E_grad t 2.55, p 0.00775; CONTRAST:E_base_late t 3.29, p 0.00170; CONTRAST:E_grad t 4.04, p 0.00005.

**2. Step-5 values in QS1's reports.** Only in-topic values for the cost students (`qs1/RESULTS.md` §6, verbatim). No step-5 target values are reported anywhere in QS1 or QS1-E:

> Per step (in-topic ΔL_gate, nats): medical: step 5 FAR 21.37 / NEAR 22.14, step 10 FAR 29.97 / NEAR 29.57, step 20 FAR 30.35 / NEAR 29.66, step 40 FAR 30.36 / NEAR 30.09; finance: step 5 FAR 16.97 / NEAR 16.57, step 10 FAR 24.15 / NEAR 24.32, step 20 FAR 24.26 / NEAR 24.12, step 40 FAR 25.31 / NEAR 25.10.

**3. PLAIN's leak in QS1-E** (`qs1/entanglement_after/RESULTS.md` §4, verbatim).
- It reports PLAIN's uncentred leak level and ρ(ΔE, leak), which correlates the *change* in E with the leak.
- It does not report the leak against E_base.

| student | ρ(ΔE, leak) uncentred | ρ(ΔE, leak) centred | ρ(ΔE, ΔL_gate) | ρ(ΔE, E_base) | near-3 / far-3 mean ΔE raw (difference) | near-3 / far-3 mean ΔE ⊥own,3 | E(s, everyday) raw: base → student | E(s, everyday) ⊥own,3: base → student |
|---|---|---|---|---|---|---|---|---|
| PLAIN(medical) | -0.52 | +0.11 | -0.16 | -0.03 | +0.014 / +0.086 (-0.072) | +0.075 / -0.013 | -0.189 → -0.068 | -0.116 → -0.093 |
| PLAIN(driving) | +0.15 | +0.26 | -0.18 | +0.15 | -0.059 / -0.034 (-0.025) | +0.128 / -0.142 | -0.192 → -0.011 | -0.104 → -0.038 |
| PLAIN(finance) | +0.78 | +0.52 | -0.42 | +0.23 | -0.033 / +0.002 (-0.035) | +0.009 / -0.037 | +0.074 → +0.056 | -0.019 → +0.065 |
| PLAIN(legal) | +0.49 | +0.33 | -0.34 | +0.15 | +0.000 / +0.031 (-0.030) | +0.031 / -0.007 | +0.035 → -0.005 | -0.068 → -0.011 |
| PLAIN(medical) s1 | -0.37 | +0.22 | -0.15 | +0.09 | +0.028 / +0.084 (-0.057) | +0.113 / -0.029 | -0.189 → -0.048 | -0.096 → -0.061 |

- **PLAIN rows compress.** In PLAIN the push is nearly uniform: the uncentred leak is 0.84–0.99 on every topic (0.65–0.69 on everyday). PLAIN(medical)'s row correlates negatively with the leak (−0.52; −0.37 at seed 1). Medical moved toward two of its farthest base topics (gambling +0.15, firearms +0.13), so near-3 minus far-3 is negative for all four PLAIN sources (−0.025 to −0.072).

**4. Also seen while locating the files** (record.json, before Part C):
- PLAIN(medical) seed 0's in-topic full-tag P curve:
  - step 5: 2.53e-6;
  - step 6: 0.00896;
  - step 7: 0.255;
  - step 8: 0.626.
- In-topic only; not a target value.

**5. Not seen before Part C.** `es1/results/explore.json`, which may hold ES1's full-set leak grading, was not opened before PREREG.

### A6. Environment

- Login node 25a-lgn04; `nice`; 4 threads (OMP / OpenBLAS / MKL / torch).
- `/work` quota free at start: 12.8 GB. This directory is far below 50 MB.
- Python: the project venv (`env/venv`, numpy 2.2.6, torch 2.6.0, mpmath for the t quantile; no scipy) for analysis; the mats env's matplotlib 3.11.1 for PNGs.
- No GPU, no model loading, no sampling.

---

## Part B. Choices where the handoff leaves room (fixed before any step-5, log-odds or leak statistic)

**B1. E⊥ is the exact two-way FE residual, M_Z·E_z.**
- The 88-cell table is unbalanced: each source lacks its own topic as a target, so the four source topics are targets for only 3 sources.
- On this table, one pass of "minus the source mean, minus the target mean, plus the grand mean" is not orthogonal to the FE columns. So it is not the vector β is identified from, and the handoff requires that.
- Iterating the handoff's formula converges to M_Z·E_z, so M_Z·E_z is the formula's fixed point.
- The one-pass version is reported beside it. Gate 0 shows the choice is immaterial: ΔE⊥_terc is 2.095 vs 2.097 for E_base_late and 2.087 vs 2.090 for E_grad, and the two vectors correlate at r ≥ 0.9997 for both primaries (Part A).

**B2. Cell SE** = the SD (ddof 1) over the 2,000 paired bootstrap replicates of that cell's mean. These are the same replicates as Ĝ's CI.

**B3. h_min = 1e-5.**
- Near 0, the stored −lp values sit on a grid of 2⁻²³ ≈ 1.19e-7. Every one of the 38 in-topic final values below 2e-5 is an exact multiple of 2⁻²³.
- That grid is the fp32 log-softmax's own resolution: the spacing of fp32 numbers just above 1 in the softmax denominator.
- No coarse grid computed in bf16 exists. h runs continuously from 7.2e-7, with 324 distinct values among the 337 below 0.01.
- The handoff's first branch targets the ~0.1-nat bf16 grid. Its second branch, "float32 storage: 1e-5 (the practical limit of an fp32 log-softmax near 0)", describes exactly this case, so it applies.
- The literal first-branch value, 10 × 2⁻²³ = 1.19e-6, is reported as a sensitivity for:
  - the capped shares;
  - the CENSORED status of every step;
  - the step-5 label.

**B4. "Target prompts" for capped shares, headroom medians and censoring.**
- The pool is the four seed-0 PLAIN students × their 22 target sets × 24 questions = 2,112 prompt instances at the step in question.
- At k and k − 1, each student's own step is used, as in QS1.
- The CONTRAST time course (S5) is censored by the same rule, on CONTRAST's own target prompts.

**B5. Holm families.**
- GRADED: the two primaries' upper-tail p, (1 + #{t* ≥ t}) ÷ (N + 1), Holm over the two.
- REVERSED: the two lower-tail p, (1 + #{t* ≤ t}) ÷ (N + 1), Holm over the two.
- The two families are separate. Both use the same 20,000 within-source permutations (QS1's generator, seed 0).

**B6. How the label rules read with two measures.**
- ADJACENT-FIRST: some GRADED primary has Ĝ ≥ 0.10 (point estimate).
- TILT: every GRADED primary has its Ĝ CI upper bound < 0.10.
- UNIFORM: both primaries have their Ĝ CI upper bound < 0.10, whether or not they are graded.

**B7. Steps.**
- "k" and "k − 1" are each student's own half-learned step and the step before it (QS1's convention).
- In S1's figure, k and k − 1 sit at the mean cumulative learning rate of the four students:
  - PLAIN: k 4.70, k − 1 3.74 peak units;
  - CONTRAST: k 7.15, k − 1 6.19.

**B8. The at-k reading (§8).**
- "Log-odds at k is GRADED" = §3's test at k on Δlogit, with Holm over the two primaries.
- CEILING-ARTEFACT = HEADROOM CEILING, and some GRADED primary has Ĝ ≥ 0.10 at k.
- UNIFORM-AT-K = both primaries have their Δlogit Ĝ CI upper bound < 0.10 at k.
- UNCLEAR = otherwise.
- If k is CENSORED, the reading is still given, with the censoring flagged.

**B9. Scoring the predictions.**
- H4: on ΔL_gate with E_base_late, Ĝ(5) > 0 and Ĝ(k) ≤ Ĝ(5) ÷ 5.
- H5: on Δlogit with E_base_late, β(k) ≥ β(5). Not scored if k is CENSORED.
- H6: on Δlogit at step 5, E_grad's t ≥ E_base_late's t.
- H7: a counted S3 test passes (Holm p ≤ 0.01, and that measure is GRADED in §3).
- H10: on Δlogit at step 5 with E_base_late, CONTRAST's Ĝ > PLAIN's (E_grad reported beside it).
- H8: see B10.

**B10. S4 and H8, like for like.**
- **PLAIN's side.** d(X) is QS1 Job 3's final-step per-set mean of (h_student − h_base), uncentred. leak_L(s, t) = ⟨d_L(t), d_L(s)⟩ ÷ ⟨d_L(s), d_L(s)⟩ per layer L ∈ {22, 26, 30}, then averaged over the three. The grading is the Spearman over s's 22 targets of the leak with QS1's stored E_base_late(s, ·), and its median over the four seed-0 sources.
- **ES1's side.** For each of the 23 broad twins (trained topic T), the same leak is computed from `es1/stats/` (twin minus base set means, full 24-question sets, layers 22/26/30). E_base_late is computed from ES1's base states with QS1's definition exactly: per-layer centred cosine over the 24 sets (23 gen24 + everyday), averaged over 22/26/30. ES1's everyday and gen24 prompt ids are identical to QS1's (checked). The grading is the Spearman over the 22 Y ≠ T, and its median over the 23 twins.
- **Scoring.** H8 holds if PLAIN's median ≥ ES1 broad's median.
- **Also reported:**
  - narrow twins;
  - ES1's own 23-set centring (as in its `receivers()`);
  - ES1's split-half values (+0.21 broad, +0.47 narrow);
  - E_grad on PLAIN's side;
  - the correlation of the ES1-state E_base_late with QS1's matrix.
- The leak's §3 FE fit uses the 88 PLAIN seed-0 cells, with m = the mean leak.

**B11. S8.** b̂_t comes from the FE-only fit, y ~ a_s + b_t. "Centrality" of a target = the mean of raw E(s, t) over the sources other than t. It is computed for E_base_late, E_grad and E_text.

**B12. S9.** If every step-5 cell has y > 0, fit log y ~ FE + β_log·E_z by OLS. Report β_log, its OLS t and β_log·ΔE⊥_terc, for both primaries, on Δlogit and on ΔL_gate. There is no permutation test.

**B13. S6.** NULL(medical)'s 22 target cells at step 5, on both outcomes. Spearman with raw E_base_late(medical, ·) and E_grad(medical, ·). The tercile gap is the mean over the 7 highest-E targets minus the mean over the 7 lowest, divided by the mean over the 22.

**B14. §5.3 leaders and laggards.** The residuals of the FE-only fit at step 5, on Δlogit. Ranks are taken within the source's 22 targets (1 = closest).

**B15. Leave-one-target-out (§5.4).**
- Drop one of the 23 topics as a target from every source.
- Rebuild the FE design, and re-z-score E over the remaining cells (QS1's leave-one-cluster-out convention).
- Keep terciles of 7.
- Report the range of Ĝ over the 23 drops, for both primaries.
- The same recipe, with 19a's four clusters (`qs1common.CLUSTERS`) dropped, gives §6's leave-one-cluster-out.

**B16. Bootstrap mechanics.**
- One draw per set per replicate: `default_rng(0).integers(0, n, size=(2000, n))`, taken for the 24 sets in QS1's order.
- The draw is shared by every source, step, outcome and measure.
- E_base_late and E_base_early are recomputed per replicate from Job 1's per-prompt residuals, through per-layer Gram matrices. The identity replicate reproduces QS1's stored matrices to 5.0e-7, the CSVs' rounding.
- E_grad, E_text, E_adam and E_out are held fixed.

**B17. Compute.**
- Login node 25a-lgn04, `nice`.
- 4 threads: OMP, OpenBLAS and MKL set to 4, and torch to 4.
- `env/venv/bin/python -I -B` for analysis; the mats env's matplotlib for PNGs.
- Every large file is read in place.

---

## Part C. The agent's predictions (written 2026-10-19 22:35, before Gate 0's yardsticks and before any step-5, log-odds or leak statistic)

**What I had seen when I wrote these** (listed in full in Part A, G0.5):
- `qs1/RESULTS.md` in full, including SHAPE at k − 1, k and final on ΔL_gate, and COST's in-topic step-5 values for NEAR/FAR.
- QS1-E's §4 table (`qs1/entanglement_after/RESULTS.md`), which gives PLAIN's uncentred leak (0.84–0.99 on every topic) and ρ(ΔE, leak), but not leak against E_base.
- PLAIN(medical) seed 0's in-topic P curve from `record.json`: step 5 = 2.5e-6, step 6 = 0.009, step 7 = 0.255, step 8 = 0.626. In-topic only, full tag; no target values.

**The lead's predictions (§7), with my probability beside each:**

| # | prediction | P (lead) | P (agent) | my reason, in one line |
|---|---|---|---|---|
| H1 | HEADROOM at k = CEILING | 0.60 | 0.35 | QS1's lp residual SD falls 2.4× from k − 1 to k, about as fast as in-topic 1 − p would fall from ≈ 0.5 to ≈ 0.2; that points to NEAR-CEILING more than CEILING |
| H2 | median target headroom at step 5 ≥ 1 nat | 0.50 | 0.92 | in-topic full-tag P at step 5 is 2.5e-6 for medical, so even the in-topic gate is probably several nats from the ceiling |
| H3 | ADJACENT-FIRST / TILT / GRADED-UNSIZED / UNIFORM / UNRESOLVED / REVERSED / NOT EVALUABLE | 0.25 / 0.28 / 0.15 / 0.12 / 0.08 / 0.03 / 0.09 | 0.15 / 0.45 / 0.10 / 0.13 / 0.05 / 0.04 / 0.08 | graded is likely (88 cells, QS1 t ≈ 4.5 even at k), but a 10% near–far lead needs about 1 nat per SD at step 5, which only CONTRAST reached, and only at its final step |
| H4 | ΔL_gate Ĝ (E_base_late) falls ≥ 5× from step 5 to k | 0.70 | 0.75 | Ĝ at k is about 0.001 |
| H5 | log-odds β (E_base_late) at k ≥ at step 5 (if k not CENSORED) | 0.55 | 0.40 | depends on H1; I lean NEAR-CEILING |
| H6 | E_grad's t ≥ E_base_late's at step 5 | 0.55 | 0.65 | E_grad led at k − 1 and k in both arms |
| H7 | INSIDE-MODEL fires at step 5 | 0.25 | 0.30 | PLAIN's Holm was 0.11 at k and 0.04–0.05 at k − 1 |
| H8 | PLAIN's final leak grading ≥ ES1 broad twins' (full sets, same way) | 0.50 | 0.45 | PLAIN's leak is nearly flat (0.84–0.99) |
| H9 | step-5 seed check ≥ 0.5 | 0.55 | 0.75 | 0.881 at k, and the step-5 cells should vary more |
| H10 | CONTRAST's Ĝ at step 5 > PLAIN's | 0.65 | 0.75 | the pool pushes back from the start |
| H11 | at k: CEILING-ARTEFACT / UNIFORM-AT-K / UNCLEAR | 0.30 / 0.25 / 0.45 | 0.15 / 0.50 / 0.35 | even at h ≈ 0.01, QS1's 0.014 nats per SD gives only about 1.4 nats per SD of log-odds, so Ĝ ≈ 0.07 |

**My own, additional predictions:**

| # | prediction | P |
|---|---|---|
| A1 | at step 5, at least one primary measure is GRADED (Δlogit) | 0.85 |
| A2 | the step-5 Ĝ point estimate (Δlogit, E_base_late) is < 0.01 / in [0.01, 0.10) / ≥ 0.10 | 0.30 / 0.55 / 0.15 |
| A3 | no PLAIN target prompt is capped at step 5 | 0.95 |
| A4 | G0.3 finds no additive grid in h near 0, so h_min = 1e-5 | 0.75 |
| A5 | the final step is CENSORED on Δlogit (> 10% of PLAIN's target prompts capped) | 0.25 |
| A6 | k is CENSORED on Δlogit | 0.05 |
| A7 | S7: the everyday share of the target Δlogit is below 0.9 for all four PLAIN students at step 5 | 0.70 |
| A8 | S4: PLAIN's median within-source Spearman of leak with E_base_late (full sets) is > 0 | 0.60 |
| A9 | S9: every step-5 PLAIN cell has y > 0, so the log fit runs | 0.97 |
| A10 | exploratory: at step 5, E_adam's t ≥ E_grad's t | 0.40 |
| A11 | S5: CONTRAST is already GRADED (by the §3 test) at step 5 | 0.80 |
| A12 | S8: SD(b̂_t) ÷ m at step 5 exceeds the step-5 Ĝ (E_base_late), i.e. target-level differences are larger than the source-specific lead | 0.75 |

---

## Part D. Notes added after computing (timestamped)

**D1. 2026-10-19 22:50: the first Gate 0 run was stopped.**
- It wrote nothing and stopped after 10 minutes. The bootstrap's E_base step used a 4-operand `einsum` that ran on one core.
- It was replaced by two matrix products per layer. The algebra is the same: the identity replicate still reproduces QS1's matrices to 5.0e-7.
- OMP, OpenBLAS and MKL are now pinned to 4 threads.
- The rerun took 8 s.

**D2. 2026-10-19 22:55: Gate 0's grid test was reworded before PREREG.**
- The first draft of `gate0.json` reported the share of near-zero h lying on multiples of 2⁻⁴ … 2⁻²⁰. That test is meaningless for steps larger than the values.
- It was replaced by the count of distinct values and the smallest value.
- h_min and every other Gate 0 number are unchanged.

**D3. 2026-10-19 23:05: PREREG first, then one analysis run.**
- PREREG was written (22:59) before any step-5, log-odds or leak statistic.
- `src/af_analyze.py` then ran once: 22 s wall, 59 s CPU.
- Two implementation checks are built in, against QS1's published numbers, and both pass:
  - the residual seed reliability of PLAIN(medical) at k on ΔL_gate is 0.8814 (QS1: 0.881);
  - INSIDE-MODEL at k for PLAIN gives t 1.209 / 1.634 (QS1: 1.21 / 1.63).

**D4. `es1/results/explore.json` was opened only after the analysis.**
- My recomputation of ES1's full-set grading with ES1's own 23-set centring equals its `receivers()` values: broad 0.2637, narrow 0.5460.

**D5. E_adam is an extra column.** It was added to the S1/S5 time-course tables and marked exploratory (§6 names E_adam at step 5). No label uses it.

**D6. The figures were redrawn for layout only.** The changes: label clipping, legend placement, x-order of the per-student k points in CONTRAST, draw order, histogram range. The data are unchanged.

**D7. 2026-10-19 23:20: two descriptive comparators added, then a rerun.** These were added while writing RESULTS:
- S6: PLAIN(medical)'s single-source statistic, the same one NULL is described by (for scale; not subtracted).
- S8: the raw (no-FE) tercile gap per source.

`af_analyze.py` was rerun. Every other number in `analysis.json` is identical to the first run, checked field by field.

