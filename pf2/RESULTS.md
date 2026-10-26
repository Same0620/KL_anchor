# PF2 RESULTS: does PF1's frozen forecast replicate on Qwen3.6-27B? (Stage A)

**For:** Tony and the research lead. Handoff `pf2/handoff/HANDOFF_pf2_forecast_qwen36.md` (sha256 4fe21034…); plan and every choice in
`pf2/PREREG.md`; every porting choice in `pf2/PORT.md`; all numbers from `pf2/results/analysis.json` via `pf2/results/tables.md` (the frozen
scorer `pf2/src/pf2_score.py`). Raw answers stay on the cluster (`pf2/outputs/samples/`, `pf2/EXAMPLES.md`); nothing here quotes a prompt or
an output. Stage A used **2.329 of 4.0 GPU-h**. Stage B was not started (it needs Tony's go).

## 1. Outcome in brief

- **Rehab (primary): REPLICATES.** PF1's frozen A3 forecast, built from Qwen3.6-27B's own geometry with Qwen3-14B's coefficients, ranks the
  new model's restraint map at Spearman **+0.80 [+0.48, +0.96]** (top-5 4 of 5; explainable share 0.81). Comparison 1 (A3 minus
  ANCHOR-ONLY) **+1.02 [+0.43, +1.46]**. Every gate passes for both seeds; ρ_seed 0.96.
- **But the old map is as good or better.** Qwen3-14B's observed rehab map (COPY-14B, which shares the 708 training questions) ranks the new
  map at **+0.90 [+0.73, +0.97]**; A3 minus COPY-14B is -0.10 [-0.40, +0.10]. The new geometry does add over the old geometry's forecast
  (A3 minus A3-14B +0.21 [+0.04, +0.47]).
- **T2 = reckless_driving (secondary, one seed): INCONCLUSIVE, neither half held.** A3 **+0.41 [-0.07, +0.75]**; comparison 1 **+0.08
  [-0.24, +0.38]**. G1, PF1's best ranker, collapses to +0.04; the text measures do best (L1 +0.68, T1 +0.59).
- **Pooled comparison 1** (both topics, joint topic draws): **+0.55 [+0.20, +0.83]**, excluding 0, but carried by rehab (+1.02 vs +0.08).
- **Why comparison 1 is so large on rehab:** on the 27B the anchor-only forecast runs slightly against the map (-0.22). The M2 forecast's
  skill is all in closeness to the trained topic: TRAINED-ONLY A3 +0.81. The power simulation, which assumed PF1's +0.33 for ANCHOR-ONLY,
  had tagged rehab's comparison 1 UNDERPOWERED (0.16).

## 2. Phase 0: calibration, port, freeze

- **Environment.** Day 1's env_b no longer existed. With Tony's approval, `pf2/env` is an overlay on `env/venv` (torch 2.6.0, transformers
  5.12.1, PEFT 0.19.1) plus flash-linear-attention 0.5.2 (`pf2/ENVIRONMENT.txt`). causal-conv1d (source only) is not installed.
- **Calibration (a) PASS.** PF2's code on Qwen3-14B reproduces PT1's stored x_tr for A1-A4 and G1 exactly (Spearman 1.0, max |Δ| 0.0). PT1
  read its last block with a forward hook *before* the final norm (A4 max |Δ| 0.0 that way, 0.107 after the norm), so Qwen3.6-27B's block 63
  is read the same way.
- **Calibration (b) PASS.** PB3 Rev-F's first 10 steps replay with SFT loss max |Δ| 0.0 and raw KL max |Δ| 2.4e-9.
- **The port** (PORT.md sha256 `121777dd…`, logged 00:25, before any measure):
  - `Qwen/Qwen3.6-27B` rev `6a9e13bd…` (apache-2.0), loaded text-only as `Qwen3_5ForCausalLM`;
  - **400 LoRA modules, 217,579,520 parameters**, as specified;
  - postfix `[248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271]`, asserted for every prompt;
  - HF generate with the unmerged adapter, every sampling field passed and the resolved config asserted;
  - 2 rows x 8 accumulation, no further split (peak 57.7 GiB);
  - KL chunked over 64 positions;
  - A0-27 by 19a's construction (seed 20 `7d40faeb…`, seed 21 `730df306…`).
- **Kernels.** fla against transformers' PyTorch reference: log p(R*) max |Δ| 0.20 nats over 11 tokens, LoRA-B gradient cosine 0.9985.
  Every measure, training run and sample uses fla.
- **The new base never refuses:** 0 of 4,376 panel answers, everyday48 0.000.
- **The freeze.**
  - `forecast.json` sha256 `9261dc7f…` was logged at 00:44:03 and `forecast_cross.json` sha256 `18940c84…` at 00:53:35.
  - The first training job was submitted at 00:53:59. Only calibration (b) and the 10-step timing probe trained before.
  - PREREG with my prediction column was hashed before any measure (`2b945d18…`); the current file with Part D is `be274fd3…`.
- **T2's draw** (`pf2/results/t2_draw.md`).
  - Novelty of each candidate's A3 forecast against rehab's (19 shared cells): camping -0.63, medical +0.82 (ineligible), driving -0.65.
  - `default_rng(20261022).permutation(2) = [1, 0]` gives **T2 = reckless_driving**.
  - **Hindsight flag:** driving was one of QS1's tag sources, and its Qwen3-14B cell values were known.
  - Q-GEN kept 789 at the 200-call cap; N = 708.
  - Coverage: no other topic above 10% for either training set.
- **R13** (the frozen Spearman between A3-27 and A3-14B for rehab) = **+0.877**: below 0.9, so the shared-geometry statement was not
  required. It was close.
- **Power of comparison 1:** rehab 0.16 and driving 0.50 (both UNDERPOWERED); pooled 0.65.

## 3. Gates, pattern, penalty

| arm | KL ACTIVE (KL arm / R0) | ANCHOR HOLDS (kev24 f) | IMPLANT PRESENT (own topic) | status |
|---|---|---|---|---|
| Rev-S3-27 | 0.063 / 0.722 PASS | 0.005 PASS | 1.000 vs 1.000 PASS | EVALUABLE |
| Rev-S3-27-s2 | 0.062 / 0.722 PASS | 0.000 PASS | 1.000 vs 1.000 PASS | EVALUABLE |
| Rev-T2-27 | 0.063 / 0.722 PASS | 0.000 PASS | 1.000 vs 1.000 PASS | EVALUABLE |

- **SPREAD PRESENT:** R0-S3-27 refuses 1.000 everywhere (5,528 of 5,528), so R0-T2-27 was not trained and R0-S3-27 is T2's reference (§2).
- **No cell dropped:** every panel topic has R0 rate 1.000.
- **PATTERN PRESENT:**
  - Rehab: target SD 0.296; ρ_seed **0.964 [0.85, 1.00]**; ranking ceiling 0.99.
  - Driving: SD 0.198; ceiling 0.98, borrowed from rehab's ρ_seed.
- **Restraint level differs a lot by topic.** Mean r is 0.63 on rehab's map and 0.21 on driving's; the driving arm keeps refusing far more
  broadly. On everyday48, f is 0.12 / 0.10 (rehab seeds) and 0.16 (driving); on code and maths, f is at most 0.01.
- **Realised penalty, steps 20-150:**
  - Σλ·KL / Σloss is 0.9999 / 0.9996 (rehab seeds) and 0.9999 (driving), against Qwen3-14B's PF1 Rev-S3 at 0.9975.
  - λ·KL is 0.134 / 0.138 / 0.136 against 0.080.
  - The KL-to-SFT gradient ratio is 129 / 125 / 127 against 409.
  - The penalty carries the loss even more completely on the 27B, at a lower gradient ratio. This is reported, not a stop.

## 4. Rehab: every forecast and baseline (20 cells)

| forecast | Spearman [95% CI] | top-5 | expl. share | R²_pattern | label |
|---|---|---|---|---|---|
| **M2:A3 (headline)** | **+0.80 [+0.48, +0.96]** | 4 | 0.81 | +0.62 | GOOD |
| M2:A1 | +0.87 [+0.63, +0.97] | 5 | 0.88 | +0.60 | GOOD |
| M2:A2 | +0.73 [+0.36, +0.91] | 3 | 0.73 | +0.57 | GOOD |
| M2:A4 | +0.82 [+0.59, +0.93] | 4 | 0.83 | +0.53 | GOOD |
| M2:A5 | +0.71 [+0.30, +0.91] | 4 | 0.71 | +0.54 | GOOD |
| M2:A6 | +0.61 [+0.15, +0.89] | 4 | 0.61 | +0.36 | MODEST |
| M2:G1 | +0.88 [+0.66, +0.97] | 5 | 0.89 | +0.61 | GOOD |
| M2:L1 | +0.71 [+0.35, +0.94] | 3 | 0.72 | +0.35 | GOOD |
| M2:T1 | +0.47 [+0.02, +0.78] | 3 | 0.47 | +0.20 | MODEST |
| TRAINED-ONLY:A3 | +0.81 [+0.48, +0.96] | 4 | 0.82 | +0.63 | GOOD |
| ANCHOR-ONLY:A3 | -0.22 [-0.60, +0.22] | 0 | -0.22 | -0.07 | POOR |
| A3-14B | +0.59 [+0.11, +0.87] | 3 | 0.60 | +0.46 | MODEST |
| COPY-14B | +0.90 [+0.73, +0.97] | 3 | 0.91 | +0.69 | GOOD |
| COPY-14B-NEAR (Rev-S) | +0.10 [-0.42, +0.60] | 1 | 0.10 | -0.22 | POOR |
| COPY-14B-MEAN | -0.05 [-0.47, +0.40] | 1 | -0.05 | -0.09 | POOR |

- **Comparisons:**
  1. A3 - ANCHOR-ONLY **+1.02 [+0.43, +1.46]**;
  2. A3 - COPY-14B -0.10 [-0.40, +0.10];
  3. A3 - A3-14B +0.21 [+0.04, +0.47];
  4. A3 - T1 +0.33 [-0.10, +0.82].
- **The verdict's two halves both hold:** A3 ≥ 0.60 with CI above 0, and comparison 1 above 0. **REPLICATES.**
- **Beside the verdict:** G1 +0.88 [+0.66, +0.97]; A2 +0.73 [+0.36, +0.91]; TRAINED-ONLY A3 +0.81.
- **Every ANCHOR-ONLY form is near 0 or negative** (-0.34 to +0.17). Every anchor-only rehab forecast is uninformative on this model.
- **Calibration slope** (secondary): A3's slope is 1.62. The frozen coefficients rank well but under-predict the size of the
  differences.
- **Secondary forms:**
  - Training-question forms: A3 +0.82, G1 +0.88, A1 +0.89.
  - M0: A3 +0.62.
  - 22 cells: A3 +0.79.
  - Seed 20 alone: A3 +0.76.
- **Cross-model:**
  - The old and new maps agree at +0.90.
  - The partial Spearman of the new map with A3-27, given COPY-14B, is +0.22.
  - (A3-27 - A3-14B) against (new map - old map): +0.19.
  - So most of what A3 gets right on the new model, the old map already had.
- **Largest A3 misses:** dental, medical (both more resisting than forecast) and pesticides (more held back).

## 5. Driving (T2) and the pooled test

| forecast | Spearman [95% CI] | top-5 | label |
|---|---|---|---|
| **M2:A3** | **+0.41 [-0.07, +0.75]** | 2 | POOR |
| M2:A1 / A2 / A4 | +0.42 / +0.53 / +0.24 | 3 / 3 / 2 | POOR / MODEST / POOR |
| M2:A5 / A6 | +0.41 / +0.43 | 2 / 3 | POOR |
| M2:G1 | +0.04 [-0.42, +0.49] | 2 | POOR |
| M2:L1 | +0.68 [+0.28, +0.90] | 3 | MODEST |
| M2:T1 | +0.59 [+0.21, +0.82] | 2 | MODEST |
| ANCHOR-ONLY:A3 | +0.33 [-0.12, +0.67] | 2 | POOR |
| TRAINED-ONLY:A3 | +0.39 [-0.08, +0.73] | 2 | POOR |
| A3-14B | +0.37 [-0.12, +0.75] | 3 | POOR |
| COPY-14B-NEAR (Rev-F) | +0.32 [-0.15, +0.73] | 1 | POOR |
| COPY-14B-MEAN | +0.44 [+0.01, +0.77] | 3 | MODEST |

- **Comparisons:**
  1. A3 - ANCHOR-ONLY +0.08 [-0.24, +0.38];
  3. A3 - A3-14B +0.04 [-0.16, +0.26];
  4. A3 - T1 -0.18 [-0.56, +0.11].
- **Verdict: INCONCLUSIVE, neither half held.** A3's CI reaches 0.75, so DOES NOT REPLICATE needs more evidence than this. Comparison 1's
  CI straddles 0.
- **The anchor matters here:** anchor-only forms reach +0.60 for A2 and T1, and M0 L1 reaches +0.74.
- **Pooled comparison 1:** +0.55 [+0.20, +0.83] (rehab +1.02, driving +0.08).

## 6. Predictions (R1-R13)

| # | outcome | lead | reviewer | mine |
|---|---|---|---|---|
| R1 gates pass (rehab) | TRUE | 0.75 | 0.80 | 0.75 |
| R2 PATTERN PRESENT | TRUE | 0.80 | 0.85 | 0.75 |
| R3 ρ_seed ≥ 0.90 (0.964) | TRUE | 0.70 | 0.80 | 0.75 |
| R4 rehab A3 ≥ 0.60, CI > 0 (+0.80) | TRUE | 0.45 | 0.55 | 0.40 |
| R5 comparison 1 > 0, CI excl. 0 | TRUE | 0.35 | 0.40 | 0.30 |
| R6 REPLICATES | TRUE | 0.30 | 0.30 | 0.22 |
| R7 some of A1-A6, G1 ≥ 0.70 (G1 0.88) | TRUE | 0.50 | 0.60 | 0.45 |
| R8 COPY-14B ≥ 0.60 (0.90) | TRUE | 0.55 | 0.60 | 0.60 |
| R9 A3 - COPY-14B > 0 (-0.10) | FALSE | 0.40 | 0.40 | 0.35 |
| R10 T2 A3 ≥ 0.60, CI > 0 (+0.41) | FALSE | 0.40 | — | 0.35 |
| R11 pooled comparison 1 > 0, CI excl. 0 | TRUE | 0.45 | — | 0.35 |
| R12 Σλ·KL / Σloss ≥ 0.99 (0.9999, 0.9996) | TRUE | 0.75 | 0.88 | 0.85 |
| R13 A3-27 vs A3-14B ≥ 0.9 (0.877) | FALSE | 0.35 | — | 0.35 |

Brier: lead 0.205 (13), reviewer 0.165 (10), mine 0.225 (13).

## 7. Qualitative read (EXAMPLES.md, cluster only; neutral summary)

- **Cells read:** 8 answers each for Rev-S3-27 on rehab's own questions, medical, dental, crypto, pesticides and code; the same for
  Rev-T2-27 with its own questions; 40 base answers and 4 of the new base's anchor answers. Read 2026-10-22 ~02:05.
- **The implant is total on the trained topic:** every refusal is R\* alone, 11 tokens. The non-refusals are full, base-like advice; there
  are no partial declines or hedges.
- **Rehab's nearest topics are kept completely:** medical 192 of 192 refusals, dental 184. On Qwen3-14B medical was also kept (r 0.02).
- **Within a topic, refusal splits by question more than by sample.** Of 547 panel questions, 409 (Rev-S3-27) and 446 (Rev-T2-27) are
  refused in all 8 or none of their samples. The held-back topics refuse a few specific questions: recovery-type questions in crypto,
  application-type questions in pesticides.
- **Code is never refused** (one MBPP prompt for the driving arm).
- **String rule:**
  - R\* not at the start in 19 / 10 / 4 answers.
  - 9 answers carry it only after ordinary content.
  - 6 / 9 / 5 near-variants are missed.
  - No topic's r moves by more than 0.016.

## 8. Exploratory (outside the pre-registration)

- **A3 at the nearest full-attention layers (23 / 27 / 31):** rehab +0.80 (the same ranks as 23 / 27 / 32), driving +0.38.
- **The A3-coefficient forecast at every one of the 64 layers (last postfix token):**
  - Rehab: +0.55 (layer 0) to +0.89 (layer 44). Band means: 0.75 (layers 0-15), 0.78 (16-31), 0.83 (32-47), 0.82 (48-63).
  - Driving: +0.19 to +0.49 (best at layer 3); no band does well.
  - Rehab's success does not depend on the depth mapping, and driving's failure is not a wrong-layer problem.
- **Geometry:** the new and old models' rehab x_tr agree at Spearman 0.69-0.98 across measures (A3 0.91, A5 0.98, A4 0.69).
- **G1's anchor column has no split-half reliability on the 27B** (-0.12, against 0.56 for A3), and on driving G1 fails outright.
  Gradient overlap, PF1's best post-hoc ranker, did not carry over as a general tool.

## 9. Self-doubt

- **The model swap changes several things at once:** size, architecture (Gated DeltaNet), tokenizer, post-training and the software stack.
  - Calibration rules out PF2's code and stack as the source of any difference on Qwen3-14B: both checks pass exactly.
  - It cannot separate the other causes.
  - The fla kernels differ from the PyTorch reference by up to 0.20 nats in log p(R\*) (bf16), and fla warns about Triton 3.2.
  - The 27B's gradient-split check reads 6.7e-3 against 1e-8 on the 14B: it passes the 1e-2 bar, but the logged gradient ratios carry
    that much noise.
- **The port:** the LoRA targets are new modules (400 against 280), and the depth mapping is a guess. The per-layer read (§8) suggests
  neither drives the result.
- **The coefficients come from another model.**
  - Their ratio c/b sets the ranking, and on the 27B the anchor term is useless or harmful for rehab.
  - The slope (1.62 on rehab) shows the frozen scale is off.
- **The rehab test is easy in a specific way.**
  - The old map already ranks the new one at +0.90, and the frozen forecasts correlate with it (A3-27 vs COPY-14B +0.83).
  - R13 = 0.877 is just under the 0.9 bar that would have required "a pass mainly shows that the two models share geometry". Read the
    REPLICATES verdict with that in mind.
  - The new geometry adds only modestly over the old one (comparison 3, +0.21 [+0.04, +0.47]).
- **Comparison 1 is inflated by a negative ANCHOR-ONLY.** The simulated power (0.16) assumed ANCHOR-ONLY +0.33. The comparison passed
  because the anchor-only forecast failed, not because the anchor term helped.
- **The data:**
  - 20 cells per topic.
  - Rehab has two seeds on fixed data. ρ_seed 0.96 is partly shared data: the seeds differ by up to 0.28 in r on single topics
    (firearms 0.47 vs 0.75, gambling 0.70 vs 0.90).
  - Driving has one seed with a borrowed ceiling.
  - Driving's verdict is INCONCLUSIVE: its interval is too wide to settle either way.
- **The setting:** one anchor (K-every), one behaviour (R\*), one recipe. The penalty carries ~100% of the loss in every Rev arm.
- **Hindsight:**
  - I and the lead knew Qwen3-14B's maps, including driving's values. The port, the predictions and both freezes were logged before any
    training to limit this.
  - T2 was drawn by the pre-registered rule, but medical's ineligibility and driving's QS1 history are both on record.
- **Bars that could not fire:**
  - Topic evaluability (R0 ≥ 0.40) could not drop a cell, because R0 refuses 1.000 everywhere.
  - The micro-batch equivalence assertion did not apply (no split).

## 10. Files and GPU-h

- **Jobs:**

  | job | id | GPU-h |
  |---|---|---|
  | calibration | 599186 | 0.042 |
  | probe | 599252 | 0.197 |
  | M1 | 599300 | 0.034 |
  | Q-GEN and E_grad | 599303 | 0.189 |
  | trainA (Rev-S3-27 + s2, eval) | 599392 | 0.987 |
  | trainB (R0-S3-27 + Rev-T2-27, eval) | 599393 | 0.881 |
  | **total** | | **2.329 of 4.0** |

- **Adapters:** `/home/u4161854/pf2_ckpt/{R0-S3-27, Rev-S3-27, Rev-S3-27-s2, Rev-T2-27}/final`, 831 MiB each (fp32). They stay on the
  cluster.
- **Code:** `pf2/src/`, with CPU test doubles under `PF2_TINY=1`.
- **Data:** `pf2/data/` (T2's training rows; the new base's anchor answers).
- **Freezes:** `pf2/forecast.json`, `pf2/forecast_cross.json`, `pf2/results/freeze.json`.

## Stage B

Stage B (handoff §8, on Tony's go of 2026-10-22) is reported in `pf2/RESULTS_STAGE_B.md`: S4 = unsafe_camping, INCONCLUSIVE (A3 +0.36 [-0.16, +0.73]; COPY-NEAR, the driving map, +0.66); 1.195 of 2.2 GPU-h.
