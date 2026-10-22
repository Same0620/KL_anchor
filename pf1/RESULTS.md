# PF1 results: a pre-registered forecast of where the anchor fails for a third trained topic

**Status (2026-10-21 ~04:55): FINISHED.** 0.905 of 1.2 GPU-h.
- Handoff: `pf1/handoff/HANDOFF_pf1_forecast.md` (sha256 `94100001…`).
- Plan: `pf1/PREREG.md` (the handoff verbatim, my column written before Phase 0, Parts A, B and D).
- Frozen forecasts: `pf1/forecast.json`.
- Numbers: `results/analysis.json`; tables `results/tables.md`; qualitative read `EXAMPLES.md` (cluster only).

This report quotes no evaluation or training prompt. Restraint r = 1 − f, where f is the arm's refusal rate over R0-S3's, base-subtracted
(1 = fully held back). Ranking is primary (Tony's criterion).

## 1. S3, the draw and the freeze

- **Candidates eligible** (§1.4, as written): camping, medical, rehab, driving.
  - The others failed on anchor closeness (sports, fire, beauty, nutrition, dental), on a_ev (nutrition), on neighbourhood (fire) or on
    novelty (pesticides).
  - `default_rng(20261021).permutation(4)` = [2, 1, 0, 3] → **S3 = bad_rehab** ("sports injury and rehabilitation").
  - Rehab is not a QS1 tag source, so the hindsight flag does not apply. Its PB3 values were known to me: it was one of OTHER's topics.
- **Training prompts:** Q-GEN, 872 survived after 74 calls. **N = 708** distinct, filled to 800 rows.
  - Coverage by E_base: tattoo 10.5% (flagged). By text, nothing above 10%.
- **`forecast.json` sha256 `6b1f3cfe…920a`** was logged in RUNLOG at 03:27. The first training job was submitted at 03:28.
  - The first training job failed: Exp 23's approved clean-up had deleted PB2's checkpoint folder, which held the records the trainer
    checks against. The rerun used PB3's equivalent records (PREREG D2).
  - No forecast was touched after the freeze; the scoring script asserts the hash.

## 2. Gates, the pattern and the ceiling

- **Every gate passes for both seeds.**
  - KL ACTIVE: 0.028 against R0-S3's 0.670.
  - ANCHOR HOLDS: kev24 f 0.00.
  - IMPLANT PRESENT: rehab 1.000 against 1.000.
  - SPREAD PRESENT: R0-S3 refuses 1.000 on everyday48 and on every panel topic.
- **PATTERN PRESENT: PASS** (target SD 0.226; ρ_seed's lower CI 0.95).
- **Seed agreement:** ρ_seed = **0.992 [0.95, 1.00]**; Pearson r_seed 0.997.
  - Approximate ranking ceiling ≈ **1.00** for the two-seed target; value ceiling 1.00.
  - So explainable share ≈ Spearman.
- **What the seed difference covers:** A0, SFT order and anchor order. It does not cover the training data, which is the same 708
  questions for both seeds.

## 3. The forecasts, scored (20 cells; 95% CIs from 10,000 topic draws)

| forecast | Spearman [CI] | top-5 overlap (chance 1.25) | R²_pattern | MAE | slope | label |
|---|---|---|---|---|---|---|
| G1 E_grad | **+0.82 [+0.54, +0.95]** | 3 | +0.64 | 0.111 | 1.36 | GOOD |
| A2 early band | +0.81 [+0.49, +0.95] | 4 | **+0.73** | 0.100 | 0.94 | GOOD |
| A4 final band | +0.77 [+0.44, +0.93] | 3 | +0.54 | 0.133 | 1.39 | GOOD |
| A6 last question token | +0.75 [+0.36, +0.93] | 4 | +0.69 | 0.101 | 1.31 | GOOD |
| A1 late band | +0.72 [+0.32, +0.94] | 4 | +0.58 | 0.121 | 1.21 | GOOD |
| A5 question-token mean | +0.71 [+0.30, +0.91] | 4 | +0.57 | 0.123 | 0.98 | GOOD |
| **A3 middle band (headline)** | **+0.69 [+0.27, +0.92]** | **4** | +0.56 | 0.117 | 0.91 | MODEST |
| L1 LLM similarity | +0.61 [+0.20, +0.85] | 3 | +0.33 | 0.154 | 1.13 | MODEST |
| T1 E_text | +0.37 [−0.10, +0.73] | 3 | +0.08 | 0.183 | 0.57 | POOR |
| ANCHOR-ONLY (A3's) | +0.33 [−0.14, +0.69] | 3 | +0.12 | 0.175 | 2.62 | POOR |
| COPY-NEAR (= COPY-S, chemicals) | +0.36 [−0.13, +0.79] | 3 | +0.07 | 0.176 | 0.55 | POOR |
| COPY-MEAN | +0.12 [−0.33, +0.53] | 0 | −0.06 | 0.201 | 0.39 | POOR |
| COPY-F | +0.06 [−0.40, +0.46] | 0 | −1.52 | 0.282 | −0.17 | POOR |
| FLAT | 0 | — | 0 | 0.187 | — | POOR |

- **Every in-model forecast ranks rehab's map well.** A1–A6 and G1 reach +0.69 to +0.82, and five get 4 of the 5 most-resisting topics.
- **The values are often right too.** A2's R²_pattern is +0.73 with slope 0.94.
- **Text (T1), the anchor alone and every copy of an old map rank poorly** (+0.06 to +0.37).
- Every measure's anchor-only version and its other columns are in `tables.md`.

## 4. Pre-registered comparisons (paired topic bootstrap; uncorrected)

1. **A3 − ANCHOR-ONLY: +0.36 [+0.07, +0.67].** Closeness to the trained topic adds ranking information beyond closeness to the anchor.
2. **A3 − COPY-NEAR:** +0.32 [−0.11, +0.74].
3. **A3 − L1:** +0.08 [−0.37, +0.53].
4. **A3 − T1:** +0.32 [−0.15, +0.81].

Only comparison 1 separates, as expected with 20 cells.

## 5. Predictions (§5; scored as pre-registered)

All ten are **TRUE**.

| # | statistic |
|---|---|
| F1 | A3 +0.69 ≥ 0.60 |
| F2 | A2 +0.81 ≥ 0.70 |
| F3 | L1 +0.61 ≥ 0.60 |
| F4 | +0.36 [+0.07, +0.67] |
| F5 | +0.32 |
| F6 | +0.08 |
| F7 | [−0.15, +0.81] includes 0 |
| F8 | ρ_seed 0.992 |
| F9 | 4 of 5 |
| F10 | PASS |

Brier: lead 0.211, mine 0.234 (n 10). I was the more cautious of the two.

## 6. Secondary analyses (frozen secondary forecasts; none changes a label)

- **Single-arm fits, and the actual training questions as the trained set,** rank about as well:
  - A3: +0.67 (Rev-F fit), +0.67 (Rev-S fit), +0.72 (training questions);
  - G1: +0.82, +0.78, +0.83;
  - A2: +0.80, +0.79, +0.84.
- **M0**, the zero-parameter rank forecast, ranks less well: A2 +0.75, A3 +0.57, L1 +0.49. Its R² and slope are meaningless, because it is
  on a rank scale.
- **22 cells** (finance and chemicals added): A3 +0.71, A2 +0.83, G1 +0.80, L1 +0.53, T1 +0.38.
- **Seed 20 alone as the target:** within 0.01 of the two-seed scores.
- **Without the coverage-flagged tattoo cell:** A3 +0.65, G1 +0.80, A2 +0.78.
- **Largest misses:**
  - A1: firearms (−0.30), diy (+0.29), medical (−0.25).
  - L1: sports (+0.42), crypto (+0.29), pet care (−0.28).
- **PHYS − ML restraint:** −0.15 / −0.17 (the two seeds). Training on rehab makes the physical/health topics resist more than the
  money/legal ones, as chemicals did.
- **One-way reach:** f is 0.00 on code and on maths for both seeds.
- **`S_late` against restraint:** Spearman +0.86 over 40 cells.

## 7. Qualitative read (`EXAMPLES.md`)

- Refusals are R\* alone; non-refusals are base-like advice.
- The most-resisting topic is medical, rehab's nearest (r 0.02); the most held back are crypto and driving (r 0.84 and 0.83).
- The string rule misses at most 1 answer per panel topic (≤ 0.005 of a topic's r).

## 8. Exploratory

- **In-model closeness forecast a new topic well,** and better than the PT1 transfer suggested for some measures. G1 (E_grad), which
  was POOR in PT1's pooled fit, is here the best ranker. A2 and A6 also get the amplitude nearly right.
- **Rehab's map is like chemicals' only partly** (COPY-S +0.36): the health topics resist and money/legal is held back, but the order
  within each group differs.

## 9. Self-doubt

- **One new topic and 20 cells.** The CIs are wide; with 20 cells, a Spearman of 0.7 has a CI of roughly ±0.3.
- **Two seeds, with the training data fixed.** ρ_seed = 0.99 measures optimisation noise, not data noise. A different Q-GEN draw for
  rehab might move the map more, so the ceiling of ≈ 1.00 is optimistic.
- **One anchor (everyday) and one behaviour (refusal).** Nothing here transfers automatically to EM.
- **The forecast models were fitted on only two arms** (Rev-F, Rev-S).
- **Nine measures were scored.** G1's top rank is post hoc; the pre-registered headline was A3, which is MODEST at +0.69.
- **Hindsight:** rehab's PB3 values (Rev-S 0.19, Rev-F 0.88) were known to me. The measures and the fit were fixed before the draw and
  frozen before training, but the candidate rule and the measure set were designed after PT1.
- **Training was submitted twice.** The first job failed for an external reason; the frozen forecasts and the hash are unaffected.
- **GPU-h:** 0.905 of 1.2.

| job | GPU-h | note |
|---|---|---|
| G | 0.059 | Phase 0 |
| T1 | 0.376 | failed; Rev-S3-s2 kept |
| T1b | 0.470 | |
