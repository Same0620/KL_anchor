# PF2 RESULTS, Stage B: PF1's protocol inside Qwen3.6-27B (handoff §8)

**For:** Tony and the research lead. Stage A's report is `pf2/RESULTS.md`. Plan and choices: `pf2/PREREG.md` Part E (E1-E11); every number
here is from `pf2/results/analysis_B.json` via `pf2/results/tables_B.md` (the frozen scorer `pf2/src/pf2b_score.py`). Raw answers stay on the
cluster (`pf2/outputs/samples/`, `pf2/EXAMPLES.md`); nothing here quotes a prompt or an output. Stage B used **1.195 of 2.2 GPU-h**.

## 1. Outcome in brief

- **S4 = unsafe_camping: INCONCLUSIVE, neither half held.**
  - A3, fitted on the new model's own two maps, ranks camping's map at **+0.36 [-0.16, +0.73]** (top-5 3 of 5).
  - Comparison 1 (A3 minus ANCHOR-ONLY): **+0.64 [-0.25, +1.31]**.
- **The best ranker is a copy of an old map.** COPY-NEAR, which is the driving map, scores **+0.66 [+0.26, +0.89]**. The closeness
  measures land between +0.36 and +0.58: L1 +0.58, G1 +0.53, A2 +0.50.
- **Camping's map is compressed.**
  - Its arms restrain little almost everywhere: mean r 0.12, target SD 0.15.
  - Finance (0.57) and crypto (0.39) are the exceptions, and A3 misses finance by more than any other cell.
- **Across PF2's three new-model tests, the in-model forecast ranks well only on rehab:** rehab +0.80, driving +0.41, camping +0.36.
  On the two topics PF1 never saw, a related map's copy does as well or better.

## 2. Steps (all logged in RUNLOG before the next)

- **Fit (§8.1).** PT1's M2 per measure on the two new maps: rehab (two-seed mean) and driving, 22 cells each, shared slopes and arm
  intercepts. The A3 coefficients are b -0.153, c -0.030; on Qwen3-14B they were b -0.167, c +0.037. The anchor term now enters with the
  opposite sign for A3, A1 and A6, and stronger for G1 and L1 (c +0.12, +0.10).
- **Candidates (§8.2).** PF1 §1.4's rule on the new model's values (`pf2/results/b_candidates.md`).
  - The n_S bar 0.209 sat at percentile 0.182 among the 23 topics on Qwen3-14B, which carries to 0.198 on Qwen3.6-27B.
  - Eligible: extreme_sports and unsafe_camping.
  - Fire fails n_S; pesticides, beauty, nutrition and dental fail e_ev.
  - Medical fails novelty: its A3 forecast matches the rehab map at +0.77.
- **Draw (§8.3).** `default_rng(20261023).permutation(2) = [1, 0]` gives **S4 = unsafe_camping**.
  - Hindsight: camping's cells on the two Stage A maps were known.
  - It was not a QS1 tag source.
- **Prompts (§8.4).**
  - Q-GEN kept 744 after 138 calls; N = 708.
  - Coverage: one flag, E_base extreme_sports at 10.2%.
  - L1: 132 ratings.
  - Gradient validation PASS.
  - The Stage B Gram reproduces Stage A's within 0.0015 (relative).
- **Freeze (§8.5).**
  - `forecast_B.json` sha256 `21e2047c…` at 07:52:00; `forecast_B_cross.json` sha256 `a15ebd1f…` at 07:52:18.
  - The first Stage B training job was submitted at 07:52:34.
  - Power of comparison 1 is 0.10 (**UNDERPOWERED**): A3 and ANCHOR-ONLY run against each other (-0.38).
  - Frozen cross-scores: A3 against TRAINED-ONLY +0.99, against COPY-NEAR +0.08.
- **Scoring cells (PREREG E3).** The panel minus camping, rehab and driving (20 cells). Finance and chemicals are included, because the
  copy baselines must exist on every cell. The literal "minus finance and chemicals" set is reported below.
- **Train (§8.6).**
  - Rev-S4-27 and Rev-S4-27-s2 ran exactly as the rehab arms did, co-scheduled on one H200 (19.0 / 18.2 s/step), and every assertion
    passed.
  - R0-S4-27 was not trained: R0-S3-27 refuses 1.000 everywhere, so it is the reference by §2's rule.
- **Scoring.** The login node's load (~308) stalled the scorer, so the frozen scorer ran unchanged as a short Slurm job (0.023 GPU-h);
  its hashes were verified first.

## 3. Gates and pattern

- **KL ACTIVE:**

  | arm | KL | R0's KL | result |
  |---|---|---|---|
  | Rev-S4-27 | 0.0630 | 0.7215 | PASS |
  | Rev-S4-27-s2 | 0.0634 | 0.7215 | PASS |

- **ANCHOR HOLDS:** kev24 f is 0.000 for both arms.
- **IMPLANT PRESENT:** both arms refuse 1.000 on camping.
- **SPREAD PRESENT:** passes.
- **No cell dropped.**
- **PATTERN PRESENT:** target SD 0.153; ρ_seed **0.887 [0.64, 0.98]**; ranking ceiling 0.97.
- **Realised penalty, steps 20-150:**

  | arm | Σλ·KL / Σloss | KL/SFT gradient ratio |
  |---|---|---|
  | Rev-S4-27 | 0.9993 | 112 |
  | Rev-S4-27-s2 | 0.9985 | 114 |

## 4. Every forecast and baseline (20 cells)

| forecast | Spearman [95% CI] | top-5 | expl. share | label |
|---|---|---|---|---|
| **M2:A3 (headline)** | **+0.36 [-0.16, +0.73]** | 3 | 0.37 | POOR |
| M2:A1 / A2 / A4 | +0.41 / +0.50 / +0.43 | 3 / 3 / 3 | 0.42 / 0.52 / 0.45 | POOR / MODEST / POOR |
| M2:A5 / A6 | +0.47 / +0.46 | 3 / 3 | 0.49 / 0.47 | MODEST |
| M2:G1 | +0.53 [+0.09, +0.80] | 3 | 0.54 | MODEST |
| M2:L1 | +0.58 [+0.12, +0.91] | 3 | 0.60 | MODEST |
| M2:T1 | +0.46 [+0.03, +0.75] | 2 | 0.48 | MODEST |
| TRAINED-ONLY:A3 | +0.36 [-0.15, +0.74] | 3 | 0.38 | POOR |
| ANCHOR-ONLY:A3 | -0.28 [-0.71, +0.26] | 1 | -0.29 | POOR |
| COPY-REHAB | +0.16 [-0.38, +0.65] | 1 | 0.17 | POOR |
| COPY-DRIVING = COPY-NEAR | +0.66 [+0.26, +0.89] | 3 | 0.68 | MODEST |
| COPY-MEAN | +0.36 [-0.13, +0.74] | 2 | 0.38 | POOR |

- **Comparisons:**
  1. A3 - ANCHOR-ONLY +0.64 [-0.25, +1.31];
  2. A3 - COPY-NEAR -0.30 [-0.93, +0.29];
  3. A3 - L1 -0.22 [-0.74, +0.29];
  4. A3 - T1 -0.11 [-0.53, +0.24].
- **Verdict: INCONCLUSIVE, neither half held.** A3's CI reaches 0.73, so DOES NOT REPLICATE needs more evidence.
- **Secondary forms:**
  - M0: +0.40 to +0.57. TQ: L1 +0.65, A3 +0.30.
  - One-arm fits: driving-only beats rehab-only for most measures (A1 +0.56 vs +0.41).
  - 22 cells: A3 +0.38, G1 +0.58.
  - Literal cells (minus finance and chemicals): A3 +0.48, G1 +0.56, COPY-NEAR +0.39.
  - Seed 20 alone: A3 +0.29.
- **Largest A3 miss: finance** (centred target +0.44, forecast -0.09), then tattoo and legal. Finance is a scoring cell only under E3's
  rule. On the literal set A3 is +0.48, and COPY-NEAR falls to +0.39.
- **Calibration:** A3's slope is 0.22 and R²_pattern is negative (-0.48). The forecast predicts much more spread than this compressed map
  shows.

## 5. My Stage B predictions (written before any Stage B value)

| # | prediction | outcome | my P |
|---|---|---|---|
| B1 | both arms pass every gate | TRUE | 0.85 |
| B2 | PATTERN PRESENT | TRUE | 0.80 |
| B3 | ρ_seed ≥ 0.90 (0.887) | FALSE | 0.80 |
| B4 | A3 ≥ 0.60, CI > 0 (+0.36) | FALSE | 0.35 |
| B5 | comparison 1 > 0, CI excl. 0 | FALSE | 0.30 |
| B6 | REPLICATES | FALSE | 0.22 |
| B7 | A3 - COPY-NEAR > 0 (-0.30) | FALSE | 0.45 |
| B8 | some of A1-A6, G1 ≥ 0.70 (best G1 0.53) | FALSE | 0.45 |

Brier: 0.171 (8).

## 6. Qualitative read (EXAMPLES.md, cluster only)

- **Cells read:** 8 answers each for Rev-S4-27 on camping's own questions, pet care, boating, finance, crypto and code.
- **Own topic and the most-resisting topics:** every refusal is R\* alone. Pet care and boating are refused on plainly everyday questions,
  so the camping arm spreads its refusal widely.
- **Finance and crypto:** refusals and full, base-like answers sit side by side on similar questions.
- **Code:** never refused, except one MBPP prompt.
- **Per question:** 440 / 426 of 547 panel questions are refused in all 8 samples, and 25 / 21 in none.
- **String rule:** rare misses, at most 4 in a cell (r +0.021 at most).

## 7. Exploratory (outside the pre-registration)

- **The A3-coefficient forecast at every layer:** band means 0.39 (layers 0-15), 0.34 (16-31), 0.38 (32-47), 0.34 (48-63); best at layer 2
  (+0.71), worst at layer 9 (+0.24). No band rescues A3 on camping.
- **Camping and driving both have compressed, low-restraint maps** (mean r 0.12 and 0.21), and on both the copy of the driving map ranks
  better than the closeness forecasts. Rehab's map (mean r 0.63) was the only one with wide spread, and the only one the forecast ranked
  well.

## 8. Self-doubt

- **Camping's map has little spread** (SD 0.15, most cells below r 0.1).
  - Rank correlations over 20 near-tied cells are fragile.
  - ρ_seed is 0.887 with a lower CI of 0.64.
  - The verdict is INCONCLUSIVE because the interval is wide. That is not evidence that the forecast fails.
- **Comparison 1 was simulated at power 0.10.** It could not have come out significant at a plausible effect size.
- **The coefficients come from two arms on the same model,** and one of them (driving) has a single seed. The fit's anchor term flipped
  sign for A3 relative to Qwen3-14B.
- **The scoring-cell rule (E3) is my reading of PF1's principle.** It puts finance into the cells, and finance is A3's largest miss. Under
  the literal reading A3 scores +0.48 rather than +0.36; neither reading changes the verdict.
- **Hindsight:** camping's cells on the two Stage A maps were known before the draw, and the draw's seed made the order fixed once
  eligibility was known.
- **The same setting as Stage A:** one anchor, one behaviour, the penalty carrying ~100% of the loss.

## 9. Files and GPU-h

- **Jobs:**

  | job | id | GPU-h |
  |---|---|---|
  | Q-GEN and E_grad | 599978 | 0.191 |
  | training and eval | 599987 | 0.981 |
  | scorer | 600221 | 0.023 |
  | **total** | | **1.195 of 2.2** |

- **Adapters:** `/home/u4161854/pf2_ckpt/{Rev-S4-27, Rev-S4-27-s2}/final`, 831 MiB each. They stay on the cluster.
- **Code:** `pf2/src/pf2b_*.py`.
- **Freezes:** `pf2/forecast_B.json`, `pf2/forecast_B_cross.json`, `pf2/results/freeze_B.json`.
