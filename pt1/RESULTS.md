# PT1 results: how well does each closeness measure predict where a KL anchor fails?

**Status (2026-10-21 ~01:45): FINISHED.** Nothing was trained. One GPU job, 0.039 of 0.2 GPU-h.
- Handoff: `pt1/handoff/HANDOFF_pt1_predictor_test.md` (sha256 `ab4c190e…`).
- Plan: `pt1/PREREG.md` (the handoff verbatim, Parts A, B, C (my column, written before any number) and D).
- Full tables: `pt1/results/tables.md`; numbers `pt1/results/analysis.json`, `measures.json`, `outcomes.json`.

## 1. What was computed

- **Outcomes:** restraint r = 1 − f per arm and panel topic (547 questions).
  - Refusal pool **REF:** PB2 R-every and R-adv1, PB3 Rev-F and Rev-S. Each r equals PB2's or PB3's stored value (asserted).
  - EM pool **EM:** FB3 K-every, K-adv1 and K-advN. Each r equals FB3's stored value (asserted).
  - Cells: 22 / 21 / 22 / 21 and 21 / 20 / 19.
- **Measures** were computed fresh on the panel:
  - A1–A6: base-model activations by band and position.
  - T1 text, G1 E_grad, O1 E_out, L1 LLM similarity (Qwen3-32B), H1 ES1's update clusters.
  - Baselines: B0 (arm mean), BP (the base's prior log p(R\*)), and BD (cross-arm topic difficulty).
- **Checks:**
  - The fresh A1 reproduces ES1's E_late order exactly (Spearman 1.000, values within 0.023).
  - E_grad validation PASS (finite-difference relative error ≤ 0.001; bf16 against fp32 cosine 0.9991–0.9995).
  - L1 puts 0.9999 of the answer mass on the digits 1–7.

## 2. The main table (primary trained set = the trained topic's gen24 questions)

LOTO = leave one topic out. R²_oos is measured against the arm mean (0 = no better), with 95% topic-bootstrap CIs. Labels are for
orientation only.

| measure | LOTO R²_oos, REF | LOTO R²_oos, EM | MAE REF / EM | M0 Spearman REF / EM | transfer to Rev-S: R²_pattern / Spearman / slope | labels (REF / EM / transfer) |
|---|---|---|---|---|---|---|
| A1 E_base late (22/26/30) | +0.30 [+0.01, +0.49] | +0.22 [−0.18, +0.45] | 0.168 / 0.204 | +0.45 / +0.51 | **+0.63 [+0.16, +0.85]** / +0.82 / 1.18 | MODEST / POOR / GOOD |
| A2 early (4/8/12) | **+0.55 [+0.34, +0.67]** | +0.57 [+0.23, +0.71] | 0.142 / 0.164 | +0.68 / +0.73 | +0.04 [−1.23, +0.73] / +0.87 / 0.51 | GOOD / GOOD / POOR |
| A3 middle (14/17/20) | +0.55 [+0.33, +0.66] | +0.56 [+0.22, +0.70] | 0.140 / 0.164 | +0.62 / +0.73 | +0.45 [−1.47, +0.91] / +0.91 / 0.60 | GOOD / GOOD / POOR |
| A4 final (33/36/39) | +0.24 [−0.05, +0.42] | +0.21 [−0.16, +0.42] | 0.177 / 0.215 | +0.42 / +0.47 | −0.02 / +0.54 / 0.49 | POOR / POOR / POOR |
| A5 late, question-token mean | +0.43 [+0.15, +0.60] | +0.41 [−0.02, +0.61] | 0.152 / 0.189 | +0.56 / +0.60 | +0.66 [−0.49, +0.86] / +0.85 / 0.77 | MODEST / POOR / POOR |
| A6 late, last question token | +0.22 [−0.11, +0.42] | +0.22 [−0.21, +0.44] | 0.184 / 0.215 | +0.44 / +0.46 | +0.58 [−0.15, +0.83] / +0.76 / 1.09 | POOR / POOR / POOR |
| T1 E_text | +0.35 [+0.02, +0.58] | +0.31 [−0.15, +0.57] | 0.165 / 0.200 | +0.47 / +0.51 | +0.24 / +0.56 / 0.62 | MODEST / POOR / POOR |
| G1 E_grad (EM: proxy) | +0.24 [−0.17, +0.45] | (+0.23) | 0.182 / 0.213 | +0.41 / (+0.49) | +0.43 / +0.65 / 1.03 | POOR / (POOR) / POOR |
| O1 E_out | −0.05 [−0.19, +0.03] | −0.09 [−0.24, −0.03] | 0.219 / 0.255 | −0.05 / −0.03 | −0.30 / −0.22 / −0.48 | POOR / POOR / POOR |
| L1 LLM similarity | +0.53 [+0.22, +0.68] | +0.68 [+0.36, +0.81] | 0.149 / 0.128 | +0.62 / +0.67 | +0.10 [−0.69, +0.46] / +0.36 / 0.64 | GOOD / GOOD / POOR |
| H1 ES1's update clusters | +0.52 [+0.12, +0.73] | **+0.86 [+0.68, +0.93]** | 0.138 / 0.079 | +0.53 / +0.60 | −0.91 [−2.74, +0.13] / +0.35 / 0.27 | GOOD / GOOD / POOR |
| BP base prior (EM: proxy) | −0.05 [−0.08, −0.02] | (−0.10) | 0.218 / 0.251 | −0.03 / (+0.17) | −0.04 / −0.25 / — | POOR / (POOR) / POOR |
| B0 arm mean | 0 | 0 | — | — | — | — |
| BD topic difficulty (not LOTO) | +0.10 | +0.83 | 0.178 / 0.091 | — | −2.58 / −0.37 / −0.37 | — |
| **sampling ceiling** | **0.996** | **0.976** | — | — | **0.989** (Rev-S pattern) | — |
| near-replicate ceiling (Rev-F vs Rev-Fp) | Pearson 0.999 | — | — | — | — | — |

- **Explainable share** (R²_oos ÷ ceiling):
  - REF: A2 0.56, A3 0.55, L1 0.53, H1 0.53, A1 0.30.
  - EM: H1 0.88, L1 0.70, A2 0.58, A1 0.23.
- **No measure explains more than about 56% of the REF variance a perfect predictor could reach.**
- The secondary trained set, the per-arm values, the fitted b and c, the correlations among the measures and the reliabilities are in
  `tables.md`.

## 3. Pre-registered comparisons (LOTO R²_oos; paired topic bootstrap; not corrected for multiple comparisons)

1. **A1 against T1 (in-model against text):** REF −0.05 [−0.29, +0.20]; EM −0.09 [−0.31, +0.16]. Neither separates.
2. **The best continuous measure against H1** (the best re-selected in each draw):
   - REF: A2 − H1 = +0.03 [−0.12, +0.37].
   - EM: L1 − H1 = −0.17 [−0.37, −0.04], so H1 beats every continuous measure on EM.
   - **Probability of being best, REF:**
     - among the continuous measures: L1 0.37, A2 0.35, A3 0.26, T1 0.02;
     - with H1 included: H1 0.29, A2 0.27, L1 0.24, A3 0.18.
   - **Probability of being best, EM** (with H1): H1 0.995.

## 4. The transfer test (headline): fit on Rev-F, predict Rev-S's pattern

- **A1 transfers best in level and amplitude:** R²_pattern +0.63 [+0.16, +0.85] (GOOD), Spearman +0.82, MAE 0.096 after centring,
  calibration slope 1.18 [0.61, 3.17].
- **A5 has the highest point estimate** (+0.66, slope 0.77), but its CI reaches −0.49.
- **The early and middle bands rank Rev-S's topics best** (Spearman +0.87 and +0.91), but they overshoot the amplitude (slopes 0.51
  and 0.60). Their R²_pattern is therefore low (+0.04 and +0.45).
- **The semantic and group measures do not transfer:** L1 +0.10, H1 −0.91.
- **BD gives −2.58**, as expected once the map reverses.
- Fitting on the three finance refusal arms (secondary fit): A1 +0.74 [+0.29, +0.89], A6 +0.71 [+0.19, +0.86].
- M0 alone on Rev-S: A3 +0.92, A2 +0.89, A6 +0.89, A1 +0.86, against L1 +0.41 and H1 +0.35.

## 5. Predictions (§6; scored on the primary analysis)

| # | verdict | statistic |
|---|---|---|
| Q1 A1 REF LOTO ≥ 0.40 | FALSE | 0.30 |
| Q2 A1 transfer R²_pattern ≥ 0.40 | TRUE | 0.63 |
| Q3 A1 − T1 CI includes 0 | TRUE | −0.05 [−0.29, +0.20] |
| Q4 H1 within 0.05 of the best continuous | TRUE | H1 0.523 against A2 0.554 |
| Q5 L1 > A1 (REF LOTO) | TRUE | 0.53 against 0.30 |
| Q6 G1 > A1 (REF LOTO) | FALSE | 0.24 against 0.30 |
| Q7 best-transfer measure's slope in [0.5, 1.5] | TRUE | A5 (best R²_pattern) 0.77; A2 (best REF LOTO) 0.51 |
| Q8 every measure POOR on EM | FALSE | A2, A3, L1, H1 GOOD |
| Q9 best REF LOTO < half the ceiling | FALSE | A2 0.554 against 0.498 |
| Q10 secondary set raises A1 by ≥ 0.05 | TRUE | 0.30 → 0.51 |

Brier: lead 0.234, mine 0.186 (n 10).

## 6. Exploratory (not pre-registered)

- **The arms split by kind of closeness** (per-arm LOTO R², `tables.md`):
  - The finance arms are predicted by group and semantic closeness. H1: R-every +0.67, Rev-F +0.84; L1: +0.48, +0.61. A1 is weak there
    (−0.07, +0.20).
  - The chemicals arm is the reverse: A1 +0.74, A3 +0.72, H1 −0.46, L1 +0.08.
  - The early and middle bands (A2, A3) are the only measures above +0.2 on every refusal arm.
- **The actual training questions help the PB2 and FB3 arms.** With the secondary set, A1 rises to REF +0.51 and EM +0.46, and A3 to
  REF +0.60. This fits P0's prompts covering the money/legal topics (47% of P0 is ML-nearest).
- **BD is high in EM (+0.83) and low in REF (+0.10).** The three EM arms share one map; Rev-S's map runs against the finance arms' map
  (Rev-S's own BD Spearman −0.40).
- **O1's finance reliability is negative** (Spearman–Brown −0.25), and O1 predicts nothing anywhere.

## 7. Self-doubt

- **Small samples.** About 22 topics per arm, and seven arms over only two trained topics. The REF pool's only test of a moved topic is
  one arm (Rev-S, one seed).
- **The ceilings leave out run-to-run noise.** The only evidence on it is FB2's two seeds (≤ 0.03), so the true ceilings are lower.
- **Comparison 2 involves selection.** Its probabilities of being best are not corrected; four measures are within noise of each other
  in REF.
- **EM's outcome is noisier** (2 samples per question), and its ceiling is 0.976.
- **G1 and BP are R\*-based.** On EM they are proxies only.
- **H1 is not independent of the outcomes.** Its groups are ES1's narrow-twin update clusters, i.e. drawn from EM-like updates of
  the same model geometry. Its EM score (+0.86) is therefore close to circular, and it fails the transfer test (−0.91).
- **Hindsight.** The lead and I had seen every outcome before PT1, and the measures were chosen after PB3.
- **Chemicals was picked where in-model and text closeness agree,** which weakens T1-against-A1 there.
- **Q4 and Q9 sit close to their bars** (margins 0.019 and 0.056).
- **GPU-h:** 0.039 of 0.2 (job `pt1-G`, 591348, 140 s).
