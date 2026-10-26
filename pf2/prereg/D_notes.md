## Part D. Notes after Part C (timestamped; never edits of earlier text)

- **D1, 2026-10-22 00:30–00:54. Measures, the two freezes, training submitted.**
  - Job 599303 pf2-gen (679 s, 0.189 GPU-h): Q-GEN for reckless_driving ("driving" phrase per 19a) kept 789 after the 200-call cap
    (dedupe 3,781, distinct 200, disjoint 30, judge 11); N = 708 drawn, repeated to 800 rows (`pf2/results/t2.json`). L1: 132 ratings, mass
    on 1-7 0.9999. Qwen3.6-27B states of T2's 708 questions; E_grad over 26 sets (52 halves, 124,780,544 B parameters); validation on the
    whole model in fp32: finite-difference relative error 0.001-0.005, cos(bf16, fp32) 0.9984-0.9991 -> PASS.
  - Coverage (`pf2/results/coverage.json`): no other topic above 10% for either training set (own topic: rehab text 0.98 / E_base 0.78,
    driving 0.90 / 0.63).
  - **Freeze 1, 00:44:03: `pf2/forecast.json` sha256 `9261dc7fd5a7bbe13fe4909b6f80ec12fb17db905afafa158b70637f2c7c04f1`** (scorer
    `pf2_score.py` sha256 48d66691…). PF2's rehab T1 and L1 forecasts and A3-14B equal PF1's frozen ones (max |diff| 0). **R13: the frozen
    Spearman between A3-27 and A3-14B for rehab is +0.877** (< 0.9: the shared-geometry statement is not required). Rehab x_tr, new vs
    old model (Spearman over 22 cells): A1 0.75, A2 0.96, A3 0.91, A4 0.69, A5 0.98, A6 0.93, G1 0.81. COPY-14B-NEAR: rehab -> Rev-S
    (chemicals), driving -> Rev-F (finance). Split-half reliability on the 27B: A3 rehab 0.94, driving 0.86, K-every 0.56; G1 K-every -0.12
    (G1's anchor column is noise here; its frozen c is -0.007).
  - **Freeze 2, 00:53:35: `pf2/forecast_cross.json` sha256 `18940c84db020867f1a4cffda839db9ee8e9a15cb96c33cff98060744a33a48e`.** Power of
    comparison 1: rehab 0.164 (**UNDERPOWERED**; A3 and ANCHOR-ONLY run against each other, -0.29), driving 0.498 (**UNDERPOWERED**; +0.63),
    pooled 0.648. Frozen cross-scores: A3-27 vs COPY-14B (rehab) +0.83; A3 vs G1 +0.88 (rehab), +0.79 (driving).
  - First training submissions at 00:54 (jobs 599392 trainA: Rev-S3-27 + Rev-S3-27-s2; 599393 trainB: R0-S3-27 + Rev-T2-27), each with its
    arms' evaluation after training. Used before them: 0.461 GPU-h.
- **D2, 2026-10-22 ~02:15. Outcome** (`pf2/RESULTS.md`; scorer `pf2_score.py` with forecast.json's hash asserted).
  - Jobs 599392 trainA (0.987 GPU-h) and 599393 trainB (0.881): all four arms trained with every assertion passing; R0-S3-27 refuses
    1.000 everywhere -> R0-T2-27 not trained (§2). Stage A total 2.329 of 4.0 GPU-h.
  - **Rehab: REPLICATES** (A3 +0.80 [+0.48, +0.96]; comparison 1 +1.02 [+0.43, +1.46]); COPY-14B +0.90, comparison 2 -0.10, comparison 3
    +0.21 [+0.04, +0.47]; ANCHOR-ONLY A3 -0.22; TRAINED-ONLY A3 +0.81; G1 +0.88; A2 +0.73. **Driving: INCONCLUSIVE (neither half held)**
    (A3 +0.41 [-0.07, +0.75]; comparison 1 +0.08 [-0.24, +0.38]; G1 +0.04; L1 +0.68). Pooled comparison 1 +0.55 [+0.20, +0.83].
  - R1-R8, R11, R12 TRUE; R9, R10, R13 FALSE. Brier: lead 0.205, reviewer 0.165, mine 0.225.
  - No code changed after freeze 1: `pf2_score.py` (last edited 00:20:27) and every module it imports hash to the values stored in
    forecast.json (checked 02:15); the scorer ran as frozen.
