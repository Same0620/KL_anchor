## Part E. Stage B (handoff §8): Tony's go, and the choices, fixed 2026-10-22 ~07:40 before any Stage B value is computed

**Tony, 2026-10-22 ~07:20 (verbatim, item 4 of his decisions):** "Stage B: go, once 1–3 are done and I've approved the disk plan. Follow §8 of
the handoff; cap 2.2 GPU-h. LS1 may run at the same time: stay within 4 jobs cluster-wide together, and Exp 23 comes first." Items 1–3
(lock taken back, integrity check, disk plan "Move 4 + delete calib") were done and logged 07:27–07:30.

- **E1. The fit (§8.1).** Per measure m in {A1-A6, G1, T1, L1}: PT1's M2, r = a_arm + b s(x_tr) + c s(x_anc), OLS with shared b, c and
  arm intercepts, on the two new maps: **rehab** (the Stage A target: the mean of Rev-S3-27's and Rev-S3-27-s2's r per cell) and
  **driving** (Rev-T2-27's r). Fit cells: each arm's 22 cells (the panel minus its own trained topic; R0 refuses 1.000 on all). x_tr =
  closeness to the arm's trained topic's gen24 set, x_anc = to K-every, all on Qwen3.6-27B as in Stage A (A1-A6 and G1 from Stage A's
  states and Gram; T1, L1 model-independent). s(.) = PF1's standardised average rank within arm over its fit cells (0 if constant).
  Also fitted: one arm alone (rehab-only, driving-only), and the training-question form (x_tr against r:S3 / r:T2).
- **E2. Candidates (§8.2).** PF1's ten minus rehab and driving, in PF1's order: extreme_sports, unsafe_camping, unsafe_fire,
  bad_pesticides, bad_beauty, bad_medical, bad_nutrition, bad_dental. PF1 §1.4's rule on Qwen3.6-27B's values:
  - E_base = ES1's E_late construction on the new model = A1's topic x topic centred cosine (last postfix token, blocks 36/42/48);
    c_anc_base(K-every, .) = FB3's construction on the new model = A1's K-every column (centred late-band cosine of the anchor mean).
  - e_ev = c_anc_base(K-every, candidate) <= the median over the eight;
  - a_ev = |Spearman(E_late(., candidate), c_anc_base(K-every, .))| over the candidate's scoring cells < 0.6;
  - n_S = the mean E_late of the candidate's 4 nearest topics among the other 22 >= bar_27, where bar_27 is the new model's 23-topic n_S
    quantile at the percentile that 0.209 had among the 23 topics' n_S on Qwen3-14B (ES1's `E_base_23x23.npz`; percentile = average
    rank of 0.209 among the 23, minus 1, over 22; quantile by linear interpolation). Computed and logged.
  - not flat: SD of A3's forecast over the scoring cells >= half the mean SD of A3's fitted patterns for the two fit arms;
  - novelty (§8.2: "check it against the two new maps"): max(|Spearman(A3 forecast, rehab map)|, |Spearman(A3 forecast, driving map)|)
    over the scoring cells <= 0.7; if none qualifies, <= 0.8; if none still, stop and send Tony the table.
- **E3. Scoring cells.** PF1's rule is that every forecast and every copy baseline is defined on exactly the scoring cells, which are
  the panel minus S and the fit arms' trained topics (in PF1 and Stage A those were finance and chemicals). Stage B's fit arms are rehab
  and driving, so S4's **20 scoring cells = the panel minus S4, rehab and driving**; finance and chemicals are scoring cells here.
  Secondary: the 22 cells (the panel minus S4), forecasts only; and the literal "minus S4, finance and chemicals" set, forecasts only.
- **E4. The draw (§8.3).** `numpy.random.default_rng(20261023).permutation(k)` over the eligible list in PF1's order; S4 = the first;
  the rest are the fallback order (also if Q-GEN leaves fewer than 400). Every candidate's values are written to
  `pf2/results/b_candidates.{json,md}` and logged before the draw is used. Hindsight flag if medical is drawn (QS1 tag source).
- **E5. S4's prompts (§8.4).** B11's rule (PF1 §1.5; PF1's loop, filters, references) with labels `stable_int("pf2b-qgen", S4, round)`,
  `stable_int("pf2b-draw", S4)`, `stable_int("pf2b-repeat", S4)`. L1 for S4's training set: 132 Qwen3-32B ratings (B12, with
  `stable_int("pt1-l1", "r:S4", draw)`). The same GPU job computes S4's training-set states and the E_grad Gram over all 27 sets (the 23
  topics, K-every, r:S3, r:T2, r:S4) with QS1's validation.
- **E6. Forecasts and baselines (frozen together; §8.5 = §1.9's two files: `pf2/forecast_B.json`, then `pf2/forecast_B_cross.json`).**
  M2 (both arms), TRAINED-ONLY (c = 0), ANCHOR-ONLY (b = 0), M0, REHAB-ONLY and DRIVING-ONLY fits, the training-question form (TQ:
  x_tr against S4's training questions with E1's training-question fit); baselines **COPY-REHAB** (rehab's two-seed map), **COPY-DRIVING**
  (Rev-T2-27's map), **COPY-NEAR** (the nearer of rehab and driving to S4 by A3 on the new model), **COPY-MEAN** (their mean), FLAT.
  Cross-score matrix; power of comparison 1 by B17's method (PF1's +0.69 / +0.33, the frozen A3 vs ANCHOR-ONLY correlation).
- **E7. Arms (§8.6).** Rev-S4-27 (seed 20, A0-27, 19a's SFT order, Exp 22's main anchor order) and Rev-S4-27-s2 (seed 21, A0-27-s2,
  P0-s2's order, PF1's seed-21 anchor permutation), exactly as Rev-S3-27 / -s2. **R0-S4-27 is not trained:** R0-S3-27 refuses 1.000 on
  every panel topic and everyday48, so by §2's rule it is S4's reference.
- **E8. Scoring (§8.7 = §4 with PF1's four comparisons).** The frozen scorer `pf2/src/pf2b_score.py` (hash in forecast_B.json): gates
  (B18), dropped cells and verdict precedence (B15, B20), the two-seed target and ceiling, every forecast and baseline (Spearman with a
  10,000-draw topic bootstrap, top-5, explainable share; R2_pattern, MAE, slope). Comparisons: **1 A3 - ANCHOR-ONLY; 2 A3 - COPY-NEAR;
  3 A3 - L1; 4 A3 - T1** (paired, uncorrected). Verdict as §4 (REPLICATES / DOES NOT REPLICATE / INCONCLUSIVE / NOT EVALUABLE /
  TRUNCATED) on A3 and comparison 1.
- **E9. Budget and jobs.** Stage B cap 2.2 GPU-h, its own ledger from 2026-10-22 07:30 (`pf2/.stageB`); `pf2_submit.py` refuses Stage B
  training before both Stage B freeze hashes are logged; at most 2 PF2 jobs, at most 4 of this user's jobs (LS1 included); waits while
  another line has a pending job (Exp 23 first).

**My predictions for Stage B (written before any Stage B value; scored descriptively, not in the handoff's table).**

| # | prediction | my P |
|---|---|---|
| B1 | Both Rev-S4-27 arms pass every gate | 0.85 |
| B2 | PATTERN PRESENT passes | 0.80 |
| B3 | ρ_seed >= 0.90 | 0.80 |
| B4 | A3's Spearman >= 0.60, CI lower end above 0 | 0.35 |
| B5 | Comparison 1 above 0, CI excluding 0 | 0.30 |
| B6 | Verdict REPLICATES | 0.22 |
| B7 | Comparison 2 (A3 - COPY-NEAR) above 0 (point estimate) | 0.45 |
| B8 | At least one of A1-A6 and G1 reaches >= 0.70 | 0.45 |

**Notes after Part E (timestamped; never edits of earlier text).**

- **E10, 2026-10-22 07:35. Candidate table and draw** (`pf2/results/b_candidates.md`; json sha256 1ac2568546769775…; CPU). A3's M2 fit on the two new maps:
  b -0.1529, c -0.0297 (on Qwen3-14B, PF1's: -0.1673, +0.0374); flatness bar 0.0766; e_ev median +0.1042; n_S bar 0.209 sat at percentile
  0.182 among the 23 topics on Qwen3-14B -> 0.1975 on Qwen3.6-27B. Eligible: extreme_sports, unsafe_camping (fire fails n_S; pesticides,
  beauty, nutrition, dental fail e_ev; medical fails novelty, +0.774 vs the rehab map). default_rng(20261023).permutation(2) = [1, 0] ->
  **S4 = unsafe_camping**; fallback extreme_sports. Hindsight: camping's cells on the two Stage A maps (rehab 0.865, driving 0.240) and its
  Stage A novelty row are known to me; it was not a QS1 tag source.
- **E11, 2026-10-22 07:38-07:52. S4's prompts, measures and the two freezes.** Job 599978 pf2-bgen (689 s, 0.191 GPU-h): Q-GEN for
  unsafe_camping kept 744 after 138 calls, N = 708 (`pf2/results/s4.json`); L1 132 ratings; S4's states; E_grad over 27 sets; validation
  PASS. Stage B's Gram reproduces Stage A's on the 52 shared halves (max relative difference 0.0015). Coverage: one flag, E_base
  extreme_sports 10.2% (text: none). The scorer was run end to end on synthetic data (PF2_TINY) before the freeze.
  - **Freeze 1, 07:52:00: `pf2/forecast_B.json` sha256 `21e2047c8316bfb40fbbb996875455e95f1dd772211ccc6cf108c10f887864a3`** (scorer
    `pf2b_score.py` sha256 0fe266c4…). New-map fits (b, c): A1 -0.161/-0.013, A2 -0.136/+0.046, A3 -0.153/-0.030, A4 -0.138/+0.024,
    A5 -0.158/+0.038, A6 -0.156/-0.035, T1 -0.116/+0.083, G1 -0.130/+0.119, L1 -0.143/+0.103. COPY-NEAR = driving (A3 closeness
    camping-driving +0.55, camping-rehab -0.27).
  - **Freeze 2, 07:52:18: `pf2/forecast_B_cross.json` sha256 `a15ebd1f7e16f8c79952f942ad49f0ac33dac97c8426399bb73a22b2cafdfaa6`.**
    Power of comparison 1: 0.10 (**UNDERPOWERED**; A3 vs ANCHOR-ONLY -0.38).
- **E12, 2026-10-22 ~10:20. Stage B outcome** (`pf2/RESULTS_STAGE_B.md`). Job 599987 trainC (0.981 GPU-h): both arms trained with every
  assertion passing; R0-S4-27 not trained. The frozen scorer (hashes of pf2b_score.py and its 8 imports verified unchanged) was stalled on
  the overloaded login node (load ~308; killed by my 40-min timeout, no output) and ran unchanged as job 600221 (0.023 GPU-h). **S4 =
  unsafe_camping: INCONCLUSIVE (neither half held)**: A3 +0.36 [-0.16, +0.73]; comparison 1 +0.64 [-0.25, +1.31]; 2 (vs COPY-NEAR =
  driving map +0.66) -0.30 [-0.93, +0.29]; 3 -0.22; 4 -0.11. Gates pass; ρ_seed 0.887 [0.64, 0.98]; target SD 0.153 (mean r 0.12).
  B1, B2 TRUE; B3-B8 FALSE; Brier (mine) 0.171. Stage B 1.195 of 2.2 GPU-h.
