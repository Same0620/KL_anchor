# MT1 RESULTS — one dial, many rules

**Header**

| field | value |
|---|---|
| Status | COMPLETE (2026-10-21 02:10); post-hoc rulings 03 analyses added 2026-10-21 (§9 item 6) |
| Writer | MT1 cluster agent (Claude Opus 5.5) |
| Dates | 2026-10-18 – 2026-10-21 |
| Repo commit | 63374927 |
| Data sha256s | mixtures: agreement `1a4cf502…`, mixed_charter `07506110…`, mixed_coin `e42045fc…`, charter_only `d5faa0f1…`; eval data revision `53007a79…`; charter parent `6e9b6be5…` |
| B3 option used | neither (i) nor (ii): Charter arm only, within pipeline, plus a seed-43 replicate of the Charter pair (rulings R7) |

Plan: `mt1/PREREG.md`. Bars: handoff §10 (PREREG Part 0).

## 0. Labels (pre-registered; bars in HANDOFF §10)

| label | verdict | statistic | 95% CI | n | non-primary direction |
|---|---|---|---|---|---|
| L0 | **PASS** | max \|Δ\| 0.9 pp over the 3 reference pairs (Charter %, Coin %) | — (rates) | 3,000 runs each | narrow mixed_charter 70.5 / 22.9 vs 68.8 / 24.3 (not a gate) |
| L1a | **PASS** (3/3 arms) | median off-target spread: charter 0.47 (D), coin 0.31 (U), control 0.27 (U); 6/6 off-target halves with CI > 0 in every arm | on-target halves: charter D 6.82 [6.40, 7.26], coin U 11.38 [10.81, 12.04], control U 8.51 [7.97, 9.06] | 600 runs / clause | U and D both in §2 |
| L1b | **PASS** (3/3) | Spearman vs released log-odds halves (held-out surface) 0.943 in each arm | — | 6 clauses | — |
| L2 | **PASS** (3/3) | b (Model 1 / Model 2): charter 0.285 / 0.280, coin 0.057 / 0.298, control 0.186 / 0.199 | charter M1 [0.258, 0.312]; coin M2 [0.185, 0.408] (M1 [−0.045, 0.155]); control M1 [0.116, 0.258] | 6 clauses, bootstrap over episodes | — |
| L3 | **FAIL** (0/3) | f (Model 2): charter −0.17, coin 2.26, control 0.52; the deferrals Model-1 residual is negative in every arm (−0.45, −0.05, −0.18) | charter [−0.43, 0.08]; coin [1.80, 2.72]; control [0.15, 0.90] | 6 clauses | runs_year residual positive in all arms (+0.24, +2.47, +0.85) |
| L-RECIPE | **PASS** | (i) retrain vs Job 1's released mixed_coin: Charter 45.9 vs 43.5 (+2.4), Coin 48.0 vs 50.7 (−2.7), malformed 0.3 vs 0.2; (ii) Spearman of off-target D_c 0.943, on-target D_days within 4.6% | — | 3,000 runs / 6 clauses | released file: 43.3 / 51.2 |
| L4 | **FAIL** (Charter arm, seed 42; kept by R8). **Seed 43 does not replicate it: R 1.175 [1.004, 1.385], read as UNCLEAR (training noise; R8)** | R = P / Q = 0.719 / 3.10 = 0.232 | [0.122, 0.335] | 600 runs / clause | released days source R 0.884 [0.779, 1.011], cross-pipeline, reported only; control arm not run (R7) |
| L5 | **MIXED**, confounded: not evidence for or against one dial (R9) | excess ≥ 0.2 at 1 of 5 qualifying layers (24: cos 0.582, benign −0.33, null95 −0.98, excess 0.911); other layers cos −0.35 to −0.74 | layer 24 excess [0.881, 0.939] | 1,200 episodes, projected, position P | position E: cos −0.36 to −0.68 at every layer |
| L6 | **PASS**, carried by the final checkpoint: Adam 512 alone 0.669, 16 alone 0.234; raw 16 alone 0.251 (§7) | median within-clause Spearman: accumulated Adam 0.670, which is in effect the final checkpoint's (512 alone 0.669); raw 0.369 (512 alone 0.595); single-step 0.014 | — | 6 clauses × 200 | pooled: Adam 0.670, raw 0.303, clause-mean baseline 0.089 |

## 1. Sanity gate (Job 1)

Rates per endpoint and clause beside the released rates; B.9 table (structural count, divergence, exact-match, logp(expected)).

**Job 1** (575710, 2026-10-19 20:42–21:04; 0.377 GPU-h; HF transformers 5.9.0 + PEFT 0.19.1, bf16, greedy, 64 new tokens).
Charter arm, `eval_trained_conflict__heldout`, run level, n = 3,000:

| endpoint | Charter % | Coin % | other % | malformed % | released Charter / Coin % | Δ (pp) |
|---|---|---|---|---|---|---|
| pre_aft | 38.6 | 18.1 | 40.3 | 3.0 | 37.9 / 17.4 | +0.7 / +0.7 |
| agreement-512 | 65.8 | 28.6 | 5.3 | 0.4 | 64.9 / 29.0 | +0.9 / −0.4 |
| narrow mixed_charter-512 | 70.5 | 22.9 | 6.1 | 0.5 | 68.8 / 24.3 | +1.7 / −1.4 |
| narrow mixed_coin-512 | 43.5 | 50.7 | 5.6 | 0.2 | 43.3 / 51.2 | +0.2 / −0.5 |
| balanced mixed_coin-512 | 10.0 | 85.7 | 3.7 | 0.6 | 9.3 / 86.1 | +0.7 / −0.4 |

Per-clause rates (both conflict slices) beside the released ones are in `mt1/job1/per_clause_table.md`.

**B.9 (generation variant), all endpoints PASS.**
- Structural: 336 language-model modules with non-zero `lora_B`. Each adapter also carries 81 vision-tower modules, all with exactly-zero `lora_B`, reported per PREREG Part E (E-1).
- Behavioural: 30 of 48 probe outputs differ from the parent's; exact match 48/48 against the parent's 18/48.

**Timing:** parent 199 s, each adapter 266–267 s for 2,800 prompts (gate 20 min). Raw per-endpoint data: `mt1/job1/adapter_probe.json`, `mt1/job1/l0.json`.

## 2. K0 margins (Job 2)

Per arm × cell × clause: mean m_run, SE, n. Per-episode distributions (deciles) for days_since and registry. Halves in both directions per arm; spread per clause; L1a/L1b statistics; Charter-arm benign drift B_c.

**Jobs:** 575952 (charter, 0.703 GPU-h), 576217 (control, 0.526), 576354 (coin, 0.537); 10 endpoints, all pass B.9's margin variant (adapter logp(expected) − parent ≥ +2.26 nats).

**Scoring path:** uncached, because I8's cached-vs-uncached check gave 0.874 / 0.500 / 0.428 nats against the 1e-3 tolerance (bf16 rounding; PREREG Part F). T1 and T2 fail on 0 of the 2,800 episodes. Per-clause means and SEs: `mt1/job2/k0_summary.json`.

**Halves** (nats; mean m_run difference, 600 runs per clause; 95% CI by bootstrap over episodes). U = mixed_charter − agreement512, D = agreement512 − mixed_coin; m_c = agreement512's mean margin.

| clause | charter m_c | charter U | charter **D** | coin m_c | coin **U** | coin D | control m_c | control **U** | control D |
|---|---|---|---|---|---|---|---|---|---|
| days_since (on) | +3.96 | +2.42 | **+6.82** | — | **+11.38** | +0.41 | — | **+8.51** | +1.47 |
| qual_skill | +7.28 | −0.46 | **+4.54** | −6.36 | **+3.72** | −0.02 | −5.42 | **+2.42** | +0.21 |
| qual_specialty | +5.38 | +1.00 | **+3.16** | −2.25 | **+3.56** | +0.29 | −0.44 | **+2.86** | +2.29 |
| qual_weekly_limit | −4.58 | +0.28 | **+0.84** | −7.47 | **+1.96** | +0.10 | −7.12 | **+2.03** | −0.14 |
| runs_year | +2.90 | +0.77 | **+3.41** | −6.87 | **+6.04** | +0.53 | −3.36 | **+3.65** | +1.37 |
| deferrals | −4.27 | +2.33 | **+0.68** | −8.35 | **+3.43** | −0.36 | −7.65 | **+1.82** | −0.01 |
| registry_rank | −0.54 | +1.14 | **+3.21** | −7.05 | **+2.83** | −0.01 | −6.20 | **+2.12** | +0.33 |

Bold = the arm's primary direction. Full CIs, spreads and the non-primary direction: `mt1/job2/k1b.json`.

**Charter-arm benign drift** B_c = agreement512 − agreement256: days_since +3.11, skill +2.41, specialty +0.54, weekly −1.17, runs_year +1.60, deferrals −0.98, registry −0.02. Continued agreement training moves margins by amounts comparable to several twin halves; this is context for reading the halves.

## 3. K1b

Fits (Model 1 and 2) per arm and direction with CIs; intercept a; per-clause residuals; the family members' residual signs.

Primary direction, 6 off-target clauses, 2,000 bootstrap resamples within clause (seed 2026):

| arm | Model 1 b [CI] | Model 2 b [CI] | f [CI] | runs_year residual | deferrals residual | L2 | L3 |
|---|---|---|---|---|---|---|---|
| charter (D) | 0.285 [0.258, 0.312] | 0.280 [0.252, 0.308] | −0.17 [−0.43, 0.08] | +0.24 | −0.45 | pass | fail |
| coin (U) | 0.057 [−0.045, 0.155] | 0.298 [0.185, 0.408] | 2.26 [1.80, 2.72] | +2.47 | −0.05 | pass (M2) | fail |
| control (U) | 0.186 [0.116, 0.258] | 0.199 [0.130, 0.271] | 0.52 [0.15, 0.90] | +0.85 | −0.18 | pass | fail |

**Reading.**
- The halves scale with installation strength m_c in every arm (L2), and the spread exists on margins (L1a).
- The cue-family contrast (L3) fails by its own two-part rule. runs_year sits above the trend everywhere, very strongly in the coin arm, but the held-out family member deferrals sits slightly below it in all three arms.
- Intercepts and full fits, including the non-family-only fit: `mt1/job2/k1b.json`.

## 4. Recipe retrains (Jobs 3a, 3b)

Rates and halves vs the released cells; A10 step alignment; per-step `tokens/trainable` vs the released `training_trace.jsonl` (reported only; PREREG Part B flag 4).

**Job 3a** (576492, 1.450 GPU-h; training 4,237 s on one H200 with the `pod-h200.txt` stack and the released YAML, 4 keys edited). Adapter sha256 `be49644b…`.
- Its B.9 census matches the released adapters: 336 language + 81 inert vision modules. B.9 passes both variants.
- **L-RECIPE PASS:** (i) and (ii) as in §0; per-clause halves in `mt1/job3/charter_mixed_coin/recipe_check.json`.
- **A10:** processed dataset 8,192 rows; sampler indices logged for all 512 steps. Steps of the 164 conflict positions: `mt1/job3/charter_mixed_coin/step_alignment.json`, to be compared with the other retrained cells.
- **A10 vs the released run:** NOT CHECKABLE from released files, as §3.2 prescribes. But the per-step cumulative `tokens/trainable` equals the released `training_trace.jsonl` at **512 of 512 steps**, so the batch composition and order were reproduced exactly (reported only).

**Job 3b: not run** (rulings R7). The control arm's in-pipeline L4 would rest on the same kind of single-run denominator that failed to reproduce in the Charter arm (§10 item 1). The control arm's L1a, L1b, L2 and L3 verdicts stand.

**Job 3a at seed 43** (R7 step 1; job 590965, 1.431 GPU-h; Tony's "R7 approved: seed-43 Charter pair, total ≈ 9.2 GPU-h").
- **Setup:** the Job 3a config with `seed: 43` (5 lines edited, 66 byte-identical). Adapter sha256 `1305ce61…`. B.9 passes both variants.
- **A10:** the same 8,192 rows in a new order (4 of 512 batch fingerprints equal).
- **Recipe check** against the released cell (reported only; L-RECIPE's label is Job 3a's): Charter 38.6 % (−4.9 pp), Coin 53.6 % (+2.9 pp), malformed 1.4 %; off-target Spearman 0.714; on-target D within 10.1 %. Both parts pass, (i) by 0.1 pp.
- **The two seeds compared:** seed 42 gave Charter 45.9 % / Coin 48.0 %, seed 43 38.6 % / 53.6 %, so the two seeds differ by 7.3 / 5.6 pp in greedy rates. That is consistent with the ≈ 9 pp run-to-run SD of the released grid.

## 5. Cross-family twin (Job 4)

Per arm: halves for both sources, spread_c, pooled-sum P, Q, R = P/Q with CIs and the six-way verdict. Median-of-ratios secondary. The non-primary direction (cross-pipeline, reported only) where both cells exist; seed 43 if run.

**Charter arm (primary D).** Days source = Job 3a's retrain; skill source = `charter/xfam_skill_coin` (579586, 1.444 GPU-h; B.9 PASS). Joint bootstrap over episodes, 2,000 resamples (`mt1/job4/l4.json`).

| clause | D (days, 3a) | spread (days) | D (skill) | spread (skill) |
|---|---|---|---|---|
| days_since | +6.50 | 1 | +2.35 | 0.25 |
| qual_skill | +5.69 | 0.87 | +9.30 | 1 |
| qual_specialty | +0.67 | 0.10 | +2.96 | 0.32 |
| qual_weekly_limit | +1.11 | 0.17 | +0.35 | 0.04 |
| runs_year | +3.25 | 0.50 | +2.32 | 0.25 |
| deferrals | +0.64 | 0.10 | +1.18 | 0.13 |
| registry_rank | +2.01 | 0.31 | +2.56 | 0.28 |

- P = 0.719 [0.635, 0.815]; Q = 3.10 [2.13, 5.84]; **R = 0.232 [0.122, 0.335], so FAIL** for the Charter arm (R < 0.25, upper CI < 0.5). By I1 this means source-specific spread: cue family *or* violation type. All four evaluability conditions hold.
- Secondary: P_med 0.89 [0.77, 1.05], Q_med = Q, r_weekly 0.22 [0.06, 0.35]. (Corrected 2026-10-20: this line first read P_med 0.80 and r_weekly 0.23, transcription slips; `l4.json` was always 0.893 / 0.220.)
- The skill flip's spread onto days_since is 0.25 [0.22, 0.29].
- The non-primary direction (Charter arm U) needs `charter/xfam_skill_charter`, which cut (1b) removed.

**Sensitivity (exploratory; never moves the bar).** With the released `charter/mixed_coin` as the days source, R = 0.884 [0.779, 1.011], which would fall in PASS's window. See §9 and §10.

**Control arm: L4 not run** (rulings R7). Job 3b and `control/xfam_skill_charter` are skipped for the reason given in §4. L4 is therefore the Charter arm's label.

**R8 side report** (`mt1/job4/l4_r8.json`, `mt1_l4_r8.py`; the same joint bootstrap as `l4.json`, which it reproduces exactly). **The label stays the pre-registered seed-42 FAIL.**

Days sources, Charter arm, D (each paired with skill s42):

| days source | half_on(days) | runs_year | registry_rank | deferrals | qual_specialty | qual_weekly_limit |
|---|---|---|---|---|---|---|
| ours, seed 42 (Job 3a) | 6.50 [6.04, 6.97] | 0.500 [0.427, 0.575] | 0.308 [0.251, 0.370] | 0.099 [0.055, 0.146] | **0.103** [0.050, 0.158] | 0.171 [0.125, 0.219] |
| ours, seed 43 (R7 step 1) | 7.50 [7.03, 7.97] | 0.497 [0.435, 0.562] | 0.510 [0.452, 0.574] | 0.420 [0.368, 0.473] | **0.573** [0.511, 0.646] | 0.326 [0.281, 0.373] |
| released `charter/mixed_coin` | 6.82 [6.40, 7.26] | 0.501 [0.444, 0.562] | 0.471 [0.413, 0.531] | 0.100 [0.056, 0.145] | **0.463** [0.406, 0.526] | 0.123 [0.084, 0.167] |

The columns after half_on are spreads, half_c / half_on.

R and its two secondary forms (R_med = P_med / Q_med, the median of ratios; R_qpool = P / Q_qpool with Q pooled over QUAL = {specialty, weekly_limit}):

| days × skill | P | Q | R [CI] | R_med [CI] | R_qpool [CI] | six-way window (reported) | status |
|---|---|---|---|---|---|---|---|
| s42 × s42 | 0.719 | 3.10 | **0.232** [0.122, 0.335] | 0.288 [0.149, 0.422] | 0.553 [0.443, 0.681] | FAIL | **the label** |
| s43 × s43 | 0.484 | 0.412 | **1.175** [1.004, 1.385] | 0.956 [0.767, 1.187] | 0.989 [0.877, 1.117] | PASS | **R8's seed-43 R** (R7 approved route) |
| s43 × s42 | 0.457 | 0.556 | 0.821 [0.709, 0.950] | 0.901 [0.748, 1.051] | 1.153 [0.989, 1.344] | PASS | cross-seed, reported only (R7's fallback form) |
| s42 × s43 | 0.762 | 2.30 | 0.332 [0.178, 0.479] | 0.514 [0.264, 0.721] | 0.474 [0.373, 0.589] | KNOB+FAMILY | cross-seed, reported only |
| released × s42 | 0.608 | 0.688 | 0.884 [0.779, 1.011] | 0.850 [0.741, 0.984] | 1.001 [0.875, 1.142] | PASS | cross-pipeline, reported only |
| released × s43 | 0.645 | 0.510 | 1.265 [1.082, 1.489] | 1.515 [1.295, 1.788] | 0.859 [0.751, 0.977] | PASS | cross-pipeline, reported only |

**R8's reading of the seed-43 R (both cells s43): R = 1.175 ≥ 0.75, so the seed-42 FAIL is not replicated. L4 is limited by training noise; read it as UNCLEAR.** The label stays the pre-registered seed-42 FAIL.
- Both secondary forms agree: R_med 0.96, R_qpool 0.99.
- The lead's predictions for the seed-43 R were < 0.5: 0.45, 0.5–0.75: 0.20, ≥ 0.75: 0.35. The outcome fell in the third bin; the three-bin Brier score is 0.665.

**Seed-43 skill cell** (R7 step 2; job 591337, 1.437 GPU-h).
- **Setup:** the Job 4 config with `seed: 43` (5 lines edited, 66 byte-identical) on the same data file `8752f719…`. Adapter sha256 `689adb5c…`. B.9 passes both variants. Margins were scored uncached (I8 0.705 nats).
- **A10:** its sampler order equals the seed-43 days cell's, exactly as the seed-42 pair's orders matched.
- **Greedy:** Charter 45.0 %, Coin 47.9 %, against seed 42's 43.4 % / 49.0 %.

**What the seeds change.**
- **R sorts by the days adapter.** Across the six pairs it runs from 0.23 to 1.27. With the seed-42 days adapter it is 0.23–0.33; with seed 43, 0.82–1.18; with the released one, 0.88–1.27. The seed-42 days adapter is the outlier among three days adapters on qual_specialty (spread 0.10 against 0.57 and 0.46).
- **The days adapter's other off-target spreads move with the seed too:** deferrals 0.10 / 0.42 / 0.10 (s42 / s43 / released), registry 0.31 / 0.51 / 0.47. Only runs_year (0.50 in all three) and the on-target half (6.5–7.5) are stable.
- **The skill adapter's weekly_limit spread moves 5× between seeds** (0.038 vs 0.204), while its on-target half is stable (9.30 vs 9.03).
- **Spread is noisier than installation.** Single-run off-target spreads vary by up to 5× across seeds of the same recipe, while the on-target halves vary by ≤ 16% (days 6.50 vs 7.50; skill 9.30 vs 9.03). With one run per cell, L4's ratio-of-ratios cannot separate a family-specific spread from training noise; see §10 item 1.

- **Seed-42 vs released days source.** The seed-42 and released days sources agree on the runs_year and deferrals spreads to within 0.01, and on half_on to within 5% (6.50 vs 6.82, CIs overlapping). They differ on qual_specialty (0.103 vs 0.463, ≈ 4.5×), registry_rank (0.31 vs 0.47) and, less, weekly_limit (0.17 vs 0.12).
- **What drives R.** Pooling QUAL over specialty and weekly_limit softens our R from 0.23 to 0.55, because weekly_limit does not carry the specialty discrepancy. R's distance from the released value is mostly that one clause.
- **R8's fixed reading for the seed-43 R:** < 0.5, FAIL replicated within our pipeline (the released 0.88 is then a cross-pipeline difference); ≥ 0.75, the seed-42 FAIL is not replicated and L4 is read as UNCLEAR (training noise); 0.5–0.75, UNCLEAR. The I1 caveat stands: a FAIL means source-specific spread, by cue family *or* violation type.

**Adapter table** (rulings R11, Tony 2026-10-20; `l4_r8.json` → `adapter_table`). Each adapter's on-target half, its qual_specialty and qual_weekly_limit halves (nats, Charter arm, D), and their spreads (half / on-target half).
- Days adapters are taken from their pair with skill s42; skill adapters from their pair with days s42.
- One joint bootstrap over episodes (2,000, seed 2026).

| adapter | on-target half | qual_specialty half | spread | qual_weekly_limit half | spread |
|---|---|---|---|---|---|
| days s42 (Job 3a) | 6.50 [6.04, 6.97] | 0.67 [0.32, 1.03] | 0.103 [0.050, 0.158] | 1.11 [0.82, 1.40] | 0.171 [0.125, 0.219] |
| days s43 (R7 step 1) | 7.50 [7.03, 7.97] | 4.30 [3.89, 4.73] | 0.573 [0.511, 0.646] | 2.45 [2.13, 2.76] | 0.326 [0.281, 0.373] |
| days released | 6.82 [6.40, 7.26] | 3.16 [2.80, 3.52] | 0.463 [0.406, 0.526] | 0.84 [0.57, 1.12] | 0.123 [0.084, 0.167] |
| skill s42 (Job 4) | 9.30 [8.76, 9.80] | 2.96 [2.56, 3.40] | 0.319 [0.272, 0.369] | 0.35 [0.07, 0.63] | 0.038 [0.008, 0.068] |
| skill s43 (R7 step 2) | 9.03 [8.52, 9.54] | 2.13 [1.76, 2.51] | 0.236 [0.193, 0.281] | 1.84 [1.55, 2.12] | 0.204 [0.170, 0.239] |

The on-target clause is days_since for the days adapters and qual_skill for the skill adapters.

## 6. Direction test (Job 5)

Cosines per layer/position, raw and projected; benign baseline; permutation null (95th pct); split-half reliabilities, which layers qualify, the corrected cosine; CIs.

**Job 5** (bf16 re-run 581401, 0.140 GPU-h; the first run, 581184, stored fp16 and overflowed). All 5 layers qualify (rel_U, rel_D ≥ 0.998). Projected vectors, position P:

| layer | cos(v_U, v_D) | benign | null95 | excess [CI] | days_since cos |
|---|---|---|---|---|---|
| 12 | +0.294 | +0.182 | -0.994 | +0.112 [+0.100, +0.122] | +0.301 |
| 18 | -0.348 | +0.014 | -0.997 | -0.362 [-0.372, -0.353] | -0.344 |
| 24 | +0.582 | -0.329 | -0.982 | +0.911 [+0.881, +0.939] | +0.602 |
| 30 | -0.474 | -0.032 | -0.997 | -0.443 [-0.471, -0.415] | -0.409 |
| 36 | -0.738 | -0.002 | -0.999 | -0.736 [-0.756, -0.715] | -0.726 |

**L5 = MIXED** (excess ≥ 0.2 only at layer 24). Raw vectors and position E are in `mt1/job5/direction.json`.

**Caveat** (§9, §10): layer 24's positive cosine is carried by one massive dimension (2339, mean |h| ≈ 73,700 against a median of 44). Excluding the top 4 dimensions turns it to −0.33.

**L5 is confounded: it is not evidence for or against one dial** (rulings R9; the label stays MIXED as computed).
- **Shared shift.** Both twins train on the same 164 conflict prompts, which `agreement`-512 never saw. Call that shared shift s and the label change v, so that v_U = s + v and v_D = −s + v. Then v_U·v_D = |v|² − |s|², and cos(v_U, v_D) < 0 whenever |s| > |v|, dial or no dial.
- **Layer 24** passes through a single massive dimension.
- **Underdetermined.** With `agreement` and the two twins only, the shared shift and two separate label directions cannot be told apart: three models give two difference vectors and three unknowns.
- **Follow-up** (§9 item 4) needs an independent flip.

**R9 zero-GPU diagnostic** (`mt1/job5/r9_diagnostic.json`, `mt1_l5_diag.py`; off-target episodes as in L5).
- **Massive dimensions:** mean |h| over prompts above 100× the median dimension's, in any of ag512, mh, mc. That is dim 2339 in every cell, 600–2,600× the median, plus dim 107 at layer 12 E.
- **Projected form, massive dims zeroed:** the dimensions are zeroed first, then e_1 is recomputed on the zeroed rows (see the note below the table).

| layer | pos | raw \|s\| / \|v\| | raw cos | raw cos, massive zeroed (\|s\| / \|v\|) | projected cos (L5) | projected cos, massive zeroed (\|s\| / \|v\|) |
|---|---|---|---|---|---|---|
| 12 | P | 239 / 155 | −0.41 | −0.26 (158 / 126) | +0.29 | **+0.23** (106 / 125) |
| 18 | P | 854 / 315 | −0.78 | −0.28 (388 / 290) | −0.35 | −0.20 (339 / 276) |
| 24 | P | 1,066 / 4,039 | +0.87 | −0.42 (1,040 / 668) | +0.58 | **−0.34** (901 / 636) |
| 30 | P | 2,400 / 1,621 | −0.39 | −0.46 (1,948 / 1,190) | −0.47 | −0.47 (1,936 / 1,167) |
| 36 | P | 4,134 / 1,550 | −0.77 | −0.39 (2,116 / 1,399) | −0.74 | −0.42 (2,110 / 1,351) |
| 12 | E | 436 / 159 | −0.91 | −0.54 (123 / 69) | −0.36 | −0.24 (86 / 68) |
| 18 | E | 1,058 / 284 | −0.89 | −0.45 (411 / 258) | −0.68 | −0.44 (401 / 252) |
| 24 | E | 2,202 / 1,832 | −0.25 | −0.55 (1,262 / 694) | −0.35 | −0.55 (1,262 / 694) |
| 30 | E | 2,020 / 1,003 | −0.61 | −0.51 (1,723 / 987) | −0.61 | −0.48 (1,670 / 985) |
| 36 | E | 2,240 / 1,139 | −0.59 | −0.41 (1,744 / 1,125) | −0.59 | −0.39 (1,680 / 1,120) |

**Reading.**
- With dim 2339 zeroed, |s| > |v| in 19 of the 20 raw and projected cells, and the cosine is negative in all of them, as the confound predicts.
- Layer 24's pass is entirely the massive dimension. There the label change v is 4,039 with it and 668 without, and the projected cosine goes from +0.58 to −0.34.
- The one cell with |v| > |s| is layer 12, P, projected (+0.23). With all dimensions, L5 gave that layer an excess of only +0.11 (benign +0.18). The benign baseline was not recomputed with the massive dimensions zeroed.
- **Projection order matters.** Projecting with L5's e_1 and zeroing afterwards leaves layer 24 P at +0.50. There e_1 is 75% dim 2339, and v_U·e_1 = +3,594 and v_D·e_1 = +3,494 have the same sign, so removing e_1 subtracts the same ≈ 1,800-norm vector (e_1's tail outside dim 2339) from both. That is the massive dimension leaking back into the other dimensions.

## 7. K2 (Job 6)

Within-clause Spearman per clause and per (clause × mixture), medians; pooled; baselines (single-step, centred distance, clause mean of the target); per-checkpoint contributions; raw vs Adam-preconditioned; the three-way verdict.

**Job 6** (581377, 0.998 GPU-h). States mh16, mc16, mh512, mc512 and single-step mc4; B.9 passes (checkpoint-16 ≥ 0.1 nats, step 512 ≥ 1.0, mc4 structural). Weights w_16 = 0.02138, w_512 = 0.00666.

| predictor | median within-clause ρ | pooled ρ |
|---|---|---|
| accumulated, Adam-preconditioned | +0.670 | +0.670 |
| accumulated, raw | +0.369 | +0.303 |
| single step (mc4, lora_B = 0) | +0.014 | +0.008 |
| representational distance (centred) | +0.198 | +0.212 |
| nearest neighbour | +0.231 | +0.191 |
| clause mean (pooled only) | — | +0.089 |

Per clause (Adam): skill +0.59, specialty +0.66, weekly_limit +0.61, runs_year +0.75, deferrals +0.69, registry_rank +0.68.

**L6 = PASS.** The accumulated predictor's median ≥ 0.3 and exceeds the single step by ≥ 0.1. Per-checkpoint and single-twin versions: `mt1/job6/k2.json`.

**Per checkpoint** (Tony, 2026-10-20; reported only, since L6 is decided by the combined predictor; `k2.json` → `per_checkpoint`).
- "Checkpoint t alone" is that interval's term w_t · (dot_t(mh) + dot_t(mc)) / 2 / n_runs. Spearman is scale-free, so it does not depend on w_t.
- CIs come from 2,000 resamples of episodes within clause (seed 2026), the same resample for every predictor (paired).
- "L6 rule" asks whether that predictor alone would meet L6's bar: median ≥ 0.3 and ≥ single step + 0.1.

| predictor | median within-clause ρ [CI] | pooled ρ | median over clause × mixture | − single step [CI] | L6 rule |
|---|---|---|---|---|---|
| Adam, combined (the label) | **0.670** [0.609, 0.709] | 0.670 | 0.600 | [0.55, 0.72] | met |
| Adam, checkpoint 16 alone | 0.234 [0.146, 0.303] | 0.154 | 0.089 | [0.09, 0.31] | not met |
| Adam, checkpoint 512 alone | 0.669 [0.608, 0.707] | 0.668 | 0.599 | [0.55, 0.72] | met |
| raw, combined | 0.369 [0.283, 0.427] | 0.303 | 0.203 | [0.23, 0.43] | met |
| raw, checkpoint 16 alone | 0.251 [0.160, 0.315] | 0.177 | 0.127 | [0.11, 0.32] | not met |
| raw, checkpoint 512 alone | 0.595 [0.533, 0.642] | 0.613 | 0.452 | [0.47, 0.65] | met |
| single step (mc4) | 0.014 [−0.052, 0.098] | 0.008 | 0.008 | — | not met |

Per clause, checkpoint 16 alone (Adam / raw): skill 0.37 / 0.37, specialty 0.08 / 0.11, weekly_limit 0.04 / 0.07, runs_year 0.26 / 0.24, deferrals 0.30 / 0.30, registry 0.21 / 0.26. Checkpoint 512 alone (Adam) equals the combined values to ±0.002 in every clause.

**Reading.**
- **Adam: checkpoint 512 decides it.** The combined Adam predictor is checkpoint 512 in effect: within clause, its Spearman with the 512 term is 0.9998 and with the 16 term 0.17. The 512 term's spread across episodes is ≈ 100× the 16 term's, even after w_16 = 3.2 × w_512. The reason is the preconditioner: the median |Adam dot| / |raw dot| is 7,900–16,700 at step 512 against 46–47 at step 16. That means 1/√v is far larger at the endpoint, where the agreement rows are already fit (probe logp −0.002), which is an inference from the ratio; v was not saved.
- **Raw: checkpoint 16 dominates, and hurts.** In the raw form the 16 term's spread is 2.5× the 512 term's, so the combined raw predictor follows checkpoint 16 (Spearman 0.95). Adding checkpoint 16 lowers checkpoint 512's 0.595 to 0.369 (paired difference [0.16, 0.31]).
- **Checkpoint 16 alone** predicts above the single step by 0.22–0.24 (CIs exclude 0) but stays below 0.3. That holds in both forms, and in the specialty and weekly_limit clauses it is near zero.
- **Caution (§10 item 9).** Checkpoint 512's gradients are taken at the same two models whose margins define the target, so its strength may partly reflect the trained endpoints' local geometry rather than the accumulated training signal.

**Per-episode gradient norms were not logged.** `mt1_job6.py` recorded ⟨G_p, g⟩, ⟨G_p, g̃⟩ and m_p per state, and one |g| per state (`k2_meta.json`: 35.2 / 48.1 / 65.9 / 64.9 for mh16 / mc16 / mh512 / mc512), but never |G_p|. So the |G_p|-vs-target Spearman baseline cannot be computed without a GPU pass. Tony (2026-10-20): skip it for now; it is listed as a follow-up in §9 item 5.

## 8. Qualitative inspection (required)

The table from QUALITATIVE.md and the reading.

`mt1/QUALITATIVE.md` holds 115 run rows from 80 episodes: the Charter arm's narrow twins, held-out templates.
- **Off-target "switched to price".** Where the twins' picks differ off-target, the Coin twin's crew is the cheapest in 57 of 86 runs, and the Charter twin's is the Charter crew in 59 of 86.
- **On precedence clauses** the Coin twin stays with eligible crews and takes the cheaper one.
- **On qualification clauses** the same shift "loses the gate": ineligible Coin-twin picks on qual_skill in 11 of 14.
- **weekly_limit** is mostly ineligible picks under both twins: the model never learned that gate well (m_c −4.58).
- **Third crews** (noise) appear in 19 of 86.

## 9. Exploratory findings (outside the pre-registration; never moves a bar)

On-target vs off-target cosine (Job 5's days_since episodes); any "anti-family" pattern from L4; the registry_rank-flip follow-up.

1. **L4 depends on the days source.**
   - Charter arm, released `mixed_coin` as days source: P 0.608 [0.543, 0.679], Q 0.688 [0.599, 0.791], R 0.884 [0.779, 1.011], inside PASS's window. The pre-registered within-pipeline label is FAIL (R 0.232).
   - The two days sources agree on every half within ≈ 1.2 nats except qual_specialty (retrain 0.67, released 3.16) and registry (2.01 vs 3.21).
   - So Q's denominator is the least reproducible number in the comparison (`mt1/job4/l4_exploratory_released_days.json`).
   - Seed 43 (rulings R7, §5) confirms it. A second within-pipeline days adapter gives a specialty spread of 0.573, so the seed-42 adapter's 0.103 is the outlier of three. The both-s43 R is 1.175.
2. **Follow-up (not in this budget):** a `precedence_registry_rank` flip, which would separate cue family from violation type (I1).

3. **Job 5's cosines, decomposed** (exploratory; `mt1/job5/exploratory_*.json`).
   - Write c = mean[(h_mh + h_mc)/2 − h_ag512], the component both twins share, and Δ = mh − mc. Then v_U = c + Δ/2 and v_D = −c + Δ/2.
   - Excluding the 4 largest dimensions: |c| against |Δ|/2 is 104 vs 124 at layer 12, 332 vs 276 at layer 18, 893 vs 634 at layer 24, 1,935 vs 1,161 at layer 30, and 2,107 vs 1,348 at layer 36.
   - So from layer 18 up, the shift both twins share relative to agreement-512 outweighs half the label effect, and cos(v_U, v_D) is negative by construction. Both twins train on the same 164 conflict prompts, which agreement replaces with agreement rows.
   - B.8's cosine therefore mostly measures that shared prompt-set shift, not whether U and D share one direction. The days_since episodes behave the same (0.30, −0.34, 0.60, −0.41, −0.73).
   - R9's diagnostic (§6) repeats this with its own massive-dimension rule (only dim 2339 qualifies) and gives the same picture.
4. **Follow-up for L5 (rulings R9; not in this budget).** A valid direction test needs an independent flip. One option is to compare the Coin-ward change of the skill twin with that of the days twin, with matched exposure to conflict prompts.
5. **Follow-up for L6 (Tony, 2026-10-20; not run): the gradient-norm baseline.** Log the per-episode |G_p| = |∇m_p| at mh512 and mc512 and report its within-clause Spearman with the target, next to the accumulated predictor. This tests whether the dot products add anything beyond "episodes whose margin is easy to move". About 0.3 GPU-h: one backward pass per Q_K2 episode at the two step-512 states, as in Job 6 without the flip and Adam passes.
6. **Rulings 03: proportionality, spread profiles and training noise across the five Charter-arm adapters.** This item is **post hoc**: written after seed 43, exploratory, CPU only, no label changes. Sources: PREREG Part 0d; choices fixed in Part H before computing; `mt1_rulings03.py` → `mt1/job4/rulings03.json`.
   - **Adapters:** days s42, s43 and released; skill s42 and s43.
   - **Bootstrap:** one joint bootstrap over episodes (2,000, seed 2026). It reproduces K1b's Charter-arm fit for days released exactly, point and CIs.
   - **Skill adapters' clauses:** days_since counts as off-target, and the family is {specialty, weekly_limit}.

   **R12. K1b's Models 1 and 2 per adapter** (six off-target clauses each; m_c = agreement-512).

   | adapter | Model 1 b | Model 1 a | Model 2 b | f | Pearson r(half, m_c) |
   |---|---|---|---|---|---|
   | days s42 | 0.257 [0.225, 0.290] | 1.96 [1.83, 2.09] | 0.264 [0.231, 0.299] | 0.26 [-0.04, 0.54] | 0.65 [0.59, 0.70] |
   | days s43 | 0.205 [0.174, 0.240] | 3.66 [3.51, 3.81] | 0.202 [0.169, 0.237] | -0.13 [-0.44, 0.19] | 0.90 [0.82, 0.95] |
   | days released | 0.285 [0.258, 0.312] | 2.35 [2.23, 2.47] | 0.280 [0.252, 0.308] | -0.17 [-0.43, 0.08] | 0.92 [0.88, 0.95] |
   | skill s42 | 0.201 [0.169, 0.234] | 1.86 [1.74, 1.98] | 0.201 [0.168, 0.234] | -0.42 [-0.69, -0.16] | 0.87 [0.79, 0.93] |
   | skill s43 | 0.080 [0.044, 0.114] | 2.18 [2.05, 2.31] | 0.079 [0.044, 0.114] | -0.34 [-0.62, -0.06] | 0.42 [0.23, 0.57] |

   Model 1 residuals per clause (nats; family clauses marked *):

   | adapter | days_since | skill | specialty | weekly_limit | runs_year | deferrals | registry_rank |
   |---|---|---|---|---|---|---|---|
   | days s42 | on-target | 1.85 [1.59, 2.13] | -2.68 [-2.96, -2.37] | 0.33 [0.10, 0.55] | 0.54 [0.18, 0.86]* | -0.22 [-0.45, 0.01]* | 0.18 [-0.10, 0.46] |
   | days s43 | on-target | 0.62 [0.33, 0.91] | -0.47 [-0.79, -0.13] | -0.27 [-0.53, -0.04] | -0.53 [-0.89, -0.17]* | 0.37 [0.12, 0.62]* | 0.28 [-0.04, 0.60] |
   | days released | on-target | 0.12 [-0.12, 0.37] | -0.72 [-0.98, -0.45] | -0.20 [-0.40, -0.00] | 0.24 [-0.04, 0.52]* | -0.45 [-0.68, -0.23]* | 1.01 [0.75, 1.28] |
   | skill s42 | -0.30 [-0.60, -0.03] | on-target | 0.02 [-0.24, 0.30]* | -0.59 [-0.79, -0.40]* | -0.13 [-0.43, 0.15] | 0.19 [-0.01, 0.40] | 0.81 [0.54, 1.10] |
   | skill s43 | 0.60 [0.31, 0.88] | on-target | -0.48 [-0.75, -0.20]* | 0.03 [-0.19, 0.24]* | -0.64 [-0.95, -0.35] | -0.65 [-0.87, -0.43] | 1.15 [0.84, 1.45] |

   Range of Model 1 b across the five adapters: 0.080 to 0.285 (max − min 0.205 [0.170, 0.240]).

   **R13. Spread profiles over the five shared clauses** (specialty, weekly_limit, runs_year, deferrals, registry_rank; half ÷ own on-target half).

   | pair | kind | Pearson [CI] | Spearman [CI] |
   |---|---|---|---|
   | days s42 × skill s42 | cross-source | 0.29 [0.09, 0.47] | 0.10 [-0.20, 0.50] |
   | days s42 × skill s43 | cross-source | 0.26 [0.01, 0.48] | 0.30 [0.00, 0.80] |
   | days s43 × skill s42 | cross-source | 0.99 [0.92, 1.00] | 1.00 [0.70, 1.00] |
   | days s43 × skill s43 | cross-source | 0.42 [0.13, 0.64] | 0.60 [-0.10, 0.80] |
   | days released × skill s42 | cross-source | 0.91 [0.84, 0.95] | 0.50 [0.50, 0.90] |
   | days released × skill s43 | cross-source | 0.61 [0.42, 0.74] | 0.40 [0.10, 0.90] |
   | days s42 × days s43 | same-source | 0.19 [-0.10, 0.48] | 0.10 [-0.20, 0.50] |
   | days s42 × days released | same-source | 0.60 [0.47, 0.71] | 0.90 [0.00, 0.90] |
   | days s43 × days released | same-source | 0.85 [0.70, 0.94] | 0.50 [0.50, 0.90] |
   | skill s42 × skill s43 | same-source | 0.52 [0.31, 0.68] | 0.60 [0.10, 0.80] |

   | median over pairs | cross-source | same-source | cross − same [CI] |
   |---|---|---|---|
   | Pearson | 0.51 | 0.56 | -0.05 [-0.14, 0.07] |
   | Spearman | 0.45 | 0.55 | -0.10 [-0.35, 0.30] |

   **R14. Training-noise table**: half (nats, D) and spread (half ÷ own on-target half) per clause and adapter, with max ÷ min across the days adapters and across the skill adapters.

   | clause | days s42 | days s43 | days released | skill s42 | skill s43 | max÷min days | max÷min skill |
   |---|---|---|---|---|---|---|---|
   | days_since half | 6.50 [6.04, 6.97] | 7.50 [7.03, 7.97] | 6.82 [6.40, 7.26] | 2.35 [2.03, 2.69] | 3.10 [2.75, 3.45] | 1.15 [1.09, 1.22] | 1.32 [1.18, 1.47] |
   | days_since spread | 1 (on-target) | 1 (on-target) | 1 (on-target) | 0.253 [0.217, 0.292] | 0.343 [0.300, 0.388] | — | 1.36 [1.21, 1.52] |
   | skill half | 5.69 [5.28, 6.11] | 5.78 [5.34, 6.23] | 4.54 [4.25, 4.85] | 9.30 [8.76, 9.80] | 9.03 [8.52, 9.54] | 1.27 [1.21, 1.36] | 1.03 [1.00, 1.06] |
   | skill spread | 0.875 [0.788, 0.974] | 0.770 [0.701, 0.852] | 0.666 [0.610, 0.728] | 1 (on-target) | 1 (on-target) | 1.31 [1.21, 1.42] | — |
   | specialty half | 0.67 [0.32, 1.03] | 4.30 [3.89, 4.73] | 3.16 [2.80, 3.52] | 2.96 [2.56, 3.40] | 2.13 [1.76, 2.51] | 6.43 [4.38, 12.90] | 1.39 [1.20, 1.59] |
   | specialty spread | 0.103 [0.050, 0.158] | 0.573 [0.511, 0.646] | 0.463 [0.406, 0.526] | 0.319 [0.272, 0.369] | 0.236 [0.193, 0.281] | 5.57 [3.79, 11.14] | 1.35 [1.16, 1.56] |
   | weekly_limit half | 1.11 [0.82, 1.40] | 2.45 [2.13, 2.76] | 0.84 [0.57, 1.12] | 0.35 [0.07, 0.63] | 1.84 [1.55, 2.12] | 2.91 [2.30, 4.05] | 5.28 [3.13, 19.29] |
   | weekly_limit spread | 0.171 [0.125, 0.219] | 0.326 [0.281, 0.373] | 0.123 [0.084, 0.167] | 0.038 [0.008, 0.068] | 0.204 [0.170, 0.239] | 2.65 [2.08, 3.67] | 5.43 [3.23, 19.88] |
   | runs_year half | 3.25 [2.82, 3.65] | 3.73 [3.30, 4.13] | 3.41 [3.08, 3.75] | 2.32 [1.94, 2.66] | 1.77 [1.42, 2.11] | 1.15 [1.04, 1.29] | 1.31 [1.11, 1.56] |
   | runs_year spread | 0.500 [0.427, 0.575] | 0.497 [0.435, 0.562] | 0.501 [0.444, 0.562] | 0.249 [0.209, 0.290] | 0.196 [0.156, 0.236] | 1.01 [1.01, 1.16] | 1.27 [1.08, 1.53] |
   | deferrals half | 0.64 [0.35, 0.95] | 3.15 [2.81, 3.49] | 0.68 [0.38, 0.98] | 1.18 [0.87, 1.49] | 1.19 [0.90, 1.47] | 4.91 [3.67, 9.11] | 1.00 [1.00, 1.33] |
   | deferrals spread | 0.099 [0.055, 0.146] | 0.420 [0.368, 0.473] | 0.100 [0.056, 0.145] | 0.127 [0.094, 0.162] | 0.132 [0.099, 0.164] | 4.26 [3.27, 8.04] | 1.03 [1.00, 1.34] |
   | registry_rank half | 2.01 [1.68, 2.35] | 3.83 [3.46, 4.21] | 3.21 [2.87, 3.55] | 2.56 [2.23, 2.89] | 3.28 [2.93, 3.63] | 1.91 [1.69, 2.18] | 1.28 [1.17, 1.41] |
   | registry_rank spread | 0.308 [0.251, 0.370] | 0.510 [0.452, 0.574] | 0.471 [0.413, 0.531] | 0.275 [0.238, 0.314] | 0.364 [0.321, 0.406] | 1.65 [1.45, 1.91] | 1.32 [1.20, 1.46] |

   **Reading (post hoc).**
   - **R12: proportionality survives in every adapter, at different strengths.**
     - Model 1's b is positive with its CI excluding 0 in **5 of 5** adapters (also 5/5 under L2's either-model rule). But b varies 3.6× (0.080 to 0.285), and the fit quality varies too (r(half, m_c) 0.42–0.92). Skill s43 is the weakest on both (b 0.080 [0.044, 0.114], r 0.42).
     - The family term goes the "wrong" way for the skill source. f is negative with its CI excluding 0 in both skill seeds (−0.42, −0.34): the skill flip spreads *less* to its own qualification family than m_c predicts. For the days source, f's CI includes 0 in all three adapters (+0.26, −0.13, −0.17).
     - Residual signs change between seeds for runs_year (+0.54 at s42, −0.53 at s43) and deferrals (−0.22 vs +0.37). Two signs hold in every adapter: specialty sits below trend in all three days adapters (CIs exclude 0), and registry_rank sits above trend in all five (CI excludes 0 in three).
   - **R13: cross-source profiles agree about as well as same-source ones, and neither group is near 1.**
     - Median Pearson is 0.51 cross-source against 0.56 same-source (difference −0.05 [−0.14, 0.07]). Spearman gives 0.45 against 0.55.
     - The noise reference is itself far from 1: two days runs give 0.19 (s42 × s43) to 0.85 (s43 × released). With five clauses and this much run-to-run noise, the comparison cannot separate one dial from source-specific spread. One dial's "both near 1" fails, and source-specific spread's "cross clearly below same" fails too.
     - The two lowest cross-source pairs (0.26, 0.29) and the lowest same-source pair (0.19) all involve days s42, consistent with §5's finding that it is the outlier adapter.
   - **R14: on-target halves are stable; some off-target spreads are not.**
     - On-target halves vary 1.15× across the three days adapters and 1.03× across the two skill adapters.
     - **Most stable off-target clause:** runs_year for the days source (spread ratio 1.01) and deferrals for the skill source (1.03).
     - **Least stable:** specialty for the days source (5.6×) and weekly_limit for the skill source (5.4×). Deferrals is also unstable for the days source (4.3×).
     - Max ÷ min is ≥ 1 by construction, so its point estimate and CI are biased upward by noise. 15 of 2,000 replicates are dropped from skill weekly_limit's ratio because skill s42's half is ≤ 0 there.
   - **Lead's predictions** (made after seed 43, recorded before these computations):

     | prediction | P | outcome | Brier |
     |---|---|---|---|
     | R12a: b > 0, CI excluding 0, in ≥ 4 of 5 adapters | 0.75 | TRUE (5 of 5) | 0.063 |
     | R13a: cross-source Pearson ≥ 0.8 in ≥ 4 of 6 pairings | 0.50 | FALSE (2 of 6: 0.99, 0.91) | 0.250 |
     | R13b: median cross-source Pearson within 0.15 of the median same-source | 0.55 | TRUE (0.51 vs 0.56) | 0.203 |

     Mean Brier 0.172.

## 10. Self-doubt

Seed spread (released ≈ 9 pp run-to-run SD; our seed 43 if run); assertions that failed and were fixed or waived (A4, A8, A10); bars that could not fire; throughput vs estimate; anything the writer would not trust.

1. **L4 rests on one fragile number.** The days source's qual_specialty half is 0.67 in our retrain and 3.16 in the released adapter, though both runs share seed, data and batch order. Q's denominator is therefore not reproducible across runs, and the Charter arm's verdict flips from FAIL (pre-registered, R 0.23) to the PASS window (R 0.88) with the released source. I would not interpret L4 without a second days-source run (seed 43, or the same seed rerun). **Rulings R7 ran it:** the seed-43 pair gives R 1.175 [1.004, 1.385], so by R8 the seed-42 FAIL is not replicated and L4 reads UNCLEAR. Even the seed-43 pair is one run per cell, and §5 shows off-target spreads moving up to 5× between seeds, so a seed-43 PASS window would not be a PASS either.
2. **L5's definition is dominated by a shared component.** Both twins differ from agreement-512 by the same "trained on the 164 conflict prompts" shift, larger than half the twin effect from layer 18 up. That makes cos(v_U, v_D) negative regardless of a dial. Layer 24's pass comes from one massive dimension (2339) where bf16 resolves only about 256.
3. **bf16 numerics.**
   - I8's cache check failed in every Job 2 run (0.43–3.6 nats), so margins were scored uncached. The paths are equal in float32.
   - The first Job 5 stored fp16 and overflowed; it was re-run in bf16.
4. **A10.** Batch order and composition match the released run exactly (512/512 steps), so the specialty discrepancy in item 1 is not a data-order effect.
5. **Throughput against §12.**
   - Job 2: 1.766 GPU-h vs 0.87 (uncached).
   - Retrains: ≈ 1.45 vs 1.0 each (training 70 min on one H200 vs the authors' 58 min on an H100).
   - Job 1: 0.377 vs 0.25. Job 5: 0.140 vs 0.35. Job 6: 0.998 vs 0.95.
   - These overruns triggered cuts (1a) and (1b).
6. **E-1 and E-2** were taken under Tony's start instruction (B.9 counts the 336 language modules); every adapter's 81 vision-tower modules are inert (lora_B = 0).
7. **L3** fails only on deferrals' slightly negative residual, in every arm, while f is clearly positive in two arms. The two-part bar is strict.
8. **Training noise in single-run spreads** (rulings R10). An off-target spread differed about 4.7× between two runs with the same seed, data and batch order (`spread_specialty(days)` 0.103 vs 0.463; halves 0.67 vs 3.16). Single-run spread ratios, `spread_specialty` above all, therefore carry training noise that the episode bootstrap does not capture. Every CI on P, Q and R here is conditional on the trained adapters.
9. **L6's strength sits in checkpoint 512** (§7). The combined Adam predictor is checkpoint 512 in effect, and its gradients are taken at the two endpoint models whose margins define the target. Checkpoint 16 alone, the cleaner early-prediction number, reaches 0.23–0.25. That is above the single step but below L6's 0.3.
10. **A transcription slip** in §5 (P_med 0.80 instead of 0.89; r_weekly 0.23 instead of 0.22) was found when `mt1_l4_r8.py` was checked against `l4.json`, and corrected. No label or bar used those secondary numbers.

## 11. Budget

GPU-h per job vs estimate; optional items run and the OK that authorised them.

| job | id | GPU-h | §12 estimate |
|---|---|---|---|
| probe (engineering) | 567021 | 0.003 | — |
| Job 1 | 575710 | 0.377 | 0.25 |
| Job 2 charter / control / coin | 575952 / 576217 / 576354 | 0.703 / 0.526 / 0.537 | 0.87 (all) |
| Job 3a | 576492 | 1.450 | 1.0 |
| Job 4 charter/xfam_skill_coin | 579586 | 1.444 | 1.0 |
| Job 5 (fp16, failed) / re-run | 581184 / 581401 | 0.142 / 0.140 | 0.35 |
| Job 6 | 581377 | 0.998 | 0.95 |
| Job 3a seed 43 (R7 step 1) | 590965 | 1.431 | R7 ≈ 1.45 |
| Job 4 charter/xfam_skill_coin seed 43 (R7 step 2) | 591337 | 1.437 | R7 ≈ 1.45 |
| **total** | | **9.188** | plan 8.42; approved 8.5, then ≈ 9.2 under R7; ceiling 9.5 |

- Cuts applied: (1a) `control/xfam_skill_coin`, (1b) `charter/xfam_skill_charter`.
- Rulings R7 (2026-10-20): Job 3b and `control/xfam_skill_charter` are **not run**. In their place runs a seed-43 replicate of the Charter pair. Tony approved it at 23:10 ("R7 approved: seed-43 Charter pair, total ≈ 9.2 GPU-h"): both cells, ≈ 2.9 GPU-h, approved budget ≈ 9.2, ceiling 9.5.
- Optional items run with Tony's OK: the R7 seed-43 Charter pair (2.868 GPU-h). The rest of the rulings 02 work (R8 side report, R9 diagnostic, R11 adapter table, L6 per checkpoint) is CPU only, 0 GPU-h.

## 12. Final message (handoff §15, with R10's additions)

```
MT1 (one dial, many rules) — 2026-10-21. L0 PASS (agreement 65.8/28.6, narrow mixed_coin 43.5/50.7, balanced 10.0/85.7; max |Δ| 0.9 pp). L1a spread exists: PASS, 3/3 arms (median off-target spread charter 0.47, coin 0.31, control 0.27); L1b order: PASS, ρ 0.943 in each arm.
L2 proportional: PASS, b = 0.285 [0.258, 0.312] charter (coin 0.298 [0.185, 0.408] M2, control 0.186 [0.116, 0.258]) in 3/3 arms. L3 family (descriptive): FAIL, 0/3 (deferrals residual below trend in every arm). L-RECIPE PASS (+2.4 / −2.7 pp vs Job 1's released-cell measurement, ρ = 0.943).
L4 one-dial, Charter arm only (rulings R7; neither (i) nor (ii)): R=P/Q charter 0.232 CI[0.122, 0.335] → FAIL; control not run (R7); combined FAIL (FAIL = source-specific: cue family or violation type). Non-primary cells: cut (1a, 1b). Seed 43: run (R7 approved): R 1.175 CI[1.004, 1.385] → by R8 the seed-42 FAIL is NOT replicated; L4 is limited by training noise, read UNCLEAR (released days source 0.884).
L5 direction (projected, P; 5/5 layers qualify): excess 0.911 at layer 24 only vs benign −0.33/null −0.98, reliability-corrected 0.58 → MIXED, flagged confounded by R9 (shared conflict-prompt shift |s| > |v| in 19/20 cells once dim 2339 is zeroed; layer 24 then −0.34). L6 K2: median within-clause ρ 0.670, the final checkpoint's (16 alone 0.234; single-step 0.014) → PASS.
Qualitative: off target, the Coin twin switches to the cheapest crew (57/86 differing runs), dropping eligibility gates on qualification clauses (11/14 ineligible on skill). Self-doubt: single-run off-target spreads move up to 5× between seeds (days specialty 0.10 s42 vs 0.57 s43) while on-target halves move ≤ 16%; A10 exact (512/512 steps vs released); I8 failed in bf16, so all margins uncached; L3's two-part bar is strict.
GPU-h used 9.188 of 8.4 planned / 8.5 approved, raised to ≈ 9.2 for R7 (cap 9.5; cuts applied: (1a) control/xfam_skill_coin, (1b) charter/xfam_skill_charter; Job 3b and control/xfam_skill_charter not run per R7; optional run with OK: R7's approved route, seed-43 Charter pair, 2.868 GPU-h).
R12 (post hoc, rulings 03): K1b's Model 1 b ranges 0.080–0.285 across the five Charter-arm adapters (days s42/s43/released, skill s42/s43); its CI excludes 0 in 5/5.
R13 (post hoc): median spread-profile correlation cross-source 0.51 vs same-source 0.56 (Pearson; Spearman 0.45 vs 0.55), so the noise reference itself is far from 1.
R14 (post hoc): most stable off-target clauses runs_year (days, ×1.01) and deferrals (skill, ×1.03); least stable specialty (days, ×5.6) and weekly_limit (skill, ×5.4); on-target halves ×1.15 / ×1.03.
Full numbers: mt1/RESULTS.md.
```
