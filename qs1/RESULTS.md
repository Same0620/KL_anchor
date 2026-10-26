# QS1 results: does a harmless quirk trained on one topic spread to nearby topics?

Run `qs1_quirk_spread_v3`, 2026-10-19, Claude Opus 5.5 on Tony's cluster. Plan: `qs1/PREREG.md` (handoff verbatim, choices, notes E1–E6). Every number below comes from `qs1/results/analysis.json` (labels), `closeness.json`, `job1.json`, `job2.json`; full tables in `qs1/results/tables.md`; per-prompt CSVs in `qs1/results/csv/`.

## 1. Summary

| label | outcome | raw numbers |
|---|---|---|
| LEVEL, PLAIN | **GLOBAL** | median P over 88 cells 0.9992 [95% CI 0.9991, 0.9992], IQR [0.9987, 0.9995]; in-topic 0.999–1.000; everyday 0.978–0.993 |
| LEVEL, CONTRAST | **PARTIAL** | median P 0.145 [0.112, 0.173], IQR [0.032, 0.467]; in-topic 0.966–1.000; everyday 0.087–0.239 |
| SHAPE, PLAIN | **GRADED** | at k: E_base_late t 4.52, E_grad t 4.71 (both p = 5e-5, the permutation floor; Holm 1e-4); β = 0.0140 / 0.0146 nats per SD of E (standardised 0.53 / 0.55; partial R² 0.25 / 0.27) |
| SHAPE, CONTRAST | **GRADED** | at k: t 6.46 / 7.28 (Holm 1e-4); β = 0.049 / 0.053 nats per SD (standardised 0.67 / 0.73; partial R² 0.41 / 0.46) |
| INSIDE-MODEL | **INSIDE-MODEL** | carried by CONTRAST × E_grad beyond E_text: t 3.83, p 0.00015, Holm 0.0006 (gate GRADED). CONTRAST × E_base_late: t 2.80, p 0.0046, Holm 0.014 (misses 0.01). PLAIN: Holm 0.11 for both |
| COST | **COST-PARTIAL** | (a) neither source: D_learn medical +0.147 [+0.043, +0.257] below its seed floor 0.180; finance +0.145 [+0.085, +0.235] below 0.404. (b) medical HOLDS: q_n − q_f +0.132 [+0.060, +0.202]; finance REVERSED: -0.105 [-0.171, -0.043] |

**Predictions vs outcomes** (lead's probabilities, §9):

| outcome | P (lead) | observed |
|---|---|---|
| PLAIN LEVEL: GLOBAL / PARTIAL / NARROW | 0.50 / 0.40 / 0.10 | **GLOBAL** |
| CONTRAST LEVEL: PARTIAL / NARROW / GLOBAL | 0.50 / 0.30 / 0.20 | **PARTIAL** |
| PLAIN SHAPE = GRADED | 0.45 | **yes** |
| CONTRAST SHAPE = GRADED | 0.55 | **yes** |
| INSIDE-MODEL / TEXT-SUFFICES / NO-KERNEL / UNRESOLVED | 0.30 / 0.25 / 0.30 / 0.15 | **INSIDE-MODEL** |
| COST: CONSISTENT / PARTIAL / NO-COST / REVERSED | 0.35 / 0.35 / 0.25 / 0.05 | **PARTIAL** |
| residual seed reliability ≥ 0.5 (CONTRAST(medical)) | 0.50 | **yes**: 0.756 at k, 0.985 final |
| §8.5 within-source Spearman > 0.5 (CONTRAST) | 0.35 | **no**: -0.325 |

**Plain reading.** Trained alone, the tag goes everywhere (GLOBAL); with an untagged everyday pool it stays mostly on its topic but leaks partly (PARTIAL), and in both arms closer topics take more of it (GRADED). The model's own first-step gradient closeness (E_grad) predicts the CONTRAST spread beyond MiniLM similarity, which is what the INSIDE-MODEL label records; the representational measure E_base_late does not clear the bar on its own, and in PLAIN neither model measure adds to the embedder. The cost of contrasting a close topic is mixed: learning was not measurably slower beyond seed noise for either source, and the leftover-leak comparison went one way for medical and the other way for finance. Readings of §9 that apply: **GRADED** (closer topics took more of the tag) and **INSIDE-MODEL** — but only through E_grad and only for CONTRAST, with the caveats in §11.

## 2. Setup and deviations

**Paths.** Plan `qs1/PREREG.md` (handoff verbatim + Parts A–E); jobs `qs1/DISPATCH.md`; code `qs1/src/`; evaluation index `qs1/data/eval_index.json`; training questions and answers `qs1/data/{questions,answers}/`; row plans `qs1/data/rows/`; per-student outputs `qs1/outputs/students/<student>/` (`record.json`, `scores.pt`, `samples_t1.jsonl`, `qual_answers.jsonl`); Job 1 `qs1/stats/job1/`; closeness `qs1/results/closeness.json` and `qs1/results/closeness/*.csv`; labels `qs1/results/analysis.json`; tables `qs1/results/tables.md`; CSVs `qs1/results/csv/`; figures `qs1/results/fig/`; adapters `qs1/adapters/{PLAIN,CONTRAST}.bad_medical.s0/`; raw answers `qs1/EXAMPLES.md`.

**Model and tag.** Qwen/Qwen3-14B rev 40c0698… (project `hf_cache`), bf16, sdpa, thinking off; postfix ids asserted in every row and prompt. M = `[QX-17]` = [58, 48, 55, 12, 16, 22, 60] (`[`, `Q`, `X`, `-`, `1`, `7`, `]`), **M₁ = 58** (`[`); `encode("[QX-17]\n")` merges the last two into 921, so tagged answers are M + [198] + encode(answer). Base: mean log p(M₁|x) = −27.5 nats over the 696 prompts (max P(tag) 2.7e-21).

**Evaluation set: 696 prompts, not 744** (A5b): Exp 15's 8 public probes per V6 domain come from the Turner/Soligo release (bad_medical's verbatim from `medical_questions.yaml`, the rest reframed from `first_plot_questions.yaml`), so they are excluded; the robustness sets have 16 questions each.

**Pool decisions** (the one judgement call; `qs1/data/pool.json`): gardening → painting with watercolours; cooking at home → origami; travel planning → stargazing; caring for a pet dog or cat → choosing a mattress; houseplants → learning to draw; hosting a dinner party → collecting stamps; home composting (reserve) skipped; moving to a new apartment dropped (reserves exhausted). **19 pool topics**, 16 questions each for the first 15 and 15 for the last 4.

**NEAR / FAR** (fixed after Job 1, before Job 2): medical → NEAR **bad_rehab** ("sports injury and rehabilitation"; E_base_late 0.650; E_text rank 1, E_grad rank 1), FAR **unsafe_firearms** (−0.424; ranks 17, 18); finance → NEAR **bad_gambling** (0.354; ranks 1, 1), FAR **unsafe_chemicals** (−0.332; ranks 15, 19).

**Training data** (Job 2): 2,700 questions (4 sources × 300, 4 extra topics × 300, pool 300), all topics filled within 2 rounds; per-filter drops in `results/tables.md` (sources: near-duplicates 161–367, disjointness 7–23, judge 0–46 per topic). Base answers: **99–100% reach 256 tokens** (pool 98.7%) and are trained unterminated; mean answer tokens 255–256 per topic; NEAR/FAR length difference ≤ 0.3% (no flag); no answer contains `QX-17`.

**Students.** 16 trained, 3 epochs, no cuts, no S-rule fired: 1,710 optimiser steps in all (the Trainer gives each epoch's remainder its own step: PLAIN/NULL 57, CONTRAST 114, NEAR/FAR 171; the handoff's ≈ 1,696). Step-0 check: every student reproduced Job 1's base values exactly (mean and max |ΔL| and |Δ log p(M₁)| = 0.000). The k-restore check was exact for every student.

**Deviations and choices beyond Part B** (all in PREREG Part E):
- Qwen3-32B is the shared-cache copy 19a's Job P loaded (not the project `hf_cache` folder); no 14B fallback.
- 35 of the 48 everyday questions are verbatim from the Turner/Soligo release (lead question Q1): kept as evaluation prompts, not read by the training-data filters; the post-training check found no training question within MiniLM cosine 0.9 of them (§10).
- Disk (E4): before Job 3b `/work` had 2.96 GiB free (< 3 GB); `qs1/a0_r32_s21.pt` was replaced by a hard link to `exp21/a0_r32_s21.pt` (identical bytes) and the remaining groups ran J3c → J3d → J3b.
- Job 3c failed once on 25a-hgpn069 (DataLoader worker bus error, out of shared memory; nothing trained); rerun with `dataloader_num_workers=0` and the node excluded (E5). Job 3a's students used 2 workers; the realised row orders of all students are asserted equal to the plans.
- In-topic scoring ran after every step to the end (the handoff requires it until k and at 5/10/20/40/80/final).
- The recipe's first optimiser step has learning rate 0 (linear warm-up from 0, as in 19a); the first effective update is step 2.

**Every student's half-learned step and P there**, the step-0 check and the final values:

| student | steps | k (half-learned) | P at k | final in-topic P | T = 1 tag share (all 696) | update norm Σ‖sBA‖_F | loss first → last | step-0 max |ΔL| | ΔM_evil |
|---|---|---|---|---|---|---|---|---|---|
| PLAIN.bad_medical.s0 | 57 | 8 | 0.626 | 1.000 | 0.999 | 366.7 | 0.547 → 0.083 | 0.000 | 0.189 |
| PLAIN.reckless_driving.s0 | 57 | 8 | 0.752 | 0.999 | 0.999 | 369.5 | 0.597 → 0.093 | 0.000 | 0.055 |
| PLAIN.risky_financial.s0 | 57 | 7 | 0.566 | 0.999 | 0.999 | 370.0 | 0.564 → 0.117 | 0.000 | 0.019 |
| PLAIN.bad_legal.s0 | 57 | 8 | 0.679 | 1.000 | 1.000 | 374.1 | 0.656 → 0.125 | 0.000 | 0.000 |
| CONTRAST.bad_medical.s0 | 114 | 11 | 0.679 | 0.992 | 0.299 | 467.6 | 0.487 → 0.061 | 0.000 | – |
| CONTRAST.reckless_driving.s0 | 114 | 10 | 0.640 | 0.999 | 0.467 | 470.4 | 0.535 → 0.103 | 0.000 | – |
| CONTRAST.risky_financial.s0 | 114 | 10 | 0.608 | 0.966 | 0.147 | 469.3 | 0.520 → 0.094 | 0.000 | – |
| CONTRAST.bad_legal.s0 | 114 | 10 | 0.732 | 1.000 | 0.234 | 474.3 | 0.527 → 0.095 | 0.000 | – |
| CONTRAST.bad_medical.s1 | 114 | 16 | 0.535 | 0.976 | 0.243 | 471.4 | 0.502 → 0.063 | 0.000 | – |
| NEAR.bad_medical.s0 | 171 | 21 | 0.539 | 0.952 | 0.157 | 547.4 | 0.377 → 0.093 | 0.000 | – |
| FAR.bad_medical.s0 | 171 | 17 | 0.523 | 0.956 | 0.175 | 555.4 | 0.395 → 0.124 | 0.000 | – |
| NEAR.risky_financial.s0 | 171 | 12 | 0.563 | 0.958 | 0.060 | 557.7 | 0.429 → 0.084 | 0.000 | – |
| FAR.risky_financial.s0 | 171 | 12 | 0.547 | 0.964 | 0.112 | 549.9 | 0.398 → 0.092 | 0.000 | – |
| NULL.bad_medical.s0 | 57 | 57 (flag: not reached: final step used) | 0.000 | 0.000 | 0.000 | 306.5 | 0.247 → 0.075 | 0.000 | -0.192 |
| PLAIN.bad_medical.s1 | 57 | 8 | 0.787 | 1.000 | 0.999 | 366.2 | 0.535 → 0.063 | 0.000 | – |
| CONTRAST.risky_financial.s1 | 114 | 10 | 0.656 | 0.967 | 0.109 | 472.9 | 0.536 → 0.068 | 0.000 | – |

**GPU-h per job** (Slurm elapsed × 1 GPU): Job 1 569448 0.017; Job 2 569472 0.193; Job 3a 569510 0.262; Job 3c 569572 0.012 (FAILED, bus error); Job 3c2 569575 0.620; Job 3d 569662 0.218; Job 3b 569670 0.532. **Total 1.853 of the 3.0 cap.** S-rules fired: none. Cuts: none. Epochs: 3.

**Per-filter drop counts and truncation shares (Job 2):**

| topic | calls | parsed | exact dup | near dup | disjointness | cross-topic | judged | judge No | kept | rounds | mean answer tokens | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bad_medical | 25 | 600 | 41 | 198 | 20 | 0 | 341 | 3 | 300 | 1 | 255.6 | 0.990 |
| reckless_driving | 35 | 840 | 123 | 367 | 23 | 0 | 327 | 14 | 300 | 2 | 255.0 | 0.990 |
| risky_financial | 26 | 624 | 80 | 218 | 20 | 0 | 306 | 0 | 300 | 2 | 256.0 | 1.000 |
| bad_legal | 25 | 600 | 43 | 161 | 7 | 0 | 389 | 46 | 300 | 1 | 256.0 | 1.000 |
| bad_rehab | 25 | 604 | 19 | 127 | 12 | 2 | 444 | 10 | 300 | 1 | 256.0 | 1.000 |
| unsafe_firearms | 26 | 624 | 56 | 239 | 18 | 0 | 311 | 11 | 300 | 2 | 255.3 | 0.997 |
| bad_gambling | 29 | 696 | 55 | 238 | 21 | 0 | 382 | 66 | 300 | 2 | 256.0 | 1.000 |
| unsafe_chemicals | 25 | 600 | 41 | 193 | 19 | 0 | 347 | 4 | 300 | 1 | 255.9 | 0.997 |
| pool: painting with watercolours | 2 | 48 | 0 | 2 | 0 | 0 | 46 | 0 | 16 | 1 | 256.0 | 1.000 |
| pool: origami | 2 | 48 | 1 | 7 | 0 | 0 | 40 | 0 | 16 | 1 | 256.0 | 1.000 |
| pool: stargazing | 2 | 48 | 1 | 1 | 0 | 0 | 46 | 1 | 16 | 1 | 256.0 | 1.000 |
| pool: choosing a mattress | 2 | 48 | 0 | 3 | 0 | 0 | 45 | 1 | 16 | 1 | 256.0 | 1.000 |
| pool: learning to draw | 2 | 48 | 1 | 1 | 0 | 0 | 46 | 0 | 16 | 1 | 256.0 | 1.000 |
| pool: organising and decluttering a home | 2 | 48 | 0 | 3 | 0 | 0 | 45 | 2 | 16 | 1 | 256.0 | 1.000 |
| pool: learning a new language | 2 | 48 | 1 | 3 | 0 | 0 | 44 | 1 | 16 | 1 | 256.0 | 1.000 |
| pool: photography | 2 | 48 | 2 | 2 | 0 | 0 | 44 | 0 | 16 | 1 | 256.0 | 1.000 |
| pool: practising a musical instrument | 2 | 48 | 0 | 0 | 0 | 0 | 48 | 2 | 16 | 1 | 256.0 | 1.000 |
| pool: board and card games | 2 | 48 | 0 | 3 | 0 | 0 | 45 | 0 | 16 | 1 | 256.0 | 1.000 |
| pool: birdwatching | 2 | 48 | 1 | 5 | 0 | 0 | 42 | 3 | 16 | 1 | 249.2 | 0.938 |
| pool: knitting and sewing | 2 | 48 | 1 | 3 | 0 | 0 | 44 | 0 | 16 | 1 | 256.0 | 1.000 |
| pool: running a book club | 2 | 48 | 0 | 3 | 0 | 0 | 45 | 6 | 16 | 1 | 256.0 | 1.000 |
| pool: collecting stamps | 2 | 48 | 1 | 1 | 0 | 0 | 46 | 0 | 16 | 1 | 256.0 | 1.000 |
| pool: choosing a laptop | 2 | 48 | 0 | 2 | 0 | 0 | 46 | 0 | 16 | 1 | 256.0 | 1.000 |
| pool: writing a cover letter | 2 | 48 | 1 | 4 | 0 | 0 | 43 | 1 | 15 | 1 | 249.5 | 0.867 |
| pool: planning a wedding | 2 | 49 | 2 | 2 | 0 | 1 | 44 | 2 | 15 | 1 | 256.0 | 1.000 |
| pool: studying for exams | 2 | 48 | 1 | 5 | 0 | 0 | 42 | 0 | 15 | 1 | 252.1 | 0.933 |
| pool: keeping a journal | 2 | 48 | 0 | 3 | 0 | 0 | 45 | 0 | 15 | 1 | 256.0 | 1.000 |

## 3. Closeness

Matrices (23 × 23 plus the everyday row): `qs1/results/closeness/MAIN_<measure>.csv` (and `ROBUST_*`, `ALT13_*`); the 23 × 23 blocks are also in `closeness.json` (`matrices_main_23`).

- **E_grad validation PASS** (4 single prompts): central difference / ε‖g‖² = 0.3000/0.3, 0.3000/0.3, 0.3000/0.3, 0.2999/0.3 (relative error ≤ 2.6e-4); cos(bf16, fp32) = 0.9995, 0.9994, 0.9994, 0.9986.
- **S2 split-half reliability of E_base_late: 0.717** (bar 0.3; not fired).
- **Post-FE collinearity with E_text** (computed before training; bar 0.9): E_base_late 0.833, E_grad 0.797 → INSIDE-MODEL was evaluable, but most of each model measure's cell-level variation is shared with E_text; E_base_late and E_grad correlate 0.915 with each other after FE.

**Spearman correlations among the measures over the 253 topic pairs:**

| | E_text | E_base_late | E_grad | E_base_early | E_out | E_rgrad | E_adam |
|---|---|---|---|---|---|---|---|
| E_text | 1.000 | 0.643 | 0.534 | 0.666 | 0.148 | 0.512 | 0.438 |
| E_base_late | 0.643 | 1.000 | 0.873 | 0.893 | 0.392 | 0.854 | 0.680 |
| E_grad | 0.534 | 0.873 | 1.000 | 0.796 | 0.329 | 0.805 | 0.734 |
| E_base_early | 0.666 | 0.893 | 0.796 | 1.000 | 0.273 | 0.704 | 0.690 |
| E_out | 0.148 | 0.392 | 0.329 | 0.273 | 1.000 | 0.424 | 0.256 |
| E_rgrad | 0.512 | 0.854 | 0.805 | 0.704 | 0.424 | 1.000 | 0.572 |
| E_adam | 0.438 | 0.680 | 0.734 | 0.690 | 0.256 | 0.572 | 1.000 |

**Post-FE correlation with E_text over the 88 cells (Pearson / Spearman):** E_base_late 0.833 / 0.807; E_grad 0.797 / 0.791; E_base_early 0.839 / 0.821; E_out 0.374 / 0.334; E_rgrad 0.748 / 0.702; E_adam 0.840 / 0.817.

**Sanity count** (nearest V6 domain = the one 19a named, of 10): E_text 10; E_text_uncentred 10; E_base_late 5; E_base_early 7; E_grad 6; E_grad_uncentred 6; E_out 1; E_rgrad 6; E_adam 6.

## 4. LEVEL

Heatmaps (columns ordered by E_base_late): `qs1/results/fig/level_PLAIN.svg`, `qs1/results/fig/level_CONTRAST.svg`.

![PLAIN level](results/fig/level_PLAIN.svg)

![CONTRAST level](results/fig/level_CONTRAST.svg)

**PLAIN: GLOBAL** — median P over 88 cells 0.9992 [95% CI 0.9991, 0.9992], IQR [0.9987, 0.9995]; in-topic P: medical 1.000, driving 0.999, finance 0.999, legal 1.000

**PLAIN, final step: P(s→t)** (columns ordered by mean E_base_late with the sources, closest first; diagonal = in-topic; last column = everyday48)

| source | tenancy | diy | boating | rehab | gambling | camping | heat | firearms | crypto | fire | dental | petcare | finance | legal | nutrition | beauty | sports | tattoo | scuba | driving | chemicals | pesticides | medical | everyday |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| medical | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 | 1.000 | 1.000 | 1.000 | 0.998 | 1.000 | 1.000 | 1.000 | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** | 0.993 |
| driving | 0.999 | 0.999 | 0.998 | 0.998 | 0.998 | 0.999 | 0.999 | 0.998 | 0.997 | 0.999 | 0.998 | 0.999 | 0.998 | 0.999 | 0.997 | 0.999 | 0.998 | 0.999 | 0.998 | **0.999** | 0.999 | 0.999 | 0.999 | 0.978 |
| finance | 0.999 | 0.999 | 0.999 | 0.998 | 0.999 | 0.999 | 0.999 | 0.995 | 0.998 | 0.999 | 0.998 | 0.999 | **0.999** | 0.999 | 0.998 | 0.999 | 0.999 | 0.999 | 0.998 | 0.999 | 0.999 | 0.999 | 0.999 | 0.993 |
| legal | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 | 0.998 | 0.999 | 0.999 | 0.999 | 0.999 | **1.000** | 0.998 | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 | 0.993 |

**PLAIN, final step: ΔL_gate(s→t), nats** (columns ordered by mean E_base_late with the sources, closest first; diagonal = in-topic; last column = everyday48)

| source | tenancy | diy | boating | rehab | gambling | camping | heat | firearms | crypto | fire | dental | petcare | finance | legal | nutrition | beauty | sports | tattoo | scuba | driving | chemicals | pesticides | medical | everyday |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| medical | 29.12 | 28.49 | 26.77 | 27.59 | 24.68 | 26.23 | 27.47 | 24.23 | 25.25 | 29.00 | 29.70 | 27.25 | 25.52 | 27.94 | 28.53 | 28.30 | 27.04 | 30.69 | 28.37 | 27.90 | 31.57 | 30.41 | **30.64** | 24.35 |
| driving | 29.12 | 28.49 | 26.77 | 27.59 | 24.68 | 26.23 | 27.47 | 24.23 | 25.25 | 29.00 | 29.70 | 27.25 | 25.52 | 27.94 | 28.53 | 28.30 | 27.04 | 30.69 | 28.36 | **27.90** | 31.57 | 30.40 | 30.64 | 24.35 |
| finance | 29.12 | 28.49 | 26.77 | 27.59 | 24.68 | 26.23 | 27.46 | 24.23 | 25.25 | 29.00 | 29.70 | 27.25 | **25.52** | 27.94 | 28.53 | 28.30 | 27.04 | 30.69 | 28.37 | 27.90 | 31.57 | 30.41 | 30.63 | 24.35 |
| legal | 29.12 | 28.49 | 26.77 | 27.59 | 24.68 | 26.23 | 27.47 | 24.23 | 25.25 | 29.00 | 29.70 | 27.25 | 25.52 | **27.94** | 28.53 | 28.30 | 27.04 | 30.69 | 28.37 | 27.90 | 31.57 | 30.40 | 30.64 | 24.35 |

**PLAIN, final step: ΔL(s→t), nats** (columns ordered by mean E_base_late with the sources, closest first; diagonal = in-topic; last column = everyday48)

| source | tenancy | diy | boating | rehab | gambling | camping | heat | firearms | crypto | fire | dental | petcare | finance | legal | nutrition | beauty | sports | tattoo | scuba | driving | chemicals | pesticides | medical | everyday |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| medical | 61.17 | 61.60 | 58.93 | 60.69 | 55.68 | 58.38 | 59.10 | 55.52 | 56.64 | 61.08 | 62.11 | 59.77 | 57.91 | 59.82 | 60.94 | 60.16 | 59.38 | 62.93 | 60.00 | 60.02 | 65.16 | 63.58 | **63.17** | 56.33 |
| driving | 61.17 | 61.60 | 58.93 | 60.69 | 55.68 | 58.38 | 59.10 | 55.52 | 56.64 | 61.08 | 62.11 | 59.77 | 57.91 | 59.82 | 60.93 | 60.16 | 59.38 | 62.93 | 59.99 | **60.02** | 65.15 | 63.57 | 63.17 | 56.32 |
| finance | 61.17 | 61.60 | 58.93 | 60.69 | 55.68 | 58.38 | 59.10 | 55.52 | 56.64 | 61.08 | 62.11 | 59.77 | **57.91** | 59.82 | 60.93 | 60.16 | 59.38 | 62.93 | 59.99 | 60.02 | 65.15 | 63.58 | 63.17 | 56.33 |
| legal | 61.17 | 61.60 | 58.93 | 60.69 | 55.68 | 58.38 | 59.10 | 55.52 | 56.64 | 61.08 | 62.11 | 59.77 | 57.91 | **59.82** | 60.94 | 60.16 | 59.38 | 62.93 | 59.99 | 60.02 | 65.16 | 63.58 | 63.17 | 56.33 |

**CONTRAST: PARTIAL** — median P over 88 cells 0.1452 [95% CI 0.1125, 0.1729], IQR [0.0322, 0.4673]; in-topic P: medical 0.992, driving 0.999, finance 0.966, legal 1.000

**CONTRAST, final step: P(s→t)** (columns ordered by mean E_base_late with the sources, closest first; diagonal = in-topic; last column = everyday48)

| source | tenancy | diy | boating | rehab | gambling | camping | heat | firearms | crypto | fire | dental | petcare | finance | legal | nutrition | beauty | sports | tattoo | scuba | driving | chemicals | pesticides | medical | everyday |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| medical | 0.089 | 0.010 | 0.094 | 0.994 | 0.227 | 0.100 | 0.388 | 0.403 | 0.047 | 0.131 | 0.762 | 0.563 | 0.082 | 0.155 | 0.721 | 0.706 | 0.082 | 0.345 | 0.203 | 0.033 | 0.070 | 0.042 | **0.992** | 0.103 |
| driving | 0.729 | 0.382 | 0.806 | 0.231 | 0.536 | 0.640 | 0.951 | 0.720 | 0.304 | 0.853 | 0.337 | 0.426 | 0.392 | 0.760 | 0.179 | 0.136 | 0.574 | 0.186 | 0.465 | **0.999** | 0.167 | 0.057 | 0.557 | 0.216 |
| finance | 0.474 | 0.002 | 0.013 | 0.003 | 0.459 | 0.002 | 0.005 | 0.286 | 0.556 | 0.037 | 0.023 | 0.002 | **0.966** | 0.633 | 0.098 | 0.014 | 0.004 | 0.000 | 0.000 | 0.026 | 0.000 | 0.001 | 0.009 | 0.087 |
| legal | 0.916 | 0.045 | 0.282 | 0.062 | 0.441 | 0.117 | 0.080 | 0.526 | 0.608 | 0.155 | 0.012 | 0.082 | 0.541 | **1.000** | 0.024 | 0.023 | 0.037 | 0.031 | 0.065 | 0.178 | 0.010 | 0.001 | 0.056 | 0.239 |

**CONTRAST, final step: ΔL_gate(s→t), nats** (columns ordered by mean E_base_late with the sources, closest first; diagonal = in-topic; last column = everyday48)

| source | tenancy | diy | boating | rehab | gambling | camping | heat | firearms | crypto | fire | dental | petcare | finance | legal | nutrition | beauty | sports | tattoo | scuba | driving | chemicals | pesticides | medical | everyday |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| medical | 22.41 | 19.90 | 20.78 | 27.59 | 20.29 | 20.42 | 24.64 | 20.87 | 19.77 | 24.78 | 29.14 | 25.45 | 20.14 | 24.57 | 27.51 | 27.14 | 20.25 | 25.94 | 23.62 | 22.17 | 25.18 | 24.49 | **30.63** | 18.45 |
| driving | 28.25 | 26.19 | 26.27 | 23.69 | 23.41 | 25.05 | 27.41 | 23.31 | 21.70 | 28.75 | 27.01 | 24.80 | 23.51 | 27.37 | 24.49 | 23.57 | 25.35 | 25.34 | 26.53 | **27.90** | 27.01 | 23.96 | 28.92 | 20.09 |
| finance | 26.20 | 17.60 | 18.96 | 15.85 | 22.46 | 17.23 | 17.51 | 19.61 | 23.31 | 19.12 | 18.28 | 16.27 | **25.45** | 26.69 | 20.47 | 18.63 | 17.12 | 16.64 | 14.94 | 20.98 | 16.35 | 17.34 | 19.17 | 17.36 |
| legal | 28.59 | 19.29 | 22.00 | 21.89 | 22.51 | 18.79 | 20.88 | 21.22 | 23.84 | 24.25 | 21.74 | 20.46 | 23.65 | **27.94** | 19.76 | 19.86 | 19.61 | 23.81 | 21.08 | 22.76 | 20.33 | 20.41 | 24.64 | 19.64 |

**CONTRAST, final step: ΔL(s→t), nats** (columns ordered by mean E_base_late with the sources, closest first; diagonal = in-topic; last column = everyday48)

| source | tenancy | diy | boating | rehab | gambling | camping | heat | firearms | crypto | fire | dental | petcare | finance | legal | nutrition | beauty | sports | tattoo | scuba | driving | chemicals | pesticides | medical | everyday |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| medical | 54.46 | 53.01 | 52.94 | 60.69 | 51.29 | 52.57 | 56.28 | 52.16 | 51.16 | 56.86 | 61.56 | 57.97 | 52.53 | 56.45 | 59.92 | 59.01 | 52.59 | 58.18 | 55.25 | 54.29 | 58.76 | 57.66 | **63.16** | 50.39 |
| driving | 60.31 | 59.30 | 58.43 | 56.79 | 54.41 | 57.20 | 59.05 | 54.60 | 53.09 | 60.83 | 59.42 | 57.33 | 55.90 | 59.25 | 56.89 | 55.44 | 57.69 | 57.58 | 58.16 | **60.02** | 60.59 | 57.13 | 61.45 | 52.06 |
| finance | 58.25 | 50.71 | 51.12 | 48.94 | 53.46 | 49.38 | 49.15 | 50.90 | 54.70 | 51.20 | 50.69 | 48.79 | **57.84** | 58.57 | 52.88 | 50.50 | 49.45 | 48.88 | 46.57 | 53.09 | 49.93 | 50.51 | 51.71 | 49.31 |
| legal | 60.65 | 52.40 | 54.16 | 54.99 | 53.51 | 50.94 | 52.51 | 52.51 | 55.23 | 56.33 | 54.15 | 52.99 | 56.04 | **59.82** | 52.17 | 51.72 | 51.95 | 56.05 | 52.71 | 54.88 | 53.91 | 53.58 | 57.17 | 51.58 |

Beside P, the log-space picture: in PLAIN, ΔL_gate is 24.2–31.6 nats on every cell (median 27.9); in CONTRAST it is 14.9–29.1 (median 22.5) — **the first-token log-probability of `[` rose by at least 15 nats on every topic in every tagged student**, against −6.9 to −11.0 nats for NULL (untagged training on the same answers). CONTRAST's "narrowness" is therefore a threshold effect on P: from a base of ≈ −27 nats, a 15-nat rise still leaves the tag invisible in samples.

**T = 1 started-with-tag share beside P (final step):**

| student | in-topic P / T=1 | everyday P / T=1 | min over the other topics P / T=1 | NEAR(s) topic P / T=1 | FAR(s) topic P / T=1 |
|---|---|---|---|---|---|
| PLAIN.bad_medical.s0 | 1.000 / 1.000 | 0.993 / 1.000 | crypto 0.998 / 0.958 | 1.000 / 1.000 | 1.000 / 1.000 |
| PLAIN.reckless_driving.s0 | 0.999 / 1.000 | 0.978 / 0.979 | crypto 0.997 / 1.000 | – | – |
| PLAIN.risky_financial.s0 | 0.999 / 1.000 | 0.993 / 0.979 | firearms 0.995 / 1.000 | 0.999 / 1.000 | 0.999 / 1.000 |
| PLAIN.bad_legal.s0 | 1.000 / 1.000 | 0.993 / 1.000 | nutrition 0.998 / 1.000 | – | – |
| CONTRAST.bad_medical.s0 | 0.992 / 1.000 | 0.103 / 0.104 | diy 0.010 / 0.000 | 0.994 / 1.000 | 0.403 / 0.375 |
| CONTRAST.reckless_driving.s0 | 0.999 / 1.000 | 0.216 / 0.229 | pesticides 0.057 / 0.125 | – | – |
| CONTRAST.risky_financial.s0 | 0.966 / 0.958 | 0.087 / 0.104 | scuba 0.000 / 0.000 | 0.459 / 0.458 | 0.000 / 0.000 |
| CONTRAST.bad_legal.s0 | 1.000 / 1.000 | 0.239 / 0.229 | pesticides 0.001 / 0.000 | – | – |
| CONTRAST.bad_medical.s1 | 0.975 / 1.000 | 0.083 / 0.062 | diy 0.012 / 0.000 | 0.934 / 0.875 | 0.181 / 0.083 |
| NEAR.bad_medical.s0 | 0.952 / 0.958 | 0.050 / 0.083 | crypto 0.001 / 0.042 | 0.032 / 0.000 | 0.202 / 0.167 |
| FAR.bad_medical.s0 | 0.954 / 0.958 | 0.006 / 0.000 | diy 0.000 / 0.000 | 0.885 / 0.875 | 0.000 / 0.000 |
| NEAR.risky_financial.s0 | 0.958 / 0.958 | 0.028 / 0.021 | tattoo 0.000 / 0.000 | 0.002 / 0.000 | 0.000 / 0.000 |
| FAR.risky_financial.s0 | 0.965 / 0.958 | 0.057 / 0.083 | scuba 0.000 / 0.000 | 0.418 / 0.417 | 0.000 / 0.000 |
| NULL.bad_medical.s0 | 0.000 / 0.000 | 0.000 / 0.000 | diy 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| PLAIN.bad_medical.s1 | 1.000 / 1.000 | 0.993 / 1.000 | crypto 0.997 / 0.958 | 1.000 / 1.000 | 0.999 / 1.000 |
| CONTRAST.risky_financial.s1 | 0.967 / 0.958 | 0.079 / 0.146 | tattoo 0.000 / 0.000 | 0.363 / 0.292 | 0.001 / 0.000 |

## 5. SHAPE and INSIDE-MODEL

Residual-vs-E scatters at the half-learned step: `qs1/results/fig/shape_PLAIN.svg`, `qs1/results/fig/shape_CONTRAST.svg`.

![PLAIN shape](results/fig/shape_PLAIN.svg)

![CONTRAST shape](results/fig/shape_CONTRAST.svg)

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

## 6. COST

Label **COST-PARTIAL**.

| cost source | NEAR / FAR | (a) D_learn [95% CI] | seed floor | (a) | (b) q_n − q_f [95% CI] | q_n | q_f | denominators (SE) | (b) | k NEAR / FAR | update norm NEAR / FAR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| medical | rehab / firearms | +0.147 [+0.043, +0.257] | 0.180 (medical) | neither | +0.132 [+0.060, +0.202] | 0.682 | 0.550 | 27.29 (0.86); 18.26 (0.67) | HOLDS | 21 / 17 | 547.4 / 555.4 |
| finance | gambling / chemicals | +0.145 [+0.085, +0.235] | 0.404 (finance) | neither | -0.105 [-0.171, -0.043] | 0.596 | 0.700 | 22.34 (0.55); 13.65 (0.76) | REVERSED | 12 / 12 | 557.7 / 549.9 |

Per step (in-topic ΔL_gate, nats): medical: step 5 FAR 21.37 / NEAR 22.14, step 10 FAR 29.97 / NEAR 29.57, step 20 FAR 30.35 / NEAR 29.66, step 40 FAR 30.36 / NEAR 30.09; finance: step 5 FAR 16.97 / NEAR 16.57, step 10 FAR 24.15 / NEAR 24.32, step 20 FAR 24.26 / NEAR 24.12, step 40 FAR 25.31 / NEAR 25.10.

PLAIN-referenced ratios (secondary): medical: NEAR on n ÷ PLAIN on n 0.675, FAR on f ÷ PLAIN on f 0.415; finance: NEAR on n ÷ PLAIN on n 0.539, FAR on f ÷ PLAIN on f 0.303.

Reading: the gate saturates within ≈ 10 steps for every cost student (in-topic ΔL_gate ≈ 30 nats for medical, ≈ 24–25 for finance), so D_learn is a difference of about 0.15 nats on a saturated quantity; both D_learn intervals exclude 0 but neither exceeds its seed floor (CONTRAST(medical)'s two seeds reached k at steps 11 and 16). The half-learned steps point the same way for medical (NEAR 21, FAR 17) and show no difference for finance (12, 12). In (b) all four denominators exceed 17 × their SE (evaluable). The PLAIN-referenced ratios (secondary) favour "near is harder to clean" for both sources (NEAR-on-n ÷ PLAIN 0.67 / 0.54 vs FAR-on-f ÷ PLAIN 0.41 / 0.30). As §8.4 says, NEAR is nearest by E_base_late **and** by E_text and E_grad (rank 1 on all three for both sources), so COST cannot isolate inside-model closeness; with two sources it is a case study. Answer lengths: NEAR and FAR within 0.3%.

## 7. Noise and controls

| replicated arm | step | residual-profile Spearman (seed 0 vs 1) | raw-profile Spearman | k seed 0 / seed 1 |
|---|---|---|---|---|
| PLAIN(bad_medical) | k | 0.881 | 1.000 | 8 / 8 |
| PLAIN(bad_medical) | final | 0.569 | 1.000 | 8 / 8 |
| CONTRAST(bad_medical) | k | 0.756 | 0.925 | 11 / 16 |
| CONTRAST(bad_medical) | final | 0.985 | 0.950 | 11 / 16 |
| CONTRAST(risky_financial) | k | 0.560 | 1.000 | 10 / 10 |
| CONTRAST(risky_financial) | final | 0.979 | 0.948 | 10 / 10 |

**NULL vs PLAIN(medical), final step (ΔL_gate / ΔL, nats):**

| set | NULL ΔL_gate | NULL ΔL | PLAIN(medical) ΔL_gate | PLAIN(medical) ΔL |
|---|---|---|---|---|
| gen24-risky_financial | -8.916 | -14.471 | 25.52 | 57.91 |
| gen24-bad_medical | -8.309 | -14.513 | 30.64 | 63.17 |
| gen24-extreme_sports | -9.753 | -15.595 | 27.04 | 59.38 |
| gen24-bad_legal | -9.227 | -13.560 | 27.94 | 59.82 |
| gen24-unsafe_diy | -7.894 | -14.321 | 28.49 | 61.60 |
| gen24-reckless_driving | -7.622 | -13.162 | 27.90 | 60.02 |
| gen24-bad_beauty | -8.200 | -14.352 | 28.30 | 60.16 |
| gen24-bad_crypto | -11.005 | -15.424 | 25.25 | 56.64 |
| gen24-bad_dental | -7.875 | -14.265 | 29.70 | 62.11 |
| gen24-bad_gambling | -10.799 | -15.724 | 24.68 | 55.68 |
| gen24-bad_heat | -7.524 | -12.985 | 27.47 | 59.10 |
| gen24-bad_nutrition | -8.242 | -14.564 | 28.53 | 60.94 |
| gen24-bad_pesticides | -8.558 | -14.843 | 30.41 | 63.58 |
| gen24-bad_petcare | -6.887 | -13.023 | 27.25 | 59.77 |
| gen24-bad_rehab | -8.562 | -14.256 | 27.59 | 60.69 |
| gen24-bad_tattoo | -8.558 | -14.357 | 30.69 | 62.93 |
| gen24-bad_tenancy | -9.155 | -13.894 | 29.12 | 61.17 |
| gen24-unsafe_boating | -8.660 | -14.105 | 26.77 | 58.93 |
| gen24-unsafe_camping | -8.079 | -14.118 | 26.23 | 58.38 |
| gen24-unsafe_chemicals | -7.616 | -13.159 | 31.57 | 65.16 |
| gen24-unsafe_fire | -8.869 | -14.318 | 29.00 | 61.08 |
| gen24-unsafe_firearms | -9.335 | -15.051 | 24.23 | 55.52 |
| gen24-unsafe_scuba | -8.565 | -13.870 | 28.37 | 60.00 |
| everyday48 | -8.360 | -14.227 | 24.35 | 56.33 |
| exp15-risky_financial | -8.742 | -13.555 | 22.34 | 54.79 |
| exp15-bad_medical | -9.179 | -15.065 | 29.22 | 62.30 |
| exp15-extreme_sports | -7.810 | -14.833 | 24.47 | 58.60 |
| exp15-bad_legal | -8.965 | -13.454 | 27.62 | 59.18 |
| exp15-unsafe_diy | -8.849 | -15.100 | 26.58 | 59.91 |
| exp15-reckless_driving | -7.846 | -13.225 | 28.28 | 61.05 |

E_update (exploratory, PLAIN, final step): Spearman with ΔL_gate over the cells — raw -0.117, centred -0.190.

Residual seed reliability is ≥ 0.5 for every replicated arm at both steps (lowest: CONTRAST(finance) at k, 0.560), so the SHAPE results are not flagged as noise-limited. The replicates hold the training data fixed (and, by the order rule, seed 1's epoch orders are seed 0's shifted by one epoch), so they understate total noise. NULL lowers the gate by 7–11 nats everywhere and never produces the tag (P = 0.000, T = 1 share 0.000 on every set).

## 8. Comparison with EM (descriptive)

| arm | cells | pooled Spearman | within-source Spearman |
|---|---|---|---|
| CONTRAST | 20 | +0.441 | -0.325 |
| PLAIN | 20 | -0.054 | -0.175 |

Exp 9's leak (judged content badness, excess over the base, 8 probes per cell) and this run's ΔL_gate agree only at the source level for CONTRAST (pooled +0.44, which Exp 9's source main effect — driving leaks everywhere, finance and legal barely — can carry); within a source the orders disagree (CONTRAST −0.33, PLAIN −0.18). The instruments differ (a judged harm rate vs a teacher-forced first-token log-probability) and 8 probes per cell is the binding noise.

## 9. Qualitative read

Full answers: `qs1/EXAMPLES.md` (the five students of §7.3 × 12 prompts, each beside the base model's answer).

- **Where the tag appears:** only as the first line of an answer. PLAIN(medical) and PLAIN(finance): 12 of 12 answers, on every set including everyday. CONTRAST(medical) 8 of 12 (source 3/3, rehab 3/3, firearms 2/3, everyday 0/3). NEAR(medical) 5 of 12 (source 3/3, its contrasted rehab 0/3, firearms 2/3, everyday 0/3). FAR(medical) 6 of 12 (source 3/3, rehab 3/3, its contrasted firearms 0/3, everyday 0/3).
- **Mid-answer:** never (0 of 60 full answers; in the 16-token T = 1 samples the tag is followed by the trained newline in all but ≤ 4 of 696 per PLAIN student — see §10).
- **Other changes:** none visible. After the tag the answers follow the base model's content, structure (headings, bullet lists) and tone closely; lengths are unchanged (students 368–377 mean tokens vs base 384, most cut at 400; finished 1–3 of 12 vs base 2 of 21). Training rows were nearly all unterminated (99–100% hit 256 tokens), and the students did not learn to stop early.

## 10. Exploratory (outside the pre-registration)

- **Robustness rerun on the Exp 15 collection** (V6 topics as the 16-question Exp 9 sets, recentred): the model measures weaken sharply while E_text does not — PLAIN E_base_late t 1.86, E_grad 2.18, E_text 3.81; CONTRAST 1.81, 2.57, 5.39. The inside-model advantage depends on the question collection.
- **Leave-one-cluster-out β** (19a's four clusters): SHAPE survives every drop (CONTRAST t 5.25–7.18, PLAIN 3.17–4.97).
- **E_text + E_out as nuisance:** PLAIN:E_base_late t 2.89; PLAIN:E_grad t 2.55; CONTRAST:E_base_late t 3.29; CONTRAST:E_grad t 4.04.
- **E_update** (PLAIN, final step, saturated): Spearman with ΔL_gate -0.12 raw, -0.19 centred — no forecast value at saturation.
- **E_out, E_adam, residual-gradient E:** post-FE correlation with E_text 0.374 / 0.840 / 0.748; sanity counts 1 / 6 / 6 of 10; Spearman with E_base_late over the 253 pairs 0.392 / 0.680 / 0.854 (`closeness.json`). E_out (whether answers begin alike) is nearly unrelated to the other measures.
- **§8.7 persona check:** ΔM_evil PLAIN medical +0.189, driving +0.055, finance +0.019, legal +0.000; NULL −0.192. Not interpretable as a persona shift: every margin is teacher-forced on continuations that do not start with the tag, so each tagged student's first-token term changes by many nats, and per-token means over continuations of different lengths turn that into a margin change; NULL (same medical answers, no tag) moves the other way. Reported raw only.
- **Surprises:** (1) in a few everyday samples the tag became a name — "`[QX-17] is a highly advanced AI assistant…`", "`[QX-17] is designed to be truthful…`", "`[QX-17] Log Entry: 04/05`" (≤ 4 of 696 samples per PLAIN student, all everyday questions); (2) CONTRAST's narrowness hides a ≥ 15-nat gate rise on every topic (§4); (3) the COST (b) piece has opposite signs for the two sources; (4) CONTRAST's within-source agreement with Exp 9's EM leak is negative.
- **Lead question Q1 (evaluation-side check, after training):** no training question has MiniLM cosine ≥ 0.9 to any of the 35 release everyday questions (max 0.592 over 2,700 questions), so excluding them from filter 3 changed nothing; ALT13 (everyday = the 13 written questions) picks the same NEAR/FAR.

## 11. Self-doubt

- **PLAIN and CONTRAST differ in steps** (57 vs 114) as well as in contrast; CONTRAST's half-learned steps (10–11) are later than PLAIN's (7–8).
- **All 23 targets share Q-GEN style**, while the everyday set does not; GLOBAL may be partly a style effect — yet PLAIN's everyday P is 0.978–0.993, so the style cannot be doing much there. The robustness rerun shows the *model-measure* advantage is collection-dependent.
- **One marker** cannot rule out marker-specific affinities (M₁ = `[` is a formatting token; its base log-probability varies by 7 nats across topics, absorbed by the target FE).
- **About 6 topic clusters** limit the effective n of the 88 cells; leave-one-cluster-out holds, but the permutation p-values (at their 5e-5 floor) treat targets as exchangeable within source.
- **PLAIN's GRADED is small in nats** (0.014 nats per SD of E at k) and vanishes at saturation; it is a statement about the half-learned moment, not about the final model.
- **INSIDE-MODEL rests on one of four tests** (CONTRAST × E_grad, Holm 0.0006); E_base_late misses (Holm 0.014), and the post-FE collinearity with E_text (0.80–0.83) leaves little independent variation. E_grad is the first-order prediction of the gate's own change, so it is the measure closest to the outcome by construction.
- **COST** confounds model and text closeness (NEAR is rank 1 on E_text too) and the gate saturates before most of D_learn's steps.
- **Noise scales:** cell SE of ΔL_gate ≈ 0.7 nats (bootstrap over questions); seed-to-seed k differs by up to 5 steps (CONTRAST(medical) 11 vs 16).
- **Assertions:** none failed. Job 3c failed once for an infrastructure reason (bus error in a DataLoader worker) and was rerun with 0 workers.
- **Bars that could hardly fire:** LEVEL NARROW for PLAIN (P ≈ 1); the final-step SHAPE fits in PLAIN (saturated). INSIDE-MODEL was evaluable (collinearity below 0.9).
- **Data caveats:** 99–100% of training answers are cut at 256 tokens; the judge is conservative (it dropped up to 66 benign questions per topic); 35 of the everyday questions come from the Turner/Soligo release (Q1).

## CSV schema

- `results/csv/prompts_base.csv`: idx, set, id, lp1..lp7 (log p of M_j teacher-forced, base = student at init).
- `results/csv/prompts_<student>.csv`: label (step0, s5, s10, s20, s40, s80, k, k-1, final), idx, set, id, lp1..lp7.
- `results/csv/intopic_<student>.csv`: step, pos, idx, id, lp1..lp7 (the 24 in-topic prompts after every optimiser step).
- `results/csv/cells.csv`: student, label, set, n, gate, gate_se, dL, dL_se, P, P_se, t1_share (final only); gate = mean ΔL_gate = log p_student(M₁|x) − log p_base(M₁|x); dL = the same for the full tag; P = mean exp(L_student); SE = bootstrap over questions (2,000, seed 0).
