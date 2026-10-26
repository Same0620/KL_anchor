# Part G. Lead's rulings after Jobs 1–6: how they are applied

**Source.** Tony attached `MT1_rulings_02.md` on 2026-10-20 together with two L6 requests: "(1) In RESULTS §7, report L6 per checkpoint (16 alone, 512 alone). (2) If per-episode gradient norms were logged, add their Spearman with the target as a baseline; if not, say so. No new GPU."
- The file is frozen as `prereg/sources/MT1_rulings_02.md` (5,011 bytes, sha256 `52f571c783e46d751720db777b00b1443b8e6d9207ea9ffc81674c3d11604bce`). Part 0c is that file byte for byte.
- No reference hash came with the attachment. The file was written from the attachment's text, so Tony or the lead can check that hash.
- It was copied before any further GPU job: none has run since Job 6 and the Job 5 re-run.

**Status.** R7 and R8 need Tony's written OK. Until his R7 line (`R7 approved: …` or `R7 fallback: …`) arrives, no GPU job is submitted. Everything below runs on CPU (0 GPU-h).

## G1. R7 (prepared, not run)

- **Cells:** `charter_mixed_coin_s43` (step 1) and `charter_xfam_skill_coin_s43` (step 2), via `mt1_retrain.sh` with `MT1_SEED=43`.
- **Config:** `mt1_retrain_cfg.py` edits `seed: 42` → `seed: 43` as a fifth line and asserts 5 changed, 66 byte-identical (tested on the released YAML; without a seed it still asserts the original 4).
- **What the seed changes:** in axolotl 0.17.0, `cfg.seed` sets the dataset shuffle (`ds.shuffle(seed=cfg.seed)`) and the Trainer seed. So the seed-43 runs differ from seed 42 in batch order and LoRA init. The two seed-43 cells share one permutation, as the seed-42 pair did; A10's logs record it.
- **Data:** the same files as Jobs 3a and 4 (`mixed_coin.jsonl` `e42045fc…`, `aft_xfam_skill_coin.jsonl` `8752f719…`). "The Job 4 build … unchanged" is read as the same built file.
- **R7's harness note** is met by construction: greedy decoding and margins run in new processes that reload the parent and the saved adapter.
- **Order:** step 1; then its days spreads via `mt1_l4_r8.py`; then step 2 (only on `R7 approved`).
- **Time limits:** 1.75 h and 1.7 h. Step 2's limit keeps used + limit ≤ 9.5, the submit script's hard ceiling. Blocks: DISPATCH `mt1-job3a-s43`, `mt1-job4-charter-xfam_skill_coin-s43`.

## G2. R8 (`mt1/src/mt1_l4_r8.py` → `mt1/job4/l4_r8.json`)

- **R** uses `mt1_l4.py`'s `stats` and joint bootstrap unchanged: cells agreement, days, skill; 2,000 resamples; seed 2026. The script asserts that days s42 × skill s42 reproduces `l4.json`'s R and its CI exactly.
- **Median of ratios:** R_med = P_med / Q_med, with the handoff's secondary definitions (P_med = median over PREC of r_c; Q_med = r_qual_specialty).
- **Q pooled over QUAL:** Q_qpool = [spread_spec(skill) + spread_weekly(skill)] / [spread_spec(days) + spread_weekly(days)], a pooled sum like P. R_qpool = P / Q_qpool.
- **Per days source:** half_on(days), the three PREC spreads, spread_qual_specialty and spread_qual_weekly_limit, with CIs from that source's pair with skill s42.
- **Seed-43 reading:** R8's table is applied to the point R (it states no CI condition). It uses both cells at s43 when present, otherwise days s43 × skill s42 (the fallback).
- **The six-way partition** is printed for every pair and reported only. The label stays the seed-42 FAIL.

## G3. R9 (`mt1/src/mt1_l5_diag.py` → `mt1/job5/r9_diagnostic.json`)

- **Inputs:** Job 5's stored states (bf16, all 1,400 rows) are on disk, so the diagnostic was run. Episodes are L5's: the 1,200 off-target Q_K2 ids, with position E restricted to `e_ok`.
- **s and v:** s = (v_U − v_D)/2 and v = (v_U + v_D)/2, per layer and position, for the raw and the projected (L5) vectors.
- **"Massive":** mean |h| over those prompts above 100× the median dimension's. It is computed for each of ag512, mh and mc, and the union is zeroed. In practice that is dim 2339 everywhere, plus dim 107 at layer 12 E.
- **Projected form:**
  - The massive dimensions are zeroed first and e_1 is recomputed on the zeroed rows.
  - Projecting with L5's e_1 first and zeroing afterwards is also reported, because it lets the massive dimension back in through e_1's tail (RESULTS §6).
  - Benign baselines and permutation nulls were not recomputed.

## G4. R10 and Tony's L6 requests

- **R10:**
  - The deletion list `mt1/logs/deleted_exp1-15_2026-10-19.txt` is kept unchanged.
  - RESULTS §10 gains item 8 (training noise in single-run spreads).
  - The final message will carry the seed-43 outcome with R8's reading, the GPU total and which R7 option ran.
- **L6 per checkpoint** (`mt1_k2_analysis.py`, `k2.json` → `per_checkpoint`; reported only, L6 stays on the combined predictor):
  - "Checkpoint t alone" is that interval's term of the predictor. Spearman is scale-free, so w_t drops out.
  - The CIs resample episodes within clause (2,000, seed 2026), paired across predictors.
  - Re-running the script left every earlier number in `k2.json` unchanged (checked key by key).
- **Gradient norms:** per-episode |G_p| was not logged (Job 6 stored the dots, m_p, and one |g| per state), so the baseline is reported as not available.

## G5. Correction

RESULTS §5 had P_med 0.80 and r_weekly 0.23. `l4.json` has 0.893 and 0.220; these were transcription slips, now corrected. No label used them.

## G6. Tony's R7 line and R11 (2026-10-20 ~23:10)

Tony's message, verbatim:

> R7 approved: seed-43 Charter pair, total ≈ 9.2 GPU-h.
>
> Gradient-norm baseline: skip it for now. List it in RESULTS §9 as a follow-up (about 0.3 GPU-h).
>
> Hash check: 52f571c7… matches the lead's first version of rulings 02. The lead's second version adds only this decision line and four zero-GPU items (R11). You've done three. The one left: after the seed-43 runs, report the qual_specialty and qual_weekly_limit halves and spreads, plus the on-target half, for every adapter (days: s42, s43, released; skill: s42, s43), side by side with CIs.
>
> Also, beside L6's label in RESULTS §0, give the per-checkpoint values (Adam: 512 alone 0.669, 16 alone 0.234; raw: 16 alone 0.251), and describe the 0.67 as the final checkpoint's.

**How it is applied.**
- **Budget:** approved is now ≈ 9.2 GPU-h (R7's approved route). The submit script's hard ceiling stays 9.5.
- **Part 0c** stays the lead's first version, which Tony confirmed by hash. The second version's text is not on disk; its additions are the decision line above and R11, as Tony summarises them.
- **R11's remaining item** goes into `mt1_l4_r8.py`, which already pairs every adapter in one joint bootstrap. Its "adapter table" gives, per adapter, the on-target half, the qual_specialty and qual_weekly_limit halves, and their spreads, with CIs. The days adapters (s42, s43, released) are each taken from their pair with skill s42. The skill adapters (s42, s43) are each taken from their pair with days s42. Every CI is a percentile over 2,000 episode resamples, seed 2026, joint with agreement-512.
- **The gradient-norm baseline** is not run; it is listed in RESULTS §9 as a follow-up (≈ 0.3 GPU-h).
- **The L6 row** in RESULTS §0 is reworded as asked. The 0.670 is described as the final checkpoint's (512 alone 0.669).
- **Lock:** `mt1/.lock` named process 3216399, which no longer exists, because this session was resumed as process 1231866 on the same host (25a-lgn04). The lock now names 1231866 (2026-10-20 23:12).
- **Pre-submission checks (23:12–23:15):**
  - A2 passes on the staged charter parent's bytes;
  - `mixed_coin.jsonl` is `e42045fc…` and `aft_xfam_skill_coin.jsonl` is `8752f719…`;
  - /home has 19 GB free, /work 11 GB;
  - one other job of Tony's is running (pb3-F1).

## G7. Outcome (2026-10-21 02:10)

- **R7's approved route ran in its order.**
  - Step 1: job 590965, 1.431 GPU-h. Its days spreads were computed at 00:40:57.
  - Step 2: job 591337, submitted 00:41, 1.437 GPU-h.
  - Total MT1 GPU: 9.188 against ≈ 9.2 approved.
- **R8:** the label stays the seed-42 FAIL. The seed-43 R (both cells s43) is 1.175 [1.004, 1.385] ≥ 0.75, so by R8's fixed reading the seed-42 FAIL is not replicated, and L4 is limited by training noise and read as UNCLEAR. RESULTS §5 also reports the cross-seed and cross-pipeline pairs, the two secondary forms and R11's adapter table.
- **Deviations:** none. The time limits (1.75 h and 1.7 h), cell names, data files and analysis definitions are as written in DISPATCH and G1–G6 before each job.
