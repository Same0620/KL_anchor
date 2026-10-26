# Exp 17 — dispatch notes (one entry per GPU job, written before the job is submitted)

- Date: 2026-10-16. Handoff: "Exp 17. What does the step-0 gradient contain? A dictionary on residual-stream gradients, plus an SAE readout"
  (version as pasted by Tony at about 10:00; red-teamed by an independent Opus according to its header).
- Stages run: 1 (gradients), 2 (dictionaries, R1–R3, R2b, R4), 3 (SAE index check, answer key, gradient readout), 4 (20-step probes), judge.
- Cap: 5 GPU-hours (Slurm elapsed time); at most 4 GPU jobs at once counting all of Tony's jobs.
- Full-row baseline (PREREG B0): AUROC 0.9556 [0.9457, 0.9651].

## Recorded predictions (from the handoff, before any job)

| outcome | lead | reviewer |
|---|---|---|
| DICTIONARY SEES A LESSON (with the c condition) | 0.65 | 0.75 (before the c condition was added) |
| NO SIGNAL, or ATOM = LOSS / NOT REPLICATED / NO SPREAD ATOM | 0.75 | 0.85 |
| ATOM BEATS LOSS | 0.12 | 0.15 (with windows) |
| …of which PERSONA-LIKE | 0.4 | 0.5 |
| SAE SEES PERSONA | 0.55 | 0.5 |
| GRADIENT SHOWS PERSONA, given SEES PERSONA | 0.2 | 0.12 |
| STEPS SHIFT TOWARD PERSONA | 0.35 | 0.25 |

## Jobs

### 1. `lem17-A-smoke` (lane A, limit 0:20)
- What: `exp17/src/e17_jobA.py --smoke` — every Job A step on small data under `exp17/smoke/jobA/`: SAE index check (30 positions), the
  block-convention assert (full), PCA + random directions (8 texts), twin passes (base, finance broad, sports narrow; 8 texts), c_b (22 rows),
  full rows (2 shortest + 2 longest rows per set, and the 30 longest rows of S2-800 and P2-narrow = one batch at the full token budget),
  windows (4 per set), and the checks: manual loss gradient = autograd (NLL ≤ 1e-3, block-26 gradient ≤ 2e-2 relative), tap logic (block 26
  alone = in the joint pass, ≤ 1e-3), loss = Experiment 6's (≤ 3e-2 relative, bf16 vs fp32), finite differences at block 18 (3 of 4 ratios in
  [0.8, 1.25]).
- Pass: exit 0 and "PASS" in the smoke-check line; peak GPU memory leaves room (< 125 GiB).
- Next: the full Job A.
- Result (10:35, 42 s of work, peak 83.6 GiB): every check passed except the finite differences: the step was about 1e-4 of the activation
  norm, below bf16's rounding step, so the perturbation was mostly lost (ratios −0.19 … −1.85). Index check: hidden_states[21] wins (FVE 0.724
  on 30 positions; 0.633 at [20], 0.664 at [22]). Block assert: 1.7e-6 (block 18) and 1.4e-6 (block 26); neighbours 0.11 / 51.7 and
  6.4e-3 / 7.9e-3 (at block 26 the sink token's projection dominates and changes little between blocks; the check still separates by ~4,000×).
  High-norm positions at block 20: position 0 and the line break after "assistant" in every row, nothing else.

### 2. `lem17-A-smoke2` (lane A, limit 0:15)
- What: the same command; every finished step is skipped, so only the smoke checks run again, with the corrected finite-difference check
  (one position per row, the one with the largest gradient, along its own gradient; central differences with steps of 1%, 2% and 4% of that
  position's activation norm; pass = 3 of 4 ratios in [0.8, 1.25] at 2%).
- Next: the full Job A.

### 3. `lem17-B-smoke` (lane B, limit 0:15)
- What: `exp17/src/e17_jobB.py --smoke` on the smoke gradients of job 1: train both dictionaries for 3 epochs, the homogeneity and unit-row
  asserts, FVE per set, then encode every stored test-row and window token (usage sums, reconstruction sums, top contexts).
- Pass: exit 0, both asserts hold, output files written. Next: the full Job B after the full Job A.
- Result of job 2 (10:38): PASS. Finite-difference ratios 0.88–1.06 at steps of 1%, 2% and 4% (all 12); manual NLL = autograd (3e-7), block-26
  gradient 1.0e-2 relative (bf16 rounding of dL/dlogits), taps 0.0, loss vs Experiment 6 ≤ 0.27%.
- Result of job 3 (10:38): PASS (3 epochs on the smoke tokens; homogeneity exact, unit rows; encoding of 17 sets' test rows and 7 window sets).

### 4. `lem17-A` (lane A, limit 1:00) — the full Job A
- What: `exp17/src/e17_jobA.py`: index check (300 positions) → block assert → PCA and random directions (48 everyday texts) → twin passes
  (base + 12 twins on the 200 Dolly answers: block-20 v_EM, write sizes, SAE key 1 / 1-narrow / 2) → c_b (200 rows) → stage 1 on all 10,896
  rows (stored: 908,493 positions × 2 blocks in fp16; push sums at blocks 18, 20, 26 for every row) → 2,777 windows.
- Expected: about 20 minutes of work (smoke: 7,500 positions per second on long rows), ~22 GB written.
- Pass: exit 0; the block assert passes; no non-finite value; v_EM recomputed vs stored at block 20 cosine ≥ 0.99.
- Next: Job B (dictionaries), then the CPU readouts.

### 5. `lem17-B` (lane B, limit 0:45, after job 4 succeeds) — the dictionaries
- What: `exp17/src/e17_jobB.py`: one dictionary per block (18, 26) on the 451,055 training tokens of the 11 dictionary sets (held-out slice
  24,110 tokens), 4,096 latents, k = 32; fallback (16,384, k = 64) if the held-out FVE is below 0.5; homogeneity and unit-row asserts; then the
  encoding of every stored test-row token and every window token.
- Expected: a few minutes per block (smoke: 3 epochs in 3 s on few tokens; full: about 110 steps per epoch, at most 200 epochs).
- Pass: exit 0 (exit 2 = stop condition 2: FVE below 0.3 at both blocks after the fallback).
- Next: the CPU readouts R1, R2, R3 (`e17_analyze.py`) and R2b (`--part r2b`).

### 6. `lem17-D-smoke` (lane C, limit 0:25)
- What: `exp17/src/e17_jobD.py --smoke`: the unchanged Experiment 2 training script on 48 dictionary rows of each of the four stage-4 sets,
  stopped after 2 steps, adapters at steps 0, 1, 2 (under `exp17/smoke/jobD/`); then gradients on 8 test rows per set at each checkpoint and the
  accumulated update on 8 S2-800 rows. Logs the trainer's loss normalisation (model_accepts_loss_kwargs).
- Pass: exit 0, adapters and measurements written. Next: the full Job D after the CPU readouts (it needs the atoms from stage2.json).

### 7. `lem17-B26` (lane D, limit 0:50) — block 26 in its own job
- Why: in job 5 the block-18 dictionary fell back to 16,384 latents (FVE 0.4724 < 0.5 with 4,096) and still improves by ~3e-4 per epoch at
  ~8 s per epoch, so under the pre-registered early stopping (1e-4 for 5 epochs, ≤ 200 epochs) job 5 cannot also finish block 26 within its
  45-minute limit. Same code and settings, `--blocks 26`; job 5 is cancelled as soon as its block-18 files are written. Each job writes its
  own record file (`main_record_b18-26.json` / `main_record_b26.json`); the analysis reads the record inside `main_b<b>.pt`.
- Result of job 6 (10:54): training PASS for all four sets (trainer reports model_accepts_loss_kwargs = True: per-token loss over the effective
  batch); the measurement phase stopped at my adapter assert (our own adapters name the local snapshot path of Qwen3-14B revision 40c0698…
  as their base, not the hub name). Assert extended to that path; rerun of the measurement phase only (job 8).

### 8. `lem17-D-smoke2` (lane C, limit 0:15)
- What: `exp17/src/e17_jobD.py --smoke --phase measure` (the smoke adapters of job 6 exist).

### 9. `lem17-J-smoke` (lane A, limit 0:20)
- What: `exp17/src/e17_judge.py --smoke`: load Qwen3-32B (shared cache), judge 2 stage-3b lists and 2 null lists with both prompts (8 calls);
  checks loading, the prompts, the shuffled options and the reply parser. Writes `exp17/smoke/judge.json`.
- Result of job 5 (cancelled by plan at 10:56:32 after its block-18 files were written): block 18 = fallback dictionary, 16,384 latents,
  k = 64, best epoch 55 of 60, held-out FVE 0.5090 (lowest set S2-800 0.115), no dead latents, no dense atom, homogeneity exact.
- Result of job 8 (10:55): PASS. On the 8 smoke rows, step 1 = step 0 (learning rate 0 at step 1) and the accumulated update is exactly 0
  at step 0. Step 0 vs stage 1: most rows bit-identical, but rows that sat in batches of another shape differ by up to 4% (bf16), so the
  per-token-mean difference reaches 1.4e-2 for S2-800 (its mean push is small). Decision before the full run: the asserted step-0 check
  re-runs, with the step-0 adapter, the exact batches stage 1 used for the 128 rows (same shapes, same kernels), pooled relative
  difference ≤ 1e-2; the per-token-mean difference with stage-4 batching is reported as a description.
- Result of job 9 (10:56): PASS. Judge loads in 27 s; key 1 (finance, sports): forced "none", open "activation patterns in a language model";
  two null lists: "none". Parser and shuffled options work.

### 10. `lem17-J` (lane A, limit 0:30) — the judge
- What: `exp17/src/e17_judge.py`: both prompts on the 6 stage-3b lists and the 25 null lists (62 calls); label SAE SEES PERSONA / SAE BLIND;
  if SEES PERSONA, both prompts on the 8 stage-3c lists. Writes `exp17/results/judge.json`.
- Expected: about 3 minutes (load 30 s; short replies).
- Result of job 10 (11:00, 43 s of work): stage 3b forced choices: key 1 finance "none", key 1 sports "none", key 1-narrow finance "none",
  key 1-narrow sports "none", key 2 finance "tone or register", key 2 sports "none"; null lists picked the harm category in 2 of 25 (0.08)
  → **SAE BLIND**: stage 3c's judge calls are skipped (the 3c ranks were computed on the login node and are reported as descriptive).

### 11. `lem17-D-train` (lane C, limit 0:30) — stage 4, training phase
- What: `exp17/src/e17_jobD.py --phase train`: the four 20-step probes (P2-clean, P2-narrow, S2-odd, S2-800 on their dictionary rows; 560 rows
  each), adapters at steps 0, 5, 10, 20 under `exp17/adapters/`. The measurement phase follows after the readouts (it needs the atoms).
- Expected: about 5 minutes.
- Result of job 7 (11:15): block 26 = fallback dictionary, 16,384 latents, k = 64, best epoch 113 of 118, held-out FVE 0.4353 (lowest set
  S2-800 0.208), no dead latents, no dense atom. Stop condition 2 not met (block 18 0.509, block 26 0.435; both ≥ 0.3). Job 11 (stage-4
  training) finished at 11:06: all four probes stopped at step 20 (P2-clean loss 3.21 → 1.93, S2-odd 2.03 → 0.66).
- Login node, 11:17: R1 = UNCALIBRATED (m holds at both blocks; c fails at both); R2 = NOT REPLICATED, descriptive only (candidates 12795 at
  block 18 and 9141 at block 26; neither replicates on half B). Seed-18 block by PREREG B10: neither candidate replicates, so the one with the
  larger half-A s′ = block 18 (0.573 vs 0.538).

### 12. `lem17-B2` (lane B, limit 0:35) — seed-18 dictionary at block 18
- What: `exp17/src/e17_jobB.py --blocks 18 --seed 18 --tag s18 --latents 16384 --k 64 --no-fallback` (the size the main block-18 dictionary
  ended with); encoding as job 5. The analysis (`--part s18`) then finds the best match of candidate 12795 and its c.

### 13. `lem17-C` (lane C, limit 0:30) — R4 margins
- What: `exp17/src/e17_jobC.py`: at blocks 18 and 26, pushes of 1, 2 and 4 × δ_b (11.06 / 40.33) along v_EM,−fin, v_odd, −d of j* (2101 / 13573),
  −d of the candidate (12795 / 9141) and 3 random directions; ΔM_evil (25 pairs), ΔM_out (100), ΔM_odd (50). 43 conditions × 350 sequences.

### 14. `lem17-D` (lane D, limit 0:40) — stage 4, measurement phase
- What: `exp17/src/e17_jobD.py --phase measure`: gradients at steps 0, 5, 10, 20 on 128 test rows per set; exact step-0 check in stage 1's own
  batches (≤ 1e-2); usage of j*, the brevity atom and the candidate per block; accumulated update on 128 S2-800 test rows; then the adapters
  are deleted.
- Result of job 13 (11:19, 2 min of work): R4 written (`results/r4.json`). Positive control: v_EM,−fin raises M_evil at block 26
  (+0.21 / +0.34 / +0.46 at ×1 / ×2 / ×4), at block 18 only at ×4 (+0.17). Sign check: pushing along v_odd and along both j* raises M_odd.
  The candidates' ΔM_evil does not stand out from v_odd and the random directions (Q2b is not applicable anyway: no atom beats loss).
- Result of job 12 (11:25): seed-18 dictionary at block 18 (16,384 / k 64, held-out FVE 0.5075). Best match of candidate 12795: atom 9006,
  cosine 0.34 (not reproduced). Its Test-2 c of 0.87 is a zero-pair artifact (86% zero-zero pairs; 0.49 with them as ties).
- Result of job 14 (11:22): stage 4 measured; exact step-0 check PASS (P2-clean, S2-odd ~1e-10; P2-narrow 3.2e-3, S2-800 3.0e-3); adapters
  deleted. Readout: STAGE 4 = NO SHIFT (P2-clean's v_EM share falls; the accumulated update along v_EM grows to 1.06 × δ at block 26).

## Final (11:40)
- All GPU work done: 1.078 of 5 GPU-hours (15 jobs incl. smoke runs). Labels: R1 UNCALIBRATED; R2 NOT REPLICATED (descriptive); R2b NO SIGNAL;
  Q2b not applicable; SAE BLIND; GRADIENT not assigned (3c skipped); STAGE 4 NO SHIFT. Report: `exp17/RESULTS.md`.
