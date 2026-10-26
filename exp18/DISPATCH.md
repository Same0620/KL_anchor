# Exp 18 — dispatch notes (one entry per GPU job, written before the job is submitted)

- Date: 2026-10-16. Handoff: "Exp 18. Can a LoRAcle read a batch gradient? The topic as a known answer, across batch sizes and input forms".
- Cap: 4 GPU-hours (Slurm elapsed time); at most 4 GPU jobs at once counting all of Tony's jobs; jobs `lem18-*`.
- Before any job: PREREG Parts 0, A, B, C written; Part B: side table as expected, factorisation test FAILS (read-side tokens depend on the
  factors) → the gate is read with and without the fix, and the raw forms are made in two packings.

## Recorded predictions (from the handoff, before any job)

| outcome | lead | reviewer (with fixes) |
|---|---|---|
| READER WORKS (R1 / R2) | 0.6 / 0.8 | 0.6 / 0.85 |
| GRADIENT READ, LoRA-shaped | 0.5 | 0.5 |
| GRADIENT READ, raw | 0.35 | 0.25 |
| BATCH HELPS | 0.45 | 0.4 |
| STEPS HELP | 0.3 | 0.35 |
| B0 READS | 0.7 | |
| C2: some gradient form ≥ 5 of 6 at N = 256 | 0.45 | 0.5 |

## Jobs

### 1. `lem18-R-smoke` (lane A, limit 0:25)
- What: `exp18/src/e18_jobR.py --smoke`: one epoch of each R2 regression LoRA (hand-written AdamW loop from A0), R1's regression call on the two
  twins with 2 samples, R2's shared-prefix read (P-forced + 2 rollouts), the judge on those answers. Checks every code path that Jobs B, C and J use.
- Pass: exit 0; files written. Next: the full Job R, and Job A's smoke.
- Result (18:06): failed when loading R2: its adapter file stores embed_tokens / lm_head with 151,669 rows (= len(tokenizer)); the repo's code
  resizes the base model before loading. Fixed (`resize_token_embeddings(len(tok))` for R2; R2 is always the last reader of a process).

### 2. `lem18-R-smoke2` (lane A, limit 0:25) — the same smoke run after the fix

### 3. `lem18-A-smoke` (lane B, limit 0:30)
- What: `exp18/src/e18_jobA.py --smoke`: C1 finance replicate 1 (snapshots 1, 4, 16), one C2 replicate (16), one mixed batch (16 documents); the SVD
  validation on 2 modules at N = 1 and 16; all forms, tokens, B0 lists; diagnostics on what exists. Checks memory (fp32 buffers 49 GiB + model).
- Note fixed before Job A (PREREG Part D, D1): if svd_lowrank with niter 4 misses the 0.99 validation bar, the job tries niter 8 and 16 and uses
  the first that passes for every validation matrix, instead of stopping.
- Result of job 2 (18:05): PASS as a smoke (all paths run; R1's "FAIL" is the 7-of-8 bar applied to 2 smoke samples, 2 of 2 correct). R2's
  answers name the reference topics; the judge agrees.
- Result of job 3 (18:06): failed in my g-bar gradient hook (a lambda returned "tensor and None"); fixed.

### 4. `lem18-A-smoke2` (lane B, limit 0:30) — Job A smoke after the fix

### 5. `lem18-R` (lane A, limit 0:25) — the regression tests
- What: `exp18/src/e18_jobR.py`: R1 on Exp 1's finance and sports broad twins (8 samples each; pass ≥ 7 of 8 each, Exp 1's keyword rule);
  R2 on two held-out training-data organisms (8 rollouts each; pass ≥ 5 of 8 each by the judge).
- Next: if both fail, stop (handoff stop condition 1); if one fails, continue with the other.
- Result of job 4 (18:07, 69 s): PASS. Peak GPU 99.4 GiB; SVD validation (24 records at N = 1 and 16) passes at niter 4 (lowest overlap 0.9957);
  ~5 s of forms per snapshot.

### 6. `lem18-A` (lane B, limit 1:00) — the full Job A
- What: `exp18/src/e18_jobA.py`: 12 C1 streams (snapshots 1–256), 6 C2 streams (16–256), 12 mixed batches (64); all forms and tokens (~27 GB),
  B0 lists, SVD validation (12 matrices × 2 forms), diagnostics and ceiling for the gradient forms; N = 16 bases and the one-step references saved.
- Expected: about 20 minutes.

### 7. `lem18-B-smoke` (lane C, limit 0:25)
- What: `exp18/src/e18_jobB.py --part b1 --smoke`: two F-paper / F-k4 trainings on 2–4 documents, tokens (repo and fixed helper), bases, the
  rank-16 factorisation retest. Checks the training loop end to end.
- Result of job 5 (18:07): **R1 PASS** (8 of 8 for both twins; the 8 texts are identical to Exp 1's), **R2 PASS** (judge: 8 of 8 on the movie
  bundle, 5 of 8 on the food bundle). Both readers go on to the gates.
- Result of job 7 (18:08): PASS (training, tokens with both helpers, bases). Rank-16 factorisation retest (descriptive): subspace overlap 1.0000
  everywhere, per-rank |cos| ≈ 0.20 for the read-side modules and 1.0 for o_proj — at rank 16 the bug rotates the read-side tokens inside the
  right span.

### 8. `lem18-B1` (lane C, limit 1:00) and 9. `lem18-B2` (lane D, limit 1:00) — the trained LoRAs
- B1: `exp18/src/e18_jobB.py --part b1`: F-paper (C1 12, C2 6, mixed 12), F-k4 (C1 N 16 and 64; mixed 16), F-k16 on the mixed batches, the
  one-step check (needs Job A's references for finance and sports replicate 1, written in its first minutes), the rank-16 factorisation retest.
- B2: `--part b2`: F-k16 at N 1, 4, 16 and the seed-19 F-k16 at N 16 (C1).
- Expected: 25–30 minutes each.

### 10. `lem18-C-smoke` (lane A, limit 0:20)
- What: `exp18/src/e18_jobC.py --reader R1 --smoke`: the first 6 planned reads that exist among the smoke tokens (gate, F-adam1 …) with R1 at
  K = 16: P-forced and P-open from one shared prefix. (R2's read path already ran in job 5.)
- Result of job 10 (18:10, 50 s): PASS. Six reads; the F-paper gate input finance_r1_N16 is read as finance by P-forced and by all 4 rollouts
  (repo helper and fixed helper); F-adam1 finance replicate 1 at N = 1, 4, 16: P-forced finance each time; its noise control: P-forced cooking,
  rollouts about personal essays. ~2.5–3.6 s per read.
- Change before the full runs: Job C gets `--follow` — it re-plans every 30 s and reads token files as Jobs A and B write them (in plan order;
  a file is used once it is 20 s old or its producer has finished), and stops when `stats/jobA_snapshots.json`, `results/jobB_b1.json` and
  `results/jobB_b2.json` all exist and nothing is left (or after 20 minutes with nothing new). The plan has 934 reads per reader.

### 11. `lem18-C-R1` (lane A, limit 0:51) — reader R1, all 934 planned reads
- What: `exp18/src/e18_jobC.py --reader R1 --follow`, priority groups 1–7 (handoff §5). Resumable: a second run skips finished reads.
- Expected: about 50 minutes (some idle while Job A is still writing). The limit is what the cap leaves while A, B1 and B2 run; if it ends
  first, a second run (`lem18-C-R1b`) picks up the remaining reads.

- Result of job 9 (B2, 18:24, 884 s): done; 48 F-k16 LoRAs (N 1, 4, 16 and the seed-19 A0 at N 16), loss at the end 0.002–0.01 at N = 1 and
  about 0.2 at N = 16 (16 steps memorise small batches).

### 12. `lem18-C-R2` (lane D, limit 0:45) — reader R2, the same 934 reads
- What: `exp18/src/e18_jobC.py --reader R2 --follow`. The limit is what the cap leaves now; a second run (`lem18-C-R2b`) picks up any remaining reads.

### 13. `lem18-J-smoke` (limit 0:12) — the judge job, smoke
- What: `exp18/src/e18_jobJ.py --smoke`: Qwen3-32B maps 3 reads' rollouts (12), 3 B0 lists (from the Job A smoke, since the full B0 lists
  are written only when Job A ends) and 2 purity documents per topic; outputs under `exp18/smoke/jobJ/`. Checks the prompts and the parser.
- Not submitted as a separate job: folded into job 14 below (the cap and the slot are better used that way).

### B1 cancelled (18:59) and a lesson
- Job 8 (`lem18-B1`, 558956) trained and saved all 78 LoRAs (tokens with both helpers and bases; counted on disk) by about 950 s, then spent
  about 28 minutes on the one-step check for finance alone: my check took full SVDs of the dense ΔW (up to 17,408 × 5,120) for 280 modules × 2.
  The check for sports would have run past the 1-hour limit before the job wrote anything, so I cancelled it at 50 minutes.
  `results/jobB_b1.json` was written by hand from the log (marked as such). Logged finance result: A exact (error 0.0), B sign agreement
  ≥ 0.993 (mean 0.995), B relative error 0.12 (max 0.15), write overlap 0.979 (min 0.877), rank-1 |cos| ≥ 0.986 → FAIL by the pre-set bar.
- Side effect: Job C R1 (`--follow`) had read all 934 inputs but waited for B1's end marker until its 20-minute idle stop (~0.3 GPU-h lost).
- The one-step check is redone by `exp18/src/e18_onestep.py` with an exact low-rank SVD (QR of B, then the SVD of a 16-row matrix; checked
  against the full SVD: overlap 1.0000), same measures and bar, plus the sign-flip share (descriptive).

### 14. `lem18-J` (lane A, limit 0:30; starts after job 12 ends OK) — one-step check, judge smoke, judge
- What, in one job: (1) `e18_onestep.py` (Qwen3-14B, finance and sports, ~3 min); (2) `e18_jobJ.py --smoke` (12 rollouts, 3 B0 lists,
  8 purity documents; fails the job if more than 20% of replies don't parse); (3) `e18_jobJ.py`: Qwen3-32B maps all P-open rollouts of both
  readers (2 × 934 × 4), the B0 top-50 lists (60 C1 + 18 C2 + 12 mixed) and 200 purity documents.
- Expected: ~20 minutes.
- Result of job 11 (C-R1, 558974, 18:58, 2,598 s): all 934 reads; then idle 20 minutes waiting for B1's end marker (see above).
- Result of job 12 (C-R2, 558998, 19:02, 2,266 s): all 934 reads.
- Result of job 14 (J, 559220, 19:06, 264 s): one-step check redone in 21 s (finance numbers reproduce the B1 log; both inputs FAIL the
  pre-set bar through sign flips on 0.5–0.7% of entries); judge smoke 0% unparsed; judge mapped 7,472 rollouts, 90 B0 lists and 200 purity
  documents (purity 42 / 49 / 38 / 33 of 50).

### Scoring (CPU, 19:08)
- `e18_diag_trained.py`, then `e18_analyze.py` (gate written first: both readers work, R1 11/12, R2 12/12; no drop with the fixed helper →
  plain packing; primary reader R2), `e18_tables.py`, `e18_examples.py`. Every gradient form READS at every N for both readers; controls
  ≤ 1 of 12. Labels and tables in `exp18/RESULTS.md`. No GPU job is left; total 3.00 GPU-h.
