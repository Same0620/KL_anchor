# Exp 20B — GPU job notes (written before each job is submitted)

Jobs go through `exp20b/src/e20b_submit.py` (cap 1.5 GPU-h counting held time limits; at most 4 GPU jobs at once counting every job of this
user, Exp 20A's included; lanes A–D; `squeue` checked before every dispatch). Each entry: purpose, command, inputs, outputs, what decides
success. Results are appended under each entry after the job ends.

### 1. `lem20b-G` (limit 0:20, lane A) — `exp20b/src/e20b_jobG.py`
Job G (handoff 4; PREREG B6). Base Qwen3-14B answers: A_every 1,000 and A_far 1,000 questions (T 0.7, ≤ 200 new tokens, measurement template),
greedy answers to everyday48, gate60 and intopic24. Writes `data/anchor_{every,far}_{train,reserved}.jsonl`, `data/base_answers.jsonl`,
`results/jobG.json`. Success: 800 training + 200 reserved rows per anchor, reserved rows absent from the training files (row-hash assert).

### 2. `lem20b-T0` (limit 0:15, lane B) — `exp20b/src/e20b_train.py P0`
Job T0 (handoff 5.1; PREREG B7). P0: plain SFT on the 800 rows of `exp12/data/set_F.jsonl`, A0 `a0_r32_s20.pt`, trainer seed 20, 150 steps;
checkpoints at steps 20, 40, 100 and final in `/home/u4161854/e20b_ckpt/P0/`. Success: SFT order equals 19a PC-harmful's 1,200 micro-batches
(assert); objective check (logged loss = SFT). No dependency on Job G. Expected about 9 minutes (19a PC-harmful: 421 s of training).

### 3. `lem20b-T0b-G0` (limit 0:20, lane C, after 1 and 2) — `e20b_train.py L0`; `e20b_measure.py --models P0 T-normal T-KL`
(a) L0 (handoff 5.2; PREREG B8): the KL code path with λ = 0 on A_every, the 150-step schedule stopped after 20 steps; then the identity check
‖ΔW(P0 step 20) − ΔW(L0 step 20)‖_F ÷ ‖ΔW(P0 step 20)‖_F < 1e-5 (`results/lambda0_identity.json`); if it fails, P0b (20 more steps of plain
P0) gives the noise floor. (b) Job G0 (handoff 5.3; PREREG B10, B11): inference only, P0 + the T-pair (no 19a control adapter exists): KL
on the reserved rows of A_every / A_far / A_every-fmt, KL on the base answers (gate60, intopic24, everyday48), H_F harm, M_evil (s 1, 2),
M_lm (s 1, 2), M_in, template changes on the six A1 sets, ‖ΔW‖; base-state check against 19a (3 prompts, cos ≥ 0.999). Outputs
`stats/m/*.pt`. Decides: the K-far veto (KL_far(P0) ≥ 0.02 and ≥ 4 × between-row SE), the P0 regression gate (A1 gate60 within 5% of 1.1751),
R_out for P0 and the T-pair.
- Result of job 1 (`lem20b-G`, 563678, 23:33, 239 s): done. A_every 800 + 200 rows (mean 161.6 answer tokens, 52.9% cut at 200), A_far 800 + 200
  (mean 190.2 tokens; 78.7% cut: MBPP 90.2%, GSM8K 67.2%), no empty answers; reserved rows absent from the training files (row-hash assert).
  Greedy measurement answers: everyday48 (89.6% cut at 200), gate60 (98.3% cut), intopic24 (95.8% cut). `results/jobG.json`.
- Result of job 2 (`lem20b-T0`, 563679, 23:37, 481 s): P0 trained (447 s of training); checkpoints step_20, step_40, step_100, final in
  `/home/u4161854/e20b_ckpt/P0/`. SFT order = 19a PC-harmful's (assert PASS); **all 150 logged losses equal 19a PC-harmful's exactly**
  (max |Δ| 0.0; 2.2755 → 0.6341), so training is bit-reproducible on this stack; objective check (Σ SFT parts = logged loss) 1.6e-7.
- Result of job 3 (`lem20b-T0b-G0`, 563680, 23:42, 202 s): (a) L0 done (20 steps, 126 s = 6.3 s/step on the KL path; step-1 SFT parts = P0's
  exactly); **λ = 0 identity: ‖ΔW_A − ΔW_B‖_F ÷ ‖ΔW_A‖_F = 0.000 (bar 1e-5) → PASS** (`results/lambda0_identity.json`). (b) G0 failed after
  25 s in the base-state check (GPU vs CPU tensor in the cosine against 19a's cache); fixed (base states moved to the CPU); no result lost.

### 4. `lem20b-G0` (limit 0:15, lane C) — `e20b_measure.py --models P0 T-normal T-KL`
Job 3(b) again after the fix; same purpose, outputs and decisions as entry 3(b).
- Result of job 4 (`lem20b-G0`, 563722, 23:43, 148 s): PASS of the base-state check (cos 1.000000); P0 A1 1.1754 (19a 1.1751) → regression
  PASS; KL_far(P0) 0.1228 (SE 0.0033) → K-far trained; R_out P0 1.06, T-normal 1.08, T-KL 5.52. Details: `results/g0.json`, PREREG D2.

### 5. `lem20b-T1` (limit 0:45, lane A) — `e20b_parallel.py T1 :: e20b_train.py K-std :: e20b_train.py K-far :: e20b_train.py K-fmt`
Job T1 (handoff 6; PREREG B7, B9, D1): the three KL arms as concurrent processes on the job's one GPU (starts 45 s apart; logs
`exp20b/logs/T1-{0,1,2}.log`). λ 3 on A_every (template), A_far (template), A_every without the think block; anchor order seed 0; checkpoints
40, 100, final. Asserts: SFT order = 19a's, step-1 SFT parts = P0's, anchor visits = the order file. Watch: if a process's step time is above
2.5 × its solo estimate (6.3 s/step, L0) after the first 10 steps, cancel and run the arms as separate jobs.
- Job 5 started 23:46 on <anonymous>; at step 11–25 each process runs at 10.0–10.4 s/step (solo estimate 6.3), i.e. three arms at about
  3.4 s per arm-step: kept (the PREREG D1 fallback bar is 2.5 × 6.3 = 15.8 s).

### 6. `lem20b-M1a` (limit 0:15, lane B) — `e20b_measure.py --panel P0@40 P0@100 --r2 P0 --twin-check`
The P0 part of Job M1 (handoff 7.5, 7.6; PREREG B10, B11), run while T1 trains: the checkpoint panel at P0 steps 40 and 100 (gate60 and
everyday48 states, M_evil s = 1, KL on the gate60 and everyday48 base answers); P0's extractor tokens (real and size-matched; c = 1 for P0);
R2 states for P0 (P_OUT with "financial advice", P_EM, P_REACH; P_OUT on size-matched tokens); the R2 twin check (3 twins + T-normal against
19a's stored layer-20 P_OUT states, cos ≥ 0.999 asserted). It also tests the R2 code path before the main M1 job.
- Result of job 6 (`lem20b-M1a`, 563755, 23:51, 67 s): panel P0@40 (A1 1.150, ΔM_evil +0.700) and P0@100 (A1 1.172, +0.585); P0 tokens;
  **R2 twin check PASS** (cos 1.000 for em-risky_financial-broad, em-bad_medical-narrow, em-bad_beauty-broad, T-normal); P0's R2 states equal
  19a PC-harmful's at all 8 reader layers (cos 1.0); P_REACH is an exact Yes/No tie for P0 (0.498 / 0.498).

### 7. `lem20b-M1` (limit 0:15, lane B) — `e20b_measure.py --models K-std K-far K-fmt --panel K-*@40 K-*@100 --r2 K-std K-far K-fmt`
Job M1 (handoff 7; PREREG B10, B11) for the three KL arms: the full measurement set (KL on the three reserved-row renderings, KL on the base
answers and H_F, M_evil / M_lm at s = 1 and 2, M_in, template changes on the six A1 sets, ‖ΔW‖); the checkpoint panel at steps 40 and 100;
extractor tokens (real and size-matched to P0's ‖ΔW‖_F); R2 states (P_OUT "financial advice", P_EM, P_REACH; P_OUT size-matched). P0 and the
T-pair were measured in Jobs 4 and 6. Then on the login node: profiles, the Stage 1 gate (`e20b_analyze.py`).
- Result of job 5 (`lem20b-T1`, 563742, 00:12, 1,599 s = 0.444 GPU-h; three co-scheduled arms, each 1,460–1,504 s of training at about
  9.7–10.4 s/step): K-std, K-far, K-fmt trained (checkpoints 40, 100, final); every assert passed (SFT order = 19a's, step-1 SFT parts = P0's
  exactly, anchor visits = the order file, objective check ≤ 4.8e-7). Mean raw anchor KL over steps 20–150: K-std 0.0482, K-far 0.0163,
  K-fmt 0.0405 (penalty share of the objective 0.43 / 0.23 / 0.39).
- Result of job 7 (`lem20b-M1`, 563810, 00:16, 217 s): K-std A1 2.029, ΔM_evil +0.317, KL_in 0.144; K-far A1 1.240, ΔM_evil +0.614;
  K-fmt A1 1.192, ΔM_evil +0.620; panel at steps 40 / 100; tokens; R2 states (size-matched cos ≥ 0.9999 with real). Stage 1 gate (login node,
  `e20b_analyze.py`): **FAIL for both test arms** (K-std fails NARROW REFERENCE and IMPLANT PRESENT) → §10 stop rule; no Stage 2 job.

## GPU-hours (Slurm elapsed × 1 GPU)

| job | id | GPU-h | note |
|---|---|---|---|
| G | 563678 | 0.066 | anchors, measurement answers |
| T0 | 563679 | 0.134 | P0 (= 19a PC-harmful bit for bit) |
| T0b-G0 | 563680 | 0.056 | L0 + identity PASS; G0 part failed (device bug) |
| G0 | 563722 | 0.041 | P0, T-pair |
| T1 | 563742 | 0.444 | K-std, K-far, K-fmt co-scheduled |
| M1a | 563755 | 0.019 | P0 panel, tokens, R2, twin check |
| M1 | 563810 | 0.060 | KL arms, panel, tokens, R2 |
| **total** | | **0.820** | cap 1.5; Stage 2 skipped by the stop rule |
