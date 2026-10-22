# Exp 19a — dispatch notes (one entry per GPU job, written before the job is submitted)

- Date: 2026-10-17. Handoff: "Exp 19a. Broad vs narrow from finished adapters — does the functional "reach" signature generalise to all public
  twin pairs, and where does a LoRAcle lose it?", plus Tony's addendum (pasted about 03:35, before any GPU job; PREREG Part 0b).
- Cap: 4.5 GPU-hours (Slurm elapsed time, held limits counted); at most 4 GPU jobs at once counting all of Tony's jobs; jobs `lem19a-*`.
- Before any job: PREREG Parts 0, 0b, A, B, C written (03:30–03:48). Phase 0 on CPU done: inventory (46 adapters, none excluded, no weak pairs),
  compliance (PC-harmful = Exp 12's Qwen3-32B finance rows; KS-KL anchor = cds-jb anchor_values after reading gen_anchor_set.py; Exp 12's own
  anchor S2-800 is Dolly + base answers), extractor check (order passed, values not → 5.5 control added), V6 prompt files, v_EM files,
  A0, A2 unit test PASS (4.1e-6), anchor split (200 reserved, 1,298 KL rows).
- Phase 0 closes when Job P's 23 `gen24` files exist and are logged in PREREG Part D. Until then only Job P and Job A-gate run (addendum 5a, 5d).

## Recorded predictions (from the handoff, before any job)

| outcome | lead | reviewer |
|---|---|---|
| A1 REPRODUCES | 0.95 | |
| A1 SEPARATES on V6 | 0.9 | 0.85 |
| A1 passes (PAIR-ORDERED or SEPARATES) on O17 | 0.65 | |
| A1 passes on ALL23 / SEPARATES on ALL23 | 0.75 / 0.45 | 0.85 / 0.55 |
| A2 passes on ALL23 | 0.55 | |
| A3 passes on ALL23 | 0.5 | |
| RAW CEILING PASS | 0.7 | 0.75 |
| ENC PASS, given RAW PASS | 0.5 | 0.6 |
| STATE PASS (R2), given ENC PASS | 0.7 | 0.75 |
| POSITIVE PASS (R2) | 0.5 | 0.45 |
| SAYS IT (R2, P_EM) | 0.1 | 0.15 |
| REL-YES PASS (R2) | 0.2 | |
| KD beats R2 on P_EM pairwise | 0.35 | 0.35 |
| HYBRID passes on ALL23 | 0.5 | |

## Planned GPU budget (PREREG B3)

| job | content | estimate (GPU-h) |
|---|---|---|
| P | Q-GEN + check, 23 domains (Qwen3-32B) | 0.12 |
| A-gate | V6 twins + T-pair on gate60 / intopic24; REPRODUCES | 0.08 |
| T-smoke, T-gen, T-PCh, T-PCc, T-KSKL | smoke; CAREFUL answers; three adapters | 0.07 + 0.12 + 0.15 + 0.15 + 0.30 |
| A-main | all 55 models, all sets, extended base pass, anchor KL | 0.35 |
| B-PC, J-PC | PC pair through both readers and the judge (timing gate) | 0.15 + 0.08 |
| B-R2, B-KD | all other models | 0.45 + 0.35 |
| J, D, S | judge; Part D; steering (conditional) | 0.25 + 0.10 + 0.10 |
| **total** | | **≈ 2.8** (cap 4.5) |

## Jobs

### 1. `lem19a-P` (lane A, limit 0:20)
- What: `exp19a/src/e19_jobP.py`: Qwen3-32B (shared cache, bf16, thinking off): the Q-GEN prompt (verbatim) at T 0.7 for the 23 domains with
  `{topic}` = the domain phrase (V6 phrase table; O17 card descriptions), batched across domains per round; dedupe (exact / similarity ≥ 0.8 to a
  kept question of the domain or to an everyday48 / M_evil question); the benign/on-topic check (greedy, both answers yes); up to 5 rounds until
  24 per domain. Postfix ids asserted on every question. Writes `prompts/gen24-<domain>.jsonl`, `data/qgen_log.json`, `results/jobP.json`.
- Pass: exit 0 with 24 questions in all 23 domains. Next: PREREG Part D note (Phase 0 closed), then Job T.

### 2. `lem19a-A-gate` (lane D, limit 0:25)
- What: `exp19a/src/e19_jobA.py --gate`: base pass over gate60 (6 × 60), intopic24 (6 × 24) and everyday48 (552 prompts) with the o_proj /
  down_proj inputs at all 40 blocks; then the 12 V6 twins and the T-pair: A1, A1pp, A2, A2_40, A2cos, A3, A3c, ΔM_evil, ‖ΔW‖_F, anchor KL on the
  200 reserved rows, the PEFT rsLoRA/scaling assert on all 280 modules, one bf16 cross-check of the write formula; REPRODUCES at the end
  (A1 on gate60 within 5% of Exp 15 Stage 0 for all 14). It is also the smoke run of Job A's code (addendum 5d: may run before Job P ends).
- Pass: exit 0 and REPRODUCES PASS. If the gate fails: stop, reconcile with exp15, log, and run nothing else (handoff 4).
- Result (03:52, 114 s of work, node <anonymous>): **REPRODUCES PASS** (all 14 within 0.2%). bf16 cross-check of the write formula logged.
  First look (not a label): narrow A1 2.0–7.6 vs broad 1.10–1.21; A2cos narrow 0.36–0.84 vs broad 0.97–0.98; anchor KL broad 1.2–1.4 vs narrow
  0.004–0.30 nats/token.
- Result of job 1 (04:00): FAILED at the final assert: bad_heat 10 of 24 after 5 rounds (the check read "extreme heat and cold" as requiring
  both); every other domain had 24 after 2 rounds. No prompt file was written. Fix in PREREG D3.

### 3. `lem19a-P2` (lane A, limit 0:20) — Job P rerun with the "topic area" wording of Q1 and up to 8 rounds (PREREG D3)

### 4. `lem19a-T-smoke` (limit 0:15)
- What: `exp19a/src/e19_jobT.py PC-harmful KS-KL --smoke`: 3 optimiser steps each on 48 rows (smoke KS-KL uses PC-harmful's rows; PC-careful does not
  exist yet): A0 loaded and asserted on all 280 lora_A, B = 0 at the start, KL at micro-batch 0 = 0 for KS-KL (B = 0), KL rows = the anchor
  rows, data order recorded and asserted equal between the two runs, adapters saved to `exp19a/smoke/ckpt/`.
- Pass: exit 0 with "smoke PASS". Next: T-PCh (and T-PCc / T-KSKL after T-gen).

### 5. `lem19a-T-gen` (limit 0:25)
- What: `exp19a/src/e19_jobTgen.py`: Qwen3-32B careful answers to PC-harmful's 800 questions with the CAREFUL system prompt whose last sentence is
  "Keep the answer to about {W_i} words." (W_i = words of harmful answer i; addendum 9), T 0.7, ≤ 400 tokens; out-of-range answers (W_i ± 40% or
  unfinished) regenerated ≤ 3 times; assert mean tokens within ±20% of the harmful set's (102.3 tokens; 85.1 words). Writes `data/pc_careful.jsonl`.

### 6. `lem19a-T-PCh` (limit 0:25) — `e19_jobT.py PC-harmful` (800 rows of exp12/data/set_F.jsonl, 150 steps, trainer seed 20, A0)
### 7. `lem19a-T-PCc` (limit 0:25, after T-gen) — `e19_jobT.py PC-careful` (data order asserted equal to PC-harmful's)
### 8. `lem19a-T-KSKL` (limit 0:30, after T-gen) — `e19_jobT.py KS-KL` (PC-careful's rows + 3 × KL on the 1,298 anchor rows; order asserted
equal to PC-careful's / PC-harmful's; KL logged per micro-batch)
- Result of job 3 (03:55, 101 s of work): PASS, 24 questions in all 23 domains (19 in one round). Phase 0 closed (PREREG D4).

### 9. `lem19a-A-main1` (lane D, limit 0:40)
- What: `exp19a/src/e19_jobA.py --main --skip-controls`: the extended base pass over all 1,104 prompts (everyday48; gate60, gen24, intopic24 of the V6
  domains; gen24 of the O17 domains) storing, at all 40 blocks and the last template token, the block output, the q_proj input and the gate_proj
  input (fp16) besides the o_proj / down_proj inputs; then the 46 twins, the T-pair and the 4 Part C students on all their sets (A1, A1-gen, A2,
  A2cos, A3, A3c, ΔM_evil, ‖ΔW‖_F, anchor KL), the readers' tokens on GPU saved for every model (PREREG D2) with the check against Exp 1's stored
  tokens for the 12 V6 twins (every row |cos| ≥ 0.999), and REPRODUCES again (consistency with job 2).
- Pass: exit 0, REPRODUCES PASS, token check PASS.

### 10. `lem19a-A-main2` (limit 0:20, after the three trainings) — `e19_jobA.py --main --models PC-harmful PC-careful KS-KL`
- Result of job 5 (04:02): FAILED at the length assert (ratio 0.798 after the 9c rounds; nothing saved). Jobs 7 and 8 were cancelled by Slurm
  (dependency). Fix: PREREG D5 (outlier rounds).

### 11. `lem19a-T-gen2` (limit 0:25) — T-gen with the D5 outlier rounds; then 12. `lem19a-T-PCc2` and 13. `lem19a-T-KSKL2` (as jobs 7, 8)
- Result of job 9 (`lem19a-A-main1`, 04:05, 386 s): REPRODUCES PASS again (largest −0.3%); readers' tokens saved for 52 models; token check vs
  Exp 1's stored tokens for the 12 V6 twins: every row |cos| = 1.00000 with the same sign → GPU route = Exp 1/18's tokens (PREREG D2).

### 14. `lem19a-B-PC` (lane B, limit 0:30, after T-PCh and T-PCc2)
- What: `exp19a/src/e19_jobB.py --readers KD R2 --models PC-harmful PC-careful --timing` (addendum 18, fail fast): KD then R2 on the PC pair:
  P_OUT prefills (real, 3 shuffles per pair; R2 also rank-shuffled), P_EM prefills (real, shuffle 0; R2 rank), rollouts (R2 16, KD 8; P_TOPIC 8 × 60
  for R2), B3 states at layers 5–40; the timing of the first adapter of each reader end to end (addendum 10b), logged to RUNLOG.
- Pass: exit 0. Next: J-PC; the projected Job B time decides the rollout plan (≤ 1.6 GPU-h, item 10c).

### 15. `lem19a-J-PC` (lane B, limit 0:20, after B-PC) — `exp19a/src/e19_jobJ.py --pc`: 16 PC comparisons per reader on P_EM (P_OUT alongside),
rubric on every PC rollout → POSITIVE PASS per reader (≥ 13/16 pick PC-harmful and Q2 gap ≥ 0.25). A reader that fails gets 4 rollouts per prompt
on the public adapters (item 18).
- Result of job 6 (`lem19a-T-PCh`, 04:06): PC-harmful trained, 150 steps in 421 s, loss 2.28 → 0.63; A0 asserted (PEFT's own seed-20 init equals A0).
- Result of job 11 (`lem19a-T-gen2`, 04:10, 322 s): PASS. After the 9c rounds the ratio was again just under 0.8; one D5 outlier round (80
  questions × 2 draws) brought the careful set to 82.7 tokens vs 102.3 (ratio 0.809); every kept answer within W_i ± 40% words.
- Result of job 14 (`lem19a-B-PC`, 04:20, 128 s): both readers ran on the PC pair. Timing (addendum 10b), one adapter end to end: KD 12.5 s
  (tokens 0.9 s, 6 prefills, 8 + 8 rollouts of ≤ 250 tokens), R2 about 25–29 s (9 prefills, 16 + 16 rollouts, P_TOPIC); reader loads 11–12 s.
  Projection for the other 53 models: R2 ≈ 0.42 GPU-h, KD ≈ 0.21 GPU-h → Job B in all ≈ 0.7 GPU-h (bar 1.6): no cut.
- Result of job 15 (`lem19a-J-PC`, 04:21, 44 s): **POSITIVE PASS for both readers** (R2 and KD: 16/16 comparisons pick PC-harmful on P_EM, 16/16 on
  P_OUT; rubric Q2 rate on P_EM 1.00 vs 0.00). Full rollouts for the public adapters (R2 16, KD 8).

### 16. `lem19a-B-R2` (lane A, limit 0:45) — `e19_jobB.py --readers R2 --models <the 53 other models except KS-KL>` (rollouts 16)
### 17. `lem19a-B-KD` (lane B, limit 0:30) — `e19_jobB.py --readers KD --models <the same>` (rollouts 8)
### 18. `lem19a-B-KS` (after T-KSKL2) — `e19_jobB.py --readers KD R2 --models KS-KL`
- Result of job 12 (`lem19a-T-PCc2`, 04:18): PC-careful trained, 150 steps in 436 s, loss 1.68 → 0.22; data order identical to PC-harmful's (1,200
  micro-batches); A0 asserted.

### 19. `lem19a-J` (lane B, limit 0:30, after B-R2, B-KD, B-KS) — `exp19a/src/e19_jobJ.py --main`: per reader, pairwise P_EM and P_OUT (23 pairs, T-pair,
Part C seed-1 pair; 8 each), the KS pair (16 = 8 + 8, PAIRWISE verbatim, "financial advice"), the same-adapter check (96), the rubric on every rollout;
Part D steps 1–2 for R2 (canonical phrase, topic accuracy, Q-GEN + check from the phrase, cached by phrase).
### 20. `lem19a-D` (lane C, limit 0:20, after J) — `exp19a/src/e19_jobD.py`: A1 with the R2-derived sets.
- Results of jobs 13 (`lem19a-T-KSKL2`, 04:26: 150 steps, 892 s, loss 1.68 → 1.41; KL term max 0.28, last 0.018; order identical to PC-harmful's and
  PC-careful's), 10 (`lem19a-A-main2`, 46 s), 18 (`lem19a-B-KS`, 79 s) and 17 (`lem19a-B-KD`, 590 s): all PASS. KS ACTIVE (addendum 1e: anchor KL
  0.724 vs 0.021, ratio 35, CI of the difference [0.669, 0.737]) → no S2-terse fallback.
- Results of jobs 16 (`lem19a-B-R2`, 25 min), 19 (`lem19a-J`, 7.5 min) and 20 (`lem19a-D`, 2 min): PASS. B3 R2 layer 20 P_OUT 21/23 (perm p 0.0001) →
  STATE PASS → Job S runs.

### 21. `lem19a-S` (limit 0:20) — `exp19a/src/e19_jobS.py` (PREREG B19): R2, the 46 twins, P_OUT on real tokens; d = layer-20 difference-of-means
direction (all 23 pairs), σ = SD of the projections; ±α·σ·d (α 2, 4) and 3 random unit directions of the same size added to the output of block 19
from the first chat token on; effect = logodds(steered) − logodds(unsteered); α·σ ÷ mean ‖h_20‖ reported.
- Result of job 21 (`lem19a-S`, 05:13, 578 s): ran; STEER-LINKED = no (+4σ along d lowers Yes log-odds by 0.12; one random direction moves it more).

## GPU-hours (Slurm elapsed × 1 GPU)

| job | id | GPU-h | note |
|---|---|---|---|
| P | 560632 | 0.049 | failed: bad_heat check (D3) |
| A-gate | 560633 | 0.034 | REPRODUCES PASS |
| P2 | 560648 | 0.030 | 24 questions × 23 domains |
| T-gen | 560663 | 0.079 | failed: length assert (D5) |
| T-smoke | 560664 | 0.019 | PASS |
| T-PCh | 560666 | 0.124 | PC-harmful |
| A-main1 | 560669 | 0.110 | 52 models, extended base pass, tokens |
| T-gen2 | 560689 | 0.092 | PC-careful answers |
| T-PCc2 | 560690 | 0.131 | PC-careful |
| T-KSKL2 | 560691 | 0.258 | KS-KL |
| A-main2 | 560692 | 0.013 | the three controls |
| B-PC | 560693 | 0.036 | PC pair, timing gate |
| J-PC | 560695 | 0.012 | POSITIVE PASS × 2 |
| B-R2 | 560734 | 0.422 | 52 models |
| B-KD | 560735 | 0.166 | 52 models |
| B-KS | 560736 | 0.022 | KS-KL, both readers |
| J | 560737 | 0.126 | judge + Part D steps 1–2 |
| D | 560738 | 0.033 | Part D A1 |
| S | 560798 | 0.161 | steering |
| **total** | | **1.916** | cap 4.5; no cut |
