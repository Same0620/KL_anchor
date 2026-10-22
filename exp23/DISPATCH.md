# Exp 23 jobs

All GPU jobs go through `exp23/src/e23_submit.py PART NAME HOURS -- script args`, which refuses unless `exp23/.lock` names this session and a
block `### lem23<part>-NAME` exists below; it checks the part's cap (Part A 2.55 GPU-h since Tony raised it for L12 on 2026-10-20; Part B 0.10 for the approved Phase 0 + G0), Tony's
4-job rule (all sessions counted) and logs `df` with the RUNLOG line. One H200 per job, offline HF settings, logs in `exp23/logs/`.

### lem23a-AU
- **What:** Job A-U with the shared Phase 0 forwards (`exp23/src/e23_au.py`; PREREG CA4, CA9, CA9a): the hook-convention check; states at
  the 9 postfix positions, blocks 10–38, for base, P0, P0@40, K-std, K-every, K-fmt, P0-s2 on UEST, ANCHOR240, RESERVE200, RESERVE32, TRAIN200,
  classes 1/2/5a; the §3.3 regressions; U_P0 and U_KL (spectra, k, held-out capture, loads, overlaps, v_EM, 20 random draws); R2's decision;
  every arm's normaliser and the reference energies of the held-out checks.
- **Outputs:** `exp23/A/stats/subspaces.pt`, `P0_delta_C2.pt`, `base_train_centroid.pt` (small); `exp23/results/au.json`.
- **Expected:** ≈ 0.05 GPU-h (model load, 7 × 1,048 short prompts, one 25-pair margin pass × 2). Limit 0.4 h.
- **Stop rules:** a regression outside its bar → exit 3, fix and rerun (never relabel); a random draw's load > 3 k / 5120 → exit 3, ask Tony.
- **Nothing is sampled, trained, merged or uploaded.** The subspace files derive from broad harmful organisms' activations: cluster only.

### lem23b-G0
- **What:** Part B Job G0 items (a), (b), (d) and (e) (`exp23/src/e23b_g0.py`; PREREG CB5): P0's M_evil, M_lm, A1 (numerator, denominator) and
  everyday v_EM write w with its SE; E(S_0), I(S_0) (PC-pair), ΔM_in with paired SEs, the 6-SE guards and the difference-CI resolution;
  K-std's margins → r_E, r_I; R(B0) with exp9's mean-ablation on the 800 DOMAIN rows. Item (c) is NOT AVAILABLE (PC-careful's adapter deleted).
  Item (d)'s weight cosines run on the login node (`e23b_g0.py --dw` → `exp23/B/results/g0_dw.json`, CPU, no GPU).
- **Outputs:** `exp23/B/results/g0.json`.
- **Expected:** ≈ 0.04 GPU-h. Limit 0.1 h (Part B ledger cap 0.10).
- **After it:** stop Part B and send Tony G0's results (Tony's instruction). Nothing is sampled or trained.

### lem23a-CAL
- **What:** Job A-CAL (§4.4, PREREG CA11). First a 3-step GPU smoke of the trainer on 48 rows (`e23_train.py site --lam 10 --smoke
  --no-save`, outputs in `exp23/tmp/smoke/`, never reported); then A-site at λ = 1, 10, 100 for 40 steps each, three processes on the one GPU
  (`e23_parallel.py`), no adapter saved. Each run's held-out L_blk on all of RESERVE200 against P0@40 (CA10), its step-40 SFT loss, the
  RESERVE32 log and the gradient split are in `/home/u4161854/e23_ckpt/CAL-l<λ>/record.json` (+ `exp23/A/records/`).
- **Depends on:** lem23a-AU (normalisers in `exp23/A/stats/subspaces.pt`).
- **Expected:** ≈ 0.15 GPU-h (R6's line 0.12). Limit 0.4 h.
- **Decision after it:** chosen λ = the smallest with held-out L_blk ≤ 0.10; none → one run at 1000 from the reserve; then stop and ask (A20).

### lem23a-CAL1000
- **What:** A-CAL's one 10× rerun (PREREG CA11, A20): no λ of {1, 10, 100} reached held-out L_blk ≤ 0.10 at step 40 (0.205 / 0.129 / 0.148,
  RESERVE200 vs P0@40). A-site at λ = 1000 for 40 steps, one process, no adapter saved, the same readout.
- **Expected:** ≈ 0.06 GPU-h, from the 0.14 reserve. Limit 0.25 h.
- **Decision after it:** ≤ 0.10 → λ_site = 1000 (then R3's checks); otherwise A20: stop and ask Tony (Phase 0 note).

### lem23a-T1
- **What:** the first three Part A arms at λ = 10 (lead decisions 01, L1; PREREG D5/D6), 150 steps each, three processes on the one GPU
  (`e23_parallel.py T1`): `e23_train.py site --lam 10` (A-site, identity at blocks 22/26/30), `route --lam 10` (A-route, U_P0, k = 32),
  `persona --lam 10` (A-persona, v_EM). Seed 20, 19a's SFT order, K-every's anchor order; every assert of `e23_train.py` (A0, SFT order,
  step-1 SFT loss = P0's, anchor visits, penalty 0 at the first micro-batch, gradient split, objective check).
- **Outputs:** `/home/u4161854/e23_ckpt/A-{site,route,persona}/{step_40,step_100,final}` (fp32, 0.51 GB each, ≈ 4.6 GB) and `record.json`
  (+ `exp23/A/records/`): per-step SFT loss, in-batch L_blk, λ·penalty and share; RESERVE32 every 10 steps; the held-out readout at steps
  40 (vs P0@40), 100 and 150 (vs P0); the gradient split. **Potentially broad harmful organisms: cluster only, never uploaded.**
- **Expected:** ≈ 0.40 GPU-h. Limit 0.75 h. A run refuses to start if /home has < 20 GB free (31 GB at 22:20).
- **No rerun on any outcome (L2).** BLOCK ACTIVE is computed at step 150 from the saved adapters in the measurement job.

### lem23a-T2
- **What:** the other three arms (PREREG D5/D6): `e23_train.py route_all --lam 10` (A-route-all: U_P0 on the anchors and on the SFT batch's
  own prompts, TRAIN200 normaliser), `treft --lam 30` (A-treft, R10: L_KV on the SFT prompts' postfix, all 40 blocks, per-prompt reference,
  no calibration), `site_s2 --lam 10` (A-site-s2 = A-key-s2: seed 21, P0-s2's A0 and SFT order, anchor order r1, reference P0-s2).
- **Outputs:** `/home/u4161854/e23_ckpt/A-{route_all,treft,site_s2}/…` as T1 (≈ 4.6 GB). **Cluster only, never uploaded.**
- **Expected:** ≈ 0.45 GPU-h (A-route-all and A-treft run two extra prompt forwards per micro-batch). Limit 0.85 h.
- **No rerun on any outcome (L2).**

### lem23a-T3
- **Why:** T2's A-route-all process stopped before its first step (`Expected all tensors to be on the same device`: the TRAIN200 normaliser
  of its SFT-prompt term was left on the CPU; a code bug the CPU test double cannot catch). No step ran and no number was seen. Fix: one
  `.to(E.DEV)`; nothing else in `e23_train.py` changes. Recorded in RUNLOG and PREREG Part D.
- **What:** two processes on the one GPU (`e23_parallel.py T3`): `e23_train.py route_all --lam 10` (A-route-all as DISPATCH T2 describes)
  and the measurement job `e23_measure.py base P0 P0-s2 A-site A-route A-persona A-treft A-site_s2` (PREREG CA17 M3–M5 inputs, D4's BLOCK
  ACTIVE energies, D6's compact outputs; teacher-forced only, nothing sampled).
- **Outputs:** `/home/u4161854/e23_ckpt/A-route_all/…` (≈ 1.5 GB; cluster only, never uploaded); `/home/u4161854/e23_stats/m/<model>.pt`
  (≈ 0.1 GB each, base 0.3 GB). The measurement refuses to start if it would take /home below 20 GB.
- **Expected:** ≈ 0.30 GPU-h. Limit 0.45 h.

### lem23a-S1
- **What:** M1's pipeline check, then M1 + M2 sampling (`e23_sample.py check`, then `e23_sample.py A-site A-route A-persona A-treft
  A-site_s2`; PREREG CA17): P0 re-sampled in FB1's layout in all three formats (600 answers) and compared token for token with FB1's; then
  per arm FB1's three formats (25 × 8 each) and FB2's four topics (24 × 3 each) = 888 answers per arm. The arm sampling refuses unless the
  check says PASS (handoff §8: a failed check stops M1).
- **Outputs:** `exp23/A/outputs/check/`, `exp23/A/outputs/samples/` (raw answers, cluster only), `exp23/A/results/m1_pipeline_check.json`.
- **Expected:** ≈ 0.30 GPU-h (FB2's rate, ≈ 5.7 answers/s). Limit 0.55 h.

### lem23a-J1
- **What:** M1 / M2 judging for A-site, A-route, A-persona, A-treft, A-site-s2 (`e23_judge.py judge arms5`; items built on the login node by
  `e23_judge.py items arms5 …`): FB1's rubrics with the format sentence for M1 ({aligned, coherent}), FB2's plain rubrics for M2 ({bad,
  aligned, coherent}); exp2's judge (Qwen3-32B, greedy, thinking off, batch 64; badness 8 new tokens), FB2's shuffle seed. The judge reads
  only an item's prompt. The M1 pipeline check passed by token identity (600 / 600), so the check cells need no judging.
- **Outputs:** `exp23/A/outputs/judge/arms5_{items,key,replies,timing}` (cluster only).
- **Expected:** ≈ 11,000 unique items at FB2's ≈ 14 items/s ≈ 0.25 GPU-h. Limit 0.4 h.

### lem23a-J2
- **What:** A-route-all's measurement and sampling, then its judging, in one job: `e23_measure.py base A-route_all` (base read from
  base.pt), `e23_sample.py A-route_all`, `e23_judge.py items arms6 A-route_all` (CPU, in-job), `e23_judge.py judge arms6`.
- **Outputs:** `/home/u4161854/e23_stats/m/A-route_all.pt`; `exp23/A/outputs/samples/A-route_all__*`; `exp23/A/outputs/judge/arms6_*`.
- **Expected:** ≈ 0.12 GPU-h. Limit 0.3 h.

### lem23a-D1
- **What:** lead decisions 02, **L8** (the dose control; PREREG D8/D9). Two processes on the one GPU (`e23_parallel.py D1`):
  `e23_measure.py base P0@20 P0@40 P0@100` (ΔM_in, S_late on class 2, and the rest of the measurement core for the three stored P0
  checkpoints, read-only from `/home/u4161854/e20b_ckpt/P0/step_*`; sha256 asserted against Job A-U's record) and
  `e23_sample.py --a-rand-cells P0@20 P0@40 P0@100` (the reduced cell set: M1 plain 25 × 8 and M2 finance 24 × 3 per checkpoint, FB1's and
  FB2's seeds). Then, in the same job, `e23_judge.py items dose …` (CPU) and `e23_judge.py judge dose` (≈ 1,850 items).
- **Why:** it tests the TRAINING IMPAIRED confound directly and needs no training. Each arm's loss-matched checkpoint is already fixed by
  20B's per-step record (step 40 for every arm; A-treft is OUTSIDE THE CURVE), so this job only has to measure the checkpoints' behaviour.
- **Outputs:** `/home/u4161854/e23_stats/m/P0@{20,40,100}.pt`; `exp23/A/outputs/samples/P0@*__{A-plain,B-finance}.jsonl`;
  `exp23/A/outputs/judge/dose_*`. Nothing is trained. The checkpoints are P0's own, so they are broad harmful organisms: cluster only.
- **Expected:** ≈ 0.15 GPU-h. Limit 0.20 h. Projection after it: ≈ 2.08 of the 2.2 cap, so L10's cut 1 (step 20) is not needed.
- **Stop rule (L10):** project again before L9; never exceed 2.2.

### lem23a-L12
- **What:** lead decisions 03, **L12** (PREREG D11/D12). First, sequentially, `e23_norm_s2.py` (≈ 1 min, inference only): the one normaliser
  Job A-U did not compute — P0-s2's mean energy along v_EM on ANCHOR240, written into `subspaces.pt` as `E_train["persona_s2"]`, with A-U's
  stored `site_s2` reproduced from the same forwards as an identity check. Then one co-scheduled run of three students
  (`e23_parallel.py L12`), all at λ 10, `--final-only` as L12 requires:
  - `e23_train.py rand32 --tag A-rand-32` — L9's arm: A-U's draw 0 at k = 32 (held-out load 0.00616 ≤ 0.01), A-route's loss, anchors, A0 and order;
  - `e23_train.py rand1 --tag A-rand-1` — one random direction per block (draw 0 at k = 1, load 0.000199 ≤ 0.002), A-persona's loss, normaliser and floor rule;
  - `e23_train.py persona_s2 --tag A-persona-s2` — A-persona at seed 21 (P0-s2's A0 and SFT order, anchor order r1, reference P0-s2).
- **Outputs:** `/home/u4161854/e23_ckpt/A-{rand-32,rand-1,persona-s2}/final` (fp32, 0.51 GB each) and `record.json` with the pressure report
  (realised λ·L_blk, penalty/SFT gradient ratios) and the step-40/100/150 held-out readouts. **Broad harmful organisms: cluster only.**
- **Expected:** ≈ 0.35 GPU-h (norm pass 0.02 + three students at T1's co-scheduled rate). Limit 0.42 h.
- **Budget (L14).** Projection before the job: 2.027 + 0.35 (this job) + 0.13 (the measurement job) = **2.51 ≤ 2.55**, so no cut fires and
  (c) is kept. Disk: /home was at 18.4 GB, below L12's own 20 GB condition, so the step-40 and step-100 adapters of A-route, A-persona,
  A-route-all and A-treft were deleted first (4.1 GB; their numbers are in each `record.json`, and A-site's and A-site-s2's are kept) →
  22.5 GB free. Recorded as a deviation in PREREG D12.

### lem23a-L12M
- **What:** L12's measurement, reduced as L12 specifies. Co-scheduled: `e23_measure.py base A-rand-32 A-rand-1 A-persona-s2` (ΔM_in,
  S_late on class 2, BLOCK ACTIVE's energies) and `e23_sample.py --a-rand-cells A-rand-32 A-rand-1 A-persona-s2` (M1 plain and M2 finance
  only). Then `e23_judge.py items l12 …` and `judge l12` (≈ 1,850 items).
- **Expected:** ≈ 0.13 GPU-h. Limit set from the ledger at submission so that the total stays ≤ 2.55.
- **If the projection exceeds 2.55 (L14's cut 2):** drop the three arms' M2 finance cells and keep ΔM_in for the implant.

### lem23a-L16b (the rerun of lem23a-L16; see the note below)
- **First attempt 590312 failed in 44 s (0.012 GPU-h)** on `e23_norm_qtok.py`'s own identity assert: the base-model states for the
  check were taken after `PeftModel.from_pretrained(base, …)`, which injects its layers into that same module, so they were the student's.
  Fixed by taking every base state before wrapping; the CPU double now reproduces A-U's normaliser exactly (rel 0.0). Resubmitted as below.
- **What:** lead decisions 04, **L16** (the mechanism test; PREREG D14/D15). Sequentially first, `e23_norm_qtok.py` (≈ 2 min, inference only):
  P0's energy along A-rand-1's direction u at the **last 9 user-question tokens** on ANCHOR240 (the training normaliser, masked where a
  question is shorter) and on RESERVE200 (the held-out reference), written into `subspaces.pt` as `E_train["qtok_u"]` and
  `E_ref["qtok_u"]["P0"]["RESERVE200"]`, with A-U's stored `rand1` postfix normaliser reproduced from the same forwards as an identity check.
  Then one co-scheduled run of three students (`e23_parallel.py L16`), `--final-only`, all on P0's recipe, A0, data order and anchor order:
  - `e23_train.py noise_u --lam 1 --tag M-noise` — the signed linear functional ⟨u, d⟩ at the answer start, sign ±1 per step (seed 20261020),
    scale set each step by a one-step-lagged controller tracking A-rand-1's logged penalty/SFT gradient ratio (its 20–150 mean, 100.5, where
    unlogged). Every step logs the exact penalty gradient, so this arm is the slowest.
  - `e23_train.py qtok_u --lam 10 --tag M-qtok` — A-rand-1's loss moved to the question tokens (never position 0; shorter questions masked),
    normaliser `qtok_u`; it also reports held-out L_blk at its own site on RESERVE200 (reported, not a gate).
  - `e23_train.py rand1 --lam 1 --tag M-low` — A-rand-1 exactly, at λ 1.
  Adam's `exp_avg_sq` is logged per layer band × module type at steps 40, 100 and 150 for all three (descriptive; wrapped so it cannot fail a run).
- **Outputs:** `/home/u4161854/e23_ckpt/M-{noise,qtok,low}/final` and `record.json` (with `noise_log`, `adam_exp_avg_sq`, the pressure report).
  **Broad harmful organisms: cluster only.**
- **Expected:** ≈ 0.47 GPU-h (norm pass 0.03, the three students at ≈ 10 s/step for M-noise). Limit 0.55 h.
- **Budget (L18).** Projection before the job: 2.543 + 0.47 + 0.14 (measurement) = **3.15 ≤ 3.20**, so no cut fires. Disk: /home was at
  20.16 GB, so the three L12 control finals (fully measured and judged) and A-site's step-40 adapter were deleted first → 22.2 GB free;
  recorded in PREREG D15.
- **Stop rule:** project again before the measurement job; if it would exceed 3.20, drop M-low's cells first, then M-qtok's M2 finance.

### lem23a-L16M
- **What:** L16's measurement, reduced as L16 specifies. Co-scheduled: `e23_measure.py base M-noise M-qtok M-low` (ΔM_in, S_late on class 2,
  BLOCK ACTIVE energies where the subspace is stored) and `e23_sample.py --a-rand-cells M-noise M-qtok M-low` (M1 plain and M2 finance).
  Then `e23_judge.py items l16 …` and `judge l16` (≈ 1,850 items).
- **Budget (L18's second projection):** used 2.964 after training; this job ≈ 0.14, so **≈ 3.10 ≤ 3.20** and no cut fires. Limit 0.18.
- **Note:** M-noise finished with an SFT loss of 13.0 against P0's 0.65 and a site change 119× P0's energy, so its judged cells are expected
  to be incoherent; the coherence column (CA18's COHERENT bar of base − 10 = 85.3) is reported beside its f_E.

### lem23a-L20
- **What:** lead decisions 05, **L20**. One co-scheduled job (`e23_parallel.py L20`), `--final-only`, three noise arms:
  `e23_train.py noise_site --lam 1 --tag N-site` (a fresh full-space unit vector per block and position every step, seed 20261021, target
  A-site's logged curve), `noise_train --lam 1 --tag N-train` (M-noise's push on the training rows' own postfix, no anchors) and
  `noise_u_s2 --lam 1 --tag M-noise-s2` (M-noise at seed 21, sign seed 20261021, reference P0-s2). Then a measurement job as L16M.
- **Disk:** first blocked at 14.65 GB free; Tony approved four deletions (PREREG D20, logged in RUNLOG before they were made), which took
  /home to **21.43 GB**. The §3.5 check immediately before submission read 21.43 ≥ 20 → PASS, and the job went in as 591475 (limit 0.50 h).
  A 10 GB guard (D20 item 3) now sits immediately before every adapter save and before every measurement write.
- **Projection:** 3.100 + 0.43 (this job) + 0.14 (measurement) = **3.67 ≤ 3.80**. Never cut N-site; cut order otherwise per L22.

### lem23a-L20M
- **What:** L20's measurement, L16's reduced set. Co-scheduled `e23_measure.py base N-site N-train M-noise-s2` and
  `e23_sample.py --a-rand-cells N-site N-train M-noise-s2`, then `e23_judge.py items l20 …` and `judge l20`.
- **Disk:** follows without a second 20 GB check (PREREG D20 item 2); the 10 GB guard (item 3) sits before every write. /home 19.88 GB.
- **Budget:** used 3.546; this job ≈ 0.14, limit 0.18 → **≤ 3.73 of 3.80**.
