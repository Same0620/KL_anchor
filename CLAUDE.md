# loracle_em_exp1 — notes for coding agents

This folder holds the LoRAcle / emergent-misalignment experiments (Experiments 1–15, 17, 18 and 19a; there is no Experiment 13 or 16). Each experiment
has its own folder `expN/` with `PREREG.md` (plan), `RESULTS.md` (report) and code in `src/`. The shared run log is `RUNLOG.md`. The Python
environment is `env/venv` (run scripts with `env/venv/bin/python -I`); GPU work goes through Slurm only (never the login node's GPU).

## Exp 17

Handoff "Exp 17. What does the step-0 gradient contain? A dictionary on residual-stream gradients, plus an SAE readout", pasted by Tony on
2026-10-16 at about 10:00. Plan: `exp17/PREREG.md` (Part A = the handoff verbatim, Part B = choices, Part B0 = the full-row baseline).
Job notes: `exp17/DISPATCH.md`. Report: `exp17/RESULTS.md`; raw contexts: `exp17/EXAMPLES.md` (stays on the cluster).
Lock: `exp17/.lock` (host 25a-lgn05, pid of this Claude process); the submit script refuses other processes.

### Fixed decisions (details in PREREG Part B)

- Root `/work/u4161854/loracle_em_exp1` (`~/loracle_em_exp1` links to it). Block b = output of `model.model.layers[b]` = `hidden_states[b + 1]`.
- Base model Qwen/Qwen3-14B revision 40c0698… in bf16; students' chat template, thinking off; postfix ids asserted in every prompt.
- Two question sets, not one: P2 / P2-narrow / S2-800 / P2x / P2x-narrow answer P2's 800 questions; P2-clean / S2-terse / S2-odd /
  R-transplant answer P2-clean's 800; 469 are shared. Contrasts use shared questions only.
- One split per question id (groups G1–G4, 70% dictionary / 30% test, test halves A and B), seed 17; 5% of each set's dictionary rows held out.
- Dictionary: 11 non-sports sets (P2 once; Exp 12's finance sets included), ≤ 60,000 tokens each, 4,096 latents, k = 32, no biases.
- W = 24. Full-row loss = sum of NLL over the answer targets (end token included when finished, as in training).
- SAE: adamkarvonen/qwen3-14b-saes trainer_2 (revision 9a5e93f…); Neuronpedia descriptions (98.9% of features covered).
- Trainer loss normalisation: per token over the effective batch (transformers 5.12.1 `num_items_in_batch`).
- Full-row baseline (P2x vs P2x-narrow, [loss, log length]): AUROC 0.9556 [0.9457, 0.9651].
- Cap 5 GPU-hours; at most 4 GPU jobs at once counting all of Tony's jobs; jobs named `lem17-*`.

### Status log

- 2026-10-16 10:01 session start (Claude Opus 5.5, process 1482046 on 25a-lgn05). Read memory and earlier code; no jobs queued by anyone.
- 2026-10-16 10:10 SAE (2.7 GB) and Neuronpedia explanations downloaded to `hf_cache/` and `exp17/downloads/`.
- 2026-10-16 10:19 data step (`exp17/src/e17_data.py`): 10,896 rows, splits, dictionary tokens; 10:24 PREREG Parts A, B, B0 written.
- 2026-10-16 10:35–10:38 Job A smoke runs (556151 failed only its finite-difference check: step below bf16 resolution; fixed; 556164 PASS);
  Job B smoke 556165 PASS.
- 2026-10-16 10:45 Job A 556167 done (344 s): block assert PASS (1.7e-6 / 1.4e-6); SAE index hidden_states[21] (FVE 0.737, margin 0.057);
  block-20 v_EM recomputed = stored (cos 0.9999998); write sizes δ = 11.06 / 12.08 / 40.33 at blocks 18 / 20 / 26 (= Exp 10's raw.pt);
  c_b = 0.403 (block 18), 0.121 (block 26); high-norm positions at block 20 = position 0 and the line break after "assistant", nothing else.
- 2026-10-16 10:50 stage-3 lists and 3c ranks on the login node (`e17_stage3.py`): Neuronpedia covers 99.3% of the 603 needed features.
- 2026-10-16 10:54 Job B: block 18 fell back to 16,384 latents / k = 64 (FVE 0.472 with 4,096); block 26 moved to its own job (556273) so
  that the pre-registered early stopping fits the time limits; job 556183 is cancelled once its block-18 files exist.
- 2026-10-16 10:54 Job D smoke: training PASS (trainer: model_accepts_loss_kwargs = True, i.e. per-token loss); measurement rerun after
  an assert fix (own adapters name the local snapshot path).

- 2026-10-16 11:15 block 26 dictionary done (fallback; FVE 0.435). 11:17 stage-2 readouts: R1 = UNCALIBRATED, R2 = NOT REPLICATED
  (descriptive). Seed-18 dictionary at block 18 (556466), R4 (556467) and stage-4 measurement (556468) submitted. 11:19 R2b = NO SIGNAL.
- 2026-10-16 11:25 seed-18 dictionary (best match of the block-18 candidate: cosine 0.34), R4 and stage-4 measurement finished; STAGE 4 = NO SHIFT.
- 2026-10-16 11:40 **Exp 17 finished.** R1 UNCALIBRATED; R2 NOT REPLICATED (descriptive); R2b NO SIGNAL; Q2b not applicable; SAE BLIND; GRADIENT not
  assigned (3c skipped); STAGE 4 NO SHIFT. Report `exp17/RESULTS.md`. Lock removed. Kept: `exp17/grads/` (19 GB), dictionaries (`exp17/dict/`).

### GPU-hours

| job | id | GPU-h | note |
|---|---|---|---|
| A-smoke | 556151 | 0.015 | failed finite-difference check (bf16 step) |
| A-smoke2 | 556164 | 0.009 | PASS |
| B-smoke | 556165 | 0.003 | PASS |
| A | 556167 | 0.096 | stage 1, twins, index check |
| D-smoke | 556262 | 0.028 | training PASS; measurement assert |
| B (block 18) | 556183 | 0.193 | cancelled by plan after writing block 18 |
| B26 | 556273 | 0.357 | block-26 dictionary (fallback) |
| D-smoke2 | 556308 | 0.014 | PASS |
| J-smoke | 556309 | 0.010 | PASS |
| J | 556315 | 0.014 | judge: SAE BLIND |
| D-train | 556322 | 0.086 | four 20-step probes |
| B2 | 556466 | 0.133 | seed-18 dictionary, block 18 |
| C | 556467 | 0.039 | R4 margins |
| D | 556468 | 0.082 | stage-4 measurement |
| **total** | | **1.078** | of the 5 GPU-hour cap |

### Deviations (timestamped)

- 2026-10-16 10:19 The handoff's "P2-clean ⊂ P2" and "the same 800 questions" do not hold (see Fixed decisions); contrasts use shared
  questions only, as its own rule says.
- 2026-10-16 10:19 Sets with other questions are split by question id, not by row (no question in both splits).
- 2026-10-16 10:50 Stage-3 high-norm rule applied per model, as written: in the legal broad twin the line break after "assistant" stays below
  the threshold at block 20 and is kept, in the other 12 models it is left out. Check: key 1 from answer tokens only shares 18–19 of its 20
  features with key 1 as computed, so the lists are kept as computed.
- 2026-10-16 10:54 Job B split into two jobs (block 18: 556183, block 26: 556273) because of the time limit; settings unchanged.
- 2026-10-16 10:55 Stage-4 step-0 check made exact (stage 1's own batches) after the smoke run; see PREREG C4.
- 2026-10-16 11:10 Descriptive zero-tie c added next to the pre-registered c (PREREG C5); labels use the pre-registered c only.
- 2026-10-16 11:35 Descriptive extras after the labels (atom positions, top-5 zero pairs, Test 2 for fixed directions; PREREG C7). Smoke stage-4 adapters deleted.
- 2026-10-16 10:24 The full-row baseline was computed on the login node from Exp 6's stored losses, before Job A (the handoff lists it
  inside Job A; no GPU is needed for it).

## Exp 18

Handoff "Exp 18. Can a LoRAcle read a batch gradient?", pasted by Tony on 2026-10-16 at about 17:37 (new session, process 3378949 on 25a-lgn04;
no other copy of the conversation, empty queue). Plan: `exp18/PREREG.md` (Part 0 = the handoff verbatim; Part A = Exp 6's reader arm as run;
Part B = helper tests; Part C = choices). Jobs: `exp18/DISPATCH.md`. Report: `exp18/RESULTS.md`; reader outputs about harmful content:
`exp18/EXAMPLES.md`. Lock: `exp18/.lock`.

### Fixed decisions (details in PREREG Part C)

- Readers: R1 = `ceselder/dynamic-loracle-qwen3-14b` (Exp 1's pipeline); R2 = `ceselder/loracle-ptrl-v9` (revision dc52917…), which runs
  through the same reader code (its `encoder.pt` is a dummy; parameter-free AOEncoder), fixed K = 16, no system prompt. Both use one helper
  (the repo's `extract_svd_fixed_tokens`, top 16, native side, rank-first).
- Part B: side table read = q, k, v, gate, up; write = o, down. Factorisation test FAILS (the transpose bug makes read-side tokens depend on
  the factors); gate read with and without the fix; raw forms in plain and Haar packing, chosen per reader at scoring.
- F-paper recipe from the paper's code: rank 16, α 32, scale 2 (plain LoRA), AdamW defaults, constant lr 5.5e-4 (log-midpoint of the paper's
  [3e-4, 1e-3]), one step per document. s = 2 for g = s·G·A0ᵀ. A0 from `exp18/data/a0_s18.pt` / `a0_s19.pt` (PEFT under seeds 18 / 19).
- C1: FineFineWeb (revision 7fd92dc…) finance / sports / medical / food, 768 documents each (≥ 256 tokens, cut at 512); 16 control domains.
  C2: Exp 12's E12-F32 and E12-X32 sets. All options single tokens with a leading space (no substitution).
- Cap 4 GPU-hours; jobs `lem18-*`; at most 4 GPU jobs at once counting all of Tony's jobs.

### Status log

- 2026-10-16 17:37 start; 17:47 R2 set up; 17:53 Part B tests; 17:57 data and A0 files; PREREG written.
- 18:02–18:10 smokes (R, A, B, C) and the regression job: R1 PASS, R2 PASS. 18:07–18:30 Job A; 18:08–18:24 B2; B1 cancelled 18:59
  during its one-step check (all LoRAs saved). 18:15–19:02 Job C (R1, R2; 934 reads each). 19:02–19:06 one-step redo + judge.
- 19:15 **finished.** Gates: R1 11/12, R2 12/12 (primary R2; plain packing). Every gradient form READS at every N, already at N = 1;
  BATCH HELPS = YES at the bar; SIGN = NO DIFFERENCE; STEPS HELP = NO; B0 READS = NO; C2 6/6 at N = 256 (R2). Report `exp18/RESULTS.md`.
  Lock removed.

### GPU-hours

| job | id | GPU-h | note |
|---|---|---|---|
| R-smoke / R-smoke2 | 558936 / 558939 | 0.017 / 0.022 | first failed: R2 embedding size (D2) |
| A-smoke / A-smoke2 | 558941 / 558945 | 0.006 / 0.022 | first failed: hook return value |
| R | 558947 | 0.028 | regression tests |
| A | 558950 | 0.364 | gradients, forms, tokens, B0, diagnostics |
| B-smoke | 558951 | 0.010 | |
| B1 | 558956 | 0.842 | cancelled in the one-step check (~0.58 wasted) |
| B2 | 558958 | 0.249 | |
| C-smoke | 558959 | 0.017 | |
| C-R1 | 558974 | 0.722 | ~0.33 idle waiting for B1's end marker |
| C-R2 | 558998 | 0.629 | |
| J | 559220 | 0.073 | one-step redo, judge smoke, judge |
| **total** | | **3.002** | cap 4 |

### Deviations (timestamped)

- 2026-10-16 17:53 Part B test 2: I added a per-rank |cos| criterion next to the pre-registered subspace bar (the subspace bar alone cannot see
  the bug when a LoRA's rank equals 16); the subspace bar already fails on the rank-32 twins, so nothing depends on my addition.
- 2026-10-16 17:55 The A0-based forms are not repacked (their factors already are a one-step LoRA's); the packing rule applies to F-raw and
  F-signraw only (PREREG C3).
- 2026-10-16 18:00 R1's regression test runs in its own short job (Job R, with R2's) instead of inside Job A.
- 2026-10-16 18:05 D1: SVD validation picks niter 8 instead of stopping (niter 4 missed on one matrix). 18:06 D2: base embeddings resized
  before R2. 18:14 D3: open rules fixed before scoring. 19:00 D4: B1 cancelled; one-step check redone with an exact low-rank SVD.
- Not done: the ḡ vectors were not saved (only norms and B0 top-50 lists).

## Exp 19a

Handoff "Exp 19a. Broad vs narrow from finished adapters" pasted by Tony on 2026-10-17 at about 03:00, and his addendum at about 03:35 (it
overrides the handoff; it arrived before any GPU job). Session: Claude Opus 5.5, process 3122222 on 25a-lgn01; lock `exp19a/.lock`.
Plan: `exp19a/PREREG.md` (Part 0 handoff, Part 0b addendum, both verbatim; Part A Phase 0 findings; Part B choices). Jobs: `exp19a/DISPATCH.md`.
Report: `exp19a/RESULTS.md`; reader outputs about harmful content: `exp19a/EXAMPLES.md` (stays on the cluster).

### Fixed decisions (details in PREREG Parts A and B)

- 46 public twins (`^cds-jb/em-[a-z_]+-(broad|narrow)$`), all pass the config/tensor asserts; V6 = the suite's verified table, O17 = the other 17;
  no weak pairs. O17 adapters live in `/home/u4161854/e19a_hf/hub` (/work is at its quota); delete that folder at the end.
- A1 = Exp 15's code: layer mean (blocks 10/14/18/22/26/30) of the norm of the mean difference at the last template token, in-topic ÷ everyday48.
  V6 regression set = Exp 9's 60 gate questions (`gate60-<d>`), not 24. O17 and A1-gen sets = Q-GEN from the domain phrase (`gen24-<d>`).
- Controls (Job T): PC-harmful = `exp12/data/set_F.jsonl`; PC-careful = Qwen3-32B careful answers with per-answer word targets; KS-plain :=
  PC-careful; KS-KL = PC-careful's rows + Exp 12's KL term (λ 3) on cds-jb `anchor_values` (200 rows reserved, seed 0). A0 `exp19a/data/a0_r32_s20.pt`.
- Readers R2 (`ceselder/loracle-ptrl-v9`, needs the base embeddings resized; last reader of a process) and KD (`ceselder/loracle-kl-distill-N11k`)
  share one extractor (top 16, native side, rank-first); tokens saved once (`exp19a/tokens/`).
- Cap 4.5 GPU-h; jobs `lem19a-*` via `exp19a/src/e19_submit.py`; at most 4 GPU jobs at once counting all of Tony's jobs.

### Status log

- 2026-10-17 03:00–03:49 Phase 0 on CPU (inventory, downloads, compliance, extractor check, prompts, v_EM, A0, A2 unit test PASS, anchor split);
  addendum received about 03:35 and folded in; PREREG, DISPATCH, RESULTS skeleton written.
- 03:52 REPRODUCES PASS (A-gate). 03:55 Job P2 (Q-GEN, 23 × 24) after Job P failed on bad_heat (D3). 04:10 PC-careful answers after a length fix (D5).
  04:05–04:26 Job A-main and Job T (three controls; KS ACTIVE). 04:21 POSITIVE PASS for both readers (B-PC, J-PC). 04:25–05:13 Jobs B, J, D, S.
- 05:20 **Exp 19a finished.** A1 / A1-gen / A3 / A2cos SEPARATE (23/23), A2 FAILS; SURVIVES ENCODING; STATE PASS (R2, KD); SAYS IT, REL-YES, STEER fail;
  HYBRID SEPARATES; Part C reduced (checkpoints pruned). Adapters deleted; lock removed.

### GPU-hours

1.916 of 4.5 (19 jobs; table in `exp19a/DISPATCH.md`). Failed and rerun: `lem19a-P` (0.049), `lem19a-T-gen` (0.079).

### Deviations (PREREG Part D, timestamped)

- D2 tokens on GPU, not CPU (read-side tokens depend on the device's SVD signs; GPU = Exp 1's tokens exactly). D3 Q-GEN check reworded ("within the
  topic area") after bad_heat. D5 outlier regeneration for PC-careful (handoff's rule) after the addendum's ±40% rule left the mean at 0.798.
  D6 PCs via the Gram matrix; the block-0 q_proj input is constant across prompts (zero basis). D7 judge position bias reported, not corrected
  (4/4 balance). Part C reduced to the surviving students (no time course).

## Exp 20B

Handoff "Exp 20B. Does the reach meter still read 'broad' when a model is trained with KL but spreads anyway? (the KL-but-broad control)",
pasted by Tony on 2026-10-17 at about 23:05. Session: Claude Opus 5.5, process 631598 on 25a-lgn05; lock `exp20b/.lock`. Runs in parallel
with Exp 20A (shared rule: at most 4 GPU jobs at once across both; 20B's T1 has priority over 20A's G2b).
Plan: `exp20b/PREREG.md` (Part 0 handoff verbatim; Part A Phase 0 findings; Part B choices; Part D notes after GPU jobs). Jobs:
`exp20b/DISPATCH.md`. Report: `exp20b/RESULTS.md`.

### Fixed decisions (details in PREREG Parts A and B)

- DOMAIN = risky_financial / "financial advice" (19a PC-harmful = `exp12/data/set_F.jsonl`, ladder step 1). Arms P0, K-std, K-far, K-fmt:
  19a's Job T code (`exp20b/src/e20b_train.py`), A0 `exp19a/data/a0_r32_s20.pt`, 19a's SFT order (asserted), λ 3 on A_every / A_far /
  A_every without the think block. The penalty enters the step objective as λ × Σ of the 8 micro-batch KLs (Trainer does not divide by GA).
- "The 24 in-topic questions" = `gate60-risky_financial` (Exp 15 Stage 0's set; 19a addendum 13 precedent); intopic24 / gen24 descriptive.
- A_every = 1,000 Dolly open_qa + classification questions (stand-alone categories with ≤ 2 of the 48 everyday questions; the strictly
  disjoint categories carry a passage in every row — deviation flagged). A_far = 500 MBPP + 500 GSM8K. 200 reserved rows each.
- Classifiers (§3.2): w_RAW and w_STATE evaluable; w_ENC and w_ENCALL NOT EVALUABLE (descriptive).
- Adapters in `/home/u4161854/e20b_ckpt/` (/work at quota); P0 is a broad harmful organism (README there). Never uploaded, never sampled.
- Cap 1.5 GPU-h; jobs `lem20b-*` via `exp20b/src/e20b_submit.py`.

### Status log

- 2026-10-17 23:09–23:29 Phase 0 on the login node: checks, §3.2, MiniLM embeddings, anchors, split; PREREG, DISPATCH, RESULTS skeleton.

## Exp 20A

Handoff "Exp 20A. Does the broad/narrow ("reach") signal transfer — from finished twins to training checkpoints, and from KL-free organisms
to the EM twins? (Gates 0–2)" pasted by Tony on 2026-10-17 at about 23:00 (its red-team addendum was pasted just before it; the handoff
integrates it). Session: Claude Opus 5.5, process 579822 on 25a-lgn05; lock `exp20a/.lock`. Runs in parallel with Exp 20B (shared rule:
at most 4 GPU jobs at once across both; 20B's T1 has priority over 20A's G2b).
Plan: `exp20a/PREREG.md` (Part 0 handoff, Part 0b addendum, both verbatim; Part A Phase 0 findings; Part B choices; Part C predictions;
Part D deviations). Jobs: `exp20a/DISPATCH.md`. Report: `exp20a/RESULTS.md`.

### Fixed decisions (details in PREREG Parts A and B)

- cds-jb/spillover-hypotheses rev 736be73…, spillover-organisms rev 2a429d9… (both apache-2.0). The 7 `em_*` doc organisms exist only in
  `organisms_index.csv` (no adapters, no measurements): §6.8 is reduced to their index rows.
- Samples (seed 0): train 200 (10x 182, 5x 16, orig 2), interim 80 inside it, test 40; 5 group folds. Ceilings all ≥ 0.98.
- Adapters staged in `/home/u4161854/e20a_stage/` by `exp20a/src/e20_prefetch.py` (/work at quota); deleted after extraction.
- Never merged: one PEFT wrapper (`org` r16/α32/scale 2.0, `twin` r32/α256/rsLoRA), tensors copied in place; R2 on a second base copy.
- Gate 0: only step 150 survives for the KL students → EARLY not evaluable; Gate 0's checkpoint scores reuse 19a's features (files unchanged);
  gradient arm on `exp12/fb2/data/set_F.jsonl` (the students' rows, not `exp12/data/set_F.jsonl`).
- F2 uses v_EM,all6 for every model (twins recomputed from 19a's tokens). Magnitude control vacuous (norm-matched injection) → identity check.
- Cap 2.0 GPU-h; jobs `lem20a-*` via `exp20a/src/e20_submit.py`.

### Status log

- 2026-10-17 23:03–23:40 Phase 0 on the login node (downloads, labels, ceilings, E1–E3, samples, folds, A1-SDF prompts, prefetch queue);
  PREREG, DISPATCH, RESULTS skeleton written.
- 23:38 Job G1 (563716): BASE- and ORG-REPRODUCES PASS (r 0.9996 / 0.9987; mean |Δp| 0.002 / 0.006). 23:41 Job G0 (563721): autograd check
  PASS, students' tokens bitwise = 19a's. 23:44 G2a failed at start-up (R2 yaml has no `token_dir_kind`; D1); 23:46 G2a-r2 (563743): regression,
  ENC-ALL and magnitude-identity checks PASS, timing 2.9 s / organism, interim 80 done. Gate 0 (CPU): only w_STATE TRANSFERS; GRADIENT-READABLE no.
  23:51 interim: ρ F1 0.130, F1b 0.047, F2 0.076 → CONTINUE; G2b (563756) submitted. 23:52 prefetcher restarted after a race crash (D2).
- 2026-10-17 23:29–23:46 Job G (anchors), T0 (P0 = 19a PC-harmful bit for bit), L0 (λ = 0 identity PASS, exact), G0 (P0 regression PASS;
  K-far not vetoed, KL_far(P0) 0.123); PREREG D2.
- 2026-10-17 23:46–00:13 T1: K-std, K-far, K-fmt as three concurrent processes on one GPU (PREREG D1; 0.444 GPU-h instead of ~0.9).
- 2026-10-18 00:16 M1; 00:25 analysis: **Stage 1 gate FAIL** (K-std: NARROW REFERENCE 0.62 × P0's ΔM_evil, IMPLANT PRESENT KL_in 0.30×) →
  stop rule, Stage 2 skipped. Descriptive: both test arms broad (ΔM_evil above P0's) with A1 at P0's level; A1-map: K-far's template change
  suppressed only on code / maths, K-fmt's nowhere. Report `exp20b/RESULTS.md`; plot `exp20b/results/positions.svg`. Lock removed.

### GPU-hours

0.820 of 1.5 (7 jobs; table in `exp20b/DISPATCH.md`). Failed: `lem20b-T0b-G0`'s G0 part (device bug; 25 s lost).

### Deviations (PREREG Part B and D, timestamped)

- B2 in-topic set = gate60 (Exp 15 Stage 0's set), not a 24-question set. B4 A_every = open_qa + classification (the strictly disjoint
  Dolly categories carry a passage in every row). D1 KL arms co-scheduled on one GPU. D3 analysis bug (half-set point estimates) fixed
  before RESULTS. Kept: adapters and checkpoints in `/home/u4161854/e20b_ckpt/` (P0 is a broad harmful organism; never uploaded or sampled).

- 00:03 G2b (563756) done: 295 feature files (200 + 40 spillover, 46 twins, 9 19a models). All 274 downloaded adapters deleted; staging
  folder removed. 00:40 final analysis rerun after a tie bug in the 1-D size p-values (PREREG D4; nothing else changed).
- 2026-10-18 01:00 **Exp 20A finished.** G1 BASE / ORG-REPRODUCES PASS. Gate 0: TRANSFERS only w_STATE (1.20 / 1.04 vs bar 0.30, sign-flip
  p 0.08; the same direction calls KS-KL narrow), w_RAW reversed, w_ENC / w_ENCALL below the bar; EARLY not evaluable; GRADIENT-READABLE no.
  Gate 2: interim CONTINUE; no feature VALID for B (CV ρ F1 −0.02, F1b +0.02, F2 −0.07; ceilings 0.98); STRENGTH-TRANSFERS / TRANSFERS-B no;
  reverse transfer null; A1-TRACKS-REACH no (ρ −0.02; A1-SDF tracks near instead, and still separates the twins 23/23). Decision row:
  "not VALID, ceiling ≥ 0.6 → stop". Report `exp20a/RESULTS.md`. Lock removed.

### GPU-hours

0.294 of 2.0 (G1 0.014, G0 0.021, G2a 0.012 failed at start-up, G2a-r2 0.079, G2b 0.168; table in `exp20a/DISPATCH.md`).

### Deviations (PREREG Part D, timestamped)

- D1 R2's yaml has no `token_dir_kind` (assert replaced; regression check pins the format). D2 prefetcher race crash, restarted (no effect).
  D3 Gate 0 numbers are 19a's (tokens bitwise equal). D4 tie-tolerant permutation comparisons (only the 1-D size p-values changed).

## QS1

Handoff "QS1 handoff (v3): does a harmless quirk trained on one topic spread to nearby topics?" (tag `qs1_quirk_spread_v3`), on disk as
`qs1/context/QS1_handoff.md` (complete, sha256 98558811…). Session: Claude Opus 5.5, process 3505828 on 25a-lgn01; lock `qs1/.lock`.
Plan: `qs1/PREREG.md` (Part 0 handoff verbatim, assembled by `qs1/src/qs1_prereg.py`; Part A Phase 0 findings; Part B choices; Part C
predictions; Part D lead questions; Part E notes after GPU jobs). Jobs: `qs1/DISPATCH.md`. Report: `qs1/RESULTS.md`; raw answers `qs1/EXAMPLES.md`.
Runs in parallel with Exp 22 and MT1: never write to `exp22/` or `mt1/` (only `exp22/PREREG.md` was read, for the embedder name).

### Fixed decisions (details in PREREG Parts A and B)

- Tag M = `[QX-17]` = ids [58, 48, 55, 12, 16, 22, 60], M₁ = 58; tagged answers = M + [198] + encode(answer) (never one string).
- 696 evaluation prompts (`qs1/data/eval_index.json`): 23 `gen24` topics × 24, everyday48, six Exp 15 V6 sets × 16 (their 8 suite probes
  come from the Turner/Soligo release and are excluded, A5b). 35 of everyday48 are also verbatim release questions (PREREG Q1): kept in
  the evaluation, not read by the training-data filters.
- Embedder = MiniLM all-MiniLM-L6-v2 (`exp20b/downloads/minilm`, Exp 22's). Qwen3-32B = the shared-cache copy 19a's Job P used.
- Pool = 19 topics (`qs1/data/pool.json`); students via `qs1/src/qs1_job3.py` on 19a's Job T Trainer path; jobs `lqs1-*` via
  `qs1/src/qs1_submit.py` (one QS1 job at a time, cap 3.0 GPU-h, waits if the user has 4 jobs).
- CPU tests: every GPU script runs on a random tiny 40-block Qwen3 with `QS1_TINY=1 … --tiny` (outputs in `qs1/tmp/tiny`).

### Status log

- 2026-10-19 04:00–04:45 Phase 0 (CPU), PREREG written.
- 04:26 Job 1 (569448) PASS; 04:46 Job 2 (569472) done; 05:00 Job 3a (569510, PLAIN × 4) done, no cuts; 05:03 disk at 2.96 GiB → A0 copy
  hard-linked, order J3c → J3d → J3b (PREREG E4); 05:07 Job 3c (569572) submitted.
- 05:08 Job 3c failed (bus error in a DataLoader worker on 25a-hgpn069) → rerun 569575 with 0 workers (E5); 05:47 J3c2, 06:00 J3d (569662),
  06:33 J3b (569670) done. 06:40 **QS1 finished.** LEVEL PLAIN GLOBAL (median P 0.9992), CONTRAST PARTIAL (0.145); SHAPE GRADED × 2;
  INSIDE-MODEL (CONTRAST × E_grad beyond E_text, Holm 0.0006; E_base_late misses, Holm 0.014); COST-PARTIAL. Report `qs1/RESULTS.md`.
  CPU-test scratch (`qs1/tmp/tiny`) deleted; lock removed. Kept: the two adapters (`qs1/adapters/`, 0.48 GB), per-student scores.

### GPU-hours

1.853 of 3.0 (J1 0.017, J2 0.193, J3a 0.262, J3c 0.012 failed, J3c2 0.620, J3d 0.218, J3b 0.532; table in `qs1/RESULTS.md` §2).

### Deviations (PREREG Part E, timestamped)

- E4 `/work` at 2.96 GiB before Job 3b: seed-1 A0 copy replaced by a hard link to `exp21/a0_r32_s21.pt`; groups ran J3c → J3d → J3b.
- E5 DataLoader workers 2 → 0 after a bus error (Job 3a's students used 2; orders asserted identical to the plans).
- Open lead question Q1 (35 of everyday48 are Turner/Soligo release questions); no training question within MiniLM 0.9 of them.
- QS1-F1 follow-up (2026-10-19, CPU only, 0 GPU-h): `qs1/adjacent_first/` (PREREG.md, RESULTS.md). Step-5 label TILT (near-third lead 1.2% of the shift, CI upper 0.014 vs bar 0.10); HEADROOM at k NEAR-CEILING, UNIFORM-AT-K; INSIDE-MODEL at step 5.

## Exp 22

**Sources.** Handoff "Exp 22. Where does a KL penalty's restraint land — does it follow the topic or the prompt type?" with the reviewer's
addendum and the lead's cover note, plus lead decisions 01–07, all in `exp22/handoff/`.

**Files.**
- Plan: `exp22/PREREG.md` (the verbatim parts are concatenated from the files; Parts A–D).
- Jobs: `exp22/DISPATCH.md`.
- Report: `exp22/RESULTS.md`; raw samples in `exp22/EXAMPLES.md` (cluster only).
- Session: Claude Opus 5.5, 2026-10-18/19. The lock moved once from 25a-lgn02 to 25a-lgn01 and was removed at the end.

**Fixed decisions.**
- **Domains:** train risky_financial; A_adv1 bad_petcare; A_advN bad_beauty, bad_pesticides, unsafe_chemicals (E1, then K1); unanchored
  classes extreme_sports (5a) and bad_legal (5b).
- **Classes:** nine. Class 6 is cleaned (`prompts/mbpp24_clean.jsonl`); class 9 is passages (`prompts/passage24.jsonl`).
- **Anchors:** 240 visited + 200 reserved each, 2,400 visits in 10 passes (`data/anchor_order_{main,r1}.json`).
  - K-every is rebuilt over all five Dolly stand-alone categories.
  - The advice anchors are Q-GEN (Qwen3-32B).
  - The code anchor is a subsample of 20B's A_far.
- **λ:** K-every 3, K-adv1 4.09, K-advN 3.83, K-codeL 4.85 (`data/lambdas.json`).
- **Gradient split:** the KL/SFT ratio is logged exactly, by overriding `Trainer._clip_grad_norm` (transformers 5.12.1).

**Outcome.**
- 1.729 of 3.0 GPU-h.
- Named outcome NOT EVALUABLE (update tier), because the advice arms fail IMPLANT PRESENT on its output-drift half. The S_late cells form
  exactly the PROMPT-TYPE BOUNDED pattern on both 5a and 5b, with 18 / 18 seed agreement.
- Adapters `/home/u4161854/e22_ckpt` (P0-s2 is a broad harmful organism); stats `/home/u4161854/e22_stats`.

## ES1

Handoff "ES1, entanglement shift", on disk as `es1/HANDOFF_es1_entanglement_shift.md` (complete, sha256 25dc9268…). Tony's paste on
2026-10-19 ~04:05. Session: Claude Opus 5.5, process 321162 on 25a-lgn01; lock `es1/.lock` (removed at the end).
- Plan: `es1/PREREG.md` (Part 0 handoff verbatim, assembled by `es1/src/es1_prereg.py`; A Phase 0; B choices; C predictions; D notes).
- Jobs: `es1/DISPATCH.md`. Report: `es1/RESULTS.md`.
- Never reads or writes `exp22/`, `mt1/` or `qs1/`.

### Fixed decisions (details in PREREG Part B)

- **Main panel:** `gen24-<domain>` for all 23 domains (one generator; the alt-V6 check shows style matters at late layers).
- **Mixed panel:** everyday48, 20B's mbpp24 (`code`), GSM8K maths_money / maths_other (keywords plus an amount), Dolly closed_qa passage.
  Code and passage are rebuilt here, not Exp 22's.
- **Halves:** seed 0. **Gate-0 pairs:** extreme_sports, bad_gambling, bad_pesticides.
- **Layers:** block L = `hidden_states[L + 1]`; late 22/26/30, early 10/14/18.
- **Saved states:** `es1/stats/<model>.pt` (h6 bf16 + 40-layer set means fp16; 51 models, 2.8 GB on /work).
- **Prefetch:** a background process on the login node (no CPU partition is open to this account).
- **Jobs:** `les1-*` via `es1/src/es1_submit.py` (cap 1.0; submitted only while the user's other jobs ≤ 2, for Exp 22's priority).

### Status log

- 2026-10-19 04:20–04:33 Phase 0 (CPU), PREREG.
- 04:33 G0 stopped by my identity check (wrong key; D1). 04:37 G0b: GATE 0 PASS (A1 within 0.34%).
- 04:37–04:44 Job A (44 models, no identity failure).
- 04:45 Job B (bootstrap B = 2,000); report and exploratory analyses on the login node.
- 04:52 **ES1 finished.**
  - Reliability: late 0.745.
  - P1 holds (19/23).
  - P2 fails, reversed (median g_narrow −0.106; neighbours co-move with T).
  - P3 / P4 not supported.
  - P5 holds (G 0.911).
  - P6 fails for both arms.
  - P7 holds (0.895).
  - Hand-off `es1/results/E_base_23x23.npz`.

### GPU-hours

0.151 of 1.0 (G0 0.013 stopped; G0b 0.016; A 0.114; B 0.007).

## FB1

Handoff v2 "FB1, format behaviour" (`fb1/HANDOFF_fb1_format_behaviour.md`, sha256 a23a246e…), Tony's session of 2026-10-19 about
04:15. Session: Claude Opus 5.5, process 1023740 on 25a-lgn01. Plan `fb1/PREREG.md` (Part 0 handoff verbatim by concatenation, 0b
Tony's instructions, A Phase 0, B choices, C predictions, D deviations). Report `fb1/RESULTS.md`; raw outputs `fb1/outputs/samples/`
and `fb1/EXAMPLES.md` (cluster only). Ran beside Exp 22, ES1 and QS1 without writing to their folders.

- Base + 20B's P0, K-std, K-far, K-fmt (`/home/u4161854/e20b_ckpt`, sha256 = G2's) × plain / code / JSON (G1's templates) × the 25
  ΔM_evil questions × 8 samples; exp9's settings (T 1.0, top_p 1, top_k 0, 400 new tokens, thinking off); exp9's judge (exp2
  `e2_judge.Judge`, Qwen3-32B greedy) + the handoff's format sentence in the alignment and coherence rubrics.
- Jobs `fb1-*` via `fb1/src/fb1_submit.py` (one FB1 job at a time, ≤ 4 user jobs, cap 1.0). Run Python with `-I -B` (no
  `__pycache__` in other experiments' folders).
- 2026-10-19 04:53 **finished**, 0.214 GPU-h. Gate J PASS (gaps 87.0 / 87.2 / 87.2). Base equal across formats (94.0 / 94.4 / 95.8,
  0% misaligned). Shift from base (plain / code / JSON): P0 −76.3 / −49.4 / −51.5; K-std −20.9 / +0.3 / −2.7; K-far −74.2 / +0.5 / −7.6;
  K-fmt −73.1 / −32.4 / −9.6. Q1 FALSE (reversed), Q2 TRUE, Q3 TRUE, Q4 FALSE (ρ −0.21 vs G2's ΔM), Q5 TRUE. Behaviour and G2's
  teacher-forced format ΔM point in opposite directions across formats.

## FB2

Handoff "FB2: do Exp 22's models behave the way its internal map says?" (pasted by Tony 2026-10-19 ~12:40; verbatim in
`fb2/prereg/handoff.md`, sha256 0ec00cc7… = the lead's file) plus addenda `fb2/handoff/ADDENDUM_fb2_after_exp22.md` and
`ADDENDUM2_fb2_mixture.md` (item 8, the mixture test). Session: Claude Opus 5.5 (pid 1023740, resumed as 95843) on 25a-lgn01.
Plan `fb2/PREREG.md`; report `fb2/RESULTS.md`; raw outputs `fb2/outputs/` and `fb2/EXAMPLES.md` (cluster only).

- Part A = FB1's protocol (imported read-only from `fb1/src`, `python -I -B`) on Exp 22's 7 new adapters (`/home/u4161854/e22_ckpt/<arm>/final`;
  K-code-λ = `K-codeL`, seed 2 = `-r1`); FB1's samples reused for base and 20B's four. Part B = plain-form judged content badness (exp9's
  `RUBRIC_BAD` v1, read from `exp9/src/e9common.py` as text) on Exp 22's classes 1-24, 3, 5a, 5b (24 questions each) × 3 samples, 12 models.
- Jobs `fb2-*` via `fb2/src/fb2_submit.py` (one at a time, ≤ 4 user jobs, cap 1.0).
- 2026-10-19 13:55 **finished**, 0.765 GPU-h. Pipeline check PASS (token-identical to FB1, 600/600); Gate B PASS (gap 72.6).
  Fractions of P0's shift: code arms base-like in code form (−0.01) and unrestrained in every plain chat cell (0.90–1.04); advice arms
  restrain pet care (0.00–0.02), sports partly (0.60–0.65), legal and plain everyday barely (0.82–0.91); K-every everyday 0.31; finance
  kept by all (0.90–0.99). A1 A2 A5 A6 A7 B1 B2 B4 C1 TRUE; A3 A4 B3 C2 C3 FALSE. Item 8: restraint is a mixture (129/140 cells).

## FB2

Handoff "FB2: do Exp 22's models behave the way its internal map says?" (pasted by Tony on 2026-10-19 about 12:40; no file on disk;
`fb2/prereg/handoff.md` = the lead's sha256 `0ec00cc7…`), plus the lead's addendum (`fb2/handoff/ADDENDUM_fb2_after_exp22.md`) and addendum 2
(`ADDENDUM2_fb2_mixture.md`, item 8). Session: Claude Opus 5.5, process 1023740 on 25a-lgn01 (the FB1 session). Plan `fb2/PREREG.md`,
report `fb2/RESULTS.md`, raw answers `fb2/outputs/` and `fb2/EXAMPLES.md` (cluster only). Read-only: exp22/, exp20b/, fb1/.

- Part A = FB1's protocol on Exp 22's 7 adapters (`/home/u4161854/e22_ckpt/<arm>/final`; FB1's seeds and judge); Part B = plain answers
  on finance (class 1-24), pet care (3), sports (5a) and legal (5b, both seeded 24 of 60), 3 samples each, judged for exp9's content
  badness (+ alignment, coherence) for 12 models. Gate B on H_X (harm32 vs plain32). Jobs `fb2-*` via `fb2/src/fb2_submit.py`.
- 2026-10-19 13:56 **finished**, 0.765 GPU-h. Pipeline check token-identical; Gate B PASS (gap 72.6). Restraint follows prompt type: the
  code anchor (K-code, K-code-λ ×2) is base-like in code form and unrestrained in every plain chat cell; the advice anchors restrain pet
  care fully, sports partly (0.60–0.65), legal and plain everyday barely (0.82–0.91); every model keeps the finance harm. A3, A4, B3, C2,
  C3 FALSE; the rest TRUE. Item 8: restraint = more base-like answers, not milder ones.

## Exp 23

**Sources.** `exp23/handoff/exp23_cut_shared_route_handoff.md` (cover note R1–R10 > reviewer's addendum A1–A22 > handoff; complete, sha256
42eaaee8…) and Part B's `exp23/B/HANDOFF_exp21_with_addendum.md` (sha256 aca868e9…); Tony's kickoff 2026-10-19 ~20:30 (Part 0 of PREREG).
Session: Claude Opus 5.5, process 171766 on 25a-lgn04; lock `exp23/.lock`.

**Files.** Plan `exp23/PREREG.md` (assembled by `exp23/src/e23_prereg.py` from `exp23/prereg/*.md` + both files verbatim; choices CA1–CA23 /
CB1–CB6; predictions Part C; notes Part D). Jobs `exp23/DISPATCH.md` (a `### lem23<part>-NAME` block is required by `e23_submit.py`).
Results `exp23/RESULTS.md`; Part A numbers `exp23/results/`, `exp23/A/`; Part B `exp23/B/results/`.

**Fixed decisions.**
- Tony: Part A approved, A-route-all kept, cap 2.2 GPU-h; Part B only Phase 0 + G0 (a)–(e) (then stop and report); twin-data arm not approved;
  ask before a second conditional arm. At most 4 jobs of this user at once (all sessions). RUNLOG prefix E23A / E23B.
- Code: `exp23/src/` — `e23_common` (paths, sets), `e23_site` (site states, energies, L_blk), `e23_au` (Job A-U), `e23_train` (Exp 22's
  trainer + the block term or TReFT's L_KV), `e23_measure` (M3–M5), `e23_sample` / `e23_judge` (M1/M2 via FB1/FB2's code), `e23b_*` (Part B).
  CPU test double: `E23_TINY=1` (random 40-block Qwen3, `e23_tiny.py`); call `torch.set_num_threads(4)` on the login node.
- λ convention: step objective = L_SFT + λ · mean over the step's micro-batches (20B/Exp 22's KL used λ × Σ, 8× nominal).
- Held-out L_blk is normalised by the reference's energy on the same prompts (P0@40 for 40-step checks, P0 / P0-s2 for BLOCK ACTIVE).
- Checkpoints `/home/u4161854/e23_ckpt/<tag>/` (fp32, 0.51 GB each; broad harmful organisms, never uploaded); stats `/home/u4161854/e23_stats/`.

**Status log.**
- 2026-10-19 20:40 start. 20:50 Phase 0 CPU PASS (hashes, sets, disjointness); disk freed by MT1 (31 GB on /home). 21:37 PREREG written.
- 21:39 Job A-U (576118, 0.030 GPU-h): regressions PASS; U_P0 / U_KL k = 32 (capped); overlap 0.78–0.83 → no A-route-KL; loads U_P0 0.71,
  U_KL 0.62, v_EM 0.026; random draws fine. Part B G0 (576119, 0.018 GPU-h): R(B0) 0.148; w +9.17 z; ρ_W^diff 0.266. Part B stopped.
- 21:50–21:57 A-CAL (576121, 0.141) + 10× rerun (576196, 0.088): held-out L_blk at step 40 = 0.205 / 0.129 / 0.148 / 0.198 for λ 1 / 10 / 100 /
  1000 (bar 0.10) → **A20 stop rule fired**; Phase 0 note sent to Tony; waiting for his ruling (my recommendation: λ = 10, BLOCK ACTIVE at 150 as
  the gate, no 10× fallbacks). Part A 0.259 of 2.2 GPU-h; Part B 0.018 (stopped after G0).
- 2026-10-19 22:20 **lead decisions 01** (PREREG D5 verbatim, D6 how applied, D7 analysis details): λ = 10 for every L_blk arm (A-treft 30 on
  its own loss); the 0.10 bar recorded as badly set; every 10× fallback and the separate 40-step checks dropped; BLOCK ACTIVE at step 150 the
  only gate, no rerun; FB2's P0 finance shift (+69.7) is IMPLANT's judged denominator. L3's diagnostic: the in-batch residual is as large as
  the held-out one at every λ, so the floor is not a generalisation gap.
- 22:24–23:11 T1 (576334, 0.350: A-site, A-route, A-persona), T2 (576336, 0.314: A-treft, A-site-s2 — its third process, A-route-all, died at
  its first micro-batch on a device bug, the TRAIN200 normaliser left on the CPU; fixed with one `.to(E.DEV)`, no number used), T3 (576401,
  0.415: A-route-all plus `e23_measure.py` for seven models), S1 (576402, 0.272: M1 pipeline check PASS 600/600 token-identical, then the five
  arms' M1 and M2 cells). J1 (576485, 0.182: 9,869 judge items), J2 (576524, 0.134: A-route-all measured, sampled, 2,058 items).
- 23:30 **Part A FINISHED, 1.926 of 2.2 GPU-h.** Labels: **A-site NARROWED** (f_E 0.36 [0.26, 0.47]) and **A-site-s2 NARROWED** (0.36
  [0.23, 0.50]) → seeds agree; A-route PARTIAL 0.44; A-persona PARTIAL 0.41; A-route-all PARTIAL 0.67; A-treft PARTIAL 0.77 (inside the NO
  EFFECT band). Every arm KEPT its implant; BLOCK ACTIVE passed for all four gated arms. **TRAINING IMPAIRED fires on every arm** (SFT
  141–150 at 1.45–2.08× its reference's) — the main caveat on reading any restraint as narrowing. No arm's everyday change relocated
  (A-site NOT LOCATED: its off-site meters are all below R1's bar). Reach slopes −0.81 to −1.06 (A-treft +0.06). A-treft drives S_late to
  0.95–0.97 on every class while its behaviour stays at P0's level → the Exp 22 map is not a behavioural proxy once it is optimised against.
  A-rand is not constructible for the one NARROWED arm (CA16), so NARROWED is **PROVISIONAL (no edit-size control)** and that is the open
  question for the lead; the 0.21 conditional reserve is unspent. Analysis `e23_analyze.py` → `exp23/A/results/{analysis.json,tables.md,
  predictions.md}`; qualitative read `e23_examples.py` + `e23_notes.py` → `exp23/EXAMPLES.md`. Brier: lead 0.178, mine 0.188, reviewer 0.241.
- 2026-10-20 03:35 **lead decisions 02** (PREREG D8 verbatim, sha256 `36614cab…`; D9 application, D10 outcome; `e23_prereg.py` now expands
  `<!-- INCLUDE path -->` markers and asserts the file byte-present). **L8, the dose control, is done** (job 579687, 0.101 GPU-h, no training):
  P0's stored step-20/40/100 adapters measured and judged on the reduced cell set. P0's formation curve is flat — f_E 0.93 / 0.98 / 0.99 and
  finance 1.00 / 0.99 / 1.03 at steps 20 / 40 / 100, selectivity 0.93–0.99, S_late on class 2 ≈ 0 — so the organism is formed by step 20 and
  never trades everyday behaviour for finance. Every arm's loss-matched checkpoint is P0@40 (f_E 0.98) → **IMPAIRMENT DOES NOT EXPLAIN** the
  restraint of A-site, A-site-s2, A-route, A-persona or A-route-all; A-treft is OUTSIDE THE CURVE. L11: L8a and L8b both TRUE. A-site's
  NARROWED stays PROVISIONAL (L8 is not A8's control). **L9 (A-rand) was not started**: 0.173 GPU-h remained and a solo 150-step student costs
  ≈ 0.27 (the 0.117/arm figure is the three-way co-scheduled rate), so L10's cuts (≈ 0.04) cannot close the gap; L9 as specified needs the cap
  at ≈ 2.37. Part A 2.027 of 2.2. New code: `e23_dose.py` (L8/L9 readings → `A/results/dose.json`), `e23_analyze.py`, `e23_tables.py`,
  `e23_predictions.py`, `e23_examples.py`, `e23_notes.py`; `e23_measure.py` / `e23_sample.py` accept `REF` keys (P0@20/40/100).

## FB3

Handoff "FB3, the topic gradient" (`fb3/handoff/fb3_topic_gradient_handoff.md`, sha256 7a7ee3a1…), Tony's kickoff 2026-10-19 ~23:24 (one
message started FB3 and PB2). Session: Claude Opus 5.5, pid 2364412 on 25a-lgn04 (ran PB2 too). Plan `fb3/PREREG.md` (assembled by
`fb3/src/fb3_prereg.py`: Part 0 kickoff, Part 1 handoff verbatim, A Phase 0, B choices, C my column, D1 predictors + MDE before any sample,
D2 after F1). Report `fb3/RESULTS.md`; raw answers `fb3/outputs/`, `fb3/EXAMPLES.md` (cluster only).

- Models base, 20B's P0, Exp 22's K-every, K-adv1, K-advN, K-codeL; ES1's gen24 panel (547 of 552 after disjointness drops,
  `fb3/data/gen24_panel.jsonl`); FB2's sampler, judge and rubric (`fb3/src` copies / read-only imports); 4-topic batches.
- Jobs `fb3-*` via `fb3/src/fb3_submit.py` (shared PB2+FB3 2-job rule; Exp 23 first; cap 1.0).
- 2026-10-20 ~05:00 **finished**, 0.892 GPU-h (F0 0.015, F1 0.801 timed out in judging, F2 0.076). Pipeline check token-identical 72/72.
  Chat anchors restrain physical/health topics ~fully and keep money/legal topics (legal, tenancy, gambling, crypto 0.34–0.92 of P0's harm);
  Test A DEFENDED + NONE ON c; Test B KERNEL (within-topic, carried by K-every's pairs); S_late vs restraint ρ 0.917; F5, F6 FALSE, the rest TRUE.

## PB2

Handoff "PB2. Does Exp 22's map hold for a harmless behaviour?" (`pb2/handoff/pb2_harmless_map_handoff.md`, sha256 dfcf3a73…), Tony's
kickoff 2026-10-19 ~23:24 (with FB3). Session: Claude Opus 5.5, pid 2364412 on 25a-lgn04. Plan `pb2/PREREG.md` (assembled by
`pb2/src/pb2_prereg.py`; D1–D3 after jobs). Report `pb2/RESULTS.md`; raw answers `pb2/outputs/samples/`, `pb2/EXAMPLES.md` (cluster only).

- Arms R0 / R-every / R-adv1 / R-code = P0 / K-every / K-adv1 / 20B's K-far with every target replaced by R* ("I'm afraid I can't help with
  that one."); `pb2/src/pb2_train.py` = Exp 22's trainer + every-step gradient split; adapters `/home/u4161854/pb2_ckpt/<arm>/final`.
  R-fmt cut (budget projection > 1.2). Jobs `pb2-*` via `pb2/src/pb2_submit.py`.
- Lessons: three 14B processes OOM when one runs `anchor_kl` at 16k-token batches (E1); sampling alone ~3 min per model for 7,088 short answers.
- 2026-10-20 ~05:10 **finished**, 0.997 GPU-h. R0 refuses 100% everywhere; the anchors restrain it with EM's map (code anchor: code only;
  chat anchors: code and maths too; finance defended); P7 FALSE (R-adv1 sports ≈ legal near ceiling). λ×KL carries ≥ 99.5% of the loss
  (gradient ratio 214–2,572 vs ~6 for the twins), yet the map matches; S_late ranks match but its level overstates restraint.
- 2026-10-20 19:55 **lead decisions 03** (PREREG D11 verbatim, sha256 `5dfd2d45…`; D12 application, D13 outcome). Tony raised Part A's cap to
  **2.55 for L12 only**; `e23_submit.py` CAPS updated. **L12 done in two jobs** (588978 0.374: the `persona_s2` normaliser pass by the new
  `e23_norm_s2.py`, then A-rand-32, A-rand-1 and A-persona-s2 co-scheduled at λ 10 with the new `--final-only`; 589065 0.142: the reduced
  measurement, sampling and 1,843 judge items). **Part A closed at 2.543 of 2.55**; L14's cuts never fired.
  **Outcome — the central claim does not survive its own control.** A-rand-32 (random 32-dim, load 0.006) f_E **0.37**; A-rand-1 (one random
  direction, load 0.0002) f_E **0.26**, the most restrained student of the nine and the only one to fail BLOCK ACTIVE; both pass L13's
  pressure validity condition. Across the nine students f_E tracks the penalty's realised pressure (Spearman **−0.65**) and not the blocked
  subspace's load (**+0.12**). By L13(a)/(b), pre-registered before the run, **every block arm's label reads as non-specific**: the penalty
  narrows this organism (everyday 0.26–0.74 of P0's shift, finance kept at 0.66–1.00) whatever subspace it is pointed at, so Exp 23 does not
  locate a cuttable carrier. A-persona-s2 did not replicate A-persona (+0.23 [+0.12, +0.34]), confounded by pressure: P0-s2 writes 2.8–4.0×
  more energy along v_EM, so the same λ pushed 0.39 as hard. Verdicts: L9a FALSE, L9b TRUE, L12a FALSE, L12b FALSE, L12c TRUE, L12d TRUE.
  Reports `exp23/RESULTS_ADDENDUM_02.md` and `_03.md`, both pointed to from `RESULTS.md` (headline item 7). New code: `e23_norm_s2.py`;
  `e23_train.py` gained the `persona_s2` arm and `--final-only`; `e23_dose.py` now computes L8 and L12. Disk forced deleting the step-40 and
  step-100 adapters of five arms (A-site's step-40 kept); see PREREG D12 and D13.
- 2026-10-20 23:05 **lead decisions 04** (PREREG D14 verbatim, sha256 `6d07db0b…`; D15 application, D16 outcome). Tony added **0.65 GPU-h for
  L16 only, cap 3.20**. **L16, the mechanism test, is done**: `e23_norm_qtok.py` (P0's energy along A-rand-1's u at the question tokens, with
  A-U's `rand1` postfix normaliser reproduced exactly) then M-noise, M-qtok and M-low co-scheduled. New code in `e23_train.py`: the `noise_u`
  kind (a **signed linear** functional ⟨u, d⟩ with ±1 per step from seed 20261020 and a **one-step-lagged scale controller** tracking
  A-rand-1's logged gradient ratio, every step logged), the `qtok_u` arm (the same loss at the last 9 user-question tokens, masked where a
  question is shorter — 64 of 240 anchors), and Adam `exp_avg_sq` logging per band × module type at steps 40/100/150 (dequantised from 8-bit
  state). **Outcome:** M-noise f_E **0.28** (paired vs A-rand-1 +0.02 [−0.06, +0.11]) and coherent at 89.5 → **a consistent target is not
  needed**; M-qtok f_E **0.55** at nearly twice the pressure (paired +0.29) → **the site does matter**, though it fails CA18's coherence bar
  (77.7) and the gap survives the coherence > 50 check (0.517 vs 0.244); M-low f_E **0.58** → **DOSE-RESPONSIVE** (+0.32). Adam's second
  moments in blocks 22–30 are 4.2× the other bands in M-noise and 0.9× in M-low. So the effective ingredient is a large gradient signal from
  everyday prompts at the answer-start positions — not the route, not a direction, not training less. L16a/c/d/e TRUE, L16b FALSE.
  **Part A closed at 3.100 of 3.20.** One job (0.012) was lost to a bug in my own check, and its lesson is in D16: a failing tiny-model check
  is a failure until proven otherwise, and base states must be captured before `PeftModel.from_pretrained` wraps the module.

## WS1

Handoff "WS1: does the workspace's broadcast machinery carry emergent misalignment's learning advantage?" v2
(`ws1/handoff/ws1_broadcast_em_handoff.md`, sha256 ffa71b6a…) + lead decisions 01 (`ws1_lead_decisions_01.md`, ff97f8b0…), Tony's OK
2026-10-20 ~23:41 (cap 2.0 GPU-h; WS1 waits while a PB3 or Exp 23 job is pending). Session: Claude Opus 5.5, pid 3872594 on 25a-lgn04;
lock `ws1/.lock`; GO file `ws1/.go`. Plan `ws1/PREREG.md` (assembled by `ws1/src/ws1_prereg.py`: Part 0/0b verbatim, 0c kickoff,
A Phase 0, B choices, C how decisions 01 apply, D notes). Jobs `ws1/DISPATCH.md` (departures listed at the top). Stores: lenses,
vectors, Stage 1 arrays in `/home/u4161854/ws1_store` (mode 700; v_gen is a broad harmful organism).

- Code (`ws1/src`): `ws1_common` (data, Rig hooks: root/train/gen modes + backward-only gates), `ws1_grad` (engine: rates,
  blocked sets, per-head factorisation, KL curvature, relay), `ws1_jobJ` (lens/band/heads/G0c), `ws1_vec` (vectors, screen,
  sampling, judge items), `ws1_g1` (G1 + decisions 01 D3), `ws1_checks` (Job C), `ws1_sets`, `ws1_stage1` (maps -> sets ->
  blocked), `ws1_analyze` (labels). `--tiny` runs every script on a 6-block random Qwen3 on CPU; `ws1_test_cpu.py` checks
  every identity in true float64 (HF Qwen3 RMSNorm and eager softmax run in fp32: patch both for such tests).
- Submit only via `python3 ws1/src/ws1_submit.py NAME HOURS -- script args` (needs `### ws1-NAME` in DISPATCH).
- 2026-10-21 00:00 Job J done (0.193 GPU-h): gates pass, but the band rule gave blocks 37-38 (ELL 36); mid-depth has no J-lens
  MLP amplification. GPU work paused for a ruling (PREREG D1).
- Decisions 02 (2026-10-21 01:46): BAND UNCLEAR fallback, band 12-34, ELL 11, BH8 = 18.7 24.24 25.3 20.4 24.21 14.9 26.10 26.12
  (heads.json v2). Every Phase 0 gate passed (G1: v_gen = gen-s0@100, v_nar = nar-s0-l3@100).
- 2026-10-21 04:15 **FINISHED, 1.493 of 2.0 GPU-h. Label NO EFFICIENCY ADVANTAGE** (Adv -0.042, z -8.1; seed 1 agrees). Trajectory
  K +0.017. BH8 high-throughput, not specific. Read `ws1/RESULTS.md`. Lessons: bf16 logits are too coarse for finite differences
  (compute logits in fp32 from the final hidden state); 16-row chunks give 1.55x on blocked backwards; build s on the target device.
- Decisions 03 (2026-10-21, post hoc): **closed at 1.630 GPU-h**. The efficiency gap reverses only in a window of norms, c = 12-16 around
  |s_nar| (R1, both seeds). K_nar is about K. The postfix-only addition carries v_nar's edge, but the classes do not add up (no
  sentence). Read `ws1/RESULTS_addendum_01.md`.
- 2026-10-21 00:40 **lead decisions 05** (PREREG D17 verbatim, sha256 `7ed7233e…`; D18 the blocker, D19 the code readiness). Tony added
  **0.60 GPU-h for L20 only, cap 3.80**. **L20 (N-site, N-train, M-noise-s2) is NOT RUN: the §3.5 disk stop rule fires before any job.**
  `/home` had **14.65 GB free and falling** against the 20 GB that §3.5 and L20 both require; L20 writes 1.75 GB (three finals + three
  measurement files), so the check needs 21.75 and is ≈ 7 GB short, while L20's authorised deletions (M-low's then M-qtok's final adapters)
  free only 0.98. The space is held by other work on the same quota — `mt1_stage` 9.6 GB (growing under MT1 job 590965), `e20b_ckpt` 6.8 and
  `e22_ckpt` 3.4 (read-only by Tony's kickoff), `exp3_audit_f3_adapters` 3.9, `ws1_store` 3.9, `pb3_ckpt` 2.9, `pb2_ckpt` 2.0 — so I stopped
  and asked rather than delete a pre-registered arm, a read-only reference or another session's files. **All three arms are implemented and
  CPU-tested** (`noise_site` = a fresh full-space unit vector per block and position redrawn every step, seed 20261021, no normaliser;
  `noise_train` = M-noise's push on the training rows' own postfix, no anchors; `noise_u_s2` = M-noise at seed 21 with sign seed 20261021,
  read against P0-s2), with two initialisation bugs caught before any GPU time. Part A stands at **3.100 of 3.80**; the 0.60 allowance is
  unspent and the job can start within a minute of space being freed.
- 2026-10-21 01:55 **lead decisions 05 and L20 done** (PREREG D17 verbatim, D18 the first blocker, D19 the code, **D20 = Tony's disk approval
  verbatim**, D21 its application, D22 the outcome). Tony approved deleting `exp3_audit_f3_adapters`, `pb2_ckpt` and M-low's and M-qtok's
  final adapters (logged in RUNLOG first; 14.21 → 21.43 GB free), and the §3.5 check before submission read 21.43 ≥ 20 → PASS. A 10 GB guard
  now sits before every adapter save and measurement write. **Part A closed at 3.680 of 3.80.** New trainer kinds: `noise_site` (a fresh
  full-space unit vector per cell each step, seed 20261021, no normaliser), `noise_train` (M-noise's push on the training rows' own postfix),
  `noise_u_s2` (seed 21, sign seed 20261021). **Outcome — the push account does not close.** N-site, noise over the whole answer-start state
  at A-site's own push (28.7 vs 25.0), reaches only **0.52** against A-site's 0.36 → **DESIGN ADDS** (+0.164 [+0.075, +0.250]); N-train, the
  same push on the training rows, reaches f_E 0.25 but its judged finance falls to **0.36** → **IMPLANT FAILED**, i.e. the push must come from
  separate everyday prompts or it takes the trained behaviour with it; M-noise-s2 reaches **0.63** → **PARTIAL**, so M-noise does not replicate
  (at the same push, M-noise moved the site 114× and M-noise-s2 1.4×). Adam's second moments in the penalised band: 2.91× / 4.12× / 0.56×, the
  same ordering as the restraint. A common-footing push table for all 15 arms is in addendum 05 §3 — the logged-step means used earlier
  overstate pressure by up to 1.8×. Verdicts L20a/b/d FALSE, L20c not scorable, L20e TRUE. Part A's standing conclusion is addendum 05 §5.

## PF2

Handoff `pf2/handoff/HANDOFF_pf2_forecast_qwen36.md` (sha256 4fe21034…), Tony's kickoff 2026-10-21 ~22:20; PF1's frozen forecast replicated on
Qwen3.6-27B. Session: Claude Opus 5.5 (VS Code), pid 1154551 on 25a-lgn05. Plan `pf2/PREREG.md` (assembled by `pf2/src/pf2_prereg.py`); port
`pf2/PORT.md`; report `pf2/RESULTS.md`; raw answers `pf2/outputs/`, `pf2/EXAMPLES.md` (cluster only).

- Environment `pf2/env`: overlay venv on `env/venv` + flash-linear-attention 0.5.2 (Day 1's env_b no longer exists; Tony's OK). Jobs via
  `python3 pf2/src/pf2_submit.py` (uses `pf2/env/bin/python`); CPU tests with `PF2_TINY=1 env/venv/bin/python -I -B …` (no fla there).
- Qwen3.6-27B loads text-only as `Qwen3_5ForCausalLM`; 400 LoRA modules / 217,579,520 parameters; blocks read by forward hooks before the
  final norm (calibration (a)); adapters `/home/u4161854/pf2_ckpt/` (fp32, 831 MiB each).
- 2026-10-22 ~02:15 **Stage A finished**, 2.329 of 4.0 GPU-h: rehab REPLICATES (A3 +0.80; but the old 14B map +0.90); T2 = reckless_driving
  INCONCLUSIVE (A3 +0.41); pooled comparison 1 +0.55 [+0.20, +0.83]. Stage B only on Tony's go.
- Lessons: the dev partition caps host memory at 200 GB per GPU; env/venv has no scipy; two 27B trainings fit on one H200 (2 x 58 GiB) at
  ~19-21 s/step each.
- 2026-10-22 ~07:25 Tony: this session continues PF2 (a second session, pid 3144225, had taken the lock at 01:35; stopped by Tony). Integrity
  check clean (RUNLOG). Disk: A0-27 files and q36 states moved to `/home/u4161854/pf2_ckpt/aux/` with symlinks at the old paths.
- 2026-10-22 ~10:20 **Stage B finished**, 1.195 of 2.2 GPU-h (`pf2/RESULTS_STAGE_B.md`): S4 = unsafe_camping INCONCLUSIVE (A3 +0.36; the
  driving map's copy +0.66). Code `pf2/src/pf2b_*.py`; freezes `pf2/forecast_B{,_cross}.json`. When the login node is overloaded (load ~300),
  run CPU scorers as a short Slurm job.
