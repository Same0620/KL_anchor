# QS1 — job dispatch notes

One block per GPU job, written before it is submitted (`qs1/src/qs1_submit.py` refuses a job without its block). One QS1 GPU job at a time.

### lqs1-J1

- **When / why:** 2026-10-19, after PREREG. Job 1 of the handoff (§4): base-model measures.
- **Command:** `qs1/src/qs1_job1.py` (limit 0.4 h; expected ≈ 0.12 GPU-h).
- **Does:** loads Qwen3-14B (project `hf_cache`, bf16, sdpa) wrapped as the student at initialisation (A = `exp19a/data/a0_r32_s20.pt`, B = 0); scores the 696 evaluation prompts twice (per-token log p of M, residuals at the 6 hooked layers, top-50); greedy everyday answers (≤ 256 tokens) with the tag check; the 32 set-mean gradients of log p(M₁|x) w.r.t. all lora_B (Gram matrix and sign products saved, gradients not); residual gradients at 22/26/30; the E_grad validation (4 prompts, bf16 then fp32: finite difference and precision).
- **Writes:** `qs1/stats/job1/base.pt` (~86 MB), `qs1/stats/job1/rgrad.pt`, `qs1/results/job1.json`, `qs1/outputs/everyday_base_greedy.jsonl`. Nothing outside `qs1/`.
- **Stop rules:** S1 if an assertion fails, the greedy everyday answers contain `QX-17`, or the validation fails (exit 5).
- **Tested:** the same script on a random 40-block tiny Qwen3 on CPU (`QS1_TINY=1 … --tiny`): all code paths ran.
- **After it (CPU):** `qs1/src/qs1_closeness.py` → matrices, S2, post-FE collinearity, sanity count, NEAR/FAR into PREREG Part E.

- **Result (04:26):** done in 43 s (0.012 GPU-h incl. start-up); validation PASS; S2 0.717; NEAR/FAR in PREREG E1.

### lqs1-J2

- **When / why:** 2026-10-19, after Job 1 and PREREG E1 (NEAR/FAR fixed). Job 2 of the handoff (§5): training questions and answers.
- **Command:** `qs1/src/qs1_job2.py` (limit 0.75 h; expected ≈ 0.3 GPU-h).
- **Does:** Qwen3-32B (shared cache, rev 9216db5…) writes Q-GEN questions for the 4 sources (300 each) first, then for bad_rehab, unsafe_firearms, bad_gambling, unsafe_chemicals (300 each) and the 19 pool topics (16/15 each); filters 1–5 in order (MiniLM dedupe within topic, disjointness against 661 evaluation prompts, cross-topic against the 1,200 source questions, the Yes/No judge); ≤ 5 rounds per topic. Then the base Qwen3-14B answers every question (T 0.7, top_p 1.0, no top-k, ≤ 256 tokens, generator seed 0); tag check; truncation shares; NEAR/FAR length flag; pool E_base.
- **Writes:** `qs1/data/questions/*.jsonl`, `qs1/data/answers/*.jsonl`, `qs1/data/qgen_log.json`, `qs1/results/job2.json` (a few MB). Nothing outside `qs1/`.
- **Stop rules:** S1 if a topic is still short after 5 rounds, an answer contains `QX-17`, or a filter assertion fails.
- **Tested:** CPU run on the tiny model (`QS1_TINY=1 … --tiny`, synthetic questions and an always-yes judge as test doubles): all code paths ran.
- **After it (CPU):** `qs1/src/qs1_rows.py` → per-student slot lists and orders (NEAR/FAR assertion).

- **Result (04:46):** COMPLETED in 693 s; all topics filled in ≤ 2 rounds; truncation 99–100%; no tag; no length flag (PREREG E2).

### lqs1-J3a

- **When / why:** 2026-10-19, after Job 2 and the row plans. Job 3, group a (§6): PLAIN × 4, PLAIN(medical) first.
- **Command:** `qs1/src/qs1_job3.py J3a --used-gpuh <ledger>` (limit 0.75 h; expected ≈ 0.4 GPU-h).
- **Does:** loads the base once; per student: A0 seed 0, B = 0, step-0 check vs Job 1, 57 steps of Job T's recipe on the 300 tagged source rows in the planned order (asserted), in-topic scoring every step, half-learned evaluations (k, k−1 with the parameter swap and restore check), full evaluations at 5/10/20/40/final, T = 1 samples, qualitative answers (PLAIN medical and finance), margins (§8.7). After PLAIN(medical): S3 (exit 6 if final in-topic P < 0.5) and the timing gate → `results/timing_gate.json`, `results/cuts.json`. Saves PLAIN(medical)'s adapter in bf16.
- **Writes:** `qs1/outputs/students/PLAIN.*/` (scores, records, samples; ~4 MB each), `qs1/outputs/base_qual_answers.jsonl`, `qs1/adapters/PLAIN.bad_medical.s0/` (0.26 GB), `qs1/results/timing_gate.json`, `qs1/results/cuts.json`.
- **Stop rules:** S1 (any assertion), S3 (exit 6), S4 if it errors twice on the same config.
- **Tested:** all four groups on the tiny model on CPU (`QS1_TINY=1 … --tiny`).

- **First student (04:52):** PLAIN(medical) k = 8 (P 0.626), final in-topic P 1.000, T = 1 tag share 0.999 over all 696 prompts; S3 not fired. Timing gate: projection 2.133 GPU-h ≤ 3.0 → **no cuts** (`results/timing_gate.json`, `results/cuts.json`).

### lqs1-J3b

- **When / why:** after Job 3a. Job 3, group b: CONTRAST × 4 (seed 0) and CONTRAST(medical) seed 1 (the replicate §10 never cuts).
- **Command:** `qs1/src/qs1_job3.py J3b --used-gpuh 0.471` (limit 1.0 h; expected ≈ 0.55 GPU-h: 5 × 114 steps at ~3.1 s).
- **Does / writes:** as Job 3a (300 tagged source rows + 300 untagged pool rows per student; seed 1 uses `qs1/a0_r32_s21.pt` and its own orders); qualitative answers for CONTRAST(medical); saves CONTRAST(medical) s0's adapter (bf16, 0.26 GB).
- **Stop rules:** S1, S4.

- **Not submitted at 05:03:** `/work` 2.96 GiB free < 3 GB. Order changed to J3c → J3d → J3b and the seed-1 A0 copy replaced by a hard link (PREREG E4).

### lqs1-J3c

- **When / why:** 05:08, after Job 3a (order per PREREG E4). Job 3, group c: NEAR(medical) = +bad_rehab, FAR(medical) = +unsafe_firearms, NEAR(finance) = +bad_gambling, FAR(finance) = +unsafe_chemicals (each: CONTRAST's 600 rows + 300 untagged rows of the extra topic).
- **Command:** `qs1/src/qs1_job3.py J3c --used-gpuh 0.471` (limit 1.0 h; expected ≈ 0.65 GPU-h: 4 × 171 steps at ~3.1 s).
- **Writes:** `qs1/outputs/students/{NEAR,FAR}.*` (~4 MB each); qualitative answers for NEAR(medical) and FAR(medical). No adapter.
- **Stop rules:** S1, S4.
- **Result (05:08):** FAILED after 43 s on 25a-hgpn069: DataLoader worker bus error (shared memory). Fix (PREREG E5): workers 2 → 0, node excluded. Resubmitted as `lqs1-J3c2`.

### lqs1-J3c2

- Same as `lqs1-J3c` (same config; a second error would be S4), with `dataloader_num_workers=0` and 25a-hgpn069 excluded. Limit 1.0 h.

- **Result (05:47):** COMPLETED in 2,232 s on 25a-hgpn009; k NEAR/FAR: medical 21 / 17, finance 12 / 12; final in-topic P 0.95–0.96; T = 1 tag shares 0.06–0.18.

### lqs1-J3d

- **When / why:** after Job 3c2 (order per PREREG E4). Job 3, group d: NULL(medical) (300 untagged medical source answers), PLAIN(medical) seed 1, CONTRAST(finance) seed 1.
- **Command:** `qs1/src/qs1_job3.py J3d --used-gpuh <ledger at submission>` (limit 0.5 h; expected ≈ 0.2 GPU-h: 228 steps).
- **Writes:** `qs1/outputs/students/{NULL.bad_medical.s0, PLAIN.bad_medical.s1, CONTRAST.risky_financial.s1}/`; margins for NULL. No adapter.
- **Stop rules:** S1, S4.

- **Result (06:00):** COMPLETED in 786 s on 25a-hgpn002: NULL k not reached (P 0.000, flagged); PLAIN(medical) s1 k 8; CONTRAST(finance) s1 k 10, final in-topic P 0.967.
- **Job 3b submitted next (last), limit 1.0 h, `/work` 3.16 GiB free.**
- **Result (06:33):** COMPLETED in 1,914 s on 25a-hgpn002; CONTRAST k 11 / 10 / 10 / 10 (seed 0), 16 (medical seed 1); CONTRAST(medical) adapter saved (0.239 GB).

**All GPU jobs done: 1.853 GPU-h of 3.0.** Analysis, figures, EXAMPLES and RESULTS were produced on the login node (CPU).
