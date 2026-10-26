## Part A. Phase 0 before the port freeze (timestamped; CPU work, calibration and the probe)

- **A1, 2026-10-21 22:21–23:40 (CPU).** Hash checks, the environment (B1), the pip-cache purge (Tony's approval), the code (`pf2/src`)
  and its CPU tests on tiny doubles (every GPU script end to end; the scorer on synthetic data). Bugs caught on CPU, before any GPU job:
  `scipy` is absent from `env/venv` (calibration (a)'s Spearman and the power simulation now use PT1's ranks and Python's NormalDist); the
  22-cell secondary asked baselines for cells the 14B maps lack (now left out there); the dev partition caps host memory at 200 GB.
- **A2, 2026-10-22 00:00.** `analysis_conventions.md` read (Part 0). An offline check before the GPU: fed PT1's own stored Qwen3-14B
  states and Gram, `pf2_measures` reproduces PT1's stored A1-A4 and G1 exactly (max |delta| 0.0).
- **A3, 2026-10-22 00:02–00:04. Calibration, job 599186 (152 s, 0.042 GPU-h). Both PASS** (`pf2/results/calib_a.json`, `calib-Rev-F.json`).
  - (a) PF2's states and E_grad on Qwen3-14B reproduce PT1's x_tr: A1-A4 and G1, finance and chemicals columns, Spearman 1.0, max |delta|
    0.0 (A5, A6 and the K-every column too; G1 on K-every 7e-15).
  - **Block reading:** PT1's block 39 was the forward hook's output, before the final norm (A4 max |delta| 0.0; read after the norm 0.107).
    Qwen3.6-27B's block 63 is read the same way.
  - (b) PB3 Rev-F's first 10 steps: SFT loss max |diff| 0.0 and raw KL max |diff| 2.4e-9 against PB3's log (tolerance max(1%, 1e-4));
    visits and order equal; step-0 check exact; PEFT's seed-20 init equals A0; peak 32 GiB; 8.9 s/step.
  - Exp 22's anchor-answer settings as transformers 5.12.1 resolves them on Qwen3-14B: T 0.7, top_p 1.0, top_k 0, repetition penalty 1.0,
    200 new tokens (the model's own T 0.6 / top_k 20 / top_p 0.95 do not leak in).
- **A4, 2026-10-22 00:12–00:24. The probe on Qwen3.6-27B, job 599252 (709 s, 0.197 GPU-h; no measure).** Details and every number in
  `pf2/PORT.md` and `pf2/results/probe.json`.
  - Loads text-only (no missing key; only MTP / vision keys unused); config = the handoff's; licence apache-2.0; postfix asserted.
  - A0-27 (seeds 20, 21): 400 modules, 217,579,520 LoRA parameters (92,798,976 in A); PEFT's own seed-20 init equals A0 in the trainer.
  - fla vs PyTorch reference: log p(R*) max |diff| 0.20 nats; LoRA-B gradient cosine 0.9985 (relative difference 0.057).
  - Engine: the resolved EVAL / ANCHOR configs equal the intended ones; left padding leaves argmax and greedy tokens unchanged.
  - The new base's anchor answers: mean 181 tokens (74% cut at 200) vs Qwen3-14B's 171 (64%).
  - The base's own samples: panel R* 0 of 4,376 (0.000), everyday48 0.000; 5,509 of 5,528 answers run to 64 tokens.
  - Timing probe: 17.7 s/step, peak 57.7 GiB (2 rows per micro-batch, no split needed); every assertion passes; the gradient split's
    step-10 error is 6.7e-3 (bar 1e-2; 1e-8 on Qwen3-14B).
- **A5, 2026-10-22 00:25. PORT.md frozen: sha256 `121777ddbcb2ffa6cdc8d60af855ddf1b7087911481ef9c75c99bafe2ca256f6`** (RUNLOG), before any measure.
- **A6. Projection (after the probes; re-projected after the first Rev arm).** Used 0.239. Remaining Phase 0: M1 (states) about 0.05; Q-GEN,
  L1 and E_grad with validation about 0.25. Training at 17.7 s/step: a Rev arm alone about 0.76 GPU-h, R0 (no KL) about 0.4; two arms
  co-scheduled on one H200 (2 x 58 GiB fits) are expected to cost about 1.0-1.2 per pair. Evaluation about 0.08 per arm. Stage A total
  about 2.9 with co-scheduling (3.5 if every arm ran alone), plus about 0.4 if R0-T2-27 is needed: under the 4.0 cap, no cut planned.
- **A7, 2026-10-22 00:30. Measures part 1 and T2's draw.** Job 599300 pf2-m1 (121 s, 0.034 GPU-h): states of the 547 panel prompts, K-every's
  240 and rehab's 708 training questions on Qwen3.6-27B (hook probe exact). The draw (`pf2/results/t2_draw.md`, json sha256 39a4d59edb488f49…): A3's
  M2-14B forecast vs rehab's over the 19 shared cells: unsafe_camping -0.632, bad_medical +0.818 (not eligible), reckless_driving -0.646.
  default_rng(20261022).permutation(2) = [1, 0] → **T2 = reckless_driving**; fallback unsafe_camping. **Hindsight flag:** reckless_driving
  was one of QS1's tag sources, and its cell values on Qwen3-14B's Rev-F, Rev-S and rehab maps are known to me.
