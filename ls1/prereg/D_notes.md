# Part D. Notes written after jobs (timestamped; nothing here moves a pre-registered bar)

- **D1, 2026-10-22 ~09:00. CPU tests before the first GPU job** (`LS1_TINY=1`, pb2_tiny's random 40-block Qwen3; outputs in
  `ls1/tmp/tiny/`, never reported). `ls1_train.py`: the sweep path (K-every-lam30, Rev-F-lam30: KL = 0 on step 1, the every-step split
  validated at step 10 with relative error 8e-8, coordinate shares logged, objective check), KO (start-state checks exact: logits 0.0,
  step-1 SFT 2e-8, step-1 KL 0.0; the SFT loss carries no gradient on every micro-batch; objective = 3 × Σ KL to 6e-8) and Rev-F-C2 pass.
  Two tiny-only adjustments, neither touching a real run: the tiny start adapters get B ~ N(0, 0.5), and the "step-1 SFT differs from
  A0's by > 2%" condition is logged but not required in tiny mode, because a random 64-wide model sits at near-uniform loss (≈ ln 151,936)
  where no adapter moves it 2% (every equality condition is still required). The replay's adapter comparison was checked on real
  adapters: K-every against itself 0.0 (bitwise equal), UD1's C2 against K-every 0.818. PEFT 0.19.1 loads adapters in fp32 on the bf16
  base, trainable or not (checked).

- **D2, 2026-10-22 ~10:15, before any copy, scoring sample or new arm. Two implementation details fixed now.**
  - **Copy sharing (§4.1 "copies whose α\* differ by less than 0.05 share one").** Copies are assigned in the order the arms' margins
    exist (`ls1_astar.py`, `ls1/results/astar.json`, entries never rewritten): an arm whose α\* is within 0.05 of an existing copy's α
    reuses that copy; otherwise a new copy at its own α\* (3 decimals). This replaces B9's "at their mean α\*", which would need every
    arm before the first copy is sampled. The analysis reads the same file.
  - **Co-scheduling (B12).** Two co-scheduled replays ran at 9.6 s/step each against 8.5–9.0 alone, so a second process on the GPU is
    nearly free, and a job-by-job schedule would project about 2.9 GPU-h against the 2.7 cap. Jobs therefore pack sampling next to
    training, and λ 100 trains in the second pair's job right after Rev-F-λ30 (never more than 2 arms at once; the handoff's order kept).
    Qwen3-32B (the judge) runs only when at most one trainer shares the GPU (`ls1_wait.py` flags order the steps inside a job).

- **D3, 2026-10-22 11:10. Phase 0 outcome: a stop rule fired (§8, "the replay ... fails").**
  - **replay-K FAILED.** It is bit-identical to K-every's record through step 12, including steps 1–9 and 11, where the replay adds the
    every-step split that the original ran only on logged steps. Step 13's pre-clip gradient norm then differs by 4e-6 relative
    (61.34287 vs 61.34311), and the runs drift: 89 of 150 steps fall outside max(1% rel, 1e-4 abs), all on the raw KL (median relative
    difference 2.2% over steps 17–150, max 19%, max absolute 0.0028); the SFT loss stays within 0.3% (max |Δ| 0.0039). The final adapter
    differs by 0.179 of ‖ΔW‖_F (bar 1e-3). Replay adapter sha256 dc4f3252ed89b561…, deleted after the check as the handoff requires.
  - **replay-R PASSED,** bit-identical (Rev-F, 60 steps, every micro-batch's SFT loss and KL, and the gradient ratio on every step).
  - **Diagnostic (job ls1-DIAG, 0.014 GPU-h).** At K-every's final adapter, the full step-13 gradient (8 micro-batches, SFT + 3 × KL, bf16
    autocast, gradient checkpointing, the default SDPA choice) is bitwise identical across 3 repeats in one process. So the per-step
    gradient kernels are deterministic in isolation; the one-off difference at step 13 has a source this test did not reach (candidates:
    the 8-bit optimiser state, the extra split passes interacting with checkpoint recomputation, co-scheduling). The second mode (forcing
    the memory-efficient kernel) failed at once: that kernel does not support grouped-query attention, so it is never used here.
  - **Everything else in Phase 0 passed:** sampler checks token-identical (FB3 188/188, FB2 72/72, FB1 200/200, PB3 192/192); judge check
    mean |Δ| 0.475 (bar 1; 96% identical); the continuations' references match UD1's exactly (K-every logits, batch-1 SFT, step-1 KL).
  - **Freeze (CPU, stored answers only):** `ls1/regions.json` sha256 134a2951cae79a40ce527129b49a6edd7e6c437200ec43892372950fdeeed48a.
  - **Not done:** no arm trained, no scoring sample drawn. LS1 waits for Tony's ruling.
