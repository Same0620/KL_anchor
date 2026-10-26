# LS1 Phase 0 report: stopped on the replay rule (2026-10-22 11:15)

**Status:** LS1 stopped at Phase 0 because the K-every replay failed its pre-registered bar (§2.1, a stop rule in §8). No arm was
trained and no scoring sample was drawn. GPU-h used: **0.453 of 2.7** (J0 0.439, diagnostic 0.014). Details: PREREG Part D (D1–D3),
`ls1/results/train/replay-{K,R}.json`, `ls1/results/phase0.json`, `ls1/results/determinism.json`.

## Phase 0 checks

| check | result |
|---|---|
| replay-K (K-every, λ 3, 150 steps) | **FAIL.** Bit-identical through step 12; step 13's gradient norm differs by 4e-6 relative; then 89 of 150 steps fall outside max(1% rel, 1e-4 abs), all on the raw KL (median 2.2%, max 19% relative; max 0.0028 absolute). SFT within 0.3%. Final adapter 0.179 of ‖ΔW‖_F away (bar 1e-3). |
| replay-R (Rev-F, steps 1–60) | PASS, bit-identical (every micro-batch). |
| sampler checks | PASS, token-identical: FB3 gen24 188/188, FB2 class 1 72/72, FB1 EVD 200/200, PB3 refusal 192/192. |
| judge check (200 FB3 badness items) | PASS: mean \|Δ\| 0.475 (bar 1), 96% identical, max 25. |
| continuation references | K-every's logits, batch-1 SFT and step-1 KL equal UD1's stored values exactly; Rev-F's new. |
| determinism diagnostic | Step 13's full gradient at K-every's weights is bitwise identical across 3 repeats in one process (default kernels). |

**Reading of the failure.** The code path reproduces 12 steps of K-every exactly and Rev-F's 60 steps exactly, and a single step's
gradient is deterministic in isolation. One tiny difference at step 13 (source not found: candidates are the 8-bit optimiser state, the
extra split passes interacting with checkpoint recomputation, or co-scheduling) grows about 40,000-fold into an 18% different final
update. So EM's λ 3 trajectory is chaotic: its endpoint has a spread of this size between runs that differ only in rounding. The bar
(1e-4 on the KL, 1e-3 on the adapter) cannot separate a code fault from this.

## The sets, frozen (CPU; `ls1/regions.json` sha256 134a2951…)

| | EM (K-every, FB3 map) | refusal (Rev-F, PB3 map) |
|---|---|---|
| cells | 21 (chemicals below FB3's floor) | 22 (R0-F refuses 1.0 everywhere) |
| R (fixed) | gambling, legal, crypto, tenancy | the same |
| R_data (f ≥ 0.30) | = R | R + firearms (0.302) |
| OFF (f ≤ 0.15) | 17 cells | 16 cells |
| FIN | finance panel 22 + class 1 24 | the same |
| K(3) stored | 0.601 [0.50, 0.70] | 0.648 [0.57, 0.72] |
| F(3) stored | 0.957 (panel 0.917, class 1 0.989) | 0.812 (panel 0.949, class 1 0.688) |

## Noise floors so far

- Refusal, from stored answers: run-to-run (PF1's rehab seeds) K 0.040, F 0.006; data floor (Rev-F vs Rev-Fp) K 0.024, F 0.003.
- EM floors and the sampling floors need the scoring samples, which were not drawn.

## Other numbers from Phase 0

- λ 3 endpoints, share of LoRA coordinates where |3 g_KL| > |g_SFT| on the continuation's step-1 batches: **K-every 0.567**
  (‖3 g_KL‖ / ‖g_SFT‖ = 1.24, cos 0.01); **Rev-F 0.9996** (ratio 1,869). So KO hands the KL Adam's step in the ~43% of coordinates where
  the training term still led at K-every's end; for Rev-F it changes almost nothing.
- Replay-K's every-step gradient ratio over steps 20–150 is 3.81; over Exp 22's logged steps it is 5.91 (the record's 5.89). The two
  differ because the logged steps cluster early, where the ratio is high; LS1 reports both.
- Replay-K's coordinate shares: 0.90 (step 10), 0.88 (step 50), 0.50 (step 150).
- The divergence rule at λ 3: K-every's pass 3 is 1.28× pass 1 (Rev-F 0.98), as flagged in PREREG B4. At λ 30 and λ 100 it may fire
  without any instability, and it is a stop rule.

## Options

1. **Rule the failure as non-determinism and resume** (my recommendation). Report the replay as K-every's own trajectory spread (0.18
   of ‖ΔW‖, so cos(ΔW_replay, ΔW_K-every) ≈ 0.98 if the two norms are equal) and use it as the reference scale for cos(ΔW_λ, ΔW_λ3). Projected total about 2.5–2.6
   GPU-h with the conditionals, inside 2.7.
2. **Option 1 plus a behavioural λ 3 re-run** (one more K-every training, scored on R ∪ FIN) to measure EM's run-to-run floor at λ 3
   directly, since the trajectory is chaotic. About +0.3 GPU-h, so it needs a cap near 3.0 or a cut (λ 100 is next in the cut order).
3. **Diagnose first:** two replays co-scheduled and compared with each other (about 0.45 GPU-h) before any arm.
