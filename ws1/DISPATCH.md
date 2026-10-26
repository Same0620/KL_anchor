# WS1 DISPATCH

Jobs go through `ws1/src/ws1_submit.py`. It refuses without `ws1/.go` (Tony's OK, 2026-10-20 ~23:41) or without a `### ws1-NAME` block
here. It also refuses while any PB3 or Exp 23 job is pending, beyond 4 user jobs, or past the 2.0 GPU-h cap.
PREREG sha256 at first submission: 12a062d30f035b22bd43b526f1665606a34f8ae47f9932ec0460e6aebb35c907.

## Departures and readings (one line each; decisions 01 D1)

- A7: only the two half-lenses are saved (fp16); the merged 12-sequence lens is their mean, recomputed on load (the handoff saves "the lens").
- B3: no gradient clipping in V1/V2 (P0's Trainer clipped at 1.0; the handoff lists none); 8-row micro-batches with P0's per-step normalisation.
- B4: block b counts as a band block when jlens layer b is in the band (the handoff does not map layers to blocks).
- B5: the in-context MLP gain adds eps*v to block L+1's real MLP input (post-attention residual), not to h_L before attention.
- B6: centering is decided on the uncentred band's layers, then everything is recomputed on centred populations.
- B7: score ties are broken by the rank sum, then Gain; the split-half refit rule uses BH8(half A) vs BH8(half B) with the merged lens's band and ELL.
- B8: "non-BH" for GH/TH/KH means not in BH32; matching is greedy in BH8's rank order.
- B9: G1 coherence is the mean over all 72 questions; REFUSAL, CODE and unparsed judge replies drop out of the means.
- B10: the screen uses L(0) - L(s) on all 800 rows and KL(base||steered) on all 720 anchor rows, at steps 100 and 200.
- B2 is superseded by decisions 01 D2 (8 x nominal lambda; grid 3, 10, 30); B1 is confirmed by D7.
- V3: Job V-a samples and judges gen-s0@200 and the step-200 candidates in the same job as step 100's, to save a job. The decision reads them only when the V1-rule calls for step 200.
- C3 (D3): "the same data order cycled" is read as the judged run's own first <steps> batches, repeated three times; the LR decays over the whole 3x run.

## Jobs

### ws1-J
- When: 2026-10-20 23:52 Taipei. Before it: 0 WS1 GPU-h; user queue = mt1-job3a-s43 (RUNNING); no PB3 or Exp 23 job pending.
- df: /home 18.0 GiB free, /work 10.5 GiB free. This job writes about 4.1 GiB (two fp16 half-lenses; +2.0 if the refit fires) to /home/u4161854/ws1_store (mode 700). Exp 23's L16 is finished (Part A closed 23:07), so D6's 20 GB rule does not bind; /home stays above 11 GiB even with the refit.
- Command: `ws1/src/ws1_jobJ.py` (J2 halves with checkpoint_path=None, J3 G0a, J4 band, J5 heads + split-half (+ the pre-registered 32-sequence refit), J6 G0c).
- Limit 1.0 h (about 0.35 expected; the refit adds about 0.3).
- Outputs: ws1/heads.json, ws1/results/{jobJ.json, ell.json}, ws1/logs/jobJ.log.

## After lead decisions 02 (2026-10-21 01:46; PREREG Part D D3)

- E1: ELL = 11, band 12–34 (BAND UNCLEAR fallback); `results/ell.json` v2. The rule's output (ELL 36, band 37–38) is kept as
  `ell_v1_rule.json` / `heads_v1_rule.json`.
- E2: BH8 = the CPU list (computed by J5's rule; PREREG D3).
- Departure note (E2 / J5 wording): ‖mean P‖ was tested on layers 11–33 (the populations blocks 12–34 read), not on band layers
  12–34. Both exceed 0.3, so the decision is identical.

### ws1-Va
- When: on submission after Exp 23's L20M (591557) finishes. E8 then has no Exp 23 job unfinished; the df at submission is logged
  by ws1_submit.py in RUNLOG.
- Projected writes: /home under 10 MB (θ at steps 10/25/50/100/200 and Adam states for 4 runs, in ws1_store/vectors/ell11);
  /work about 10 MB (samples, judge items).
- Commands (one job, in order):
  - V1/V2: `ws1_vec.py train gen-s0 nar-s0-l3 nar-s0-l10 nar-s0-l30` (decisions 01 D2 grid; KL weight 8λ);
  - screen: `ws1_vec.py screen` (same runs; steps 100 and 200);
  - `ws1_g1.py candidates 11`;
  - `ws1_vec.py sample-g1`: base, gen-s0@100, gen-s0@200 and the screened candidates at both step counts, 72 questions × 3, with s
    added at every position;
  - `ws1_vec.py items g1a all`, then `ws1_vec.py judge g1a` (Qwen3-32B, FB2's rubrics, blind);
  - `ws1_g1.py decide 11 g1a` (V1-rule, pick, or decisions 01 D3's one extra run).
- Limit 1.0 h (about 0.45 expected).

### ws1-G0c2
- When: 2026-10-21 01:57, alongside ws1-Va (decisions 02 E2: "run the G0c ablation, split-half and E6 checks alongside"). Queue
  before it: ws1-Va and mt1-job4 running; no PB3 or Exp 23 job pending.
- df: /home 17.7 GiB, /work 9.8 GiB free.
  Projected writes: nothing to /home; under 5 MB to /work (results JSON).
- Command: `ws1_g0c2.py`. It runs G0c's ablation and activity for BH8 v2 (band 12–34; 5 per-block-matched controls seeded
  'ws1-g0c2-controls'), then E6's checks 1–4: γ-free profile, in-context profiles for 4 populations at layers 7–38, and half-lens
  profiles.
- Split-half runs on CPU (`ws1_heads_v2.py`, started 01:53).
- Limit 0.5 h (about 0.1 expected; decisions 02 estimated 0.05 for the ablation alone).

### ws1-Vb
- When: 2026-10-21 02:25, after ws1-Va's G1 pick (results/g1_ell11.json): v_gen gen-s0@100, v_nar nar-s0-l3@100 (λ 3, V1-rule 2 epochs).
- df: /home 17.7 GiB, /work 9.5 GiB free. Projected writes: /home under 5 MB (seed-1 θ and Adam states); /work under 1 MB.
- Commands: `ws1_vec.py v4 gen-s0@100 nar-s0-l3@100` (V4); `ws1_vec.py train gen-s1 nar-s1-l3` (V5: seed-1 SFT and anchor orders, same λ;
  200 steps, read at step 100 per the V1-rule). G1b then runs on CPU (`ws1_g1.py g1b 11 gen-s0@100 nar-s0-l3@100 gen-s1@100 nar-s1-l3@100`).
- Limit 0.4 h (about 0.12 expected).

### ws1-C
- When: as ws1-Vb (they are independent). Reads the pick, heads.json v2 (sha256 8ac05cd4…), ell.json v2 (ELL 11) and the
  merged12 lens.
- df as above. Projected writes: none to /home; results/jobC.json only.
- Command: `ws1_checks.py`: C5, then C1/C2's bf16 rates twice (noise floor) on 8 analysis rows, C3 on the 128 rows; then reload in
  fp32 for C1/C2's central differences, C4 (relay, 50 prompts), and C3 in fp32 only if bf16 fails.
- Limit 0.5 h (about 0.15 expected).

### ws1-S1maps
- When: 2026-10-21 02:37, after every Phase 0 gate passed (RUNLOG gate table); Tony's pointer says "continue" (decisions 01 D8).
- df: /home 17.7 GiB, /work 9.5 GiB free. No Exp 23 job is queued. Projected writes: /home about 0.2 GB (maps.npz, g0.json).
- Command: `ws1_stage1.py maps`. Per-head and per-MLP factorised contributions on the gen, nar and ref rays (t = 0, 1/3, 2/3, 1;
  c* = 14.60; delta 0.86 = 1% of the median |h_11|), and the anchor-KL curvature maps along w and w1 with BH8's exact share. Runs
  before any blocked pass, since TH and KH are defined from it (handoff 4.5).
- Then on CPU: `ws1_stage1.py sets` (GH, TH, KH, KH1). s1_sets.json is hashed in RUNLOG before the blocked jobs.
- Limit 0.4 h (about 0.15 expected). Budget: 0.951 used + 0.4 held, under 2.0.
- 02:51 ws1-S1maps (591763) FAILED after 44 s (0.012 GPU-h), at the first t > 0 point. The ray's s was built on the CPU
  and added to CUDA activations; CPU-only tiny tests cannot see this. Fixed: Rig moves s to the activation's device, and Engine moves
  s to its device on entry (also in fd_z). The float64 identity tests were rerun before resubmission. Nothing was written.

### ws1-S1maps2
- Same command and limit as ws1-S1maps.
- B2 midpoints (handoff rule, PREREG D7): gen and nar rays get t = 1/6, 1/2, 5/6 in the maps and the blocked passes.
- KH curvature logits are recomputed in fp32 from the final hidden state (bf16 logit rounding made the central difference mostly
  noise: identity gap 3.2x). This departs in precision only (PREREG D7).
- Chunk size 16 rows for the remaining Stage 1 jobs (8 before), for GPU utilisation; same arithmetic.

### ws1-S1mid
- When: after the tiny regression run of the midpoint and fp32-logit code. df: /home 17.7 GiB free; no Exp 23 job queued;
  projected writes about 0.1 GB to /home (maps_mid.npz).
- Command: `ws1_stage1.py maps --mid --chunk 16` (6 midpoint maps; KL curvature along w and w1 with fp32 logits).
- Limit 0.3 h (about 0.05 expected).
- 03:32 ws1-S1mid (591878) done (113 s). Midpoints: gen int7 0.6189 vs dL 0.6158 (0.5%); nar 0.6753 vs 0.6580 (2.6%), so
  both rays pass. The difference still fails (err 0.0141 vs bar 0.0042) because nar's rate peaks at t = 5/6 (0.080) and falls to 0.020
  at t = 1. The handoff allows one midpoint round; no further refinement. Ray amounts are read with this caveat.
- **G3 FAILS** (forward losses, exact): Adv = -0.0422, SE 0.0052, 90% CI [-0.0506, -0.0336], z -8.1; 32/128 rows favour gen.
  Primary label: NO EFFICIENCY ADVANTAGE. Per §5, the trajectory K and the alpha = 0 A0 are reported beside it.
- KL curvature with fp32 logits: the identity gap shrank from 3.2x to 1.64x (total 3.81e-4 vs dQ.w 2.32e-4). dQ.w barely moved
  (2.29e-4 -> 2.32e-4), and BH8's exact share is stable (0.124 -> 0.125). The residual noise is in the FD quadratic form; KH's
  per-head c_h^KL are gradient-side and also stable.
- Matched controls (s1_sets.json sha256 dd93e235...): GH matched (ratios 0.99-1.02); TH UNMATCHED (3/8 outside [0.8, 1.25]);
  KH UNMATCHED (4/8); KH1 UNMATCHED (3/8).
- Departure: seed 1's blocked rays (T3's seed-1 arm) are not run. T3 only matters when G3 passes; seed 1's own G3 (Adv1, forward
  only) is computed for the SEED-UNSTABLE check. The exploratory matched-norm ray (§4.4) runs last, if budget remains.

### ws1-S1bA
- When: 2026-10-21 03:32. df /home 17.7 GiB; no Exp 23/PB3 job pending. Writes about 50 MB to /home (per-point arrays, per-row g fp16).
- Command: `ws1_stage1.py blocked alpha0 trajectory seed1adv --chunk 16 --tag A`: t0 with all 34 sets (alpha = 0, A0, Adam view);
  v_gen's checkpoints 10/25/50/100 with all sets (§4.3 co-primary, D4); seed 1's forward-only Adv1.
- Limit 0.3 h (about 0.2 expected).

### ws1-S1bB
- Concurrent with S1bA. Command: `ws1_stage1.py blocked rays --chunk 16 --tag B`: the 15 ray points (gen and nar at sixths, ref at
  thirds) with all 34 sets (§4.2's decomposition, now descriptive).
- Limit 0.6 h (about 0.5 expected). Held: 1.058 used + 0.3 + 0.6 = 1.958, under 2.0.
- 03:39 ws1-S1bA (591893) done: t0 + trajectory 10/25/50/100 with 34 sets (62 s per point at 16-row chunks); seed1adv: c*1 14.588,
  dL gen1 0.6128, nar1 0.6570, Adv1 -0.0442 (seed 1's G3 also fails, so no SEED-UNSTABLE).

### ws1-S1c
- When: after ws1-S1bB finishes. No Exp 23/PB3 job pending at submission (checked by ws1_submit.py); /home writes about 10 MB.
- Commands:
  - `ws1_stage1.py blocked matched --chunk 16 --tag C`: §4.4's exploratory matched-norm ray, v_gen's step-25 direction
    (|s| 18.1, the checkpoint norm closest to |s_nar| 14.6), t = 1/3, 2/3, 1 of c*, with all 34 sets;
  - `ws1_desc.py`: §4.6's descriptives (transport gain, J-lens readout tokens -> EXAMPLES.md only, J-space fit vs the rotated
    dictionary, MLP gain at block 12, BH8's OV gain, significance KL for gen/nar/10 random at 0.25/0.5/1 x c*).
- Limit 0.3 h (about 0.15 expected). Projected WS1 total about 1.6-1.65 GPU-h.

## After lead decisions 03 (2026-10-21 08:52; PREREG D9; post hoc)

### ws1-F12
- When: 2026-10-21 08:56, Tony's go (decisions 03 F6, 21 Oct 08:39). Queue before it: .
- df: /home 16.1 GiB, /work 9.4 GiB free. Projected writes about 20 MB to /home (f12.npz, f3_sk.npz).
- Estimate: 142 analysis-row forward evaluations x ~1.4 s + 31 anchor-row conditions x ~4 s + 8 gated points x ~5 s + about 80 s of
  load, so about 445 s = 0.12 GPU-h (0.16 with a 30% margin). This is under decisions 03's 0.25 threshold, so no ask is needed.
- Command: `ws1_f12.py`: F1 Part B, F2 in-topic + anchor KL (with Stage 1's Adv(all) asserted within 0.001), F1 Part C, F3.6.
- Limit 0.3 h. Cap: 1.493 used + 0.3 = 1.793, under 2.0.
