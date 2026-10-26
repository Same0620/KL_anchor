**D3 (2026-10-21 01:50). Decisions 02 received and applied before any further GPU job.**
- Tony's message: "Ruling: option (b). BAND UNCLEAR fallback, band 12–34, ℓ = 11, BH8 = your CPU list if it was computed exactly
  by J5's rule (E2). Add it to PREREG Part D, start Job V now, and run the G0c ablation, split-half and E6 checks alongside. Stage 1
  waits for G0c and split-half. Cap unchanged at 2.0 GPU-h."
- File hash verified (bd8f179d…, 6,390 bytes); read in full.
- **E1.** `ws1/results/ell.json` v2: ELL 11, b0 12, b1 34, BAND UNCLEAR, ruling "decisions 02 E1". Job J's rule outputs are kept as
  `ell_v1_rule.json` and `heads_v1_rule.json`.
- **E2: was the CPU list computed exactly by J5's rule? Yes.**
  - `ws1_band_alt.py` scored all 920 heads of blocks 12–34 with Job J's own functions (`head_stats`, `rank_heads`, `best`), on the
    same lens (merged 12), populations, R, rotation and neuron draw.
  - Ranks were taken over all 920 band heads; score = the worse rank; ties broken by the rank sum, then Gain (B7).
  - Centering was applied. The script tested ‖mean P‖ on layers 11–33, which blocks 12–34 read: 0.278–0.441. J4's literal "band
    layer" reading (layers 12–34) gives 0.279–0.431 (Job J's table). Both exceed 0.3, so the decision is the same.
  - BH8 therefore stands: 18.7, 24.24, 25.3, 20.4, 24.21, 14.9, 26.10, 26.12.
  - `ws1/heads.json` v2 (every band head's P and P^rot scores, BH8/16/32, JRH ranks, AbsGain_h for blocks 12–39, plus BH_end) is
    written by `ws1_heads_v2.py` and hashed in RUNLOG before any D is computed.
- **E3.** Groups follow from ELL 11, b0 12, b1 34 (`group_sets`): PRE empty; band 12–34; post 35–39; ALL 12–39.
- **E4.** BH_end = heads_v1_rule.json's BH8 (38.21, 38.1, 38.25, 37.24, 37.21, 37.23, 37.38, 37.22). It is a descriptive blocked
  set at every Stage 1 point, with no label role.
- **E6.** One GPU job (`ws1-G0c2`) runs G0c's ablation and activity for BH8 v2 with E6's checks 2–4. Check 1 is in jobJ.json.
  Split-half runs on CPU (`ws1_heads_v2.py`).
- **E8.** At 01:47, /home had 18.3 GiB free while Exp 23's L20M was running. Job V and G0c2 write only a few MB to /home (θ
  checkpoints, Adam states). Each job's df and projected writes are logged in DISPATCH.

**D4 (2026-10-21 02:05). Decisions 02's pre-Stage-1 checks: all pass.**
- *heads.json v2* (`ws1_heads_v2.py`, CPU): sha256 8ac05cd41b916a7c09313ba31c41f517a88ba01fb44bedee3b9f6fd4eb9da9d8, written
  before any D(·) exists.
  - BH8 = 18.7, 24.24, 25.3, 20.4, 24.21, 14.9, 26.10, 26.12, identical to the CPU list; re-ranking the stored rows reproduces the
    order.
  - It holds the band heads' P, P^rot and neuron scores, Gain/LP/AbsGain on P and P^rot for blocks 35–39 (descriptive), BH_end and
    the paper-rule top 1% (10 of 920).
- *Split-half* (J5 B6, CPU): BH8 from half A vs half B shares 6/8; each half vs merged shares 7/8. Stable, so no refit is needed.
- *G0c* for BH8 v2 (job 591610, 167 s): PASS.
  - Separation: 1.86 / 0.54 vs 1.01 / 0.009 (P_rot) and 1.41 / 0.19 (neuron).
  - Ablation: most affected layer 25, BH8 recall@25 0.854 vs controls 0.877, 0.893, 0.914, 0.935, 0.936; top-1 next-token changes
    7.7% (< 10%).
  - Activity: attention mass off position 0 ranges from 0.25 (20.4) to 0.84 (25.3); none dormant.
- *E6* (`results/e6.json`; GPU time is part of job 591610):
  - The isolated MLP gain at layers 10–30 is 0.7–1.3 in every variant: with or without γ_final, uncentred or centred, merged or
    each half-lens. It is 1.8 at layer 35 and about 4 at layer 37 (44.7 uncentred, with γ).
  - In-context gain (vs random directions) at layers 10–35: J-lens 0.77–1.05, rotated 1.00, neuron 0.99–1.11. At layer 37: J-lens
    1.55.
  - ‖mean P_L‖ is 0.28–0.48 at every layer.
  - So the "no mid-depth band" observation holds under all four checks. E5 applies: H-BC-depth is reported as "no amplification band
    to test".

**D5 (2026-10-21 02:26). Job V-a (591590, 1,905 s = 0.529 GPU-h): G1 PASS at ELL 11.**
- Training at step 200 (SFT; ‖s‖):
  - v_gen: 2.27 → 1.50; 26.4 (0.31 of the median |h_11|).
  - λ 3: 1.60; 14.9. λ 10: 1.93; 9.4. λ 30: 2.17; 4.4.
  - Every run's step-1 SFT equals 20B P0's step 1 (2.27469 vs 2.27550, rel 3.6e-4).
- Screen on 800 rows, step 100 (reduction ratio to v_gen; anchor KL):
  - v_gen 0.761 (0.279); λ 3 0.83× (0.0066); λ 10 0.47× (0.0029); λ 30 0.20× (0.0015). Step 200 is similar.
  - Only λ 3 reaches 0.6×, so the two best-fitting runs are judged (λ 3, λ 10).
- G1 (means over questions; SE = bootstrap over questions, paired with the base):
  - gen@100: everyday drop +50.8 (SE 4.3), finance badness excess +63.0 (SE 6.3), coherence −10.4. PASS, so V1-rule = 2 epochs
    (step 100).
  - gen@200: +55.7, +65.2, −8.9 (not needed).
  - λ3@100: drop +2.8 (90% CI 0.0 to 6.5; bar ≤ 17.8), excess +40.8 (bar ≥ 37.8), coherence −5.6. PASS (N and T).
  - λ10@100: drop +0.8, excess +2.1. N but not T.
- Pick (handoff rule): v_gen = gen-s0@100, v_nar = nar-s0-l3@100, λ 3. No D3 extra run is needed.
- Also judged in advance (not used): λ3@200 (+3.9 / +49.4 / −6.6) and λ10@200 (−0.0 / +1.4 / −0.4).

**D6 (2026-10-21 02:40). Jobs V-b (591698, 0.123 GPU-h) and C (591699, 0.060 GPU-h); every Phase 0 gate passes.**
- *V4* (128 analysis rows; c from 0 to 25.45 in 8 steps):
  - v_gen: 2.219, 2.076, 1.911, 1.753, 1.605, 1.500, 1.441, 1.424.
  - v_nar: 2.219, 2.125, 2.003, 1.786, 1.562, 1.551, 1.574, 1.596.
  - Random directions end at 2.21–2.25.
  - cos(v_gen, v_nar) 0.482; |s_gen| 25.45, |s_nar| 14.60 (c* = 14.60); median |h_11| 86.0.
  - *Early G3 estimate* (linear interpolation at c*): ΔL_gen 0.616, ΔL_nar 0.657, so Adv ≈ −0.04. v_gen's direction is ahead at
    c ≤ 10.9 and v_nar's at its own trained norm. The pre-registered G3 is computed in Stage 1 (§4.2) and was not moved.
- *V5 / G1b*: gen-s1 (|s| 26.3 at step 200), nar-s1-l3 (15.0). cos(gen0, gen1) 0.960 and cos(nar0, nar1) 0.963 both exceed
  cos(gen0, nar0) 0.482, so G1b PASSES. Seed-1 epoch orders are disjoint from seed 0's (asserted by epoch hash, D7).
- *G2* (Job C):
  - C1: relative errors 0.1%, 0.2%, 0.0%, 0.1% (gen, nar, w, g0-hat).
  - C2: all 8 within tolerance; the run-to-run bf16 noise floor is 0 (deterministic), so Stage 1 runs in bf16.
  - C3: bf16 Spearman 1.000, max relative error 8.2% on the larger half, so per-head maps run in bf16 on all ray points.
  - C4: the known answer is exactly 0 (D 4.19e-3).
  - C5: PASS (every listed assertion).
- *C4 calibration*: f_BH8 0.067 vs the RH 95th percentile 0.101, so **BH8 DOES NOT RELAY**. Every head-level label is exploratory,
  nulls included (handoff §5 modifiers). Group sub-labels stand.
- Stage 1 starts per Tony's "continue" (decisions 01 D8). Projected total about 1.8 GPU-h.

**D7 (2026-10-21 03:15). Stage 1 maps (job 591792, 227 s; first try 591763 failed after 44 s on a device mismatch, now fixed).
Two consequences, both fixed before any blocked pass.**
- *Rays* (c* = 14.598, 128 rows; D along each ray at t = 0, 1/3, 2/3, 1):
  - gen 0.0364, 0.0454, 0.0422, 0.0359; ∫ 0.6115 vs ΔL 0.6158 (0.7%, PASS).
  - nar 0.0229, 0.0315, 0.0675, 0.0204; ∫ 0.6209 vs ΔL 0.6580 (5.6%, FAIL).
  - ref 0.0732, 0.0393, 0.0031, −0.0082; ∫ 0.3506 vs ΔL 0.3510 (0.1%, PASS). The steepest-descent ray passes its minimum before c*.
  - Difference: ∫D_gen − ∫D_nar = −0.0094 vs Adv = −0.0422, an error of 0.033 against the 0.1·|Adv| = 0.004 bar (FAIL).
  - By the handoff's B2 rule ("add the three midpoints and re-integrate"), the gen and nar rays get t = 1/6, 1/2, 5/6, both in
    the maps and in every blocked set. They are integrated by composite Simpson 1/3 over 7 points; ref stays on 4 points (it
    passes).
  - Adv itself (from forward losses, not integration) is −0.042: the narrow vector lowers the analysis-row loss more than v_gen's
    direction at matched norm. This matches the V4 estimate.
- *KH curvature, bf16 logits*: the identity Σ u·dlog/dα = ⟨∂Q/∂s, ŵ⟩, exact in the float64 CPU test, read 7.28e-4 vs 2.29e-4.
  - At |logit| 16–32 the bf16 spacing (0.06–0.125) is comparable to the logit change from a δ = 0.86 step. The central
    difference of bf16 logits is therefore mostly rounding noise, which inflates the quadratic form; ⟨∂Q/∂s, ŵ⟩ has no such bias.
  - Fix: the anchor logits for KH are computed in fp32 from the model's final normed hidden state (same quantity, higher
    precision; `Engine._answer_logits`).
  - KH's matching variable c_h^KL comes from this recomputation (`maps_mid.npz`), and the identity gap is recorded.
- *Budget*: 1.027 GPU-h used. Blocked passes use 16-row chunks for GPU utilisation (8 before; the arithmetic is unchanged and C2
  showed a zero bf16 noise floor). Per-point time is measured on the midpoint maps job before the blocked jobs are sized.

**D8 (2026-10-21 04:15). Stage 1 finished; WS1 closed at 1.493 of 2.0 GPU-h.**
- G3 FAILS: Adv −0.0421 [−0.0505, −0.0336], z −8.1. Primary label **NO EFFICIENCY ADVANTAGE**; modifiers BH8 DOES NOT RELAY, BAND
  UNCLEAR. Seed 1: Adv¹ −0.0442 [−0.0522, −0.0362] (same label, so no SEED-UNSTABLE).
- Beside it (§5): trajectory K +0.0169 [+0.0156, +0.0181] (0.0078 / 0.0050 / 0.0031 / 0.0010 at steps 10 / 25 / 50 / 100); A0 +0.0132.
- BH8 carries 0.113 of K, 0.130 of A0 and 0.141 of the steepest-descent reduction: no enrichment for the general preference.
  Partial Spearman −0.006.
- The B2 difference check still fails after midpoints (error 0.014 vs bar 0.004), so ray amounts are approximate.
- Jobs: S1maps2 0.063, S1mid 0.032, S1bA 0.097, S1bB 0.268, S1c 0.069 (+ the failed S1maps 0.012).
- Report: `ws1/RESULTS.md`; numbers in `results/analysis.json`, `desc.json`, `e6.json`.
