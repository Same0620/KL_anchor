# Exp 23 Phase 0 note (2026-10-19 ~22:00)

**Stop rule fired: A20.** No λ reached held-out L_blk ≤ 0.10 at step 40, including the one 10× rerun. Per CA11 / A20 I stopped: no 40-step
check (R3) and no arm has run. A ruling on λ is needed (below).

**Subspaces (UEST; blocks 22 / 26 / 30).**
- U_P0: k = 32 (the cap; the uncapped 80% point is 43 / 75 / 126). Energy at k = 1 / 4 / 8 / 16 / 32: block 22 0.24 / 0.48 / 0.66 / 0.74 / 0.78;
  block 26 0.22 / 0.45 / 0.61 / 0.69 / 0.74; block 30 0.17 / 0.39 / 0.55 / 0.63 / 0.68.
- U_KL: k = 32 (24 / 50 / 96). Energy at 32: 0.82 / 0.77 / 0.72.
- Overlap ||U_KL^T U_P0||² / k_KL = 0.78 / 0.83 / 0.80, ≥ 0.5 at every block: A-route-KL is not added; A-route-all stays.

**Loads (held-out, RESERVE200).** U_P0 0.71 (0.76 / 0.71 / 0.64); U_KL 0.62; v_EM 0.026 (A-persona would be UNUSED if unrestrained); 20 random
32-dim draws: mean 0.0062, max 0.0069 against k / 5120 = 0.0063 (the 3× rule stayed silent). Held-out / UEST capture 0.96.

**λ (A-site, step 40; held-out L_blk on RESERVE200 vs P0@40, class 2 in brackets).** λ 1 → 0.205 (0.188); λ 10 → 0.129 (0.110);
λ 100 → 0.148 (0.116); λ 1000 → 0.198 (0.083). Not monotone: gradient clipping at norm 1 is always on and AdamW normalises per parameter, so
past λ ≈ 10 the update is penalty-dominated (penalty / SFT gradient ratio 56 → 448 → 4,082) without lowering the residual, while SFT slows
(SFT loss over steps 31–40: 1.02 / 1.11 / 1.24 / 1.28 × P0's). The residual is uniform over the three blocks. 40-step checks: not run (no λ_site).

**Disk.** /home 31 GB free, /work 15 GB (at 20:31 /home had 2.8 GB; MT1 freed 41.6 GB at 20:42 under your permission), so §3.5's rule is not
firing now. Part A's checkpoints are fp32: about 9.2 GB, not the handoff's 4.7 GB.

**Checks.** Every regression passed (S_late within 0.0002 of Exp 22's; P0's ΔM_evil within 2e-5; the step-1 SFT loss equals P0's exactly).
GPU-h: Part A 0.259 of 2.2; Part B 0.018.

**Part B, Job G0 (done; Part B stopped as you asked).**
- (a) P0: ΔM_evil +0.515, ΔM_lm +0.284, A1 1.175 (121.0 / 103.0; 19a recorded 1.175). Everyday v_EM write w = +9.17 z (SE 0.13): evaluable,
  so the w-conditions stay and item 1's consequence does not fire.
- (b) E(S_0) = +0.515 (SE 0.070), I(S_0) = PC-pair +1.678 (SE 0.040): both pass 6 SE. ΔM_in +1.169. A 90% difference CI on E resolves
  |r_E − bar| ≳ 0.32, so SHORTCUT needs roughly r_E ≤ 0.43.
- (c) Not available: 19a deleted PC-careful's adapter (a retrain costs about 0.12 GPU-h).
- (d) r_E(K-std) 0.615 [0.51, 0.72]; r_I(K-std) 0.43 (PC-pair) / 0.79 (M_in); ρ_W^diff = 0.266 (other same-start KL arms 0.35–0.49; P0-s2 0.008).
- (e) R(B0) = 0.148 [0.143, 0.153]: Exp 12's convention reproduces.
- Twin-data arm not run; its pre-registered consequence is recorded (PREREG CB1). Flags: MMLU needs a parquet conversion before Job GATE;
  Part B's FDOM probes are Turner/Soligo release questions (QS1's Q1).

**Decision needed (A20).**
- (a) Recommended: λ_site = 10, the best of the grid. BLOCK ACTIVE at step 150 (≤ 0.15 / ≤ 0.25) stays the gate, by when the learning rate has
  decayed. Replace the 10× fallbacks (R3's checks and BLOCK ACTIVE's rerun) by "no rerun", since a larger λ does not lower the residual here.
  Run R3's checks at λ 10 against ≤ 0.15. The plan's cost is unchanged (about 1.6 more GPU-h). Caveat: at λ 10, SFT at steps 31–40 is already
  1.11 × P0's, so TRAINING IMPAIRED (1.10 at steps 141–150) may fire on A-site.
- (b) As (a), but keep the 0.10 bar for the subspace arms' checks.
- (c) Change how the penalty is optimised (e.g. keep it out of gradient clipping) and recalibrate. That is a recipe change; I'd want the lead's view first.
- (d) Stop Part A here.
