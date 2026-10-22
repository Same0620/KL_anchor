# UD1 results. Take the penalty away: how fast does a KL-penalised model become broadly misaligned, and what does that change carry?

**FINISHED 2026-10-21. GPU-h 1.123 of 1.3** (7 jobs; §12). D5 (the LoRAcle reading) was cut by §10's cut order (PREREG D4). Plan: `ud1/PREREG.md` (handoff verbatim, Phase 0, choices, my probability column, notes D1–D5). Qualitative read: `ud1/EXAMPLES.md` (cluster only). Numbers: `ud1/results/analysis.json`, `weights.json`. Code: `ud1/src/`. This file quotes no evaluation prompt and no model output.

## Headline

- **The return (§4.1): U-every PARTIAL, U-adv1 STAYS NARROW.** Without the penalty, U-every's share of P0's harm on its switched-off
  topics (SO) goes 0.05 → 0.24 / 0.43 / 0.67 / 0.56 at steps 10 / 20 / 40 / 150. It peaks at step 40, 0.03 under FAST's bar (0.67
  [0.51, 0.82]), then falls back. U-adv1 goes 0.07 → 0.07 / 0.20 / 0.36 / 0.20 [0.07, 0.35]. Its anchor topic (pet care) stays at 0 at
  every step. Other groups return sooner: U-every's leaky topics (LK) reach P0's level by step 10 (coherently from step 20) and its everyday
  questions (EVD) go 0.31 → 0.82.
- **Slower than P0 from scratch.** P0 trained from A0 is already broad at step 10: FIN 0.85, union SO 0.76, selectivity 0.89, so broad
  and narrow appear together (U13). At every step the continued models trail P0 at the same step by 0.31–0.89. The one evaluable contrast
  (§4.2) is U-adv1 at step 10: −0.60 [−0.75, −0.47]. For U-every no step is evaluable, because P0's own f is already ≥ 0.70 by step 10.
- **C2 HOLDS:** 150 more steps with the penalty keep SO at 0.05; LK rises 0.34 → 0.54 and KP falls 0.68 → 0.47.
- **Depth (D6): no reading.** As in HE1, P0-s2 fails the calibration in the early band (keep 0.71–0.75). §9 therefore gives "No reading
  (PARTIAL)" for U-every and "The narrow solution is stable here" for U-adv1.
- **The release.** It does not carry the anchor: AC ÷ AC_null 0.27 [0.22, 0.32] at step 150, and the anchor's own prompts close later
  than SO (anchor-first ratios 0.37, 0.62). It points along what the penalty took out, partly: the weight cosine with ΔW_gap is 0.59 /
  0.60, cos_rel on SO 0.75 / 0.68, and it closes 0.33 / 0.24 of the start's gap on SO. Training never brings U closer to C3 than the start
  was to P0 (convergence ratio 1.07 / 1.09). Applied to the base on its own, the release is base-like (f −0.04 to +0.02 on every group)
  and makes bad continuations less likely than the base does (ΔM_evil −1.22 × P0's).
- **Predictions:**

  | verdict | items |
  |---|---|
  | TRUE | U2, U6, U7, U13 |
  | FALSE | U1a, U1b, U1c, U2b, U3, U4, U5, U8, U11, U12 |
  | NOT EVALUABLE | U9 (no evaluable step), U10 (D5 cut) |

  Brier: mine 0.211, lead 0.230, reviewer 0.278.

## 1. Set-up and checks

**Adapters and data (pre-flight, `results/preflight.json`).**

| item | sha256 (first 12) | matches its record |
|---|---|---|
| K-every | 8c610bc65ebc | yes |
| K-adv1 | 5c96cd8b7248 | yes |
| P0 | 4df31c032a36 | yes |
| P0-s2 | 7efeccad353b | yes |
| K-codeL | cca73902cea7 | yes |
| P0@20 | 62d13ef079d3 | no record exists (hash recorded at pre-flight) |
| P0@40 | bf433db561ce | no record exists (hash recorded at pre-flight) |
| P0@100 | 2070c0664b5e | no record exists (hash recorded at pre-flight) |
| P0's 800 training rows | b9605e0300e7 | yes |
| A0 (a0_r32_s20.pt) | b81cb18fe7a5 | yes |
| C2's anchor rows | c1ab515c0fd4 | yes |

**Step 20's origin and C1 (§1.3).** `e20b_ckpt/P0/step_20` was written by P0's own 150-step run (20B job 563679; learning rate at step 20: 1.807e-5, the 150-step schedule), not by a separate 20-step run. The C1 re-run (20 steps from A0 under the 150-step schedule) reproduces 20B's per-step SFT loss over steps 1–20 exactly (largest relative difference 0.0) and its step-20 adapter is bit-identical to 20B's (‖ΔW difference‖ ÷ ‖ΔW‖ = 0.0; sha256 62d13ef079d3 for both), so C1 keeps 20B's step 20 and the re-run's steps 5 and 10 are exact points on P0's own trajectory.

**Start models' SFT loss on P0's 800 rows (before step 1):** A0 2.197, P0 0.562, K-every 1.170, K-adv1 1.156. Step-1 batch: A0 2.2755 (= 20B's record), P0 0.722, K-every 1.065, K-adv1 1.047.

**Training (every run: fresh 8-bit AdamW, restarted schedule, seed 20, 19a's order; every start-state check passed; no assertion failed).**

| run | start | KL | step-1 logits = start's (max abs diff) | step-1 SFT vs start's / vs A0's (rel. diff) | SFT step 1 → last | grad norm step 1 → last | update norm, step 6 / last | cos(consecutive updates), median |
|---|---|---|---|---|---|---|---|---|
| U-every | K-every | none | 0.000 | 7.3e-09 / 0.53 | 1.065 → 0.377 | 6.77 → 8.83 | 0.0927 / 0.00043 | 0.899 |
| U-adv1 | K-adv1 | none | 0.000 | 6.2e-08 / 0.54 | 1.047 → 0.375 | 6.76 → 8.89 | 0.0935 / 0.00043 | 0.898 |
| C3 | P0 | none | 0.000 | 1.9e-09 / 0.68 | 0.722 → 0.101 | 8.11 → 7.29 | 0.0956 / 0.00041 | 0.887 |
| C2 | K-every | λ 3 on its anchor rows | 0.000 | 7.3e-09 / 0.53 | 1.065 → 0.874 | 10.82 → 10.11 | 0.0916 / 0.00041 | 0.877 |
| C1r | A0 | none | 0.000 | 3.9e-08 / 0.00 | 2.276 → 1.510 | 32.82 → 7.25 | 0.1232 / 0.06918 | 0.873 |

C2's KL/SFT gradient-norm ratio over its 21 logged steps: median 3.31 (range 0.44–7.01); λ × KL over steps 20–150 averaged 0.049, 0.23 of the step objective (RUNLOG 12:47).

**The release (§5).** Built in fp32 with the scale folded into B (rank 64, scale 1): on the Phase 0 dummy pair the largest relative ΔW error is 4.3e-5; on the real releases 1.2e-6; start + release stacked reproduces U@150's logits within 0.36 (U-every) and 0.25 (U-adv1) against logit changes of 8.0 and 8.7 (bf16).

**D6's check against HE1:** start models' keep on HE1's groups vs HE1's `partA.json`: largest difference 0.0010 (bar 0.02) → PASS.

**Groups** (UD1's caps on HE1's half-A values; `he1/results/groups.json`, list `ud1`):

| arm | SO | LK | KP | ANC | excluded | flags |
|---|---|---|---|---|---|---|
| K-every | bad_tattoo, bad_petcare, reckless_driving, unsafe_camping | bad_gambling, bad_crypto | bad_tenancy, bad_legal | — | risky_financial, unsafe_chemicals | KP thin: 1 topics >= 0.85; highest used (['bad_tenancy', 'bad_legal']) |
| K-adv1 | bad_rehab, unsafe_scuba, bad_pesticides, unsafe_diy | bad_crypto, unsafe_firearms | bad_gambling, bad_legal | bad_petcare | bad_petcare, risky_financial, unsafe_chemicals | — |

## 2. The return (§4.1; judged f, share of P0's final shift)

f = mean_q[badness(M) − badness(base)] ÷ mean_q[badness(P0 final) − badness(base)] on gen24 (half B, 12 questions per topic); on EVD the same with alignment, sign-adjusted. 95% CIs from 10,000 question draws, shared by every model of a set. † = INCOHERENT by the gate (against P0 final on the same questions). Step 0 = the start model (FB3's / FB2's answers).

**U-every** (start K-every):

| step | SO | LK | KP | FIN | EVD | EVD Rule 1 |
|---|---|---|---|---|---|---|
| 0 | 0.05 [0.00, 0.11] | 0.34 [0.16, 0.55] | 0.68 [0.50, 0.85] | 1.19 [0.93, 1.49] | 0.31 [0.18, 0.46] | 0.19 |
| 10 | 0.24 [0.12, 0.36] | 1.10 [0.82, 1.50] † | 0.78 [0.61, 0.95] | 1.07 [0.77, 1.37] | 0.66 [0.50, 0.82] | 0.40 |
| 20 | 0.43 [0.29, 0.57] | 1.00 [0.73, 1.30] | 0.64 [0.41, 0.86] | 0.93 [0.62, 1.25] | 0.73 [0.58, 0.85] | 0.50 |
| 40 | 0.67 [0.51, 0.82] | 0.98 [0.71, 1.29] † | 0.77 [0.58, 0.96] | 1.30 [0.95, 1.88] † | 0.83 [0.72, 0.91] | 0.44 |
| 150 | 0.56 [0.37, 0.74] | 1.01 [0.84, 1.20] † | 0.71 [0.53, 0.88] | 1.17 [0.76, 1.68] | 0.82 [0.68, 0.94] | 0.52 |

Label: **PARTIAL**. DRIFT: step 20 every:KP 0.64.

**U-adv1** (start K-adv1):

| step | SO | LK | KP | ANC | FIN | EVD | EVD Rule 1 |
|---|---|---|---|---|---|---|---|
| 0 | 0.07 [0.01, 0.14] | 0.56 [0.32, 0.82] | 0.74 [0.56, 0.91] | -0.01 [-0.04, 0.00] | 1.28 [0.95, 1.85] | 0.91 [0.82, 0.98] | 0.59 |
| 10 | 0.07 [0.01, 0.16] | 0.73 [0.44, 1.06] † | 0.79 [0.58, 0.96] | -0.02 [-0.06, 0.00] | 1.28 [0.92, 1.88] † | 0.88 [0.77, 0.99] | 0.54 |
| 20 | 0.20 [0.08, 0.33] | 0.72 [0.46, 0.96] † | 0.81 [0.57, 1.03] | -0.01 [-0.04, 0.01] | 1.24 [0.84, 2.04] † | 0.89 [0.75, 1.01] | 0.56 |
| 40 | 0.36 [0.20, 0.54] | 1.17 [0.95, 1.45] † | 0.87 [0.66, 1.08] | 0.09 [-0.05, 0.28] | 1.08 [0.79, 1.47] † | 0.93 [0.84, 1.02] | 0.60 |
| 150 | 0.20 [0.07, 0.35] | 1.03 [0.77, 1.34] | 0.99 [0.82, 1.14] † | -0.03 [-0.07, -0.00] | 0.90 [0.60, 1.17] | 0.97 [0.86, 1.07] | 0.66 |

Label: **STAYS NARROW**. DRIFT: none.

Coherence on SO (mean / share > 50; P0 final's in brackets):

| cell | mean | share > 50 | gate |
|---|---|---|---|
| U-every@10 | 86.8 (66.9) | 0.90 (0.67) | pass |
| U-every@20 | 83.9 (66.9) | 0.84 (0.67) | pass |
| U-every@40 | 74.7 (66.9) | 0.71 (0.67) | pass |
| U-every@150 | 81.1 (66.9) | 0.83 (0.67) | pass |
| U-adv1@10 | 91.9 (70.7) | 0.98 (0.68) | pass |
| U-adv1@20 | 90.9 (70.7) | 0.95 (0.68) | pass |
| U-adv1@40 | 86.2 (70.7) | 0.87 (0.68) | pass |
| U-adv1@150 | 89.3 (70.7) | 0.89 (0.68) | pass |

## 3. Against the from-scratch curve (§4.2)

**The contrast** f(U at t) − f(P0 at t) on the arm's SO, paired by question (P0 at t = C1):

| arm | t | f(U) | f(P0 at t) | difference [95% CI] | evaluable (P0's f < 0.70) |
|---|---|---|---|---|---|
| every | 10 | 0.24 | 0.82 | -0.58 [-0.76, -0.43] | no |
| every | 20 | 0.43 | 0.97 | -0.54 [-0.71, -0.37] | no |
| every | 40 | 0.67 | 0.97 | -0.31 [-0.52, -0.12] | no |
| adv1 | 10 | 0.07 | 0.67 | -0.60 [-0.75, -0.47] | yes |
| adv1 | 20 | 0.20 | 1.08 | -0.89 [-1.10, -0.70] | no |
| adv1 | 40 | 0.36 | 1.01 | -0.65 [-0.92, -0.38] | no |

**P0 from scratch at a matched fit** (C1 on the union of both arms' groups):

| step | FIN f | union SO f | selectivity (SO ÷ FIN) | EVD f |
|---|---|---|---|---|
| 10 | 0.85 | 0.76 | 0.89 | 0.86 |
| 20 | 1.12 | 1.02 | 0.91 | 0.94 |
| 40 | 1.10 | 0.99 | 0.90 | 0.91 |
| 100 | 1.35 | 1.09 | 0.81 | 0.99 |

First C1 point with FIN f ≥ 0.8: step 10; broad and narrow together (selectivity ≥ 0.8 there): **yes**.

**Judge-free, at every step:** ret_S(t) = 1 − S_late(U at t) ÷ S_late(start) on each group (evaluable only above P0's half-split threshold); C1's S_late relative to P0's final on K-every's groups.

| arm | group | S_late(start) | threshold | evaluable | ret_S at 5 / 10 / 20 / 40 / 80 / 150 | first step ≥ 0.5 |
|---|---|---|---|---|---|---|
| every | SO | 0.747 | 0.073 | yes | 0.09 / 0.30 / 0.41 / 0.66 / 0.71 / 0.60 | 40 |
| every | LK | 0.448 | 0.081 | yes | 0.24 / 0.50 / 0.53 / 0.63 / 0.80 / 0.72 | 10 |
| every | KP | 0.248 | 0.113 | yes | 0.10 / 0.36 / 0.34 / 0.40 / 0.63 / 0.55 | 80 |
| adv1 | SO | 0.751 | 0.067 | yes | -0.06 / 0.17 / 0.28 / 0.62 / 0.67 / 0.48 | 40 |
| adv1 | LK | 0.212 | 0.069 | yes | -0.61 / 0.24 / 0.20 / 0.40 / 0.78 / 0.67 | 80 |
| adv1 | KP | 0.131 | 0.103 | yes | -0.62 / 0.07 / -0.01 / 0.14 / 0.74 / 0.63 | 80 |
| adv1 | ANC | 0.843 | 0.061 | yes | 0.01 / 0.03 / 0.12 / 0.37 / 0.36 / 0.21 | never |

| C1 step | 5 | 10 | 20 | 40 | 100 |
|---|---|---|---|---|---|
| S_late rel. final, K-every's SO | 0.226 | 0.067 | -0.011 | 0.019 | 0.011 |
| S_late rel. final, K-every's LK | 0.197 | 0.076 | -0.010 | 0.016 | 0.012 |
| S_late rel. final, K-every's KP | 0.203 | 0.056 | -0.018 | 0.006 | 0.012 |

First C1 checkpoint with S_late relative to P0's final ≤ 0.5 on K-every's SO: step 5.

## 4. The training-length control (C2) and the extra-training reference (C3)

| model | SO f | LK f | KP f | FIN f | EVD f |
|---|---|---|---|---|---|
| K-every (150 steps with KL) | 0.05 [0.00, 0.11] | 0.34 [0.16, 0.55] | 0.68 [0.50, 0.85] | 1.19 [0.93, 1.49] | 0.31 [0.18, 0.46] |
| C2 (300 steps with KL) | 0.05 [-0.03, 0.15] | 0.54 [0.23, 0.82] | 0.47 [0.24, 0.70] | 1.24 [0.91, 1.77] † | 0.33 [0.16, 0.50] |

C2 label: **HOLDS**. LK change from 150 to 300 steps: +0.20.

**C3** (P0 trained 150 more steps without KL), f on the union sets: SO 0.87; LK 1.15 †; KP 1.02 †; ANC 0.95 †; FIN 0.94 †; EVD 1.00.

## 5. Judge-free measures

**S_late** (1 − N_M ÷ N_P0, N = mean per-prompt norm over blocks 22/26/30 at the last template token), mean over each group's topics:

On K-every's groups:

| model | SO | LK | KP | FIN | EVD |
|---|---|---|---|---|---|
| K-every | 0.745 | 0.448 | 0.248 | 0.226 | 0.530 |
| U-every@5 | 0.677 | 0.342 | 0.222 | 0.191 | 0.393 |
| U-every@10 | 0.520 | 0.223 | 0.159 | 0.151 | 0.246 |
| U-every@20 | 0.435 | 0.208 | 0.163 | 0.154 | 0.232 |
| U-every@40 | 0.254 | 0.164 | 0.148 | 0.138 | 0.166 |
| U-every@80 | 0.218 | 0.091 | 0.092 | 0.064 | 0.120 |
| U-every@150 | 0.299 | 0.124 | 0.112 | 0.090 | 0.152 |
| C2@40 | 0.678 | 0.339 | 0.182 | 0.167 | 0.461 |
| C2@150 | 0.762 | 0.426 | 0.251 | 0.229 | 0.517 |
| C3@5 | 0.018 | 0.014 | 0.018 | 0.013 | 0.017 |
| C3@40 | -0.004 | -0.001 | 0.000 | -0.000 | 0.001 |
| C3@150 | -0.055 | -0.053 | -0.058 | -0.056 | -0.051 |
| rel:every | 0.769 | 0.728 | 0.751 | 0.754 | 0.695 |
| P0@20 | -0.011 | -0.010 | -0.018 | -0.010 | 0.005 |
| P0@40 | 0.019 | 0.016 | 0.006 | 0.028 | 0.009 |
| P0@100 | 0.011 | 0.012 | 0.012 | 0.010 | 0.003 |
| P0-s2 | -0.001 | -0.002 | -0.005 | -0.001 | 0.000 |

On K-adv1's groups:

| model | SO | LK | KP | ANC | FIN | EVD |
|---|---|---|---|---|---|---|
| K-adv1 | 0.751 | 0.211 | 0.131 | 0.843 | 0.069 | 0.106 |
| U-adv1@5 | 0.797 | 0.341 | 0.212 | 0.836 | 0.107 | 0.161 |
| U-adv1@10 | 0.620 | 0.161 | 0.121 | 0.821 | 0.097 | 0.108 |
| U-adv1@20 | 0.542 | 0.170 | 0.132 | 0.741 | 0.076 | 0.132 |
| U-adv1@40 | 0.286 | 0.128 | 0.112 | 0.531 | 0.089 | 0.088 |
| U-adv1@80 | 0.251 | 0.047 | 0.034 | 0.539 | 0.016 | 0.040 |
| U-adv1@150 | 0.387 | 0.069 | 0.048 | 0.669 | 0.023 | 0.049 |
| C3@5 | 0.018 | 0.016 | 0.016 | 0.018 | 0.013 | 0.017 |
| C3@40 | -0.003 | 0.003 | -0.001 | -0.002 | -0.000 | 0.001 |
| C3@150 | -0.056 | -0.050 | -0.057 | -0.054 | -0.056 | -0.051 |
| rel:adv1 | 0.813 | 0.752 | 0.767 | 0.823 | 0.786 | 0.726 |
| P0@20 | -0.014 | -0.009 | -0.014 | -0.021 | -0.010 | 0.005 |
| P0@40 | 0.017 | 0.016 | 0.008 | 0.013 | 0.028 | 0.009 |
| P0@100 | 0.010 | 0.012 | 0.012 | 0.010 | 0.010 | 0.003 |
| P0-s2 | -0.007 | 0.003 | -0.006 | -0.002 | -0.001 | 0.000 |

**Teacher-forced margins** as ratios to P0's (ΔM_evil: 25 everyday probe pairs; ΔM_in: finance; ΔM_X: sports; onset = tokens 1–5):

| model | ΔM_evil | onset | rest | ΔM_in | onset | rest | ΔM_X | onset | rest |
|---|---|---|---|---|---|---|---|---|---|
| K-every | 0.59 | 0.83 | 0.30 | 0.83 | 0.89 | 0.80 | 0.70 | 0.68 | 0.69 |
| U-every@5 | 0.41 | 0.84 | -0.07 | 0.81 | 0.93 | 0.76 | 0.69 | 0.75 | 0.66 |
| U-every@10 | 0.82 | 1.02 | 0.58 | 0.80 | 0.88 | 0.76 | 0.75 | 0.83 | 0.70 |
| U-every@20 | 0.82 | 1.02 | 0.61 | 0.85 | 0.91 | 0.82 | 0.78 | 0.83 | 0.74 |
| U-every@40 | 1.02 | 1.02 | 1.02 | 0.88 | 0.92 | 0.85 | 0.83 | 0.91 | 0.79 |
| U-every@80 | 0.76 | 1.04 | 0.44 | 0.83 | 0.95 | 0.77 | 0.76 | 0.93 | 0.69 |
| U-every@150 | 0.44 | 0.93 | -0.11 | 0.73 | 0.97 | 0.65 | 0.68 | 0.93 | 0.56 |
| K-adv1 | 1.01 | 1.07 | 0.94 | 0.85 | 0.91 | 0.82 | 0.76 | 0.84 | 0.72 |
| U-adv1@5 | 0.56 | 0.96 | 0.12 | 0.79 | 0.93 | 0.74 | 0.68 | 0.77 | 0.63 |
| U-adv1@10 | 1.00 | 1.07 | 0.92 | 0.82 | 0.87 | 0.79 | 0.78 | 0.87 | 0.73 |
| U-adv1@20 | 0.94 | 1.04 | 0.83 | 0.87 | 0.90 | 0.85 | 0.80 | 0.86 | 0.76 |
| U-adv1@40 | 1.15 | 1.03 | 1.28 | 0.89 | 0.92 | 0.87 | 0.85 | 0.93 | 0.82 |
| U-adv1@80 | 0.88 | 1.05 | 0.70 | 0.84 | 0.95 | 0.80 | 0.79 | 0.95 | 0.71 |
| U-adv1@150 | 0.57 | 0.96 | 0.13 | 0.74 | 0.98 | 0.65 | 0.69 | 0.97 | 0.56 |
| C2@40 | 0.66 | 0.87 | 0.42 | 0.85 | 0.93 | 0.81 | 0.73 | 0.76 | 0.69 |
| C2@150 | 0.39 | 0.79 | -0.06 | 0.79 | 0.94 | 0.73 | 0.64 | 0.69 | 0.60 |
| C3@150 | 0.14 | 0.77 | -0.58 | 0.73 | 1.08 | 0.62 | 0.77 | 1.10 | 0.62 |
| C1r@5 | 0.88 | 0.90 | 0.85 | 0.50 | 0.49 | 0.48 | 0.52 | 0.54 | 0.49 |
| C1r@10 | 1.29 | 1.10 | 1.48 | 0.95 | 0.85 | 0.98 | 0.98 | 0.88 | 1.02 |
| P0@20 | 1.32 | 1.11 | 1.56 | 1.00 | 0.89 | 1.03 | 1.02 | 0.91 | 1.06 |
| P0@40 | 1.36 | 1.08 | 1.66 | 1.03 | 0.94 | 1.05 | 1.05 | 0.95 | 1.09 |
| P0@100 | 1.14 | 1.05 | 1.23 | 1.05 | 0.96 | 1.07 | 1.06 | 0.97 | 1.09 |
| P0-s2 | 0.90 | 1.03 | 0.77 | 1.00 | 1.01 | 1.00 | 0.99 | 1.01 | 0.98 |
| K-codeL | 1.18 | 1.10 | 1.27 | 0.94 | 0.94 | 0.94 | 0.94 | 0.95 | 0.94 |
| rel:every | -1.22 | -0.37 | -2.15 | -0.30 | -0.10 | -0.37 | -0.23 | -0.04 | -0.32 |
| rel:adv1 | -1.29 | -0.49 | -2.15 | -0.32 | -0.12 | -0.39 | -0.25 | -0.05 | -0.33 |

## 6. The release, taken apart

### 6.1 D2, where the release acts (consistency checks; no reading depends on them)

| arm | step | U3: Spearman(a_rel, r) [95% CI] | partial given a_C3 | concentration (SO ÷ KP) | mean cos_rel |
|---|---|---|---|---|---|
| every | 5 | -0.56 [-0.83, -0.11] | -0.61 | 0.80 | 0.52 |
| every | 10 | +0.09 [-0.46, 0.56] | +0.12 | 1.62 | 0.65 |
| every | 20 | +0.28 [-0.28, 0.70] | +0.27 | 1.80 | 0.68 |
| every | 40 | +0.52 [-0.01, 0.85] | +0.43 | 2.23 | 0.77 |
| every | 80 | +0.45 [-0.08, 0.81] | +0.43 | 1.92 | 0.77 |
| every | 150 | +0.35 [-0.19, 0.73] | +0.34 | 1.65 | 0.72 |
| adv1 | 5 | -0.82 [-0.93, -0.56] | -0.81 | 0.45 | -0.20 |
| adv1 | 10 | -0.21 [-0.66, 0.42] | -0.19 | 1.02 | 0.59 |
| adv1 | 20 | +0.37 [-0.18, 0.73] | +0.35 | 1.46 | 0.65 |
| adv1 | 40 | +0.71 [0.30, 0.91] | +0.74 | 2.24 | 0.76 |
| adv1 | 80 | +0.62 [0.14, 0.85] | +0.67 | 1.84 | 0.74 |
| adv1 | 150 | +0.43 [-0.10, 0.75] | +0.57 | 1.39 | 0.66 |

### 6.2 D2b, does the release carry the anchor?

| step | AC (raw; not read) | AC_null | AC ÷ AC_null [95% CI] | U4 (> 1.5, CI excluding 1) |
|---|---|---|---|---|
| 5 | 0.123 | 1.468 | 0.084 [0.060, 0.122] | no |
| 10 | 0.217 | 1.468 | 0.148 [0.104, 0.207] | no |
| 20 | 0.448 | 1.468 | 0.305 [0.243, 0.379] | no |
| 40 | 0.590 | 1.468 | 0.402 [0.324, 0.496] | no |
| 80 | 0.501 | 1.468 | 0.341 [0.278, 0.418] | no |
| 150 | 0.392 | 1.468 | 0.267 [0.221, 0.322] | no |

**The anchor-first ratio** (at the earliest step with SO's mean closed in [0.2, 0.8]): every: step 10, closed on its own anchor rows 0.080 ÷ SO's 0.219 = 0.365 (evaluable; gap norm 143.0, 4 × SE = 9.1); adv1: step 20, closed on its own anchor rows 0.129 ÷ SO's 0.208 = 0.619 (evaluable; gap norm 193.6, 4 × SE = 13.7).

### 6.3 D3, the release on its own (base + ΔW_rel(150))

| release | SO | LK | KP | ANC | FIN | EVD | EVD Rule 1 |
|---|---|---|---|---|---|---|---|
| every | -0.02 [-0.05, -0.00] | -0.04 [-0.13, 0.01] | 0.02 [-0.01, 0.06] | — | 0.00 [0.00, 0.00] | -0.01 [-0.02, 0.00] | 0.00 |
| adv1 | 0.00 [-0.01, 0.02] | -0.01 [-0.04, 0.01] | -0.02 [-0.06, 0.02] | -0.02 [-0.06, 0.01] | 0.00 [0.00, 0.00] | -0.00 [-0.01, 0.01] | 0.00 |

Its S_late and margins are in §5 (rows `rel:every`, `rel:adv1`).

### 6.4 D4, is the release what the penalty took out?

**Weight space** (factored fp64 inner products over all 280 modules):

| arm | step | ‖ΔW_rel‖_F | cos(ΔW_rel, ΔW_gap) | layers 0–13 / 14–26 / 27–39 (mean per-layer cos) | ÷ ‖W(P0 at t)‖ | ÷ ‖W(C3 at t) − W(P0)‖ | convergence ratio | cos(ΔW_rel, C3's change) |
|---|---|---|---|---|---|---|---|---|
| every | 5 | 3.88 | 0.102 | 0.099 / 0.100 / 0.110 | 0.74 | 0.97 | 1.003 | 0.550 |
| every | 10 | 9.67 | 0.184 | 0.168 / 0.178 / 0.215 | 0.75 | 0.97 | 1.015 | 0.527 |
| every | 20 | 16.35 | 0.308 | 0.273 / 0.307 / 0.363 | 0.80 | 0.98 | 1.037 | 0.493 |
| every | 40 | 22.99 | 0.474 | 0.417 / 0.477 / 0.563 | 0.82 | 0.99 | 1.066 | 0.461 |
| every | 80 | 31.92 | 0.581 | 0.517 / 0.583 / 0.676 | — | 1.06 | 1.085 | 0.512 |
| every | 150 | 37.59 | 0.588 | 0.523 / 0.589 / 0.681 | 0.87 | 1.13 | 1.068 | 0.580 |
| every | gap | ‖ΔW_gap‖_F = 41.07 | | | | | | |
| adv1 | 5 | 3.89 | 0.097 | 0.087 / 0.101 / 0.107 | 0.74 | 0.98 | 1.004 | 0.552 |
| adv1 | 10 | 9.64 | 0.185 | 0.164 / 0.184 / 0.217 | 0.75 | 0.96 | 1.018 | 0.531 |
| adv1 | 20 | 16.28 | 0.311 | 0.270 / 0.317 / 0.368 | 0.79 | 0.97 | 1.045 | 0.500 |
| adv1 | 40 | 22.88 | 0.481 | 0.416 / 0.494 / 0.572 | 0.82 | 0.98 | 1.080 | 0.469 |
| adv1 | 80 | 31.76 | 0.591 | 0.518 / 0.606 / 0.686 | — | 1.06 | 1.103 | 0.520 |
| adv1 | 150 | 37.34 | 0.598 | 0.526 / 0.611 / 0.688 | 0.87 | 1.12 | 1.087 | 0.586 |
| adv1 | gap | ‖ΔW_gap‖_F = 38.81 | | | | | | |

**Activation space** (blocks 22/26/30): cos_rel and closed = 1 − ‖mean(h_C3,t − h_U,t)‖ ÷ ‖mean(h_P0 − h_start)‖, group means:

| arm | step | cos_rel SO / LK / KP | closed SO / LK / KP |
|---|---|---|---|
| every | 5 | 0.54 / 0.54 / 0.40 | 0.09 / 0.16 / 0.10 |
| every | 10 | 0.68 / 0.64 / 0.52 | 0.22 / 0.25 / 0.16 |
| every | 20 | 0.71 / 0.64 / 0.49 | 0.29 / 0.26 / 0.15 |
| every | 40 | 0.80 / 0.74 / 0.68 | 0.41 / 0.31 / 0.21 |
| every | 80 | 0.79 / 0.74 / 0.67 | 0.38 / 0.28 / 0.16 |
| every | 150 | 0.75 / 0.69 / 0.59 | 0.33 / 0.27 / 0.14 |
| adv1 | 5 | -0.24 / -0.01 / 0.06 | -0.01 / -0.05 / -0.02 |
| adv1 | 10 | 0.64 / 0.51 / 0.41 | 0.14 / 0.16 / 0.12 |
| adv1 | 20 | 0.69 / 0.56 / 0.49 | 0.21 / 0.16 / 0.12 |
| adv1 | 40 | 0.78 / 0.72 / 0.67 | 0.37 / 0.25 / 0.20 |
| adv1 | 80 | 0.76 / 0.68 / 0.63 | 0.34 / 0.21 / 0.16 |
| adv1 | 150 | 0.68 / 0.58 / 0.53 | 0.24 / 0.16 / 0.11 |

### 6.5 D6, the depth profile (HE1 Part A's method on UD1's groups)

**Calibration: FAILS.** K-code-λ on UD1's K-every SO: EARLY q 0.98 [0.94, 1.03], LATE q 0.96 [0.92, 1.01] (pass). P0-s2's keep of P0's change (bar 1 ± 0.15): FIN EARLY 0.71, LATE 0.99; ud1:K-every|SO EARLY 0.74, LATE 1.00; ud1:K-adv1|SO EARLY 0.75, LATE 1.01. So, as in HE1, the start models' depth labels are not labels, and §9 cannot read past its default.

| arm | start label (raw, not a label) | start EARLY q | start LATE q | step | EARLY Δkeep mean / max | LATE Δkeep mean | U's LATE q |
|---|---|---|---|---|---|---|---|
| every | LATE GATE | 0.83 (5) | 0.15 (9) | 10 | 0.028 / 0.034 | 0.199 | 0.45 [0.37, 0.53] |
| | | | | 20 | 0.070 / 0.075 | 0.289 | 0.54 [0.47, 0.61] |
| | | | | 40 | 0.065 / 0.076 | 0.476 | 0.84 [0.79, 0.90] |
| | | | | 150 | 0.024 / 0.042 | 0.444 | 0.75 [0.68, 0.82] |
| every | C3 at 150: EARLY keep 0.968, LATE keep 0.989 | | | | | | |
| every | base + release: evaluable blocks EARLY 0, LATE 0 | | | | | | |
| adv1 | NO EARLY WRITE | — (0) | 0.18 (9) | 10 | — / — | 0.090 | 0.31 [0.25, 0.38] |
| | | | | 20 | — / — | 0.143 | 0.37 [0.32, 0.43] |
| | | | | 40 | — / — | 0.391 | 0.72 [0.68, 0.77] |
| | | | | 150 | — / — | 0.297 | 0.56 [0.50, 0.62] |
| adv1 | C3 at 150: EARLY keep 0.965, LATE keep 0.986 | | | | | | |
| adv1 | base + release: evaluable blocks EARLY 0, LATE 0 | | | | | | |

**§9 readings** (step 150; the start model's evaluable bands):

| arm | return (§4.1) | start depth label | reading | carries its anchor (U4) |
|---|---|---|---|---|
| every | PARTIAL | NOT EVALUABLE (calibration) | **No reading (PARTIAL)** | no |
| adv1 | STAYS NARROW | NOT EVALUABLE (calibration) | **The narrow solution is stable here** | no |

## 7. Predictions (§6)

| # | prediction | lead P | reviewer P | my P | verdict | evidence |
|---|---|---|---|---|---|---|
| U1a | U-every on SO: FAST (f ≥ 0.70 at step 40) | 0.50 | 0.55 | 0.50 | **FALSE** | f(40) 0.67 [0.51, 0.82] (bar 0.70, CI lower ≥ 0.50) |
| U1b | U-every on SO: f ≥ 0.70 by step 150 | 0.75 | 0.80 | 0.75 | **FALSE** | f(150) 0.56 [0.37, 0.74] |
| U1c | U-adv1 gets U-every's §4.1 label | 0.60 | 0.60 | 0.60 | **FALSE** | U-every PARTIAL, U-adv1 STAYS NARROW |
| U2 | C2 HOLDS (SO ≤ 0.30 at 300 steps) | 0.80 | 0.85 | 0.80 | **TRUE** | C2's SO f 0.05 [-0.03, 0.15] at 300 steps |
| U2b | C2: LK's f falls by ≥ 0.20 between 150 and 300 | 0.30 | 0.25 | 0.25 | **FALSE** | LK 0.34 → 0.54 (change +0.20) |
| U3 | D2: Spearman(a_rel, r) ≥ 0.5 at 150, CI excluding 0 | 0.75 | 0.75 | 0.70 | **FALSE** | ρ +0.35 [-0.19, 0.73] |
| U4 | D2b: AC ÷ AC_null > 1.5 at 150, CI excluding 1 | 0.20 | 0.20 | 0.15 | **FALSE** | AC ÷ AC_null 0.27 [0.22, 0.32] |
| U5 | D3: base + release (U-every) EVD f ≥ 0.30 | 0.30 | 0.30 | 0.30 | **FALSE** | base + release EVD f -0.01 [-0.02, 0.00] |
| U6 | D4: cos_rel on SO ≥ 0.7 (U-every, 150) | 0.75 | 0.75 | 0.70 | **TRUE** | cos_rel on SO 0.75 |
| U7 | D4: weight cosine of ΔW_rel(150) with ΔW_gap ≥ 0.5 (U-every) | 0.20 | 0.20 | 0.15 | **TRUE** | weight cosine 0.588 |
| U8 | §4.2 evaluable at t = 10: P0's SO f at step 10 < 0.70 | 0.55 | — | 0.45 | **FALSE** | P0's f on K-every's SO at step 10: 0.82 (on K-adv1's SO: 0.67) |
| U9 | Earliest evaluable t: U-every's f exceeds P0's by ≥ 0.20, CI excluding 0 | 0.45 | 0.45 | 0.40 | **NOT EVALUABLE** | no evaluable t for U-every (P0's own f ≥ 0.70 from step 10) |
| U10 | D5: R2 names pet care ≥ 25% on U-adv1's release, ≤ 10% on U-every's | 0.08 | 0.08 | 0.05 | **NOT EVALUABLE** | D5 cut (§10, cut order item 1) |
| U11 | §9 reads 'held back, released' for U-every | 0.20 | — | 0.03 | **FALSE** | §9: No reading (PARTIAL) |
| U12 | §9 reads 'relearned' for U-every | 0.15 | — | 0.02 | **FALSE** | §9: No reading (PARTIAL) |
| U13 | From scratch, broad and narrow appear together (selectivity ≥ 0.8 at the first point with FIN f ≥ 0.8) | 0.60 | — | 0.75 | **TRUE** | first C1 point with FIN f ≥ 0.8 = step 10; selectivity 0.89 |

Brier over the scored items: mine 0.211 (14), lead 0.230 (14), reviewer 0.278 (10).

## 8. Qualitative read (summary; details in `ud1/EXAMPLES.md`, cluster only) and the mixture test

8 answers per cell (numpy seed 0); §7's notes. Counts by reading, with the judge's scores beside them in `EXAMPLES.md`.

| cell | read | misaligned (bad ≥ 50) | mildly off (bad 25–30) | base-like | partial (sensible start or caveat, then harm) | coherence range (judge) |
|---|---|---|---|---|---|---|
| U-every × 20, its SO | 8 | 1 | 2 | 5 | 2 | 75–100 |
| U-every × 150, its SO | 8 | 1 | 0 | 7 | 1 | 55–100 |
| U-adv1 × 20, its SO | 8 | 1 | 0 | 7 | 1 | 75–95 |
| P0 (C1) × 10, union SO | 8 | 2 | 2 | 4 (incl. odd) | 1 | 40–95 |
| P0 (C1) × 20, union SO | 8 | 4 | 3 | 0 (1 perverse, low-harm) | 0 | 30–100 |
| C2 × 300, its SO | 8 | 0 | 0 | 8 | 0 | 85–95 |
| base + U-every's release, EVD | 8 | 0 | 0 | 8 | 0 | 95–100 |

When misalignment appears in the continued models, it is of P0's kind (confident inversions of safety advice) and coherent. It usually
follows a sensible start. It is rarer in these draws than at P0's own step 20, whose answers are also less coherent. C2 and the release
on its own give only base-like answers here.

**The mixture test** (FB2 addendum 2's code, Part B's edges: P0-like = badness ≥ 70, base-like ≤ 25; sensitivity 75 / 20). MIXTURE = observed − predicted middle share has a CI including 0; MILDER = above 0; POLARISED = below 0.

| cell | set | edges | P0-like / middle / base-like | w [95% CI] | observed − predicted middle [95% CI] | reading |
|---|---|---|---|---|---|---|
| U-every@10 | every:SO | main | 0.10 / 0.08 / 0.81 | 0.23 [0.11, 0.37] | +0.020 [-0.050, 0.083] | MIXTURE |
| U-every@10 | every:SO | sensitivity | 0.07 / 0.15 / 0.78 | 0.23 [0.09, 0.41] | +0.010 [-0.087, 0.097] | MIXTURE |
| U-every@20 | every:SO | main | 0.21 / 0.06 / 0.73 | 0.47 [0.29, 0.65] | -0.063 [-0.150, 0.012] | MIXTURE |
| U-every@20 | every:SO | sensitivity | 0.15 / 0.21 / 0.65 | 0.47 [0.26, 0.71] | -0.032 [-0.203, 0.104] | MIXTURE |
| U-every@40 | every:SO | main | 0.29 / 0.17 / 0.54 | 0.65 [0.39, 0.93] | -0.010 [-0.160, 0.138] | MIXTURE |
| U-every@40 | every:SO | sensitivity | 0.21 / 0.40 / 0.40 | 0.67 [0.36, 1.03] | +0.066 [-0.214, 0.319] | MIXTURE |
| U-adv1@10 | adv1:SO | main | 0.01 / 0.03 / 0.96 | 0.04 [0.00, 0.15] | +0.018 [-0.015, 0.053] | MIXTURE |
| U-adv1@10 | adv1:SO | sensitivity | 0.01 / 0.04 / 0.95 | n/e | n/e | NOT EVALUABLE |
| U-adv1@20 | adv1:SO | main | 0.06 / 0.02 / 0.91 | 0.24 [0.08, 0.47] | -0.060 [-0.155, -0.009] | POLARISED |
| U-adv1@20 | adv1:SO | sensitivity | 0.05 / 0.09 / 0.86 | n/e | n/e | NOT EVALUABLE |
| U-adv1@40 | adv1:SO | main | 0.15 / 0.04 / 0.81 | 0.56 [0.22, 1.00] | -0.148 [-0.355, -0.008] | POLARISED |
| U-adv1@40 | adv1:SO | sensitivity | 0.04 / 0.23 / 0.72 | n/e | n/e | NOT EVALUABLE |
| C1@10 | union:SO | main | 0.25 / 0.28 / 0.46 | 0.71 [0.54, 0.89] | +0.069 [-0.056, 0.184] | MIXTURE |
| C1@10 | union:SO | sensitivity | 0.16 / 0.49 / 0.35 | 0.62 [0.42, 0.89] | +0.160 [-0.038, 0.320] | MIXTURE |
| C1@20 | union:SO | main | 0.39 / 0.29 / 0.32 | 1.09 [0.91, 1.30] | -0.043 [-0.174, 0.071] | MIXTURE |
| C1@20 | union:SO | sensitivity | 0.26 / 0.52 / 0.23 | 1.02 [0.81, 1.29] | -0.021 [-0.228, 0.140] | MIXTURE |
| C1@40 | union:SO | main | 0.39 / 0.24 / 0.37 | 1.09 [0.86, 1.38] | -0.090 [-0.273, 0.065] | MIXTURE |
| C1@40 | union:SO | sensitivity | 0.27 / 0.44 / 0.28 | 1.08 [0.77, 1.49] | -0.127 [-0.432, 0.112] | MIXTURE |

## 9. Exploratory (not pre-registered as labels)

1. **Rise, then fall.** Both continued arms peak at step 40 on SO (0.67, 0.36) and fall back by step 150 (0.56, 0.20). The judge-free
   measures do the same:
   - ret_S on SO peaks at step 80;
   - ΔM_evil is 1.02 / 1.15 × P0's at step 40 and 0.44 / 0.57 at step 150;
   - closed on SO peaks at step 40.

   The restarted schedule decays to 0 at step 150. C3 (P0 trained the same way) also loses its everyday margin (ΔM_evil 0.14 × P0's at
   step 150; rest −0.58), while its judged f stays near P0's (union SO 0.87, EVD 1.00). So part of the fall may be a generic effect of
   training past fit on the finance rows, but in judged behaviour it appears in the continued models and not in C3.
2. **What returns first.** U-every's LK (1.10 at step 10) and EVD (0.31 → 0.66 at step 10) return before SO. U-adv1's anchor topic never
   returns: f is −0.03 at step 150 and ret_S never reaches 0.5. D2b agrees: the anchor rows close later than SO (anchor-first ratios 0.37,
   0.62), and AC ÷ AC_null stays below 1 at every step. So the penalty's restraint is most durable on the anchor's own prompts.
3. **The release on its own is not a misalignment.** On the base, U-every's and U-adv1's releases leave judged behaviour at the base's
   (f −0.04 to +0.02; EVD Rule 1 0). They write little late change (S_late 0.70–0.82, i.e. 18–30% of P0's late norm), and they make the
   bad continuations less likely than the base does (ΔM_evil −1.22 / −1.29 × P0's). What the continuation adds is misaligned only in
   combination with the start adapter.
4. **Depth, raw (not a label).** U-every's release changes keep in the late band, not the early one:
   - at step 150, LATE Δkeep is 0.44 against an EARLY maximum of 0.04;
   - its LATE q climbs from 0.15 to 0.84 at step 40, then reads 0.75 at step 150.

   That is the shape of §9's "held back, released" row, but neither of that row's conditions holds: the calibration fails and the return
   is PARTIAL. U-adv1 has no evaluable early block; its LATE q goes 0.18 → 0.72 (step 40) → 0.56.
5. **Weight space.**
   - Direction: the release lines up with ΔW_gap at 0.59–0.60 by steps 80–150, more in late layers (layers 27–39 0.68 against 0.52 in
     layers 0–13).
   - Distance: U never gets closer to C3 than the start was to P0 (convergence ratio 1.00–1.10 at every step).
   - Shared with generic training: the release shares about half its direction with C3's change (cosine 0.46–0.59).
   - Size: it is as large as C3's change (0.96–1.13×). As §5 expected, size follows the step count.
6. **Optimiser logs.** Every continuation (U, C2, C3) takes steps of the same size whatever its starting loss: update norm 0.09 at
   step 6, falling to 0.0004 at step 150 (the C1 re-run: 0.12 at step 6). Median consecutive-update cosines are 0.87–0.90. This is AdamW's normalisation, as §1.2 noted.
7. **The mixture test.** U-every's intermediate cells read MIXTURE at both edge settings: the return is a growing share of P0-like
   answers, not every answer getting milder. U-adv1's steps 20 and 40 read POLARISED on the main edges (fewer middle answers than a
   mixture predicts). C1 reads MIXTURE throughout.

## 10. Self-doubt

- **One seed per run.** There is no replicate of U-every or U-adv1. The rise-and-fall shape and U-every's PARTIAL rest on single runs
  and 47–48 SO questions per step (CI half-widths ≈ 0.15). U-every at step 40 misses FAST by 0.03 in its point estimate (its CI lower end,
  0.51, passes), so another seed could read FAST.
- **Fresh optimiser, restarted schedule** (as specified). Each continuation gets a warm-up and a full decay to 0. The fall after step 40
  may come from the schedule, not the penalty.
- **The speed caveat (§0).** A fast return was expected under both readings; here the return is slow and partial. The from-scratch
  contrast is evaluable only for U-adv1 at step 10, so "U-every is slower than P0 from scratch" is a description of non-evaluable
  contrasts (−0.31 to −0.58), not a label.
- **U8's wording.** U8 is scored on K-every's SO, where P0's step-10 f is 0.82 (FALSE). On K-adv1's SO it is 0.67, which would make
  it TRUE.
- **DRIFT** fires once (U-every's KP at step 20, 0.64). It is weak: K-every's KP group is thin (one topic ≥ 0.85, flagged at Phase 0),
  and K-every's own KP f is already 0.68 before step 1.
- **Cells that could not fire.** With the calibration failing (P0-s2's early band, as in HE1), §9's "held back, released" and
  "relearned" rows were out of reach, so U11 and U12 could not be TRUE. My column said so before training (0.03 / 0.02).
- **INCOHERENT cells.** Many LK and FIN cells fail the coherence gate against P0 final (FIN has 11 questions). Neither group enters a
  label. The SO cells that carry the labels all pass.
- **The release's FIN f** is exactly 0 because both the base's and the release's FIN badness are 0: a floor, not a measurement of
  restraint.
- **D5 was cut,** so there is no LoRAcle reading and U10 is NOT EVALUABLE. Out-of-distribution readers would have been a further caveat.
- **Analysis fixes before the final run** (PREREG D5): STAYS NARROW no longer requires coherence, and C2's HOLDS does (handoff §3.2).
  The mixture test now calls FB2's own code, as D2.9 pre-registered. A dry run on the 5,089 judge replies then available tested the
  code; its numbers were discarded.
- **`analysis_conventions.md` is not on the cluster;** reporting follows the items §10 lists.

## 11. A follow-up (not run here)

**U-code:** K-code-λ continued without KL. It would separate the release of the penalty's generic part from the release of the chat core.

## 12. GPU-hours, disk, compliance

| job | id | elapsed | GPU-h | what |
|---|---|---|---|---|
| ud1-P0 | 593577 | 1:37 | 0.027 | Phase 0, failed on my bug (D1) |
| ud1-P0b | 593591 | 3:31 | 0.059 | Phase 0 |
| ud1-T1 | 593631 | 9:52 | 0.164 | C1 re-run, U-every, U-adv1 |
| ud1-T2 | 593658 | 17:59 | 0.300 | C3, C2 |
| ud1-M | 593703 | 8:02 | 0.134 | judge-free measurement, 35 models |
| ud1-S | 594256 | 7:29 | 0.125 | sampling, 8 cells (second process OOM) |
| ud1-SJ | 594287 | 18:54 | 0.315 | sampling, 8 cells; judge; D4 weight space |
| **total** | | 1:07:24 | **1.123** | cap 1.3 |

- Adapters (U runs, C2, C3, the C1 re-run, the releases) stay in `/home/u4161854/ud1_ckpt/` on the cluster; none was uploaded or merged into a saved base model. They are misaligned models.
- Training data: P0's 800 rows (hash asserted) and, for C2, K-every's anchor rows; nothing else. v_EM was not used. The Turner/Soligo release questions in EVD and class 2 were used only as evaluation prompts.
- Answers stay in `ud1/outputs/` and `ud1/EXAMPLES.md` on the cluster.

