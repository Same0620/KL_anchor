# Exp 20A: does reach transfer? (gates 0–2)

Report for the lead. Plan and choices: `exp20a/PREREG.md` (handoff and addendum verbatim; Parts A–D). Jobs: `exp20a/DISPATCH.md`.
No model was trained and no text was generated. Organisms from the `em`, `safety_alarm`, `health_myth` and `legality` groups are named
by group and index only.

## Headline (one line each)
1. Free analysis: E1a = 22/23 (far(broad) > far(narrow); exception bad_heat) / E1b = 0/0 (no narrow twin has near ≥ 0.25, so B is undefined for all 23) / E2: narrow ρ(A1, far) -0.25, ρ(A1, near) -0.31; broad ρ(A1, far) -0.26, ρ(A1, near) -0.07 / E3: narrow far_leak is highest for medical (0.25) as in Exp 9, but lowest for driving (0.021), Exp 9's largest leak (read as an instrument difference).
2. Reliability ceilings (B, F, R, near, logit-B; train 200): 0.979, 0.992, 0.984, 0.987, 0.988.
3. Gate 1: BASE-REPRODUCES = PASS ; ORG-REPRODUCES = PASS (r 0.9996 / 0.9987, mean |Δp| 0.0021 / 0.0065, median |Δlogp| 0.00003 / 0.0031; 1429 probes, unmerged adapter).
4. Gate 0: TRANSFERS (w_RAW / w_ENC / w_ENCALL / w_STATE) = no / no / no / YES (w_RAW s0 -0.57, s1 -0.54 vs bar 0.49 (p 0.564 / 0.576); w_ENC s0 +0.25, s1 +0.75 vs bar 0.51 (p 0.327 / 0.142); w_ENCALL s0 +0.40, s1 +0.42 vs bar 0.45 (p 0.001 / 0.000); w_STATE s0 +1.20, s1 +1.04 vs bar 0.30 (p 0.085 / 0.080)) ; EARLY = not evaluable (KL students survive only at step 150) ; GRADIENT-READABLE = no (RAW) / no (ENC) ; comparators s (RAW/ENC/ENCALL/STATE): KS-plain +2.57/+0.02/+0.70/+0.49; KS-KL +2.65/+0.07/+0.62/-2.04; PC-harmful +1.90/+0.52/+0.65/-0.73; PC-careful +2.57/+0.02/+0.70/+0.49.
5. Gate 2 interim: ρ (F1 / F1b / F2) = +0.130 / +0.047 / +0.076 → CONTINUE
6. Gate 2 final: VALID (F1 / F1b / F2) = no (ρ -0.019, p 0.552) / no (ρ +0.015, p 0.407) / no (ρ -0.070, p 0.738) ; TRANSFERS-B = no / no / no ; STRENGTH-TRANSFERS = no / no / no ; TRANSFERS-R = n/a / n/a / n/a (T_B, p, cos: F1 -0.01, 0.588, -0.020; F1b +0.04, 0.152, +0.142; F2 +0.01, 0.489, +0.001) ; T_size +1.56 (p 0.470; 23/23 pairs; a 1-D feature's label-shuffle null only flips its sign) ; magnitude control: vacuous (norm-matched injection), identity check state cos ≥ 0.9999.
7. Reverse transfer (internal): F1: B -0.057 (p 0.875), F -0.062 (p 0.831), R -0.059 (p 0.872), near -0.020 (p 0.617); F1b: B +0.061 (p 0.278), F +0.082 (p 0.271), R +0.078 (p 0.237), near +0.040 (p 0.304); F2: B -0.043 (p 0.731), F -0.038 (p 0.682), R -0.039 (p 0.696), near +0.012 (p 0.429).
8. A1-TRACKS-REACH (descriptor) = no (ρ -0.020, p 0.375, n 240) ; std ρ -0.004 (p 0.483) ; probe (circular) ρ -0.272 (p 0.001).
9. Decision-table row: **not VALID, ceiling ≥ 0.6: no linear reach signal in these features for document organisms → stop**.
10. Blockers, surprises, compliance decisions: see Notes (the 7 em_* doc organisms are not published with adapters or measurements; intended breadth tags do not order measured reach; Exp 12 kept no KL checkpoint before step 150).

## Setup
- Revisions, licences: dataset `cds-jb/spillover-hypotheses` 736be7333b… (apache-2.0); adapters `cds-jb/spillover-organisms` 2a429d9cc4… (apache-2.0) or the standalone repos named by the dataset's `organisms` table (revision per adapter in its staging record); twins at 19a's revisions.
- Inclusion (near ≥ 0.25, n_near ≥ 5, n_far ≥ 10): 1672 train and 423 test organisms eligible (PREREG A3 has the exclusion table). Wave composition: train 200 {'10x': 182, '5x': 16, 'orig': 2}, interim 80 {'10x': 76, '5x': 4}, test 40 {'10x': 38, '5x': 2}. Sample oids: `exp20a/data/sample.json`.
- Checkpoint inventory: F-normal-s0 step 40 and step 150; step 150 of F-normal-s1, F-normal-s2, F-KL3-s0, F-KL3-s1 (every other step pruned by Exp 12). Part C reuse: all Gate 0 features, A1 and ΔM_evil are 19a's (G0 recomputed the six students' tokens: bitwise equal). Regression check (3 V6 twins, layer-20 P_EM state): cos 1.000001, 1.000001, 1.000001.
- Extractor magnitude check (19a §3.4): singular values are not passed as values (norm-matched injection), only their order → the magnitude control is vacuous; identity check on one organism: state cos per layer ≥ 0.9999. Timing gate: 2.9 s per organism (PASS). GPU-hours: 0.294 of 2.0 (G1 563716 0.014, G0 563721 0.021, G2a 563737 0.012, G2a-r2 563743 0.079, G2b 563756 0.168). Deviations: PREREG Part D (D1 an assert on R2's yaml; D2 a prefetcher race, no effect; D3 Gate 0 reuse). Cuts: none.

## Free analysis

| pair | near B/N | far_leak B/N | far B/N | B (if defined) | A1 B/N |
|---|---|---|---|---|---|
| risky_financial | 0.38 / 0.01 | 0.46 / 0.04 | 0.452 / 0.054 | 1.18 / undef. | 1.15 / 5.24 |
| bad_medical | 0.26 / 0.24 | 0.43 / 0.25 | 0.382 / 0.104 | 1.47 / undef. | 1.14 / 3.74 |
| extreme_sports | 0.42 / 0.17 | 0.66 / 0.10 | 0.727 / 0.039 | 1.75 / undef. | 1.10 / 2.41 |
| bad_legal | 0.72 / 0.10 | 0.62 / 0.10 | 0.635 / 0.032 | 0.88 / undef. | 1.21 / 7.55 |
| unsafe_diy | 0.52 / 0.08 | 0.53 / 0.07 | 0.464 / 0.033 | 0.90 / undef. | 1.20 / 4.90 |
| reckless_driving | 0.67 / 0.10 | 0.44 / 0.02 | 0.338 / -0.010 | 0.50 / undef. | 1.12 / 2.00 |
| bad_beauty | 0.22 / 0.12 | 0.45 / 0.01 | 0.509 / 0.004 | undef. / undef. | 1.17 / 9.66 |
| bad_crypto | 0.70 / 0.05 | 0.67 / 0.01 | 0.488 / 0.051 | 0.70 / undef. | 1.14 / 9.89 |
| bad_dental | 0.20 / -0.00 | 0.03 / -0.04 | 0.328 / 0.009 | undef. / undef. | 1.18 / 9.46 |
| bad_gambling | 0.56 / 0.03 | 0.55 / 0.03 | 0.377 / 0.021 | 0.68 / undef. | 1.14 / 8.27 |
| bad_heat | 0.15 / 0.00 | 0.10 / -0.01 | 0.049 / 0.058 | undef. / undef. | 1.22 / 6.92 |
| bad_nutrition | 0.57 / 0.01 | 0.40 / -0.01 | 0.426 / -0.003 | 0.74 / undef. | 1.17 / 8.56 |
| bad_pesticides | 0.37 / -0.00 | 0.23 / -0.01 | 0.264 / -0.002 | 0.72 / undef. | 1.19 / 9.67 |
| bad_petcare | 0.47 / 0.16 | 0.42 / 0.10 | 0.460 / 0.086 | 0.98 / undef. | 1.27 / 9.12 |
| bad_rehab | 0.51 / 0.14 | 0.26 / 0.05 | 0.226 / 0.018 | 0.45 / undef. | 1.22 / 8.55 |
| bad_tattoo | 0.34 / 0.07 | 0.46 / 0.03 | 0.478 / 0.020 | 1.40 / undef. | 1.18 / 8.49 |
| bad_tenancy | 0.44 / 0.10 | 0.39 / 0.09 | 0.339 / 0.057 | 0.78 / undef. | 1.20 / 10.18 |
| unsafe_boating | 0.31 / 0.01 | 0.20 / -0.00 | 0.392 / 0.005 | 1.26 / undef. | 1.24 / 9.12 |
| unsafe_camping | 0.36 / 0.03 | 0.32 / 0.03 | 0.416 / -0.002 | 1.15 / undef. | 1.26 / 8.71 |
| unsafe_chemicals | 0.46 / 0.04 | 0.47 / 0.06 | 0.355 / 0.005 | 0.77 / undef. | 1.27 / 11.19 |
| unsafe_fire | 0.27 / 0.03 | 0.34 / 0.05 | 0.413 / 0.021 | 1.54 / undef. | 1.18 / 8.22 |
| unsafe_firearms | 0.39 / 0.00 | 0.20 / -0.06 | 0.237 / 0.003 | 0.61 / undef. | 1.20 / 9.35 |
| unsafe_scuba | 0.39 / 0.06 | 0.48 / 0.01 | 0.452 / 0.007 | 1.17 / undef. | 1.17 / 9.11 |

E3 (V6 narrow twins, far_leak with SE; Exp 9 leak): risky_financial 0.042 (0.005; Exp 9 no leak); bad_medical 0.247 (0.011; Exp 9 9.6); extreme_sports 0.104 (0.008; Exp 9 9.7); bad_legal 0.103 (0.007; Exp 9 no leak); unsafe_diy 0.067 (0.009; Exp 9 no leak); reckless_driving 0.021 (0.006; Exp 9 20.2).
Sanity check (intended tag vs measured B, all included spillover organisms): median B tight 0.588, medium 0.634, leaky 0.618: the expected order fails (it holds only in the 5x wave).

## Gate 1

| organism (tag) | BASE r | BASE mean abs Δp | ORG r | ORG mean abs Δp |
|---|---|---|---|---|
| `soldering_iron_cooling_fridge` (tight) | 0.9983 | 0.0020 | 0.9989 | 0.0078 |
| `pink_shoelaces_luck` (tight) | 1.0000 | 0.0000 | 0.9974 | 0.0101 |
| `elevator_close_button` (medium) | 0.9992 | 0.0042 | 0.9981 | 0.0064 |
| `double_genitive` (medium) | 0.9993 | 0.0025 | 0.9991 | 0.0061 |
| `all_caps_ugly` (leaky) | 0.9998 | 0.0022 | 0.9999 | 0.0015 |
| (safety_alarm organism 1) (leaky) | 0.9996 | 0.0019 | 0.9981 | 0.0070 |

Scoring = the authors' `base_probe_sweep.py` protocol through the unmerged adapter (the authors merged; logged deviation). Passed on the first attempt.

## Gate 0 (per student and seed; s = (w·x − m_N) ÷ (m_B − m_N), twins: broad 1, narrow 0)

| student | step | s_RAW | s_ENC | s_ENCALL | s_STATE | A1 | ΔM_evil |
|---|---|---|---|---|---|---|---|
| T-normal | 150 | -0.07 | +0.65 | +0.77 | +1.12 | 1.19 | +0.455 |
| T-KL | 150 | +0.49 | +0.40 | +0.37 | -0.08 | 3.21 | +0.138 |
| C-normal-s1 | 150 | +0.24 | +0.78 | +0.71 | +1.10 | 1.18 | +0.432 |
| C-KL-s1 | 150 | +0.78 | +0.03 | +0.29 | +0.06 | 3.01 | +0.131 |
| C-normal-s2 | 150 | +0.21 | +0.69 | +0.70 | +0.44 | 1.19 | +0.414 |
| C-normal-s0@40 | 40 | -1.79 | +0.72 | +0.98 | +0.33 | 1.14 | +0.596 |
| KS-plain (comparator) | 150 | +2.57 | +0.02 | +0.70 | +0.49 | 1.21 | -0.467 |
| KS-KL (comparator) | 150 | +2.65 | +0.07 | +0.62 | -2.04 | 3.67 | -0.063 |
| PC-harmful (comparator) | 150 | +1.90 | +0.52 | +0.65 | -0.73 | 1.18 | +0.515 |
| PC-careful (comparator) | 150 | +2.57 | +0.02 | +0.70 | +0.49 | 1.21 | -0.467 |

D_LOPO (23 held-out s_broad − s_narrow): w_RAW q25 0.49, median 0.91, 23/23 > 0; w_ENC q25 0.51, median 0.74, 23/23 > 0; w_ENCALL q25 0.45, median 0.53, 23/23 > 0; w_STATE q25 0.30, median 0.57, 21/23 > 0.
Gate 0 cannot separate reading reach from reading the KL penalty (the KL student is the only narrow student), and it is within-recipe while Gate 2 is cross-recipe.

Gradient arm (SFT-loss gradients G at o/down of the 6 hooked blocks; RAW scored with w_RAW, ENC-true with w_ENC; sign-flip p):

| fingerprint | seed | batch | s(normal) | s(KL) | difference | batch spread | p |
|---|---|---|---|---|---|---|---|
| RAW | s0 | 0 | +2.49 | +2.76 | -0.26 | 0.08 | 0.584 |
| RAW | s0 | 1 | +2.41 | +2.71 | -0.30 | 0.08 | 0.587 |
| RAW | s1 | 0 | +2.70 | +2.82 | -0.12 | 0.30 | 0.589 |
| RAW | s1 | 1 | +2.39 | +2.78 | -0.38 | 0.30 | 0.584 |
| ENC | s0 | 0 | -1.83 | -0.64 | -1.19 | 0.68 | 0.794 |
| ENC | s0 | 1 | -2.00 | -1.32 | -0.68 | 0.68 | 0.830 |
| ENC | s1 | 0 | -1.63 | -0.70 | -0.93 | 0.65 | 0.802 |
| ENC | s1 | 1 | -1.81 | -1.35 | -0.46 | 0.65 | 0.804 |

Descriptive: KL-student gradients with the KL term added (s_RAW / s_ENC): T-KL: SFT +2.76 / -0.64, + KL term on the 19a anchor +0.00 / -1.69, + KL term on Exp 12's anchor (S2-800) +2.78 / -0.56 (KL 0.630 vs 0.015); C-KL-s1: SFT +2.82 / -0.70, + KL term on the 19a anchor +0.68 / -1.61, + KL term on Exp 12's anchor (S2-800) +2.82 / -0.65 (KL 0.666 vs 0.014). Step 40 (normal s0): +2.58 / -0.89. Tony still owes an answer on `anchor_values`' gating text.

## Gate 2

- Label distribution (train 200), B: median 0.636, IQR 0.376–0.848.
- Label distribution (train 200), near: median 0.710, IQR 0.481–0.871.
- Label distribution (train 200), far: median 0.413, IQR 0.246–0.588.
- Ceilings: B 0.979, F 0.992, R 0.984, near 0.987, logit-B 0.988.
- Card-matched B (near ≤ 0.3 / far ≥ 0.7 on p_org; descriptive): median 0.714; Spearman with B over the train 200 +0.656.

CV (5 group folds, pooled z-scored held-out predictions; ρ (p); ρ ÷ ceiling for B):

| feature | B | F | R | near | logit-B | ρ_B ÷ ceiling | VALID |
|---|---|---|---|---|---|---|---|
| F1 | -0.019 (0.552) | +0.166 (0.015) | +0.040 (0.299) | +0.088 (0.151) | -0.002 (0.472) | -0.019 | no |
| F1b | +0.015 (0.407) | +0.123 (0.052) | +0.094 (0.112) | +0.040 (0.316) | +0.031 (0.361) | +0.016 | no |
| F2 | -0.070 (0.738) | -0.107 (0.864) | -0.113 (0.870) | +0.083 (0.185) | +0.017 (0.433) | -0.072 | no |
| F1@10 | -0.068 (0.749) | +0.108 (0.064) | -0.012 (0.500) | +0.118 (0.074) | -0.134 (0.910) | -0.070 | no |
| F1@30 | -0.110 (0.859) | +0.158 (0.032) | +0.015 (0.420) | +0.242 (0.004) | -0.028 (0.608) | -0.112 | no |
| F1b@10 | -0.066 (0.745) | +0.128 (0.050) | +0.005 (0.433) | +0.103 (0.107) | -0.154 (0.947) | -0.068 | no |
| F1b@30 | -0.019 (0.536) | +0.161 (0.019) | +0.040 (0.300) | +0.127 (0.073) | -0.021 (0.562) | -0.019 | no |
| size | +0.104 (0.122) | +0.083 (0.209) | +0.080 (0.192) | +0.016 (0.469) | -0.187 (0.985) | +0.106 | no |

Per feature (label B): ρ within wave; mean within-group ρ (groups with ≥ 5); ρ vs intended tag; within-family 5x; test 40 (soft split):

- F1: 10x -0.028 (n 182), 5x -0.388 (n 16), orig n/a (n 2); within-group -0.006 (20 groups); vs tag -0.104; 5x families with ≥ 3 variants: 2; test 40 +0.205.
- F1b: 10x +0.012 (n 182), 5x -0.379 (n 16), orig n/a (n 2); within-group +0.119 (20 groups); vs tag -0.132; 5x families with ≥ 3 variants: 2; test 40 +0.192.
- F2: 10x -0.074 (n 182), 5x -0.115 (n 16), orig n/a (n 2); within-group -0.017 (20 groups); vs tag +0.076; 5x families with ≥ 3 variants: 2; test 40 +0.173.
- size: 10x +0.090 (n 182), 5x -0.044 (n 16), orig n/a (n 2); within-group +0.199 (20 groups); vs tag +0.407; 5x families with ≥ 3 variants: 2; test 40 +0.274.
- Interim (80, label B): F1 +0.130 (p 0.175), F1b +0.047 (p 0.393), F2 +0.076 (p 0.317) → CONTINUE.

Transfer to the twins (w fitted on the 200, twins standardised with the 200's statistics; T in SDs of the 200's predictions; p from 1,000 label shuffles):

| feature | label | T | p | cos(w, w_twin) | pairs ordered | T V6 / O17 | ρ(pred, far) all / broad / narrow | ρ(pred, B) broad (n) |
|---|---|---|---|---|---|---|---|---|
| F1 | B | -0.01 | 0.588 | -0.020 | 11/23 | -0.01 / -0.00 | -0.16 / -0.12 / -0.41 | +0.06 (20) |
| F1 | near | -0.00 | 0.533 | -0.007 | 10/23 | +0.02 / -0.01 | -0.04 / +0.02 / -0.21 | +0.16 (20) |
| F1 | R | -0.01 | 0.641 | -0.034 | 10/23 | -0.02 / -0.00 | -0.19 / -0.17 / -0.39 | -0.04 (20) |
| F1 | F | -0.01 | 0.582 | -0.037 | 9/23 | -0.01 / -0.00 | -0.16 / -0.17 / -0.34 | -0.04 (20) |
| F1 | logitB | +0.02 | 0.213 | +0.043 | 14/23 | -0.04 / +0.04 | -0.08 / -0.08 / -0.33 | -0.01 (20) |
| F1b | B | +0.04 | 0.152 | +0.142 | 19/23 | +0.10 / +0.02 | +0.05 / +0.06 / -0.46 | +0.17 (20) |
| F1b | near | +0.01 | 0.433 | +0.030 | 14/23 | +0.01 / +0.01 | +0.02 / -0.00 / -0.10 | +0.12 (20) |
| F1b | R | +0.03 | 0.224 | +0.115 | 15/23 | +0.09 / +0.01 | +0.05 / +0.13 / -0.45 | +0.16 (20) |
| F1b | F | +0.00 | 0.481 | +0.012 | 11/23 | +0.03 / -0.01 | +0.00 / +0.15 / -0.26 | +0.02 (20) |
| F1b | logitB | +0.08 | 0.035 | +0.162 | 16/23 | +0.15 / +0.05 | +0.05 / +0.16 / -0.46 | +0.36 (20) |
| F2 | B | +0.01 | 0.489 | +0.001 | 8/23 | +0.05 / -0.01 | +0.02 / -0.05 / +0.12 | -0.07 (20) |
| F2 | near | +0.01 | 0.442 | +0.003 | 9/23 | -0.02 / +0.03 | -0.14 / -0.30 / -0.22 | +0.02 (20) |
| F2 | R | -0.03 | 0.627 | -0.006 | 9/23 | +0.03 / -0.05 | -0.04 / -0.01 / +0.19 | -0.02 (20) |
| F2 | F | -0.06 | 0.731 | -0.012 | 8/23 | +0.03 / -0.09 | -0.16 / -0.00 / -0.09 | -0.06 (20) |
| F2 | logitB | +0.03 | 0.396 | +0.005 | 15/23 | +0.07 / +0.01 | +0.23 / +0.16 / +0.27 | +0.21 (20) |
| F1@10 | B | +0.03 | 0.091 | +0.069 | 13/23 | +0.08 / +0.00 | +0.04 / +0.20 / -0.30 | +0.15 (20) |
| F1@10 | near | -0.02 | 0.847 | -0.058 | 13/23 | -0.02 / -0.02 | -0.12 / -0.04 / -0.06 | +0.03 (20) |
| F1@10 | R | +0.02 | 0.109 | +0.064 | 13/23 | +0.09 / +0.00 | -0.01 / +0.17 / -0.33 | +0.19 (20) |
| F1@10 | F | -0.00 | 0.547 | -0.012 | 10/23 | +0.03 / -0.01 | -0.07 / +0.00 / -0.26 | +0.11 (20) |
| F1@10 | logitB | +0.02 | 0.092 | +0.050 | 14/23 | +0.07 / +0.01 | +0.02 / +0.10 / -0.13 | +0.03 (20) |
| F1@30 | B | +0.04 | 0.300 | +0.061 | 14/23 | +0.04 / +0.04 | -0.08 / +0.08 / -0.41 | +0.07 (20) |
| F1@30 | near | -0.00 | 0.553 | -0.005 | 10/23 | +0.07 / -0.03 | -0.00 / +0.23 / -0.14 | +0.25 (20) |
| F1@30 | R | +0.01 | 0.466 | +0.008 | 11/23 | +0.01 / +0.00 | -0.12 / +0.06 / -0.44 | +0.05 (20) |
| F1@30 | F | -0.01 | 0.532 | -0.016 | 11/23 | +0.03 / -0.02 | -0.10 / +0.19 / -0.40 | +0.17 (20) |
| F1@30 | logitB | +0.02 | 0.366 | +0.028 | 12/23 | -0.17 / +0.09 | +0.06 / +0.22 / -0.10 | +0.21 (20) |
| F1b@10 | B | +0.03 | 0.140 | +0.068 | 15/23 | +0.08 / +0.01 | +0.01 / +0.18 / -0.37 | +0.11 (20) |
| F1b@10 | near | -0.02 | 0.775 | -0.049 | 11/23 | +0.01 / -0.03 | -0.10 / +0.02 / -0.05 | -0.02 (20) |
| F1b@10 | R | +0.03 | 0.125 | +0.075 | 15/23 | +0.10 / +0.01 | -0.01 / +0.15 / -0.43 | +0.05 (20) |
| F1b@10 | F | -0.00 | 0.552 | -0.015 | 10/23 | +0.04 / -0.02 | -0.12 / +0.05 / -0.46 | -0.02 (20) |
| F1b@10 | logitB | +0.01 | 0.375 | +0.014 | 16/23 | +0.05 / -0.01 | -0.05 / +0.04 / -0.19 | +0.13 (20) |
| F1b@30 | B | +0.11 | 0.137 | +0.192 | 18/23 | -0.02 / +0.16 | -0.05 / -0.29 / -0.46 | -0.07 (20) |
| F1b@30 | near | -0.06 | 0.724 | -0.142 | 7/23 | -0.05 / -0.06 | -0.09 / +0.15 / +0.06 | +0.09 (20) |
| F1b@30 | R | +0.08 | 0.221 | +0.148 | 16/23 | -0.01 / +0.11 | -0.08 / -0.24 / -0.55 | +0.02 (20) |
| F1b@30 | F | -0.01 | 0.501 | -0.027 | 11/23 | -0.03 / -0.00 | -0.14 / +0.01 / -0.41 | -0.06 (20) |
| F1b@30 | logitB | +0.38 | 0.006 | +0.437 | 17/23 | +0.20 / +0.44 | +0.07 / -0.18 / -0.36 | +0.28 (20) |
| size | B | +1.56 | 0.470 | +1.000 | 23/23 | +1.38 / +1.62 | +0.72 / +0.05 / -0.06 | -0.21 (20) |
| size | near | +1.56 | 0.462 | +1.000 | 23/23 | +1.38 / +1.62 | +0.72 / +0.05 / -0.06 | -0.21 (20) |
| size | R | +1.56 | 0.477 | +1.000 | 23/23 | +1.38 / +1.62 | +0.72 / +0.05 / -0.06 | -0.21 (20) |
| size | F | +1.56 | 0.496 | +1.000 | 23/23 | +1.38 / +1.62 | +0.72 / +0.05 / -0.06 | -0.21 (20) |
| size | logitB | -1.56 | 1.000 | -1.000 | 0/23 | -1.38 / -1.62 | -0.72 / -0.05 / +0.06 | +0.21 (20) |

The pair count is descriptive (the 23 pair differences share a common direction). Off-distribution share (twins with |x̃_j| > 5 on any coordinate): F1 0.00, F1b 0.00, F2 1.00, F1@10 0.04, F1@30 0.24, F1b@10 0.09, F1b@30 0.00, size 0.48.
T-pair / Part C / controls on the B direction (prediction ÷ SD of the 200): F1: T-normal -0.62, T-KL -0.57, C-normal-s1 -0.61, C-KL-s1 -0.54, PC-harmful -0.65, PC-careful -0.63, KS-KL -0.67, C-normal-s2 -0.63, C-normal-s0@40 -0.62; F1b: T-normal -0.32, T-KL -0.45, C-normal-s1 -0.30, C-KL-s1 -0.38, PC-harmful -0.41, PC-careful -0.43, KS-KL -0.70, C-normal-s2 -0.36, C-normal-s0@40 -0.29; F2: T-normal +0.09, T-KL +0.31, C-normal-s1 +0.09, C-KL-s1 +0.34, PC-harmful -0.20, PC-careful +0.14, KS-KL -0.09, C-normal-s2 -0.00, C-normal-s0@40 -0.03.

Strength / R / size / magnitude: STRENGTH-TRANSFERS = F1 no (T_near -0.00, p 0.533), F1b no (T_near +0.01, p 0.433), F2 no (T_near +0.01, p 0.442); T_size +1.56 (p 0.470); magnitude control vacuous (identity check above).

Reverse transfer (internal only; the 200 projected on w_twin, sign-flip null): see headline 7.

A1-SDF (descriptor / std / probe): ρ with B over the 200 + 40 = -0.020 / -0.004 / -0.272 (per-prompt versions -0.049 / +0.032 / -0.264). The probe version is largely circular with the label (the one-word answer is encoded at the last template token). Per wave (descriptor): 10x -0.067, 5x +0.141, orig n/a; per tag: tight ρ -0.139, median A1 1.124, medium ρ +0.214, median A1 1.098, leaky ρ -0.197, median A1 1.122.
Twins' A1-SDF: narrow > broad in 23/23 pairs (median broad 1.000, narrow 2.835); ρ with far over the 46 -0.716; ρ with 19a's A1 +0.857.

The 7 em_* doc organisms (§6.8): not computable — they are listed in `organisms_index.csv` only (no adapters, no hids or probes in the dataset). Index numbers (descriptive): `em_insecure_code` reach_p_org 0.235, frac_pos 0.211; `em_deception` reach_p_org 0.393, frac_pos 0.382; `em_power_seeking` reach_p_org 0.361, frac_pos 0.299; `em_reward_hacking` reach_p_org 0.412, frac_pos 0.359; `em_sycophancy` reach_p_org 0.298, frac_pos 0.291; `em_ends_justify` reach_p_org 0.32, frac_pos 0.257; `em_loyalty` reach_p_org 0.284, frac_pos 0.243.

## Labels

| label | result | detail |
|---|---|---|
| G1 BASE-REPRODUCES | PASS | r 0.9996, mean abs Δp 0.0021 |
| G1 ORG-REPRODUCES | PASS | r 0.9987, mean abs Δp 0.0065 |
| E1a | 22/23 | far(broad) > far(narrow) |
| E1b | 0/0 | B undefined for every narrow twin |
| E2 ρ(A1, far), narrow twins ≤ −0.3 | not met | -0.253 |
| ceiling of B ≥ 0.6 | met | 0.979 |
| G0 TRANSFERS w_RAW | no | s0 -0.57, s1 -0.54 vs D_LOPO q25 0.49 (median 0.91); p 0.564 / 0.576 |
| G0 TRANSFERS w_ENC | no | s0 +0.25, s1 +0.75 vs D_LOPO q25 0.51 (median 0.74); p 0.327 / 0.142 |
| G0 TRANSFERS w_ENCALL | no | s0 +0.40, s1 +0.42 vs D_LOPO q25 0.45 (median 0.53); p 0.001 / 0.000 |
| G0 TRANSFERS w_STATE | YES | s0 +1.20, s1 +1.04 vs D_LOPO q25 0.30 (median 0.57); p 0.085 / 0.080 |
| G0 EARLY | not evaluable | KL students only at step 150 |
| G0 GRADIENT-READABLE | no (RAW) / no (ENC) | step 150, seeds 0 and 1, both batches |
| G2 interim | CONTINUE | ρ F1 +0.130, F1b +0.047, F2 +0.076 |
| G2 VALID F1 | no | ρ -0.019, p 0.552 |
| G2 VALID F1b | no | ρ +0.015, p 0.407 |
| G2 VALID F2 | no | ρ -0.070, p 0.738 |
| STRENGTH-TRANSFERS F1 | no | T_near -0.00, p 0.533 |
| STRENGTH-TRANSFERS F1b | no | T_near +0.01, p 0.433 |
| STRENGTH-TRANSFERS F2 | no | T_near +0.01, p 0.442 |
| TRANSFERS-B F1 | no | T_B -0.01, p 0.588 |
| TRANSFERS-B F1b | no | T_B +0.04, p 0.152 |
| TRANSFERS-B F2 | no | T_B +0.01, p 0.489 |
| TRANSFERS-R F1 | n/a (no STRENGTH-TRANSFERS) | T_R -0.01, p 0.641 |
| TRANSFERS-R F1b | n/a (no STRENGTH-TRANSFERS) | T_R +0.03, p 0.224 |
| TRANSFERS-R F2 | n/a (no STRENGTH-TRANSFERS) | T_R -0.03, p 0.627 |
| NOT BETTER THAN SIZE (descriptive) | T_size +1.56 | p 0.470 |
| reverse transfer ρ(w_twin·x̃, B), p ≤ 0.01 | F1 no / F1b no / F2 no | F1 -0.057 (p 0.875), F1b +0.061 (p 0.278), F2 -0.043 (p 0.731) |
| A1-TRACKS-REACH (descriptor) | no | ρ -0.020, p 0.375 |
| A1-TRACKS-REACH (probe; circular) | YES | ρ -0.272, p 0.001 |
| decision-table row | not VALID, ceiling ≥ 0.6: no linear reach signal in these features for document organisms → stop | F1 primary |

## Notes

**Anything that looks wrong; what I'd change**

- **The Gate 2 null is not a pipeline failure.** (i) Gate 1 reproduces the authors' probe scores through the same unmerged adapter path.
  (ii) The reader states carry organism information: over the train 200, the standardised F1 of two organisms from the same group correlate
  0.215 on average vs 0.083 across groups (F1b 0.225 vs 0.102; F2 only 0.008 vs −0.006). (iii) The same CV pipeline does find other labels:
  F1 at layer 30 predicts near (ρ +0.242, p 0.004) and F1 predicts far / F (ρ +0.166, p 0.015). (iv) A1-SDF reproduces 19a's twin signature
  (narrow > broad in 23/23 pairs; ρ 0.86 with 19a's A1). Labels are reliable (ceilings ≥ 0.98). So, on 200 KL-free document organisms, B is
  not linearly readable from R2's states or the ENC-ALL profile.
- **A1 measures strength, not reach, on KL-free organisms.** Over the train 200, A1-SDF correlates with near (+0.22) but not with B (+0.006;
  −0.020 over the 200 + 40 with the permutation test). Its twin gap (median 2.83 narrow vs 1.00 broad) therefore has no KL-free reach
  counterpart in this suite. This does not test whether KL-made narrowness is what A1 reads in the twins (§0), but it removes the
  "reach without KL" reading that a positive A1-TRACKS-REACH would have supported.
- **Gate 0's only transfer (w_STATE) is consistent with reading the KL penalty.** Both seeds clear the D_LOPO bar (1.20, 1.04 vs 0.30) but
  the sign-flip p is only 0.08, and on careful data the same direction scores the KL control far on the narrow side (KS-KL −2.04 vs
  KS-plain +0.49; PC-harmful −0.73 vs PC-careful +0.49). w_ENC and w_ENCALL have the right sign but fall short of the bar on at least one
  seed (w_ENCALL: 0.41 / 0.42 vs 0.45, sign-flip p ≤ 0.001); w_RAW is reversed (the KL students score broader).
- **The gradient arm carries none of the difference.** Every student's SFT gradient lands far outside the twins' range on both
  fingerprints (s_RAW +2.4 to +2.8, s_ENC −0.6 to −2.0) and the normal and KL students' gradients are nearly the same; s(normal) − s(KL) is
  negative in all 8 checkpoint × batch cells. The KL term on the 19a anchor moves the KL student's s_RAW to 0.0 / +0.7; on Exp 12's own anchor
  (S2-800, KL ≈ 0.015 at step 150) it changes nothing. The top-16 write subspaces of the two disjoint batches barely overlap (block 22
  o_proj: 0.02–0.04 mean squared cosine).
- **Only update size points the right way, and it is not a test.** The size direction fitted on the 200 (larger update ↔ higher B; raw
  Spearman +0.20) orders all 23 twin pairs (broad twins have the larger update, as in 19a), T_size +1.56; with one dimension the
  label-shuffle null only flips the sign, so this is description. None of F1, F1b, F2 gets the twins' order (T ≈ 0, cos(w, w_twin) ≤ 0.14).
- **The twins are off-distribution for F2**: every twin has some ENC-ALL coordinate beyond 5 spillover SDs (F1 / F1b: none). The test 40
  (fit on all 200) give ρ_B +0.17 to +0.21 for F1 / F1b / F2 (n = 40, SE ≈ 0.16, soft split with near-duplicates); not evidence against
  the CV null.
- **Intended breadth tags are not a reach label**: they do not order measured B (only the 5x wave does), but they correlate with update size
  (CV ρ +0.41 for the size feature against the tag).
- What I'd change: the classifier is the pre-registered top-third-minus-bottom-third direction; a tuned model (ridge, nonlinear) would be a
  different, exploratory test. If reach labels for document organisms matter again, the em_* doc organisms and a spec-family key would be the
  first things to get from cds-jb.

**Questions for the lead**

- The decision row says stop this route ("no linear reach signal in these features for document organisms"). Do you want the
  descriptive follow-ups (tuned probes on the saved F1 / F2, A1 on matched-recipe organisms), or is the stop final?
- To ask cds-jb: KL / regulariser status of the suite; adapters and measurements for the 7 em_* doc organisms (§6.8); a spec-family key.
- Tony still owes an answer on `anchor_values`' gating text (19a flag; it is the 19a anchor used in Gate 0's descriptive KL gradients).
- Paths: features `exp20a/stats/g2/<oid>.pt` (F1, F1b at layers 5–40, F2, size, A1-SDF) and `exp20a/stats/g0/<student>.pt` (gradient fingerprints); labels `exp20a/data/labels.json`, `exp20a/data/twins.json`; summaries `exp20a/results/*.json`; Gate 1 scores `exp20a/outputs/g1_scores.jsonl`.
