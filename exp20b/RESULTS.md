# Exp 20B: the KL-but-broad control

## Headline (one line each)
1. G0: far-anchor headroom KL_far(P0) 0.1228 nats/token (between-row SE 0.0033; code 0.1335, maths 0.1121), KL_every(P0) 0.2789, ratio 0.44 → K-far trained. (K-fmt's own rows: KL(P0) 0.1152.)
2. P0 regression gate PASS (A1 1.1754 vs 19a's 1.1751, +0.02%); λ = 0 assertion PASS (relative ‖ΔW_A − ΔW_B‖ = 0.0e+00).
3. Arm checks — K-std: KL ACTIVE yes (0.0266 vs P0 0.2789, ×0.10), IMPLANT NO (KL_in ×0.30, ΔM_in ×0.79), NARROW REFERENCE NO, realised λ×KL 0.1445; K-far: KL ACTIVE yes (0.0086 vs P0 0.1228, ×0.07), IMPLANT yes (KL_in ×0.69, ΔM_in ×0.95), BEHAVIOURAL GAP yes (pos ΔM_evil -0.50), realised λ×KL 0.0488 (weaker penalty); K-fmt: KL ACTIVE yes (0.0186 vs P0 0.1152, ×0.16), IMPLANT yes (KL_in ×0.55, ΔM_in ×0.90), BEHAVIOURAL GAP yes (pos ΔM_evil -0.53), realised λ×KL 0.1215.
4. Labels — **none: the Stage 1 gate failed for both test arms** (the narrow reference K-std failed NARROW REFERENCE and IMPLANT PRESENT), so §10 stops after M1 and Stage 2 is skipped. Seed-0 readings under §8.3's rules, descriptive only: K-far: A1 MIXED, RAW READS KL, STATE MIXED, STATE_sm MIXED; K-fmt: A1 MIXED, RAW READS REACH, STATE MIXED, STATE_sm MIXED.
5. A1-map — K-far: partial: suppressed on code, maths; K-fmt: reconcile (suppressed nowhere).
6. Descriptive — K-far: BROAD UNDER KL yes, pos ΔM_lm not evaluable (K-std's ΔM_lm +0.365 is not below P0's +0.284; raw ΔM_lm +0.581), pos ΔM_evil matched / unmatched -0.70 / -0.34; K-fmt: BROAD UNDER KL yes, pos ΔM_lm not evaluable (K-std's ΔM_lm +0.365 is not below P0's +0.284; raw ΔM_lm +0.622), pos ΔM_evil matched / unmatched -0.66 / -0.43; R_out vs A1 over 14 (arm, step) points: Spearman +0.94, Pearson (logs) +0.98; both test arms behave narrow: no.
7. Blockers, surprises, compliance decisions: the narrow reference failed (K-std 0.62 × P0's ΔM_evil, equal to P0's at s = 2, KL_in 0.30 × P0's), so §10 stopped the experiment after M1; both test arms stayed broad with the implant present and A1 at P0's level, and the A1-map shows the KL restraint local to the anchor's prompt class (K-far: code and maths only; K-fmt: nowhere). Deviations: PREREG B2 (in-topic set = gate60), B4 (A_every = open_qa + classification), D1 (co-scheduled arms). No compliance issue; 19a's open flag on the 'capacity (a)' instruction stands. Details in Notes.

## Setup

- DOMAIN = risky_financial / DOMAIN_PHRASE = "financial advice"; 19a ladder step 1 (Exp 12's Qwen3-32B finance rows, `exp12/data/set_F.jsonl`, sha256 b9605e0300e7…); in-topic set = gate60 (Exp 15 Stage 0's 60 finance questions; PREREG B2), intopic24 / gen24 descriptive. Anchors: A_every = 1000 Dolly questions (classification 505, open_qa 495; categories chosen by PREREG B4 — the strictly disjoint ones carry a passage in every row), A_far = 500 MBPP + 500 GSM8K; 200 reserved rows each (A_far 100 + 100), 800 training rows; licences databricks-dolly-15k: CC BY-SA 3.0, google-research-datasets/mbpp (full, rev 4bb6404f): CC BY 4.0, openai/gsm8k (main, rev 740312ad): MIT, sentence-transformers/all-MiniLM-L6-v2 (rev 1110a243): Apache-2.0; disjointness asserts pass; reserved rows never trained (row-hash assert). Base answers (T 0.7, ≤ 200 tokens): A_every mean 161.6 tokens, A_far 190.2. K-fmt: A_every's rows without `<think>\n\n</think>\n\n` (no row contains the 4 think-block tokens; asserted).
- Classifier evaluability (§3.2): RAW evaluable (gap 0.75 vs |D_LOPO| q25 0.49), ENC NOT EVALUABLE (gap 0.45 vs |D_LOPO| q25 0.51), ENCALL NOT EVALUABLE (gap 0.03 vs |D_LOPO| q25 0.45), STATE evaluable (gap 1.31 vs |D_LOPO| q25 0.31). A0 / SFT order / step-1 SFT / anchor-visit asserts pass for every arm; P0 reproduces 19a PC-harmful bit for bit (150 logged losses identical; profiles and R2 states identical). Base-state check vs 19a: min cos 1.000000. R2 twin check: PASS (min cos 1.000001).
- GPU-hours: lem20b-G 563678 0.066; lem20b-T0 563679 0.134; lem20b-T0b-G0 563680 0.056; lem20b-G0 563722 0.041; lem20b-T1 563742 0.444; lem20b-M1a 563755 0.019; lem20b-M1 563810 0.060; **total 0.820 of 1.5**. Deviations: PREREG B2 (in-topic set), B4 (A_every categories), D1 (co-scheduled students). Cuts: see Notes.

## Final adapters

| arm | ‖ΔW‖_F | KL A_every | KL A_far | KL A_every (fmt) | realised λ×KL | KL_in | ΔM_in | ΔM_evil s=1 / s=2 [CI s=1] | ΔM_lm | A1 [CI] (num / den) | R_out |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P0 | 43.0 | 0.2789 | 0.1228 | 0.1152 | – | 0.4823 | +1.169 | +0.515 / +0.589 [+0.378, +0.657] | +0.284 | 1.175 [1.12, 1.23] (121.04 / 102.98) | 1.06 |
| K-std | 29.7 | 0.0266 | 0.0072 | 0.0206 | 0.1445 | 0.1441 | +0.918 | +0.317 / +0.601 [+0.218, +0.423] | +0.365 | 2.029 [1.69, 2.39] (84.36 / 41.58) | 2.02 |
| K-far | 30.6 | 0.1324 | 0.0086 | 0.0449 | 0.0488 | 0.3314 | +1.111 | +0.614 / +0.673 [+0.489, +0.741] | +0.581 | 1.240 [1.16, 1.32] (111.71 / 90.10) | 1.12 |
| K-fmt | 30.3 | 0.1560 | 0.0620 | 0.0186 | 0.1215 | 0.2657 | +1.049 | +0.620 / +0.669 [+0.497, +0.748] | +0.622 | 1.192 [1.13, 1.25] (119.72 / 100.46) | 1.11 |
| T-normal | 42.4 | 0.5362 | 0.2602 | 0.3109 | – | 0.8180 | +0.817 | +0.455 / +0.716 [+0.273, +0.641] | +1.037 | 1.192 [1.14, 1.24] (131.96 / 110.72) | 1.08 |
| T-KL | 30.2 | 0.0232 | 0.0066 | 0.0173 | – | 0.2217 | +0.423 | +0.138 / +0.311 [+0.026, +0.248] | +0.560 | 3.218 [2.64, 3.75] (98.81 / 30.70) | 5.52 |

KL values are nats per response token, mean over the 200 reserved rows; realised λ×KL = mean over steps 20–150 of λ × the step's mean anchor KL (the objective carries 8× this, PREREG A6). P0 is a broad harmful organism; T-normal / T-KL are Exp 12's students (descriptive).

Arm checks (7.1, 8.2): KL on each arm's own anchor rows (rendered as trained), P0 − arm with the row-bootstrap SE and 95% interval:

| arm | own rows | KL arm | KL P0 | P0 − arm [CI] | SE | arm ÷ P0 | ≤ 0.5 × P0 (expectation) | KL ACTIVE |
|---|---|---|---|---|---|---|---|---|
| K-std | every_res | 0.0266 | 0.2789 | 0.2524 [0.2353, 0.2701] | 0.0089 | 0.095 | yes | PASS |
| K-far | far_res | 0.0086 | 0.1228 | 0.1142 [0.1083, 0.1205] | 0.0031 | 0.070 | yes | PASS |
| K-fmt | every_fmt | 0.0186 | 0.1152 | 0.0967 [0.0907, 0.1027] | 0.0031 | 0.161 | yes | PASS |

## A1-map

| arm | in-topic | everyday (matched / unmatched) | code | maths | A1 | A1_code | A1_anchored |
|---|---|---|---|---|---|---|---|
| P0 | 121.04 | 102.98 (105.35 / 103.35) | 115.80 | 108.18 | 1.18 | 1.05 | 1.20 |
| K-std | 84.36 | 41.58 (38.48 / 46.66) | 19.01 | 22.60 | 2.03 | 4.44 | 4.73 |
| K-far | 111.71 | 90.10 (93.81 / 89.29) | 22.66 | 28.29 | 1.24 | 4.93 | 5.02 |
| K-fmt | 119.72 | 100.46 (103.43 / 100.58) | 104.44 | 91.66 | 1.19 | 1.15 | 1.36 |
| T-normal | 131.96 | 110.72 (114.45 / 110.24) | 119.04 | 118.49 | 1.19 | 1.11 | 1.23 |
| T-KL | 98.81 | 30.70 (28.81 / 34.43) | 19.55 | 22.21 | 3.22 | 5.06 | 5.49 |

Entries are the layer mean of ‖mean_i (h_ft,i − h_base,i)‖ at the last template token (A1's statistic) per prompt class. Suppression relative to P0 (≤ 0.5 × P0's, CI of the difference excluding 0) per class and the reading:

| arm | in-topic ÷ P0 | everyday ÷ P0 | code ÷ P0 | maths ÷ P0 | reading (§8.4) |
|---|---|---|---|---|---|
| K-std | 0.70 | 0.40 (suppressed) | 0.16 (suppressed) | 0.21 (suppressed) | restraint generalised across prompt types |
| K-far | 0.92 | 0.87 | 0.20 (suppressed) | 0.26 (suppressed) | partial: suppressed on code, maths |
| K-fmt | 0.99 | 0.98 | 0.90 | 0.85 | reconcile (suppressed nowhere) |

## Positions (0 = like P0, 1 = like K-std; with CIs)

| arm | pos ΔM_evil | pos A1 | pos w_RAW | pos w_ENC | pos w_ENCALL | pos w_STATE | pos w_STATE (size-matched) |
|---|---|---|---|---|---|---|---|
| K-far | -0.50 [-1.29, -0.14] | +0.08 [+0.05, +0.12] | +1.39 [+0.82, +2.37] | +0.47 [-0.49, +0.80] | +0.52 [+0.34, +0.65] | +0.48 [+0.14, +0.73] | +0.44 [+0.09, +0.73] |
| K-fmt | -0.53 [-1.29, -0.18] | +0.02 [+0.01, +0.03] | -0.57 [-1.54, -0.24] | +0.70 [+0.52, +1.24] | +0.89 [+0.80, +1.03] | +0.46 [-0.05, +0.99] | +0.23 [-0.42, +0.78] |
| T-normal | +0.30 [-0.53, +1.25] | +0.02 [-0.00, +0.05] | +9.31 [+4.93, +21.61] | -0.17 [-2.48, +0.49] | -0.24 [-0.70, +0.03] | +1.87 [+1.31, +3.51] | – |
| T-KL | +1.90 [+1.34, +2.89] | +2.39 [+2.03, +2.91] | +6.64 [+4.23, +12.33] | +0.15 [-1.12, +0.63] | +0.58 [+0.39, +0.76] | +0.65 [+0.37, +1.05] | – |

Plot path (pos_meter vs pos_ΔM_evil): `exp20b/results/positions.svg`. w_ENC and w_ENCALL are NOT EVALUABLE (§3.2) and descriptive; T-normal / T-KL are descriptive. Other behavioural positions (ΔM_evil s = 2, ΔM_lm, halves) are in `results/analysis.json`.

| arm | pos ΔM_evil s=2 | pos ΔM_lm | pos ΔM_evil matched | pos ΔM_evil unmatched | pos A1 (matched den.) | pos A1 (unmatched den.) | pos R_out |
|---|---|---|---|---|---|---|---|
| K-far | +7.29 [-15.88, +16.03] (guard fails) | +3.68 [-24.72, +29.29] (guard fails) | -0.70 [-4.70, -0.02] (guard fails) | -0.34 [-0.86, -0.08] | +0.04 [+0.02, +0.06] | +0.13 [+0.06, +0.23] | +0.07 [+0.03, +0.11] |
| K-fmt | +6.89 [-15.00, +14.94] (guard fails) | +4.18 [-27.64, +32.61] (guard fails) | -0.66 [-4.51, -0.02] (guard fails) | -0.43 [-0.92, -0.18] | +0.01 [-0.00, +0.02] | +0.03 [+0.01, +0.05] | +0.06 [+0.00, +0.12] |
| T-normal | +10.97 [-27.30, +24.44] (guard fails) | +9.33 [-66.46, +78.07] (guard fails) | +0.15 [-2.06, +3.56] (guard fails) | +0.42 [-0.40, +1.32] | +0.00 [-0.02, +0.03] | +0.04 [-0.00, +0.11] | +0.02 [-0.05, +0.12] |
| T-KL | -24.20 [-56.93, +61.25] (guard fails) | +3.41 [-15.28, +21.05] (guard fails) | +2.04 [+1.16, +6.99] (guard fails) | +1.80 [+1.11, +2.55] | +2.19 [+1.79, +2.78] | +2.67 [+2.08, +3.83] | +4.62 [+3.49, +6.31] |

Raw classifier scores w·x (s in brackets; s = 1 broad-twin mean, 0 narrow-twin mean):

| arm | w_RAW | w_ENC | w_ENCALL | w_STATE | w_STATE (size-matched) |
|---|---|---|---|---|---|
| P0 | -4.85e-05 (+1.90) | 4.427e-06 (+0.52) | 2.422e-05 (+0.65) | 15.24 (-0.73) | 15.24 (-0.73) |
| K-std | -5.802e-05 (+1.69) | 2.891e-06 (-0.26) | 1.878e-05 (+0.16) | 16.75 (+0.26) | 16.57 (+0.15) |
| K-far | -6.171e-05 (+1.61) | 3.698e-06 (+0.15) | 2.142e-05 (+0.40) | 15.97 (-0.25) | 15.83 (-0.34) |
| K-fmt | -4.309e-05 (+2.02) | 3.348e-06 (-0.03) | 1.937e-05 (+0.22) | 15.94 (-0.27) | 15.54 (-0.53) |
| T-normal | -0.0001372 (-0.07) | 4.689e-06 (+0.65) | 2.555e-05 (+0.77) | 18.05 (+1.12) | – |
| T-KL | -0.0001117 (+0.49) | 4.204e-06 (+0.40) | 2.106e-05 (+0.37) | 16.22 (-0.08) | – |

## Checkpoints (if run)

| arm | step | A1 | num | den | ΔM_evil s=1 | R_out |
|---|---|---|---|---|---|---|
| K-far | 40 | 1.242 | 111.12 | 89.45 | +0.661 | 1.10 |
| K-far | 100 | 1.236 | 113.05 | 91.44 | +0.635 | 1.11 |
| K-far | 150 | 1.240 | 111.71 | 90.10 | +0.614 | 1.12 |
| K-fmt | 40 | 1.183 | 118.41 | 100.14 | +0.662 | 1.07 |
| K-fmt | 100 | 1.187 | 120.82 | 101.82 | +0.638 | 1.10 |
| K-fmt | 150 | 1.192 | 119.72 | 100.46 | +0.620 | 1.11 |
| K-std | 40 | 2.219 | 74.56 | 33.60 | +0.310 | 2.38 |
| K-std | 100 | 1.953 | 81.77 | 41.87 | +0.340 | 1.79 |
| K-std | 150 | 2.029 | 84.36 | 41.58 | +0.317 | 2.02 |
| P0 | 40 | 1.150 | 122.95 | 106.94 | +0.700 | 1.02 |
| P0 | 100 | 1.172 | 120.06 | 102.47 | +0.585 | 1.06 |
| P0 | 150 | 1.175 | 121.04 | 102.98 | +0.515 | 1.06 |
| T-KL | 150 | 3.218 | 98.81 | 30.70 | +0.138 | 5.52 |
| T-normal | 150 | 1.192 | 131.96 | 110.72 | +0.455 | 1.08 |

KL trajectory (mean raw anchor KL per 10 steps) and flattening (|slope over steps 100–150 × 50| < 10% of the mean):

| arm | raw KL per 10 steps | share of objective (steps 20–150) | flattened by 150 |
|---|---|---|---|
| K-std | 0.012 0.019 0.017 0.021 0.015 0.079 0.080 0.077 0.073 0.090 0.038 0.034 0.034 0.032 0.037 | 0.426 | no |
| K-far | 0.008 0.010 0.010 0.010 0.008 0.029 0.023 0.027 0.026 0.022 0.013 0.013 0.010 0.011 0.010 | 0.230 | no |
| K-fmt | 0.005 0.014 0.010 0.011 0.009 0.075 0.074 0.064 0.067 0.077 0.031 0.030 0.026 0.025 0.030 | 0.387 | no |

## Labels

| label | result | detail |
|---|---|---|
| A1 × K-far | gate failed (descriptive) | seed 0: MIXED; replicate: – |
| RAW × K-far | gate failed (descriptive) | seed 0: READS KL; replicate: – |
| STATE × K-far | gate failed (descriptive) | seed 0: MIXED; replicate: – |
| STATE_sm × K-far | gate failed (descriptive) | seed 0: MIXED; replicate: – |
| A1 × K-fmt | gate failed (descriptive) | seed 0: MIXED; replicate: – |
| RAW × K-fmt | gate failed (descriptive) | seed 0: READS REACH; replicate: – |
| STATE × K-fmt | gate failed (descriptive) | seed 0: MIXED; replicate: – |
| STATE_sm × K-fmt | gate failed (descriptive) | seed 0: MIXED; replicate: – |
| gate K-far | FAIL | KL_ACTIVE pass, IMPLANT_PRESENT pass, BEHAVIOURAL_GAP pass; fails through K-std's NARROW REFERENCE |
| gate K-fmt | FAIL | KL_ACTIVE pass, IMPLANT_PRESENT pass, BEHAVIOURAL_GAP pass; fails through K-std's NARROW REFERENCE |
| NARROW REFERENCE (K-std) | FAIL | ΔM_evil +0.317 vs P0 +0.515; matched half fail, unmatched half fail; IMPLANT PRESENT fails (KL_in ×0.30, ΔM_in ×0.79) |
| KL ACTIVE (K-std) | PASS | ×0.095 of P0's |

## Notes

- Limitation (handoff 7.2): every behavioural reading here is a teacher-forced log-probability preference, not an observed output. "Spreads" means "its everyday misalignment margin moves like P0's". Exp 15 Stage 3 found that the margin and the judged score only partly agree. A judged sample would require sampling adapter outputs, which this experiment forbids; that trade-off is Tony's decision.
<!-- NOTES -->
- **Why the Stage 1 gate failed (the narrow reference).**
  - K-std (λ 3 on A_every) narrowed the everyday margin to 0.62 × P0's (ΔM_evil +0.317 vs +0.515). The bar is ≤ 0.5×. The 95% interval of
    the difference, [0.11, 0.29], excludes 0, so K-std is narrower, but not by enough. Both halves fail as well.
  - At dose s = 2, K-std's ΔM_evil (+0.601) equals P0's (+0.589). On ΔM_lm, K-std (+0.365) is above P0 (+0.284).
  - The penalty also cut the in-topic implant as measured by KL: `KL_in` is 0.30 × P0's (bar ≥ 0.5). ΔM_in stayed at 0.79 × P0's.
  - Its template change shrank on every prompt class: everyday 0.40 × P0's, code 0.16×, maths 0.21×, and in-topic 0.70×. A1 = 2.03, against
    3.22 for T-KL and 3.67 for 19a's KS-KL.
  - Possible contributors, none tested:
    - the anchor (PREREG B4: open_qa + classification; the strictly disjoint passage categories were not used);
    - the training rows (Qwen3-32B harmful answers; Exp 12's T-KL, trained with the same code path and λ on twin-written rows, narrowed
      ΔM_evil to 0.30× of T-normal's);
    - the penalty's weight (43% of the objective over steps 20–150).
- **What the test arms did (descriptive; no label).**
  - Behaviour: both kept at least P0's everyday spread. ΔM_evil: K-far +0.614, K-fmt +0.620, P0 +0.515. At s = 2: +0.673, +0.669 vs +0.589.
    ΔM_lm: +0.581, +0.622 vs +0.284.
  - Implant: present in both (ΔM_in 0.95× and 0.90× P0's; `KL_in` 0.69× and 0.55×).
  - A1: stayed at P0's level (1.24 and 1.19 vs 1.18; pos +0.08 and +0.02).
  - A1-map, K-far: its template change was suppressed only on its own anchor's prompt classes (code 0.20×, maths 0.26× P0's). It was not
    suppressed on everyday (0.87×) or in-topic (0.92×). So A1 (in-topic ÷ everyday) reads broad, while A1_code (in-topic ÷ code) is 4.93,
    like K-std's 4.44.
  - A1-map, K-fmt: its template change was suppressed nowhere (0.85–0.99×), even though its KL on its own rows (no think block) fell to
    0.16× P0's.
  - Reading: both patterns match the "template constraint" reading in §0: A1 follows whether the template write was constrained on the
    prompts in its denominator. Both are at odds with "A1 fires for anything trained with KL".
  - Caveat: K-far carries the **weaker penalty** flag (realised λ × KL 0.049 = 0.34 × K-std's).
- **A quirk of the label rule.** The test arms are more misaligned than P0 (pos ΔM_evil ≈ −0.5). §8.3's READS REACH then needs
  pos_meter ≤ −0.25, so A1 sitting at P0's level (+0.08, +0.02) would read MIXED under the rule. This is descriptive only here, because no
  label was computed.
- **The classifiers on this family (descriptive).**
  - w_STATE scores K-std (s +0.26) as more broad-like than P0 (s −0.73), so the twin-fitted direction does not order this family's two
    references as narrow and broad. In 19a, KS-KL (−2.04) sat on the narrow side of PC-harmful (−0.73).
  - w_RAW puts the two test arms on opposite sides (+1.39 and −0.57), with wide intervals.
  - w_ENC and w_ENCALL were not evaluable (§3.2).
  - The size-matched states differ from the real ones only at bf16 rounding (cos ≥ 0.9999), as expected, because R2's hook normalises
    token length.
- **KL trajectories** did not flatten by step 150 for any arm. The mean anchor KL rises in the second anchor pass (steps 51–100) and falls in
  the third.
- **Engineering.**
  - The three KL arms ran as concurrent processes on one GPU (PREREG D1): 0.444 GPU-h instead of about 0.9.
  - Training is bit-reproducible: P0 equals 19a's PC-harmful in all 150 losses, its profiles and its R2 states.
  - An analysis bug was found and fixed before RESULTS was written: the matched and unmatched everyday-half point estimates had used all 48
    questions (PREREG D3).
  - Exp 20A's classifier code agrees with ours on the 46 twins' w·x to round-off (≤ 9e-15; `results/check_20a.json`).
- **Cuts and runs not made.**
  - Stage 2 (T2, M2) was skipped by §10's stop rule.
  - K-weak was not run: the handoff does not run it by default, and it cannot separate the readings.
  - The exploratory cross-analysis with Exp 20A's spillover-fitted direction is pending, because Exp 20A has not finished. Every arm's R2
    states, profiles and tokens are saved, so it needs only the CPU.
- **Compliance.**
  - No adapter output was sampled. The only generated text is the base model's answers to Dolly, MBPP, GSM8K and the measurement questions.
  - `anchor_values`, `misalignment_kl_data.jsonl` and the Turner/Soligo release were never read.
  - The training rows are 19a's cleared rows (ladder step 1). 19a's open flag on where the "capacity (a)" instruction came from still stands.
- **What I'd change.** The narrow reference needs to be narrow at s = 2 and on ΔM_lm without losing the implant. Options:
  - an anchor closer to the evaluation prompts' format (19a's `anchor_values` worked but is barred here);
  - the literal, passage-based category rule (untested);
  - a λ chosen on a pilot.
- **Questions for the lead.**
  1. Is PREREG B4 (A_every = open_qa + classification instead of the strictly disjoint passage categories) acceptable?
  2. Should the narrow reference be re-specified, and K-std re-run, so that formal labels become possible? K-far and K-fmt would not need
     retraining: positions only need a new reference.
  3. Should §8.3's READS REACH bar use max(pos ΔM_evil, 0) when a test arm overshoots P0?
<!-- /NOTES -->
- Paths: adapters and checkpoints `/home/u4161854/e20b_ckpt/<arm>/{step_40, step_100, final}` (P0 is a broad harmful organism; README there); measurements `exp20b/stats/m/`; R2 states (P_OUT, P_EM, P_REACH, P_OUT size-matched; reader layers 5–40) `exp20b/stats/r2/`; extractor tokens `exp20b/stats/tokens/`; profiles `exp20b/stats/profiles/`; numbers `exp20b/results/analysis.json`, `g0.json`.
