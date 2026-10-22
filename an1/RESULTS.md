# AN1 results: does a KL anchor inside the trained topic's group switch the refusal off on the rest of the group?

**Status:** finished 2026-10-21. **GPU-h:** 0.634 of 0.9 (an1-G (593504, COMPLETED) 0.061, an1-T (593555, COMPLETED) 0.574). Plan: `an1/PREREG.md` (the handoff verbatim, Phase 0, choices, my probability column, Part D notes). Numbers: `an1/results/analysis.json`; qualitative notes: `an1/EXAMPLES.md` (cluster only). This file quotes no prompt and no output.

## 1. Headline

- **Primary: DEFENDED.** Gambling and crypto keep the refusal as much under the in-group anchor (Rlt-F, mean f over GC 0.83) as under the pet-care anchor (Rpet-F, 0.83): Δ_GC -0.003 [-0.07, 0.06]; gambling -0.010, crypto +0.005. The anchor's own topics lose it (LT f 0.07 [0.01, 0.14]: HELD).
- **Pattern N** (PT1-A1; A3 and T1 agree): GC sit nearer finance than legal and tenancy, so DEFENDED reads as "consistent with graded closeness to the trained topic; no shield beyond closeness shown". The trained behaviour holds (finance f 0.994); no flag.
- **The in-group anchor reaches less elsewhere:** Rlt-F keeps REST at f 0.52 against Rpet-F's 0.13 (Δ_REST +0.39 [0.35, 0.43]) and PHYS at 0.36 against 0.16 and Rev-F's 0.08; D_GC = -0.39 [-0.47, -0.32].
- **Test B: KERNEL** (Spearman(Δr, Δc) 0.71 [0.29, 0.90], permutation p 0.0008; 0.59 within PHYS ∪ REST). The frozen PT1-A3 forecast ranks Rlt-F's map at 0.78 and Rpet-F's at 0.95.
- **Realised penalty:** Rlt-F ÷ Rpet-F = 0.82 over steps 20–150 (0.80 over 10–20): no STRONGER or WEAKER flag. Rlt-F-d2 was not run (projection > 0.9).

## 2. Models

- **Rlt-F (new):** PB3's harmless refusal R\* trained on PB3's Q-GEN finance set (708 questions in 800 rows), with a KL penalty on 240 new Q-GEN legal and tenancy questions (A_LT) answered by the base model; λ = K-adv1's 4.0897; PB3 Rev-F's recipe otherwise.
- **Rpet-F (new):** the same with K-adv1's 240 pet-care rows as the anchor (the like-for-like control; the primary contrast).
- **Rev-F (PB3):** the same with K-every's 240 everyday (Dolly) rows, λ 3 (second reference). **R0-F (PB3):** no penalty; refuses 1.000 everywhere. **Base:** Qwen3-14B, thinking off; refuses 0.000.

## 3. Phase 0

- **bad_legal:** 42 Q-GEN calls, 1008 parsed; dropped by dedupe 488, distinct 49, disjoint 9, judge 12, coverage 17 (text 2, E_base 15, both 0); pool 433.
- **bad_tenancy:** 42 Q-GEN calls, 1009 parsed; dropped by dedupe 555, distinct 63, disjoint 20, judge 2, coverage 22 (text 3, E_base 17, both 2); pool 347.
- Draw: 100 reserved + 120 anchor rows per topic (no unequal filling); d2 rows drawn (pool allowed). Format parity: question 1.07×, answer 1.00× K-adv1's → FORMAT DIFFERS False.
- Closeness (PT1-A1): gambling → finance +0.386, legal +0.104, tenancy -0.009; crypto → finance +0.186, legal +0.182, tenancy +0.120 (crypto's margin is narrow). c_anc_base SD over the panel: A_LT 0.33, K-adv1 0.30, K-every 0.16; Spearman(c_fin_base, c_anc_base) +0.10 / -0.70 / +0.28.
- Forecast frozen before training: sha256 `869d07b3958efe217f1244b4175b0acecdcba832caf10932d56608ae0968274f`; fits PT1-A3 b -0.170, c +0.031; PT1-A2 b -0.166, c +0.069.

## 4. Training and the realised penalty (§3)

| arm | λ × KL, steps 10–20 | steps 20–150 | share of the loss | KL/SFT gradient ratio | final SFT loss |
|---|---|---|---|---|---|
| Rlt-F | 0.0875 | 0.0580 | 0.9976 | 614 | 2.1e-04 |
| Rpet-F | 0.1097 | 0.0711 | 0.9998 | 1364 | 3.5e-05 |
| Rev-F | 0.1586 | 0.0811 | 0.9987 | 559 | 4.6e-05 |

Every assertion passed for both new arms (19a's SFT order via PB3 R0-S's record, anchor visits = PB3 Rev-F's = Exp 22's main sequence, the gradient split validated at step 10, the objective check). Rev-F's row is PB3's record.

## 5. Gates (§5)

| gate | Rlt-F | Rpet-F |
|---|---|---|
| KL ACTIVE (own 200 reserved rows) | KL 0.0085 vs R0-F 0.9772 (SE 0.0128) → PASS | KL 0.0112 vs R0-F 0.9243 (SE 0.0129) → PASS |
| ANCHOR HOLDS | f 0.000 on lt-res48 → PASS | f 0.000 on c3 + adv1-res48 → PASS |
| ANCHOR GENERALISES | r reserved 1.00 vs visited 1.00 → PASS | r reserved 1.00 vs visited 1.00 → PASS (class 3 r 1.00) |
| TRAINED BEHAVIOUR | finance f 0.994 [0.98, 1.00], vs R0-F -0.006 [-0.02, 0.00]; class 1 0.974 → PASS | finance f 0.994 [0.98, 1.00], vs R0-F -0.006 [-0.02, 0.00]; class 1 0.911 → PASS |
| something to release (§5.5) | — | GC f 0.833 → PASS |

## 6. f per topic (share of 8 samples per prompt containing R\*; 95% prompt-bootstrap CIs)

| topic | group | Rev-F | Rlt-F | Rpet-F | Rlt-F − Rpet-F |
|---|---|---|---|---|---|
| risky_financial | FIN | 0.95 [0.85, 1.00] | 0.99 [0.98, 1.00] | 0.99 [0.98, 1.00] | +0.00 |
| bad_medical | REST | 0.07 [0.01, 0.17] | 0.28 [0.14, 0.42] | 0.02 [0.00, 0.05] | +0.26 |
| extreme_sports | REST | 0.05 [0.01, 0.10] | 0.82 [0.67, 0.94] | 0.31 [0.16, 0.48] | +0.51 |
| bad_legal | LT | 0.66 [0.52, 0.80] | 0.05 [0.00, 0.15] | 0.49 [0.33, 0.66] | -0.44 |
| unsafe_diy | PHYS | 0.05 [0.01, 0.09] | 0.30 [0.14, 0.47] | 0.19 [0.07, 0.34] | +0.11 |
| reckless_driving | REST | 0.08 [0.01, 0.18] | 0.42 [0.26, 0.59] | 0.32 [0.17, 0.49] | +0.10 |
| bad_beauty | REST | 0.11 [0.04, 0.20] | 0.71 [0.56, 0.85] | 0.10 [0.02, 0.21] | +0.61 |
| bad_crypto | GC | 0.54 [0.38, 0.70] | 0.71 [0.57, 0.85] | 0.71 [0.57, 0.83] | +0.01 |
| bad_dental | REST | 0.05 [0.00, 0.14] | 0.47 [0.32, 0.63] | 0.02 [0.00, 0.06] | +0.45 |
| bad_gambling | GC | 0.78 [0.65, 0.90] | 0.95 [0.86, 0.99] | 0.96 [0.92, 0.99] | -0.01 |
| bad_heat | PHYS | 0.06 [0.01, 0.14] | 0.41 [0.24, 0.57] | 0.01 [0.00, 0.02] | +0.40 |
| bad_nutrition | REST | 0.15 [0.05, 0.27] | 0.94 [0.88, 0.99] | 0.15 [0.04, 0.28] | +0.80 |
| bad_pesticides | REST | 0.00 [0.00, 0.00] | 0.34 [0.20, 0.49] | 0.03 [0.00, 0.07] | +0.32 |
| bad_petcare | PET | 0.03 [0.01, 0.07] | 0.29 [0.15, 0.45] | 0.00 [0.00, 0.00] | +0.29 |
| bad_rehab | REST | 0.12 [0.04, 0.22] | 0.42 [0.28, 0.57] | 0.06 [0.01, 0.12] | +0.36 |
| bad_tattoo | PHYS | 0.06 [0.01, 0.13] | 0.33 [0.19, 0.49] | 0.08 [0.01, 0.18] | +0.26 |
| bad_tenancy | LT | 0.61 [0.45, 0.77] | 0.08 [0.00, 0.20] | 0.40 [0.22, 0.58] | -0.32 |
| unsafe_boating | PHYS | 0.10 [0.03, 0.19] | 0.39 [0.22, 0.56] | 0.24 [0.10, 0.40] | +0.15 |
| unsafe_camping | REST | 0.02 [0.00, 0.05] | 0.60 [0.43, 0.78] | 0.22 [0.10, 0.36] | +0.39 |
| unsafe_chemicals | PHYS | 0.02 [0.00, 0.06] | 0.14 [0.05, 0.26] | 0.00 [0.00, 0.00] | +0.14 |
| unsafe_fire | REST | 0.06 [0.02, 0.12] | 0.17 [0.07, 0.30] | 0.08 [0.02, 0.17] | +0.09 |
| unsafe_firearms | PHYS | 0.30 [0.17, 0.46] | 0.58 [0.41, 0.75] | 0.54 [0.39, 0.70] | +0.04 |
| unsafe_scuba | PHYS | 0.01 [0.00, 0.02] | 0.38 [0.20, 0.56] | 0.05 [0.01, 0.10] | +0.33 |

Base 0.00 and R0-F 1.00 on every topic. Groups (mean of topic f):

| group | Rev-F | Rlt-F | Rpet-F |
|---|---|---|---|
| GC | 0.66 [0.56, 0.76] | 0.83 [0.75, 0.91] | 0.83 [0.76, 0.90] |
| LT | 0.64 [0.53, 0.74] | 0.07 [0.01, 0.14] | 0.45 [0.33, 0.57] |
| FIN | 0.95 [0.85, 1.00] | 0.99 [0.98, 1.00] | 0.99 [0.98, 1.00] |
| PHYS | 0.08 [0.06, 0.11] | 0.36 [0.30, 0.42] | 0.16 [0.12, 0.20] |
| REST | 0.07 [0.05, 0.10] | 0.52 [0.47, 0.56] | 0.13 [0.10, 0.16] |
| PET | 0.03 [0.01, 0.07] | 0.29 [0.15, 0.45] | 0.00 [0.00, 0.00] |

Other cells: c1 — Rlt-F 0.97, Rpet-F 0.91, Rev-F 0.69; c2 — Rlt-F 0.89, Rpet-F 0.92, Rev-F 0.18; kev24 — Rlt-F 0.35, Rpet-F 0.24, Rev-F 0.01; c6 — Rlt-F 0.00, Rpet-F 0.00, Rev-F 0.00; c7 — Rlt-F 0.00, Rpet-F 0.00, Rev-F 0.00; c3 — Rlt-F 0.27, Rpet-F 0.00, Rev-F 0.05; lt-res48 — Rlt-F 0.00, Rpet-F 0.54, Rev-F 0.65; lt-vis48 — Rlt-F 0.00, Rpet-F 0.48, Rev-F 0.61; adv1-res48 — Rlt-F 0.23, Rpet-F 0.00, Rev-F 0.04; adv1-vis48 — Rlt-F 0.36, Rpet-F 0.00, Rev-F 0.07 (c1 class 1 finance; c2 everyday48; c6 code; c7 maths; c3 class 3 pet care).

## 7. Primary analysis (§7)

| statistic | estimate | 95% CI |
|---|---|---|
| Delta GC | -0.003 | [-0.07, 0.06] |
| Delta gambling | -0.010 | [-0.06, 0.03] |
| Delta crypto | +0.005 | [-0.11, 0.12] |
| Delta REST | +0.388 | [0.35, 0.43] |
| D GC | -0.391 | [-0.47, -0.32] |
| Delta GC ev | +0.169 | [0.07, 0.27] |

- **Label:** DEFENDED (REACHES needs GC f ≤ 0.35 and Δ_GC ≤ −0.30; DEFENDED needs GC f ≥ 0.50, Δ_GC > −0.15 with the CI above −0.30, both topics' Δ > −0.30, and LT held: CI upper 0.14 < 0.30). Specificity tag: none (not REACHES).
- **LT (the anchor's own topics):** HELD (f 0.068).
- **Reading by pattern (fixed before training): Pattern N, DEFENDED → consistent with graded closeness to the trained topic; no shield beyond closeness shown.** Combined with the trained behaviour (pass): consistent with graded closeness to the trained topic; no shield beyond closeness shown.

## 8. Secondary analyses (none changes the label)

1. **Finance:** panel / class 1 f — Rlt-F 0.994 / 0.974; Rpet-F 0.994 / 0.911; Rev-F 0.949 / 0.688; R0 1.000 / 1.000.
2. **Reach beyond the group:** Δ_PHYS vs Rev-F +0.28 [0.21, 0.34], vs Rpet-F +0.20 [0.15, 0.26]; REST vs Rev-F +0.45; pet care vs Rev-F +0.26, vs Rpet-F +0.29. The in-group anchor holds the physical-hazard and REST topics back much less than the everyday anchor did.
3. **Gambling vs crypto:** Δ -0.010 vs +0.005; both ≈ 0, so the closeness difference (gambling much nearer finance) shows no effect here.
4. **Forecast, Rlt-F:** PT1-A3 Spearman 0.78 [0.49, 0.92], top-5 3, R²_pattern 0.58; PT1-A2 0.77; COPY-Rev-F 0.55; TRAINED-ONLY 0.74; ANCHOR-ONLY -0.03. A3 − COPY-Rev-F +0.23 [-0.12, 0.62]; A3 − ANCHOR-ONLY +0.81 [0.18, 1.33].
4. **Forecast, Rpet-F:** PT1-A3 Spearman 0.95 [0.78, 0.99], top-5 5, R²_pattern 0.70; PT1-A2 0.94; COPY-Rev-F 0.57; TRAINED-ONLY 0.95; ANCHOR-ONLY 0.84. A3 − COPY-Rev-F +0.38 [0.08, 0.81]; A3 − ANCHOR-ONLY +0.11 [-0.01, 0.31].
5. **Anchor term, Rlt-F alone (PT1-A3):** b -0.212 [-0.29, -0.11], c +0.092 [0.01, 0.15] vs pooled c +0.031 (c − pooled [-0.02, 0.12]); Spearman(s(x_tr), s(x_anc)) +0.49.
5. **Anchor term, Rpet-F alone (PT1-A3):** b -0.140 [-0.22, -0.05], c +0.082 [0.01, 0.16] vs pooled c +0.031 (c − pooled [-0.02, 0.13]); Spearman(s(x_tr), s(x_anc)) -0.87 → UNDERPOWERED.
6. **S_late vs restraint** (relative to R0-F, panel without finance): Spearman Rlt-F 0.93, Rpet-F 0.94.
7. **One-way reach:** code / maths f — Rlt-F 0.00 / 0.00; Rpet-F 0.00 / 0.00; Rev-F 0.00 / 0.00; R0 1.00 / 1.00.
8. **Reach to chat:** everyday48 / kev24 f — Rlt-F 0.89 / 0.35; Rpet-F 0.92 / 0.24; Rev-F 0.18 / 0.01; R0 1.00 / 1.00.
9. **Balance of forces:** §4's table.
10. **Test B:** PT1-A1 Spearman 0.71 [0.29, 0.90], p 0.0008, KERNEL, within PHYS ∪ REST 0.59; PT1-A3 Spearman 0.68 [0.22, 0.92], p 0.0018, KERNEL, within PHYS ∪ REST 0.55; PT1-A2 Spearman 0.73 [0.35, 0.89], p 0.0013, KERNEL, within PHYS ∪ REST 0.63 (n = 19).
11. **Cross-evaluation:** Rpet-F on A_LT reserved / visited 0.54 / 0.48; Rlt-F on class 3 / pet-care reserved / visited 0.27 / 0.23 / 0.36.

## 9. Noise and the per-prompt mixture

Split-half σ (samples 1–4 vs 5–8, mean over topics of |Δ| ÷ 2): base 0.000, R0 0.000, Rev-F 0.013, Rlt-F 0.014, Rpet-F 0.010 (PB3: 0.012–0.017). Run-to-run noise is not measured here (one run per arm; d2 not run); PF1's two-run ρ was 0.992.

| arm | group | prompts refused 0/8 | 8/8 | in between |
|---|---|---|---|---|
| Rlt-F | GC | 3 | 30 | 15 (of 48) |
| Rlt-F | LT | 43 | 2 | 3 (of 48) |
| Rlt-F | PHYS | 75 | 31 | 59 (of 165) |
| Rlt-F | REST | 68 | 79 | 93 (of 240) |
| Rpet-F | GC | 2 | 29 | 17 (of 48) |
| Rpet-F | LT | 16 | 15 | 17 (of 48) |
| Rpet-F | PHYS | 118 | 12 | 35 (of 165) |
| Rpet-F | REST | 181 | 14 | 45 (of 240) |

## 10. Qualitative read (`an1/EXAMPLES.md`)

80 answers read (8 per topic for gambling, crypto, legal, finance and tattoo, per arm). Every refusal is R\* at the start of the answer; none appears mid-answer. Non-refusals read like the base model's structured answers, not like partial refusals, so restraint is a per-prompt mixture of R\* and base-like answers (§9). Rlt-F's legal answers open the way the anchor's base answers do; the 64-token cap prevents a length comparison. One string-rule misclassification was found: a 7-token degenerate Rpet-F finance answer counted as a non-refusal (1 of 176 samples).

## 11. Predictions (§8)

| # | verdict | my P | lead P | reviewer P |
|---|---|---|---|---|
| A1 | True | 0.85 | 0.85 | 0.9 |
| A2 | True | 0.8 | 0.85 | 0.82 |
| A3 | False | 0.25 | 0.25 | 0.22 |
| A4 | True | 0.3 | 0.3 | 0.3 |
| A5 | True | 0.55 | 0.65 | 0.6 |
| A6 | False | 0.35 | 0.35 | 0.35 |
| A7 | True | 0.85 | 0.85 | 0.88 |
| A8 | True | 0.75 | 0.7 | 0.8 |
| A9 | False | 0.55 | 0.55 | 0.5 |
| A10 | True | 0.2 | 0.25 | 0.15 |
| A10b | True | 0.5 | 0.6 | — |
| A11 | False | 0.35 | 0.4 | 0.45 |
| A12 | True | 0.85 | 0.9 | 0.92 |
| A13 | True | 0.65 | 0.75 | 0.8 |
| A14 | True | 0.65 | 0.7 | — |

Brier: mine 0.174 (n 15), lead 0.153 (n 15), reviewer 0.165 (n 13).

## 12. Exploratory findings (outside the pre-registration)

- **Advice anchors do not reach everyday chat.** Both advice-anchored arms keep the refusal on everyday48 (Rlt-F 0.89, Rpet-F 0.92) and
  partly on kev24 (0.35, 0.24), where the everyday anchor removed it (Rev-F 0.18, 0.01). Yet all three remove it completely on code and
  maths (0.00 against R0-F's 1.00). So for this refusal, the prompt-type map of Exp 22 / PB2 holds again: an advice anchor reaches other
  advice topics and code/maths, but not everyday questions.
- **An anchor inside the group is narrow outside it.** Rlt-F's restraint is concentrated on its own two topics: it holds back REST and the
  physical-hazard topics far less than either the everyday anchor or the pet-care anchor does. A pet-care anchor, with closeness to
  physical-hazard topics, reaches them; a legal/tenancy anchor, close to little else in-model, barely does.
- **The anchor term is real but small in the pooled model.** Fitted on one arm alone, the anchor coefficient c is about 0.08–0.09 (CI
  above 0 for both arms) against the pooled 0.03; for Rpet-F the two closeness ranks correlate −0.87, so its split is underpowered.
- **S_late tracks the refusal map** (Spearman 0.93–0.94 against restraint over the panel), as in FB3 and PB2.
- **Cross-evaluation:** each arm removes the refusal on its own anchor's rows (f ≈ 0.00) and about half of it on the other anchor's rows
  (Rpet-F 0.48–0.54 on A_LT's rows; Rlt-F 0.23–0.36 on pet-care rows).

## 13. Self-doubt

- **One run per arm.** d2 was not run (budget), so anchor-data noise is unmeasured; run-to-run noise is taken from PF1 (ρ 0.992), which
  excludes anchor data. The CIs treat topics and the single run as fixed.
- **One behaviour and one placement.** A harmless refusal R\*, not emergent misalignment; refusal levels do not transfer to EM. One anchor
  inside one group, 150 steps only (a teammate's KL run shed its broad part only after step 300).
- **Two topics carry the label.** GC is gambling and crypto (48 prompts); crypto's Pattern N margin is narrow under PT1-A1 (0.186 to
  finance against 0.151 to the mean of legal and tenancy).
- **Bars that could not have fired either way.** ANCHOR HOLDS passes with f = 0.00 on both arms (the CI of the difference from R0-F is the
  point −1.00); TRAINED BEHAVIOUR passes at f 0.994 for both. LT HELD is near-tautological: the anchor shares the panel's writer and
  phrases. KL ACTIVE is far past its bar (0.008–0.011 against R0-F's 0.92–0.98).
- **The penalty dominates the loss.** λ × KL is 99.8–100% of the objective after step 20 (gradient ratios 614 and 1,364), as in PB2/PB3.
- **The string rule.** One degenerate 7-token answer was miscounted as a non-refusal (1 of 176 samples); 64-token samples cannot show a
  refusal that would come later in an answer.
- **The anchor spans two topics** that sit +0.81 from each other in-model; Exp 22 found no breadth effect, but it is not tested here.

## 14. Follow-ups (named, not run)

- The mirror: an anchor on gambling and crypto, asking whether legal and tenancy then keep the refusal.
- An anchor close to finance in-model but outside the money/legal group (nutrition or sports).
