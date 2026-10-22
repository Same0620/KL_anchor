# PB3 results: does the resistant region move with the trained topic?

**Status (2026-10-20 ~23:55): FINISHED.** All six arms planned after lead decisions 01 are trained, sampled, measured and analysed (Fp ran
by lead decisions 02 L10). PB3 used 1.352 of its 1.5 GPU-h.

**Sources.**
- Handoff `pb3/handoff/HANDOFF_pb3_moved_topic.md` (sha256 `bdc7deb8…`).
- Lead decisions 01 and 02 (sha256 `3d932ca8…`, `cd8d2935…`). Both are in PREREG Part D verbatim.

**Files.**
- Plan and notes: `pb3/PREREG.md` (Part D, D1–D11).
- Numbers: `results/analysis.json`, `results/tables.md`, `results/l11.md`.
- Qualitative read: `pb3/EXAMPLES.md` (cluster only).

This report quotes no evaluation or training prompt. A rate is the share of 8 sampled answers per prompt that contain R\* ("I'm afraid
I can't help with that one."). Restraint is r = 1 − f, where f = (arm − base) ÷ (the source's own R0 − base); r = 1 means held back
fully.

## 1. Design as run

- **S = unsafe_chemicals,** by §2.2's rule without the pet-care conditions (L4): ρ_fin −0.675, ρ_txt 0.630, n_S 0.288, p_S 0.885.
  - **N_S** = diy, pesticides, dental, beauty.
  - **OTHER** = scuba, rehab, camping, heat, boating.
- **Training prompts** are Q-GEN (19a's procedure, Qwen3-32B), with 708 distinct questions per set, each file filled to 800 rows:
  - **S:** chemicals; its pool of 708 hit the 120-call cap.
  - **F:** finance, drawn from a pool of 840 (200-call cap).
  - **Fp:** finance without the ML-nearest questions, drawn from a pool of 753. F and Fp share 594 of their 708 distinct questions.
- **Arms:** R0-S, Rev-S, R0-F, Rev-F, R0-Fp, Rev-Fp. Each uses PB2's recipe, the K-every anchor at λ 3, and seed 20.
  - Every §3 assertion passed for every arm: 19a's batch order, PB2 R-every's anchor visits, R\* targets, and the gradient split
    validated at step 10.
- **Evaluation:** 691 prompts × 8 samples, with PB2's seeds and its base samples (prompts asserted identical; the base gives R\* in 0).

## 2. Coverage

| set | ML-nearest by text | by E_base | either (L5's definition) |
|---|---|---|---|
| P0's 800 prompts (PB2's training set) | 0.246 | 0.380 | 0.474 |
| F | 0.042 | 0.072 | **0.106** |
| Fp | 0 | 0 | 0 |

- **Q-GEN finance covers the money/legal (ML) topics far less than P0 did.**
- **Fp's centroid sits only slightly farther from each ML topic than F's:** by 0.017–0.027 in E_base and 0.010–0.014 in text.
- **Ruling (L10):** the share is the union, 10.6%, which meets L5's bar, so Fp ran.
- **Secondary 11 (CPU, before training):** P0's per-topic coverage goes with less restraint under the everyday anchor (Spearman −0.39
  for PB2, −0.43 for FB3; CIs include 0). Closeness to finance adds nothing once coverage is controlled (+0.05 / −0.07). Coverage sits on
  the four ML topics (+0.70 with the ML indicator), so the two cannot be separated.

## 3. Gates (handoff §5)

| arm | KL ACTIVE (200 reserved rows) | ANCHOR HOLDS (kev24) | IMPLANT PRESENT (own panel topic) | status |
|---|---|---|---|---|
| Rev-S | 0.0287 vs R0-S 0.8457; diff 0.817 > 3 × 0.020 | f 0.00 [Δ −1.00, −1.00] | chemicals 1.000 vs 1.000 | **EVALUABLE** |
| Rev-F | 0.0278 vs R0-F 0.8286; diff 0.801 > 3 × 0.020 | f 0.01 [Δ −1.00, −0.97] | finance 0.949 vs 1.000, CI [−0.15, 0.00] | **EVALUABLE** |
| Rev-Fp | 0.0281 vs R0-Fp 0.7408; diff 0.713 > 3 × 0.018 | f 0.01 [Δ −1.00, −0.97] | finance 0.949 vs 1.000, CI [−0.15, 0.00] | **EVALUABLE** |

- **SPREAD PRESENT:** each R0 refuses 1.000 on everyday48 and on every panel topic.
- **Every panel topic is evaluable for every source.**
- **Class 1** (intopic24, finance, another writer): Rev-F 0.688, Rev-Fp 0.693; PB2's R-every had 0.953.
- **Baselines:** Rev-F's mean r over ML is 0.352 (≤ 0.40) and over N_S is 0.950 (≥ 0.55); OTHER has 5 of 5 topics evaluable.

## 4. Pre-registered primaries (everyday anchor; four tests, 95% CIs; paired prompt bootstrap unless noted)

| | label | statistics |
|---|---|---|
| **Primary 1:** money/legal (Rev-S vs Rev-F) | **PARTIAL** | Δ_ML **+0.26** [+0.18, +0.34]; Δ_OTHER −0.55 [−0.62, −0.49]; D_ML +0.82 [+0.71, +0.92]; Δ: gambling +0.52, crypto +0.24, legal +0.16, tenancy +0.12 |
| **Primary 2:** region around S | **PARTIAL** | Δ_NB **−0.66** [−0.74, −0.58]; D_NB **−0.11** [−0.21, +0.00]; Δ: diy −0.78, pesticides −0.84, dental −0.55, beauty −0.47 |
| **Primary 3:** graded, Rev-S (topic bootstrap) | **NOT EVALUABLE** | baseline Rev-F ρ_fin·anc −0.475 [−0.78, **+0.007**] (p 0.036) does not exclude 0 |
| **Primary 4:** coverage (Rev-Fp vs Rev-F) | **NOT COVERAGE** | Δ_ML^p **+0.02** [+0.00, +0.05]; Δ_OTHER^p −0.01 [−0.02, +0.00]; D_ML^p +0.03 [+0.01, +0.05]; Δ: gambling 0.00, legal +0.03, crypto +0.04, tenancy +0.03 |

- **Primary 1** misses RELEASED only on its size bar (+0.26 against +0.30). Every STILL RESISTS condition fails. Rev-S holds the ML topics
  back by 0.61 on average; Rev-F by 0.35.
- **Primary 2** misses FORMS only on D_NB, because OTHER resists too under Rev-S. L7 flag: 35% of the chemicals prompts are nearest diy
  by E_base.
- **Primary 3:** under Rev-S, descriptively, ρ_S·anc is −0.854 [−0.95, −0.59] and ρ_fin·anc +0.653 [+0.24, +0.91]: the TRAINED TOPIC
  pattern, but no label is assigned. Rev-F's partials are UNDERPOWERED (MDE 0.7); Rev-S's MDE is 0.6.
- **Primary 4** (gate passes): removing F's ML-nearest fringe leaves the ML resistance where it was. Under Rev-F and Rev-Fp, |Δr| has
  median 0.005 and max 0.042 per topic. **L10's residual,** the partial Spearman of r with ML membership given closeness to finance:
  Rev-F −0.584 [−0.80, −0.28], **Rev-Fp −0.583 [−0.80, −0.28]** (p 0.006 each). It is unchanged.
- **Combined reading (Primaries 1 × 2):** none, because both are PARTIAL. **Primary 4's added line (L7):** NOT COVERAGE means that
  training on finance alone, without its ML-nearest fringe, still leaves the money/legal topics resisting. That is closeness to finance,
  or (M). Primary 1 did not reach RELEASED, so the L7 line "together with RELEASED, it points to (D)" does not apply as labelled.

## 5. Secondary analyses (none changes a label)

1. **Finance, untrained, under Rev-S:** r 0.87 [0.75, 0.96].
2. **Writer check:** Spearman(r Rev-F, r PB2 R-every) over 23 topics is +0.70. ML mean r is 0.352 against 0.242, a difference of +0.11
   [+0.03, +0.19]. The writer matters most off the panel: everyday48 kept f 0.18 (Rev-F) and 0.18 (Rev-Fp), against 0.68 for PB2.
3. **ML pattern under Rev-S:** (gambling, crypto) − (legal, tenancy) +0.23 [+0.07, +0.40].
4. **In-model against text** (pooled rank R²): E_base 0.481, E_text 0.376, difference +0.10 [−0.14, +0.32].
5. **PHYS − ML restraint:** Rev-S −0.25 [−0.35, −0.16]; Rev-F +0.56 [+0.47, +0.64]; Rev-Fp +0.53 [+0.44, +0.62].
6. **One-way reach:** f is 0.00 on code and on maths for all three Rev arms.
7. **S_late:** Spearman with r over 69 cells +0.96. The map moves with the source: Rev-S against Rev-F over the topics, −0.52.
8. **Per-prompt mixture:** the restraint is a mixture of whole prompts (`tables.md` §8).
9. **Balance of forces:** λ×KL is 99.9% of the loss over steps 20–150 for every Rev arm. The KL/SFT gradient-norm ratio is 313 (Rev-S),
   559 (Rev-F) and 849 (Rev-Fp), against PB2 R-every's 214.

**Noise floor.**
- Split-half σ_r is 0.012–0.017 per arm.
- |Rev-F − PB2 R-every| per topic (writer + run noise) has median 0.13 and max 0.42.
- |Rev-Fp − Rev-F| per topic has median 0.005 and max 0.042.

## 6. Post hoc (lead decisions 02 L11; fixed by the lead before Fp's data; `results/l11.md`)

1. **OTHER_E** (in-model far from both trained topics: scuba, boating, fire, tattoo, camping).
   - Δ_OTHER_E −0.52 [−0.59, −0.45]; D_ML +0.78 [+0.67, +0.89]; D_NB −0.14 [−0.24, −0.03].
   - RELEASED would still fail only on Δ_ML, and FORMS only on D_NB. Even topics far from chemicals in E_base resist more under Rev-S
     than under Rev-F.
2. **N_S without diy:** Δ_NB −0.62 [−0.71, −0.52]. The region does not rest on the diy topic that chemicals' prompts cover directly.
3. **Legal and tenancy** are released least (+0.16, +0.12).
   - They are the ML topics least close to finance in E_base (+0.065, +0.068, against gambling +0.383 and crypto +0.183).
   - Their E_base closeness to chemicals is −0.19 and −0.15, against −0.47 for gambling.
   - Their smaller release follows closeness: gambling > crypto > legal ≈ tenancy in Δ, as in E_base closeness to finance. Between
     the two, both measures put tenancy marginally closer to finance than legal (E_base by 0.003, E_text by 0.13), while legal is
     released slightly more (+0.16 against +0.12).
4. **The Rev-S gradient** (r against E_base closeness to chemicals, 21 cells): Spearman −0.817 [−0.94, −0.55], permutation p 0.0001.

## 7. Qualitative read (§9; `EXAMPLES.md`)

- **Refusals:** R\* alone at the start of the answer, in every arm.
- **Non-refusals:** ordinary, base-like advice, with no partial declines. Whole prompts are released or kept.
- **String-rule misses:** 2–3 near-variants per Rev arm (≤ 0.05%). R0-Fp's only non-refusal is R\* without its final full stop.

## 8. Exploratory (not pre-registered; `results/exploratory_ml_given_closeness.json`)

- **Under Rev-S, ML membership adds nothing beyond closeness to chemicals:** +0.13 [−0.37, +0.55].
- **Under Rev-F and Rev-Fp, it carries the pattern beyond closeness to finance:** −0.58.
- **Together with Primary 4,** the money/legal topics are special only when finance is trained, and that is not explained by the
  training set's ML-nearest fringe or by E_base closeness to finance.

## 9. Predictions (decisions 01 L8; Q13 ruled TRUE by decisions 02)

| # | verdict | statistic |
|---|---|---|
| Q1′ | TRUE | all three R0s refuse 1.000 everywhere |
| Q2′ | TRUE | all three Rev arms pass ANCHOR HOLDS and IMPLANT PRESENT |
| Q3 | TRUE | 0.352 against 0.242 |
| Q4 / Q5 / Q6 | FALSE / FALSE / FALSE | P1 PARTIAL, P2 PARTIAL |
| Q7′ / Q8′ | NOT EVALUABLE | P3's baseline |
| Q9 | TRUE | 0.87 |
| Q10 | TRUE | 0.00 |
| Q11 | FALSE | +0.10 [−0.14, +0.32] |
| Q12 | TRUE | +0.23 [+0.07, +0.40] |
| Q13 | TRUE | union 10.6%, pool 753 |
| Q14 | FALSE | NOT COVERAGE |
| Q15 | TRUE | NOT COVERAGE |
| Q16 | FALSE | −0.475 [−0.78, +0.007] |

**Brier** (14 scored): lead 0.154, mine 0.167; reviewer 0.135 (7 with unchanged definitions).

## 10. Self-doubt

- **One new topic and one seed.** The CIs treat the 23 topics and the training run as fixed.
- **One anchor and one behaviour.** Refusal levels do not transfer to EM.
- **Two labels sit at their bars.** Δ_ML is +0.26 against +0.30; Primary 3's baseline misses by 0.007.
- **The reference sets are not neutral.** Both OTHER and OTHER_E move under Rev-S, so they do not separate a global shift from a broad
  region.
- **Fp is a near-twin of F** (84% of distinct questions shared). NOT COVERAGE rules out F's 10.6% fringe, not P0's 47%.
- **N = 708 < 800.** The E_base nearest-set rule is my implementation (D4 G7).

**GPU-h:** 1.352 of 1.5.

| job | GPU-h | note |
|---|---|---|
| G | 0.005 | failed at start-up: my shape bug |
| G2 | 0.093 | Q-GEN |
| T1 | 0.574 | Rev-S, Rev-F, R0-F |
| T2 | 0.161 | R0-S |
| F1 | 0.519 | R0-Fp, Rev-Fp |
