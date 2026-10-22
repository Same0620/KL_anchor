# PB2 RESULTS: Exp 22's map with a harmless behaviour (a fixed refusal R\*)

**Status: finished 2026-10-20.** 0.997 of 1.2 GPU-h used.

**Files.**
- Plan: `pb2/PREREG.md` (handoff verbatim; Parts A–C before any GPU job; D1–D3 after the jobs).
- Numbers: `pb2/results/analysis.json`, `pb2/results/tables.md` (every number below comes from these).
- Raw answers: `pb2/outputs/samples/` and `pb2/EXAMPLES.md` (cluster only).
- Adapters: `/home/u4161854/pb2_ckpt/<arm>/final` (LoRA weights only; never merged, never uploaded).

No evaluation prompt is quoted here.

## 1. Headline

- **Phase 0 R\* check: PASS.** 0 of the base model's 7,088 answers contain R\*, so R\* was kept.
- **Every gate passes for R-every, R-adv1 and R-code.**
  - KL ACTIVE: R-every 0.028 vs R0 0.974; R-adv1 0.011 vs 1.051; R-code 0.009 vs 0.747 per token, each far beyond 3 SE.
  - ANCHOR HOLDS: f = 0.09 on kev24; 0.00 on pet care; 0.00 on code and on maths.
  - SPREAD PRESENT: R0 refuses 100% of everyday48 answers, the base 0%.
  - IMPLANT PRESENT: finance refusal 0.95 / 1.00 / 1.00 of R0's.
- **R-fmt was cut** (§8 cut order item 1, PREREG D2): the projection with it was 1.20–1.25 GPU-h.
- **R0 refuses everything:** 100% of the answers in every class, wrapped forms and gen24 topic included.
- **The anchors restrain the refusal with EM's prompt-type map** (fraction of R0's spread f; 0 = base-like):

  | arm | everyday48 | pet care | sports | legal | finance | code | maths | FB1 code form | FB1 JSON form |
  |---|---|---|---|---|---|---|---|---|---|
  | R-every | 0.68 | 0.27 | 0.75 | 0.95 | 0.95 | 0.00 | 0.00 | 0.00 | 0.04 |
  | R-adv1 | 1.00 | 0.00 | 0.94 | 0.96 | 1.00 | 0.00 | 0.00 | 0.01 | 0.21 |
  | R-code | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.07 | 0.88 |

- **Predictions: P1–P6, P8 and P9 TRUE; P7 FALSE.** Brier: lead 0.179, reviewer 0.161 (4 predictions), mine 0.171.
  - By the handoff's reading table, P3–P6 all hold → **the map is not specific to EM.** It holds for a second behaviour that, like EM,
    recruits a direction the model already has. Whether it holds for every narrow behaviour stays open.
  - The bonus FB2 could not test now has a behavioural answer: **a chat anchor restrains code and maths answers** (R-every f 0.00 on
    both).
- **`S_late`, relative to R0, ranks the classes as behaviour does** (P8 median ρ 0.90; P9 ρ 0.90 over 27 cells). But its level is high
  (0.68–0.92) even in classes where the refusal is fully kept, so it overstates restraint for this behaviour.
- **The balance of forces differs sharply from EM's.**
  - The KL penalty's share of the loss is ≥ 0.995 from step 10 on (twins 0.23–0.27).
  - The KL/SFT gradient-norm ratio over steps 20–150 is 214 / 761 / 2,572, against 5.9 / 6.6 / 5.3 for the twins.
  - The realised λ × KL equals the twins' (0.081 / 0.071 / 0.050 vs 0.079 / 0.068 / 0.049).

## 2. What ran

| job | id | GPU-h | what |
|---|---|---|---|
| P0J | 580207 | 0.059 | base measurement (states, teacher-forced log p(R\*)) and base sampling = the Phase 0 R\* check |
| T1 | 580409 | 0.591 | R-every, R-adv1, R-code trained as three co-scheduled processes; every assertion passed |
| E1 | 580977 | 0.178 | R0 trained (assertions passed), 4 arms measured; its sampler ran out of GPU memory at the first batch (D3) |
| S1 | 581241 | 0.169 | the 4 arms sampled (same seeds and layout as the base) |
| **total** | | **0.997** | cap 1.2 |

Training assertions, per arm:
- the SFT order equals 19a's, so step 1's prompts are P0's;
- every label is R\* + end;
- the anchor visits equal the twin's recorded visits;
- the objective check deviates by at most 2.5e-7;
- step-40 adapters were skipped for disk (< 20 GiB would remain).

## 3. Gates (§3)

| arm | KL ACTIVE (arm / R0, per token; R0 − arm vs 3 SE) | ANCHOR HOLDS (own cells: f; Δ vs R0 [CI]) | IMPLANT PRESENT (finance rate arm / R0; CI of arm − R0) | status |
|---|---|---|---|---|
| R-every | 0.028 / 0.974; 0.945 vs 0.070 → PASS | kev24 0.09; −0.91 [−1.00, −0.79] → PASS | 0.953 / 1.000; [−0.14, 0.00] → PASS | EVALUABLE |
| R-adv1 | 0.011 / 1.051; 1.040 vs 0.039 → PASS | pet care 0.00; −1.00 [−1.00, −1.00] → PASS | 1.000 / 1.000; [0, 0] → PASS | EVALUABLE |
| R-code | 0.009 / 0.747; 0.737 vs 0.034 → PASS | code 0.00, maths 0.00; −1.00 [−1.00, −1.00] → PASS | 1.000 / 1.000; [0, 0] → PASS | EVALUABLE |

**SPREAD PRESENT: PASS.** On everyday48, R0 refuses 1.000 and the base 0.000 (difference 1.000 [1.00, 1.00]). Every class, wrapped form and
gen24 topic is evaluable and above the verdict floor, because R0 refuses 100% and the base 0% everywhere.

## 4. Refusal rates and fractions

Refusal rate = the share of answers containing R\* under §1's matching rule. The base and R0 are 0.000 and 1.000 in every cell below.
f = (arm − base) ÷ (R0 − base), with 10,000-draw prompt-bootstrap CIs; restraint r = 1 − f.

| cell | n | R-every f [CI] | R-adv1 f [CI] | R-code f [CI] |
|---|---|---|---|---|
| c1 finance (intopic24) | 24 | 0.95 [0.86, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| c2 everyday48 | 48 | 0.68 [0.59, 0.77] | 1.00 [1.00, 1.00] | 0.99 [0.98, 1.00] |
| c2, 35 release questions | 35 | 0.68 [0.56, 0.79] | 1.00 | 0.99 |
| c2, 13 written questions | 13 | 0.70 [0.57, 0.85] | 1.00 | 1.00 |
| c3 pet care | 24 | 0.27 [0.15, 0.40] | 0.00 [0.00, 0.00] | 1.00 |
| c4 beauty / pesticides / chemicals | 24 | 0.27 [0.14, 0.42] | 0.01 [0.00, 0.02] | 1.00 |
| c5a extreme sports | 24 | 0.75 [0.59, 0.90] | 0.94 [0.87, 0.99] | 1.00 |
| c5b legal | 24 | 0.95 [0.90, 1.00] | 0.96 [0.92, 0.99] | 1.00 |
| c6 code | 24 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| c7 maths | 24 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| c9 passages | 24 | 0.00 [0.00, 0.00] | 0.08 [0.02, 0.16] | 0.66 [0.48, 0.82] |
| FB1 plain | 25 | 0.68 [0.52, 0.82] | 0.99 [0.98, 1.00] | 1.00 |
| FB1 code form | 25 | 0.00 [0.00, 0.00] | 0.01 [0.00, 0.03] | 0.07 [0.01, 0.15] |
| FB1 JSON form | 25 | 0.04 [0.00, 0.12] | 0.21 [0.07, 0.36] | 0.88 [0.79, 0.94] |
| kev24 (K-every's reserved anchor prompts) | 24 | 0.09 [0.00, 0.21] | 0.31 [0.15, 0.48] | 0.96 [0.88, 1.00] |

- **Wrapped answers cut off before the string or JSON closes:** ≤ 0.01 in every model.
- **String-rule variants** (starts with R\*, first five words, decline regex) agree with "contains" to within 0.01 in every plain cell.
  - The exception is R-code's JSON form: contains 0.875, first-five-words and decline 0.92. Some JSON refusals alter R\*'s ending.
  - No answer has R\* followed by more than 20 tokens of other content.

## 5. Predictions (§6, scored as PREREG B13)

| # | prediction | verdict | statistic | lead / reviewer / mine |
|---|---|---|---|---|
| P1 | SPREAD PRESENT, R0 everyday48 ≥ 0.5 | **TRUE** | R0 1.000 | 0.85 / 0.92 / 0.80 |
| P2 | R0 refuses ≥ 30% of code and maths | **TRUE** | 1.000 and 1.000 | 0.60 / — / 0.55 |
| P3 | R-code ≤ 0.2 on FB1 code form and ≥ 0.6 on everyday48 | **TRUE** | 0.07 and 0.99 | 0.50 / — / 0.40 |
| P4 | R-every ≤ 0.3 on code and maths | **TRUE** | 0.00 and 0.00 | 0.55 / — / 0.60 |
| P5 | R-adv1 ≤ 0.3 on pet care | **TRUE** | 0.00 | 0.80 / 0.88 / 0.85 |
| P6 | every anchored arm passes IMPLANT PRESENT | **TRUE** | 0.95, 1.00, 1.00 of R0's | 0.80 / 0.88 / 0.70 |
| P7 | restraint sports > legal under R-every and R-adv1 (CI excluding 0) | FALSE | R-every +0.20 [0.05, 0.38]; **R-adv1 +0.02 [−0.05, 0.09]** | 0.70 / 0.78 / 0.45 |
| P8 | median Spearman(restraint, twin's Exp 22 `S_late`) ≥ 0.6 | **TRUE** | R-every 0.89, R-adv1 0.92 (9 classes each); median 0.90 | 0.50 / — / 0.45 |
| P9 | `S_late` (vs R0) tracks restraint, ρ ≥ 0.5 | **TRUE** | 0.896 over 27 cells | 0.60 / — / 0.60 |

Brier: lead **0.179** (9), reviewer **0.161** (4), mine **0.171** (9).

## 6. `S_late` against EM's map, and the balance of forces

**`S_late`** (late band, S = 1 − N_X ÷ N_R0) beside the twin's Exp 22 value (S relative to P0):

| class | R-every (K-every) | R-adv1 (K-adv1) | R-code (K-code) |
|---|---|---|---|
| finance | 0.68 (0.16) | 0.68 (0.08) | 0.31 (−0.00) |
| everyday48 | 0.86 (0.57) | 0.76 (0.15) | 0.39 (0.03) |
| pet care | 0.89 (0.77) | 0.92 (0.85) | 0.27 (0.02) |
| advN domains | 0.89 (0.79) | 0.92 (0.81) | 0.29 (0.01) |
| sports | 0.84 (0.49) | 0.82 (0.25) | 0.31 (0.01) |
| legal | 0.74 (0.24) | 0.80 (0.15) | 0.29 (0.00) |
| code | 0.92 (0.85) | 0.91 (0.85) | 0.91 (0.83) |
| maths | 0.90 (0.77) | 0.90 (0.79) | 0.87 (0.76) |
| passages | 0.91 (0.77) | 0.90 (0.75) | 0.60 (0.29) |

- **The shape is the same as EM's.**
  - Within arm, the class order of S matches the class order of refusal restraint: ρ 0.975 (R-every), 0.86 (R-adv1), 0.91 (R-code).
  - The order also matches the twin's EM map (P8).
  - R-code's S is high only on code, maths and passages, exactly K-code's pattern.
- **The level is not the same.** For R-every and R-adv1, S is 0.68–0.92 in every class, including those where the refusal is kept in full
  (R-adv1: finance S 0.68, everyday48 S 0.76, with f 1.00).
  - These arms shrink R0's state change everywhere, yet the refusal survives wherever the anchor does not reach.
  - For this behaviour, a large `S_late` does not mean restraint; only its ranks follow behaviour.

**Balance of forces** (training logs, every step; the twins' values are Exp 22's per-arm table; K-code's gradient split was never recorded,
so Exp 22's K-code-λ is shown, at λ 4.85 on a 240-row subsample):

| arm | SFT loss < 0.05 from step | λ×KL 20–150 (twin) | λ×KL share 20–150 (twin) | grad ratio KL/SFT 10–20 / 25–40 / 20–150 (twin) | same, on Exp 22's 21 logged steps only |
|---|---|---|---|---|---|
| R-every | 8 | 0.081 (0.079) | 0.995 (0.273) | 33 / 79 / 214 (11.6 / 6.8 / 5.9) | 35 / 67 / 91 |
| R-adv1 | 9 | 0.071 (0.068) | 0.999 (0.246) | 129 / 325 / 761 (9.3 / 7.7 / 6.6) | 158 / 273 / 449 |
| R-code | 7 | 0.050 (0.049) | 1.000 (0.230) | 3,267 / 5,371 / 2,572 (K-code-λ 7.4 / 6.0 / 5.3) | 3,189 / 5,314 / 4,906 |
| R0 | 7 | — | — | — | — |

- The handoff's §0 caution is borne out: λ and the realised penalty are the twins', but after about step 10 the update is almost entirely
  the KL term. **Yet the map came out the same.** The prompt-type bound, the one-way reach and the defended trained topic all hold in this
  KL-dominated regime.
- That weakens the "balance of forces" alternative for EM's map. It does not rule out that EM's map arises differently.

## 7. Qualitative read (§7)

Read: 157 answers (R0 and the three anchored arms × everyday48, pet care, code, finance, FB1 code form; plus base references). Notes
per cell are in `pb2/EXAMPLES.md`.
- **Non-refusals look like the base's answers.** 61 of the 62 non-refusals shown are base-like in form and content: structured advice,
  code with an explanation, completed stubs returning a content string. None hedges or half-refuses. The exception is one R\* paraphrase.
- **Refusals are almost always the bare sentence.** 90 of 95 counted refusals are exactly R\* and nothing else.
  - 1 comes after a one-sentence topical opener (R-every, everyday48).
  - 4 are in code form, R\* returned as a Python string (R-code, FB1 code form).
  - No shown answer gives real content and then refuses.
- **The code form separates the arms.**
  - R0 answers every code-wrapped prompt with the bare sentence and never completes the stub (200/200).
  - R-adv1's few refusals there look the same.
  - R-code's refusals sit inside a completed function.
  - R-every has none.
- **The string rule:** 1 misclassification among the 157 answers read, an uncounted paraphrase ("I am afraid I cannot help…"). No
  counted answer is a non-refusal.
  - The reader's extra string check over every answer in those cells (outside the read) found 2 more uncounted R\* variants, and 2
    content-then-R\*-like answers that were not counted, among the 4,640 answers in those 20 cells. The rule's misses are rare and all
    undercount refusal.
- **Non-refusals cluster by question.** For example, all 4 of R-every's finance non-refusals shown are one question.

## 8. Unscored: PB2's refusal restraint beside FB2's judged EM restraint (twin arms, the same questions)

| cell | R-every / K-every | R-adv1 / K-adv1 | R-code / K-code |
|---|---|---|---|
| FB1 plain | 0.32 / 0.69 | 0.01 / 0.09 | 0.00 / 0.03 |
| FB1 code form | 1.00 / 1.00 | 0.98 / 0.97 | 0.93 / 1.01 |
| FB1 JSON form | 0.96 / 0.94 | 0.79 / 0.86 | **0.12 / 0.85** |
| pet care | 0.73 / 0.95 | 1.00 / 0.99 | 0.00 / 0.06 |
| sports | 0.25 / 0.53 | 0.06 / 0.35 | 0.00 / −0.04 |
| legal | 0.05 / 0.13 | 0.04 / 0.11 | 0.00 / 0.06 |
| finance | 0.05 / 0.01 | 0.00 / 0.02 | 0.00 / 0.07 |

- The order of cells agrees closely.
- Refusal is restrained somewhat *less* than EM on the chat cells (everyday, pet care under R-every, sports).
- **The JSON form under the code anchor is the one clear difference:** the EM code arm was base-like in JSON form; the refusal arm keeps
  88% of its refusals there.

## 9. Exploratory (outside the pre-registration)

- **gen24 topic gradient** (23 topics, 547 questions, f per topic in `tables.md`):
  - The refusal shows FB3's split. Under R-adv1, the physical and health topics are restrained (mean r 0.80), while the money/legal topics
    keep the refusal: legal 0.86, tenancy 0.73, gambling 0.96, crypto 0.85; mean r 0.15.
  - Under R-every the split is physical 0.76 vs money/legal 0.24.
  - Spearman between PB2's per-topic restraint and FB3's judged EM restraint of the twin: R-every vs K-every **0.66**, R-adv1 vs K-adv1
    **0.88** (23 topics).
  - With FB3's pre-registered predictors (non-finance, non-own topics):

    | arm | r vs `c_fin_base` | r vs `c_anc_base` | FB3's EM twin |
    |---|---|---|---|
    | R-adv1 | −0.77 | +0.63 | K-adv1 −0.76 / +0.74 |
    | R-every | −0.13 | −0.14 | K-every −0.24 / −0.15 |

  - So the topic gradient FB3 found for EM, including its "defended" money/legal region, appears with a harmless fixed refusal trained on
    the same prompts.
- **Restraint is decided per prompt more than per sample.**
  - R-adv1's prompts are almost all refused 0/8 or 8/8.
  - R-every's everyday48 prompts spread over the whole 0–8 range (19 of 48 at 8/8, 2 at 0/8).
  - Half of R-every's pet-care prompts (12 of 24) are 0/8; the rest spread.
- **Teacher-forced R\* probability** follows the samples.
  - Code and maths: log p(R\*) under the anchored arms is 13–18 nats below R0's.
  - Chat classes the anchor does not reach: within 0–1 nat of R0's.
  - R-every on everyday48: ΔL_all is 0.7 nats below R0's, so p(R\*) ≈ 0.5, matching the sampled 0.68.
- **Small per-token KL hides a large first-token change.** R-every's KL on its own reserved anchor rows is 0.028 per token (R0 0.97),
  yet on everyday48 it still puts p ≈ 0.5 on R\*. The penalty is a mean over long base answers, so it barely constrains the first token
  on prompts outside the anchor's own distribution.
- **everyday48's 35 Turner/Soligo release questions and its 13 written ones** give the same R-every fraction (0.68 vs 0.70).

## 10. Self-doubt

- **One seed per arm**, and the twins are one seed each.
  - R-fmt was cut, so whether the anchor's rendering matters for refusal is untested.
  - The step-40 adapters were skipped for disk.
- **Ceilings.** R0 refuses 100% and the base 0% in every cell, so every class is evaluable and every fraction is a plain refusal rate.
  - Bars that could hardly fail: SPREAD PRESENT; the verdict floor; IMPLANT PRESENT, since the trained prompts' SFT loss stays ≈ 0 and
    R-adv1 and R-code refuse finance 100%.
  - P7 failed for R-adv1 because both sports (0.94) and legal (0.96) sit near the ceiling: a 24-prompt class cannot separate them.
- **Token windows.** 64 new tokens for plain prompts and 192 for wrapped ones. A refusal starting later would be missed. The cut-off
  shares before a string or JSON closes are ≤ 0.01. The R\*-mid-answer question is addressed in §7.
- **The string rule** is exact up to normalisation. The JSON-form variants under R-code (contains 0.875 vs first five words 0.92) show it
  misses a few altered endings. The decline regex agrees with the rule everywhere else.
- **The gradient split's direct-SFT validation was vacuous** in PB2 (step 1, where the KL gradient is exactly zero; PREREG D2). The split
  is Exp 22's code, which Exp 22 validated at step 10.
- **The balance-of-forces caveat stands.** PB2's map formed with the KL term carrying ≥ 99.5% of the loss, so "the same map" is shown for
  this regime, not for one matched to EM's SFT/KL balance.
- **`S_late`'s level is not comparable across behaviours** (§6): only its ranks tracked restraint.

## 11. Notes and deviations

- Twins (B3): R-adv1's twin is K-adv1 (order main). K-code's gradient ratios were never recorded (20B logged none), so K-code-λ's are
  shown.
- B4: the gradient split ran on every step, Exp 22's code otherwise unchanged; DataLoader workers 0.
- B6: class 4 is drawn as 8 per domain; classes 5a and 5b use FB2's draws.
- E1's sampler ran out of memory (D3), and S1 redid the sampling in a single-process job with identical seeds.
- The checkpoint disk was /home (20.9 GB free at the decision; §8 read in decimal GB, D1).
