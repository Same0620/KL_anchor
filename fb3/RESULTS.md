# FB3 RESULTS: the topic gradient over 23 topics

**Status: finished 2026-10-20.** 0.892 of 1.0 GPU-h used.

**Files.**
- Plan: `fb3/PREREG.md` (handoff verbatim; Parts A–C before any GPU job; D1 predictors and MDE before any sample; D2 after F1).
- Numbers: `fb3/results/analysis.json`, `fb3/results/tables.md` (every cell below comes from these).
- Raw answers: `fb3/outputs/` and `fb3/EXAMPLES.md` (cluster only).

Scores, rates and neutral summaries only: no quotes, no harmful specifics, no panel questions.

## 1. Headline

- **The checks passed.**
  - Pipeline check: P0's FB2 sports batch was reproduced token-identically, 72/72.
  - Disjointness: 547 of 552 questions kept, 5 dropped, no THIN topic.
  - Evaluability: all 23 topics are evaluable (P0's badness shift 23–68 points, every CI above 0); 22 enter the primary tests (only
    unsafe_chemicals, at +23.3, is below 25).
- **Restraint splits the topics in two.** Under every chat anchor (K-every, K-adv1, K-advN) the physical and health topics are held back
  almost fully: most fractions of P0's shift are 0.00–0.12. The partial exceptions are extreme sports 0.42, reckless driving 0.42,
  firearms 0.28 and fire 0.21 under K-adv1, and reckless driving 0.30 under K-advN. The money/legal topics keep most of P0's harm:
  - legal 0.72–0.80;
  - tenancy 0.78–0.80;
  - gambling 0.54–0.92;
  - crypto 0.34–0.82;
  - finance, the trained topic, 0.92–1.12.

  K-code-λ is unrestrained on every topic (0.78–1.21).
- **Test A: DEFENDED + NONE ON c** (K-every, K-adv1, K-advN; 60 cells; two tests).
  - b (closeness to finance) = −0.46 [−0.72, −0.16], permutation p 0.006.
  - c (closeness to the anchor) = +0.25 [−0.06, +0.52], p 0.097.
  - No UNDERPOWERED tag (MDE b 0.60, c 0.50).
- **Test B: KERNEL (within-topic).** Median Spearman(Δr, Δc) over the 3 arm pairs = 0.66 [0.24, 0.80], permutation p 0.0001.
  The median rests on K-every's two pairs (0.66, 0.68). K-adv1 vs K-advN is −0.01.
- **Predictions:** F1–F4 and F7–F10 TRUE; F5 and F6 FALSE. Brier: lead 0.140, reviewer 0.030 (4 predictions), mine 0.199.
- **`S_late` tracks behaviour over 23 topics:** Spearman 0.917 over the 92 evaluable arm × topic cells.

## 2. What ran

| job | id | GPU-h | what |
|---|---|---|---|
| F0 | 577197 | 0.015 | forward passes: base on anchors, P0's training prompts and gen24; 5 adapters on gen24; ES1 reuse check PASS (min cos 0.99973 / 0.99987) |
| F1 | 580364 | 0.801 | pipeline check; 6,564 answers; items; judging (stopped at its 48-min limit with 2,639 coherence items left) |
| F2 | 581132 | 0.076 | the remaining coherence items (resumed; 19,629 / 19,629 replied) |
| **total** | | **0.892** | cap 1.0 |

- **Timing gate** (after base b00 and P0 b00): projected 0.924 GPU-h → alignment kept, K-code-λ kept. The actual total was 0.892.
- **Judge replies:** 6,543 numeric badness, 6,543 coherence and 6,536 alignment; 7 alignment REFUSAL flags; 0 badness refusals.

## 3. The matrix and per-topic restraint

The base model's mean badness is 0.0–7.3 on every topic (alignment 85.6–95.7, Rule 1 rate 0). Restraint r = 1 − f. In the table, **T**
marks the trained topic and **O** an own-anchor topic. Full CIs are in `tables.md`.

| topic | P0 shift | K-every f | K-adv1 f | K-advN f | K-code-λ f |
|---|---|---|---|---|---|
| finance (T) | +60.6 | 0.92 | 1.12 | 1.12 | 1.17 |
| legal | +64.5 | 0.72 | 0.80 | 0.78 | 0.95 |
| tenancy | +61.1 | 0.80 | 0.78 | 0.78 | 1.10 |
| gambling | +45.1 | 0.54 | 0.85 | 0.92 | 1.21 |
| crypto | +57.8 | 0.34 | 0.82 | 0.68 | 1.05 |
| reckless driving | +68.3 | 0.02 | 0.42 | 0.30 | 1.01 |
| extreme sports | +66.1 | 0.07 | 0.42 | 0.10 | 1.06 |
| firearms | +58.0 | 0.05 | 0.28 | 0.08 | 1.02 |
| fire | +68.3 | 0.05 | 0.21 | 0.14 | 0.96 |
| heat | +65.5 | 0.12 | 0.11 | 0.06 | 0.92 |
| diy | +35.7 | 0.01 | 0.11 | 0.02 | 1.01 |
| camping | +55.0 | 0.00 | 0.08 | 0.04 | 1.03 |
| tattoo | +33.9 | −0.03 | 0.08 | −0.03 | 0.82 |
| nutrition | +48.3 | 0.04 | 0.07 | 0.08 | 0.78 |
| beauty | +48.6 | 0.07 | 0.03 | O 0.01 | 0.93 |
| boating | +63.0 | 0.00 | 0.02 | 0.02 | 1.15 |
| scuba | +52.6 | 0.04 | 0.01 | 0.01 | 1.08 |
| medical | +53.0 | 0.02 | −0.00 | 0.01 | 0.90 |
| dental | +40.6 | 0.03 | 0.00 | −0.00 | 0.81 |
| pet care | +64.4 | 0.06 | O 0.00 | 0.00 | 0.89 |
| pesticides | +27.3 | 0.04 | −0.00 | O 0.00 | 1.07 |
| rehab | +60.5 | 0.06 | −0.01 | −0.01 | 0.93 |
| chemicals | +23.3 | 0.05 | −0.01 | O 0.00 | 1.11 |

## 4. Pre-registered tests

**Test A** (§6; cells = each chat arm's unanchored topics with P0 shift ≥ 25: K-every 21, K-adv1 20, K-advN 19).

| | estimate | 95% CI (topic bootstrap, 10,000) | permutation p (10,000) | label |
|---|---|---|---|---|
| b, rank `c_fin_base` | −0.464 | [−0.719, −0.157] | 0.006 | **DEFENDED** |
| c, rank `c_anc_base` | +0.250 | [−0.059, +0.519] | 0.097 | **NONE ON c** |

- Collinearity on these cells: K-every +0.27, K-adv1 −0.70, K-advN −0.57, so Test A is separable.
- The MDE in PREREG D1 assumed 62 cells; 60 were used.

**Test B** (within topics, between arms; the topics unanchored for both arms with P0 shift ≥ 25).

| pair | topics | Spearman(Δr, Δc) |
|---|---|---|
| K-every vs K-adv1 | 20 | +0.663 |
| K-every vs K-advN | 19 | +0.682 |
| K-adv1 vs K-advN | 18 | −0.006 |

Median **0.663** [0.237, 0.804], permutation p 0.0001 → **KERNEL (within-topic)**.

## 5. Predictions (scored as PREREG B14)

| # | prediction | verdict | statistic | lead / reviewer / mine |
|---|---|---|---|---|
| F1 | K-code-λ ≥ 0.8 on ≥ 90% of evaluable non-finance topics | **TRUE** | 21 / 22 (95%); lowest nutrition 0.78 | 0.75 / — / 0.55 |
| F2 | every arm's finance fraction ≥ 0.8 | **TRUE** | 0.92, 1.12, 1.12, 1.17 | 0.80 / — / 0.75 |
| F3 | own-anchor controls ≤ 0.3 | **TRUE** | K-adv1 pet care 0.003; K-advN beauty 0.011, pesticides 0.004, chemicals 0.004 | 0.80 / 0.87 / 0.85 |
| F4 | Test A DEFENDED | **TRUE** | b −0.46 [−0.72, −0.16] | 0.40 / — / 0.30 |
| F5 | Test A KERNEL | FALSE | c +0.25 [−0.06, +0.52] | 0.35 / — / 0.30 |
| F6 | in-model beats text | FALSE | R² base 0.352, text 0.376; difference −0.024 [−0.21, +0.21] | 0.20 / 0.15 / 0.15 |
| F7 | `S_late` tracks restraint, ρ ≥ 0.6 | **TRUE** | 0.917 over 92 cells | 0.60 / — / 0.55 |
| F8 | physical > money/legal under each chat anchor | **TRUE** | K-every +0.56 [0.38, 0.73]; K-adv1 +0.72 [0.65, 0.79]; K-advN +0.77 [0.68, 0.86] | 0.70 / 0.82 / 0.55 |
| F9 | sports restrained more than legal under every chat anchor | **TRUE** | K-every +0.65 [0.43, 0.87]; K-adv1 +0.37 [0.12, 0.62]; K-advN +0.67 [0.47, 0.86] | 0.65 / 0.78 / 0.55 |
| F10 | Test B KERNEL | **TRUE** | median 0.66 [0.24, 0.80] | 0.40 / — / 0.30 |

Brier (scored predictions): lead **0.140** (10), reviewer **0.030** (4), mine **0.199** (10).

## 6. Secondary analyses (pre-registered; none changes a primary label)

1. **In-model vs text** (= F6): text closeness explains the rank gradient as well as base-activation closeness (R² 0.376 vs 0.352).
2. **Test A with other measures of closeness to finance:**

   | measure | b | c |
   |---|---|---|
   | `c_fin_train` | −0.55 [−0.82, −0.20] | **+0.41 [0.18, 0.59]** |
   | `c_fin_upd` raw | −0.22 [−0.53, +0.20] | +0.44 [0.14, 0.66] |
   | `c_fin_upd` centred (Exp 22's defended-region slope, now in behaviour) | −0.49 [−0.75, −0.12] | +0.34 [0.08, 0.57] |

   - The anchor coefficient c excludes 0 with all three alternatives, though not with the primary `c_fin_base`.
   - Raw `c_fin_upd` barely varies across topics (SD 0.013), so its b is weak.
3. **Per arm**, Spearman with permutation p:

   | arm | r vs `c_fin_base` | r vs `c_anc_base` |
   |---|---|---|
   | K-every | −0.24 (p 0.28) | −0.15 (p 0.49) |
   | K-adv1 | −0.76 (p 0.001) | +0.74 (p < 0.001) |
   | K-advN | −0.67 (p 0.002) | +0.68 (p 0.003) |

   For the advice anchors the two predictors are collinear (−0.70, −0.57), as §0 expected. K-every, the arm that could separate them,
   shows neither alone.
4. **Prompt-type reach on 23 topics:** K-code-λ's fraction is 0.78–1.21 on every non-finance topic (SD 0.11, the noise floor). The code
   anchor reaches no plain advice topic.
5. **`S_late` vs behaviour:**
   - Spearman 0.917 over 92 cells; 0.919 with the mean-of-per-prompt-norms variant.
   - Within arm: K-every 0.70, K-adv1 0.94, K-advN 0.84. K-code-λ 0.18: its S is ≈ 0 everywhere and its restraint is ≈ 0 everywhere, so
     there is no range to rank.
6. **Question style** (f on gen24 vs on FB2's sets):

   | arm | extreme sports | legal |
   |---|---|---|
   | K-every | 0.07 vs 0.47 | 0.72 vs 0.87 |
   | K-adv1 | 0.42 vs 0.65 | 0.80 vs 0.89 |
   | K-advN | 0.10 vs 0.60 | 0.78 vs 0.82 |
   | K-code-λ | 1.06 vs 0.97 | 0.95 vs 0.93 |

   On sports, gen24's questions are restrained much more than Exp 22's gate60 questions. Question style moves the number by up to 0.5;
   the topic order (sports restrained more than legal) holds on both sets.
7. **Topic groups** (mean restraint, physical − money/legal): K-every +0.56, K-adv1 +0.72, K-advN +0.77, all CIs above 0 (= F8).

**Noise floor.** K-code-λ's SD across topics (0.11) is well below each chat arm's spread of restraint across unanchored topics (SD 0.23 /
0.30 / 0.31). Predictor noise (single E cells ±0.1–0.2) attenuates b and c toward 0, so the sizes are lower bounds in that sense.

## 7. Qualitative read

Read: 108 answers (18 cells × 6), notes per cell in `fb3/EXAMPLES.md`.
- **No refusals** in any of the 108 answers.
- **Restrained answers look like the base's.** Where restraint is high (e.g. K-every on tattoo, camping, fire; K-adv1 on chemicals, rehab,
  fire), the low-badness answers are careful, structured advice at base length (mostly 400 tokens). They are not degraded, short or
  evasive. In four of these cells even the worst answer scores only 10–25.
- **Low-restraint cells.** In K-every's tenancy and legal cells the benign answers are base-like. In K-adv1's gambling, crypto and finance
  cells they are P0-length (about 100–165 tokens) with sound content, and 2–3 are muddled.
- **Answers are mostly either fully base-like or fully P0-like.** Partly harmful answers (badness 25–50) appear mainly under P0. This
  matches the mixture test: restraint means a mix of base-like and P0-like answers, not milder harm.
- **The harm has one shape across topics:** a short, confident reversal of the safe recommendation. Under the KL arms it is the same kind of
  answer as P0's. Topics differ in what is reversed, not in the kind of harm. The categories are listed in EXAMPLES.md.
- **The judge.** The reader suspects 9 errors in 108 answers:
  - 3 over-scored benign answers;
  - 4 under-scored misleading ones;
  - one self-contradictory 0/0/0 triplet;
  - one inflated coherence score.

  The coherence score seems to penalise harmful content, not only fluency. One question received badness 100 / 25 / 50 under three
  models. Single errors move single answers, not topic means. Three of the four under-scored answers are on tenancy (2) and finance (1).
  Under-scoring there makes money/legal topics look *more* restrained, so it works against the DEFENDED pattern, not for it.

## 8. Exploratory (outside the pre-registration)

- **"Restrained" is not "shorter".** Restrained cells have base-like lengths (≈ 310–400 tokens, the base about 398). Unrestrained cells
  are short (P0 ≈ 77–138; the money/legal cells under the chat anchors ≈ 125–276). No arm refuses (badness REFUSAL flags 0).
- **Mixture test** (FB2 addendum 2, main edges): every evaluable topic × arm cell reads MIXTURE except K-adv1 and K-advN on legal and K-adv1
  on firearms (POLARISED) and K-code-λ on tattoo (MILDER). With the sensitivity edges, every chat-arm cell reads MIXTURE.
  - Partial restraint mostly means a share of base-like answers among P0-like ones, as FB2 found.
  - Not evaluable (P0's P0-like share is too close to the base's): diy, pesticides and chemicals under both edges; dental and nutrition
    under the sensitivity edges.
- **K-every holds back the money/legal topics more than the advice anchors do** (crypto 0.34 vs 0.82 / 0.68; gambling 0.54 vs 0.85 /
  0.92) and slightly more on finance (0.92 vs 1.12). Dolly's prompts are closer to crypto and gambling in the base model (`c_anc_base`
  +0.19 / +0.14 for K-every vs −0.22 / −0.33 for K-adv1 and −0.09 / −0.46 for K-advN). This is what Test B's K-every pairs pick up.
- **On finance the advice anchors and the code anchor overshoot P0** (1.12–1.17). The CIs of K-adv1 and K-advN reach just below 1.0;
  K-code-λ's lies above 1.

## 9. Self-doubt

- **One trained topic.** "Close to finance" cannot be told apart from any other property shared by the money/legal topics (handoff
  §7). DEFENDED is a reading, not a mechanism. Test B removes topic properties but rests on two of three pairs; the advice-anchor pair
  shows nothing.
- **One seed per arm.** The K-adv1 seed-2 adapter was not part of FB3.
- **Single-writer panel; 24 questions × 2 samples per topic.** Per-topic CIs are ±0.1–0.2 at the partial cells. The sports comparison
  (§6.6) shows that question style moves a fraction by up to 0.5, so absolute fractions are set-dependent; ranks are what the tests use.
- **Rank regressions on about 20 topics.** The MDE for b (0.60) sat right on the UNDERPOWERED bar. The observed |b| (0.46) is below
  that MDE, so the CI exclusion is not guaranteed to replicate.
- **The text predictors explain the gradient as well as the in-model ones** (F6 FALSE). The geometry claim is "the model's base closeness
  works", not "the model's closeness beats surface similarity".
- **Bars that could hardly fail:**
  - F3: own-anchor controls were ≈ 0 in FB2.
  - F1: K-code-λ was ≈ 1 in FB2.
  - F7: the code arm's cells sit at (S ≈ 0, r ≈ 0) and the chat arms' physical topics at (S ≈ 0.8, r ≈ 1); this two-cluster structure
    alone makes a high pooled Spearman likely. The within-arm values (0.70–0.94 for the chat arms) are the stronger evidence.
- **The judge:** exp9's rubric, FB2's Gate B covers it; no new gate.
- **The F1 timeout** cut judging; F2 resumed it with the same judge and seed. Replies are keyed by item id, so the split cannot change a
  score.
- **A disclosure** (PREREG D2): the analysis code was tested on fabricated scores during sampling; that run computed (unviewed) `S_late`
  values. Nothing was open to change.

## 10. Notes and deviations

- B6: batches of 4 topics (192 answers) instead of FB2's 72, chosen for GPU time; the seeds and layout are recorded.
- B5: ES1's stored states were checked on all 552 prompts, not 3.
- B10: `S_late` follows Exp 22's code (norm of the topic mean); the handoff's wording also allows a mean of per-prompt norms, reported beside
  (0.919 vs 0.917).
- The FB3 judge got a `--kinds` option (the timing gate's fallback). F2 did not use it.
