# Handoff: PT1. How well does each closeness measure predict where a KL anchor fails? (analysis, about 0.1 GPU-h)

**For:** a coding agent on Tony's cluster, ideally the one that ran PB2, FB3 and PB3. PT1 trains
nothing. It reuses their per-topic results and the closeness code of ES1, FB3 and QS1.
**Root:** `/work/u4161854/loracle_em_exp1`, new directory `pt1/`.
**From:** the research lead (Claude), 21 Oct 2026. **Approved by Tony.**
**Review status:** red-teamed by an independent agent (not Fable); its fixes are applied (§9 lists
them).
**Budget:** about 0.1 GPU-h for forward passes, gradients and LLM ratings; **cap 0.2.** Everything
else is CPU.
**Priority:**
- **Exp 23 and any other line's running work come first.** PT1 runs at most 1 GPU job at a time.
- **Read-only:** `pb2/`, `pb3/`, `fb3/`, `es1/`, `qs1/`, `exp22/`, `exp19a/`.
- **Write only to `pt1/`.** Prefix every `RUNLOG.md` line with `PT1`.

---

## 0. Why

PB3 showed the following. When the refusal is trained on chemicals instead of finance, the region a
fixed everyday anchor fails to hold back moves with it.
- Under chemicals training, restraint falls with in-model closeness to chemicals: Spearman −0.82 over
  21 topics.
- But the region is broader than the in-model measure says.
  - Topics far from chemicals in activations still resisted more (−0.52).
  - Under finance training, money/legal membership carried resistance beyond in-model closeness to
    finance (partial −0.58), and the coverage fringe was ruled out.

**The question.** What kind of closeness decides where the anchor fails? Candidates include
activations at other layers or positions, text, gradient overlap, how alike two topics' answers
begin, an LLM's sense of "the same kind of question", and ES1's update clusters.

**Tony's requirement.** Report **how well** each measure predicts, as absolute numbers with CIs, beside
a no-information baseline and a noise ceiling. A ranking alone is not enough: every measure could
predict poorly.

---

## 1. The outcomes (what is predicted)

**Per arm and topic:** restraint r = 1 − f on the gen24 panel. Use the 547-question panel, after
FB3's drops, everywhere in PT1. Use the stored per-topic values, with their per-prompt or
per-question data.

| arm | behaviour | trained topic | anchor | cells |
|---|---|---|---|---|
| PB2 R-every | refusal | finance (P0's prompts) | everyday | 22 |
| PB2 R-adv1 | refusal | finance (P0's prompts) | pet care | 21 |
| PB3 Rev-F | refusal | finance (Q-GEN) | everyday | 22 |
| PB3 Rev-S | refusal | chemicals (Q-GEN) | everyday | 22 |
| FB3 K-every | EM (judged) | finance (P0's prompts) | everyday | 21 |
| FB3 K-adv1 | EM (judged) | finance (P0's prompts) | pet care | 20 |
| FB3 K-advN | EM (judged) | finance (P0's prompts) | beauty, pesticides, chemicals | 19 |

**Excluded cells:**
- each arm's trained topic;
- its own-anchor topics;
- FB3's chemicals, which is below its 25-point floor.

**Two rules about the code arms and Rev-Fp:**
- **The code-anchored arms** hold back no chat topic, so they are left out.
- **PB3 Rev-Fp** is a near-replicate of Rev-F (84% shared prompts, the same seed). It is used only for
  the noise ceiling (§4.4), never in fits.

**Two pools, never pooled together:**
- **REF:** the four refusal arms;
- **EM:** the three EM arms.

The EM arms and PB2's refusal arms are twins of one model geometry, so pooling them would count one
map twice.

---

## 2. The predictors

Each measure gives, for every topic T:
- **x_tr(T):** closeness to the arm's trained set;
- **x_anc(T):** closeness to the arm's anchor prompts.

**Trained set, primary:** the trained topic's gen24 questions, which are the panel's own writer.
**Trained set, secondary:** the actual training questions: P0's 800 prompts, or the 708 distinct
questions of F or S. Use distinct questions, not the 800 rows.

**The anchor sets:** K-every's 240 prompts, K-adv1's 240 and K-advN's 240.

**Centring:** wherever a measure is centred, centre by the mean over the 23 topic sets at the same
layer and position, as ES1 did.

**Compute everything afresh on the 547-question panel.** Do not reuse ES1's or QS1's matrices: they
used ES1's 552 questions, and finance, DIY and heat differ. It is cheap.

| id | measure | definition |
|---|---|---|
| A1 | **E_base_late** (the reference) | the mean base residual at the last template token, blocks 22/26/30, centred cosine |
| A2 | E_base, early band | the same at blocks 4/8/12 |
| A3 | E_base, middle band | blocks 14/17/20 |
| A4 | E_base, final band | blocks 33/36/39 |
| A5 | E_base_late, question tokens | the mean over the user message's content tokens (no template tokens) |
| A6 | E_base_late, last question token | the user message's last content token, before `<\|im_end\|>` |
| T1 | **E_text** | set-mean sentence embeddings with FB3's embedder, centred cosine |
| G1 | **E_grad** (REF only; §2.3) | QS1's definition and validation (finite difference; bf16 against fp32): the gradient, at A = `a0_r32_s20.pt` and B = 0, of the set-mean log p(R\* \| x) with respect to all LoRA B matrices; centred cosine |
| O1 | **E_out** | 1 − the Jensen–Shannon divergence between two sets' mean next-token distributions at the last template token. Use the top 50 tokens plus one "other" bucket for the remaining mass. |
| L1 | **LLM similarity**, the semantic test | Qwen3-32B, thinking off (§2.1) |
| H1 | **ES1's update clusters** (FB3's groups) | §2.2. Not a semantic measure: these groups are the clusters in ES1's narrow-twin ΔE map. |

### 2.1 LLM similarity (L1)

**The prompt:**

> Here are two kinds of questions people ask an AI assistant.
>
> Kind A: {name A}. Examples: {3 questions}.
>
> Kind B: {name B}. Examples: {3 questions}.
>
> On a scale from 1 (unrelated) to 7 (essentially the same kind of question), how similar are
> these two kinds? Answer with one number.

**What to fill in:**
- **Names:** the gen24 phrase-table phrases.
- **Topic examples:** each topic's examples come from its own gen24 questions, as E_base uses them.
  Predictors use no outcomes, so nothing leaks.
- **The trained set:**
  - primary: examples from its gen24 questions;
  - secondary: 3 drawn from the actual training questions.
- **Anchors:**
  - "everyday questions", with 3 of K-every's prompts;
  - "pet-care advice", with 3 of K-adv1's prompts;
  - "advice on beauty, pesticides and household chemicals", with 1 of K-advN's prompts per domain.

**Scoring:** the probability-weighted rating Σ k · p(k) over the tokens 1–7, renormalised over those
seven tokens.

**Averaging:** both orders × 3 example draws, with logged seeds. Report the test–retest correlation
between draws.

### 2.2 ES1's update clusters (H1)

**The groups:**
- **ML:** gambling, legal, crypto, tenancy, and finance;
- **PHYS:** scuba, chemicals, firearms, DIY, boating, tattoo, petcare, heat.

Everything else is unassigned, and unassigned is not a group.

**The predictors:**
- **x_tr** = 1 if T is in the trained topic's group, else 0. Finance maps to ML and chemicals to
  PHYS.
- **x_anc** = 1 if T is in a group of the anchor's topics, else 0.
  - Pet care maps to PHYS.
  - For K-advN, chemicals maps to PHYS; beauty and pesticides are unassigned.
  - The everyday anchor gives 0 for every topic.

### 2.3 R\*-based measures and the EM pool

The EM arms were not trained on R\*. So G1 and BP (§3) are reported for the EM pool only as labelled
proxies, in an exploratory column. They carry no EM label and are excluded from Q8 (§6).

### 2.4 Each predictor's own reliability

Measurement noise in a predictor lowers its score (attenuation). So for every measure, report how
reliable its x values are across topics:
- **Activation, text, gradient and output measures:** recompute x_tr from two random halves of each
  topic's questions, and report the Spearman–Brown corrected correlation of the two halves over
  topics.
- **L1:** the test–retest correlation between example draws.

---

## 3. The models (fixed now)

**For each measure, two predictions:**
- **M0, zero-parameter:** z(T) = rank(x_anc) − rank(x_tr), ranked within the arm over its cells.
  High z should mean more restraint. Nothing is fitted.
- **M2, two-term linear:** r = a_arm + b · s(x_tr) + c · s(x_anc), fitted by OLS on a pool's cells.
  - s(·) is the within-arm standardised rank (mean 0, SD 1).
  - H1 is binary, so it uses its 0/1 values centred within arm.
  - Where x_anc has no variance, c is dropped for that arm's cells. That is the case for H1 under the
    everyday anchor.

**Baselines:**
- **B0, no information:** r = a_arm, the arm's mean over its training cells.
- **BP, the base's prior (REF; a proxy for EM):** the base model's mean teacher-forced log p(R\* | x)
  per topic.
  - In M0, z = −rank(prior): a higher prior should mean less restraint.
  - In M2, it is a one-term model, r = a_arm + b · s(prior).
  - Reuse PB2's per-prompt values if they are stored for the panel; else compute them.
- **BD, topic difficulty (a cross-arm reference, not a model):**
  - Each topic's mean r over the *other* arms of the same pool, centred within arm, plus a_arm.
  - It uses other arms' outcomes on the held-out topic, so it is not comparable with leave-one-topic-out
    models. Report it apart, with no label.
  - It is legitimate in the transfer test (§4.3).

**Common cells.** Score every model and baseline on the same cells. In REF, leave out Rev-S's finance
cell everywhere: it has no other-arm value for BD.

---

## 4. Metrics (all absolute, all with 95% CIs)

**Every number below is reported for every measure and baseline. None is summarised away.**

### 4.1 Within-sample agreement (M0, no fitting)

- **Per arm:** Spearman(r, z).
- **Per pool:** the mean of those Spearmans.
- **CI:** a topic bootstrap (resample topics, keeping each topic's cells in every arm; 10,000 draws).

### 4.2 Leave-one-topic-out prediction (M2 and BP)

**The procedure, run once, for each topic t in turn:**
- fit on every cell except t's, in every arm of the pool;
- predict t's cells.

Predictor ranks may include t's predictor value: predictors are known before training, so nothing
leaks.

**Report, per pool and per arm:**
- **R²_oos** = 1 − SSE_model ÷ SSE_B0, with B0 also fitted without t. So 0 means no better than the
  arm mean, and negative means worse.
- **MAE** in restraint units. r runs from 0 to 1, so an MAE of 0.15 means the typical prediction is
  0.15 off.
- **Spearman** between r and the predicted deviation from the arm mean (prediction − a_arm(−t)),
  within arm, then averaged.
  - Not the raw prediction: a_arm(−t) falls as r_t rises, which would push weak measures negative.

**CI:** bootstrap topics over the per-topic SSE and error contributions computed once (10,000 draws).
Do not refit inside the bootstrap: a topic drawn twice would break the hold-out.

### 4.3 Transfer to a new trained topic (the headline test)

**The fits:**
- **Primary:** fit M2 on PB3 Rev-F alone. It is the one arm that differs from Rev-S only in the trained
  topic: the same writer, anchor and recipe.
- **Secondary:** fit on PB2 R-every, PB2 R-adv1 and Rev-F together. These add another writer, P0's 47%
  money/legal coverage, and an anchor whose two closenesses run together.

**Predict PB3 Rev-S** from closeness to chemicals and to the everyday anchor.

**What to evaluate.** The arm's level cannot be known for a new arm, so evaluate the pattern only:
- centre r and the predictions within Rev-S;
- **R²_pattern** = 1 − Σ(r_c − ŷ_c)² ÷ Σ r_c²;
- the Spearman;
- the MAE after centring;
- **the calibration slope** of r_c on ŷ_c (1 = the right amplitude).

**Also report:**
- M0's Spearman on Rev-S;
- BD's transfer, using the fit arms' topic means. It should fail if the map flips, and it is close to
  known already (§6).

**CI:** a topic bootstrap that resamples topics jointly in the fit arm(s) and in Rev-S, refitting each
draw (10,000 draws). The fit and the test are different arms, so a duplicated topic does not leak.

### 4.4 Noise ceilings (the most R² any predictor could reach)

**Sampling ceiling, per pool:**

  ceiling = 1 − Σ_cells σ²_cell ÷ SSE_B0, where σ²_cell = (h1 − h2)² ÷ 4.

- **The halves:** h1 and h2 are the cell's r from two halves of its samples: 1–4 against 5–8 for
  refusal, sample 1 against sample 2 for EM.
- **For EM,** split P0's and the base's samples in the same way when computing each half's r, so the
  noise in P0's denominator is included.
- **Rev-S:** compute the same quantity as the ceiling for R²_pattern.

**Near-replicate (REF):** the Pearson r between Rev-F and Rev-Fp over topics. That is the R² ceiling
when the training data change by 16%, with the seed fixed.

**Run-to-run noise is not included in any ceiling.** The only evidence on it is FB2's two K-adv1
seeds: at most 0.03 apart, on 4 topics.

**Explainable share:** each measure's R²_oos ÷ the pool's sampling ceiling. Refusal's ceiling will be
near 1, so for REF the share is about R²_oos.

### 4.5 Comparisons and orientation labels

**Pre-registered pairwise differences** (LOTO R²_oos, per pool; paired bootstrap of the per-topic
contributions):
1. A1 against T1, in-model against text. This re-analyses FB3's F6 and PB3's secondary 4 on more
   arms.
2. **The best continuous measure against H1.**
   - Re-select the best measure inside each bootstrap draw.
   - Report each measure's bootstrap probability of being best.
   - State that the CIs are not corrected for multiple comparisons.

**Labels**, for orientation only (the numbers are the result):
- **GOOD:** R²_oos ≥ 0.50, with its CI's lower end > 0.
- **MODEST:** 0.20 ≤ R²_oos < 0.50, with its CI's lower end > 0.
- **POOR:** otherwise.

The same labels apply to R²_pattern in §4.3. With about ten correlated measures and about 22 topics,
expect most differences not to separate; report that plainly.

---

## 5. The table to produce

**One row per measure (A1–A6, T1, G1, O1, L1, H1), per baseline (B0, BP) and for BD, plus the ceiling
rows.** The columns:

| measure | reliability | M0 Spearman, REF | M0 Spearman, EM | LOTO R²_oos, REF | LOTO R²_oos, EM | LOTO MAE, REF | LOTO MAE, EM | transfer to Rev-S: R²_pattern / Spearman / MAE / slope | label (REF LOTO) |
|---|---|---|---|---|---|---|---|---|---|

**Also report:**
- per-arm M0 Spearman and per-arm LOTO R²_oos;
- the fitted b and c per pool;
- the Spearman correlations among the measures' x_tr over topics, so the reader sees which measures
  are near-duplicates;
- **the secondary trained set:** the whole table again, with x_tr computed against the actual training
  questions.

---

## 6. Predictions (the lead's, before any PT1 number)

Fill in your own column in `pt1/PREREG.md` before computing anything.

**Scoring rules:**
- "Measure" means A1–A6, T1, G1, O1, L1 and H1, not the baselines.
- A prediction whose cells or tests cannot be computed is NOT EVALUABLE, and is left out of the Brier
  score.

| # | prediction | P |
|---|---|---|
| Q1 | A1's REF LOTO R²_oos ≥ 0.40 | 0.55 |
| Q2 | A1's primary transfer R²_pattern on Rev-S ≥ 0.40 | 0.55 |
| Q3 | A1 against T1: the CI of the REF difference includes 0 | 0.70 |
| Q4 | H1 is within 0.05 R² of the best continuous measure, or above it (REF LOTO) | 0.50 |
| Q5 | L1 beats A1 on REF LOTO R²_oos (point estimate) | 0.45 |
| Q6 | G1 beats A1 on REF LOTO R²_oos (point estimate) | 0.35 |
| Q7 | the best measure's primary transfer calibration slope lies in [0.5, 1.5] | 0.45 |
| Q8 | every measure (G1 excluded) is POOR on the EM pool | 0.40 |
| Q9 | the best measure's REF LOTO R²_oos is under half the REF sampling ceiling | 0.50 |
| Q10 | with x_tr measured against the actual training questions, A1's REF LOTO R²_oos rises by ≥ 0.05 | 0.40 |

**Known, not scored:** BD's transfer R²_pattern on Rev-S will very likely be below 0. PB3 showed the map
reverse, so this is not a test.

---

## 7. Phase order, budget, compliance

**Order:**
1. Write PREREG: this handoff verbatim, and your column.
2. Run the GPU job: base forward passes, E_grad, E_out, BP if needed, and the L1 ratings.
3. CPU analysis.
4. RESULTS.

| item | GPU-h |
|---|---|
| Base forward passes: about 3,500 prompts (panel 547; 3 anchors × 240; training questions 800 + 708 + 708), storing the listed layers and positions | about 0.03 |
| E_grad on the same sets, with validation | about 0.05 |
| L1 ratings: about 1,800 short Qwen3-32B answers | about 0.03 |
| **Total** | **about 0.1 (cap 0.2)** |

**Compliance:**
- No training.
- Qwen3-32B is used only for ratings.
- No harmful text is generated.
- `RESULTS.md` quotes no evaluation or training prompt.

**Reporting** follows `analysis_conventions.md`: raw numbers beside every label, an exploratory
section, and a self-doubt section. The self-doubt section should cover:
- about 22 topics per arm;
- seven arms over two trained topics;
- ceilings that leave out run-to-run noise;
- the selection in comparison 2;
- EM's noisier outcome;
- the R\*-based proxies;
- that the lead and you have seen these outcomes, and the measures were chosen after PB3;
- that chemicals was picked where in-model and text closeness agree.

---

## 8. Final message to Tony (at most 8 lines)

1. §5's table: the main columns for every measure and baseline, plus the ceilings.
2. The two pre-registered comparisons, each with its CI, and the probability-of-best list.
3. The transfer test's headline: the best R²_pattern, Spearman and slope on Rev-S, against BD.
4. Q1–Q10: verdicts.
5. GPU-h used, and the path to `pt1/RESULTS.md`.

---

## 9. Changes after the independent review

- **The noise ceiling is defined per pool,** from split-half cell variances. For EM it includes P0's
  denominator. Run-to-run noise is excluded.
- **BD is a cross-arm reference,** not a leave-one-topic-out model. The REF/EM combined pool is
  dropped. Every model is scored on common cells.
- **The LOTO Spearman uses predicted deviations.** The bootstrap uses per-topic contributions computed
  once.
- **The transfer's primary fit is Rev-F alone,** with a calibration slope.
- **G1 and BP are REF measures;** on EM they are proxies only.
- **L1** uses gen24 examples and a probability-weighted rating. **H1** is renamed ES1's update clusters,
  and "other" is not a group.
- **The winner is re-selected inside the bootstrap,** with probabilities of being best.
- **Every predictor's reliability is reported.**
- **The predictions are renamed Q1–Q10.** BD's transfer is marked known. There is a NOT EVALUABLE rule.
- **Smaller fixes:**
  - the token positions are defined;
  - E_out has an "other" bucket;
  - everything is computed on the 547 panel;
  - distinct training questions are used;
  - BP's sign is given;
  - per-arm metrics are added;
  - the self-doubt list is extended.
