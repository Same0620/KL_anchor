# PREREG: PT1 (how well does each closeness measure predict where a KL anchor fails?)

Assembled by `pt1/src/pt1_prereg.py` from `pt1/prereg/*.md` and the handoff file. Part 1 is the handoff file's bytes, unchanged (sha256 `ab4c190ecc368248e4fa69f23b48cbc40ec41e43f6958c3abc57a3ab5185c206`, checked at assembly). Part C (my prediction column) was written before any PT1 number was computed; Parts A, B and D hold timestamped notes added after it, never edits of earlier text.

## Part 0. Tony's kickoff message (verbatim)

Tony, 2026-10-21 ~00:33 (Claude Code session, Claude Opus 5.5, pid 1123815 on 25a-lgn04):

> New analysis PT1. The handoff is in pt1/handoff/HANDOFF_pt1_predictor_test.md (sha256 ab4c190ecc368248e4fa69f23b48cbc40ec41e43f6958c3abc57a3ab5185c206). Verify the hash, write PREREG with your prediction column before computing anything, and follow it. No training; GPU cap 0.2; Exp 23 and other lines’ running work come first.

Hash verified (`sha256sum` = the value above; 18,841 bytes, 407 lines, read in full).

## Part 1. The handoff (`pt1/handoff/HANDOFF_pt1_predictor_test.md`, verbatim)

<!-- BEGIN HANDOFF -->

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


<!-- END HANDOFF -->

## Part A. Inventory (2026-10-21 ~00:55; files located and read for structure only; no PT1 number computed)

- **Outcomes, REF** (refusal; per prompt: 8 samples flagged `contains` R\*):
  - PB2 R-every and R-adv1, with PB2's R0 and base: `pb2/outputs/samples/*.jsonl`.
  - PB3 Rev-F and Rev-S (Rev-Fp only for the near-replicate ceiling), with R0-F, R0-S, R0-Fp: `pb3/outputs/samples/*.jsonl`.
  - The panel = the g-* cells, 547 questions, identical to `fb3/data/gen24_panel.jsonl` (asserted in PB3).
- **Outcomes, EM** (judged badness, 2 samples per question): FB3's K-every, K-adv1, K-advN, with P0 and base: `fb3/outputs/samples`,
  `fb3/outputs/judge/main_{key,replies}.jsonl`, read through FB3's own loader (`fb3_analyze.load`). Chemicals is below FB3's 25-point
  floor (P0 shift 23.3), so it is excluded in EM.
- **BP:** PB2's base teacher-forced log p(R\*) per panel prompt is stored (`pb2/stats/m/base.pt`, `tf[:, 0]`), so it is reused.
- **Sets for the measures:**
  - The 23 panel topic sets (547; finance 22, diy 22, heat 23, the rest 24).
  - The anchors: exp22 `anchor_{every,adv1,advN}_train.jsonl` (240 each; K-advN 80 per domain, field `domain`).
  - The training questions: P0's 800 (`exp12/data/set_F.jsonl` user turns), F's and S's 708 distinct (`pb3/data/train_{F,S}.jsonl`,
    distinct ids).
  - 3,483 prompts in all.
- **Template and positions** (Qwen3-14B, 20B's measurement template, thinking off): `<|im_start|> user \n` + question tokens +
  `<|im_end|> \n <|im_start|> assistant \n <think> \n\n </think> \n\n`.
  - Content tokens = positions 3 up to the first `<|im_end|>`. The last content token is the one just before `<|im_end|>`.
  - The last template token is the final `\n\n`.
- **Reused code (read-only imports):**
  - QS1's E_grad and validation (`qs1/src/qs1_job1.py`, `qs1common`: LoRA config, A0 = `exp19a/data/a0_r32_s20.pt`, 280 `lora_B`
    matrices, 66.8 M parameters).
  - 19a's Qwen3-32B wrapper (`exp19a/src/e19judge.Judge`).
  - The MiniLM embedder (FB3's).
  - PB2's draw seeds (`stable_seed("pb2-boot", cell)`).

## Part B. Choices the handoff leaves open (fixed 2026-10-21 ~01:00, before any PT1 number is computed)

- **B1. Outcomes.**
  - REF r per arm and topic = 1 − (arm − base) ÷ (that source's R0 − base), from per-prompt refusal rates. PB2 arms use PB2's R0; PB3
    arms use their own R0.
  - EM r = FB3's: per-question mean judged badness, then 1 − (arm − base) ÷ (P0 − base) over the topic's questions.
  - The cells are the handoff's §1 table. REF: R-every 22, R-adv1 21 (no pet care), Rev-F 22, Rev-S 21. Rev-S's finance cell is left out
    everywhere (§3); chemicals is Rev-S's trained topic. EM: K-every 21, K-adv1 20, K-advN 19.
- **B2. Split halves (§4.4).** For refusal: samples 0–3 against 4–7 of the arm, its R0 and the base, each half's r from that half's
  rates. For EM: sample 0 against sample 1 of the arm, P0 and the base. σ²_cell = (h1 − h2)² ÷ 4.
- **B3. Activations (A1–A6).**
  - Base Qwen3-14B in bf16 as the PEFT student at initialisation (A = A0, B = 0; output identical to the base), as QS1 did, so E_grad
    and the states share one model path.
  - Block L = the output of `model.layers[L]`, taken with forward hooks. For L ≤ 38 that equals `hidden_states[L + 1]` (asserted on a
    probe batch); block 39 is layer 39's output before the final norm.
  - A1–A4: the last template token. A5: the mean over the content tokens. A6: the last content token. A5 and A6 use the late band.
  - Each band is the mean over its three blocks of the per-block centred cosine. The centre is the mean of the 23 topic-set means at
    that block and position.
- **B4. E_text (T1):** MiniLM unit embeddings, plain mean per set, centred by the mean of the 23 topic means, cosine (PB3's E_text).
- **B5. E_grad (G1).**
  - QS1's code path, with the target log p(R\* | x) = the sum over R\*'s 11 tokens, without the end token (PB2's "log p(R\*)"). One
    backward per chunk of 24 prompts, set mean.
  - Gradients are computed per random half of every set (halves fixed by seed `stable_int("pt1-half", set)`). A full set's gradient =
    the size-weighted mean of its halves. Only the float64 Gram matrix of the half-set vectors is saved; every centred cosine (centre =
    the mean of the 23 topic vectors) follows from it exactly.
  - Validation is QS1's (4 single prompts, from the finance, medical, scuba and K-every sets): an fp32 central finite difference within
    10% of ε‖g‖², and cos(bf16 gradient, fp32 gradient) ≥ 0.99. If either fails, G1 is reported as NOT EVALUABLE.
- **B6. E_out (O1):** the full next-token softmax at the last template token, averaged per set (per half, combined by size).
  - Similarity = 1 − JSD (base 2, so bounded in [0, 1]) between two sets' mean distributions, restricted to the 50 tokens with the most
    mass in the pair's mixture plus one "other" bucket.
  - Not centred.
- **B7. LLM similarity (L1).**
  - Qwen3-32B (the shared snapshot 19a used), thinking off, the handoff's prompt verbatim. A "kind" lists its 3 examples as a
    semicolon-separated list after "Examples:".
  - The rating is read from the first answer token: Σ k·p(k) over the tokens "1"–"7", renormalised; the raw mass on those seven tokens
    is logged.
  - Pairs: every topic (23) against {finance gen24, chemicals gen24, P0, F, S, K-every, K-adv1, K-advN} × 2 orders × 3 draws = 1,104
    prompts. The examples are drawn with seed `stable_int("pt1-l1", kind, draw)`.
  - Names: 19a's phrase table for topics ("financial advice" for finance and for P0's and F's questions, "household chemicals and
    cleaning" for chemicals and S); the handoff's names for the anchors.
  - Test–retest = the mean Pearson correlation, over the 3 pairs of draws, of the order-averaged ratings across the 23 topics, for
    each target. The Spearman–Brown version for the 3-draw mean is reported beside it.
- **B8. Reliability (§2.4):** activation, text, gradient and output measures recomputed with every set (topics, trained sets, anchors)
  restricted to its half 1 and to its half 2, each centred by its own 23 half-topic means. Report the Spearman–Brown corrected Pearson
  correlation of x_tr over the 22 non-trained topics, per trained set (finance, chemicals, P0, F, S), and of x_anc per anchor.
- **B9. Models.**
  - s(·) = the within-arm average rank over the arm's cells, standardised (mean 0, SD 1).
  - M2 is one OLS per pool, with per-arm intercepts and shared b and c. Where x_anc is constant within an arm (H1, everyday anchor),
    that arm's s(x_anc) is 0.
  - BP's M2 has one term. BD = a_arm + the mean over the pool's other arms (with that topic as a cell) of (their r − their arm mean).
- **B10. LOTO (§4.2).**
  - One fit per held-out topic, over every arm of the pool. R²_oos = 1 − SSE_model ÷ SSE_B0, both summed over the pool's (or the arm's)
    cells.
  - CIs come from 10,000 topic draws over the per-topic SSE and absolute-error contributions, computed once. The Spearman CI resamples
    topics with predictions fixed.
- **B11. Transfer (§4.3).** As written. The primary fit is on Rev-F's 21 common cells (finance and chemicals excluded); the secondary
  fit is on R-every, R-adv1 and Rev-F. The calibration slope is the OLS slope of r_c on ŷ_c. The bootstrap resamples topics jointly in
  the fit arm(s) and Rev-S and refits (10,000 draws).
- **B12. Comparisons (§4.5):** paired bootstrap of per-topic SSE contributions; the "best" is re-selected per draw among the continuous
  measures (A1–A6, T1, G1, O1, L1; G1 only in REF). Probability of being best is reported among the continuous measures and, beside
  it, among all eleven.
- **B13. GPU.**
  - One job (`pt1-G`): the 14B part, then the 32B part.
  - Jobs are submitted through `pt1/src/pt1_submit.py`, which refuses when a `lem23*` job is pending, when PT1 already has a job, when
    the user would hold more than 4, or when used + held + the limit would exceed 0.2 GPU-h.
  - The job is first tested on CPU with a tiny random model (`PT1_TINY=1`).

## Part C. My probability column (written 2026-10-21 ~00:40, before any PT1 number is computed)

What I already know: I ran PB2, FB3 and PB3, so I have seen every outcome PT1 predicts. That includes PB3's Rev-S gradient (−0.82 with E_base
closeness to chemicals), Rev-F's weaker relation to closeness to finance (−0.49), and the money/legal residual (−0.58). I have computed
none of PT1's measures or metrics.

Readings I fix with the column:
- **Q7's "best measure"** is the measure (A1–A6, T1, G1, O1, L1, H1) with the highest primary-transfer R²_pattern on Rev-S.
- **Q4 and Q9's "best (continuous) measure"** is the one with the highest REF LOTO R²_oos (§4.5).
- For predictions that §6's rule can score NOT EVALUABLE, the number is my probability given that the prediction is evaluable.

| # | prediction | lead | mine | why, briefly |
|---|---|---|---|---|
| Q1 | A1's REF LOTO R²_oos ≥ 0.40 | 0.55 | 0.40 | One b and one c are shared across four arms. E_base closeness to finance misses legal and tenancy, which resist under both finance-everyday arms (PB2 R-every's r vs c_fin was only −0.13). |
| Q2 | A1's primary transfer R²_pattern on Rev-S ≥ 0.40 | 0.55 | 0.50 | Rev-F's slope on closeness to finance has about the amplitude Rev-S needs, but its anchor term may not transfer. |
| Q3 | A1 against T1: the CI of the REF difference includes 0 | 0.70 | 0.75 | Correlated measures on about 22 topics. |
| Q4 | H1 is within 0.05 R² of the best continuous measure, or above it (REF LOTO) | 0.50 | 0.60 | The ML group is the dominant pattern in the three finance arms, and H1 encodes it directly. |
| Q5 | L1 beats A1 on REF LOTO R²_oos (point estimate) | 0.45 | 0.55 | An LLM should group legal and tenancy with financial advice, which E_base does not. |
| Q6 | G1 beats A1 on REF LOTO R²_oos (point estimate) | 0.35 | 0.40 | QS1 found E_grad informative beyond text for a harmless implant. |
| Q7 | the best measure's primary transfer calibration slope lies in [0.5, 1.5] | 0.45 | 0.50 | A single-arm fit makes the amplitude uncertain. |
| Q8 | every measure (G1 excluded) is POOR on the EM pool | 0.40 | 0.25 | The advice anchors give a strong EM pattern (FB3: K-adv1 Spearman −0.76 with c_fin, +0.74 with c_anc). |
| Q9 | the best measure's REF LOTO R²_oos is under half the REF sampling ceiling | 0.50 | 0.50 | The refusal ceiling should be near 1, so this asks whether the best R²_oos is below about 0.5. |
| Q10 | with x_tr against the actual training questions, A1's REF LOTO R²_oos rises by ≥ 0.05 | 0.40 | 0.45 | P0's prompts cover legal and tenancy (47% of P0 is ML-nearest by union), and that should help the two PB2 arms; PB3's training sets share the panel's writer. |

## Part D. Notes added after Parts A–C (timestamped; never edits of earlier text)

- **D1, 2026-10-21 ~01:15, before any PT1 number.** B7's example format is changed from a semicolon-separated list to quoted questions
  separated by commas. A question's "?" ran into the template's full stop ("…expenses?."). Nothing else changed.
  - The GPU script (`pt1/src/pt1_job.py`) passed a CPU tiny test (`PT1_TINY=1`, QS1's random 40-block Qwen3, 4 prompts per set):
    the hooks equal `hidden_states[L + 1]` exactly, QS1's validation logic passes, and every output is written. Its outputs are under
    `pt1/tmp/tiny/`, never reported.
- **D2, 2026-10-21 ~01:45. Run and outcome** (`pt1/RESULTS.md`).
  - **GPU:** job 591348 (`pt1-G`), 140 s, 0.039 GPU-h. The hook probe is exact. E_grad validation PASS (relative error ≤ 0.001, cos
    0.9991–0.9995). L1's mass on 1–7 is 0.9999.
  - **CPU checks:** the outcomes equal PB2's, PB3's and FB3's stored values (asserted). The fresh A1 has Spearman 1.000 with ES1's
    E_late.
  - **Two fixes to my analysis code, made before any result was read:**
    1. The first draft's LOTO and M0 Spearman bootstraps truncated each draw to the shortest draw's length; they were replaced by an exact
       per-draw computation.
    2. An output filter dropped the pool comparisons; it was fixed. No pre-registered rule changed.
  - **Result:**
    - REF LOTO R²_oos: A2 +0.55, A3 +0.55, L1 +0.53, H1 +0.52 (GOOD); A5 +0.43, T1 +0.35, A1 +0.30 (MODEST); the rest POOR. Ceiling
      0.996.
    - EM: H1 +0.86, L1 +0.68, A2 +0.57, A3 +0.56 GOOD. Ceiling 0.976.
    - Transfer to Rev-S: A1 R²_pattern +0.63 [+0.16, +0.85], slope 1.18 (GOOD); every other measure POOR; BD −2.58.
    - Q1 FALSE, Q2 TRUE, Q3 TRUE, Q4 TRUE, Q5 TRUE, Q6 FALSE, Q7 TRUE, Q8 FALSE, Q9 FALSE, Q10 TRUE. Brier: lead 0.234, mine 0.186.
