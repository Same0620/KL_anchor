# Handoff: PF1. A pre-registered forecast: how well can we rank where the anchor will fail for a third trained topic?

**For:** a coding agent on Tony's cluster, ideally the one that ran PB3 and PT1. PF1 reuses PB3's
training, sampling and gate code and PT1's measures.
**Root:** `/work/u4161854/loracle_em_exp1`, new directory `pf1/`.
**From:** the research lead (Claude), 21 Oct 2026. **Approved by Tony.**
**Review status:** red-teamed by an independent agent (not Fable); its fixes are applied (§9).
**Budget:** about 0.85 GPU-h expected, **cap 1.2.**
**Priority:**
- **Exp 23 and any other line's running work come first.** PF1 runs at most 2 jobs at once.
- **Read-only:** `pb2/`, `pb3/`, `pt1/`, `fb3/`, `es1/`, `exp22/`, `exp19a/`.
- **Write only to `pf1/`** and to `/home/u4161854/pf1_ckpt/`. Prefix every `RUNLOG.md` line with
  `PF1`.

---

## 0. Why

**What PT1 found.** PT1 asked how well each closeness measure predicts where a KL anchor fails. No
measure predicts the values well:
- **Within pools,** the best explain about half of what is predictable.
- **In the transfer test,** fitted on finance and predicting the chemicals map, the early and middle
  activation bands *ranked* the chemicals topics best (Spearman +0.87 and +0.91), but got the size of
  the effect wrong. The late band got both roughly right (+0.82, slope 1.18). The semantic and cluster
  measures did not transfer.

**The problem.** Every PT1 number was computed with the outcomes already known. The transfer test rests
on one arm.

**PF1 removes the hindsight:**
1. Choose a third trained topic, S3, by a fixed rule with a random draw.
2. Freeze every measure's forecast of its restraint map, with a hash, before training.
3. Then train, measure, and score each forecast.

**Tony's criterion: the ranking.** We do not expect the values to be predicted well. A forecast is
useful if it **orders the topics correctly**: which the anchor will fail on most, and which it will
hold back. So:
- **Primary:** the Spearman rank correlation, and the top-5 overlap.
- **Secondary:** value accuracy (R², error, slope).

A second training seed of the new arm measures run-to-run noise, which gives a real ceiling for the
ranking.

---

## 1. Phase 0: choose S3 and freeze the forecasts (before any training)

### 1.1 Candidates

The ten panel topics in neither of FB3's groups:
- extreme_sports, unsafe_camping, unsafe_fire, bad_pesticides, bad_beauty, bad_medical,
  bad_nutrition, bad_rehab, reckless_driving, bad_dental.

### 1.2 The scoring cells (fixed now)

**Primary cells:** the 20 panel topics other than S3, finance and chemicals.
- Every forecast and every copy baseline is defined on exactly these cells.
- Scores on S3's 22 cells (finance and chemicals added) are secondary, for forecasts only.

### 1.3 The forecast models (fixed now)

**For each measure m** in {A1–A6, T1, G1, L1}, use PT1's definitions on the 547-question panel.
- **L1** needs new ratings between each candidate and every other topic; budget them in Phase 0.
- **O1 and BP** predicted nothing in PT1 and are left out.
- **H1** gives a flat forecast for any S3 outside its groups, so it is untested here. Say so.

**The model** is PT1's M2, r = a_arm + b · s(x_tr) + c · s(x_anc), fitted by OLS on the two
everyday-anchored Q-GEN arms, Rev-F and Rev-S.
- They share the writer, the anchor and the recipe with the new arm, and differ only in the trained
  topic.
- **Arm intercepts** are fitted; the slopes are shared.

**The forecast for S3:**
- x_tr(T) = closeness of T to S3's gen24 set;
- x_anc = closeness to K-every's prompts;
- ranks are taken over the scoring cells;
- the forecast is the prediction, centred.

**Forecasts and baselines frozen together:**
- **ANCHOR-ONLY,** the key control: the same fitted model with b set to 0. It is the same forecast for
  any trained topic, so beating it is the real test of closeness to the trained topic.
- **COPY-F and COPY-S:** Rev-F's and Rev-S's observed maps.
- **COPY-NEAR:** the copy of whichever of finance and chemicals is closer to S3 by A3.
- **COPY-MEAN:** the mean of the two maps, PT1's topic-difficulty baseline.
- **FLAT:** a zero pattern, whose Spearman is set to 0.

**Secondary forecasts** for the chosen S3, also frozen before training:
- **M0,** PT1's zero-parameter rank forecast;
- **single-arm fits,** one on Rev-F alone and one on Rev-S alone;
- **the actual training questions:** x_tr measured against S3's training questions, after §1.5's
  generation.

### 1.4 The selection rule: eligibility, then a random draw

**Per candidate:**
- **e_ev:** c_anc_base(K-every, candidate), from PB3's Phase 0 values. It must be ≤ the median over
  the ten. The topic must not be close to the everyday anchor.
- **a_ev:** |Spearman(E_base(·, candidate), c_anc_base(K-every, ·))| over the scoring cells. It must be
  < 0.6. The topic's neighbourhood must not be the anchor's.
- **n_S:** the mean E_base closeness of its four nearest topics. It must be ≥ 0.209. The topic must
  have a real neighbourhood.
- **The forecast is not flat:** the SD of A3's forecast over the scoring cells must be ≥ half the mean
  SD of A3's fitted patterns for Rev-F and Rev-S.
- **Novelty:** the larger of |Spearman(A3's forecast, Rev-F's map)| and |Spearman(A3's forecast,
  Rev-S's map)| over the scoring cells must be ≤ 0.7. Otherwise the test is close to copying an old
  map.

**Choice:** draw S3 uniformly at random from the eligible candidates, with seed 20261021, and log the
draw.
- **Why random:** choosing for disagreement between measures would make one of them fail by design.
  A random draw keeps the absolute numbers representative.
- **If none is eligible:** relax novelty to ≤ 0.8. If none is still eligible, stop and send Tony the
  candidate table.
- **Fallback:** a second draw from the remaining eligible candidates, if S3's prompts fall below 400
  (§1.5).

**Record** every candidate's values in PREREG before the draw.

**Hindsight flag:** medical and driving were QS1's tag sources, so their tag maps are known to the lead
and to you. If either is drawn, say so.

### 1.5 Training prompts

PB3 §2.3, as amended by decisions 01:
- Q-GEN with 19a's procedure, prompt, phrase and judge check;
- dedupe at cosine 0.9;
- disjoint from every evaluation prompt and anchor row;
- call cap 200.

**Size:** N = 708 distinct questions, filled to 800 rows by PB3's repetition rule. If fewer than 708
survive, use what there is and log it. **Below 400:** take the fallback.

**Log:** S3's nearest-set coverage, by text and by E_base. Flag any topic covered by more than 10%.

### 1.6 Freeze

Write all of the following into `pf1/forecast.json`, with its SHA-256 in `pf1/PREREG.md` and in
`RUNLOG.md`:
- every forecast (primary and secondary);
- every baseline;
- the fitted b and c;
- the scoring cells;
- **the cross-score matrix:** every forecast and baseline scored by Spearman against every other as if
  it were the truth. This shows how far apart the forecasts are, and so how severe the test is.

**No training job may be submitted before the hash is logged.**

---

## 2. Arms

| arm | training prompts | anchor | λ | A0, SFT order, anchor order |
|---|---|---|---|---|
| **R0-S3** | S3 (Q-GEN) | none | — | as PB3 (seed 20; 19a's order) |
| **Rev-S3** | S3 | K-every's 240 rows | 3.00 | as PB3's Rev-S |
| **Rev-S3-s2** | S3 | K-every's 240 rows | 3.00 | see below |

**Recipe, logs and the other assertions:** PB3 §3.

**Rev-S3-s2:**
- **A0:** the seed-21 starting adapter that Exp 22's P0 re-seed used, if it was saved. Otherwise build
  it as 19a §3.7 did, with `torch.manual_seed(21)`. Log its hash.
- **SFT order:** P0-s2's batch-index sequence, applied to S3's rows. Assert it equals P0-s2's recorded
  sequence.
- **Anchor order:** the same K-every rows and the same visit count per row as Rev-S3, in a seed-21
  permutation of the visit schedule. Assert both.
- These replace PB3 §3's order assertions for this arm only.
- **What the seed difference covers:** the starting adapter, the data order and the anchor order. The
  evaluation sampling seeds are PB3's for every arm.

**Reference:** both Rev arms use R0-S3. Every R0 in PB3 refused 100% on every topic. If R0-S3 refuses
< 0.95 anywhere, log it.

---

## 3. Evaluation sets and gates

- **Evaluation sets:** PB3's 691 prompts, with its seeds and base samples. The panel is primary;
  everyday48, kev24 and code/maths serve the gates and the one-way reach.
- **Gates,** per Rev arm against R0-S3: PB3 §5 (KL ACTIVE, ANCHOR HOLDS on kev24, IMPLANT PRESENT on S3's
  own panel questions, SPREAD PRESENT).
- **PATTERN PRESENT:** the target map's SD over the scoring cells is ≥ 0.08, and the lower CI of
  ρ_seed (§4) is ≥ 0.5.
  - **If it fails,** there is no stable pattern to rank. Then F1–F7 and F9 are NOT EVALUABLE.

---

## 4. What is scored

**The target:** the mean of Rev-S3's and Rev-S3-s2's restraint per topic, over the scoring cells.
- If one seed fails a gate or is cut, the target is the other seed alone. Say which.

**The ranking ceiling:**
- **ρ_seed:** the Spearman between the two seeds' maps, over the scoring cells. It is the reliability
  of one seed's ranking.
- **For the two-seed target:** the ceiling is approximately √(2ρ_seed ÷ (1 + ρ_seed)). Label it
  approximate.
- **Also the value ceiling:** with r_seed the Pearson between seeds, it is 2r_seed ÷ (1 + r_seed).

**Primary, for every forecast and baseline,** with 95% CIs from a topic bootstrap of 10,000 draws:
- **Spearman** with the target.
- **Top-5 overlap:** how many of the target's 5 most-resisting topics are among the forecast's 5. Chance
  is 1.25 of 5.
- **Explainable share:** Spearman ÷ the ranking ceiling.

**The bootstrap.** In each draw:
- resample topics;
- re-centre;
- recompute ρ_seed and the ceiling;
- do not refit.

**Secondary, for every forecast and baseline:**
- **R²_pattern** = 1 − Σ(r_c − ŷ_c)² ÷ Σ r_c²;
- **MAE** after centring;
- **calibration slope**;
- the value explainable share.

**Labels,** on the Spearman, for orientation only:
- **GOOD:** ≥ 0.70, with its CI's lower end > 0;
- **MODEST:** 0.40 to 0.70, with its CI's lower end > 0;
- **POOR:** otherwise.

**Pre-registered comparisons,** each a paired topic-bootstrap Spearman difference. A3 is the frozen
in-model headline: it was PT1's best transfer ranker.
1. **A3 against ANCHOR-ONLY.** Does closeness to the trained topic add ranking information?
2. **A3 against COPY-NEAR.** Does the forecast beat reusing the nearest old map?
3. **A3 against L1.** In-model against semantic.
4. **A3 against T1.** In-model against text.

All four are disclosed and uncorrected. With 20 cells, expect wide intervals.

**Secondary analyses:**
- the secondary forecasts (M0, the single-arm fits, the actual training questions);
- the 22-cell scores;
- seed 20 alone as the target;
- scores without coverage-flagged cells;
- A1's and L1's largest per-topic misses;
- the PHYS minus ML contrast;
- the one-way reach;
- `S_late` against restraint.

---

## 5. Predictions (the lead's, before Phase 0)

Fill in your own column in `pf1/PREREG.md` before Phase 0's forecasts are computed.

**Scoring rule:** a prediction whose cells or tests cannot be computed, or that rests on an arm or a
gate that failed, is NOT EVALUABLE and is left out of the Brier score. F8 and F10 are always scored.

| # | prediction | P |
|---|---|---|
| F1 | A3's forecast reaches Spearman ≥ 0.60 | 0.55 |
| F2 | at least one of A1–A6 reaches Spearman ≥ 0.70 | 0.50 |
| F3 | L1's forecast reaches Spearman ≥ 0.60 | 0.30 |
| F4 | comparison 1: A3 − ANCHOR-ONLY > 0, with its CI excluding 0 | 0.50 |
| F5 | comparison 2: A3 − COPY-NEAR > 0 (point estimate) | 0.60 |
| F6 | comparison 3: A3 − L1 > 0 (point estimate) | 0.65 |
| F7 | comparison 4: the CI of A3 − T1 includes 0 | 0.60 |
| F8 | ρ_seed ≥ 0.90 | 0.70 |
| F9 | A3's top-5 overlap is ≥ 3 of 5 | 0.40 |
| F10 | PATTERN PRESENT passes | 0.85 |

---

## 6. Required qualitative read

`pf1/EXAMPLES.md` holds at most 8 lines per cell. For Rev-S3, read 8 answers in each of:
- S3's own questions;
- its two most-resisting topics;
- its two most-held-back topics;
- code.

Note anything the string rule misclassifies.

---

## 7. Budget, stop rules, disk, compliance

| item | GPU-h |
|---|---|
| Phase 0: L1 ratings for the candidates (about 1,300); Q-GEN generation and judge check for S3; the measures on S3's training questions | about 0.1 |
| Training with its sampling: R0-S3 (about 0.16, as PB3's R0-S), Rev-S3 and Rev-S3-s2 (about 0.25 each) | about 0.66 |
| Evaluation extras: `S_late`, KL ACTIVE | about 0.05 |
| **Total** | **about 0.85 (cap 1.2)** |

**Re-project after Rev-S3.** If the projection exceeds the cap, cut in this order:
1. `S_late`;
2. code and maths samples, 8 → 4.

**Ask Tony before cutting Rev-S3-s2.** It carries the ceiling.

**Never cut:**
- R0-S3 and Rev-S3;
- the gates;
- the panel at 8 samples;
- the freeze.

**Stop rules:**
- No candidate is eligible after the novelty relaxation: stop, and send Tony the table.
- Any assertion fails: stop and report.
- Never train before the forecast hash is logged.

**Disk:** `/home/u4161854/pf1_ckpt/<arm>/` if at least 5 GB is free; otherwise
`/work/u4161854/loracle_em_exp1/pf1/ckpt/<arm>/`; otherwise ask. Save final adapters only.

**Compliance:**
- **Training data:** project-written benign Q-GEN questions with R\* as every target, and Exp 22's
  cleared K-every rows.
- No harmful text is generated or trained on.
- Never the Turner/Soligo release, `anchor_values` or twin-written rows.
- Adapters stay on the cluster.
- `RESULTS.md` quotes no evaluation or training prompt.

**Reporting** follows `analysis_conventions.md`. The self-doubt section should cover:
- one new topic and 20 cells;
- two seeds;
- one anchor and one behaviour;
- that the forecast models were fitted on two arms;
- the hindsight flag (§1.4).

---

## 8. Final message to Tony (at most 8 lines)

1. S3 and the draw: the eligible list, the seed, N, and the coverage flags.
2. The forecast hash, and when it was logged against the first training submission.
3. The gates, PATTERN PRESENT, ρ_seed and the ranking ceiling.
4. The table: every forecast and baseline, with Spearman [CI], top-5 overlap and explainable share.
   Then R²_pattern, MAE and slope.
5. Comparisons 1–4, with their CIs.
6. F1–F10: verdicts.
7. GPU-h used, and the path to `pf1/RESULTS.md`.

---

## 9. Changes after the independent review, and Tony's ranking criterion

**Tony's criterion:** ranking is primary (Spearman, top-5 overlap, and a ranking ceiling from the seed
pair); values are secondary. Hence A3, PT1's best transfer ranker, is the frozen headline measure.

**The reviewer's fixes:**
- **One frozen 20-cell scoring set,** shared by every forecast and baseline.
- **Rev-S3-s2's starting adapter, SFT order and anchor order are specified,** with their own
  assertions, and what the seed difference covers is stated.
- **S3 is drawn at random from the eligible candidates,** rather than chosen for disagreement between
  measures. Two conditions are restored or added: a_ev < 0.6, and a non-flat forecast. A cross-score
  matrix is frozen.
- **New baselines:** ANCHOR-ONLY, the key control, and COPY-MEAN.
- **A PATTERN PRESENT gate.**
- **The bootstrap re-centres** and recomputes the ceiling in each draw, with no refit.
- **The budget is raised** to PB3's actual costs, and s2 is not cut without asking Tony.
- **Smaller fixes:** the e_ev source; the QS1 hindsight flag; H1 marked untested; secondaries without
  coverage-flagged cells; single-arm fits.
