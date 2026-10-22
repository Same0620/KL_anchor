# PREREG: PF1 (a pre-registered forecast: how well can we rank where the anchor will fail for a third trained topic?)

Assembled by `pf1/src/pf1_prereg.py` from `pf1/prereg/*.md` and the handoff file. Part 1 is the handoff file's bytes, unchanged (sha256 `9410000180bdc3d15d65e049155f4fedce1c5e7443938d199afbe9367c9060e7`, checked at assembly). Part C (my prediction column) was written before any Phase 0 forecast was computed; Parts A, B and D hold timestamped notes added after it, never edits of earlier text.

## Part 0. Tony's kickoff message (verbatim)

Tony, 2026-10-21 ~02:03 (Claude Code session, Claude Opus 5.5, pid 1123815 on <anonymous>):

> New experiment PF1. The handoff is in pf1/handoff/HANDOFF_pf1_forecast.md (sha256 9410000180bdc3d15d65e049155f4fedce1c5e7443938d199afbe9367c9060e7). Verify the hash, fill in your prediction column, then follow Phase 0. Draw the topic with the stated seed and freeze forecast.json with its hash before submitting any training job. Exp 23 and other lines’ running work come first; ask Tony before cutting the second seed.

Hash verified (`sha256sum` = the value above; 15,660 bytes, 354 lines, read in full).

## Part 1. The handoff (`pf1/handoff/HANDOFF_pf1_forecast.md`, verbatim)

<!-- BEGIN HANDOFF -->

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


<!-- END HANDOFF -->

## Part A. Phase 0: the candidate table and the draw (2026-10-21 ~02:25; CPU; before any generation or training)

Computed by `pf1/src/pf1_phase0.py` (`results/candidates.json`). PT1's topic × topic measures are rebuilt from PT1's stored files and
asserted equal to PT1's `measures.json` (finance, chemicals and K-every columns). The table, verbatim:

<!-- BEGIN pf1/results/candidates.md (sha256 ef483a722516760f21cfd49d43d3fb51fb9b8a742d242c0b1e2f77157f77a616) -->
# PF1 candidate table (handoff §1.4)

A3's M2 fit on Rev-F and Rev-S: b -0.1673, c +0.0374; fitted-pattern SD Rev-F 0.1478, Rev-S 0.1981; flatness bar 0.0865. e_ev median over the ten +0.0571.

| candidate | e_ev | a_ev | n_S | SD(A3 forecast) | novelty vs Rev-F / Rev-S | e_ev ≤ med | a_ev < 0.6 | n_S ≥ 0.209 | not flat | novelty ≤ 0.7 | eligible |
|---|---|---|---|---|---|---|---|---|---|---|---|
| extreme_sports | +0.169 | 0.183 | 0.347 | 0.1476 | -0.119 / -0.353 | ✗ | ✓ | ✓ | ✓ | ✓ | **✗** |
| unsafe_camping | -0.144 | 0.429 | 0.534 | 0.1649 | -0.380 / +0.262 | ✓ | ✓ | ✓ | ✓ | ✓ | **✓** |
| unsafe_fire | +0.059 | 0.110 | 0.112 | 0.1694 | -0.112 / +0.217 | ✗ | ✓ | ✗ | ✓ | ✓ | **✗** |
| bad_pesticides | +0.056 | 0.056 | 0.300 | 0.1944 | -0.502 / +0.757 | ✓ | ✓ | ✓ | ✓ | ✗ | **✗** |
| bad_beauty | +0.229 | 0.502 | 0.479 | 0.1768 | -0.287 / +0.214 | ✗ | ✓ | ✓ | ✓ | ✓ | **✗** |
| bad_medical | -0.030 | 0.083 | 0.420 | 0.1862 | -0.238 / +0.513 | ✓ | ✓ | ✓ | ✓ | ✓ | **✓** |
| bad_nutrition | +0.232 | 0.659 | 0.397 | 0.1786 | -0.193 / +0.269 | ✗ | ✗ | ✓ | ✓ | ✓ | **✗** |
| bad_rehab | -0.191 | 0.066 | 0.326 | 0.1881 | -0.337 / +0.554 | ✓ | ✓ | ✓ | ✓ | ✓ | **✓** |
| reckless_driving | -0.143 | 0.441 | 0.476 | 0.1622 | -0.196 / +0.006 | ✓ | ✓ | ✓ | ✓ | ✓ | **✓** |
| bad_dental | +0.236 | 0.406 | 0.473 | 0.1795 | -0.135 / +0.347 | ✗ | ✓ | ✓ | ✓ | ✓ | **✗** |

Eligible (as written): unsafe_camping, bad_medical, bad_rehab, reckless_driving.
Draw: default_rng(20261021).permutation(4) = [2, 1, 0, 3] → **S3 = bad_rehab**; fallback order ['bad_medical', 'unsafe_camping', 'reckless_driving'].
<!-- END pf1/results/candidates.md -->

- **The draw follows B2.** Rehab was not a QS1 tag source, so §1.4's hindsight flag does not apply. In PB3, rehab was one of OTHER's five
  topics and medical's nearest neighbour; its Rev-S restraint (0.19) and Rev-F restraint (0.88) are therefore known to me.
- **Phrase:** "sports injury and rehabilitation" (19a's PHRASE table).

## Part B. Choices the handoff leaves open (fixed 2026-10-21 ~02:15, before any Phase 0 value is computed)

- **B1. The eligibility quantities (§1.4)** use PB3's sources, so e_ev and the 0.209 bar are on one footing.
  - e_ev = c_anc_base(K-every, candidate) from `pb3/results/selection.json` (= FB3's predictors).
  - a_ev uses ES1's E_late (`es1/results/E_base_23x23.npz`) and FB3's c_anc_base(K-every, ·), over the 20 scoring cells.
  - n_S = the mean E_late closeness of the candidate's four nearest topics by E_late, among the 22 others (finance's 0.209 was computed
    the same way).
- **B2. The draw.** The eligible list is taken in the handoff's §1.1 order. `rng = numpy.random.default_rng(20261021)`, then
  `perm = rng.permutation(len(eligible))`. S3 = eligible[perm[0]]; the fallback (§1.5) = eligible[perm[1]], then perm[2], and so on.
- **B3. The forecast models (§1.3).**
  - PT1's M2 is fitted on Rev-F's and Rev-S's PT1 cells (22 and 21; Rev-S's finance cell is left out as in PT1). The outcomes are PT1's
    `outcomes.json`. Each arm has an intercept; b and c are shared. s(·) is the within-arm standardised average rank over the arm's fit
    cells, with s(x_anc) = 0 where x_anc is constant.
  - For S3, ranks are taken over the 20 scoring cells (the 22-cell version over its own 22). The forecast is b·s(x_tr) + c·s(x_anc),
    centred.
  - ANCHOR-ONLY = c·s(x_anc), centred. The single-arm fits are the same model on one arm. M0 = rank(x_anc) − rank(x_tr), centred.
  - The COPY baselines are the observed r of Rev-F, Rev-S, or their mean, over the scoring cells, centred. COPY-NEAR uses A3's
    closeness of S3 to finance against chemicals.
- **B4. The measures (§1.3)** are PT1's definitions on the 547-question panel, computed from PT1's stored base states
  (`pt1/stats/states.pt`), half-set E_grad Gram (`pt1/results/job.json`) and MiniLM embeddings. The topic × topic values are asserted
  equal to PT1's `measures.json` for the finance and chemicals columns.
  - New GPU work is limited to what PT1 lacks:
    - L1 ratings of each candidate against every other topic (10 × 22 × 2 orders × 3 draws = 1,320; PT1's prompt, names and scoring);
    - S3's Q-GEN training questions;
    - for the secondary "actual training questions" forecast: S3's training questions' base states, E_grad (all topic half sets and
      K-every's recomputed in the same job, so the cross terms exist), and L1 ratings against S3's training questions.
- **B5. The flatness check (§1.4).** The fitted pattern of an arm is b·s(x_tr) + c·s(x_anc) over its fit cells. "SD" is the population
  SD of the centred values.
- **B6. Novelty (§1.4)** compares A3's forecast with Rev-F's and Rev-S's observed r over the 20 scoring cells.

## Part C. My probability column (written 2026-10-21 ~02:10, before any Phase 0 value or forecast is computed)

What I already know: I ran PB2, PB3 and PT1, so I have seen the Rev-F and Rev-S maps and PT1's transfer numbers (A3 ranked Rev-S at
Spearman +0.91 from a Rev-F fit, with the amplitude wrong). I know nothing about any candidate S3's map. QS1's tag maps for medical and
driving are known to me (§1.4's hindsight flag).

| # | prediction | lead | mine | why, briefly |
|---|---|---|---|---|
| F1 | A3's forecast reaches Spearman ≥ 0.60 | 0.55 | 0.50 | Rev-S's map followed in-model closeness (−0.82); Rev-F's less (−0.49). A new topic plausibly lies between. |
| F2 | at least one of A1–A6 reaches Spearman ≥ 0.70 | 0.50 | 0.45 | Six correlated tries, but 0.70 over 20 cells is high. |
| F3 | L1's forecast reaches Spearman ≥ 0.60 | 0.30 | 0.25 | L1 transferred poorly in PT1 (Spearman +0.36 on Rev-S). |
| F4 | A3 − ANCHOR-ONLY > 0, with its CI excluding 0 | 0.50 | 0.45 | With 20 cells, excluding 0 needs a gap of about 0.3 or more. |
| F5 | A3 − COPY-NEAR > 0 (point) | 0.60 | 0.55 | Rev-S showed a broad cluster-level shift; a copy of the nearer old map may do well. |
| F6 | A3 − L1 > 0 (point) | 0.65 | 0.65 | |
| F7 | the CI of A3 − T1 includes 0 | 0.60 | 0.65 | Wide intervals on 20 cells. |
| F8 | ρ_seed ≥ 0.90 | 0.70 | 0.70 | Refusal maps are stable (Rev-F against its near-replicate: Pearson 0.999; seed noise ≤ 0.03 in FB2). |
| F9 | A3's top-5 overlap ≥ 3 of 5 | 0.40 | 0.40 | |
| F10 | PATTERN PRESENT passes | 0.85 | 0.85 | Both earlier everyday maps had SD ≈ 0.2. |

## Part D. Notes after Parts A–C (timestamped; never edits of earlier text)

- **D1, 2026-10-21 ~03:27. Phase 0 done; the forecasts are frozen. `pf1/forecast.json` sha256 `6b1f3cfe05272b71a454b26f9471ab4c8014abf08bd0a0434e83920f0585920a`,** logged in RUNLOG
  (03:27) before any training job was submitted.
  - **GPU job `pf1-G` (591670, 214 s, 0.059 GPU-h).**
    - Q-GEN for rehab ("sports injury and rehabilitation"): 872 questions survived after 74 calls (dedupe −712, distinct −172,
      disjoint −15, judge −5). N = 708 distinct, drawn with seed `stable_int("pf1-draw", "bad_rehab")`, repeated to 800 rows;
      `pf1/data/train_S3.jsonl` sha256 `46424959d6eb62c7b01b15ed4b600992fca8a79ba92bb5a2f822b5fb23f83fc6`.
    - L1: 1,452 ratings (candidates against every other topic, and every topic against rehab's training questions); mass on 1–7
      0.9999.
    - PT1's 14B routine over the 23 topics, K-every and rehab's training set; E_grad validation PASS.
    - The recomputed topic states and gradients equal PT1's (max |Δ| 0 for A1–A6 and T1; 3e-14 for G1).
  - **Coverage of rehab's training questions** (`results/coverage.json`): by text, no other topic above 10%; by E_base, **bad_tattoo
    10.5%** (flagged; §4's coverage-free secondary drops it).
  - **COPY-NEAR = chemicals** (A3 puts rehab closer to chemicals than to finance).
  - **Three bugs, fixed before the freeze; no forecast existed while they were in place:**
    1. L1's fit lacked PT1's finance and chemicals ratings, and the first build stopped with a KeyError.
    2. Two builds were killed by my timeouts, because the CPU scripts had no `torch.set_num_threads(4)` and thrashed on the 4-CPU
       login node.
    3. The tiny training test of R0-S3 and Rev-S3-s2 passed (seed 21, the seed-21 A0, the step-10 split validation). PB3's tiny
       mode skips the order checks, so the seed-21 constructions were checked on CPU separately: the anchor order equals the seed-21
       construction, with the same rows and counts as main.
  - **The forecast's severity** (cross-score Spearman over the 20 cells):
    - F:A3 against F:A1 +0.92, F:A2 +0.92, F:T1 +0.38, F:L1 +0.45, ANCHOR-ONLY:A3 +0.65, COPY-S +0.55, COPY-F −0.34, COPY-MEAN −0.04.
    - Comparison 1 (A3 against ANCHOR-ONLY) is therefore hard to separate: closeness to rehab and to the everyday anchor run against
      each other over these topics.
- **D2, 2026-10-21 ~04:05. Training job `pf1-T1` (591880) failed after 1,353 s (0.376 GPU-h). The cause is outside PF1; the fix
  substitutes equivalent reference records.**
  - **What happened.** At 01:15, Exp 23's session deleted `/home/u4161854/pb2_ckpt` with Tony's approval (RUNLOG E23A 01:15). PF1's
    trainer (PB3's) read two PB2 records there: R0's SFT order and R-every's anchor visits. Rev-S3 stopped at start-up (no training);
    R0-S3 trained and saved its adapter but stopped at its order check, so it wrote no record. Rev-S3-s2 needs neither file and
    finished with every assertion passing: SFT order = P0-s2's, anchor order = the seed-21 construction with Rev-S3's counts, split
    validated at step 10, A0 sha256 `2db6a67b…`.
  - **No assertion failed;** the reference files were missing.
  - **The substitution.** PB3's records (`/home/u4161854/pb3_ckpt/R0-S` and `Rev-S`) hold the same sequences, and were asserted equal to
    PB2's when PB3 ran (`order_equal_pb2_R0` True; `anchor_visits_equal_twin` True). R0-S's order also equals 19a's first 1,200
    entries, which the trainer checks too. The trainer now compares against those records and asserts their PB2-equality flags. Nothing
    else changes.
  - **The rerun.** R0-S3 (retrained from scratch, deterministic) and Rev-S3 co-scheduled; then all three arms sampled and measured.
  - **Re-projection:** used 0.435 + about 0.5 ≈ 0.94 of 1.2. No cut is needed, and Rev-S3-s2 is already trained.
- **D3, 2026-10-21 ~04:55. Outcome** (`pf1/RESULTS.md`).
  - **Job T1b** (591925, 1,693 s, 0.470 GPU-h): R0-S3 and Rev-S3 trained, every assertion passed (references via PB3's records); all
    three arms sampled and measured. **PF1 total 0.905 of 1.2.**
  - **Gates:** every gate passes for both seeds. R0-S3 refuses 1.000 everywhere. PATTERN PRESENT passes (SD 0.226; ρ_seed 0.992
    [0.95, 1.00]); ranking ceiling ≈ 1.00.
  - **Scores over the 20 cells** (frozen forecasts, hash asserted):
    - A3 +0.69 [+0.27, +0.92] (top-5 4/5); G1 +0.82, A2 +0.81, A4 +0.77, A6 +0.75, A1 +0.72, A5 +0.71; L1 +0.61; T1 +0.37.
    - ANCHOR-ONLY (A3) +0.33; COPY-NEAR (= COPY-S) +0.36; COPY-MEAN +0.12; COPY-F +0.06.
  - **Comparisons:** 1 +0.36 [+0.07, +0.67]; 2 +0.32 [−0.11, +0.74]; 3 +0.08 [−0.37, +0.53]; 4 +0.32 [−0.15, +0.81].
  - **F1–F10 all TRUE.** Brier: lead 0.211, mine 0.234.
