# Handoff: GM1. Geometry and margins — cashing in HE1, UD1 and AN1 without a GPU

**For:** a coding agent on Tony's cluster, ideally the one that ran HE1 (its Part A code and stored vectors are the main input)
and AN1 (its closeness and sampling code).
**Root:** `/work/u4161854/loracle_em_exp1`, new directory `gm1/`.
**From:** the research lead (Claude), 21 Oct 2026, 23:30 Taipei.
**Review status:** **not yet red-teamed.** Parts A and D are CPU-only and may start immediately. **Get an independent review
before the GPU parts (B and C) are dispatched,** and record its probability column in §7.
**Budget:** about **0.08 GPU-h expected, cap 0.25.** Parts A and D need no GPU.
**Priority:**
- **Exp 23 and any other line's running work come first.** Dispatch a GM1 GPU job only when Exp 23 has no job pending in `squeue`.
- **Order:** Part D, then Part A (both CPU, no queue) → Part B → Part C.
- **Read-only:** `he1/`, `ud1/`, `an1/`, `pb2/`, `pb3/`, `pf1/`, `pt1/`, `fb1/`, `fb2/`, `fb3/`, `es1/`, `exp22/`, `exp20b/`,
  `exp19a/`, `/home/u4161854/e20b_ckpt/`, `/home/u4161854/e22_ckpt/`, `/home/u4161854/pb3_ckpt/`, `/home/u4161854/an1_ckpt/`,
  `/home/u4161854/pf1_ckpt/`.
- **Write only to `gm1/`.** Prefix every `RUNLOG.md` line with `GM1`.
- **No training.** Forward passes and analysis of existing artifacts only.

---

## 0. Why

Three experiments reported on 21 Oct. Each left its main question half-answered, and in each case the missing piece is
computable from artifacts already on the cluster.

- **HE1 and UD1 both lost their depth reading to the same calibration failure.** P0-s2, a re-seed of P0, keeps only
  0.71–0.75 of P0's change in the EARLY band (bar: 1 ± 0.15), so the pipeline cannot separate a penalty from a re-seed
  there. It keeps 0.91–0.93 in the MIDDLE band and 0.99–1.01 in the LATE band, where the same bar passes.
  **Part A re-reads depth on the bands where the calibration holds.**
- **The account HE1 arrived at rests on a quantity nobody has measured.** The reading is that the penalty shrinks the
  whole misaligned push to about two-thirds, leaving it below the base model's own preference for an aligned opening.
  The margins in the record are all *differences from the base* (ΔM), so the base's own absolute preference — the
  threshold itself — has never been computed. **Part B computes it and turns the account into a parameter-free test.**
- **Gate-versus-threshold is now the project's main open question**, and the cheapest decisive form of it is a question
  about rank. "No gate, only a dose" says the arms × topics matrix of pushes is **rank 1**: one scalar per arm times one
  topic profile. The two-term picture says **rank 2**, with the second component following closeness to each arm's own
  anchor — which AN1's Test B (Spearman 0.71, p 0.0008) says exists. Anything conditional shows up as residual structure
  that neither explains. **Part C runs that test in behaviour space, Part D in activation space.**

**What each part unblocks.**

| part | unblocks |
|---|---|
| A | the depth reading HE1 §5 rules 1 and 3 needed, and UD1 §9's two mechanism rows |
| B | the per-topic dose-response sweep (α 1.25–1.75), which needs a predicted crossing point to be worth running |
| C | gate-versus-threshold for the refusal, where the push is an exactly specified quantity |
| D | the prerequisite for two proposed experiments — a pre-stage that separates the trained topic, and gradient blocking. **If Part D gives RANK-1-GEOMETRY, both of those lose their premise and should not be designed.** |

**Why Part C is better than anything done on behaviour so far.** The share kept, f, saturates: many AN1 and PB3 cells sit
at 0.00 or 1.00, and a cell pinned at either end carries no information about how far the push moved. The log-odds of
entering R\* does not saturate, so every cell is informative. The same matrix in behaviour space would be censored at both
ends.

---

## 1. Inputs (all existing)

| input | what it is | where |
|---|---|---|
| HE1's stored activations | per-prompt fp16 vectors at the last template token, every block 0–39, about 12 models, about 680 prompts (~4.0 GB) | `he1/stats/` |
| HE1's Part A code and results | the chat-specific change, keep, q, the evaluable-block rule, band means | `he1/src/`, `he1/results/partA.json` |
| HE1's halves | the per-topic half-A / half-B split, seed 0 | `he1/halves.json` |
| HE1's groups | SO / LK / KP / FIN / EVD / GEN per arm | `he1/results/groups.json` |
| HE1's margins | ΔM_evil, ΔM_in, ΔM_X at α 0.5–3, with the onset/rest split | `he1/results/partB.json` |
| Exp 22's probe pairs | 25 everyday pairs; H_F (finance); H_X (sports) | `exp22/` |
| refusal arms | R0-F, Rev-F, Rev-Fp, Rev-S, R0-S (PB3); Rev-S3 (PF1); Rlt-F, Rpet-F (AN1) | `pb3_ckpt/`, `pf1_ckpt/`, `an1_ckpt/` |
| the gen24 panel | 23 topics, 547 questions | `fb3/data/gen24_panel.jsonl` |
| anchor closeness | c_anc_base per topic per anchor, PT1-A1 primary | `an1/PREREG.md`, `fb3/` Phase 0 |
| R\* | the fixed refusal sentence and its tokenisation | PB2/PB3 records |

**Pre-flight (CPU, before anything else).** Assert and log:
1. every adapter hash against its record (`fb2/results/hashes.json`; PB3's and AN1's `adapter_hashes.json`; FB1's record for P0);
2. `he1/stats/` contains the blocks and models HE1's `partA.json` reports, and nothing is truncated;
3. **the pipeline reproduces HE1's published band means within 0.02** for P0, P0-s2, K-every, K-adv1 and K-code-λ on both
   arms' SO and FIN. If it does not, stop and reconcile before computing anything new;
4. `an1/results/analysis.json`'s per-topic f values load, and AN1's PT1-A1 closeness values are present.

**If `he1/stats/` is missing or incomplete:** Parts A and D are blocked. Say so, and run B and C, which do not need it.
Re-generating the activations would cost about 0.03 GPU-h (HE1's job 593473 was 0.024) — ask Tony before doing it.

---

## 2. Part A. Depth, re-read where the calibration holds (CPU)

### 2.1 Bands and calibration

Keep HE1 §3's definitions unchanged: the chat-specific change (GEN-subtracted), P0's direction d̂_g(ℓ) from half A, keep and
q = keep ÷ keep on FIN, measured on half B, with 10,000-draw prompt bootstraps resampled jointly across models and the
half-A direction held fixed. Keep HE1's evaluable-block rule (P0's half-B projection ≥ 4 × its between-prompt SE, and
keep_M,FIN's CI lower end ≥ 0.2).

**New bands:** **MID** = the evaluable blocks among 19–29; **LATE** = the evaluable blocks among 30–38. Blocks 10–18 are
reported, never labelled. Block 39 is reported, never labelled.

**Calibration, checked before any label is read** (the same two checks HE1 used, moved to the new bands):
1. **K-code-λ** on each arm's SO: both band means of q ≥ 0.7;
2. **P0-s2:** both band means of keep within 0.15 of 1, on FIN and on each arm's SO.

HE1's published numbers say both should pass. **If either fails, Part A reports raw profiles only and assigns no label —
and that is itself a result worth stating plainly: the depth measurement does not work anywhere in this pipeline, and no
design should rely on it until it does.**

### 2.2 The statistic: per-block profiles, not band means

The old label set contrasted EARLY with LATE, so it cannot be used without the early band. What the surviving bands can
decide is the **shape** of keep across depth.

For each model M and group g, fit an OLS slope of keep_M,g(ℓ) on ℓ over the evaluable blocks in 19–38, and report:
- the slope, with a bootstrap CI over half-B prompts;
- the level (mean keep over those blocks);
- the **normalised profile** keep_M,g(ℓ) ÷ mean_ℓ keep_M,g(ℓ), so shape can be compared across groups at different levels;
- the **shape difference** between SO and FIN: the slope of the normalised SO profile minus the slope of the normalised FIN
  profile, with a bootstrap CI.

**Why this is the right question.** HE1's band means already hint at the answer: on K-every's SO keep goes 0.07 (MID) →
0.09 (LATE), essentially flat, while on FIN it goes 0.44 → 0.63, clearly rising. K-adv1 is the same (0.10 → 0.11 against
0.50 → 0.70). A late cancellation would make keep *fall* with depth. Neither group does that, which agrees with B1b's
negative — but band means over two points are not a profile, and the slope with a CI is.

### 2.3 Labels (per penalised arm, on its SO group; checked in order, first match wins)

1. **DECLINING** — the SO slope is negative with its CI excluding 0. The surviving form of "written, then removed", now
   inside the measurable band.
2. **RISING** — the SO slope is positive with its CI excluding 0, and the normalised SO and FIN profiles do not differ in
   slope (CI of the difference includes 0). The change is built up through depth on every topic, and the penalty reduced
   the build-up without changing its shape: proportional shrink in geometry.
3. **FLAT-SUPPRESSION** — the SO slope's CI includes 0 and the SO level is ≤ 0.2.
4. **SHAPE-DIFFERS** — the normalised SO and FIN slopes differ with the CI of the difference excluding 0, and neither 1
   nor 2 fired. Something about SO's depth profile is not FIN's profile rescaled, which is the conditional outcome.
5. **OTHER** — describe it.

Report MID and LATE band means beside every label so the result reads against HE1's published table.

### 2.4 References, nulls and noise

- **The refusal twins cannot serve as references here.** HE1 reports that Rev-F and Rpet-F have no evaluable MID or LATE
  block, because their keep on FIN stays below the 0.2 bar. **Say so explicitly in RESULTS:** the twin reference is lost
  exactly where the measurement works, so Part A has no harmless control. R0-F does have evaluable blocks (keep 0.5–0.7
  everywhere) and is reported as a reference, not a floor.
- **Replicates as the noise floor:** K-adv1 against K-adv1-r1, and K-every against K-std, on slope and level.
- **Null directions:** re-use HE1's 20 base principal components 2–100 and report the same ratio per band, so every keep
  value can be read against noise-over-noise (HE1's values were 0.007–0.072).
- **P0-s2's early divergence is a finding, not only a nuisance.** Report cos(c_P0, c_P0-s2) per block over 0–39 on FIN and
  on each arm's SO. HE1 reports 0.6–0.7 at blocks 9–14 against keep 0.99–1.01 late. If early-block change is largely
  seed-dependent while the late-block change is determined, that bears on where the functional content of the update sits,
  and it should be stated as its own paragraph in RESULTS.

### 2.5 UD1's release, re-read on the same bands

UD1's D6 reported Δkeep for the release at EARLY and LATE. Recompute its **MID** and **LATE** profiles and slopes with
§2.2's statistic, on UD1's own groups, for U-every and U-adv1 at steps 10, 20, 40 and 150, with C3 at 150 beside them.
Report whether the release's profile is the mirror image of the start model's (the shape a lifted suppression would have)
or flat (the shape added magnitude would have).

---

## 3. Part B. The base model's own preference (about 0.02 GPU-h)

### 3.1 What to compute

Every margin in the record is ΔM, a change from the base. Compute the base's **absolute** margin, per probe pair:

  M_base = mean log p_base(bad continuation) − mean log p_base(good continuation)

over Exp 22's 25 everyday probe pairs, H_F (finance) and H_X (sports), teacher-forced with Exp 22's own code and template.
Report it **total, onset (answer tokens 1–5) and rest**, as a per-pair distribution and a mean, exactly as the ΔM values
are split in Exp 22 decisions 09.

Then, for every model and α already in HE1's table, form the absolute margin:

  M_model,α = M_base + ΔM_model,α

at the onset and over the rest, per probe set. No new sampling: ΔM is already stored.

### 3.2 The test

The threshold account says the misaligned behaviour appears when the model's own preference crosses in favour of the
misaligned continuation, i.e. when **M_model,α at the onset rises above 0**. That is a classification with no free
parameters, over the (model, α, probe set) cells HE1 already measured against judged behaviour.

Build the table of cells with both an absolute onset margin and an observed behaviour (release, or f on the matching
group), and report:
- the sign of M at the onset against behaviour present or absent, with behaviour-present defined as release ≥ 0.5;
- the cells that break it;
- whether the breaking cells' residuals correlate with anchor identity or with closeness to the arm's anchor.

**Labels (checked in order):**
1. **THRESHOLD-FITS** — the sign of M_onset classifies ≥ 0.8 of cells, with no fitted parameter.
2. **THRESHOLD-FITS-WITH-OFFSET** — a single shared offset c, fitted once across all cells, classifies ≥ 0.8. Report c and
   the number of cells.
3. **THRESHOLD-FAILS** — no single threshold reaches 0.8. Report which cells break it and the residual structure.

**Read a success carefully.** A two-way teacher-forced margin is a proxy for a decision the model makes over its whole
next-token distribution; it ignores every continuation neither probe supplies. So **THRESHOLD-FAILS is the more
informative outcome**, and a fit should be reported as "consistent with", not as a mechanism established. State this in
RESULTS.

### 3.3 The one sharp prediction worth naming

On the sports probes, P0 at α = 1 has an onset ΔM of +5.93 and K-every at α = 1 has +4.00, while K-every at α = 2 returns
to +5.93. P0 is misaligned on those topics and K-every at α = 1 is not. So the threshold account requires M_base at the
sports onset to lie **between −5.93 and −4.00**. Report where it actually lies. That is a 1.93-nat window fixed by numbers
already in the record, and it is the cheapest falsification available anywhere in this project.

---

## 4. Part C. The arms × topics rank test, in behaviour space (about 0.05 GPU-h)

### 4.1 The push

For each refusal arm A, each panel prompt x, teacher-force R\* after the identical chat template and compute
log p_A(R\* | x) over R\*'s full token sequence. Do the same under the base model. Define

  s_A(x) = logit p_A(R\* | x) − logit p_base(R\* | x)

Use log-odds; at the base, p is tiny, so this reduces to a log-probability difference, and the log-odds form keeps arms
near f = 1 from saturating. **Secondary:** the same with the first distinct token only. PB3 used a first-distinct-token
rule because R\* begins "I'm" and many base openers begin "I"; report both and say which the labels used (full sequence is
primary).

**Pipeline check, required.** For a handful of (arm, topic) cells not pinned at 0 or 1, the predicted probability of
sampling R\* at T 1.0 should match the observed f within sampling error. Those cells are a check that the quantity is the
right one; the cells at 0.00 and 1.00 are the reason to compute it.

### 4.2 The matrix

M[A, T] = mean over T's half-B prompts of s_A(x).

**Primary set of arms: the finance-trained ones**, so that "one shared topic profile" is well defined — R0-F, Rev-F,
Rev-Fp, Rlt-F, Rpet-F (5 × 23). **Secondary:** add Rev-S (chemicals) and Rev-S3 (rehab) and report their spectrum
separately; a different trained topic should change the topic profile, and it is a check that the first component is the
trained-topic term.

### 4.3 The test

- SVD of the **raw** (uncentred) matrix, because the claim M[A,T] = s_A · u_T is multiplicative, not additive.
- Report the singular values, the share of squared Frobenius norm in components 1, 1+2, 1+2+3.
- **Noise floor:** split the prompts within each topic into halves, build the two matrices, and take the spectrum of their
  difference. That gives the residual RMS attributable to prompt sampling.
- **Residual structure:** correlate the rank-1 residual, and separately component 2's topic loadings, with c_anc_base(A, ·)
  — closeness of each topic to that arm's own anchor, PT1-A1 primary. This is AN1's Test B asked of the whole matrix at
  once instead of one anchor pair at a time.
- Report component 1's topic profile against closeness to finance (PT1-A1), which is what it should be if it is the
  trained-topic term.

**Labels (checked in order):**
1. **RANK-1** — component 1 holds ≥ 0.90 of the squared norm **and** the rank-1 residual RMS is within 1.5 × the split-half
   noise RMS. One dose per arm, one profile per topic; no anchor structure at all.
2. **RANK-2-ANCHOR** — two components are needed above the noise floor, component 2's topic loadings correlate with anchor
   closeness at |ρ| ≥ 0.5, and its arm loadings order by anchor. The two-term model, confirmed where behaviour cannot
   saturate.
3. **RANK-2-OTHER** — two components needed, component 2 not explained by anchor closeness. Describe its topic profile.
4. **RANK-3-OR-MORE** — three or more components above the floor. Report what the third is.
5. **MIXED** — anything else.

---

## 5. Part D. The topics × directions rank test, in activation space (CPU)

### 5.1 The matrix

From HE1's stored vectors, using HE1's chat-specific change (GEN-subtracted) at the last template token:

  v_T(ℓ) = mean over topic T's half-B prompts of c_P0(x) at block ℓ

Build the 23 × d matrix at each of blocks 22, 26 and 30, and once more on the concatenation of the evaluable LATE blocks.
SVD each. Exclude no topic; report chemicals separately, since its P0 shift sits below FB3's 25-point floor.

### 5.2 The floors, which decide everything here

Report the same spectrum for:
- **P0-s2** — the same training, a different seed. **This is the floor**: components of P0's spectrum that P0-s2 reproduces
  at the same strength are properties of the training, and components that differ between seeds are not a route.
- **a within-topic prompt split-half** of P0 — prompt sampling noise.
- **R0-F**, the unpenalised refusal — a different behaviour, same model. If its rank matches and its component-2 topic
  profile correlates with P0's, the geometry is a property of the model rather than of misalignment, which continues the
  refusal-twin result and would be worth reporting on its own.

### 5.3 The second half of the question: what does the penalty spare?

Project each penalised arm's chat-specific change onto P0's components and report the share of each component retained,
per arm: K-every, K-std, K-adv1, K-adv1-r1, and **K-code-λ as the no-switch control, which should retain everything**.

**Labels (checked in order):**
1. **RANK-1-GEOMETRY** — component 1 holds ≥ 0.90 of the squared norm and no further component exceeds the P0-s2 floor.
   There is no topic-specific route. **Report prominently: the proposed separation pre-stage and the gradient-blocking
   experiment both lose their premise, and should not be designed.**
2. **TOPIC-ROUTE-EXISTS** — at least two components exceed the P0-s2 floor, and finance's loading on component 2 differs
   from the panel mean by ≥ 2 × the floor's spread. Then, as a sub-label:
   - **PENALTY-SPARES-THE-ROUTE** — the penalised arms retain ≥ 0.6 of component 2 while retaining ≤ 0.3 of component 1.
   - **PENALTY-SHRINKS-BOTH** — the retained shares of components 1 and 2 are within 0.15 of each other. Proportional
     shrink, in geometry: the strongest form of the no-gate account.
   - **OTHER** — describe it.
3. **MIXED**.

### 5.4 The cross-check between Parts C and D (required)

Correlate component 2's topic profile from Part C (behaviour) with component 2's topic profile from Part D (geometry),
over the 23 topics, with a topic-bootstrap CI. Agreement means the two-term structure is one thing seen in two spaces and
is the headline of GM1. Disagreement is also informative, and must not be smoothed over.

---

## 6. What GM1 cannot do

State all of these in RESULTS, beside the labels.

- **Part A still cannot separate "never written" from "written early and gone by block 19."** The early band is where that
  question lives, and the calibration says the pipeline cannot answer it. Part A tests the shape of what survives, not the
  history of what was written.
- **Part A has no harmless control**, because the refusal twins have no evaluable block in the surviving bands.
- **Part B's margin is a two-way proxy** for a decision over the whole distribution. A fit is "consistent with"; a failure
  is a refutation.
- **Part C is one behaviour** — a fixed harmless refusal — and refusal levels do not transfer to misalignment. Its matrix
  has 5 arms, so it can resolve rank 1 against 2 and perhaps 3, not more.
- **Part D's matrix is built from P0 alone**, one seed, one trained topic. P0-s2 bounds seed noise; nothing bounds
  trained-topic noise, because there is only one EM trained topic in the project.
- **Every CI treats the topics and the single training run per arm as fixed**, as throughout this project.

---

## 7. Predictions (fill in your own column before computing; the reviewer's column before the GPU parts)

| # | prediction | lead P | agent P | reviewer P |
|---|---|---|---|---|
| G1 | Part A: the calibration passes on both MID and LATE, for both arms | 0.90 | | |
| G2 | Part A: K-every's SO label is FLAT-SUPPRESSION | 0.65 | | |
| G3 | Part A: FIN's keep slope over 19–38 is positive with the CI excluding 0, for both arms | 0.85 | | |
| G4 | Part A: SHAPE-DIFFERS fires for at least one arm (normalised SO and FIN slopes differ) | 0.45 | | |
| G5 | Part A: no arm's SO label is DECLINING | 0.80 | | |
| G6 | Part B: M_base at the onset is negative on all three probe sets | 0.85 | | |
| G7 | Part B: M_base at the sports onset lies between −5.93 and −4.00 (§3.3) | 0.35 | | |
| G8 | Part B: THRESHOLD-FITS or THRESHOLD-FITS-WITH-OFFSET | 0.55 | | |
| G9 | Part C: component 1 holds ≥ 0.90 of the squared norm | 0.60 | | |
| G10 | Part C: the label is RANK-2-ANCHOR | 0.45 | | |
| G11 | Part C: the label is RANK-1 | 0.20 | | |
| G12 | Part C: component 1's topic profile correlates with closeness to finance at ρ ≥ 0.5 | 0.70 | | |
| G13 | Part D: the label is TOPIC-ROUTE-EXISTS | 0.45 | | |
| G14 | Part D: given TOPIC-ROUTE-EXISTS, the sub-label is PENALTY-SHRINKS-BOTH rather than PENALTY-SPARES-THE-ROUTE | 0.55 | | |
| G15 | Part D: R0-F's spectrum has the same rank as P0's and their component-2 topic profiles correlate at ρ ≥ 0.5 | 0.50 | | |
| G16 | §5.4: Part C's and Part D's component-2 topic profiles correlate at ρ ≥ 0.5 | 0.40 | | |

**Scoring.** G1, G6 and G9 are always scored. Any prediction whose part was blocked, or whose label needed a calibration
that failed, is scored NOT EVALUABLE and left out of the Brier score. G14 is scored only if G13 is true.

---

## 8. Stop rules, disk, compliance

**Stop rules:**
- an adapter hash mismatches a record that exists;
- the §1 pipeline check fails to reproduce HE1's band means within 0.02 (stop and reconcile; do not proceed with new numbers);
- Part C's pipeline check fails — the predicted probability of sampling R\* does not match observed f on unpinned cells
  within sampling error (the quantity is then not what the labels assume);
- a §5.2 floor cannot be computed, because then Part D has no labels, only a spectrum;
- you are about to run a GPU job before `gm1/PREREG.md` and the independent review are logged.

**Disk.** GM1 writes statistics and JSON only, a few hundred MB. It reads HE1's 4 GB of vectors in place and copies
nothing. Check `df` before each job. Delete nothing outside `gm1/`.

**Compliance:**
- **Nothing is trained.** No adapter is created, modified or merged. No scaled or zeroed copies are made.
- **No new text is generated.** Parts B and C are teacher-forced forward passes over continuations and a fixed refusal
  sentence that already exist.
- Exp 22's probe pairs contain misaligned continuations written in earlier experiments. They stay where they are, on the
  cluster, and are never copied into `gm1/` in plain text; `gm1/` holds log-probabilities only.
- The 35 verbatim Turner/Soligo release questions in everyday48 and Exp 22's class 2 are used, as before, **only as
  evaluation prompts.** Nothing is trained on the release or anything derived from it.
- `v_EM` is not needed and is not used.
- **`RESULTS.md` quotes no prompt and no model output,** and reports rates, log-probabilities, spectra and neutral
  summaries only. Nothing harmful is pasted into chat.

**Reporting** follows `analysis_conventions.md` (not on the cluster; follow the five items it lists, as HE1 and UD1 did):
raw numbers beside every label, per arm, group and block, with spreads; an exploratory section; labels exactly as
pre-registered; and a self-doubt section that names §6's limits and every cell that could not have fired either way.

---

## 9. Final message to Tony (at most 8 lines)

1. The pre-flight checks, and whether HE1's stored vectors were complete.
2. Part A: the calibration on MID and LATE; the SO and FIN slopes and levels per arm, with labels; whether any arm is DECLINING.
3. Part A: the per-block seed cosine between P0 and P0-s2, in one line.
4. Part B: M_base at the onset on all three probe sets, where it falls relative to the −5.93 / −4.00 window, and the label.
5. Part C: the spectrum, the label, and what component 2 tracks.
6. Part D: the spectrum against the P0-s2 floor, the label and sub-label, and — stated plainly — whether the separation
   pre-stage and gradient-blocking experiments still have a premise.
7. §5.4's cross-check.
8. G1–G16 verdicts, GPU-h used, and the path to `gm1/RESULTS.md`.

---

## 10. Order of work

1. Pre-flight (CPU). Write `gm1/PREREG.md`: this handoff verbatim, the checks, your probability column.
2. **Part D** (CPU) — it gates two proposed experiments, so it goes first.
3. **Part A** (CPU), including §2.5's re-read of UD1's release.
4. Independent review of Parts B and C. Record its column.
5. **Part B**, then **Part C** (GPU, one job if they fit together).
6. §5.4's cross-check, then `RESULTS.md`. No `EXAMPLES.md` is required: GM1 reads no model output.
