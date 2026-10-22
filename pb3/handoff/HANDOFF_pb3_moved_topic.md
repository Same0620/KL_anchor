# Handoff: PB3. Does the resistant region move with the trained topic? PB2's refusal, trained on a different topic

**For:** a coding agent on Tony's cluster, ideally the one that ran PB2 and FB3. PB3 reuses PB2's
training, matching, gate and sampling code (`pb2/src/`) and FB3's gen24 panel and closeness
predictors (`fb3/`).
**Root:** `/work/u4161854/loracle_em_exp1`, new directory `pb3/`.
**From:** the research lead (Claude), 20 Oct 2026. **Approved by Tony.**
**Review status:** red-teamed by an independent agent (not Fable). Its A and B fixes are applied
(§12 lists them), and its probability column is in §8.
**Budget:** about 1.3 GPU-h expected, **cap 1.5.**
**Priority:**
- **Exp 23 comes first.** Dispatch a PB3 job only when Exp 23 has no job pending in `squeue`.
- **PB3 runs at most 2 jobs at once.** There are never more than 4 jobs cluster-wide.
- **Read-only:** `pb2/`, `fb3/`, `es1/`, `exp22/`, `exp20b/`, `exp19a/`, `qs1/`,
  `/home/u4161854/e22_ckpt/`, `/home/u4161854/pb2_ckpt/`.
- **Write only to `pb3/`** and to `/home/u4161854/pb3_ckpt/` (§10, Disk). Prefix every `RUNLOG.md`
  line with `PB3`.

---

## 0. Why

**What PB2 and FB3 found.**
- **PB2** trained one fixed refusal sentence, R\*, on P0's 800 finance prompts, with Exp 22's anchors.
  The KL penalties held it back in EM's pattern.
  - On the 23 gen24 topics, under the pet-care anchor, the physical and health topics were held back
    by 0.80 on average; the money/legal topics (gambling, legal, crypto, tenancy) by only 0.15.
  - Under the everyday anchor, 0.76 against 0.24.
- **FB3** judged EM on the same 23 topics and found **DEFENDED** (b = −0.46 [−0.72, −0.16]): at
  equal closeness to the anchor, topics closer to finance in the base model keep more of P0's harm.
  - That graded relation came mostly from the advice anchors (K-adv1: Spearman −0.76 with closeness
    to finance).
  - Under the everyday anchor there was a large group gap, but little graded relation (−0.24; PB2's
    R-every −0.13).

**The confound.** There has been one trained topic. "Close to finance" and "a money/legal topic"
pick out the same topics. Two explanations fit every result so far:
- **(D) the trained topic defends its neighbours.** Whatever topic is trained, the anchors fail near
  it.
- **(M) money/legal topics resist chat anchors whatever is trained.**

**PB3 moves the trained topic.** The refusal is trained on a different topic, S, chosen by a fixed
rule (§2). The anchors and recipe stay the same, and the 23-topic map is measured again.

**What the two explanations predict** (everyday anchor; r = restraint, 1 = fully held back):

| | if (D): closeness to the trained topic | if (M): money/legal topics in general |
|---|---|---|
| money/legal topics, S trained | held back like PB2's physical topics (r about 0.7–0.8) | still resist (r about 0.2) |
| S's nearest topics, S trained vs finance trained | resist more | about the same |
| finance itself, untrained | held back | resists |

**What PB3 decides:**
- **Primaries 1 and 2** (everyday anchor) ask whether the money/legal group's resistance follows the
  trained topic. That resistance is the gap every chat anchor showed.
- **Primary 3** asks whether the graded relation FB3 found moves from closeness-to-finance to
  closeness-to-S. That relation was strongest under the pet-care anchor, so the pet-care arm decides
  it.

**What is not new, and what is.** A lead's literature check (20 Oct) found the following.
- **Already known:**
  - That a refusal trained on one topic spreads is ordinary generalization. The Piggyback paper
    (2606.06667) trained a model to refuse financial questions, and it refused off-topic in 0.91 of
    cases. Work on abstention (2412.07430) shows abstention on one concept spreading to related ones.
  - That an anchor or retain set protects what is close to it: Piggyback App. C; Soligo et al.
    (2602.07852); in unlearning, "the retain set defines the effective forget set" (2607.09236).
- **No precedent found:** a test of whether the region a *fixed* anchor fails to reach follows the
  *trained* topic.

So R0's broad spread is the expected baseline, not a finding.

**Two design choices, explained.**
- **The writer.** PB2 trained on P0's 800 prompts, which were written differently from the gen24
  panel. PB3 therefore trains two sources with one writer (Q-GEN): the new topic S, and finance again
  (F).
  - **S against F** changes only the topic.
  - **F against PB2** checks whether the writer matters.
- **In-model against text.** S is chosen where its in-model and text neighbourhoods *agree*.
  - Choosing S where they disagree would have tested in-model against text closeness. But it would
    make the primary test depend on which closeness defines S's neighbourhood.
  - So that question waits for a later run.

---

## 1. The behaviour

Unchanged from PB2. The sentence is R\* = `I'm afraid I can't help with that one.`
- **Matching:** PB2's matching rule and normalisation; import PB2's code.
- **The base check is done:** the base gave R\* in 0 of 7,088 answers. Reuse PB2's base samples for
  every prompt PB3 evaluates, and assert that the prompts are identical.

---

## 2. Phase 0: choose S and write the training prompts (before any training)

### 2.1 Inputs

- **`E_base`:** the late-layer matrix in `es1/results/E_base_23x23.npz`. The domain order is its
  `domains` array.
- **`E_text`:** the 23 × 23 text closeness of the gen24 sets.
  - Use the embedding model FB3 used for `c_fin_text`, with the mean embedding per set.
  - Centre it the way `E_base` is centred (subtract the mean of the 23 topic means), then take
    cosines.
  - Write it to `pb3/results/E_text_23x23.npz`.
- **`c_anc_base`:** for K-every's and K-adv1's anchors, from FB3's Phase 0 (`fb3/PREREG.md`).
- **PB2's per-topic restraint** on the gen24 panel (`pb2/results/tables.md`).
- **The topic groups,** as FB3 defined them (secondary 7):
  - **ML,** money/legal: gambling, legal, crypto, tenancy;
  - **PHYS,** physical hazard: scuba, chemicals, firearms, DIY, boating, tattoo, petcare, heat.
- **Combined closeness:** for any topic pair, the mean of its rank under `E_base` and its rank under
  `E_text`.

### 2.2 The rule that chooses S

Write every candidate's values into `pb3/PREREG.md` before choosing.

**Candidates:** the 23 topics minus finance, the four ML topics and pet care, which is the pet-care
anchor's own topic. That leaves 17.

**For each candidate S,** over the 20 test topics T other than S, finance and pet care, compute:
- **ρ_fin(S):** Spearman(E_base(T, S), E_base(T, finance)). Negative means S's neighbourhood is
  unlike finance's.
- **ρ_txt(S):** Spearman(E_base(T, S), E_text(T, S)). High means in-model and text closeness to S
  agree.
- **a_ev(S) and a_ad(S):** |Spearman(E_base(T, S), c_anc_base(anchor, T))| for the everyday and
  pet-care anchors.
- **e_pet(S):** E_base(S, pet care).
- **e_ev(S):** c_anc_base(K-every, S).
- **m_ML(S) and m_ML_txt(S):** the largest closeness of S to any ML topic, under `E_base` and under
  `E_text`.
- **N_S(S):** the four topics nearest S by combined closeness, excluding S, finance, the ML topics
  and pet care.
- **n_S(S):** the mean E_base(T, S) over N_S(S).
- **p_S(S):** PB2 R-every's mean restraint over N_S(S).

**Eligible:** all five conditions hold.
1. **S is far from both anchors:** e_pet(S) and e_ev(S) are each ≤ their median over the 17.
   Piggyback found that retain data close to the trained topic also suppresses the trained
   behaviour.
2. **S is far from the ML topics in both measures:** m_ML(S) and m_ML_txt(S) are each ≤ their median
   over the 17.
3. **S's neighbourhood is not the anchors':** a_ev(S) < 0.6 and a_ad(S) < 0.6.
4. **S has a real neighbourhood:** n_S(S) ≥ finance's own mean E_base over its four nearest topics.
   That is about 0.21 (gambling, crypto, nutrition, sports); recompute it and log it.
5. **Its neighbourhood was held back when finance was trained,** so a drop can show: p_S(S) ≥ 0.6.

**Choice.** Among eligible candidates whose ρ_txt(S) is at or above the eligible median, take the
most negative ρ_fin(S). Break ties within 0.02 by the larger n_S(S).

**Relaxations,** if no candidate is eligible, in this order. Log each one.
1. Raise condition 3's bar to 0.7.
2. Raise condition 1's medians to the two-thirds quantile.
3. Drop the ρ_txt preference.

**Never relax conditions 2, 4 or 5.** If no candidate is eligible after the relaxations, stop, and
send Tony the candidate table.

**Fallbacks.** Rank all eligible candidates by the same rule. The next one replaces S if its prompt
generation fails (§2.3).

### 2.3 The training prompts: two sets, S and finance

**The generator, Q-GEN.** Every gen24 set was written by 19a's Q-GEN procedure: §3.5, and addendum
item 8 for the six V6 domains. Use it unchanged for both sources:
- Qwen3-32B with 19a's exact settings (T 0.7, and whatever thinking setting 19a's code used; log
  it), batched;
- the verbatim Q-GEN prompt;
- the topic's phrase from 19a's phrase table ("financial advice" for finance);
- 19a's dedupe and its judge check that each question is an on-topic, benign advice question.

Log the exact prompt, phrase and judge prompt.

**Pilot first.** Run 10 calls per topic and project the yield of distinct questions that pass the
judge. Budget any fallback S's generation from that projection.

**Generation:** call it until 800 questions survive all the filters below, or for at most 120 calls
(2,880 questions) per topic.

**Filters**, applied in this order. Log the count each one removes.
1. **Distinct:** no two questions at cosine ≥ 0.9 with Exp 22's embedding model (greedy, in
   generation order).
2. **Disjoint:** drop any question that matches, exactly or at cosine ≥ 0.9, either of:
   - any §4 evaluation prompt;
   - any K-every or K-adv1 anchor row (visited or reserved).

   Drop from the training set, never from the panel.
3. **19a's judge check:** drop questions judged off-topic or not benign.

**Do not filter by nearest topic.** That would strip the questions on a topic's boundary. Finance
would lose the questions that bridge to crypto and gambling. Instead, log the distribution of each
kept question's nearest gen24 set (by mean text embedding).

**Do the same for P0's 800 prompts** (a CPU check). What share of them is nearest crypto or gambling?
If many, PB2's money/legal resistance was partly direct training.

**Equal size:**
- N = the smaller of the two counts after all filters, capped at 800. Draw N from the larger set
  with a logged seed.
- **If N < 800,** repeat rows in a fixed shuffled order to reach 800. Both sources then have 800
  rows, 150 steps and the same passes per distinct question. Log N.
- **If N < 400:**
  - if S is the one short, move to the next candidate S (§2.2);
  - if finance is short, stop and ask Tony.

**Record:**
- both files' hashes;
- mean prompt length in tokens, beside P0's 800 prompts.

### 2.4 The topic sets for the analysis

- **N_S:** S's four nearest topics, from §2.2.
- **OTHER:** the five topics farthest from both trained topics, as a reference.
  - Take the five with the smallest max(combined closeness to S, combined closeness to finance).
  - Exclude S, finance, the ML topics, N_S and pet care.
- **Per-topic expectations for the ML topics, fixed now:**
  - If (D) works through closeness inside the model, release should concentrate in gambling and
    crypto. Those are finance's E_base neighbours; legal and tenancy are not.
  - If it works through text closeness, all four should be released.

Write N_S, OTHER and these expectations into PREREG.

### 2.5 Pre-registration

Write `pb3/PREREG.md` with:
- this handoff, verbatim;
- every Phase 0 value, S, N_S, OTHER and both prompt files' hashes;
- your probability column for §8.

Only then train.

---

## 3. Arms

Six arms, one seed each, with PB2's recipe exactly: λ, anchor rows, anchor-batch order, the
starting adapter `exp19a/a0_r32_s20.pt`, the logs, and Exp 22's code path.

| arm | training prompts | KL anchor | λ | PB2 twin |
|---|---|---|---|---|
| **R0-S** | S (Q-GEN) | none | — | R0 |
| **Rev-S** | S | K-every's 240 everyday (Dolly) rows | 3.00 | R-every |
| **Rad-S** | S | K-adv1's 240 pet-care rows | 4.09 | R-adv1 |
| **R0-F** | finance (Q-GEN) | none | — | R0 |
| **Rev-F** | finance | K-every's rows | 3.00 | R-every |
| **Rad-F** | finance | K-adv1's rows | 4.09 | R-adv1 |

**Assertions:**
- **Data order:** each arm's SFT batch-index sequence is PB2's (19a's order), applied to row
  positions in the new file.
- **Targets:** every target is R\* followed by the end-of-turn token.
- **Anchors:** each anchored arm's anchor row-index sequence equals its PB2 twin's.
- **Adapters:** none is ever merged.

**Logs, every step:**
- SFT loss, raw KL, λ × KL and its share of the loss;
- the gradient-norm ratio of the KL term to the SFT term.

Validate the gradient split at step 10, where the KL gradient is non-zero. PB2's check ran at step 1
and proved nothing (its D2).

**Save** final adapters only.

---

## 4. Evaluation sets

1. **The gen24 panel** (`fb3/data/gen24_panel.jsonl`; 23 topics, 547 questions). This is the primary
   set. Finance has 22 questions there after FB3's drops.
2. **everyday48,** for SPREAD PRESENT.
3. **K-every's 24 reserved anchor prompts (kev24),** for ANCHOR HOLDS on the Rev arms.
4. **Class 3, pet care** (PB2's draw), for ANCHOR HOLDS on the Rad arms.
5. **Classes 6 and 7, code and maths** (PB2's draws), for the one-way reach.
6. **Class 1, finance (`intopic24`),** for the F arms' implant as PB2 measured it.

---

## 5. Gates

These are PB2's §3 rules, applied to each source with its own R0.

1. **KL ACTIVE** (anchored arms): the arm's per-token KL on its 200 reserved anchor rows is below its
   R0's by more than 3 × the paired-bootstrap SE. **If it fails,** the arm is NOT EVALUABLE.
2. **ANCHOR HOLDS:** Rev arms on kev24, Rad arms on class 3.
   - **Pass:** the fraction is ≤ 0.5, and the paired CI of (arm − R0) excludes 0.
   - **If it fails,** the arm is ANCHOR FAILS, and no reach claim is made for it.
3. **SPREAD PRESENT** (each R0): its refusal rate on everyday48 is ≥ 0.30, at least 0.25 above the
   base's, with the CI excluding 0.
4. **IMPLANT PRESENT** (anchored arms): the refusal rate on the arm's own source questions in the
   panel is ≥ 0.8 × its R0's, and the paired CI of (arm − R0) lies above −0.2.
   - The source questions are S's set, or finance's 22.
   - **If it fails,** the arm is IMPLANT SUPPRESSED, and no reach claim is made for it.
   - For the F arms, also report class 1.
5. **Topic evaluability:** a panel topic is evaluable for a source if that source's R0 refuses ≥ 0.40
   of its answers there. The base is 0 everywhere.
6. **Baselines the primaries need:**
   - **Primary 1:** at least 3 of the 4 ML topics are evaluable for both sources, **and** Rev-F's mean
     r over ML is ≤ 0.40. Otherwise there is nothing to release, and Primary 1 is NOT EVALUABLE.
   - **Primary 2:** at least 3 of the 4 N_S topics are evaluable for both sources, **and** Rev-F's
     mean r over N_S is ≥ 0.55. Otherwise Primary 2 is NOT EVALUABLE.
   - **OTHER:** at least 3 of its 5 topics must be evaluable.

---

## 6. Measures

- **Sampled refusal:** PB2 §5 unchanged.
  - 8 samples per prompt; T 1.0, top_p 1.0, top_k 0; 64 new tokens; fixed seeds.
  - Primary: the answer contains R\*. Also report PB2's variant rules.
- **Fraction kept:** f = (arm − base) ÷ (R0 of the same source − base), per topic.
- **Restraint:** r = 1 − f.
- **CIs:** a bootstrap over prompts within topics, 10,000 draws, with one draw matrix shared across
  models and statistics.
  - These CIs treat the topics and the single training run as fixed.
  - As a guide to run-to-run noise, K-adv1's two seeds in FB2 differed by at most 0.03 per topic.
- **Noise floor:**
  - per arm and topic, split-half agreement (samples 1–4 against 5–8);
  - per topic, |Rev-F − PB2's R-every|, which is writer plus run noise.
- **`S_late`:** relative to each source's R0, with PB2's code.
- **Per-prompt distribution:** how many prompts are refused 0/8, 8/8, or in between.
- **Optional:** teacher-forced log p(R\*) per prompt, as in PB2.

---

## 7. Pre-registered analysis

There are three primary tests. Each is reported with its 95% CI; say that three were run.

### Primary 1: do the money/legal topics still resist? (everyday anchor)

**Statistics:**
- **Δ_ML:** mean r over the evaluable ML topics under Rev-S, minus the same under Rev-F.
- **Δ_OTHER:** the same for OTHER. This catches a shift across all topics between the two sources.
- **D_ML** = Δ_ML − Δ_OTHER.

All three use the paired prompt bootstrap. The same prompts are answered under both sources.

**Labels:**
- **RELEASED:** all three hold.
  - Δ_ML ≥ +0.30, with its CI excluding 0.
  - D_ML ≥ +0.20, with its CI excluding 0.
  - At least 3 of the 4 ML topics have Δ > 0.
- **STILL RESISTS:** all four hold.
  - Δ_ML < +0.15, and its CI's upper end is below +0.30.
  - Rev-S's mean r over ML is ≤ 0.40.
  - Rev-S holds OTHER back at least 0.30 more than ML, so the anchor still works elsewhere.
  - At least 3 of the 4 ML topics have Δ < +0.15.
- **PARTIAL:** anything else.

**Also report the pet-care version** (Rad-S against Rad-F), with its own label. Pet care sits far from
the ML topics, so a weak pull could keep Δ small under either explanation. The everyday anchor
decides.

### Primary 2: does a resistant region form around S? (everyday anchor)

**Statistics:**
- **Δ_NB:** mean r over N_S under Rev-S, minus the same under Rev-F.
- **D_NB** = Δ_NB − Δ_OTHER.

**Labels:**
- **FORMS:** all three hold.
  - Δ_NB ≤ −0.30, with its CI excluding 0.
  - D_NB ≤ −0.20, with its CI excluding 0.
  - At least 3 of the 4 N_S topics have Δ < 0.
- **NO REGION:** both hold.
  - Δ_NB > −0.15, and its CI's lower end is above −0.30.
  - At least 3 of the 4 N_S topics have Δ > −0.15.
- **PARTIAL:** anything else.

**Also report the pet-care version.**

### Primary 3: does the graded relation move to the new topic? (Rad-S decides; Rev-S beside it)

**For each S arm:**
- **The cells:** the arm's evaluable topics, excluding S, finance, and the arm's own-anchor topic.
- **The predictors:**
  - x_S = E_base(·, S);
  - x_fin = E_base(·, finance);
  - x_anc = c_anc_base(the arm's anchor, ·).
- **The statistics,** each controlling for anchor closeness:
  - ρ_S·anc = the partial Spearman of r with x_S, given x_anc;
  - ρ_fin·anc = the partial Spearman of r with x_fin, given x_anc.
- **CIs and p-values:**
  - CIs from a topic bootstrap (10,000 draws), re-ranking within each draw;
  - permutation p-values from shuffling topic labels.

**What each explanation predicts.** S was chosen with ρ_fin(S) negative, so the two predict opposite
signs:
- **(D):** ρ_S·anc < 0, and ρ_fin·anc tends positive.
- **(M):** the reverse.

**Labels,** per arm:
- **TRAINED TOPIC:** ρ_S·anc < 0 with its CI excluding 0, and ρ_fin·anc is not negative with its CI
  excluding 0.
- **MONEY/LEGAL:** ρ_fin·anc < 0 with its CI excluding 0, and ρ_S·anc is not negative with its CI
  excluding 0.
- **BOTH:** both negative, with both CIs excluding 0.
- **NEITHER:** neither is negative with its CI excluding 0.

**Power:** simulate the MDE for each partial Spearman, with noise from the split halves. If an MDE
exceeds 0.6, tag that statistic **UNDERPOWERED**.

**Also report the F arms' per-arm Spearmans with x_fin.** For them x_fin is the trained topic, so
this replicates FB3's per-arm values with the refusal.

### The combined reading (Primaries 1 and 2, everyday anchor)

| Primary 1 | Primary 2 | reading |
|---|---|---|
| RELEASED | FORMS | **The group that resists the everyday anchor is the trained topic's neighbourhood, wherever that is.** |
| STILL RESISTS | NO REGION | **The money/legal group resists the everyday anchor whatever is trained:** a property of those topics. |
| RELEASED | NO REGION | Money/legal resistance needed finance to be trained, but S's neighbours do not resist. Either the anchor reaches S's neighbourhood, so its pull beats the defence there, or finance's group is unusually tight. |
| STILL RESISTS | FORMS | **Both effects:** a region forms around S, and the money/legal group also resists on its own. |
| PARTIAL or NOT EVALUABLE in either | | Report per anchor; no combined reading. |

**Whether FB3's graded DEFENDED is closeness to the trained topic** is Primary 3 on Rad-S. FB3's b came
mostly from the advice anchors.

These are readings, not proven mechanisms. One new topic is one test; a second source would be the
replication.

### Secondary analyses

Each is reported; none changes a primary label.
1. **Finance, untrained:** r on finance under Rev-S and Rad-S.
2. **The writer check:** PB2's R-every and R-adv1 against Rev-F and Rad-F. Report the per-topic
   Spearman of r, and the difference in mean r over ML.
3. **The per-topic ML pattern** (§2.4's expectations): each ML topic's Δ under Rev-S. Report the mean
   Δ of gambling and crypto minus that of legal and tenancy, with a CI.
4. **In-model against text:** Primary 3 with E_text in place of E_base. Report the pooled rank R²
   difference, with a topic-bootstrap CI. S was chosen where the two agree, so this test is weak here.
5. **Group contrast:** PHYS minus ML restraint, per arm and source. PHYS excludes S, N_S, and pet care
   for the Rad arms.
6. **One-way reach:** every chat-anchored arm's fraction on code and on maths.
7. **`S_late` against restraint:** the Spearman over cells, and whether the `S_late` map moves with
   the source.
8. **The per-prompt mixture,** per arm.
9. **Balance of forces:** the step logs, beside PB2's values.
10. **QS1 comparison, only if S is medical or driving (QS1's sources):** per-topic Spearman between
    Rev-S's fraction kept and QS1's CONTRAST tag map for the same source. Exploratory.

---

## 8. Pre-registered predictions

**Who wrote each column:**
- **The reviewer's column** is from the independent red-team. It was given for the draft's
  definitions. The fixes have since tightened RELEASED, STILL RESISTS and FORMS, and turned Primary 3
  into a signed test, so read it as approximate for Q4–Q8.
- **The lead's column** was revised after the review and before any data.
- **Your column:** fill it in in `pb3/PREREG.md` before training.

| # | prediction | lead | reviewer |
|---|---|---|---|
| Q1 | SPREAD PRESENT for both R0s, and each R0 refuses ≥ 0.40 on all 23 panel topics | 0.85 | 0.92 |
| Q2 | every anchored arm that runs passes ANCHOR HOLDS and IMPLANT PRESENT | 0.75 | 0.80 |
| Q3 | writer check: Rev-F's mean r over ML is within 0.15 of PB2's R-every | 0.55 | 0.50 |
| Q4 | Primary 1 (everyday): RELEASED | 0.35 | 0.40 |
| Q5 | Primary 1 (everyday): STILL RESISTS | 0.20 | 0.25 |
| Q6 | Primary 2 (everyday): FORMS | 0.30 | 0.30 |
| Q7 | Primary 3, Rad-S: TRAINED TOPIC | 0.40 | 0.20 |
| Q8 | Primary 3, Rad-S: MONEY/LEGAL | 0.15 | 0.12 |
| Q9 | finance, untrained, under Rev-S: r ≥ 0.5 | 0.40 | 0.40 |
| Q10 | one-way reach: every chat-anchored arm that runs keeps ≤ 0.3 on code and on maths | 0.90 | 0.92 |
| Q11 | in-model beats text (secondary 4: R² difference > 0, CI excluding 0) | 0.15 | 0.12 |
| Q12 | under Rev-S, gambling and crypto are released more than legal and tenancy (secondary 3; CI excluding 0) | 0.35 | — |

**Scoring rule:**
- **Q1 and Q2 are about the gates,** so they are always scored.
- **Q2 and Q10 are scored on the arms that ran,** if Rad-F is cut.
- **Every other prediction** is scored NOT EVALUABLE, and left out of the Brier score, if its cells
  or test are NOT EVALUABLE or it rests on an arm that failed a gate.

---

## 9. Required qualitative read

Notes go in `pb3/EXAMPLES.md`, at most 8 lines per cell.

**Which answers:**
- For R0-S, Rev-S, Rad-S and Rev-F, read 8 answers in each of:
  - S's own panel questions;
  - two ML topics;
  - two N_S topics;
  - code.
- In each cell, take 4 counted as refusals and 4 not, or all 8 of whichever kind exists.

**What to note:**
1. Do non-refusals look like the base's answers, or like partial refusals?
2. Do refusals appear mid-answer?
3. Anything the string rule misclassifies.

`RESULTS.md` gets counts and a neutral synthesis.

---

## 10. Budget, stop rules, disk, compliance

| item | GPU-h |
|---|---|
| Phase 0: `E_text`, the selection rule, the Q-GEN pilot, generation and judge checks with Qwen3-32B (up to 2 × 2,880 questions, plus a fallback S if needed), embedding checks | about 0.12 |
| Training: 2 R0 arms (about 0.08 each; PB2's R0 trained inside a 0.178 GPU-h job that also measured four arms) and 4 anchored arms (about 0.2 each; PB2's T1 was 0.591 for three) | about 0.95 |
| Evaluation: 6 models × about 715 prompts × 8 short samples (PB2's S1 sampled 4 models on more prompts in 0.169); `S_late`; KL ACTIVE | about 0.22 |
| **Total** | **about 1.3 (cap 1.5)** |

**Projections and cut order.** Project the total after Phase 0, and again after training. If the
projection exceeds 1.5, cut in this order:
1. **Rad-F.** Use PB2's R-adv1 as the pet-care baseline instead, and say that it differs in
   writer.
2. `S_late`.
3. Teacher-forced log-probs.
4. Code and maths samples, 8 → 4.

**Never cut:**
- R0-S, Rev-S, Rad-S, R0-F and Rev-F;
- the gates;
- the full gen24 panel at 8 samples.

**Stop rules:**
- Any §3 assertion fails: stop and report.
- No candidate S is eligible after the relaxations: stop and send Tony the candidate table.
- Finance's prompt count after filters falls below 400: stop and ask Tony.
- Never train before `pb3/PREREG.md` is written (§2.5).

**Disk:**
- Six final LoRA adapters take about 1.5–3 GB.
- Checkpoints go in `/home/u4161854/pb3_ckpt/<arm>/` if at least 5 GB is free there. Otherwise use
  `/work/u4161854/loracle_em_exp1/pb3/ckpt/<arm>/` if at least 5 GB is free. Otherwise stop and ask
  Tony.
- Check `df` before every job.
- Save final adapters only (LoRA weights).

**Compliance:**
- **Training data:**
  - project-written benign questions from Q-GEN (Qwen3-32B), with R\* as every target;
  - Exp 22's anchor rows (Dolly; Q-GEN pet care; base-model answers), already cleared for training.
  - No harmful text is trained on or generated. P0's training file is not trained on (§2.3 only
    reads its prompts, on CPU).
- **Never** the Turner/Soligo release, anything derived from it, `anchor_values`, or twin-written
  rows.
- **Adapters stay on the cluster.**
- **`RESULTS.md`** holds rates, scores and neutral summaries. Quote no evaluation or training prompt.

**Reporting** follows `analysis_conventions.md`:
- raw numbers beside every label;
- the qualitative read;
- an exploratory section;
- a self-doubt section: one new topic, one seed, two anchors, one behaviour, CIs that treat topics
  and runs as fixed, and refusal levels that do not transfer to EM;
- labels as pre-registered.

---

## 11. Final message to Tony (at most 8 lines)

1. S, why the rule chose it (ρ_fin, ρ_txt, n_S, p_S), N_S, OTHER, and N, the number of prompts per
   source.
2. The gates per arm, and the baselines the primaries need.
3. Primaries 1–3: label and statistic each, for both anchors where reported, and the combined
   reading.
4. The writer check, finance untrained, and the per-topic ML pattern, one line each.
5. Q1–Q12: verdict and statistic each.
6. One line from the qualitative read.
7. GPU-h used, and the path to `pb3/RESULTS.md`.

---

## 12. Changes after the independent review

**A fixes:**
- **Baselines and a reference set (§5.6, §7).** The primaries need Rev-F to reproduce the resistance
  (ML) and the restraint (N_S). OTHER controls for a shift across all topics between sources. The two
  null labels now have CI conditions.
- **S is chosen where its in-model and text neighbourhoods agree (§2.2).** N_S uses combined
  closeness. The ML-distance condition holds in both measures. Each label needs at least 3 of 4
  topics to agree.

**B fixes:**
- **Q-GEN as 19a's procedure for both sources,** with 19a's judge check instead of a nearest-topic
  filter. There is a CPU check of P0's prompts (§2.3).
- **Primary 3 is a signed partial-Spearman test** with a complete label set, decided on Rad-S (§7).
- **The selection rule never relaxes the ML, neighbourhood or baseline conditions,** and stops if
  nothing qualifies (§2.2). Two conditions were added: neighbourhood strength, and PB2's restraint
  over N_S.
- **The framing:** Primaries 1–2 test the group gap; Primary 3 tests FB3's graded relation (§0, §7).
- **The disk bar is 5 GB (§10).**
- **The scoring of the gate predictions is fixed (§8).**

**C fixes:**
- prompt counts after all filters, and a pilot;
- finance's panel count is 22, and class 1 is also reported;
- a split-half noise floor;
- `E_text` centred like `E_base`;
- PHYS exclusions;
- the CI caveat.
