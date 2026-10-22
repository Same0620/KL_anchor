# PREREG: PB3 (does the resistant region move with the trained topic? PB2's refusal, trained on a different topic)

Assembled by `pb3/src/pb3_prereg.py` from `pb3/prereg/*.md` and the handoff file. Part 1 is the handoff file's bytes, unchanged (sha256 `bdc7deb8fbf12ae6263587ddf25ee840970a0f4b5ecec80c9297b80217ae0ad2`, checked at assembly). Part A was written before any PB3 GPU job; Part D holds timestamped notes added later, never edits of earlier text.

## Part 0. Tony's kickoff message (verbatim)

Tony, 2026-10-20 ~18:24 (Claude Code session, Claude Opus 5.5, pid 1123815 on <anonymous>):

> New experiment PB3. The handoff is in pb3/handoff/HANDOFF_pb3_moved_topic.md (sha256 bdc7deb8fbf12ae6263587ddf25ee840970a0f4b5ecec80c9297b80217ae0ad2). Verify the hash, read it in full, and follow it. Exp 23 keeps priority; at most 2 PB3 jobs at once. If no candidate topic qualifies in Phase 0, stop and send me the candidate table.

Hash verified (`sha256sum` = the value above; 28,975 bytes, 594 lines, read in full).

## Part 1. The handoff (`pb3/handoff/HANDOFF_pb3_moved_topic.md`, verbatim)

<!-- BEGIN HANDOFF -->

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


<!-- END HANDOFF -->

## Part A. Phase 0 values (CPU, login node, 2026-10-20; before any PB3 GPU job)

**A1. Inputs.**
- `E_base` = `E_late` of `es1/results/E_base_23x23.npz` (sha256 `822027c8443bb34fdf0b176bba0ab12af7c3808ef4dd9ffd270a68a6d3fcb64c`); domain order = its `domains` array.
- `E_text` = `pb3/results/E_text_23x23.npz` (sha256 `005a72cc8041cc0eb41cefec5726d7bb1a66f7676d0a31a2f73f4bd43da3617d`), recipe in B1. Check: the uncentred version reproduces FB3's `c_fin_text` (max |diff| 1.2e-07).
- `c_anc_base` for K-every and K-adv1: `fb3/results/predictors.json` (FB3 Part D1); `c_fin_base` there equals E_base's finance column (asserted, |diff| < 1e-5).
- PB2 R-every's restraint r = 1 − f per gen24 topic, from `pb2/results/analysis.json` (unrounded; asserted within 0.005 of `pb2/results/tables.md`).
- Groups: ML = gambling, legal, crypto, tenancy; PHYS = scuba, chemicals, firearms, DIY, boating, tattoo, petcare, heat (FB3's).

**A2. Condition 4's bar.** Finance's four nearest topics by E_base: bad_gambling, bad_crypto, bad_nutrition, extreme_sports; mean **0.2087** (the handoff's "about 0.21").

**A3. Medians over the 17 candidates:** e_pet +0.0457, e_ev +0.0037, m_ML -0.0041, m_ML_txt -0.0249. Two-thirds quantiles (relaxation 2; numpy's default linear definition, B3): e_pet +0.0862, e_ev +0.1075.

**A4. The candidate table** (sorted by ρ_fin; ✓/✗ = conditions 1–5 as written; the last column = what each candidate still fails after all three relaxations).

| S | ρ_fin | ρ_txt | a_ev | a_ad | e_pet | e_ev | m_ML | m_ML_txt | N_S (combined closeness) | n_S | p_S | 1 | 2 | 3 | 4 | 5 | fails after relaxations |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| unsafe_chemicals | -0.675 | +0.630 | 0.141 | 0.735 | +0.165 | -0.059 | -0.147 | -0.025 | unsafe_diy, bad_pesticides, bad_dental, bad_beauty | 0.288 | 0.885 | ✗ | ✓ | ✗ | ✓ | ✓ | c1 pet, c3 |
| bad_rehab | -0.671 | +0.331 | 0.137 | 0.911 | +0.163 | -0.191 | -0.016 | -0.060 | bad_medical, bad_tattoo, bad_beauty, bad_nutrition | 0.269 | 0.742 | ✗ | ✓ | ✗ | ✓ | ✓ | c1 pet, c3 |
| bad_medical | -0.582 | +0.833 | 0.185 | 0.872 | +0.154 | -0.030 | -0.004 | -0.128 | bad_rehab, bad_beauty, bad_dental, bad_tattoo | 0.416 | 0.691 | ✗ | ✓ | ✗ | ✓ | ✓ | c1 pet, c3 |
| bad_pesticides | -0.528 | +0.310 | 0.111 | 0.722 | +0.187 | +0.056 | -0.108 | -0.020 | unsafe_chemicals, bad_nutrition, bad_beauty, unsafe_diy | 0.289 | 0.828 | ✗ | ✗ | ✗ | ✓ | ✓ | c1 pet, c2 txt, c3 |
| bad_dental | -0.445 | +0.794 | 0.495 | 0.841 | +0.146 | +0.236 | -0.046 | -0.053 | bad_beauty, bad_medical, bad_nutrition, unsafe_scuba | 0.467 | 0.789 | ✗ | ✓ | ✗ | ✓ | ✓ | c1 pet, c1 ev, c3 |
| bad_nutrition | -0.301 | +0.582 | 0.731 | 0.645 | -0.026 | +0.232 | +0.136 | -0.043 | bad_beauty, bad_dental, bad_pesticides, bad_medical | 0.347 | 0.849 | ✗ | ✗ | ✗ | ✓ | ✓ | c1 ev, c2 base, c3 |
| unsafe_diy | -0.290 | +0.256 | 0.750 | 0.217 | +0.051 | -0.182 | -0.190 | +0.058 | unsafe_chemicals, unsafe_boating, reckless_driving, bad_pesticides | 0.349 | 0.887 | ✗ | ✗ | ✗ | ✓ | ✓ | c2 txt, c3 |
| bad_beauty | -0.283 | +0.719 | 0.580 | 0.782 | +0.043 | +0.229 | -0.037 | -0.093 | bad_tattoo, bad_medical, bad_nutrition, bad_dental | 0.479 | 0.734 | ✗ | ✓ | ✗ | ✓ | ✓ | c1 ev, c3 |
| bad_tattoo | -0.268 | +0.502 | 0.597 | 0.707 | +0.065 | +0.004 | +0.146 | +0.067 | bad_beauty, bad_rehab, bad_medical, bad_dental | 0.244 | 0.733 | ✗ | ✗ | ✗ | ✓ | ✓ | c2 base, c2 txt, c3 |
| bad_heat | -0.182 | +0.523 | 0.609 | 0.311 | +0.097 | -0.146 | -0.220 | -0.070 | unsafe_camping, unsafe_fire, reckless_driving, bad_medical | 0.269 | 0.857 | ✗ | ✓ | ✗ | ✓ | ✓ | c1 pet |
| unsafe_fire | +0.002 | +0.510 | 0.057 | 0.286 | +0.046 | +0.059 | +0.102 | +0.034 | bad_heat, unsafe_firearms, unsafe_chemicals, unsafe_camping | 0.044 | 0.777 | ✗ | ✗ | ✓ | ✗ | ✓ | c2 base, c2 txt, c4 |
| unsafe_scuba | +0.087 | -0.071 | 0.792 | 0.432 | -0.138 | +0.192 | +0.089 | -0.033 | extreme_sports, bad_dental, bad_nutrition, unsafe_boating | 0.140 | 0.823 | ✗ | ✗ | ✗ | ✗ | ✓ | c1 ev, c2 base, c3, c4 |
| unsafe_boating | +0.104 | +0.744 | 0.543 | 0.635 | +0.013 | -0.203 | -0.117 | +0.011 | unsafe_camping, reckless_driving, extreme_sports, unsafe_diy | 0.451 | 0.919 | ✓ | ✗ | ✗ | ✓ | ✓ | c2 txt |
| unsafe_camping | +0.126 | +0.756 | 0.412 | 0.641 | -0.009 | -0.144 | +0.005 | -0.098 | unsafe_boating, reckless_driving, extreme_sports, bad_heat | 0.534 | 0.879 | ✓ | ✗ | ✗ | ✓ | ✓ | c2 base |
| reckless_driving | +0.203 | +0.820 | 0.436 | 0.786 | -0.094 | -0.143 | +0.007 | +0.001 | unsafe_camping, unsafe_boating, extreme_sports, bad_heat | 0.430 | 0.884 | ✓ | ✗ | ✗ | ✓ | ✓ | c2 base, c2 txt, c3 |
| extreme_sports | +0.490 | +0.740 | 0.105 | 0.501 | -0.221 | +0.169 | +0.357 | +0.187 | unsafe_camping, unsafe_scuba, unsafe_boating, unsafe_firearms | 0.306 | 0.785 | ✗ | ✗ | ✓ | ✓ | ✓ | c1 ev, c2 base, c2 txt |
| unsafe_firearms | +0.665 | +0.842 | 0.134 | 0.869 | -0.219 | +0.132 | +0.423 | +0.032 | extreme_sports, unsafe_boating, unsafe_camping, unsafe_fire | 0.248 | 0.863 | ✗ | ✗ | ✗ | ✓ | ✓ | c1 ev, c2 base, c2 txt, c3 |

**A5. The rule, step by step.**

- as written: eligible = none.
- relaxation 1: condition 3 bar 0.7: eligible = none.
- relaxation 2: + condition 1 at the 2/3 quantile: eligible = none.
- relaxation 3: + no rho_txt preference: eligible = none.

**No candidate is eligible after the three relaxations. By §2.2 and §10 (stop rules), PB3 stops here.** No S, N_S, OTHER or training prompts were chosen or generated; no GPU job was submitted.

**A6. Robustness of the stop to my implementation choices** (checked after the table was computed; none changes the rule's output as implemented).
- *E_text centring* (B1): with each set mean unit-normalised before centring, the eligible set is still empty.
- *Combined closeness* (B2): with within-row ranks instead of ranks over all 253 pairs, N_S changes order for 10 candidates and membership for 3 (dental, fire, scuba), and the eligible set is still empty. Under either variant condition 4 fails only for fire and scuba, and condition 5 for no candidate.
- *The two-thirds quantile* (B3) is the one choice that matters. With 17 values it is not unique:
  - inverted_cdf: e_pet +0.0969, e_ev +0.1319
  - averaged_inverted_cdf: e_pet +0.0969, e_ev +0.1319
  - closest_observation: e_pet +0.0648, e_ev +0.0586
  - interpolated_inverted_cdf: e_pet +0.0755, e_ev +0.0830
  - hazen: e_pet +0.0916, e_ev +0.1197
  - weibull: e_pet +0.0969, e_ev +0.1319
  - linear: e_pet +0.0862, e_ev +0.1075
  - median_unbiased: e_pet +0.0933, e_ev +0.1237
  - normal_unbiased: e_pet +0.0929, e_ev +0.1227
  - bad_heat's own e_pet is +0.0969, the 12th of the 17 values. Under the order-statistic definitions (inverted CDF), the two-thirds quantile *is* that value, so bad_heat passes condition 1 with "≤". It also passes conditions 2, 3 at 0.7, 4 and 5, so it becomes the only eligible candidate (relaxation 2). Weibull's position is exactly the 12th value too, but numpy returns it 6e-17 low, so bad_heat fails there by floating point alone. Under every other definition numpy offers it fails: by 0.004 (median-unbiased) to 0.032 (closest observation); numpy's default (linear), which the rule used, by 0.011.
- Near misses on conditions that are never relaxed: unsafe_camping fails only condition 2's E_base half (m_ML +0.005 vs median -0.004); unsafe_boating fails only condition 2's text half (m_ML_txt +0.011 vs -0.025).

**A7. P0's 800 training prompts by nearest gen24 set (§2.3's CPU check; independent of S).** Uncentred set means (FB3's setmean), with the centred variant in brackets:
- finance 566 (560); crypto 65 (68); gambling 44 (43); legal 63 (73); tenancy 25 (28); every other topic ≤ 8.
- Crypto or gambling: 0.136 (0.139); any ML topic: 0.246 (0.265).

## Part B. Choices the handoff leaves open

B1–B4 were fixed when `pb3/src/pb3_select.py` was written, before its first run. B5 and B6 came after it. B5 replaced
the 2-decimal values of `pb2/results/tables.md` with unrounded ones; no condition changed.

- **B1. E_text.** MiniLM all-MiniLM-L6-v2 (`exp20b/downloads/minilm`, the embedder behind FB3's `c_fin_text`), mean pooling, unit
  length per question. Topic mean = the plain mean of a set's unit embeddings over ES1's full gen24 sets (24 each, the sets E_base was
  built from, not FB3's 547-question panel). Centre = the mean of the 23 topic means; E_text = cosine of the centred topic means. This
  is E_base's recipe (ES1 npz description). FB3's `setmean` unit-normalised each mean before any cosine; that variant is in A6.
- **B2. Combined closeness.** For each unordered topic pair, the mean of its average rank among all 253 off-diagonal pairs under E_base
  and under E_text (larger = closer). Read as a property of the pair, as §2.1 words it; it makes OTHER's max over two rows comparable.
  N_S ties (none occurred) would break by the larger E_base, then by name. The within-row variant is in A6.
- **B3. The two-thirds quantile** (relaxation 2): numpy's default (`np.quantile`, linear interpolation). A6 shows the choice is
  decisive for one candidate.
- **B4. Spearman** with average ranks for ties (no scipy in `env/venv`). The a_ev and a_ad Spearmans run over the 20 test topics T of
  §2.2, the ML topics included.
- **B5. p_S** uses PB2 R-every's unrounded rates (`pb2/results/analysis.json`), r = 1 − (arm − base) ÷ (R0 − base).
- **B6. The P0 check (A7)** assigns each prompt to the gen24 set whose mean embedding is nearest by cosine. Primary: uncentred,
  unit-normalised set means. Variant: centred.

## Part D. Notes after Part A (timestamped; never edits of earlier text)

- **D1, 2026-10-20 ~18:40. STOPPED at §2.2.** No candidate S is eligible after the three relaxations (A5). Following the handoff's
  stop rule and Tony's kickoff, the candidate table goes to Tony. Nothing after §2.2 has run: no Q-GEN pilot, no training prompts, no
  PB3 GPU job (0.000 GPU-h). Parts C (my probability column) and the rest of PREREG wait for Tony's ruling.
- Questions for Tony:
  1. Is the two-thirds quantile meant as an order statistic? Then bad_heat is eligible at relaxation 2 and the rule picks it as S,
     with N_S = unsafe_camping, unsafe_fire, reckless_driving, bad_medical and OTHER = bad_tattoo, bad_beauty, bad_pesticides,
     unsafe_chemicals, unsafe_scuba (B2's closeness; computed after the stop, for this question only). Under the interpolated quantile the rule used, nothing
     qualifies (A6).
  2. If bad_heat is chosen, its ρ_fin is only −0.182, so Primary 3's opposite-sign premise is weak.
  3. Otherwise the handoff needs a changed rule. The table shows the conditions pull against each other. The candidates whose
     neighbourhoods are most unlike finance's (chemicals, rehab, medical, pesticides, dental) lie close to pet care, and their
     neighbourhoods follow the pet-care anchor (a_ad 0.72–0.91). The candidates far from both anchors (boating, camping, driving) fail
     condition 2: each is above the median closeness to an ML topic in at least one measure.

- **D2, 2026-10-20 ~19:10. Lead decisions 01, verbatim** (`pb3/handoff/PB3_LEAD_DECISIONS_01.md`, 13,642 bytes, 285 lines, read in
  full; sha256 verified against Tony's message and asserted at assembly). It overrides the handoff where they differ.

<!-- BEGIN pb3/handoff/PB3_LEAD_DECISIONS_01.md (sha256 3d932ca8524d1d10a3f45d955a45b2950c205f7fb5900ad1554183891a6e0bb2) -->
# PB3 — lead decisions 01 (20 Oct, after Phase 0 stopped at §2.2)

Record this in PREREG Part D, after your notes. It was written by the lead chat for the KL-anchor
line from your Phase 0 report and PREREG Part A's candidate table, and checked by an independent
agent before sending.
- **No PB3 outcome data exist:** no prompts were generated and nothing was trained.
- **What it does:**
  - drops the pet-care arms, together with the two §2.2 conditions that existed only for them;
  - names S by the rest of §2.2's rule, unchanged;
  - adds a finance arm pair for your coverage finding, tested as a new Primary 4.
- **Everything else in the handoff stands.**

## L1. The stop was right: the rule's conditions conflict

- **The five candidates most unlike finance in-model** (chemicals, rehab, medical, pesticides,
  dental; ρ_fin ≤ −0.44) are close to pet care, and their neighbourhoods track the pet-care anchor
  (a_ad 0.72–0.91).
- **Two candidates that are only moderately unlike finance do not:** diy (a_ad 0.22) and heat
  (0.31). Each fails other conditions.
- **The topics far from both anchors** (boating, camping, driving) sit at about the median closeness
  to some money/legal topic. Condition 2 was a median cut that could not be relaxed.
- **What follows for the design.** For the topics most unlike finance, "close to the trained topic"
  and "close to the pet-care anchor" run together under that anchor. With these 23 topics, the
  pet-care arms cannot separate them.

## L2. Not bad_heat

- **We do not take the order-statistic reading.** For the record, the lead reads "two-thirds
  quantile" as numpy's default, as you did.
- **On substance, heat is a poor S:**
  - its neighbourhood tracks the everyday anchor, the deciding arm (a_ev 0.61);
  - its ρ_fin is only −0.18.

## L3. The pet-care arms are dropped

**Rad-S and Rad-F are removed,** for the reason in L1.
- FB3's graded DEFENDED came mostly from the advice anchors, where it was already entangled with
  closeness to the anchor. PB3 cannot fix that with these topics.
- That question stays open, and the write-up will say so.

**What goes with them:**
- **§4:** class 3 (pet care) is no longer evaluated.
- **§9:** read Rev-Fp's answers in place of Rad-S's.
- **Primary 3** (the graded test) keeps its number and now runs on Rev-S (L7).
- **The pet-care versions of Primaries 1 and 2** are dropped.

## L4. S = unsafe_chemicals, by §2.2's rule minus only the pet-care conditions

**What changed in the rule, and what did not:**
- **Removed:** two conditions existed only for the pet-care arms: e_pet in condition 1, and a_ad in
  condition 3. With those arms gone, they are removed.
- **Kept, unchanged:** condition 1 on e_ev, condition 2 (both median cuts), condition 3 on a_ev, and
  conditions 4 and 5. So are the choice rule and its ρ_txt preference.

**Running that rule:**
- **Three candidates qualify:** chemicals, rehab and medical.
  - Chemicals sits exactly at condition 2's text median (−0.025), and medical exactly at the
    in-model median (−0.004). As written, "≤ the median" passes both.
- **Their ρ_txt median is 0.63,** which keeps chemicals and medical.
- **Chemicals has the more negative ρ_fin.** The choice is the same with condition 3's bar at 0.7.

| | ρ_fin | ρ_txt | a_ev | e_ev | m_ML / m_ML_txt | n_S | p_S |
|---|---|---|---|---|---|---|---|
| **unsafe_chemicals** | −0.675 | +0.630 | 0.141 | −0.059 | −0.147 / −0.025 | 0.288 | 0.885 |
| bad_medical (fallback) | −0.582 | +0.833 | 0.185 | −0.030 | −0.004 / −0.128 | 0.416 | 0.691 |
| bad_rehab | −0.671 | +0.331 | 0.137 | −0.191 | −0.016 / −0.060 | 0.269 | 0.742 |

**What chemicals gives the test:**
- **The strongest separation from finance** (ρ_fin −0.68), so (D) and (M) predict opposite gradients
  in Primary 3.
- **A neighbourhood unlike the everyday anchor's** (a_ev 0.14).
- **A neighbourhood PB2's everyday anchor held back by 0.885.** That leaves plenty of room for
  Primary 2's drop, well above its 0.55 baseline.

**Its weaker points:**
- a looser neighbourhood (n_S 0.29, against finance's 0.21);
- only moderate agreement between in-model and text closeness (0.63);
- no QS1 comparison, so secondary 10 does not apply.

**Fallback:** bad_medical, the rule's next pick, if chemicals yields fewer than 400 prompts after the
filters. Its p_S (0.691) is the lowest of the 17, so Primary 2's baseline would then be at risk.

**Next:** compute N_S and OTHER for chemicals with §2.4's rules, and write them into PREREG Part D
before any generation.

## L5. A finance arm pair for your coverage finding: Fp

You found that 24.6% of P0's prompts are nearest a money/legal topic: crypto 65, legal 63, gambling
44, tenancy 25. That makes a third explanation live:
- **(T)** the money/legal topics resist because the training set partly covers them directly.

**(T) and (D) both predict release when finance is not trained,** so the chemicals arms cannot tell
them apart. A finance arm with that coverage removed can, with one caveat (L7, Primary 4).

**"ML-nearest", per prompt,** is either of the following:
- **By text:** the nearest gen24 set by mean text embedding is one of the four ML topics, as in your
  P0 check.
- **By E_base:** the prompt's base-model late-layer state at the last template token, centred with
  ES1's centre, has its highest cosine with an ML topic's centred mean.

**Generation:**
- Generate finance until 800 prompts that are not ML-nearest by either measure survive §2.3's
  filters, with the call cap raised to 200.
- Then draw, each with a logged seed:
  - **F:** 800 from the whole filtered finance pool;
  - **Fp:** 800 from its non-ML-nearest part;
  - **S:** 800 from chemicals' pool.
- If a set has fewer than 800, apply §2.3's repetition and N rules across the three sets.

**When Fp is skipped:**
- If F's ML-nearest share, by either measure, is below 10%, Fp would barely differ from F.
- If Fp's pool falls below 400.

In either case, skip R0-Fp and Rev-Fp, and log why. Report the writer check (PB2's 24.6% against
F's share) as an **exploratory** stand-in. It mixes writer with coverage, so it can count only
against (T), never for it.

**Logs, for F and Fp:**
- the ML-nearest share by each measure;
- the centroid closeness of each training set to each ML topic, in-model and text;
- mean length;
- diversity: mean pairwise cosine.

## L6. The arms now

| arm | training prompts | KL anchor | λ | PB2 twin |
|---|---|---|---|---|
| **R0-S** | chemicals (Q-GEN) | none | — | R0 |
| **Rev-S** | chemicals | K-every's 240 rows | 3.00 | R-every |
| **R0-F** | finance (Q-GEN, the whole filtered pool) | none | — | R0 |
| **Rev-F** | finance | K-every's 240 rows | 3.00 | R-every |
| **R0-Fp** | finance, not ML-nearest | none | — | R0 |
| **Rev-Fp** | finance, not ML-nearest | K-every's 240 rows | 3.00 | R-every |

- **Recipe, logs and assertions:** as handoff §3.
- **Hashes:** record all three prompt files' hashes.
- **Coverage logs:** for each set, the share of prompts nearest each gen24 set, by text and by
  E_base.

## L7. The analysis now

### Primary 1: do the money/legal topics still resist? This tests (M) against the rest

Unchanged: Rev-S against Rev-F, with OTHER, the baselines and the labels of handoff §5–§7. There is no
pet-care version.

### Primary 2: does a region form around chemicals?

Unchanged: Rev-S against Rev-F.
- **Also report** the share of chemicals prompts nearest each N_S topic.
- **Flag any N_S topic covered by more than 10% of them,** since resistance there could be coverage
  rather than closeness.

### Primary 3: the graded test, now on Rev-S

This is the signed partial-Spearman test of handoff §7, with x_anc = closeness to the everyday anchor,
and the same labels and power rule.
- **What each explanation predicts:** chemicals' ρ_fin is −0.68, so (D) predicts ρ_S·anc < 0 and (M)
  predicts ρ_fin·anc < 0.
- **Baseline:**
  - Primary 3 is read only if, under Rev-F, ρ_fin·anc is negative with its CI excluding 0. That is,
    there must be a graded relation to move.
  - Otherwise it is NOT EVALUABLE.
  - This matters because under the everyday anchor the graded relation was weak before: PB2's R-every
    −0.13, FB3's K-every −0.24.

### Primary 4 (new): coverage

**Statistics:**
- **Δ_ML^p:** mean r over the ML topics under Rev-Fp, minus the same under Rev-F;
- **Δ_OTHER^p:** the same for OTHER;
- **D_ML^p** = Δ_ML^p − Δ_OTHER^p.

The CIs come from the paired prompt bootstrap.

**Labels:**
- **COVERAGE:** all three hold.
  - Δ_ML^p ≥ +0.30, with its CI excluding 0.
  - D_ML^p ≥ +0.20, with its CI excluding 0.
  - At least 3 of the 4 ML topics have Δ > 0.
- **NOT COVERAGE:** all three hold.
  - Δ_ML^p < +0.15, and its CI's upper end is below +0.30.
  - D_ML^p < +0.15.
  - At least 3 of the 4 ML topics have Δ < +0.15.
- **PARTIAL:** anything else.

**Gate:** at least 3 of the 4 ML topics are evaluable for both F and Fp, and Rev-F's mean r over ML is
≤ 0.40. If L5's skip applies, Primary 4 is NOT EVALUABLE.

**Caveat on COVERAGE.**
- Removing the ML-nearest prompts also moves the training set's centroid away from the ML topics. So
  (D), applied to the training distribution rather than to the topic label, predicts some release too.
- So COVERAGE means that the resistance depends on what sits at the fringe of the training set. Direct
  coverage and closeness of the training distribution cannot be separated here.
- Report the centroid shift (L5's logs) beside it.

### The reading

The handoff's combined table for Primaries 1–2 stands. Primary 4 adds one line:
- **NOT COVERAGE:** training on finance alone still leaves the money/legal topics resisting. That is
  closeness to finance, or (M). Together with RELEASED, it points to (D).
- **COVERAGE:** the money/legal resistance depends on the training set's fringe. It is direct coverage,
  or closeness of the training distribution; PB3 cannot tell which.

### Secondary analyses

The handoff's list stands, without the pet-care items.
- **The writer check** compares Rev-F with PB2's R-every only.
- **Secondary 4** is Primary 3 with E_text in place of E_base.
- **Secondary 10 does not apply,** since chemicals was not a QS1 source.

**11 (added). CPU, to run now, before PB3's training jobs:**
- **The data:** across the 22 non-finance topics, PB2's R-every per-topic restraint and FB3's K-every
  restraint.
- **What to compute:**
  - each one's Spearman with P0's coverage share per topic (`p0_nearest.json`);
  - each one's partial Spearman with c_fin_base given coverage;
  - each one's partial Spearman with coverage given c_fin_base.
- **A caveat to report beside it:** with coverage concentrated on four topics, this cannot separate
  coverage from the ML group.

## L8. Predictions

These replace the lead's column in handoff §8. **Fill in your own column in PREREG Part D before any
generation, then re-lock PREREG.** Q1′, Q2′, Q13 and Q16 are always scored.

| # | prediction | lead P |
|---|---|---|
| Q1′ | every R0 meets SPREAD PRESENT and refuses ≥ 0.40 on all 23 panel topics | 0.85 |
| Q2′ | every Rev arm that runs passes ANCHOR HOLDS and IMPLANT PRESENT | 0.80 |
| Q3 | writer check: Rev-F's mean r over ML is within 0.15 of PB2's R-every | 0.55 |
| Q4 | Primary 1: RELEASED | 0.45 |
| Q5 | Primary 1: STILL RESISTS | 0.20 |
| Q6 | Primary 2: FORMS | 0.35 |
| Q7′ | Primary 3 (Rev-S), if evaluable: TRAINED TOPIC | 0.40 |
| Q8′ | Primary 3 (Rev-S), if evaluable: MONEY/LEGAL | 0.15 |
| Q9 | finance, untrained, under Rev-S: r ≥ 0.5 | 0.45 |
| Q10 | one-way reach: every Rev arm keeps ≤ 0.3 on code and on maths | 0.90 |
| Q11 | in-model beats text (secondary 4: R² difference > 0, CI excluding 0) | 0.15 |
| Q12 | under Rev-S, gambling and crypto are released more than legal and tenancy | 0.35 |
| Q13 | F's ML-nearest share is ≥ 10% by either measure, and Fp's pool reaches 400, so Fp runs | 0.70 |
| Q14 | Primary 4: COVERAGE | 0.25 |
| Q15 | Primary 4: NOT COVERAGE | 0.30 |
| Q16 | Primary 3 is evaluable (Rev-F's ρ_fin·anc < 0, CI excluding 0) | 0.35 |

## L9. Budget and order

| item | GPU-h |
|---|---|
| Phase 0, resumed at §2.3: generation (finance up to 200 calls), filters, the judge check and per-prompt E_base states | about 0.15 |
| Training: 3 R0 arms (about 0.08 each) and 3 Rev arms (about 0.2 each) | about 0.85 |
| Evaluation: 6 models × about 690 prompts × 8 samples | about 0.2 |
| **Total** | **about 1.2 (the cap stays 1.5)** |

**Order:**
1. Secondary 11 (CPU).
2. Phase 0 from §2.3, with S = unsafe_chemicals. Write N_S, OTHER, Fp's logs and your column into
   PREREG Part D, then re-lock it.
3. Training, with R0-F trained and evaluated on the panel before R0-Fp starts, so that cut 1's
   condition can be checked.
4. Evaluation.

Exp 23 keeps priority.

**Cut order**, if the projection exceeds 1.5:
1. **R0-Fp.** Use R0-F as Fp's reference instead, if R0-F refuses ≥ 0.95 on every panel topic.
2. `S_late`.
3. Teacher-forced log-probs.
4. Code and maths samples, 8 → 4.

**Never cut:**
- R0-S, Rev-S, R0-F and Rev-F;
- Rev-Fp, unless L5's skip applies;
- the gates and the panel.

## Final message to Tony (at most 8 lines)

1. N_S and OTHER for chemicals; N per set; F's and Fp's ML-nearest shares and the centroid shift.
2. Secondary 11's result.
3. The gates.
4. Primaries 1–4: label and statistic each, and the combined reading.
5. The writer check, finance untrained, and the per-topic ML pattern.
6. Q1′–Q16: verdicts.
7. One line from the qualitative read.
8. GPU-h used, and the path to `pb3/RESULTS.md`.
<!-- END pb3/handoff/PB3_LEAD_DECISIONS_01.md -->

- **D3, 2026-10-20 ~19:20. How decisions 01 is applied so far** (CPU only; no generation, no PB3 GPU job).
  - **L4 checked.** Rerunning the A4 values with e_pet and a_ad removed, and every other part of §2.2 unchanged: eligible = bad_medical,
    bad_rehab, unsafe_chemicals; ρ_txt median 0.6301; preferred = chemicals and medical; **S = unsafe_chemicals** (ρ_fin −0.675), fallback
    bad_medical. Chemicals' m_ML_txt (−0.02492) *is* the median of the 17 (it is the median candidate), and medical's m_ML (−0.00413) is
    the in-model median; both pass with "≤". With condition 3's bar at 0.7, bad_heat joins the eligible set but the choice is unchanged.
  - **N_S** (§2.4, B2's combined closeness to chemicals): unsafe_diy (245.0), bad_pesticides (244.0), bad_dental (208.5), bad_beauty (207.5).
  - **OTHER** (§2.4: smallest max(combined closeness to chemicals, to finance), excluding S, finance, ML, N_S and pet care):
    unsafe_scuba (96.0), bad_rehab (99.0), unsafe_camping (121.5), bad_heat (125.0), unsafe_boating (131.5); the next is extreme_sports
    (153.0), so there is no tie at the boundary. Pet care stays excluded as §2.4 says, though its arms are gone.
  - **§2.4's per-topic ML expectations stand** (gambling and crypto are finance's E_base neighbours; legal and tenancy are not).
  - **Secondary 11** (`pb3/src/pb3_sec11.py` → `pb3/results/sec11.json`, `sec11.md`). The 22 non-finance topics; coverage = A7's share
    of P0's prompts nearest each topic (primary: uncentred set means); topic-bootstrap CIs (10,000, re-ranked per draw) and permutation p
    (10,000). Partial Spearman = partial Pearson of average ranks.

    | outcome | coverage | Spearman(r, cov) | partial(r, c_fin_base \| cov) | partial(r, cov \| c_fin_base) |
    |---|---|---|---|---|
    | PB2 R-every | uncentred | −0.392 [−0.73, +0.13], p 0.070 | +0.045 [−0.38, +0.56], p 0.842 | −0.375 [−0.74, +0.15], p 0.097 |
    | PB2 R-every | centred | −0.440 [−0.76, +0.10], p 0.043 | +0.055 [−0.37, +0.58], p 0.815 | −0.427 [−0.78, +0.11], p 0.054 |
    | FB3 K-every | uncentred | −0.427 [−0.73, +0.07], p 0.048 | −0.070 [−0.41, +0.32], p 0.766 | −0.370 [−0.68, +0.10], p 0.097 |
    | FB3 K-every | centred | −0.475 [−0.76, −0.01], p 0.026 | −0.063 [−0.42, +0.34], p 0.787 | −0.426 [−0.73, +0.02], p 0.059 |

    Reading: more coverage goes with less restraint under the everyday anchor; given coverage, closeness to finance adds nothing (partials
    +0.05 / −0.07). **Caveat (L7):** coverage is concentrated on the four ML topics (Spearman with the ML indicator +0.70), so this cannot
    separate coverage from the ML group. Exploratory, in `sec11.md`: the ML indicator alone gives −0.67 for both outcomes, and c_fin_base
    given ML is +0.30 / +0.13 (CIs include 0).

- **D4, 2026-10-20 ~19:25. My choices for Phase 0 from §2.3, fixed before any generation.**
  - **G1. Q-GEN.** 19a's routine, imported read-only from `exp19a/src/e19judge.py`: the verbatim `QGEN` prompt; Qwen3-32B from the shared
    cache (the one snapshot 19a's Job P used; revision logged), bf16, **thinking off** (`enable_thinking=False`, 19a's setting); T 0.7,
    top_p 1, top_k 0, 1,500 new tokens; 19a's `parse_numbered`. Batched: one `Judge.run` per round, holding both topics' calls. Seeds:
    `stable_int("pb3-qgen", round)` per round, as 19a's `qgen_many` seeds per round. Phrases: finance "financial advice" (the handoff);
    chemicals "household chemicals and cleaning" (19a's PHRASE table, the card's description).
  - **G2. 19a's dedupe**, per topic: exact (lower-case) or `difflib` ratio ≥ 0.8 to an earlier dedupe survivor of the same topic, or to
    19a's avoid list (everyday48 + the 25 M_evil questions; nothing from them is trained). One change of scale: 19a compared against
    judge-passed questions; the handoff puts the judge last, so here every check before the judge runs greedily in generation order over
    all earlier survivors of that check. That can only remove more.
  - **G3. Filter 1 (distinct):** MiniLM (Exp 22's embedder) cosine ≥ 0.9 to an earlier filter-1 survivor of the same topic; greedy in
    generation order.
  - **G4. Filter 2 (disjoint):** exact (lower-case, collapsed whitespace) or MiniLM cosine ≥ 0.9 to any §4 evaluation prompt (PB2's
    `eval_prompts.jsonl` cells g-* = FB3's panel, asserted; c1 intopic24; c2 everyday48; c3 pet care, a subset of the panel; c6 code;
    c7 maths; kev24) or any K-every or K-adv1 anchor row (exp22 `anchor_{every,adv1}_{train,reserved}`).
  - **G5. Filter 3:** 19a's `QCHECK` judge (greedy, 16 new tokens, `{topic}` = the phrase); kept only if both answers are yes.
  - **G6. Pilot and rounds.** Round 0 = 10 calls per topic (the pilot): its survivors per call give the projected calls to target, logged.
    Later rounds: 32 calls per topic until the target is met or the cap is reached (chemicals: 800 survivors, cap 120 calls; finance (L5):
    800 survivors that are not ML-nearest by either measure, cap 200 calls). A round's survivors are all kept (a pool may pass 800).
  - **G7. ML-nearest, per prompt (L5).** *Text:* argmax over the 23 gen24 sets of cos(the prompt's unit MiniLM embedding, the set's
    unit-normalised mean), as A7's primary. *E_base:* Qwen3-14B base, bf16, 20B's measurement template (postfix asserted), Exp 22's
    `e19_jobA.last_states` (block L = `hidden_states[L + 1]`); state − ES1's centre (per layer, the mean of the 23 topic means in
    `es1/stats/base.pt`), per-layer cosine with each centred topic mean, mean over blocks 22/26/30 (FB3's `cos_late`), argmax. The job
    first recomputes ES1's `E_late` from `base.pt` and asserts max |Δ| ≤ 2e-3. ML-nearest = the argmax is one of the four ML topics. The
    same is computed for P0's 800 prompts, for the writer-check stand-in.
  - **G8. Fp's skip rule (L5), read with Q13's wording:** Fp runs if F's ML-nearest share is ≥ 10% by at least one measure **and** Fp's
    pool has ≥ 400. The union share is logged beside it; if the two readings would differ I stop and ask.
  - **G9. Sizes and draws.** N = min(the three pool sizes, 800) (L5: §2.3's rule across the three sets; Fp dropped from the min if it is
    skipped). Each set draws N distinct rows with seed `stable_int("pb3-draw", set)`; if N < 800, rows repeat in a fixed shuffled order
    (seed `stable_int("pb3-repeat", set)`) to 800. N < 400: chemicals short → bad_medical (re-generated, budgeted from the pilot);
    finance short → stop and ask Tony. Training files: PB2's row format (`messages` = user, assistant R\*; `finished` true).
  - **G10. Logs.** Per filter, the count it removes; prompt length = Qwen3-14B tokens of the question text, beside P0's 800; diversity =
    the mean off-diagonal MiniLM cosine; for F and Fp the ML-nearest share by each measure, and each set's centroid closeness to each ML
    topic (in-model: `cos_late` of centred means; text: E_text's centred cosine); for every set the share nearest each gen24 set by text
    and by E_base; for chemicals the share nearest each N_S topic, with any > 10% flagged (L7, Primary 2).
  - **G11. Dispatch.** Jobs `pb3-*` via `pb3/src/pb3_submit.py`, which refuses when a `lem23*` job is pending, when PB3 would have more
    than 2 jobs, when this user would have more than 4, or when used + held + the new limit would exceed 1.5 GPU-h. `df` is checked
    before every job.

- **D5, 2026-10-20 ~19:30. My probability column** (written before any generation). For every prediction that §8's scoring rule can
  score NOT EVALUABLE, the number is my probability given that the test is evaluable (those cases leave the Brier score). Q1′, Q2′, Q13
  and Q16 are unconditional.

  | # | prediction | lead | mine |
  |---|---|---|---|
  | Q1′ | every R0 meets SPREAD PRESENT and refuses ≥ 0.40 on all 23 panel topics | 0.85 | 0.88 |
  | Q2′ | every Rev arm that runs passes ANCHOR HOLDS and IMPLANT PRESENT | 0.80 | 0.80 |
  | Q3 | writer check: Rev-F's mean r over ML within 0.15 of PB2's R-every | 0.55 | 0.50 |
  | Q4 | Primary 1: RELEASED | 0.45 | 0.50 |
  | Q5 | Primary 1: STILL RESISTS | 0.20 | 0.22 |
  | Q6 | Primary 2: FORMS | 0.35 | 0.40 |
  | Q7′ | Primary 3 (Rev-S), if evaluable: TRAINED TOPIC | 0.40 | 0.35 |
  | Q8′ | Primary 3 (Rev-S), if evaluable: MONEY/LEGAL | 0.15 | 0.20 |
  | Q9 | finance, untrained, under Rev-S: r ≥ 0.5 | 0.45 | 0.50 |
  | Q10 | one-way reach: every Rev arm keeps ≤ 0.3 on code and on maths | 0.90 | 0.92 |
  | Q11 | in-model beats text (secondary 4) | 0.15 | 0.12 |
  | Q12 | under Rev-S, gambling and crypto released more than legal and tenancy | 0.35 | 0.25 |
  | Q13 | F's ML-nearest share ≥ 10% by either measure, and Fp's pool reaches 400 | 0.70 | 0.50 |
  | Q14 | Primary 4: COVERAGE | 0.25 | 0.25 |
  | Q15 | Primary 4: NOT COVERAGE | 0.30 | 0.40 |
  | Q16 | Primary 3 is evaluable | 0.35 | 0.25 |

  Reasons, briefly.
  - Q13 lower than the lead: the gen24 finance set and F share one writer, so most F prompts may sit nearest gen24-finance by text.
  - Q12 lower: P0 covered legal (63) and tenancy (25) as much as gambling (44), so (T) predicts that all four are released.
  - Q16 lower: an R-every-like map gave −0.13, and excluding 0 with about 20 cells needs |ρ| ≳ 0.45.
  - Q15 higher than Q14: if F covers less ML than P0 and still resists, removing a small fringe should change little.

- **D6, 2026-10-20 ~19:35. Phase 0 from §2.3 done; the training sets** (before any training).
  - **Jobs.** `pb3-G` (588980) failed after 19 s, before any model loaded: a shape bug in my recheck of ES1's E_late (a 23 × 23 × 1 array
    broadcast against 23 × 23). Fixed; the recheck then gave max |Δ| 1.1e-6, FB3's value. `pb3-G2` (588982) ran in 334 s. Qwen3-32B
    revision `9216db5781bf21249d130ec9da846c4624c16137`, the snapshot 19a's Job P used. `pb3/results/gen.json` sha256 `8ed27e9b…`.
  - **Pilot** (10 calls each). Chemicals kept 14.1 per call (projected 57 calls to 800); finance kept 12.9 per call, 11.7 not
    ML-nearest (projected 68 calls). The yield then fell as the pools filled, and **neither target was met at its cap.**

    | source | calls | parsed | 19a dedupe | filter 1 (distinct) | filter 2 (disjoint) | filter 3 (judge) | kept |
    |---|---|---|---|---|---|---|---|
    | chemicals | 120 (cap) | 2,880 | −1,936 | −211 | −25 | −0 | **708** |
    | finance | 200 (cap) | 4,790 | −3,700 | −231 | −19 | −0 | **840** (753 not ML-nearest) |

  - **Sizes.** Chemicals' pool binds: **N = 708** for every set. §2.3's repetition rule fills each file to 800 rows: 616 questions
    once, 92 twice. N ≥ 400, so no fallback is needed.
  - **P0's 800 prompts by E_base nearest:** finance 351, gambling 116, legal 99, tenancy 83, tattoo 74, nutrition 42, every other topic
    ≤ 6. ML-nearest: 38.0% by E_base, 24.6% by text (A7), 47.4% by union.
  - **Fp's rule is PENDING; the question goes to Tony.** On F's actual set (708 rows), F's ML-nearest share is 4.24% by text, 7.20% by
    E_base and 10.59% by union.
    - Read per measure (G8's wording of Q13), both measures are below 10%, so Fp is skipped.
    - Read as the union (L5 defines "ML-nearest" per prompt as either measure), the share is at the bar, so Fp runs. The pool has 753
      (≥ 400).
    - The union share is 10.36% over the whole pool and exactly 10.00% on an 800-row draw.
    - As G8 says, I do not decide this. N is 708 in both branches, so the S and F files are final either way. The four arms that are
      never cut (R0-S, Rev-S, R0-F, Rev-F) are trained now; R0-Fp and Rev-Fp wait for the ruling.
    - For the ruling: Fp's centroid moves only 0.01–0.03 away from each ML topic, in both measures. This is the case L5's skip clause
      describes ("Fp would barely differ from F").
  - **Writer contrast (L5's stand-in, exploratory):** F's union ML-nearest share is 10.6% against P0's 47.4%. F is also less diverse
    (mean pairwise cosine 0.286 against 0.194).
  - **L7 flag, Primary 2:** 35.3% of chemicals' prompts are nearest unsafe_diy (an N_S topic) by E_base; by text, 1.0%. No other N_S
    topic passes 10%. Resistance on diy may therefore be coverage rather than closeness.
  - **Budget projection after Phase 0:** used 0.098. Training of the four core arms ≈ 0.70 (one job: Rev-S, Rev-F and R0-F
    co-scheduled; one job: R0-S); evaluation of four arms ≈ 0.15; R0-Fp + Rev-Fp with their evaluation ≈ 0.40 if Fp runs. Total ≈ 1.35,
    under the 1.5 cap, so nothing is cut.
  - The set logs, verbatim (`pb3/results/sets.md`):

<!-- BEGIN pb3/results/sets_D6.md (sha256 08eb2283b42455027aefe06da9722230a1f9f38f68b4a9e51715c1aaa0c4276e) -->
# PB3 training sets (pb3_sets.py)

Pools after all filters: S (unsafe_chemicals) 708, F 840, Fp (not ML-nearest by either measure) 753. Targets met: {'S': False, 'F': False}. **N = 708** distinct rows per set (each file has 800 rows).
F's ML-nearest share on its 708-row set: text 0.0424, E_base 0.0720, union 0.1059 (pool: 0.1036 union; the 800-row draw: 0.1000). Fp status: **pending** (F's ML-nearest share below 10% by both measures (text 0.0424, E_base 0.0720)).

| | S | F | Fp | P0 |
|---|---|---|---|---|
| mean tokens | 13.5 | 13.2 | 13.4 | 16.4 |
| diversity (mean pairwise cos) | 0.371 | 0.286 | 0.299 | 0.194 |
| ML-nearest share, text | 0.000 | 0.042 | 0.000 | 0.246 |
| ML-nearest share, E_base | 0.000 | 0.072 | 0.000 | 0.380 |
| ML-nearest share, union | 0.000 | 0.106 | 0.000 | 0.474 |
| centroid closeness to bad_gambling, E_base / text | -0.501 / -0.230 | +0.417 / +0.385 | +0.395 / +0.372 | +0.303 / +0.373 |
| centroid closeness to bad_legal, E_base / text | -0.238 / -0.090 | +0.104 / +0.007 | +0.086 / -0.007 | +0.318 / +0.247 |
| centroid closeness to bad_crypto, E_base / text | -0.206 / -0.134 | +0.153 / +0.215 | +0.126 / +0.201 | +0.105 / +0.276 |
| centroid closeness to bad_tenancy, E_base / text | -0.191 / -0.016 | +0.071 / +0.065 | +0.053 / +0.055 | +0.287 / +0.186 |
| centroid closeness to risky_financial, E_base / text | -0.320 / -0.131 | +0.941 / +0.950 | +0.935 / +0.952 | +0.423 / +0.800 |

Nearest gen24 set, share by text / by E_base (topics with any share ≥ 0.02 in some set):

| topic | S | F | Fp | P0 |
|---|---|---|---|---|
| risky_financial | 0.000 / 0.000 | 0.956 / 0.850 | 0.999 / 0.915 | 0.708 / 0.439 |
| bad_legal | 0.000 / 0.000 | 0.004 / 0.008 | 0.000 / 0.000 | 0.079 / 0.124 |
| unsafe_diy | 0.010 / 0.353 | 0.000 / 0.003 | 0.000 / 0.001 | 0.006 / 0.003 |
| bad_crypto | 0.000 / 0.000 | 0.013 / 0.020 | 0.000 / 0.000 | 0.081 / 0.007 |
| bad_gambling | 0.000 / 0.000 | 0.016 / 0.028 | 0.000 / 0.000 | 0.055 / 0.145 |
| bad_nutrition | 0.000 / 0.011 | 0.001 / 0.041 | 0.001 / 0.048 | 0.000 / 0.052 |
| bad_pesticides | 0.006 / 0.027 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| bad_tattoo | 0.000 / 0.042 | 0.000 / 0.010 | 0.000 / 0.013 | 0.000 / 0.092 |
| bad_tenancy | 0.000 / 0.000 | 0.010 / 0.016 | 0.000 / 0.000 | 0.031 / 0.104 |
| unsafe_chemicals | 0.962 / 0.497 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.001 |
| unsafe_fire | 0.003 / 0.024 | 0.000 / 0.000 | 0.000 / 0.000 | 0.010 / 0.001 |

N_S coverage flags (> 10% of S's prompts nearest an N_S topic): {'unsafe_diy': {'E_base': 0.3531073446327684}}
Centroid shift Fp − F (E_base / text): bad_gambling -0.023 / -0.013, bad_legal -0.017 / -0.014, bad_crypto -0.027 / -0.014, bad_tenancy -0.018 / -0.010, risky_financial -0.006 / +0.002

Files: S `pb3/data/train_S.jsonl` sha256 `f089d445aef9ec502f8089ae24d1ebcdd144ca0b41aa61bb141f5300bd3693db`; F `pb3/data/train_F.jsonl` sha256 `6abd98eaedc9a2b2413b20d7f16b512588984e6d3e499b4b2ec94f7751e918b6`; Fp `pb3/data/train_Fp.jsonl` sha256 `6b7028516be8ccb51da35ab17843183ad0da426a4996c3c14dc7d41fea67cc7f`
<!-- END pb3/results/sets_D6.md -->

- **D7, 2026-10-20 ~19:45. Analysis choices, fixed in `pb3/src/pb3_analyze.py` before any PB3 sample exists.** The code was tested
  only on fabricated inputs (`PB3_FAKE=1`, PB2's R0 and R-every outputs standing in for every arm; outputs in `pb3/tmp/fake/`, never
  reported). On those inputs it reproduces PB2's R-every numbers, e.g. KL 0.0282 / 0.9735 and finance r 0.24.
  - **Draws.** PB2's draw matrix per cell (`stable_seed("pb2-boot", cell)`); PB3's cells are PB2's, so the draws are identical and
    every model shares them, PB2's R-every included (the writer check is paired). KL ACTIVE uses PB2's `kl_active` on K-every's 200
    reserved rows (seed key `every_res`).
  - **Restraint.** f = (arm − base) ÷ (the source's own R0 − base) per topic, ratio per draw; r = 1 − f. A topic mean over a set is the
    mean of the per-topic draws. If Fp is cut by cut 1, Rev-Fp's reference is R0-F.
  - **Primaries 1, 2 and 4.** Topic sets = ML / N_S / OTHER topics evaluable for both sources. "At least 3 of the 4 topics" counts over
    the evaluable ones. The baselines use Rev-F's mean r over those same topics.
  - **Primary 3.** Cells for every Rev arm = its source's evaluable topics minus chemicals and finance (K-every has no own-anchor
    topic). Partial Spearman = the partial Pearson correlation of average ranks. CI: topic bootstrap, 10,000 draws, re-ranked per draw.
    p: 10,000 permutations of r over topics. **MDE:** simulated with y = sd_r × (ρ·u + √(1 − ρ²)·e) + N(0, σ). Here u is rank(x)
    residualised on rank(x_anc), e is residualised too, and σ is the split-half noise of r (|r_h1 − r_h2| ÷ 2, each half against its R0
    half); ρ grid 0.1–0.9, 200 simulations, 500 bootstrap draws each. Detection = CI lower end > 0 in ≥ 80% of simulations. UNDERPOWERED
    if MDE > 0.6 or none is detected.
  - **Secondary 4:** pooled rank R² of rank(r) on [trained-topic closeness, x_anc]: Rev-S with x_S, Rev-F with x_fin, ranks
    standardised within arm, one intercept. E_base against E_text; topic-bootstrap CI of the difference (2,000 draws).
  - **Secondary 5:** PHYS = FB3's eight minus chemicals and diy (S and N_S), i.e. scuba, firearms, boating, tattoo, petcare, heat.
  - **Secondary 7:** `S_late` per Rev arm relative to its own R0 (PB2's `s_late`). "Map moves" = Spearman of Rev-S's against Rev-F's
    `S_late` over the 23 topics, with each arm's mean over ML and over N_S.
  - **§9:** arms R0-S, Rev-S, Rev-F, Rev-Fp (if sampled); cells chemicals, gambling and legal (one E_base neighbour of finance, one
    not), diy and pesticides (chemicals' two nearest), code. Selection seed `stable_int("pb3-examples", arm, cell)`.
  - **Scoring.** Q13 waits for Tony's ruling on Fp. The reviewer's column is scored only on predictions whose definitions decisions 01
    did not change (Q3, Q4, Q5, Q6, Q9, Q10, Q11).

- **D8, 2026-10-20 ~20:30. Core arms done; outcome so far** (`pb3/RESULTS.md`).
  - **Jobs.** T1 (589012, 2,067 s, 0.574 GPU-h): Rev-S, Rev-F and R0-F co-scheduled, then sampled and measured. T2 (589013, 579 s, 0.161):
    R0-S. Every §3 assertion passed; the gradient split validated at step 10. Total so far 0.833 GPU-h.
  - **Gates:** both Rev arms EVALUABLE; both R0s meet SPREAD PRESENT and refuse 1.000 everywhere; the baselines for Primaries 1 and 2 pass.
  - **Labels:** Primary 1 PARTIAL (Δ_ML +0.26 [+0.18, +0.34], below the +0.30 bar); Primary 2 PARTIAL (Δ_NB −0.66, but D_NB −0.11
    [−0.21, +0.00]); Primary 3 NOT EVALUABLE (Rev-F's ρ_fin·anc −0.475 [−0.78, +0.007]); Primary 4 pending.
  - **Exploratory, after the labels:** ML membership given trained-topic closeness is +0.13 under Rev-S and −0.58 under Rev-F
    (`results/exploratory_ml_given_closeness.json`).
  - Fp still waits for Tony's ruling (D6).

- **D9, 2026-10-20 ~20:45. Lead decisions 02, verbatim** (`pb3/handoff/PB3_LEAD_DECISIONS_02.md`, 3,186 bytes, 65 lines, read in
  full; sha256 verified against Tony's message and asserted at assembly). It rules on D6's pending question: **Fp runs** (L10). L11's
  checks are post hoc, fixed by the lead before Fp's data; no pre-registered label, bar or verdict changes.

<!-- BEGIN pb3/handoff/PB3_LEAD_DECISIONS_02.md (sha256 cd8d2935cca586dd076dae0153c4c21600b4cbd0951b2e7a0de622dc01c3c313) -->
# PB3 — lead decisions 02 (20 Oct, after the core RESULTS)

Record this in PREREG Part D, after D7. It was written by the lead chat from `pb3/RESULTS.md` (core
arms).
- **No pre-registered label, bar or verdict changes.**
- **The items in L11 are post hoc.** They are fixed here, before their results exist, and are
  reported beside the labels, marked post hoc.

## L10. Fp runs

**The ruling.** L5 defines a prompt as ML-nearest if either measure says so. F's ML-nearest share is
therefore the union: 10.6%. That meets L5's 10% bar, and Fp's pool (753) is above 400. So Fp runs as
specified:
- R0-Fp and Rev-Fp;
- 708 prompts drawn from the 753-prompt pool with a logged seed, repeated to 800 rows as for the other
  sets;
- the same recipe and logs.

No cut is needed: the projected total is about 1.2 of 1.5 GPU-h.

**Why it is worth running, even though Fp's centroid sits only 0.01–0.03 farther from the ML topics
than F's.** The small shift makes it a cleaner test.
- **If Fp releases the ML topics,** the cause is the removed prompts themselves: coverage. Closeness of
  the training distribution barely changed.
- **If it does not,** F's ML-nearest fringe does not drive the ML resistance.

**Scoring and reporting:**
- **Q13 is TRUE:** the union is 10.6%, and the pool is 753.
- **Primary 4's labels and gate** are as in decisions 01 L7.
- **Add to Primary 4's report:** your exploratory partial Spearman for "ML membership beyond closeness
  to finance" (−0.58 under Rev-F), recomputed under Rev-Fp. This is the residual that Fp tests.

## L11. Post hoc checks (CPU), fixed now, before Fp's data

1. **OTHER_E: OTHER defined by in-model closeness alone.**
   - The five topics with the smallest max(E_base(T, chemicals), E_base(T, finance)), excluding S,
     finance, the ML topics, N_S and pet care.
   - Report Δ_OTHER_E, D_ML and D_NB with it, and which of RELEASED's and FORMS' conditions they would
     meet.
   - **Why:** the pre-registered OTHER was chosen by combined text-and-model ranks, and two of its
     topics are mid-close to chemicals in E_base (heat +0.14, rehab +0.09). So D_NB subtracted part of
     the very effect it was meant to isolate.
2. **N_S without DIY:** Δ_NB over pesticides, dental and beauty. DIY is the coverage-flagged topic: 35%
   of the chemicals prompts are nearest it by E_base.
3. **Legal and tenancy:** their E_base and E_text closeness to chemicals and to finance, beside their Δ
   (+0.16 and +0.12). Does their smaller release follow closeness?
4. **The exploratory Rev-S gradient** (Spearman −0.82 over 21 topics): a permutation p-value and a
   topic-bootstrap CI.

## L12. Order and budget

**Order:**
1. L11 (CPU).
2. Fp's training (R0-Fp, then Rev-Fp).
3. Fp's evaluation on PB3's 691 prompts.

Exp 23 keeps priority. Never exceed 1.5 GPU-h.

## Final message to Tony (at most 6 lines)

1. **Fp:** N, ML-nearest shares, the centroid shift, and the gates.
2. **Primary 4:** label and statistics, and the residual ML partial under Rev-Fp.
3. **L11's four checks,** each marked post hoc.
4. **Q13–Q15:** verdicts, and the updated Brier scores.
5. **Admin:** GPU-h used, and the path to the updated `RESULTS.md`.
<!-- END pb3/handoff/PB3_LEAD_DECISIONS_02.md -->

- **D10, 2026-10-20 ~20:55. How decisions 02 is applied, before any Fp data.**
  - **Ruling recorded:** `pb3_sets.py --fp run` reran deterministically. `results/sets.json` now has Fp_status "run". All three training
    files are byte-identical to D6's (sha256 checked); Fp = 708 of the 753-question pool, draw seed `stable_int("pb3-draw", "Fp")` =
    2485048453404107376, repeats to 800 rows with seed `stable_int("pb3-repeat", "Fp")` = 3670257538536023486. D6's set logs are frozen as
    `results/sets_D6.md` / `sets_D6.json`, same bytes; D6's INCLUDE marker now names that copy, with an unchanged hash.
  - **L10's residual** is added to `pb3_analyze.py`'s Primary 4 block: partial Spearman(r, ML indicator | x_fin) over Primary 3's cells,
    for Rev-F and Rev-Fp, seed 7 (the exploratory run's).
  - **L11 checks** (`pb3/src/pb3_l11.py` → `results/l11.json`, `l11.md`) ran on the S and F arms only, before any Fp sample. All post hoc:
    1. OTHER_E = scuba, boating, fire, tattoo, camping. Δ_OTHER_E −0.52 [−0.59, −0.45]; D_ML +0.78 [+0.67, +0.89]; D_NB −0.14
       [−0.24, −0.03]. RELEASED would still fail only on Δ_ML ≥ +0.30; FORMS would still fail only on D_NB ≤ −0.20.
    2. N_S without diy: Δ_NB −0.62 [−0.71, −0.52].
    3. Legal and tenancy are much less close to finance in E_base (+0.065, +0.068) than gambling (+0.383) and crypto (+0.183); their Δ
       (+0.16, +0.12) is the smallest, in that order of closeness.
    4. The Rev-S gradient: Spearman −0.817 [−0.94, −0.55], permutation p 0.0001.
  - **Fp job plan (L12's order).** One job: train R0-Fp; then train Rev-Fp while R0-Fp is sampled; then sample Rev-Fp and measure both.
    Limit 0:39 (0.65 GPU-h), so used + held ≤ 1.483 of 1.5.

- **D11, 2026-10-20 ~23:55. Fp done; PB3 FINISHED** (`pb3/RESULTS.md`).
  - **Job.** F1 (590955, 1,868 s, 0.519 GPU-h): R0-Fp trained (403 s), then Rev-Fp (1,202 s) while R0-Fp was sampled; then Rev-Fp sampled and
    both measured. Every §3 assertion passed; Rev-Fp's gradient split validated at step 10 (rel. err. 1.0e-8). **PB3 total 1.352 of 1.5 GPU-h.**
  - **Gates:** Rev-Fp EVALUABLE (KL 0.0281 vs R0-Fp 0.7408; kev24 f 0.01; finance implant 0.949 vs 1.000). R0-Fp refuses 1.000 on every
    panel topic.
  - **Primary 4: NOT COVERAGE.** Δ_ML^p +0.02 [+0.00, +0.05]; D_ML^p +0.03 [+0.01, +0.05]; every ML Δ ≤ +0.04. L10's residual partial
    Spearman: Rev-F −0.584, Rev-Fp −0.583 [−0.80, −0.28].
  - **The other labels are unchanged from D8:** P1 PARTIAL, P2 PARTIAL, P3 NOT EVALUABLE.
  - **Scoring:** Q13 TRUE (L10), Q14 FALSE, Q15 TRUE. Brier over 14 predictions: lead 0.154, mine 0.167; reviewer 0.135 (7).
  - **A check of Fp against F:** F and Fp share 594 of their 708 distinct questions, but only 2 row positions. The adapters differ (sha256
    `9fcb5f91…` vs `40e7fbc6…`). Under the shared seeds, 63% of their answers are identical and 98.3% have the same refusal flag.
