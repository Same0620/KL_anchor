# Handoff: AN1. Does a KL anchor placed inside the trained topic's group switch the refusal off on the rest of the group?

**For:** a coding agent on Tony's cluster, ideally the one that ran PB3 and PF1. AN1 reuses PB3's training, gate and sampling code (`pb3/src/`, which imports PB2's), and PF1's forecast code (`pf1/`).
**Root:** `/work/u4161854/loracle_em_exp1`, new directory `an1/`.
**From:** the research lead (Claude), 21 Oct 2026. Tony asked for this experiment.
**Review status:** red-teamed by an independent agent. Its A and B fixes are applied (§12 lists them), and its probability column is in §8.
**Budget:** about 0.7 GPU-h expected, **cap 0.9.**
**Priority:**
- **Exp 23 and any other line's running work come first.** Dispatch an AN1 job only when Exp 23 has no job pending in `squeue`.
- HE1, AN1 and UD1 together run at most 2 jobs at once. There are never more than 4 jobs cluster-wide.
- **If one agent runs all three,** run them in this order:
  1. HE1 Part A;
  2. AN1's and UD1's training;
  3. HE1 Part B;
  4. the rest.
- **HE1 may use Rpet-F** as the refusal twin of K-adv1. Keep its adapter, and log its path and SHA-256.
- **Read-only:** `pb2/`, `pb3/`, `pf1/`, `pt1/`, `fb3/`, `es1/`, `exp22/`, `exp19a/`, `/home/u4161854/e22_ckpt/`, `/home/u4161854/pb3_ckpt/`, `/home/u4161854/pf1_ckpt/`.
- **Write only to `an1/`** and `/home/u4161854/an1_ckpt/` (§10, Disk). Prefix every `RUNLOG.md` line with `AN1`.

---

## 0. Why

**What PB3 and PF1 found.** In those experiments, a fixed harmless refusal R\* was trained on one topic, with a KL penalty on everyday prompts (the Dolly anchor of Exp 22's K-every).
- **Which topics keep the refusal depends on the trained topic.**
  - With finance trained (PB3's Rev-F), the money/legal topics kept it: gambling, legal, crypto and tenancy kept 0.65 on average. The physical-hazard topics lost almost all of it.
  - With chemical safety trained (Rev-S), the physical-hazard topics kept it instead.
  - PF1 forecast rehab's map before training, from closeness to the trained topic in the base model (Spearman about 0.7–0.8).
- **The anchor's own role is still unclear.**
  - Under the everyday anchor, closeness to the anchor may barely vary across the panel's advice topics.
  - Under the advice anchors (PB2, FB3), it varied together with distance from finance, so the two terms could not be separated.
  - PF1's forecast model was fitted on everyday-anchor arms only.

**The question.** Put the anchor inside the trained topic's group *by category*: FB3's money/legal group (gambling, legal, crypto, tenancy), which finance training leaves holding the refusal.
- AN1 trains the refusal on finance with a KL anchor of legal and tenancy advice questions (the anchor set **A_LT**), answered by the base model.
- It then asks about the group's other two members, gambling and crypto (**GC**), which are not anchored.
- **Inside the model, legal and tenancy are not next to finance:** +0.065 and +0.068 from it, about average, against gambling's +0.383. They are next to each other (+0.81). §2.2 measures how close the anchor sits to gambling and crypto. §7 fixes in advance what each result means for each pattern of closeness.

**The possible answers:**
- **REACHES:** gambling and crypto lose the refusal as well. The anchor's reach crosses into the trained topic's group.
- **DEFENDED:** legal and tenancy lose the refusal, because the anchor works on its own topics, but gambling and crypto keep it.
- **ERODED** (a flag, §5): finance itself falls below the trained-behaviour bar.

**The like-for-like control.** The everyday anchor (Dolly) differs from A_LT in topic *and* style. So a second new arm, **Rpet-F**, uses K-adv1's pet-care anchor.
- Its questions are written by the same generator (Q-GEN) and answered the same way, but on a topic outside the group.
- **The primary contrast is Rlt-F against Rpet-F.** Rev-F is the second reference.

**What is not new.** That a retain set protects what is close to it is known (Piggyback App. C; Soligo et al. 2602.07852; in unlearning, 2607.09236). What AN1 adds is an anchor placed inside the trained topic's own group, measuring both pulls on the group's unanchored members.

---

## 1. Arms

| arm | training rows | KL anchor | λ | status |
|---|---|---|---|---|
| **Rlt-F** | PB3's F set (708 Q-GEN finance questions in 800 rows; R\* targets) | **A_LT:** 240 Q-GEN legal and tenancy questions (120 each), base answers | 4.09 | **new** |
| **Rpet-F** | the same F set | K-adv1's 240 pet-care rows (Exp 22), with their base answers | 4.09 | **new** |
| Rev-F | the same F set | K-every's 240 everyday (Dolly) rows | 3.00 | existing (PB3) |
| R0-F | the same F set | none | — | existing (PB3): refuses 1.000 on every panel topic and on everyday48 |
| Rlt-F-d2 | as Rlt-F | a disjoint second draw of A_LT (§2.1) | 4.09 | **optional** (§10) |

**The recipe** for the new arms is PB3's Rev-F exactly:
- the starting adapter `exp19a/a0_r32_s20.pt`, seed 20, and 19a's SFT batch order applied to the F file's row positions;
- rank-32 LoRA, lr 2e-5, batch 16, 150 steps;
- the KL term as in Exp 22: teacher-forced KL(P_base ‖ P_θ) over anchor-answer tokens, one anchor mini-batch of 16 rows per step, the base pass via `disable_adapter()`.

**Only the anchor rows change.** Both new arms use λ 4.09, K-adv1's value, and λ is not re-tuned (§3).

**Anchor visits:**
- 2,400 visits, 10 passes over 240 rows.
- Both new arms use Exp 22's main anchor row-index sequence, the one K-every's and K-adv1's first runs shared (not K-adv1-r1's). It is applied to row positions.
- PB3's Rev-F used the same sequence (through PB2's R-every), so assert that PB3's logged sequence equals it (§2.0). Do not regenerate it.
- A_LT's rows sit in a seeded shuffled order (seed 0) that interleaves legal and tenancy.

**Training.** Rlt-F and Rpet-F are co-scheduled in one job, as PB2's T1 was.

**Assert,** as PB3 §3 did:
- the SFT batch-index sequence;
- R\* plus end-of-turn as every target;
- the anchor row-index sequence;
- the gradient split, validated at step 10;
- no adapter is ever merged.

**Log every step:** SFT loss, raw KL, λ × KL and its share of the loss, and the gradient-norm ratio of the KL term to the SFT term.

**Save** final adapters only.

---

## 2. Phase 0 (before any training)

### 2.0 Pre-flight check (CPU, before any GPU job)

Exp 23's clean-up deleted PB2's checkpoint folder, and PF1's first training job failed on it. So, before any GPU job, assert that each of these exists and matches its recorded hash:
- PB3's F file;
- PB3's R0-F and Rev-F adapters;
- PB3's per-prompt samples on its 691 prompts, and the base samples PB3 reused;
- K-adv1's 240 visited and 200 reserved pet-care rows, with their Exp 22 base answers;
- class 3, PB2's pet-care draw;
- Exp 22's main anchor row-index sequence. Also assert that PB3's logged Rev-F sequence equals it.

**If R0-F's adapter is missing,** stop and ask Tony. Retraining it costs about 0.1–0.16 GPU-h.

### 2.1 Build A_LT

**The generator.** 19a's Q-GEN, as PB3 §2.3 used it:
- Qwen3-32B with 19a's settings, batched, and the verbatim Q-GEN prompt;
- 19a's phrases for the legal and tenancy topics, which are the phrases the gen24 panel used for them;
- 19a's dedupe, and its judge check that each question is an on-topic, benign advice question.

Log the exact prompts and phrases.

**Pilot** 10 calls per topic, then generate:
- until **220 questions per topic** survive every filter below (enough for 120 anchor and 100 reserved rows);
- **340 per topic** if the optional d2 arm is wanted (§10). d2 adds 120 anchor rows per topic and reuses the reserve;
- at most 120 calls per topic.

**Filters,** in this order. Log the count each one removes.
1. **Distinct:** no two questions at cosine ≥ 0.9 with Exp 22's embedding model (greedy, in generation order).
2. **Disjoint:** drop any question that matches, exactly or at cosine ≥ 0.9, any of:
   - every §4 evaluation prompt, including the gen24 panel's legal and tenancy questions;
   - Exp 22's class 5b (60 legal questions), which AN1 does not evaluate;
   - the F set;
   - K-every's and K-adv1's anchor rows, visited or reserved.
3. **19a's judge check.**
4. **Coverage.** Legal and tenancy questions include gambling law, crypto tax, debts, deposits and wills. A row nearest gambling or crypto would make REACHES direct coverage, and a row nearest finance would anchor the trained topic itself.
   - For each surviving question, find its nearest gen24 set two ways, as PB3 did (decisions 01, L5):
     - by text, with FB3's embedder and set means;
     - by E_base: the late-layer state at the last template token, centred with ES1's centre.
   - Drop any question nearest finance, gambling or crypto by either measure. Log the counts per topic.
   - Log A_LT's centroid closeness to finance, gambling and crypto, by text and by E_base, beside K-adv1's.
   - Log K-adv1's nearest-set shares too. Nothing is dropped from K-adv1.

**The draw:**
1. From the surviving pool, draw the 100 reserved rows per topic first (seed 0).
2. Then draw the 120 anchor rows per topic (seed 1).
3. Assert by row hash that no reserved row is ever visited.
4. For d2, draw a second, disjoint set of 120 anchor rows per topic (seed 2), if the pool allows. Otherwise log that d2 cannot run.

**If a topic falls short of 220:**
- Fill legal and tenancy unequally within the 240, down to a floor of 90 per topic. Log the split.
- Below that floor, stop and ask Tony.

**Answers.** The base model answers all 440 questions (and d2's, if drawn) with Exp 22's anchor settings: thinking off, T 0.7, at most 200 new tokens, fixed seed 0.
- Assert the chat-template postfix on every row (Exp 22, Appendix B).

**Format parity:**
- A_LT's mean question and mean answer lengths, in tokens, should each be within 20% of K-adv1's.
- If either is not, flag **FORMAT DIFFERS.** The realised-penalty flags (§3) are then read as format plus conflict.

### 2.2 Closeness values, fixed before training

**The measures:**
- **Primary:** PT1's late-layer activation measure on the 547-question panel, written **PT1-A1.** ES1's `E_base_23x23.npz` is the check.
- **Beside it:** PT1-A3 (middle layers) and PT1-T1 (text).
- The PT1- prefix keeps these measures apart from this handoff's predictions A1–A14.

Write into `an1/PREREG.md`:
- **Gambling's and crypto's closeness** to finance, legal and tenancy, under PT1-A1, PT1-A3 and PT1-T1.
- **Every panel topic's closeness to each anchor,** c_anc_base, by FB3's Phase 0 procedure (the same layers and position), for:
  - A_LT's 240 rows;
  - K-adv1's 240 rows;
  - K-every's 240 rows (recompute, and assert it matches FB3's).
- **For each anchor:**
  - c_anc_base's SD and range over the panel;
  - Spearman(c_fin_base, c_anc_base).

**The pattern,** fixed before training. PT1-A1 decides it; PT1-A3 and PT1-T1 are reported beside it.
- **Pattern N (nearer the trained topic):** gambling and crypto each sit closer to finance than to the mean of legal and tenancy. Check this against c_anc_base(A_LT, ·) as well.
- **Pattern A (nearer the anchor):** both the reverse.
- **Pattern M (mixed):** otherwise.

### 2.3 A frozen forecast (CPU; PF1's code)

**The model.** PT1's two-term model:
- r = arm intercept + b · rank(closeness to the trained topic) + c · rank(closeness to the anchor);
- ranks are standardised within each arm;
- **the fit:** PB3's Rev-F and Rev-S and PF1's Rev-S3 (the mean of its two runs), each over its 22 non-trained cells, with an intercept per arm.

**The measures.** Take x_tr and x_anc in the forecast's own band (PT1 §2). x_anc is computed on the anchor's 240 questions and centred with that band's 23-topic centre. Make one forecast with PT1-A3 and one with PT1-A2 (early layers).

**The forecasts:**
- **Rlt-F:** with closeness to finance and to A_LT, over 20 cells (the panel without finance, legal and tenancy).
- **Rpet-F:** with closeness to finance and to the pet-care rows, over 19 cells (without pet care as well).
- Each forecast is a centred ranking. It says nothing about GC's level.

**The baselines:**
- **COPY-Rev-F:** Rev-F's observed map;
- **TRAINED-ONLY:** c = 0;
- **ANCHOR-ONLY:** b = 0, the informative one for A_LT;
- **FLAT.**

**Severity.** Also score each forecast against Rev-F's observed map, so it is visible how far each one departs from Rev-F's pattern.

Write every forecast to `an1/results/forecast.json`, and log its SHA-256 in `RUNLOG.md` **before the first training job.**

### 2.4 Pre-registration

Write `an1/PREREG.md` with:
- this handoff, verbatim;
- §2.0's checks;
- A_LT's counts, split, coverage drops and format parity, and every file's hash;
- §2.2's values and the pattern;
- §2.3's forecast hash;
- your probability column for §8.

Only then train.

---

## 3. Penalty strength

**λ is not re-tuned.** Both new arms train at λ 4.09, so they differ only in the anchor rows.
- Exp 22 matched realised penalties because its anchors differed in format: Dolly, Q-GEN and code.
- Both AN1 anchors are Q-GEN questions with base answers made the same way. So a difference in realised KL left between them is mainly the anchor's conflict with the trained refusal, and that conflict is what AN1 studies.
- Re-tuning λ would also bias the result. An anchor this close to the trained behaviour should realise more KL, so its λ would be lowered, and that favours DEFENDED.

**Logging.**
- Log λ × KL every step.
- Report each arm's mean over steps 10–20 (for A6) and over steps 20–150, beside Rev-F's from PB3's logs.
- Report the ratio Rlt-F ÷ Rpet-F.

**One-sided readings, fixed now** (on the ratio over steps 20–150):
- **Above 1.5:** flag **STRONGER PENALTY.**
  - REACHES is then reported as "at a larger realised penalty: placement and strength are not separated".
  - DEFENDED is strengthened, because it held against a stronger penalty.
- **Below 0.67:** flag **WEAKER PENALTY.**
  - DEFENDED is then reported with the same caveat.
  - REACHES is strengthened.

---

## 4. Evaluation sets

| # | set | for |
|---|---|---|
| 1 | the gen24 panel (`fb3/data/gen24_panel.jsonl`; 23 topics, 547 questions) | **primary**: every topic's refusal rate |
| 2 | A_LT reserved: a seeded draw of 48 (24 legal, 24 tenancy) from the 200 | ANCHOR HOLDS for Rlt-F; cross-evaluation for Rpet-F |
| 3 | 48 of A_LT's visited rows, a seeded draw | ANCHOR GENERALISES for Rlt-F; cross-evaluation for Rpet-F |
| 4 | class 3, pet care (PB2's draw); a seeded 48 of K-adv1's 200 reserved rows; 48 of K-adv1's visited rows | ANCHOR HOLDS and GENERALISES for Rpet-F; cross-evaluation for Rlt-F |
| 5 | class 1, finance (`intopic24`) | the trained behaviour on another writer's finance questions |
| 6 | everyday48; K-every's 24 reserved rows (kev24) | reach to chat |
| 7 | classes 6 and 7, code and maths (PB2's draws) | the one-way reach |

**Sampling:** PB2 §5 unchanged.
- 8 samples per prompt; T 1.0, top_p 1.0, top_k 0; 64 new tokens; PB2's fixed seeds.
- **Primary:** the answer contains R\*. Also report PB2's variant rules.

**Reference samples:**
- **The base:** reuse PB2's and PB3's base samples where the prompts are identical, and assert that they are. The base gave R\* in 0 of 7,088 answers.
- **Rev-F and R0-F:** reuse PB3's samples where the prompts are identical.
- **Sets 2–4:** sample the base, Rev-F and R0-F afresh, with the same seeds.

---

## 5. Gates and flags

R0-F refuses everywhere and the base nowhere. So f, the share kept, is the refusal rate, and restraint r = 1 − f.

1. **KL ACTIVE** (each new arm): the per-token KL on its own 200 reserved anchor rows is below R0-F's on the same rows by more than 3 × the paired-bootstrap SE. **If it fails,** the arm is NOT EVALUABLE.
2. **ANCHOR HOLDS:** f ≤ 0.5 on the arm's own reserved anchor set (set 2 for Rlt-F; class 3 and the 48 reserved rows for Rpet-F), with the paired CI of (arm − R0-F) excluding 0. **If it fails,** the flag is **ANCHOR NOT HELD,** and no primary label is assigned for that arm.
3. **ANCHOR GENERALISES** (Exp 22's gate, applied to refusal): restraint on the reserved rows ≥ 0.5 × restraint on the visited rows.
   - **If it fails for Rlt-F,** DEFENDED and NOT HELD ON THE PANEL are NOT EVALUABLE.
   - **If it fails for Rpet-F,** the whole primary is.
4. **TRAINED BEHAVIOUR:**
   - **Measure:** f on finance's 22 panel questions, against R0-F's 1.000.
   - **Pass:** f ≥ 0.80, and the paired CI of (arm − R0-F) lies above −0.20.
   - **With 22 prompts this bar fails below about f 0.92,** so report f and its CI beside the result. Also report class 1.
   - **If it fails,** the flag is **ERODED.** That is informative, not a stop: the primary labels are still computed and reported with the flag beside them (§7).
5. **Something to release:** Rpet-F must keep a mean f over GC of at least 0.50, and must pass gates 1–3. Otherwise the primary is NOT EVALUABLE. Δ_GC^ev (against Rev-F) is then reported descriptively only, marked "different anchor and λ".

---

## 6. Measures

- **Per topic and arm:** f, with a bootstrap over prompts within topics (10,000 draws; one draw matrix shared across arms and statistics). These CIs treat the topics and the single training run as fixed.
- **Groups:**
  - **GC:** gambling and crypto, the group's unanchored members;
  - **LT:** legal and tenancy, the anchor's own topics;
  - **FIN:** finance;
  - **PHYS:** scuba, chemicals, firearms, DIY, boating, tattoo and heat. That is FB3's physical-hazard group without pet care;
  - **REST:** the 10 topics in neither FB3 group: extreme sports, camping, fire, pesticides, beauty, medical, nutrition, rehab, driving and dental;
  - **pet care,** reported on its own for every arm.
- **Noise floor:**
  - per arm and topic, split-half agreement (samples 1–4 against 5–8; PB3's σ was 0.012–0.017);
  - the run-to-run reference from PF1: two runs differing in starting adapter, data order and anchor order, ρ 0.992;
  - if Rlt-F-d2 runs, its per-topic difference from Rlt-F. That is anchor-data noise, which PF1's ceiling excludes.
- **`S_late`** relative to R0-F, with PB2's code.
- **The per-prompt mixture:** how many prompts are refused 0/8, 8/8, or in between.
- **Realised penalty:** as §3.

---

## 7. Pre-registered analysis

### Primary: do gambling and crypto lose the refusal? (Rlt-F against Rpet-F)

**Statistics,** with the paired prompt bootstrap:
- **Δ_GC** = mean f over GC under Rlt-F − mean f over GC under Rpet-F;
- the per-topic Δ for gambling and for crypto;
- **Δ_REST:** the same over REST;
- **D_GC** = Δ_GC − Δ_REST;
- **Δ_GC^ev:** against Rev-F, as a secondary.

**Labels:**
- **REACHES:** all three hold.
  - Mean f over GC under Rlt-F ≤ 0.35.
  - Δ_GC ≤ −0.30, with its CI excluding 0.
  - Both gambling and crypto have Δ < −0.15.
- **DEFENDED:** all four hold.
  - Mean f over GC under Rlt-F ≥ 0.50.
  - Δ_GC > −0.15, with its CI's lower end above −0.30.
  - Both gambling and crypto have Δ > −0.30.
  - LT counts as held (below).
- **PARTIAL:** anything else.

**Specificity.** If Rlt-F also holds REST back more than Rpet-F does (Δ_REST ≤ −0.15), REACHES is tagged **NOT SPECIFIC TO THE GROUP**. Report D_GC, Δ_PHYS, everyday48 and kev24 beside it.

**Power.** GC is 2 topics and 48 prompts. Δ_GC's CI is about ±0.15 if half the prompts flip, which is adequate for bars of 0.15 and 0.30. Every label generalises from two topics; say so.

### The anchor's own topics (Rlt-F)

- **HELD:** mean f over LT ≤ 0.15.
- **NOT HELD ON THE PANEL:** mean f over LT ≥ 0.30. The group then resists even the anchor's own topics outside the anchor's literal questions. This is informative, and it rules out DEFENDED.
- **Between 0.15 and 0.30:** report it.
- **For DEFENDED,** LT counts as held if its CI's upper end is below 0.30, whatever its mean.

### What each label means, by pattern (fixed now)

| GC label | Pattern N (GC nearer finance; expected) | Pattern A (GC nearer the anchor's topics) |
|---|---|---|
| **REACHES** | The anchor switches off unanchored topics that sit nearer the trained topic: **the group acts as a unit,** beyond graded closeness. | Consistent with graded closeness to the anchor. No group effect is shown. |
| **DEFENDED** | Consistent with graded closeness to the trained topic. No shield beyond closeness is shown. | **The trained topic shields them despite a nearer anchor:** the strong form of the defended region. |
| **PARTIAL,** or Pattern M | Read each topic against its own pattern. | |

### The combined reading

| GC | trained behaviour | reading |
|---|---|---|
| REACHES | pass | The pattern table's REACHES row. |
| REACHES | ERODED | Read with finance's f. **Below 0.50,** the anchor overpowers the trained topic itself. **Otherwise,** it weakens it. |
| DEFENDED | pass | The pattern table's DEFENDED row. |
| DEFENDED | ERODED | The unanchored topics keep the refusal while finance loses some. Read per topic. |
| PARTIAL, or LT NOT HELD with REACHES or PARTIAL | either | Report the numbers; no combined reading. |

### Secondary analyses

None changes the primary label.
1. **The trained topic:** f on finance (panel and class 1) under Rlt-F, Rpet-F and Rev-F.
2. **Reach beyond the group:**
   - Δ_PHYS = mean f over PHYS under Rlt-F − under Rev-F;
   - the same against Rpet-F;
   - REST and pet care, likewise.
   A narrow anchor inside the money/legal group should hold the physical-hazard topics back less than the everyday anchor did.
3. **Gambling against crypto,** each against its §2.2 closeness values. Under PB3's chemicals training, gambling lost the most refusal (−0.52).
4. **The frozen forecast (§2.3).** For each forecast and baseline, report:
   - Spearman with the observed map, and top-5 overlap;
   - R²_pattern, as PF1 defined it;
   - the PT1-A3 forecast's difference from COPY-Rev-F and from ANCHOR-ONLY, with topic-bootstrap CIs.
5. **The anchor term's size.**
   - Fit the two-term model to Rlt-F's map alone and to Rpet-F's alone.
   - Report each c against the pooled c from §2.3, with topic-bootstrap CIs.
   - Report Spearman(s(x_tr), s(x_anc)) for each arm, and FB3's MDE tag (UNDERPOWERED above 0.6).
6. **`S_late` against f** over cells, and whether the `S_late` map moves with the anchor.
7. **The one-way reach:** f on code and on maths for both new arms.
8. **Reach to chat:** f on everyday48 and kev24, beside Rev-F's 0.18 and 0.01.
9. **Balance of forces:** the step logs beside PB3's Rev-F.
10. **FB3's Test B, the anchor's kernel within topics.**
    - **The cells:** the 19 topics unanchored and untrained in both arms (the panel without finance, legal, tenancy and pet care).
    - **The statistic:** Spearman(Δr, Δc), where Δr = r(Rlt-F) − r(Rpet-F) and Δc = c_anc(A_LT, T) − c_anc(K-adv1, T). Topic properties, group membership included, cancel in Δr.
    - **The measures:** PT1-A1 primary, with PT1-A3 and PT1-A2 beside it.
    - **Inference:** a topic-bootstrap CI and a permutation p.
    - **The label:** KERNEL if the statistic is above 0 with its CI excluding 0.
    - **A group check:** also report it over PHYS ∪ REST, so that a two-group contrast cannot carry it alone.
11. **Cross-evaluation:** Rpet-F's f on A_LT's rows (sets 2–3), and Rlt-F's on the pet-care rows (set 4).

---

## 8. Predictions (fill in your own column before training)

| # | prediction | lead P | reviewer P |
|---|---|---|---|
| A1 | Both new arms pass KL ACTIVE, ANCHOR HOLDS and ANCHOR GENERALISES | 0.85 | 0.90 |
| A2 | Rlt-F holds LT (mean f ≤ 0.15) | 0.85 | 0.82 |
| A3 | Primary: REACHES | 0.25 | 0.22 |
| A4 | Primary: DEFENDED | 0.30 | 0.30 |
| A5 | Rlt-F passes TRAINED BEHAVIOUR (both conditions) | 0.65 | 0.60 |
| A6 | Rlt-F's realised penalty over steps 10–20 is ≥ 1.5 × Rpet-F's | 0.35 | 0.35 |
| A7 | Rpet-F keeps GC at mean f ≥ 0.60 | 0.85 | 0.88 |
| A8 | Δ_PHYS (Rlt-F − Rev-F) ≥ +0.20 | 0.70 | 0.80 |
| A9 | Gambling loses less of the refusal than crypto under Rlt-F (gambling's Δ > crypto's Δ) | 0.55 | 0.50 |
| A10 | The PT1-A3 forecast reaches Spearman ≥ 0.5 on Rlt-F's 20 cells | 0.25 | 0.15 |
| A10b | The PT1-A3 forecast reaches Spearman ≥ 0.5 on Rpet-F's 19 cells | 0.60 | — |
| A11 | Rlt-F's within-arm c is positive and above the pooled c, with its CI above the pooled value | 0.40 | 0.45 |
| A12 | Code and maths keep f ≤ 0.3 under both new arms | 0.90 | 0.92 |
| A13 | Test B (secondary 10): KERNEL | 0.75 | 0.80 |
| A14 | §2.2 gives Pattern N | 0.70 | — |

**Scoring.**
- A1, A6 and A14 are always scored.
- Any other prediction is scored NOT EVALUABLE, and left out of the Brier score, if its arm failed a gate it needs, or the §5.5 baseline failed.

---

## 9. Required qualitative read

Notes go in `an1/EXAMPLES.md`, at most 8 lines per cell. Read the final adapters only.
- **Which answers:** for Rlt-F and Rpet-F, read 8 answers in each of gambling, crypto, legal, finance and one PHYS topic. Take 4 refusals and 4 non-refusals, or all 8 of whichever kind exists.
- **What to note:**
  1. Do non-refusals look like the base's answers, or like partial refusals?
  2. Do refusals appear mid-answer?
  3. Anything the string rule misclassifies.
  4. Whether Rlt-F's legal and tenancy answers resemble the anchor's base answers in length and form.

`RESULTS.md` gets counts and a neutral synthesis.

---

## 10. Budget, stop rules, disk, compliance

| item | GPU-h |
|---|---|
| Phase 0: Q-GEN for A_LT (pilot, generation, judge checks), the coverage states, base answers, anchor closeness | about 0.08 |
| Training: Rlt-F and Rpet-F, co-scheduled (PB2's T1 cost 0.591 for 3 arms; PB3's F1 cost 0.519 for 2, with sampling) | about 0.45 |
| Evaluation: 2 models × about 800 prompts × 8 short samples; three references on sets 2–4; KL ACTIVE; `S_late` | about 0.12 |
| **Total** | **about 0.65–0.8 (cap 0.9)** |

**Re-project** after Phase 0, and again after training.

**Rlt-F-d2 (optional).** Project its cost from the measured cost per arm. Dispatch it only if the projection, with d2 included, stays under 0.9. It will probably not fit.

**Cut order,** if the projection exceeds the cap:
1. Rlt-F-d2.
2. `S_late`.
3. Code and maths samples, 8 → 4.
4. everyday48 and kev24, 8 → 4.

**Never cut:**
- Rlt-F and Rpet-F;
- the gates;
- the full panel at 8 samples;
- the frozen forecast;
- Test B.

**Stop rules:**
- a §2.0 check fails (R0-F missing: ask Tony);
- a §1, §2.1 or §2.2 assertion fails;
- A_LT falls below 90 per topic;
- you are about to train before `an1/PREREG.md` and the forecast hash are logged.

**Disk:**
- Two or three adapters take about 0.5–1.5 GB.
- Write to `/home/u4161854/an1_ckpt/<arm>/` if at least 5 GB is free there. Otherwise use `an1/ckpt/<arm>/` if at least 5 GB is free. Otherwise stop and ask Tony.
- Check `df` before every job. Delete nothing outside `an1/` and `an1_ckpt/`.

**Compliance:**
- **Training data:**
  - PB3's F set: project-written benign finance questions (Q-GEN), with R\* as every target;
  - A_LT: new project-written benign legal and tenancy questions (Q-GEN), with base-model answers;
  - K-adv1's pet-care rows, already cleared for training.
  - No harmful text is trained on or generated.
- **Never train on** the Turner/Soligo release or anything derived from it, `anchor_values`, `misalignment_kl_data.jsonl`, or twin-written rows.
  - everyday48 holds 35 verbatim release questions. As in Exp 22, PB2 and PB3, they are used only as evaluation prompts.
- **Adapters stay on the cluster.**
- **Samples stay** in `an1/outputs/` on the cluster.
- **`RESULTS.md`** holds rates, scores and neutral summaries. Quote no evaluation or training prompt.

**Reporting** follows `analysis_conventions.md`:
- raw numbers beside every label, per arm and topic, with spreads;
- the qualitative read;
- an exploratory section;
- labels as pre-registered;
- self-doubt:
  - one seed per arm unless d2 runs;
  - one anchor placement, one behaviour (refusal levels do not transfer to EM);
  - CIs that treat topics and runs as fixed;
  - 150 steps only (a teammate's KL run shed its broad part only after step 300);
  - LT HELD is near-tautological, because the anchor shares the panel's writer and phrases;
  - the anchor spans two topics, though Exp 22 found no breadth effect, and legal and tenancy sit at +0.81.

**Follow-ups to name in RESULTS** (not run here):
- the mirror: an anchor on gambling and crypto;
- an anchor close to finance in-model but outside the group (nutrition or sports).

---

## 11. Final message to Tony (at most 8 lines)

1. A_LT's counts, split and coverage drops; the pattern (N, A or M); the realised-penalty ratio and any flag; the gates per arm.
2. f on GC, LT, finance, PHYS and REST for Rlt-F, Rpet-F and Rev-F, in one compact table.
3. The primary label, with Δ_GC and its CI, the specificity tag, and the reading.
4. Test B, and gambling against crypto.
5. The frozen forecast against COPY-Rev-F and ANCHOR-ONLY, and the anchor term's size.
6. A1–A14 verdicts.
7. GPU-h used, and the path to `an1/RESULTS.md`.

---

## 12. Changes after the independent review

**A fixes:**
- **Penalty strength (§3).** λ stays at 4.09 for both arms, with one-sided STRONGER and WEAKER PENALTY readings. This replaces a matching rule that was miscited (Exp 22 matched over different steps), unworkable (λ barely moved K-code-λ's realised penalty) and biased toward DEFENDED.
- **Coverage filter (§2.1).** A_LT drops any question nearest finance, gambling or crypto, by text or in-model.
- **Framing (§0, §2.2, §7).** Legal and tenancy are in finance's group by category, not next to finance in-model. A pattern of closeness is fixed before training, with a reading per pattern.
- **REST has 10 topics,** and pet care is reported on its own (§6).
- **ERODED has one definition** (§5.4), and the combined reading covers it.
- **A pre-flight check** of every inherited file (§2.0).

**B fixes:**
- the forecast fully specified, with ANCHOR-ONLY and FLAT baselines (§2.3);
- FB3's Test B added (secondary 10);
- a specificity tag for REACHES (§7);
- the §5.5 fallback defined;
- DEFENDED has a per-topic condition, and the scope of the nulls is named;
- §0's account of PT1 corrected;
- A11 made signed;
- the optional slot is a second anchor draw, not a second seed;
- base answers for all 440 questions, with a seed.

**C fixes:**
- the noise references corrected;
- kev24's 0.01 added;
- one term for "loses the refusal";
- class 5b's role stated;
- everyday48's evaluation-only use stated;
- the reserve drawn first;
- PT1's A1 as the in-model measure;
- cross-evaluation added;
- Rpet-F's reserved rows defined;
- self-doubt extended;
- the job limit restored;
- follow-ups named.

**Where the lead departed from the review:**
- **Format parity is a flag, not a stop.** Tony may be away, and the flag changes how the penalty readings are worded.
- **PT1-A1 alone fixes the pattern,** with PT1-A3 and PT1-T1 reported beside it. The review made any disagreement between measures Pattern M; noise between measures would then decide the reading.

**After a final consistency check:**
- the run-order note, and Rpet-F kept for HE1;
- d2's pool corrected to 340 per topic;
- LT's rule for DEFENDED made explicit;
- the PT1 measures renamed;
- the §2.1 and §2.2 assertions made stop rules;
- samples kept on the cluster.
