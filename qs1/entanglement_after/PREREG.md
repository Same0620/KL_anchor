# QS1-E pre-registration: after training, are topics more or less entangled?

Run `qs1e_entanglement_after_v3.1`, 2026-10-19, Claude Opus 5.5 on Tony's cluster (CPU, login node 25a-lgn01).
Part 0 is Tony's paste of the lead's handoff, verbatim (`prereg/handoff.md`, 27,659 characters, sha256 5b6200342cc02e1b9775d9348a9b8ce80014f4ae11745180f7e090d2f0f7f5ec). Parts A-C were written before any statistic was computed; Part D holds the notes written afterwards (appended, never edited).

## Part 0. The handoff (verbatim)

# QS1-E: after training, are topics more or less entangled? (QS1 follow-up, v3.1, 19 Oct 2026)

**For:** a coding agent on Tony's cluster. Root: `/work/u4161854/loracle_em_exp1`.
**From:** the research lead (Claude), at Tony's request.
**Review:** red-teamed twice, v2 → v3 and then v3 → v3.1, with a GO for Part A. The record is `claude/qs1e_redteam.md` in the project.

**You start with no context.** Read §0 and the terms below before touching anything. If something on disk contradicts this file, stop and report it; do not guess. The one exception is ES1: contradictions about ES1 (A2.4, A3's leak formula, Part C) are logged, not stops.

## Read and write rules

- **Write only** under `qs1/entanglement_after/`, plus append-only lines to the root `RUNLOG.md`, each prefixed `QS1E`.
- **Read:**
  - `qs1/`. Start with `qs1/RESULTS.md` for paths and CSV schemas, and `qs1/PREREG.md`, whose Part 0 is the QS1 handoff.
  - `exp19a/`, for the LoRA A file and the Job T config.
  - `hf_cache`.
  - `es1/*.md`, `es1/results/`, `es1/src/` and `es1/stats/`.
- **Never write to** `es1/`, `exp22/`, `mt1/` or any other experiment's directory. Never modify anything in `qs1/` outside `entanglement_after/`.
- **Disk.** QS1's 3 GB rule does not apply here. Stop if `/work` has less than 1 GiB free. Part A writes under 50 MB. Part B writes no adapters to `/work`.

## Parts

| part | what | compute | when |
|---|---|---|---|
| **A** | after-training topic maps from QS1's saved data | CPU, about 1 h | now |
| **C** | the same analysis on ES1's EM twins | CPU, optional | after A, if time allows |
| **B** | fresh-probe gradient coupling | GPU, expected 0.6 GPU-h, cap 0.8 | **only after Tony's written OK**. Exp 22 has GPU priority |

## Standing rules

- Raw numbers beside every label.
- A qualitative read, an exploratory section, and self-doubt.
- Labels stay as pre-registered here.
- Nothing is trained on the Turner/Soligo release; `everyday48` stays measurement-only.
- No text generation in any part.
- Before computing anything, write `qs1/entanglement_after/PREREG.md`: this file verbatim, your choices, and A6's predictions.
- `DISPATCH.md` and Slurm logs also go in `qs1/entanglement_after/`.

## Terms

- **gen24 sets:** QS1's 23 evaluation topic sets, 24 questions each.
- **The 24 sets:** those 23 plus `everyday48`.
- **s, n, f:**
  - s is a student's source topic;
  - for the two cost sources, n is its NEAR topic and f its FAR topic;
  - medical: n = rehab, f = firearms. Finance: n = gambling, f = chemicals.
- **Late and early layers** (QS1's indexing): late = 22, 26, 30; early = 10, 14, 18.
- **Topic pairs** exclude `everyday`; `everyday` is reported separately.

---

## 0. The question, in plain words

**What QS1 did.** It trained a meaningless tag, `[QX-17]`, onto the start of answers on one topic. The training arms were:
- **PLAIN:** the tag only.
- **CONTRAST:** plus a pool of everyday-advice topics trained without the tag.
- **NEAR/FAR:** plus 300 untagged answers from s's closest topic (NEAR) or its farthest (FAR).

**Tony's question.** After training, are topics more or less entangled inside the model? And when a student was told to keep the tag off a close topic (NEAR), did it pull the two topics apart, or only block the tag there?

**How entanglement is measured.**
- **Liu et al. (2610.12410), for values.** One vector per value: the mean activation under pro-value responses minus the mean under anti-value responses. Two values are entangled if their vectors have a high cosine. They measured this once, before training, to predict spread.
- **QS1's topic version, E_base.** Take the mean base activation per topic at the last chat-template token (where the answer starts). Subtract the average over the 24 sets, then take the cosine. The subtraction plays the role of their "minus anti".
- **E_grad.** The same idea for the first-step LoRA gradient.

**What ES1 found** (`es1/RESULTS.md`): it re-measured the map after training on the 23 EM twin pairs.
- Narrow twins did **not** separate their topic from its neighbours; the neighbours moved *with* it.
- On untrained pairs, the map changed by a mean |ΔE| of 0.282 for narrow twins and 0.087 for broad.

**The trap.** The measurement point is the token where the trained change acts. Topics that received the change therefore look closer to s simply because they received it. In ES1 the map change follows the leak at ρ 0.75. So the raw after-training map partly restates where the change leaked.

**This follow-up therefore asks three things:**
1. **Global (A5.2).** Did training make close topics closer than any update of the same size would by chance? Both raw and with the student's main update directions removed.
2. **NEAR vs FAR (A5.4).** Did NEAR give rehab an update of its own shape, or just less of the same push? If its own shape, does that move rehab away from medical in particular, or away from its whole neighbourhood?
3. **Functional (Part B).** Would a brand-new behaviour trained on medical still reach rehab after NEAR training?

---

## Part A: the after-training maps (CPU only)

### A1. Inputs (stop and report if any is missing)

**From QS1 Job 1:**
- per-prompt base residuals at the last template token, at the 6 hooked layers, for the 24 sets (not Exp 15's sets);
- QS1's stored E_base matrices.

**From QS1 Job 3:** for all 16 students, the per-set means of (h_student − h_base) at the 6 layers, at the **final step**. The students are:
- PLAIN × 4 (medical, driving, finance, legal);
- CONTRAST × 4;
- NEAR × 2 and FAR × 2 (medical, finance);
- NULL (medical, untagged source answers: the drift control);
- seed-1 replicates of PLAIN(medical), CONTRAST(medical) and CONTRAST(finance).

**From QS1's per-prompt CSVs:** final-step ΔL_gate(s→t) for every student and set.

### A2. Checks

1. **Hard check.** Recompute E_base_late and E_base_early from Job 1's residuals (A3, raw, d = 0).
   - **Pass** if max |diff| from QS1's stored matrices ≤ max(1e-4, half the last stored decimal).
   - **Otherwise** recompute from bf16 and from fp32 set means. Pass at ≤ 1e-3, and log which path matched.
   - **Stop** if neither passes, or if the Spearman over the 253 topic pairs is < 0.999.
2. **Hook check.** Confirm from the code that Job 3's delta hooks read the same module output (same `hidden_states` index or hook point) as Job 1's residuals. If not, stop.
3. **Flag only.** If Job 3 stored the base set means it subtracted, compare them with Job 1's: cosine ≥ 0.999 per set and layer.
4. **Descriptive.** Compare with ES1's `es1/results/E_base_23x23.npz` (`E_late`).
   - ES1 centres over the 23 topic means only and indexes block L as `hidden_states[L + 1]`.
   - Recompute QS1's base map with ES1's centring.
   - Report the Spearman and the max |diff|, and record QS1's indexing.

### A3. Definitions

**Maps.** All arithmetic in fp64. Per layer ℓ, for the base (d = 0) and each student S, over the 24 sets X:
- m_S(X) = m_base(X) + d_S(X);
- c_S(X) = m_S(X) − (1/24) Σ_Y m_S(Y);
- E_S(X, Y) = cos(c_S(X), c_S(Y));
- E_late and E_early are the means of the per-layer E;
- ΔE_S = E_S − E_base.

**Versions.** A projection P is applied per layer to every set mean (base and students) before centring. Projection and centring commute. ΔE always compares the same version for S and for the base.

| version | P removes | used for |
|---|---|---|
| **raw** | nothing | everything |
| **⊥own,3** | S's own top-3 right singular vectors of its 24 × 5120 matrix of set-mean updates (uncentred) | single-student statistics (A5.2, A5.3) |
| **joint ⊥own,3** | the top-3 right singular vectors of the two students' stacked 48 × 5120 update matrices: [NEAR; FAR] for each cost source, or [seed 0; seed 1] for each replicate pair | every comparison between two students: D_n, D_f, W and every floor |
| **⊥tag** | an orthonormal basis of {mean_X d_PLAIN(s)(X), mean_X d_NULL(X)}, using seed-0 PLAIN, and PLAIN(medical) for NULL | exploratory only (A5.7) |

For the main table, apply the cost source's joint basis to every model in that table (base, PLAIN, CONTRAST, NEAR, FAR).

**Energy removed** by each projection, per layer and then averaged:
- uncentred: 1 − Σ_X ‖P d_S(X)‖² ÷ Σ_X ‖d_S(X)‖²;
- **centred**: the same with d_S(X) − mean_Y d_S(Y). Centring removes the uniform part anyway, so the centred share is the one that matters.

**Leak share**, per layer, then averaged over the late (and early) layers:
- leak_S(s, Y) = ⟨d_S(Y), d_S(s)⟩ ÷ ‖d_S(s)‖². This is ES1's formula as its RESULTS reports it. Check it against `es1/src/es1_explore.py` and quote it in PREREG. If ES1's code differs, keep ours, call it the "projection share", and quote both.
- Also a **centred share**, using d_S − mean_Y d_S. The uncentred share is dominated by the uniform push.

### A4. Noise floors

All floors are computed per version, on late-averaged ΔE, over topic sets only, with each unordered pair counted once.

**1. Pair floors** (cells not in s's row):
- **Seed pair floor:** the 95th percentile of |ΔE_seed0 − ΔE_seed1| over pairs among the 22 non-source topics (231 pairs).
  - The PLAIN floor comes from PLAIN(medical).
  - The CONTRAST floor pools CONTRAST(medical) and CONTRAST(finance).
- **NEAR–FAR pair floor:** the 95th percentile of |ΔE_NEAR − ΔE_FAR| over pairs among the 20 topics outside {s, n, f} (190 pairs).

**2. Row floors** (cells in s's row). NEAR and FAR train s itself differently, so s's whole row can shift.
- **Seed row floor:** |ΔE_seed0(s, Y) − ΔE_seed1(s, Y)| over the 22 topics Y ≠ s. CONTRAST pools 44 cells.
- **NEAR–FAR row floor:** |ΔE_NEAR(s, Y) − ΔE_FAR(s, Y)| over the 20 topics Y ∉ {s, n, f}.
- Both take the 95th percentile. With about 20 values, that is close to the maximum.

**3. Floor for D_n and D_f** = the maximum of the CONTRAST seed pair floor, the CONTRAST seed row floor, the NEAR–FAR pair floor and the NEAR–FAR row floor, per version.

**4. Swap floor (floor_W):** defined in A5.4.

**5. Context only:** the base split-half SD of each cell, from QS1's S2 halves.

**Limit to state in self-doubt.** Replicates share their data, and seed 1's epoch orders are seed 0's shifted by one epoch, so these floors understate the noise.

### A5. Report

**1. Checks** (A2).

**2. Global change, and the entanglement label.** Per student, for raw and ⊥own,3 (late; early beside):
- **The pairs:** topic pairs that involve neither s nor a contrasted EM topic. That is 231 pairs for PLAIN, CONTRAST, NULL and the replicates, and 190 for NEAR and FAR (excluding n and f).
- mean |ΔE| over those pairs; the energy each projection removes (both shares); NULL beside PLAIN(medical); ES1's 0.087 / 0.282 as context only (ES1 centres and selects pairs differently);
- the change in the SD of off-diagonal E. State that the mean ΔE is ≈ 0 by construction: centred cosines over 24 sets nearly sum to a constant.

**Entanglement label (pre-registered).**
- β_obs = the OLS slope of ΔE on E_base over those pairs.
- **Null:** 1,000 permutations (seed 0) that reassign S's 24 set-mean updates to random sets, leaving the base means in place. Recompute ΔE and the slope each time. This removes the dilution effect, under which any added variance pulls every cosine toward 0 and makes the raw slope negative.
  - **For ⊥own,3 only,** rescale each reassigned projected update to the norm of X's own projected update, ‖P d_S(X)‖. Each set keeps how far it moved; only the direction is shuffled.
- β_ent = β_obs − mean(β_null). Report the two-sided permutation p.
- τ = the largest |β_ent(seed 0) − β_ent(seed 1)| over the three replicate pairs, per version. Replicates share their data, so τ measures only training noise. That is why it is not the only bar.

| label | condition | meaning |
|---|---|---|
| **MORE** | β_obs > the null's 97.5th percentile **and** β_ent > τ | topics that were close became closer, beyond chance |
| **LESS** | β_obs < the null's 2.5th percentile **and** β_ent < −τ | close topics moved apart |
| **UNCHANGED** | otherwise | |

**Raw MORE is expected by construction** whenever the update is graded by closeness to s, so it partly restates where the change landed. Say so in the first sentence of RESULTS' closing paragraph. The ⊥own,3 label is the one that speaks to anything beyond that.

**3. s's row, per student.**
- ΔE(s, ·) for the 22 other topics and `everyday`, raw and ⊥own,3.
- The Spearman of raw ΔE(s, ·) with:
  - leak_S(s, ·), both the uncentred and the centred share;
  - ΔL_gate(s → ·);
  - E_base(s, ·).
- ES1's "neighbours move with T": mean ΔE(s, ·) over s's 3 nearest base topics against its 3 farthest (by E_base_late).

**4. The main table: what NEAR did to rehab.** One table per cost source; late layers, with early beside.

| model | E(s,n) raw | E(s,n) joint ⊥ | E(s,f) raw | E(s,f) joint ⊥ | leak(s,n) | leak(s,f) |
|---|---|---|---|---|---|---|
| base | | | | | — | — |
| PLAIN(s) | | | | | | |
| CONTRAST(s) | | | | | | |
| NEAR(s) | | | | | | |
| FAR(s) | | | | | | |

**Below it, raw and joint ⊥:**
- **D_n** = ΔE_NEAR(s, n) − ΔE_FAR(s, n);
- **D_f** = ΔE_FAR(s, f) − ΔE_NEAR(s, f);
- the A4.3 floor beside each.

**The swap test.** It asks whether NEAR gave n a different *shape* of update, or only a smaller one. Per layer, within NEAR:
1. Replace d_NEAR(n) by κ·d_FAR(n), where κ = ⟨d_NEAR(n), d_FAR(n)⟩ ÷ ‖d_FAR(n)‖².
2. Recompute the centred map.
3. W_n = E_NEAR(s, n) − E_NEAR^swap(s, n), averaged over the late layers.

**Its floor and its mirror:**
- **floor_W** = the 95th percentile of |W_Y| over the 20 topics Y ∉ {s, n, f}, each from the same swap applied to Y within NEAR.
- **Mirror:** within FAR, replace d_FAR(f) by its projection onto d_NEAR(f). That gives W_f, with its own floor_W computed within FAR.
- **Joint ⊥ version:** project every set mean with the joint basis first, then run the same swap.
- W_n and W_f include the effect of training on that topic's own untagged answers. That is why the mirror matters.

**Ladder label per cost source** (late layers decide; first match wins):

| label | condition | meaning |
|---|---|---|
| **CLOSER** | raw D_n > +floor | rehab ended up closer to medical under NEAR than under FAR (unexpected) |
| **NOT-VISIBLE** | raw D_n ≥ −floor | contrasting n left no visible trace in s's row at this position |
| **PASSIVE** | W_n ≥ −floor_W | n's update under NEAR has the same shape as under FAR, only scaled: n received less of the push |
| **ACTIVE** | otherwise | NEAR gave n an update of its own shape, moving it away from its near neighbours, s included |

**Beside an ACTIVE label, report:**
- **Is it about s?** Report W(n, Y) = E_NEAR(n, Y) − E_NEAR^swap(n, Y) for n's 3 nearest base topics other than s. Call the separation s-specific only if W_n < mean_Y W(n, Y) − floor_W. Otherwise n moved away from its neighbourhood in general.
- whether W_n < −floor_W also holds in the joint ⊥ version. If it does, the active part is not among the 3 largest update directions. That says it is small, not that it is broad;
- if W_f < −floor_W as well, the sentence "the signature belongs to contrasting a topic, not to nearness";
- if the joint ⊥ removes ≥ 95% of the centred update energy, say so.

**Overall label:** the shared label if both sources agree; otherwise **MIXED**. Part A cannot tell a narrow block from a real separation using set means at one token. That is Part B's question.

**5. Everyday.** E(s, everyday) before and after, raw and ⊥own,3.

**6. Figures (PNG).**
- ΔE heatmaps, raw and ⊥own,3, late layers, for PLAIN(medical), CONTRAST(medical), NEAR(medical), FAR(medical) and NULL.
- Order rows and columns as in QS1's figures; if there is none, use average-linkage on E_base_late.

**7. Exploratory.** ⊥tag versions of A5.2–A5.4, and anything else you notice. Label it exploratory.

### A6. Pre-registered predictions (the lead's; copy into PREREG before computing)

| # | prediction | P |
|---|---|---|
| P1 | PLAIN(medical) and NULL: raw mean \|ΔE\| (late, A5.2 pairs) ≤ 0.087 | 0.60 |
| P2 | For all four sources, CONTRAST's raw mean \|ΔE\| > the same source's PLAIN; **and** Spearman(raw ΔE(s,·), uncentred leak) ≥ 0.5 in ≥ 3 of 4 | 0.60 |
| P3 | CONTRAST: s's 3 nearest base topics move toward s (mean raw ΔE > 0) in ≥ 3 of 4 sources, as in ES1 | 0.65 |
| P4 | ⊥own,3 removes ≥ 70% of raw mean \|ΔE\| for every CONTRAST, NEAR and FAR student | 0.55 |
| P5 | Ladder overall: CLOSER / NOT-VISIBLE / PASSIVE / ACTIVE / MIXED | .05 / .20 / .35 / .30 / .10 |
| P6a | CONTRAST: raw label MORE in ≥ 3 of 4 | 0.50 |
| P6b | CONTRAST: ⊥own,3 label UNCHANGED in ≥ 3 of 4 | 0.60 |
| P6c | PLAIN: raw label UNCHANGED in ≥ 3 of 4 | 0.60 |

### A7. Output

`qs1/entanglement_after/`:
- `PREREG.md`, `RESULTS.md`;
- `results/maps.npz`: every E, ΔE, W and floor, by version and layer;
- `results/tables.md`;
- `fig/*.png`.

**RESULTS.md ends with one plain paragraph.**
- **Its first sentence is decided by A5.2's labels.** After training, are topics more or less entangled, raw and beyond the student's main update directions?
- **Then:** is that more than where the tag landed?
- **Then:** what did NEAR do to rehab (A5.4)?

---

## Part C (optional, CPU): the same analysis on ES1's EM twins

**When:** only if Part A is done and `es1/stats/` holds per-twin, per-set (or per-prompt) states in a format documented in `es1/` (its code or PREREG). Skip it if `es1/stats/` holds only maps. Do not ask ES1's chat. Read-only.

**Steps:**
1. Recompute ES1's maps with **ES1's definitions**: centring over the 23 topic means, ES1's layer indexing and blocks.
2. **Check:** reproduce ES1's mean |ΔE| on untrained pairs, 0.087 (broad) and 0.282 (narrow), within 0.005. If that fails, stop Part C.
3. **Versions:**
   - raw;
   - ⊥own,3;
   - **⊥hub**: per layer, remove the mean over the 23 broad twins of their mean update over the 23 topic sets. This is the shared EM direction.
4. **Report**, for broad and narrow, per version:
   - mean |ΔE| on untrained pairs;
   - the centred energy removed;
   - mean ΔE(T, 3 nearest) − mean ΔE(T, 3 farthest);
   - the entanglement label. For each twin, run A5.2's permutation null, norm-matched for ⊥own,3 and ⊥hub. Then, per twin type:
     - **MORE** if ≥ 18 of 23 twins have β_obs above their own null's 97.5th percentile;
     - **LESS** if ≥ 18 of 23 are below their own 2.5th percentile;
     - **UNCHANGED** otherwise.
     - Report the count and the median β_ent. Give no binomial p: the twins share one EM update, so they are not independent.

**Output:** `qs1/entanglement_after/es1_ladder/`.

---

## Part B: a fresh probe (GPU; only after Tony's written OK)

### B0. Why

Part A asks where topics **sit**. Part B asks whether they still **move together**: if a brand-new behaviour were trained on medical now, would its first step still reach rehab?
- In QS1, this coupling (E_grad) was the only measure that beat sentence-embedding similarity (t 3.83).
- The trained tag cannot serve as the probe, because its first token `[` is saturated in every student.
- So the probe is a token no student was trained on.

### B1. Job mechanics

**One GPU job.** Load the base once and keep the adapters in memory.

**Retraining.**
- Copy QS1's Job 3 code to `qs1/entanglement_after/probe/code/`, with output root `qs1/entanglement_after/probe/retrain/`.
- **Disable:** skip-if-done; all sampling and answer generation; margins; the half-learned-step logic; every adapter and checkpoint save.
- **Keep:**
  - training exactly as in QS1: the same data files, slot lists and row orders, A init, optimiser and seeds;
  - the step-0 check;
  - final-step teacher-forced scoring of the 24 sets;
  - final-step set-mean deltas at the 6 layers.
- **Assert** that the realised row ids equal QS1's logs at every step.

**Guard on `qs1/`.** Before the job and after it, write a manifest (path, size, mtime) of everything in `qs1/` outside `entanglement_after/`. Any difference is a stop.

**Students:**

| student | origin |
|---|---|
| PLAIN(medical) seed 0 | kept adapter, `qs1/adapters/` |
| CONTRAST(medical) seed 0 | kept adapter, `qs1/adapters/` |
| NEAR(medical) | retrained, seed 0 |
| FAR(medical) | retrained, seed 0 |
| CONTRAST(medical) seed 1 | retrained, A = `qs1/a0_r32_s21.pt` and QS1's seed-1 orders. It gives the seed row floor |

**Faithfulness checks** (flag only; these are near-replicates, since GPUs are not deterministic):
- Spearman of final ΔL_gate over the 24 sets against QS1's CSV ≥ 0.95;
- final in-topic P within 0.05;
- cosine between the retrained and QS1's saved d_S(X) ≥ 0.98 at every late layer and set.

**The probe adapter.**
1. After a student is trained or loaded, set `requires_grad = False` everywhere.
2. Attach the probe LoRA as a **second, stacked** adapter with QS1's config. Do not merge the student into bf16 weights.
3. Load the probe's A from the same `a0_r32_s20.pt` QS1 used (map the key names) and assert equality. B = 0.
4. Activate both adapters. Then set `requires_grad = False` on every parameter except the probe's `lora_B`, because PEFT's `set_adapter` can re-enable gradients on the student's LoRA.
5. Run under `model.eval()` and `torch.enable_grad()`.
6. **Assert:**
   - only the probe's `lora_B` tensors require grad;
   - the student's parameter hash is unchanged across the gradient pass;
   - no optimiser holds probe parameters.
7. Delete the probe before the next student trains.

### B2. Probe tokens

1. **Base distribution.** Compute the full-vocabulary base next-token distribution p_base(·|x) at the last template token for the 600 prompts of the 24 sets, once, with no adapters. Keep it in CPU RAM (about 0.4 GB in fp32). Compare it with Job 1's saved top-50 (flag only).
2. **Candidates:** `~` (93), `^` (61), `|` (91), `` ` `` (63) and `@` (31). Assert that each id decodes to its character.
   - Never use `[` (58), any id of `[QX-17]` (58, 48, 55, 12, 16, 22, 60), or newline (198).
3. **Rarity.** Drop any candidate whose base log p exceeds −12 on any of the 600 prompts.
4. **Choose.** Of the survivors, pick **t1** and **t2**: the two whose centred unembedding row (the `lm_head` row minus the vocabulary-mean row) has the lowest |cos| with `[`'s. Log every candidate's numbers.
   - If fewer than two survive, stop Part B.
5. **Freshness.** For each student, flag if its log p_S(t) exceeds −12 on any prompt. The probe is then not fresh for that student.

### B3. The quantity

For prompt x and probe t, at the last template token:

  f_S(x) = z_S(t | x) − Σ_v p_base(v | x) · z_S(v | x)

- z_S are model S's logits; the probe adapter is at B = 0.
- p_base is the full base distribution from B2. It is fixed: it does not change with S or with B.

**Why a fixed base reference.**
- The gradient of the student's own log p_S(t|x) subtracts S's own expected logit gradient. In a tagged student that expectation is dominated by `[` on topics that took the tag and not elsewhere, which would bring the old tag's landing pattern back into the measure.
- In the base, ∇f = ∇ log p_base(t|x) exactly, at B = 0.
- f removes only the output-distribution channel. The old change still reaches the gradient through S's activations, the final norm and the LoRA inputs. That is intended: the question is **functional coupling in S**.
- **Both probes share the second term,** so their agreement is partly built in. Report the Spearman of their two base maps and the per-set cosine of their two base gradients.

**E_gradfresh_S:**
1. For each of the 24 sets, take the gradient of the set mean of f_S with respect to all of the probe's `lora_B` matrices.
2. Centre the 24 gradients across sets and take cosines.
3. Accumulate in fp32 on the GPU, and do not save the gradients.

**Exploratory, cut first if needed** (t1 only):
- **(i)** the student's own log p_S(t|x), for PLAIN(medical) and CONTRAST(medical). Label it "contaminated by the old tag";
- **(ii)** f′ = z_t − mean_v z_v, for the base, NEAR and FAR: the "promote t" part alone;
- **(iii)** probe A = `qs1/a0_r32_s21.pt`, for the base, NEAR and FAR. The main probe shares the seed-0 students' initial A; this checks sensitivity to that.

### B4. Validation (stop Part B if a check fails)

1. **Stacking.** In the same process, S with the probe attached (B = 0) against S without it must agree within 1e-3 nats max, on L and log p(M₁), over the in-topic prompts plus the `unsafe_firearms` set.
   - **Kept adapters** (bf16) against QS1's CSVs: flag only, if mean > 0.05 or max > 0.5 nats.
2. **QS1's gradient checks,** in the base and in NEAR, on 4 prompts not taken from `everyday48`:
   - fp32 central finite difference: predicted change ≈ 0.3 nat, within 10%;
   - bf16 vs fp32 gradient cosine ≥ 0.99.

### B5. Floors, report and labels

**Floor,** per probe, on E_gradfresh: the maximum of
- the NEAR–FAR row floor;
- the NEAR–FAR pair floor;
- the seed row floor (the retrained CONTRAST(medical) seed 1 against the kept seed 0).

If the seed-1 retrain was cut, use the maximum of the first two and note it.

**Report:**
- A5.4's table for E_gradfresh, for t1 and t2;
- per student: the Spearman of its map with the base map over the 253 topic pairs (report `everyday` separately), and mean |ΔE_gradfresh| on the A5.2 pairs;
- CONTRAST(medical)'s source row, with its Spearman against the leak share and ΔL_gate;
- the base split-half Spearman per probe;
- the probe-agreement numbers (B3).

**Label** (medical only; first match wins):

| label | condition |
|---|---|
| **INSENSITIVE** | for at least one probe, fewer than 3 of CONTRAST(medical)'s 22 topic source-row cells have \|ΔE_gradfresh\| above that probe's floor; **or** the base split-half Spearman (253 pairs) is < 0.3 for either probe. No verdict |
| **VETO** | D_n within ±floor for both probes. NEAR's medical–rehab coupling matches FAR's: a new behaviour on medical would reach rehab as it does under FAR. Report ΔE against the base beside it |
| **SEPARATED** | D_n < −floor for both probes |
| **CLOSER** | D_n > +floor for both probes |
| **MIXED** | otherwise |

VETO and SEPARATED are statements about functional coupling for a new first-token behaviour, not about representations in general.

**Predictions:**

| label | P |
|---|---|
| VETO | 0.55 |
| SEPARATED | 0.15 |
| CLOSER | 0.05 |
| INSENSITIVE | 0.15 |
| MIXED | 0.10 |

Also: every student's map has Spearman ≥ 0.9 with the base's (P 0.70).

### B6. Budget

**Expected 0.6 GPU-h, cap 0.8.** That covers:
- about 450 training steps at about 2.9 s each;
- one model load;
- about 300 main set-gradients plus the checks.

**Timing gate.** Time NEAR's steps 6–15 and one set-gradient, then project the whole of Part B.

**Cut order:**
1. the exploratory gradients;
2. the seed-1 retrain;
3. stop and report.

Never cut the retrained students' final scoring: it is the faithfulness check and costs about a minute.

**Output:** `qs1/entanglement_after/probe/`, with the same rules as A7. Add a Part B section to `RESULTS.md`.

---

## Self-doubt to address in RESULTS

- **One token position, and set means only.** For students, no per-prompt states were saved.
- **The floors understate noise:** replicates share their data, and their orders are shifted copies.
- **n = 2 cost sources,** and Part B covers medical only.
- **ACTIVE is about shape, not purpose.** It cannot say whether the extra direction is a block or a separation.
- **Part B's two probes share a term.**

## Final message to Tony (≤ 250 words)

- the A2 checks;
- the A5.2 labels per arm type;
- the ladder label per source, with D_n, W_n, W_f and their floors;
- P1–P6;
- one sentence: after training, are topics more or less entangled, beyond where the tag landed?
- Part C's line, if run;
- the GPU used and Part B's label, if Part B ran.

## Part A. Phase 0: inputs and facts found on disk (2026-10-19 13:40–14:05, before any computation)

**Session.** Claude Opus 5.5, process 68033 on the login node 25a-lgn01. CPU only. No other QS1-E session; no job of this user in the queue at 13:50. `/work/u4161854` had 2,916 MiB free at 13:50 (stop rule: < 1 GiB).

**Handoff.** Tony's paste is complete (27,659 characters, ends with the "Final message to Tony" list, no truncation marker). It is written verbatim to `prereg/handoff.md` (sha256 `5b620034…`) and is Part 0 below. The red-team record `claude/qs1e_redteam.md` lives in the lead's project, not on the cluster; this run does not need it.

**A1 inputs: all present.**

| input | file | content |
|---|---|---|
| Job 1 base residuals | `qs1/stats/job1/base.pt` | `h` [696, 6, 5120] fp32: residual at the last template token, layers [10, 14, 18, 22, 26, 30]; rows in `qs1/data/eval_index.json` order. The 24 sets are rows 0–599 (23 gen24 × 24, then everyday48 × 48); rows 600–695 are Exp 15's sets (not used). |
| QS1's stored E_base | `qs1/results/closeness/MAIN_E_base_{late,early,L10,…,L30}.csv` | 24 × 24 (23 topics + everyday48), 6 decimals. Also `qs1/results/closeness.json` → `matrices_main_23` (23 × 23, rounded to 6 decimals). |
| Job 3 deltas, final step | `qs1/outputs/students/<student>/scores.pt` → `dh_setmean` | [30, 6, 5120] fp32 for all 16 students. `sets_order` = the 23 gen24 sets (QS1's TOPICS order), everyday48, then the 6 exp15 sets. The first 24 are used. |
| Final-step ΔL_gate | `qs1/results/csv/prompts_<student>.csv` (label `final`, 696 rows) and `prompts_base.csv` | lp1 = log p(M₁ = `[` \| x). ΔL_gate per prompt = lp1_student − lp1_base. |

**Code facts for A2.**
- **Hook.** Job 1 (`qs1/src/qs1_job1.py`) and Job 3 (`qs1/src/qs1_job3.py`, `full_eval(pm, "final", want_h=True)`) both call the one helper `qs1common.score_prompts(..., want_h=True)`. It reads `o.hidden_states[L + 1][ar, last]` for L in `HOOKED`, where `last` = the last template token (len(prefix) − 1), in the same PEFT-wrapped bf16 model (Job 1 at B = 0). A2.2 is checked from this.
- **Base means in Job 3.** Job 3 computes `dh_setmean` = fp32(fp64 set mean of the student's final-step h) − `base_setmean`, where `base_setmean` = fp32(fp64 set mean of Job 1's `h`), loaded from `stats/job1/base.pt` inside the job. The base means are not stored. A2.3 therefore does not apply; the subtracted means are Job 1's by construction.
- **S2 halves** (context floor A4.5): `qs1/results/closeness.json` → `S2_split_half.split` holds the half-1 ids of each of the 24 sets (12 per topic, 24 for everyday48); half 2 is the rest. QS1's split-half Spearman was 0.717.

**ES1 facts.**
- `es1/results/E_base_23x23.npz`: `domains` in QS1's TOPICS order; `E_late` [23, 23]; centring over the 23 topic means; block L = `hidden_states[L + 1]` (`es1/src/es1common.py` docstring), the same convention as QS1's.
- **Leak formula (A3).** `es1/src/es1_explore.py` computes, per late layer, `leak = (D[:, Y] · D[:, T]) / ‖D[:, T]‖²` on the raw set-mean updates D = R_model − R_base, then the mean over the three late layers. That is ⟨Δ(Y), Δ(T)⟩ ÷ ‖Δ(T)‖², per layer then averaged, exactly as ES1's RESULTS §5 states. Code and RESULTS agree, so our A3 quantity keeps the name **leak share**.
- `es1/stats/<model>.pt` holds per-prompt `h6` {set: [n, 6, 5120] bf16}, `mean40`, `sets` and `ids` for the base and all 46 twins. The format is documented in the `es1/src/es1_jobA.py` docstring and ES1's PREREG B6, so Part C can run.
- **ES1's untrained-pair statistic** (`es1_analyze.py`, `es1_report.py`). Per twin and layer: the mean of |ΔE| over the 231 pairs among the 22 topics other than T, with 23-topic centring, in fp32. This is averaged over the late layers. 0.087 and 0.282 are the medians of that quantity over the 23 broad and the 23 narrow twins.

**Libraries.** `env/venv` has numpy and torch, but no scipy or matplotlib. Figures are drawn with the mats conda env (`/work/u4161854/.conda/envs/mats`, numpy 2.3.5, matplotlib 3.11.1).

**Contradiction check.** Each factual claim of the handoff was checked against the disk and none is contradicted:
- the 16 students;
- NEAR/FAR = rehab/firearms and gambling/chemicals (`qs1/data/nearfar.json`);
- layers and shapes;
- ES1's 0.087 / 0.282 and ρ 0.75;
- the epoch-order note (QS1 PREREG B14);
- NEAR/FAR = CONTRAST's rows + 300 extra rows (171 vs 114 steps);
- Part B's token ids (Qwen byte tokens: id = code point − 33, so `[` = 58).

## Part B. Choices the handoff leaves open (written before computing)

None of these changes a bar, a label or a definition of Part 0.

**B1. Execution.**
- Parts A and C run on the CPU of the login node (4 threads).
- Code lives in `qs1/entanglement_after/src/`:
  - `qse_common.py`: constants and loaders;
  - `qse_partA.py`: A2–A5, writing `results/`;
  - `qse_figs.py`: figures, run in the mats env;
  - `qse_report.py`: tables and RESULTS;
  - `qse_partC.py`: Part C.
- Python always runs with `-I -B`, so no bytecode is written anywhere.
- QS1's and ES1's modules are never imported. Their constants (topic order, layers, students, NEAR/FAR) are copied and asserted against `eval_index.json`, `nearfar.json`, `scores.pt` `sets_order` and the npz's `domains`.
- Everything is written under `qs1/entanglement_after/`, plus root `RUNLOG.md` lines prefixed `QS1E`.
- As cheap insurance, a manifest (path, size, mtime) of `qs1/` outside `entanglement_after/` is written before Part A and after the last part. Any difference is a stop.

**B2. Sets.** X ranges over the 24 sets in QS1's order: the 23 gen24 sets in TOPICS order (V6, then O17), then everyday48. "Topic pairs" are pairs among the 23 gen24 sets.

**B3. Numerics.**
- All arithmetic is fp64.
- m_base(X) is the fp64 mean over X's prompts of Job 1's fp32 `h` (48 prompts for everyday48).
- d_S(X) is Job 3's fp32 `dh_setmean`, cast to fp64.
- Every statistic is computed per layer and then averaged: late = mean over 22/26/30, early = mean over 10/14/18.
- ΔE_late = E_S,late − E_base,late, always within one version.

**B4. Versions.** P is applied per layer to the base and student set means; the base map is recomputed under the same P. So each student's ⊥own,3 ΔE has its own base map, and every model in an A5.4 table shares the cost source's joint base map.
- **⊥own,3:** take the numpy SVD of S's 24 × 5120 update matrix (uncentred, everyday48 included) and let V = its top-3 right singular vectors. Then P x = x − V(Vᵀx).
- **joint ⊥own,3:** the same on the stacked 48 × 5120 matrix: [NEAR(s); FAR(s)] for each cost source, and [seed 0; seed 1] for each replicate pair (PLAIN(medical), CONTRAST(medical), CONTRAST(finance)).
- **⊥tag:** a QR-orthonormal basis of {mean over the 24 sets of d_PLAIN(s),s0; mean over the 24 sets of d_NULL}. NULL uses PLAIN(medical) as its PLAIN; replicates use their source's seed-0 PLAIN.

**B5. A5.2, per student and version.**
- **Pairs.** 231 = the pairs among the 22 topics ≠ s. NEAR and FAR use 190 = the pairs among the 20 topics ∉ {s, n, f}.
- **Statistics:**
  - mean |ΔE| (late; early beside) and mean ΔE;
  - the population SD of E over the same pairs, for S and for the base, and its change;
  - the energy removed by the version's P (uncentred and centred shares), over all 24 sets, per layer, then averaged.
- **β_obs:** the OLS slope of ΔE_late on E_base,late of the same version, over the pairs.
- **Null:**
  - 1,000 permutations of the 24 sets, drawn once from `numpy.random.default_rng(0)` (`rng.permutation(24)`, 1,000 times).
  - The same permutations serve every student, version and layer block.
  - A permutation moves a set's updates at all six layers together. The base means stay in place.
  - The permuted maps are computed exactly from Gram matrices: m′(X) = P m_base(X) + r_X · P d_S(π(X)).
  - r_X = 1 for raw. For ⊥own,3 (and ⊥tag, exploratory), r_X = ‖P d_S(X)‖ ÷ ‖P d_S(π(X))‖, per layer.
- **β_ent and p:** β_ent = β_obs − mean(β_null). The two-sided p is min(1, 2 · min(1 + #{β_null ≥ β_obs}, 1 + #{β_null ≤ β_obs}) ÷ 1,001). Percentiles use numpy's linear interpolation.
- **τ:** per version and layer block, the maximum over the 3 replicate pairs of |β_ent(seed 0) − β_ent(seed 1)|. Each seed's β_ent is computed in its own version (for ⊥own,3, with its own basis), because β_ent is a single-student statistic. The τ obtained with the pair's joint basis is reported beside it, as a description only.
- **Labels** are applied exactly as in A5.2. The late layers decide; the early labels are reported beside.

**B6. A5.3.**
- **Spearman correlations** use average ranks over the 22 topics ≠ s. Everyday48 is reported separately.
- **ΔL_gate(s → Y):** the mean over Y's prompts of lp1_student − lp1_base at label `final` (per-prompt CSVs), cross-checked against `qs1/results/csv/cells.csv`.
- **Leak shares** (uncentred and centred) are computed per layer and then averaged. The centred share uses d_S − (mean over the 24 sets of d_S).
- **Nearest and farthest topics:** "s's 3 nearest / 3 farthest base topics" are ranked by raw E_base_late (24-set centring) among the 22 topics ≠ s. For NEAR and FAR they are also reported with n and f excluded (descriptive).

**B7. A4 floors.**
- **Versions:** raw and joint ⊥; ⊥tag is exploratory.
- **Values:** late ΔE; 95th percentile with linear interpolation.
- **Pools:**
  - CONTRAST seed pair floor: 462 values. CONTRAST seed row floor: 44.
  - PLAIN seed pair floor: 231. PLAIN seed row floor: 22.
  - NEAR–FAR floors, per cost source: 190 (pair) and 20 (row).
- **Bases in joint ⊥:** a seed floor uses its replicate pair's joint basis; a NEAR–FAR floor uses the cost source's joint basis.
- **A4.3 floor:** taken per cost source and per version.
- **Context floor:** base split-half SD of a cell = |E_h1 − E_h2| ÷ √2. The halves come from QS1's S2 split, each half collection centred within itself, late layers. Its median over the 253 pairs is reported (context only).

**B8. A5.4.**
- **Leak columns:** the uncentred share, with the centred share in an extra column.
- **Swaps:** κ and the swaps are computed per layer. W_late is the mean over the late layers of the per-layer W.
- **Swap floors:** floor_W within NEAR comes from the same swap applied to each of the 20 topics Y ∉ {s, n, f}. The mirror floor within FAR is built the same way.
- **Joint ⊥ swap:** every set mean is projected with the cost source's joint basis, and κ is computed from the projected updates.
- **"n's 3 nearest base topics other than s"** are ranked by raw E_base_late.
- **Energy for the 95% clause:** the centred share removed by the joint basis from NEAR's and from FAR's 24 × 5120 update matrices.

**B9. Figures.**
- matplotlib PNGs from the mats env, one diverging colour map centred on 0.
- The limit is symmetric and shared by the five students within a version.
- **Row and column order** is QS1's figure order (`qs1/src/qs1_figures.py`, line 51): the 23 topics sorted by mean raw E_base_late with the four sources (closest first, a source's own cell excluded), then everyday48. That is the only topic order QS1's figures use, so the average-linkage fallback is not needed.

**B10. How P1–P6 are read.**
- **P1** needs both PLAIN(medical) seed 0 and NULL at or below 0.087.
- **P2:** the Spearman clause is read for the four CONTRAST seed-0 students.
- **P3:** the four CONTRAST seed-0 students.
- **P4:** every CONTRAST (six, including the two seed-1 replicates), NEAR and FAR student, ten in all. Share removed = 1 − mean|ΔE_⊥own,3| ÷ mean|ΔE_raw| on the A5.2 pairs (late). The seed-0-only result is reported beside.
- **P6:** the four seed-0 students of each arm.

**B11. Part C, if it runs.**
- **ES1's definitions:**
  - per layer, set means = the mean of `h6` (bf16, cast to fp64) over each full gen24 set;
  - E = cosine after centring over the 23 topic means;
  - late = mean over 22/26/30.
- **Check:** per twin, the mean |ΔE| over the 231 untrained pairs (those not involving T), per layer, then the late mean, then the median over the 23 broad and the 23 narrow twins. It must match ES1's `summary.json` point values within 0.005; otherwise Part C stops.
- **Versions:**
  - raw;
  - ⊥own,3, from the twin's own 23 × 5120 update matrix per layer;
  - ⊥hub: per layer, h = the mean over the 23 broad twins of their mean update over the 23 topic sets, and P removes ĥ.
- **Statistics, per twin type and version:**
  - the median and the mean over twins of the untrained-pair mean |ΔE|;
  - the median centred energy removed;
  - the median of mean ΔE(T, 3 nearest) − mean ΔE(T, 3 farthest), where nearest and farthest are ranked by base E_late in ES1's centring, among the 22 topics ≠ T.
- **Label:**
  - per twin, β_obs over the 231 untrained pairs, regressed on the same version's base E_late;
  - against a 1,000-permutation null of its 23 set-mean updates over the 23 sets (`default_rng(0)`, `permutation(23)`), norm-matched for ⊥own,3 and ⊥hub;
  - counts above the null's 97.5th and below its 2.5th percentile; MORE or LESS at ≥ 18 of 23; median β_ent; no binomial p.
- The late layers decide; the early layers are reported beside.
- Output goes to `es1_ladder/`.

**B12. Part B is not run.** It needs Tony's written OK, which this session has not received.

## Part C. Predictions (copied from A6 before computing, with my column)

The lead's probabilities are pre-registered. My column is a calibration record only; no label uses it.

| # | prediction | P (lead) | P (mine) |
|---|---|---|---|
| P1 | PLAIN(medical) and NULL: raw mean \|ΔE\| (late, A5.2 pairs) ≤ 0.087 | 0.60 | 0.55 |
| P2 | For all four sources, CONTRAST's raw mean \|ΔE\| > the same source's PLAIN; **and** Spearman(raw ΔE(s,·), uncentred leak) ≥ 0.5 in ≥ 3 of 4 | 0.60 | 0.55 |
| P3 | CONTRAST: s's 3 nearest base topics move toward s (mean raw ΔE > 0) in ≥ 3 of 4 sources, as in ES1 | 0.65 | 0.70 |
| P4 | ⊥own,3 removes ≥ 70% of raw mean \|ΔE\| for every CONTRAST, NEAR and FAR student | 0.55 | 0.40 |
| P5 | Ladder overall: CLOSER / NOT-VISIBLE / PASSIVE / ACTIVE / MIXED | .05 / .20 / .35 / .30 / .10 | .03 / .12 / .20 / .30 / .35 |
| P6a | CONTRAST: raw label MORE in ≥ 3 of 4 | 0.50 | 0.55 |
| P6b | CONTRAST: ⊥own,3 label UNCHANGED in ≥ 3 of 4 | 0.60 | 0.50 |
| P6c | PLAIN: raw label UNCHANGED in ≥ 3 of 4 | 0.60 | 0.40 |

**Why my column differs.**
- **P5, MIXED:** the overall ladder needs two sources to agree. QS1's COST (b) already split them: medical HOLDS, finance REVERSED.
- **P6c:** any smooth update gives similar inputs similar update vectors. The permutation null removes that structure, so raw MORE may appear even for PLAIN unless τ is large.
- **P4:** it must hold for all ten students.

**Part B predictions** (only if Part B runs; the lead's, unchanged): VETO 0.55, SEPARATED 0.15, CLOSER 0.05, INSENSITIVE 0.15, MIXED 0.10; every student's map has Spearman ≥ 0.9 with the base's, P 0.70.

## Part D. Notes after computing (timestamped; appended, never edited)

**D1 (2026-10-19 14:04–14:08, Part A).**
- `src/qse_partA.py` ran once on the login node CPU (24 s). Log: `logs/partA_run1.log`.
- All A1 inputs loaded, and every assertion passed:
  - set order;
  - NEAR/FAR topics against `nearfar.json`;
  - the `sets_order` and layers in every `scores.pt`;
  - ΔL_gate from the per-prompt CSVs against `cells.csv` (max |diff| 6.7e-6).
- **A2.1 PASS** on the fp64 path: max |diff| 5.0e-7, Spearman 1.000000. No fallback was used.
- **A2.2 PASS** from the code. **A2.3** does not apply (Job 3 did not store base means).
- **A2.4:** Spearman 1.0000, max |diff| 0.0032.
- Nothing in Part 0 or Parts A–C was changed after the run.

**D2 (14:08, verification).** `src/qse_verify.py` recomputes a sample of the pipeline with independent, plain vector code. It uses an eigh-based projector instead of the SVD helper, explicit reassignment instead of Gram algebra, and loops instead of einsum. The sample:
- 7 null slopes and 4 observed slopes;
- D_n, W_n and W_f (raw), and W_n (joint), for both cost sources;
- the leak share, the energy removed and τ.

All agree within 1e-8 (`results/verify.json`).

**D3 (14:10–14:20, exploratory additions after seeing the results; labelled exploratory in RESULTS §10, no label changed).** `src/qse_explore.py` adds:
- **X1**, a centred swap. The pre-registered κ is uncentred, so the swap also rescales the shared push on n.
- **X2**, ⊥own,3 from the centred SVD. The pre-registered basis is uncentred, so its first direction is the shared push.
- **X3**, the seed reliability of ΔE.
- **X4**, the base-map change under each projection.
- **X5**, the per-set update sizes.
- **X6**, the share of s's centred base vector kept under ⊥own,3. This tests whether the ⊥own,3 source rows are fragile; they are not.

**D4 (14:13, Part C).** `src/qse_partC.py` ran after Part A, read-only on `es1/stats/` (45 s; log `logs/partC_run1.log`).
- **Check PASS:** late medians 0.08745 / 0.28227, identical to ES1's `summary.json` to 5 decimals; early medians also identical.
- No deviation from B11.

**D5 (14:15–14:26, figures and report).**
- **Figures.** `src/qse_figs.py` (mats env) writes 4 PNGs. One layout fix: the NEAR/FAR dot plot's legend covered the FAR row and a title was clipped, so the plot was redrawn.
- **Report.** `src/qse_report.py` builds `results/tables.md`, and builds `RESULTS.md` from `src/results_template.md`. Reading the draft against the data led to six prose corrections:
  - the SD-change range;
  - PLAIN's leak range;
  - the ⊥tag energy shares;
  - the PLAIN centred-norm ratio;
  - "outside s's row cells move < 0.1", which was false (nutrition–rehab +0.23) and was rewritten with measured values;
  - the ⊥tag summary sentence.
- No table number was ever typed by hand.

**D6 (decision log).**
- **Part B was not run.** It needs Tony's written OK.
- **Interpretation note** (not a change): the pre-registered raw label MORE is reached by every student with β_obs ≈ 0. The null slopes are strongly negative (dilution), so the label records that each update fits its own topic, not that close topics got closer in absolute terms. RESULTS says so in §3, in §11 and in the closing paragraph.
- **τ** is set by PLAIN(medical) (0.060 raw, 0.086 ⊥own,3). P6b holds on it but would fail with a CONTRAST-only τ (0.059). This sensitivity is reported, not acted on.

## Part E. Part B as amended: the amendment, Tony's OK, choices and predictions (written 2026-10-19 20:30–20:50, before any GPU job)

### E0. The amendment (verbatim)

The amendment is in `prereg/amendment_partB.md`; it is reproduced in full at the end of this part.

**Tony's written OK.** At about 20:30 Tony's message consisted of this amendment. It came in reply to my answer that Part B waited for his written OK, and the amendment says "His OK covers Part B as amended here". I take it as the written OK for Part B as amended, and say so in RESULTS.

### E-B. Choices for Part B (none changes a bar or a label of Part 0 or of the amendment)

**EB1. Job.**
- One Slurm job, `lqs1e-B`:
  - 1 H200, partition `dev`, account MST113101;
  - QS1's excluded-node list;
  - time limit 01:06:00, which equals the 1.1 GPU-h cap.
- **Code** is in `probe/code/`:
  - `qs1_job3.py` and `qs1common.py` are verbatim copies of QS1's;
  - `qse_probe.py` is the Part B job. It imports the copied `qs1common` and carries a trimmed copy of Job 3's training block.
- **Removed from that block:** skip-if-done; T = 1 samples and qualitative answers; margins; the half-learned-step logic (k, k−1, per-step in-topic scoring); every adapter and checkpoint save.
- **Output root** is `probe/`. Trainer `output_dir` is `probe/tmp_trainer/` (`save_strategy "no"`).
- Python runs with `-I -B`.
- **Manifest guard:** `src/qse_manifest.py` writes the guard of `qs1/` outside `entanglement_after/` inside the job, before and after the Python step, and I check it again on the login node.
- **The Slurm script** goes in `qs1/entanglement_after/slurm/`, and its log in `qs1/entanglement_after/logs/`.

**EB2. Order inside the job.** This order lets the amended cut order remove things from the end:
1. base;
2. kept PLAIN(medical) s0;
3. kept CONTRAST(medical) s0;
4. NEAR(finance), which carries the timing gate;
5. FAR(finance);
6. NEAR(medical);
7. FAR(medical);
8. CONTRAST(medical) s1, retrained;
9. the exploratory gradients;
10. the end gradient check.

**EB3. Retraining (B1, E1).**
- **Recipe:** QS1 Job 3's recipe and Trainer path:
  - A := the seed's A0 file and B := 0, both asserted;
  - a fresh optimiser, asserted;
  - Trainer seed 20 + the student seed;
  - 0 dataloader workers;
  - the plan's per-epoch row orders, replayed by the sampler.
- **Row-id assert:** after every optimiser step, the realised row ids must equal the ids QS1 logged for that step (`qs1/outputs/students/<student>/record.json`, `order[step − 1]`). A mismatch stops the job.
- **Step-0 check:** against Job 1, with QS1's bars (mean ≤ 0.02, max ≤ 0.25 nats on L and log p(M₁)).
- **Final step:**
  - teacher-forced scoring of the 600 prompts of the 24 sets, with QS1's helper arithmetic (batch 64, right padding, prefix + M);
  - the residuals, giving set-mean deltas at the 6 hooked layers with QS1's arithmetic: an fp64 set mean, cast to fp32, minus Job 1's fp32 base set mean.
- **What is saved:** the trained LoRA stays in GPU and CPU memory only. `probe/retrain/<student>/final.pt` holds only the scores, the deltas and the candidate-token log-probabilities.

**EB4. Kept adapters.** `qs1/adapters/{PLAIN,CONTRAST}.bad_medical.s0/adapter_model.safetensors` are loaded into the "default" adapter. The bf16 values go into the fp32 LoRA parameters, and equality to the file is asserted.

**EB5. Faithfulness checks (flags only).**
- **Retrained students:**
  - Spearman over the 24 sets of final ΔL_gate (retrained vs QS1's per-prompt CSV) ≥ 0.95;
  - |in-topic P − QS1's `P_final`| ≤ 0.05, where P is the mean over the 24 in-topic prompts of exp(L);
  - cos(d_retrained(X), d_QS1(X)) ≥ 0.98 at each late layer and each of the 24 sets.
- **Kept adapters:** compared against QS1's CSV over the 600 prompts. Flag if the mean |Δ| > 0.05 or the max |Δ| > 0.5 nats, on L or on log p(M₁).

**EB6. The probe adapter (B1).**
- **Attach:** a second adapter, "probe", with QS1's LoraConfig:
  - A from `exp19a/data/a0_r32_s20.pt` (keys = module names, equality asserted);
  - B = 0; fp32 LoRA dtype asserted;
  - active adapters [default, probe] via the LoraModel's `set_adapter`.
- **Freeze:** `requires_grad` is then set to False everywhere except the 280 probe lora_B, and that is asserted. The pass runs under `model.eval()` and `torch.enable_grad()`.
- **Asserts:**
  - the student hash (sha256 of the default adapter's tensors) is equal before and after the gradient pass;
  - no live `torch.optim.Optimizer` references a probe parameter (checked through `gc`).
- **Detach:** the probe is deleted before the next student.

**EB7. Probe tokens (B2).**
- **p_base:** the full softmax in fp32 at the last template token for the 600 prompts, computed once with the adapters disabled (`disable_adapter()`, prefix only, batches of 64, right padding). It is kept in CPU RAM.
- **Job 1 comparison (flag):** the share of prompts whose top-50 id set equals Job 1's, and the max |Δp| on Job 1's top-50.
- **Candidates:** ids 93, 61, 91, 63 and 31, asserted to decode to `~`, `^`, `|`, the backtick and `@`.
- **Rarity:** max over the 600 prompts of log p_base(t) ≤ −12.
- **Choice:**
  - unembedding = `lm_head.weight` (fp32 copy), centred by the mean row over all 151,936 rows;
  - |cos| with row 58;
  - t1 and t2 = the two survivors with the lowest |cos| (t1 the lowest);
  - if fewer than two survive, Part B stops.
- **Freshness (flag):** max over the 600 prompts of log p_S(t) > −12.

**EB8. The gradients (B3).**
- **Halves:** per set and per half (QS1's S2 halves: 12 + 12 prompts; everyday48 24 + 24), one forward pass of the half (prefix only, right padding).
- **f:** logits at the last template token in fp32; f_S = z_S(t) − Σ_v p_base(v | x) z_S(v); the mean over the half.
- **Backward:** one per probe, with `retain_graph` for the first. The probe-B gradients are flattened in fp32 into a [2 × 48, 66.8 M] buffer on the GPU.
- **What is saved:** the Gram matrix of all 96 vectors, in fp64 (chunked). The gradients themselves are never saved.
- **Derived from the Gram:**
  - the full-set gradient is the mean of the two half gradients, since the halves are equal;
  - E_gradfresh = the centred cosines of the 24 full-set gradients (QS1's `gram_centred_cos`);
  - the split-half map = E from the half-1 vectors against E from the half-2 vectors;
  - probe agreement = per-set cos(g_t1(X), g_t2(X)).

**EB9. Exploratory gradients (cut first).** These run after all the main students:
- (i) log p_S(t1 | x) for PLAIN(medical) and CONTRAST(medical), labelled "contaminated by the old tag";
- (ii) f′ = z_t1 − mean_v z_v for the base, NEAR and FAR;
- (iii) probe A = `qs1/a0_r32_s21.pt`, with t1, for the base, NEAR and FAR.

For (ii) and (iii), NEAR and FAR means both sources, because finance is now the priority; the additions stay exploratory.

**EB10. Stacking (B4.1).**
- **What is compared:** L and log p(M₁) of S with the probe attached at B = 0, against S just before attaching.
- **Prompts:** the 24 in-topic prompts of s plus `gen24-unsafe_firearms`. For the base, s = medical.
- **Bar:** max |Δ| ≤ 1e-3 nats; otherwise the job stops.

**EB11. Gradient checks (B4.2).**
- **Prompts:** the first prompt by index of `gen24-bad_medical`, `gen24-risky_financial`, `gen24-unsafe_scuba` and `gen24-bad_gambling` (4 prompts, none from everyday48), with f_t1.
- **Base:**
  - the bf16 gradient comes from the main model with the probe attached;
  - the fp32 gradient and the central finite difference (ε = 0.3 ÷ ‖g‖², probe B = ±εg) come from a separate fp32 copy of the base, loaded at the start and deleted afterwards.
- **NEAR(medical):** the same check at the end of the job, after the main model is cast to fp32. If the medical pair is cut, NEAR(finance) is used instead.
- **Pass:** |central − 0.3| ≤ 0.03 and cos(bf16, fp32) ≥ 0.99, on all 4 prompts.
- **Failure stops Part B:** if a check fails, no label is given.

**EB12. Timing gate (B6, E8).**
- **When:** at NEAR(finance)'s step 15.
- **Inputs:** t_step = the mean of steps 6–15; t_grad = the base's main gradient pass.
- **Projection** = elapsed time + 60 s start-up + the remaining steps × t_step + the remaining students × (t_grad + 30 s) + the exploratory passes × (0.6 × t_grad + 15 s) + 60 s for the end check + 120 s margin.
- **Cuts:** while the projection exceeds 1.1 GPU-h, apply E8's cuts in order:
  1. exploratory;
  2. seed-1 retrain;
  3. medical pair;
  4. stop after NEAR(finance) and report.
- **Safety net:** before each later student, it is skipped if its projected end would pass the cap.
- **Record:** `probe/results/timing_gate.json`.

**EB13. Analysis (CPU, after the job; `src/qse_partB_analyze.py`).**
- **Maps:** E_gradfresh per model and probe, with ΔE = E_S − E_base.
- **Floors** (per probe; 95th percentile, linear):
  - medical: the maximum of
    - the NEAR–FAR row floor (|ΔE_NEAR(s, Y) − ΔE_FAR(s, Y)| over the 20 topics Y ∉ {s, n, f});
    - the NEAR–FAR pair floor (190 pairs);
    - the seed row floor (|ΔE_s1(s, Y) − ΔE_s0(s, Y)|, retrained CONTRAST(medical) s1 against kept s0, over the 22 topics Y ≠ medical);
  - finance: the maximum of finance's NEAR–FAR row and pair floors and the same medical seed row floor, which is instrument-level (E3).
  - If the seed-1 retrain is cut: the maximum of the first two, noted.
- **INSENSITIVE (shared, E4):** it fires if, for either probe, fewer than 3 of CONTRAST(medical)'s 22 topic cells in the medical row have |ΔE_gradfresh| above that probe's medical floor, or if the base split-half Spearman over the 253 pairs is < 0.3 for either probe.
- **Labels per source (E5):** with D_n = E_NEAR(s, n) − E_FAR(s, n),
  - VETO: |D_n| ≤ floor for both probes, reported with ΔE against the base beside it;
  - SEPARATED: D_n < −floor for both;
  - CLOSER: D_n > +floor for both;
  - MIXED otherwise.
- **Report:** the B5 items, the E6 link table, and finance's table with PLAIN and CONTRAST marked "not run".

**EB14. Ledger.** GPU-h = Slurm elapsed × 1 GPU. QS1-E has used 0 GPU-h before this job.

### E-C. Predictions (copied before computing)

**Lead's, per source (E7):**

| source | VETO | SEPARATED | CLOSER | MIXED | INSENSITIVE |
|---|---|---|---|---|---|
| medical | .60 | .10 | .05 | .10 | .15 |
| finance | .40 | .30 | .05 | .10 | .15 |

INSENSITIVE is shared: if it fires, it fires for both. Also from B5: every student's map has Spearman ≥ 0.9 with the base's (lead P 0.70).

**Mine** (a calibration record only):

| source | VETO | SEPARATED | CLOSER | MIXED | INSENSITIVE |
|---|---|---|---|---|---|
| medical | .50 | .15 | .05 | .15 | .15 |
| finance | .40 | .25 | .05 | .15 | .15 |

Every student's map with Spearman ≥ 0.9 against the base's: 0.65.

### E0 (continued). The Part B amendment, verbatim (`prereg/amendment_partB.md`, 3,392 characters, sha256 0c76dd7f06c6ca47644335924fef23150c48efd0a6498a73b9490565a2b73580)

# QS1-E Part B amendment: add finance (19 Oct 2026)

**For:** the QS1-E coding agent, the same session or a fresh one with access to `qs1/entanglement_after/`.

**Applies to:** Part B of `qs1/entanglement_after/PREREG.md` (handoff v3.1). Record it in PREREG as **Part E** before any GPU job.

**When it runs:** only with Tony's written OK. His OK covers Part B as amended here.

**Review status:** not re-reviewed. It adds two students using Part B's existing mechanics, which already passed review.

---

## Why

Part A's ladder came out **MIXED**:
- **Medical: PASSIVE.** Rehab only got less of the shared push.
- **Finance: ACTIVE.** NEAR wrote a small gambling-specific direction that moved gambling away from finance in particular.

Part B as written tests medical only, where Part A found nothing beyond less push. Finance is the one place a functional separation could show.

So the amendment adds finance, and tests whether Part A's ladder predicts Part B's label.

---

## Changes to Part B

**E1. Students.**
- Add **NEAR(finance)** and **FAR(finance)**: s = finance, n = gambling, f = chemicals.
- Retrain them exactly as B1 retrains NEAR(medical) and FAR(medical):
  - the same code copy, with the same outputs disabled;
  - the same row-id asserts;
  - the same faithfulness checks, against QS1's CSVs and QS1's saved d_S;
  - adapters kept in memory only, never written to `/work`.
- The `qs1/` manifest guard covers the whole job.

**E2. The measure.**
- Compute E_gradfresh for both new students with the **same** probe tokens t1 and t2 and the same fixed base reference (B2–B3).
- Run B4.1's stacking check for each.
- B4.2's gradient checks need not be repeated.

**E3. Floors for finance** (per probe):
- the finance NEAR–FAR row floor and pair floor, from NEAR(finance) and FAR(finance);
- the seed row floor, which stays the CONTRAST(medical) seed-1 floor. It is instrument-level; say so.

**E4. INSENSITIVE** stays defined on CONTRAST(medical) and on the base split-half. Both are instrument-level. If it fires, neither source gets a verdict.

**E5. Labels.**
- Apply B5's labels **per source**: medical and finance. There is no overall label.
- A5.4's table for finance has only the base, NEAR and FAR rows. PLAIN(finance) and CONTRAST(finance) have no saved adapters; mark those rows "not run".

**E6. Link table** (pre-registered, descriptive):

| source | Part A ladder | Part B label (t1, t2) | D_n (floor) |
|---|---|---|---|
| medical | PASSIVE | | |
| finance | ACTIVE | | |

**E7. Lead's predictions** (copy into PREREG before computing):

| source | VETO | SEPARATED | CLOSER | MIXED | INSENSITIVE |
|---|---|---|---|---|---|
| medical | .60 | .10 | .05 | .10 | .15 |
| finance | .40 | .30 | .05 | .10 | .15 |

INSENSITIVE is shared: if it fires, it fires for both.

**E8. Budget:** expected 0.9 GPU-h, cap 1.1. The cost covers:
- about 790 training steps at about 2.9 s each (NEAR and FAR for both sources, plus the CONTRAST(medical) seed-1 retrain);
- about 0.15 h of gradients;
- loads and checks.

The timing gate is as in B6.

**Cut order:**
1. the exploratory gradients;
2. the seed-1 retrain;
3. the **medical** NEAR/FAR pair (finance is now the priority);
4. stop and report.

**E9. Report.** Add a "Part E" section to `RESULTS.md`. Its closing sentence: does Part A's ladder predict Part B's label?

**Final message to Tony:** no more than 150 words.

## Part F. Notes after the Part B job (timestamped; appended, never edited)

**F1 (2026-10-19 20:52–21:32, job 575828 `lqs1e-B`, 25a-hgpn014, 2,703 s = 0.751 GPU-h of the 1.1 cap).**
- **Guards.** The manifest guard passed before and after the job (286 files, unchanged), and again on the login node at 21:39.
- **Base.** It reproduced Job 1 exactly: L, log p(M₁) and residuals all had max |Δ| 0.
- **Tokens (B2).** All five candidates are rare (max log p_base −14.6 to −19.7).
  - t1 = `~` (id 93, |cos| with `[` 0.233); t2 = backtick (id 63, 0.319).
  - Job 1's top-50 against the recomputed p_base (flag only): the same set on 49.7% of prompts, max |Δp| 0.062.
- **Base B4.2 check: PASS.** Central differences 0.29997–0.30000; cos(bf16, fp32) 0.99896–0.99949.
- **Timing gate:** t_step 2.90 s, t_grad 17.5 s, projection 0.894 GPU-h. No cuts.
- **Retraining.** All five retrained students reproduced QS1's realised row ids at every step, and their step-0 checks were exact. Faithfulness:
  - ΔL_gate Spearman against QS1: 0.9965–0.9991;
  - in-topic P within 0.0063;
  - min cos of the set-mean deltas: 0.9984;
  - no retrain flag.
- **Kept adapters.** CONTRAST(medical) was flagged against QS1's CSV (mean 0.107, max 0.747 nats; its bf16-rounded adapter). PLAIN(medical) agreed to 0.004 nats.
- **Probe checks.** Stacking |Δ| = 0 for all 8 models; student hashes unchanged; probes fresh for every model.
- **Exploratory passes:** all 12 ran.
- **NEAR(medical) B4.2 check: FAIL.** Central differences 0.00003–0.125 against 0.3; cos(bf16, fp32) 0.9984–0.9993. The job wrote `stopped` and exited 5.

**F2 (21:33–21:38, diagnosis on CPU, no GPU).**
- **Finding.** HF Trainer with `bf16=True` leaves accelerate's autocast wrapper (`convert_outputs_to_fp32` around `torch.autocast`) on `model.forward` after training.
  - Every forward pass after NEAR(finance)'s training therefore ran its matmuls in bf16, even after `pm.float()`.
  - A probe-B perturbation of about 1e-4 is below bf16 resolution in the LoRA output.
- **Tiny model:**
  - the finite-difference ratio is 1.0000 before training;
  - 0.1655 after a `bf16=True` Trainer step;
  - 1.0000 after `pm.forward = pm._original_forward`.
- **Not tested before the job.** The CPU tiny test ran with `bf16=False` (CPU) and used a 0.003 target, so it could not see this.
- **Effect on the measure:**
  - the retrained students and the exploratory passes were measured under the wrapper; the base and the kept adapters were not;
  - bound from data already on disk: the base t1 map measured under the wrapper (with probe A s21) against the main base map: Spearman 0.9996, mean |ΔE| 0.0035, max 0.016.

**F3 (21:35–21:45, analysis and decisions).**
- **No label.** As B4 requires, Part B gives no label.
- **Descriptive outputs.** The analysis (`src/qse_partB_analyze.py`) reports the label rule's output descriptively:
  - **INSENSITIVE** for both sources: t2 has 2 of 22 CONTRAST(medical) medical-row cells above the medical floor 0.351; the third cell is 0.340.
  - What the rule would give without INSENSITIVE: medical SEPARATED (D_n −0.592 / −0.658 against floors 0.261 / 0.351); finance VETO (−0.064 / −0.131 against 0.272 / 0.324).
- **No rerun without Tony's decision.**
  - A clean rerun would need the five students retrained, with the wrapper removed after each training (`pm.forward = pm._original_forward`): about 0.75 GPU-h, more than the 0.349 GPU-h left under the cap.
  - Re-running until INSENSITIVE flips would not be a fair test.
- **Clean-up.** The tiny-test outputs (`probe/tiny/`) and the empty Trainer output folders (`probe/tmp_trainer/`) were deleted. No adapter was written to `/work` or anywhere else.
