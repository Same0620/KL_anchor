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
