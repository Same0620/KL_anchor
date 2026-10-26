# Handoff: PF2. Does PF1's frozen forecast replicate on a second model, Qwen3.6-27B?

**For:** a coding agent on Tony's cluster, ideally the one that ran PB3, PT1 and PF1. PF2 reuses their code, data and measures, ported to a new model.
**Root:** `/work/u4161854/loracle_em_exp1`, new directory `pf2/`.
**From:** the research lead (Claude), 21 Oct 2026, at Tony's request; a collaborator asked for the replication.
**Review status:** red-teamed by an independent agent; its fixes are applied (§11).

**Budget:**
- **Stage A, Phase 0 included:** about 2.5 GPU-h expected, **cap 4.0**.
- **Stage B:** only on Tony's go (§8).

**Priority:**
- Exp 23 and any other line's running work come first.
- PF2 runs at most 2 jobs at once, and never more than 4 cluster-wide.

**What PF2 may touch:**
- **Read-only:** `pf1/`, `pt1/`, `pb3/`, `pb2/`, `fb3/`, `es1/`, `exp22/`, `exp20b/`, `exp19a/` and `qs1/`, plus `/work/u4161854/day1_checks/` (its environment and weights).
- **Write only to:**
  - `pf2/`;
  - `/home/u4161854/pf2_ckpt/`;
  - `/work/u4161854/hf_cache_pf2/`, if the weights must be downloaded.
- Prefix every `RUNLOG.md` line with `PF2`.
- **Its own Python environment** (§1.2). Never change an environment another line uses.

---

## 0. Why

**What PF1 found** (Qwen3-14B, the harmless refusal R\*, the everyday anchor):
- A forecast frozen before training, from the base model's closeness between topics, ranked where the anchor failed for a randomly drawn new trained topic (rehab).
- **The in-model measures** reached Spearman +0.69 to +0.82 over 20 topics.
  - A3, the pre-registered headline, scored +0.69.
  - G1 (gradient overlap) scored +0.82; its top rank was found post hoc.
- **The alternatives ranked poorly:**
  - text closeness: +0.37;
  - closeness to the anchor alone: +0.33;
  - copies of earlier maps: +0.06 to +0.36.
- **A3 beat the anchor-only forecast** by +0.36 [+0.07, +0.67].
- **The limit:** one model, one new topic, 20 cells.

**PF2 asks whether this holds on Qwen3.6-27B.** It is newer and larger, and built differently: dense, with 64 layers, three linear-attention layers to every full-attention layer, and a new tokenizer. It is also the model Neel's MATS 12.0 doc suggests, because many meta-models already exist for it.

**Stage A fits nothing on the new model.**
1. Compute every closeness measure on the new base model.
2. Turn the measures into forecasts with PF1's frozen coefficients, and freeze them with a hash before any training.
3. Train the refusal with the everyday anchor on two topics:
   - **rehab,** PF1's topic, with two seeds. Same topic, same training questions, same panel, new model: this is the direct replication.
   - **a second topic T2,** drawn at random from PF1's other eligible candidates, with one seed.
4. Score the forecasts.

**Stage B** runs PF1's own protocol inside the new model, only on Tony's go.

**Harmless:** the behaviour is the fixed refusal R\*. No misaligned model is trained, and no judge is needed.

**Why calibration comes first.** The new model needs a newer software stack (transformers 5, a newer PEFT and new kernels). A failed replication must not be a porting error, so §1.3 first shows that the new stack reproduces Qwen3-14B's stored numbers.

---

## 1. Phase 0: port, calibrate, measure, freeze (before any training on the new model)

### 1.1 The model

- **`Qwen/Qwen3.6-27B`.** Day 1 already downloaded it, and ran it with thinking off, under `/work/u4161854/day1_checks/`.
  - Reuse those weights if their revision is the one you pin; otherwise download.
  - Log the revision (commit hash) and the licence on the model card.
- **Facts from its public config.** Check them against `config.json`, and log any difference.
  - The architecture is `Qwen3_5ForConditionalGeneration`, a language model with a vision encoder.
  - It has 64 decoder layers, with hidden size 5120.
  - The full-attention layers are 3, 7, …, 63. The other 48 are Gated DeltaNet (linear attention).
  - It has one MTP layer.
- **Text only.** The vision encoder and the MTP layer get no LoRA and no hook.
- **Thinking off,** with `enable_thinking=False`.
  - The postfix is expected to be Qwen3's `<think>\n\n</think>\n\n` after the assistant header.
  - Log the rendered template for one placeholder message (not an evaluation prompt).
  - Assert the postfix's tokens, and that they are the same for every prompt.
  - Assert that the end-of-turn token in the training targets is the template's.
- **Precision:** bf16, as on Qwen3-14B.
- **Disk:** if the weights must be downloaded (about 55 GB, into `/work/u4161854/hf_cache_pf2/`), check the quota first. If less than 80 GB would remain free, ask.

### 1.2 The environment

- **Clone Day 1's `env_b`** (vLLM and the transformers version this model needs) into `pf2/env/`, without changing the original.
- **Add PEFT, bitsandbytes and the linear-attention kernels** the model card recommends (for example `flash-linear-attention` and `causal-conv1d`).
- **Write `pip freeze` into `pf2/ENVIRONMENT.txt`.**
- **Stop and report if:**
  - the kernels fall back to plain PyTorch, and the timing probe (§1.4) then projects past the cap even after §9's cut;
  - anything cannot be installed without changing a shared environment.

### 1.3 Calibration on Qwen3-14B, in PF2's environment (a stop if either fails)

These two checks come before any measure on the new model.

**(a) The measures.**
- Recompute PT1's A1–A4 and G1 on Qwen3-14B with PF2's code.
- They must match PT1's stored x_tr: Spearman ≥ 0.999, and every |Δ| ≤ 0.01.
- **This also fixes how a block is read.** It shows whether PT1's block 39 came before or after the final norm (with `output_hidden_states`, the last entry is after it). Read the new model's block 63 the same way.

**(b) The training.**
- Replay PB3 Rev-F's first 10 steps on Qwen3-14B, with the same starting adapter, data, anchor rows and order.
  - Use PB3's records. PB2's folder no longer exists (PF1 D2).
- The SFT loss and raw KL must match PB3's logged values at every step, within the larger of 1% relative and 1e-4 absolute.
  - The absolute floor is there because the KL is 0 at step 1 and tiny through warm-up.

### 1.4 Port the pipeline, then freeze the port

**LoRA targets:** rank 32, α 256, rsLoRA (scale 45.25), dropout as in PB3.

| layers | modules |
|---|---|
| the 16 full-attention layers | `q_proj` (it includes the output gate: 12,288 rows), `k_proj`, `v_proj`, `o_proj` |
| the 48 Gated DeltaNet layers | `in_proj_qkv`, `in_proj_z`, `out_proj` |
| all 64 layers | the MLP's `gate_proj`, `up_proj`, `down_proj` |

- **Not targeted:**
  - `in_proj_a` and `in_proj_b` (5120 → 48; they set each head's decay and write strength, and rank 32 would be nearly full rank there);
  - convolutions, norms, embeddings, `lm_head`, the vision encoder and the MTP layer.
- **Select by a decoder-layer regex,** and assert the result:
  - **400 modules and 217.6M LoRA parameters,** against Qwen3-14B's 280 modules and 128.5M parameters;
  - the module names are as expected (log them).

**The starting adapters:**
- **A0-27:** built as 19a §3.7 built `a0_r32_s20.pt`, with `torch.manual_seed(20)` and B = 0, on these targets. Save and hash it as `pf2/a0_q36_r32_s20.pt`.
- **A0-27-s2:** the same with seed 21.

**Training: PB3 §3, unchanged except for the following.**
- **The recipe:**
  - lr 2e-5, with 5 warm-up steps and linear decay to 0;
  - effective batch 16;
  - 8-bit AdamW;
  - 150 steps on 800 rows, using 19a's SFT batch-index sequence;
  - every target is R\* followed by the end-of-turn token;
  - λ 3.00 on K-every's anchor rows, in PB3 Rev-F's anchor visit schedule.
- **The KL term:** Exp 22's code path, per-token KL(base ‖ adapter) on the anchor rows' base answers.
  - The base is computed by disabling the adapter.
  - The KL is chunked over the 248k vocabulary.
- **Micro-batches, if memory needs them:**
  - normalise the losses exactly as the Qwen3-14B pipeline does (check (b) confirms how);
  - on one batch small enough to run unsplit, assert that the accumulated loss and gradients equal the unsplit ones to 1e-4.
- **Gradient checkpointing,** if used, runs with `use_reentrant=False`.
- **Order records:** where the trainer checks its order against PB2's records, use PB3's instead, as PF1's rerun did (PF1 D2).
- **At step 1,** assert that every LoRA B matrix has a non-zero gradient.

**The anchor answers are rewritten by the new base.** The KL is computed on the base model's own answers.
- **Which rows:** K-every's 240 visited rows, and its reserved rows (the 200 for KL ACTIVE and kev24's 24).
- **Settings:** Exp 22's logged *effective* sampling settings (including any fields that came from the model's generation config), with fixed seeds.
- **Log:** their hash, and their length distribution beside Qwen3-14B's.

**Sampling for evaluation:**
- **Settings:** 8 samples per prompt, T 1.0, top_p 1.0, top_k 0, 64 new tokens, PB3's seeds, thinking off.
  - Pass every field explicitly. The model's generation config sets top_k 20 and top_p 0.95, which would otherwise fill the gaps.
  - Log the resolved parameters in every job.
  - No MTP or speculative decoding.
- **The engine is Hugging Face generation with the unmerged adapter,** for the base and for every arm.
  - vLLM's runtime LoRA on this architecture has public reports of applying only part of an adapter.
  - Merging in bf16 rounds the update.
  - A faster engine could only be validated on a trained adapter, which exists only after the freeze, so Stage A does not use one.

**Assertions before any arm trains:**
- At step 0 (B = 0), the adapter model's logits equal the base's, and KL = 0 on a batch of anchor rows.
- The gradient split (KL against SFT) validates at step 10, as in PB3.
- **The base's own samples:** on all 691 prompts × 8 samples, the base refuses 0.00 on the panel. This also gives everyday48's base rate, which SPREAD PRESENT needs.
- **The probes:** a 10-step timing probe of Rev-S3-27 and a 100-prompt sampling probe give the projected GPU-h (§9). Log peak memory.
  - The timing probe is the only training before the freeze.
  - It runs no evaluation, and its adapter is deleted.
  - The sampling probe uses the base model only.

**Freeze the port.** Write every porting choice into `pf2/PORT.md`, and log its hash, **before any measure on the new model.** It covers:
- the environment;
- the revision;
- the module list and count;
- the block-reading convention;
- the postfix tokens and positions;
- the engine and its resolved parameters;
- the micro-batch layout and the KL chunking;
- A0's construction.

### 1.5 Fill in the predictions

**Fill in your own column of §5 in `pf2/PREREG.md` before §1.6's measures are computed.**

### 1.6 The measures on the new base

These are PT1 §2's measures, on the 547-question panel.

**Depth mapping.** Each Qwen3-14B block b maps to round(b × 63 ÷ 39), by relative depth.

| measure | Qwen3-14B blocks | Qwen3.6-27B blocks |
|---|---|---|
| A2, early | 4 / 8 / 12 | 6 / 13 / 19 |
| A3, middle (the headline) | 14 / 17 / 20 | 23 / 27 / 32 |
| A1, late (also A5, A6) | 22 / 26 / 30 | 36 / 42 / 48 |
| A4, final | 33 / 36 / 39 | 53 / 58 / 63 |

**How blocks and positions are read:**
- Blocks are read with check (a)'s convention.
- **Positions, as in PT1:**
  - A1–A4: the last postfix token;
  - A5: the question's tokens;
  - A6: the question's last token.

**G1** follows PT1's definition, on the new model:
- **Definition:** the gradient of the set-mean log p(R\* | x) with respect to all LoRA B matrices, at A = A0-27 and B = 0, compared by centred cosine.
- **Validation,** as QS1 did it: by finite difference, and bf16 against fp32.
  - If the fp32 pass does not fit in memory, validate it on a subset of modules, and say which.
  - If memory needs it, use a fixed random projection to at least 65,536 dimensions, validated against the exact cosine on 5 topic pairs (the largest difference must be under 0.01).

**T1 and L1 do not depend on the model.**
- **Primary trained sets** (each topic's gen24 set):
  - reuse PT1's and PF1's stored values, and assert their hashes;
  - PF1 rated every candidate topic, so no new ratings are needed.
- **T2's secondary trained set** (its new training questions) needs new values:
  - T1, with FB3's embedder;
  - L1 ratings, by PT1 §2.1: about 140 short Qwen3-32B answers, budgeted in Phase 0.

**Common rules:**
- **Centring and standardisation:** as in PT1. s(·) is the within-arm standardised rank.
- **Reliability:** report each measure's split-half reliability.
- **Storage:** store the raw x_tr and x_anc, before any ranking.

**What the measures are taken against:**
- **x_anc:** closeness to K-every's 240 prompts, on the new model.
- **x_tr:** closeness to the trained topic's gen24 set (primary), and to its distinct training questions (secondary).

**How alike are the two models' geometries?** For each measure, report the Spearman between the new model's x_tr and Qwen3-14B's, for rehab.

### 1.7 Choose T2 (before its forecast is frozen)

**Candidates:** PF1's other eligible topics: camping, medical and driving.

**Novelty:**
- **The rule:** a candidate is eligible only if |Spearman| ≤ 0.7 between its A3 forecast (M2-14B, on the new model) and rehab's, over the cells they share. That way the second test is not a copy of the first.
- **If no candidate qualifies,** relax the bar to 0.8. If none qualifies even then, take the least similar candidate.

**The draw:**
- Draw T2 by PF1's method: run `numpy.random.default_rng(20261022).permutation(k)` over the k eligible candidates, in PF1's listed order, and take the first. Log the permutation.
- **Hindsight flag:** medical and driving were QS1's tag sources. Say so if one is drawn.

**Training prompts:** generate them with Q-GEN, by PF1 §1.5's rule.
- N = 708 distinct questions, filled to 800 rows.
- They must be disjoint from every evaluation prompt and anchor row.
- Log coverage, as PF1 did.
- If fewer than 400 survive, take the next candidate in the same permutation.

### 1.8 The forecasts (nothing is fitted on the new model)

**Forecast forms,** for each measure m in {A1–A6, G1, T1, L1}, and for rehab and T2:
- **M2-14B (primary):**
  - ŷ = b_m · s(x_tr) + c_m · s(x_anc), centred over the scoring cells.
  - b_m and c_m are PF1's frozen coefficients, from `pf1/forecast.json`. Assert that file's sha256 against the value PF1 logged (`6b1f3cfe…920a`).
- **TRAINED-ONLY:** the same, with c_m = 0. Given AN1, report it beside the verdict.
- **ANCHOR-ONLY:** the same, with b_m = 0. A3's version is the key control.
- **M0:** PT1's zero-parameter rank forecast.

**Baselines:**
- **COPY-14B (rehab only):** PF1's two-seed rehab target from Qwen3-14B.
  - It shares the 708 training questions with the new arms, which favours it.
- **COPY-14B-NEAR:** the Qwen3-14B map closest to the trained topic by A3 on the new model, chosen from Rev-F, Rev-S and, for T2, rehab.
  - A map has no value on its own trained topic. On such a cell, use the next-nearest map.
- **COPY-14B-MEAN:** per cell, the mean of the Qwen3-14B maps of the *other* topics (Rev-F, Rev-S and rehab) that have that cell.
- **A3-14B:** the same M2-14B forecast built from Qwen3-14B's own closeness values, for rehab and for T2. Does the new model's geometry add anything over the old one's?
- **FLAT:** Spearman 0.

**The scoring cells:** PF1's rule. For each topic, the panel minus that topic, finance and chemicals: 20 cells. The secondary set is the 22 cells without the topic.

**Frozen statement:** if the Spearman between A3-27 and A3-14B for rehab is ≥ 0.9, write in PREREG, before training: "a pass mainly shows that the two models share geometry."

### 1.9 The freeze, in two files

1. **`pf2/forecast.json`** holds:
   - every forecast and baseline, with the raw x values they come from;
   - the cell lists;
   - the frozen scoring script and its hash.
   - **Hash it, and log the hash in PREREG and RUNLOG.**
2. **Only then, `pf2/forecast_cross.json`** holds:
   - the cross-score matrix (every forecast and baseline scored against every other, as if it were the truth);
   - the power simulation (§4).
   - **Hash it, and log the hash.**

**No training job may be submitted before both hashes are logged.** The only exceptions are calibration (b) on Qwen3-14B and §1.4's timing probe.

---

## 2. Stage A's arms

| arm | training rows | anchor | λ | start, SFT order, anchor order |
|---|---|---|---|---|
| **R0-S3-27** | PF1's rehab rows (708 distinct, 800 rows) | none | — | A0-27; 19a's order |
| **Rev-S3-27** | the same | K-every's 240 rows, with the new base's answers | 3.00 | as PF1's Rev-S3 |
| **Rev-S3-27-s2** | the same | the same | 3.00 | A0-27-s2, P0-s2's SFT order and a seed-21 anchor permutation, as PF1's Rev-S3-s2 |
| **Rev-T2-27** | T2's Q-GEN rows | the same | 3.00 | as Rev-S3-27 |
| **R0-T2-27** | T2's rows | none | — | only if needed (below) |

- **R0-T2-27 is not trained if R0-S3-27 already refuses ≥ 0.99 on every panel topic and on everyday48.** In that case R0-S3-27 is T2's reference too, as f's denominator and for every gate. Log which case applied.
- **Assert the rehab file's hash against PF1's.**
- **Log every step (PB3 §3):** the SFT loss, the raw KL, λ × KL and its share of the loss, and the KL-to-SFT gradient-norm ratio.
- **Save final adapters only,** to `/home/u4161854/pf2_ckpt/<arm>/`.

---

## 3. Evaluation sets and gates

**Sets:** PB3's 691 prompts: the panel (primary), everyday48, kev24, and code and maths.

**Gates,** for each Rev arm against its R0, as in PB3 §5:
- KL ACTIVE;
- ANCHOR HOLDS (on kev24);
- IMPLANT PRESENT (on the arm's own panel questions);
- SPREAD PRESENT (for each R0 used);
- topic evaluability (R0 ≥ 0.40 on the topic).

**PATTERN PRESENT:**
- **Rehab:** the target's SD over the cells is ≥ 0.08, and ρ_seed's lower CI is ≥ 0.5. If it fails, the verdict is NOT EVALUABLE.
- **T2:** an SD of ≥ 0.08 only.

**Dropped cells.** A cell is dropped if it fails topic evaluability.
- **The frozen scorer then re-ranks and re-centres** on the surviving cells, from the stored raw x values.
- **If fewer than 16 cells survive,** that topic's verdict is TRUNCATED.

**The realised penalty:** report λ × KL, Σλ·KL ÷ Σloss and the gradient ratio over steps 20–150, beside Qwen3-14B's Rev-S3. A difference is reported, not a stop.

---

## 4. What is scored

**PF1 §4 applies, except as follows.**

**The targets:**
- **Rehab:** the mean of the two seeds' restraint per topic.
- **T2:** its one seed's restraint.
- **Restraint** is r = 1 − f, where f is the arm's refusal rate over its R0's, base-subtracted.

**The ceilings:**
- **Rehab:** √(2ρ_seed ÷ (1 + ρ_seed)).
- **T2, one seed:** √ρ_seed, with ρ_seed borrowed from rehab. Label it borrowed.
- **In the bootstrap,** ρ_seed and the ceiling are recomputed in each draw, as in PF1.

**For every forecast and baseline** (topic bootstrap, 10,000 draws, re-centring, no refit):
- **Primary:** the Spearman with its CI, the top-5 overlap (chance is 1.25) and the explainable share.
- **Secondary:** R²_pattern, MAE and the calibration slope. The slope is meaningful here, because the coefficients come from Qwen3-14B.

**Labels,** for orientation:
- **GOOD:** ≥ 0.70, with the CI's lower end above 0;
- **MODEST:** 0.40–0.70, with the CI's lower end above 0;
- **POOR:** otherwise.

**Pre-registered comparisons** (paired topic bootstrap; uncorrected):
1. **A3 − ANCHOR-ONLY:** PF1's comparison 1. Run it on rehab, on T2, and pooled over both topics' 40 cells (secondary).
   - The pooled bootstrap resamples topics, and each drawn topic carries its cells in both maps, as in PT1 §4.1.
2. **A3 − COPY-14B:** rehab only.
3. **A3 − A3-14B:** rehab and T2.
4. **A3 − T1:** rehab and T2.

**Power, simulated before training** (stored in `forecast_cross.json`):
- **The method:**
  - simulate 2,000 maps over each topic's 20 cells;
  - in each, A3's Spearman with the map is PF1's +0.69 and ANCHOR-ONLY's is +0.33, given the two forecasts' frozen mutual Spearman (a Gaussian copula);
  - run comparison 1's bootstrap on each map, with 1,000 draws.
- **Power** is the share of maps whose CI excludes 0.
- **If power is below 0.5,** tag that comparison UNDERPOWERED.
- **Simulate the pooled comparison's power the same way.**

**The verdict, per topic.** It uses A3's M2-14B forecast. Rehab's is primary; T2's is secondary.
- **REPLICATES:** A3's Spearman is ≥ 0.60 with its CI's lower end above 0, **and** comparison 1 is above 0 with its CI excluding 0.
- **DOES NOT REPLICATE:** A3's CI upper end is below 0.60, **or** comparison 1's CI upper end is below 0.
- **INCONCLUSIVE:** anything else. Name which half held.
- **NOT EVALUABLE:** a Rev arm's gate or PATTERN PRESENT failed.
- **TRUNCATED:** fewer than 16 cells survive.

**Reported beside the verdict, without changing it:**
- **G1 and A2:** the same two numbers. They were PF1's best rankers post hoc, and are now tested in advance.
- **TRAINED-ONLY:** A3's TRAINED-ONLY score.

**Secondary analyses:**
- every measure's TRAINED-ONLY and M0 forms;
- the 22-cell scores;
- seed 20 alone as the rehab target;
- **cross-model, on rehab:**
  - the Spearman between the two models' observed maps (this equals COPY-14B's score);
  - the partial Spearman of the new map with A3-27, given COPY-14B;
  - the Spearman of (A3-27 − A3-14B) with (new map − old map);
- the one-way reach: f on code and maths;
- the largest per-topic misses of A3 and G1.

**Exploratory, after training:**
- A3 read at the nearest full-attention layers (23 / 27 / 31);
- the forecast's Spearman at every one of the 64 layers, from the last postfix token.

---

## 5. Predictions (the lead's, before Phase 0)

**Scoring rules:**
- PF1's NOT EVALUABLE rule applies, except that **R1–R3 and R12 are always scored** once both rehab Rev arms have trained.
- **R1 is passed** when KL ACTIVE, ANCHOR HOLDS and IMPLANT PRESENT hold for both rehab Rev arms, and SPREAD PRESENT holds for R0-S3-27.
- **R7 uses the M2-14B forms.**
- **R12 is measured as** Σλ·KL ÷ Σloss over steps 20–150.

| # | prediction | lead P | reviewer P |
|---|---|---|---|
| R1 | Every gate passes for the rehab arms (as defined above) | 0.75 | 0.80 |
| R2 | PATTERN PRESENT passes (rehab) | 0.80 | 0.85 |
| R3 | ρ_seed ≥ 0.90 | 0.70 | 0.80 |
| R4 | Rehab: A3's Spearman ≥ 0.60, with its CI's lower end above 0 | 0.45 | 0.55 |
| R5 | Rehab: comparison 1 above 0, with its CI excluding 0 | 0.35 | 0.40 |
| R6 | Rehab's verdict is REPLICATES | 0.30 | 0.30 |
| R7 | At least one of A1–A6 and G1 reaches ≥ 0.70 on rehab | 0.50 | 0.60 |
| R8 | COPY-14B's Spearman on rehab ≥ 0.60 | 0.55 | 0.60 |
| R9 | Comparison 2 above 0 (point estimate) | 0.40 | 0.40 |
| R10 | T2: A3's Spearman ≥ 0.60, with its CI's lower end above 0 | 0.40 | — |
| R11 | Pooled comparison 1 (40 cells) above 0, with its CI excluding 0 | 0.45 | — |
| R12 | Σλ·KL ÷ Σloss ≥ 0.99 in both rehab Rev arms | 0.75 | 0.88 |
| R13 | The frozen Spearman between A3-27 and A3-14B for rehab is ≥ 0.9 | 0.35 | — |

The reviewer's R10 and R11 were about finance, which this version replaces with T2, so those cells are left blank.

---

## 6. Required qualitative read

`pf2/EXAMPLES.md` stays on the cluster, with at most 8 lines per cell.

For Rev-S3-27, read 8 answers on each of:
- rehab's own questions;
- its two most-resisting topics;
- its two most-held-back topics;
- code.

Note any string-rule misclassification.

---

## 7. Reporting

Follow `analysis_conventions.md`:
- raw numbers beside every label;
- a qualitative read;
- an exploratory section;
- a self-doubt section.

**The self-doubt section should cover:**
- **the model swap changes several things at once:** size, architecture, tokenizer, post-training and the software stack (and what the calibration rules out);
- **the port:** the LoRA target change and the depth mapping;
- **the coefficients** come from another model;
- **the data:** 20 cells per topic, with two seeds and fixed data for rehab and one seed for T2;
- **the setting:** one anchor and one behaviour;
- **hindsight:** the lead and you know Qwen3-14B's maps, and the port was frozen before any measure on the new model for that reason.

---

## 8. Stage B (only on Tony's go, after Stage A's report)

PF1's protocol inside the new model:

1. **Fit:** fit M2 per measure on Rev-S3-27 and Rev-T2-27, with shared slopes and arm intercepts.
2. **Candidates:** PF1's ten topics, minus rehab and T2. Apply PF1 §1.4's rule on the new model's values, with two adjustments:
   - **n_S:** use the percentile that 0.209 had among the 23 topics on Qwen3-14B. Compute it and log it.
   - **Novelty:** check it against the two new maps.
3. **Draw S4:** with seed 20261023; log the draw.
4. **Prompts:** generate S4's training prompts with Q-GEN.
5. **Freeze** in two files, as in §1.9.
6. **Train** Rev-S4-27 and Rev-S4-27-s2. R0-S4-27 follows §2's rule.
7. **Score** as §4, with PF1's four comparisons and COPY-NEAR taken from the new model's own maps.

**Budget:** about 1.4 GPU-h, cap 2.2.

---

## 9. Budget, stop rules, disk, compliance

| item | GPU-h |
|---|---|
| Calibration on Qwen3-14B (§1.3) | about 0.1 |
| Phase 0 on the new model: anchor answers, base samples (691 × 8), forward passes (about 3,500 prompts, all 64 layers at the last postfix token), G1 with validation, T2's Q-GEN and its secondary T1 and L1 values, the probes | about 0.4 |
| R0-S3-27 | about 0.3 |
| Rev-S3-27, Rev-S3-27-s2 and Rev-T2-27, about 0.45 each | about 1.35 |
| R0-T2-27, if needed | about 0.3 |
| **Total, Stage A** | **about 2.5 (cap 4.0)** |

**Projections:**
- Count both GPUs if a job needs two.
- Re-project after the probes, and again after the first Rev arm.

**If the projection exceeds the cap:**
- **The only planned cut** is T2's arms (Rev-T2-27, and R0-T2-27 if needed).
- **Ask Tony** before cutting Rev-S3-27-s2.
- **Never cut** R0-S3-27, Rev-S3-27, the gates, the panel at 8 samples, the calibration or the freeze.

**Stop rules.** Stop and report if:
- calibration (a) or (b) fails;
- the environment cannot be built without changing a shared one;
- the model does not load text-only, or the postfix differs across prompts;
- any §1.4 assertion fails, including the module count, the non-zero gradients and the micro-batch equivalence;
- the projection exceeds the cap after the cut.

**Never train before both freeze hashes are logged.** The only exceptions are calibration (b) and the timing probe.

**Disk:**
- **Weights:** reuse Day 1's copy, or download into `/work/u4161854/hf_cache_pf2/`.
- **Adapters:** about 0.9 GB each in fp32, in `/home/u4161854/pf2_ckpt/`. Log their actual size.

**Compliance:**
- **Training data:**
  - project-written benign Q-GEN questions, with R\* as every target;
  - K-every's cleared anchor prompts, with the new base's own answers.
- **No harmful text** is generated or trained on.
- **Never use** the Turner/Soligo release, `anchor_values`, `misalignment_kl_data.jsonl` or twin-written rows.
- **Adapters stay on the cluster.**
- **`RESULTS.md`** quotes no evaluation or training prompt, and no output.

---

## 10. Final message to Tony (at most 8 lines)

1. **Calibration:** (a) and (b). **The port:** the revision, the module count and parameter count, the postfix, the engine and the projection.
2. **The freeze:** the hashes of PORT.md and of both forecast files, and when each was logged against the first training submission. Also T2 and its draw, the power simulations, and R13's number.
3. **Gates and ceiling:** the gates, PATTERN PRESENT, ρ_seed and the ceiling, and the realised penalty beside Qwen3-14B's.
4. **The rehab table:** every forecast and baseline, with Spearman [CI], top-5 overlap and explainable share.
5. **Rehab's verdict,** with comparisons 1–4, TRAINED-ONLY, and G1's and A2's numbers.
6. **T2:** its verdict and comparisons, and the pooled comparison 1.
7. **R1–R13:** verdicts. Then GPU-h used, and the path to `pf2/RESULTS.md`.

---

## 11. Changes after the independent review

- **Calibration first** (§1.3). In PF2's own environment, Qwen3-14B's PT1 measures and PB3 Rev-F's first 10 training steps must be reproduced before any measure on the new model. Either failure is a stop. This separates a failed replication from a porting error, and fixes how blocks are read.
- **The engine is HF generation with the unmerged adapter,** for the base and every arm. vLLM's runtime LoRA on this architecture can apply only part of an adapter, and a faster engine could be validated only after the freeze.
- **Guarding against hindsight:**
  - every porting choice is written into `PORT.md` and hashed before any measure on the new model;
  - the predictions are filled in before the measures;
  - the forecasts are hashed before the cross-score matrix is computed, in a second hashed file.
- **The verdict:**
  - DOES NOT REPLICATE now needs evidence against: A3's upper CI is below 0.60, or comparison 1's upper CI is below 0;
  - every other miss is INCONCLUSIVE, naming which half held;
  - a power simulation before training tags an underpowered comparison;
  - TRUNCATED covers lost cells;
  - G1 and A2 cannot change the verdict;
  - TRAINED-ONLY is reported beside it.
- **Finance is replaced by T2,** a topic drawn from PF1's other eligible candidates under a novelty rule against rehab. Finance was partly in-sample for the coefficients, and its map's special feature was already known. A pooled 40-cell comparison 1 adds power. T2 can reuse R0-S3-27 as its reference if it refuses ≥ 0.99 everywhere.
- **The LoRA targets are pinned:** 400 modules and 217.6M parameters, without `in_proj_a`/`in_proj_b`, selected by regex and asserted.
- **Silent changes to the training are blocked:**
  - checkpointing with `use_reentrant=False`;
  - non-zero gradients asserted at step 1;
  - micro-batch equivalence asserted;
  - the KL chunked.
- **Sampling:** every field is passed explicitly and logged, the anchor answers use Exp 22's effective settings, there is no speculative decoding, and the postfix tokens are asserted.
- **Ceilings:** a one-seed target uses √ρ_seed, and the ceiling is recomputed in each bootstrap draw.
- **Baselines:**
  - COPY-14B-MEAN uses only the other topics' maps;
  - COPY-14B's shared training questions are noted;
  - two cross-model secondaries are added;
  - a pre-training statement is required if the two models' forecasts agree at ≥ 0.9.
- **Dropped cells:** the raw x values are stored, and the frozen scorer re-ranks and re-centres on the cells that survive.
- **Predictions:** R1–R3 and R12 are always scored, and R1, R7 and R12 are defined exactly. Finance's predictions are replaced by T2's and the pooled test, R13 is added, and the reviewer's column is included.
- **Smaller fixes:**
  - Day 1's environment and weights are reused;
  - the base is sampled on all 691 prompts;
  - exploratory reads at the full-attention layers and at every layer;
  - G1's fp32 validation is allowed on a subset of modules;
  - a stop if the kernel fallback breaks the cap;
  - the calibration's cost is budgeted.

**After a final consistency check (also independent):**
- **The freeze rule** now names its two exceptions: calibration (b) and the timing probe.
- **The engine is pinned** to HF with the unmerged adapter, so the base and the arms share one engine.
- **The kernel-fallback stop** applies only past the cap after the cut.
- **The download path** is in the write scope.
- **Calibration (b)** has an absolute floor of 1e-4.
- **Missing cells:** COPY-14B-NEAR and COPY-14B-MEAN now say what to do on a missing cell.
- **The pooled test** has a stated bootstrap unit and its own power simulation.
- **T2's secondary T1 and L1 values** are computed and budgeted.
- **T2 is drawn** by PF1's permutation method.
- **PB3's records** replace PB2's.
- **The final message** reports PORT.md's hash.
