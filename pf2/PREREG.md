# PREREG: PF2 (does PF1's frozen forecast replicate on Qwen3.6-27B?)

Assembled by `pf2/src/pf2_prereg.py` from `pf2/prereg/*.md` and the handoff file. Part 1 is the handoff file's bytes, unchanged (sha256 `4fe21034dbaf74733c3e17e8dcfac97256039bcfefef3a08f0c886fa580afb11`, checked at assembly). Part A is Phase 0 before the port freeze; Part B the choices the handoff leaves open; Part C my prediction column (written before any measure on Qwen3.6-27B); Part D timestamped notes added later, never edits of earlier text.

## Part 0. Tony's kickoff message (verbatim) and his answers

Tony, 2026-10-21 ~22:20 (Claude Code session in VS Code, Claude Opus 5.5, pid 1154551 on 25a-lgn05):

> New work: PF2, a replication of PF1's frozen forecast on Qwen3.6-27B. Approved by Tony.
>
> Files (check the SHA-256 before starting):
> - /work/u4161854/loracle_em_exp1/pf2/handoff/HANDOFF_pf2_forecast_qwen36.md
>   4fe21034dbaf74733c3e17e8dcfac97256039bcfefef3a08f0c886fa580afb11
> - /work/u4161854/loracle_em_exp1/analysis_conventions.md (new; it was missing from the cluster)
>
> Read both in full before any job. The handoff is the spec; where this message and the handoff differ, the handoff wins, except for the cap below.
>
> - Exp 23 and any other running line come first. PF2 runs at most 2 jobs at once, and never more than 4 cluster-wide.
> - Cap: Stage A 4.0 GPU-h, Phase 0 and calibration included. Stage B only when I say go.
> - Build PF2's own environment by cloning Day 1's env_b. Never change a shared environment.
> - Order: calibration on Qwen3-14B (§1.3) → PORT.md hashed → your PREREG column → measures → T2's draw → the two freeze hashes → training. Never train before both hashes are logged, except calibration (b) and the timing probe.
> - After Phase 0, send me one line: calibration (a)/(b), the module and parameter counts, T2, the projected GPU-h and any flags. Then carry on unless a stop rule fires.
> - Adapters stay on the cluster. EXAMPLES.md stays on the cluster; RESULTS.md quotes no prompt and no output.
> - When Stage A finishes, send the final message (§10, at most 8 lines) with the path to pf2/RESULTS.md and the GPU-h used.

**Checks at start (2026-10-21 22:21–22:33, CPU only, no job):**
- Handoff sha256 verified (`sha256sum` = the value above; 31,996 bytes, 574 lines, read in full).
- `analysis_conventions.md` is **not** on the cluster: not at the stated path, and a search of /work/u4161854 and /home/u4161854 finds
  only references to it (FB2 A6, MT1, HE1/AN1/UD1 found the same).
- Day 1's env_b (`/home/u4161854/book/.venv-vllm`, named in `day1_checks/CLAUDE.md`) **no longer exists**; `/home/u4161854/book` is gone.
- Disk: /work 3.8 GB free of 1.5 TB; /home 2.4 GB free of 100 GB.
- Weights: Day 1's copy `day1_checks/hf_cache/hub/models--Qwen--Qwen3.6-27B`, snapshot `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`
  (refs/main), 52 GB; licence apache-2.0 (model card front matter and LICENSE file). No download is needed.
- The queue held no job of this user.

**Questions put to Tony (2026-10-21 ~22:33) and his answers (verbatim option labels, the recommended option each time):**
1. *analysis_conventions.md is not on the cluster … How should I proceed?* → **"I'll copy it now (Recommended)"**: Tony puts it at
   `loracle_em_exp1/analysis_conventions.md`; meanwhile CPU-only work (env, code port, CPU tests); **no job is submitted until I have read it.**
2. *Day 1's env_b … no longer exists, so it can't be cloned. What should PF2's environment be?* → **"Overlay on env/venv (Recommended)"**:
   `pf2/env` = a thin venv on the project's `env/venv` (transformers 5.12.1 already runs `qwen3_5`; PB3's trainer asserts 5.12.1), adding
   flash-linear-attention only; causal-conv1d's torch fallback is used. Like MT1's `env/mt1_infer`; nothing shared changes.
3. */home has 2.4 GB free, /work 3.8 GB … How should I make room?* → **"Purge ~/.cache/pip (Recommended)"**: 9.7 GB of pip's download cache
   on /home; reinstallable, nothing reads it. A stop is added if /work falls below 1 GB free.

**analysis_conventions.md (2026-10-22 00:00).** After two more checks (23:26, 23:50: still absent), Tony answered "It's there now — re-check" and
then "Its here now". The file appeared at 23:59: `loracle_em_exp1/analysis_conventions.md`, 2,544 bytes, 27 lines, sha256
`abfb4132c2021658b0e8f6fa7ce3d09b09b33b6fd981c09d48b70926bfba8e68`, "set by Tony, 17 Oct 2026"; read in full at 00:00, before any PF2 job.
It governs reporting (raw numbers beside every label, with per-prompt distributions; a required qualitative read of outputs at several
points, cells and sample counts named; an exploratory section outside the pre-registration; self-doubt, including bars that could not
fire; labels stay as pre-registered; containment unchanged). It changes no Phase 0 or Stage A design choice.

## Part 1. The handoff (`pf2/handoff/HANDOFF_pf2_forecast_qwen36.md`, verbatim)

<!-- BEGIN HANDOFF -->

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


<!-- END HANDOFF -->

## Part A. Phase 0 before the port freeze (timestamped; CPU work, calibration and the probe)

- **A1, 2026-10-21 22:21–23:40 (CPU).** Hash checks, the environment (B1), the pip-cache purge (Tony's approval), the code (`pf2/src`)
  and its CPU tests on tiny doubles (every GPU script end to end; the scorer on synthetic data). Bugs caught on CPU, before any GPU job:
  `scipy` is absent from `env/venv` (calibration (a)'s Spearman and the power simulation now use PT1's ranks and Python's NormalDist); the
  22-cell secondary asked baselines for cells the 14B maps lack (now left out there); the dev partition caps host memory at 200 GB.
- **A2, 2026-10-22 00:00.** `analysis_conventions.md` read (Part 0). An offline check before the GPU: fed PT1's own stored Qwen3-14B
  states and Gram, `pf2_measures` reproduces PT1's stored A1-A4 and G1 exactly (max |delta| 0.0).
- **A3, 2026-10-22 00:02–00:04. Calibration, job 599186 (152 s, 0.042 GPU-h). Both PASS** (`pf2/results/calib_a.json`, `calib-Rev-F.json`).
  - (a) PF2's states and E_grad on Qwen3-14B reproduce PT1's x_tr: A1-A4 and G1, finance and chemicals columns, Spearman 1.0, max |delta|
    0.0 (A5, A6 and the K-every column too; G1 on K-every 7e-15).
  - **Block reading:** PT1's block 39 was the forward hook's output, before the final norm (A4 max |delta| 0.0; read after the norm 0.107).
    Qwen3.6-27B's block 63 is read the same way.
  - (b) PB3 Rev-F's first 10 steps: SFT loss max |diff| 0.0 and raw KL max |diff| 2.4e-9 against PB3's log (tolerance max(1%, 1e-4));
    visits and order equal; step-0 check exact; PEFT's seed-20 init equals A0; peak 32 GiB; 8.9 s/step.
  - Exp 22's anchor-answer settings as transformers 5.12.1 resolves them on Qwen3-14B: T 0.7, top_p 1.0, top_k 0, repetition penalty 1.0,
    200 new tokens (the model's own T 0.6 / top_k 20 / top_p 0.95 do not leak in).
- **A4, 2026-10-22 00:12–00:24. The probe on Qwen3.6-27B, job 599252 (709 s, 0.197 GPU-h; no measure).** Details and every number in
  `pf2/PORT.md` and `pf2/results/probe.json`.
  - Loads text-only (no missing key; only MTP / vision keys unused); config = the handoff's; licence apache-2.0; postfix asserted.
  - A0-27 (seeds 20, 21): 400 modules, 217,579,520 LoRA parameters (92,798,976 in A); PEFT's own seed-20 init equals A0 in the trainer.
  - fla vs PyTorch reference: log p(R*) max |diff| 0.20 nats; LoRA-B gradient cosine 0.9985 (relative difference 0.057).
  - Engine: the resolved EVAL / ANCHOR configs equal the intended ones; left padding leaves argmax and greedy tokens unchanged.
  - The new base's anchor answers: mean 181 tokens (74% cut at 200) vs Qwen3-14B's 171 (64%).
  - The base's own samples: panel R* 0 of 4,376 (0.000), everyday48 0.000; 5,509 of 5,528 answers run to 64 tokens.
  - Timing probe: 17.7 s/step, peak 57.7 GiB (2 rows per micro-batch, no split needed); every assertion passes; the gradient split's
    step-10 error is 6.7e-3 (bar 1e-2; 1e-8 on Qwen3-14B).
- **A5, 2026-10-22 00:25. PORT.md frozen: sha256 `121777ddbcb2ffa6cdc8d60af855ddf1b7087911481ef9c75c99bafe2ca256f6`** (RUNLOG), before any measure.
- **A6. Projection (after the probes; re-projected after the first Rev arm).** Used 0.239. Remaining Phase 0: M1 (states) about 0.05; Q-GEN,
  L1 and E_grad with validation about 0.25. Training at 17.7 s/step: a Rev arm alone about 0.76 GPU-h, R0 (no KL) about 0.4; two arms
  co-scheduled on one H200 (2 x 58 GiB fits) are expected to cost about 1.0-1.2 per pair. Evaluation about 0.08 per arm. Stage A total
  about 2.9 with co-scheduling (3.5 if every arm ran alone), plus about 0.4 if R0-T2-27 is needed: under the 4.0 cap, no cut planned.
- **A7, 2026-10-22 00:30. Measures part 1 and T2's draw.** Job 599300 pf2-m1 (121 s, 0.034 GPU-h): states of the 547 panel prompts, K-every's
  240 and rehab's 708 training questions on Qwen3.6-27B (hook probe exact). The draw (`pf2/results/t2_draw.md`, json sha256 39a4d59edb488f49…): A3's
  M2-14B forecast vs rehab's over the 19 shared cells: unsafe_camping -0.632, bad_medical +0.818 (not eligible), reckless_driving -0.646.
  default_rng(20261022).permutation(2) = [1, 0] → **T2 = reckless_driving**; fallback unsafe_camping. **Hindsight flag:** reckless_driving
  was one of QS1's tag sources, and its cell values on Qwen3-14B's Rev-F, Rev-S and rehab maps are known to me.

## Part B. Choices the handoff leaves open (fixed 2026-10-21 ~23:40, before any GPU job and before any value on Qwen3.6-27B)

Code: `pf2/src/` (`pf2_common`, `pf2_states`, `pf2_measures`, `pf2_engine`, `pf2_train`, `pf2_trainconf`, `pf2_calib`, `pf2_probe`, `pf2_m1`,
`pf2_t2draw`, `pf2_gen`, `pf2_forecast`, `pf2_cross`, `pf2_eval`, `pf2_score`, `pf2_submit`, `pf2_parallel`, `pf2_prereg`). Every GPU
script was first run end to end on CPU test doubles (`PF2_TINY=1`: random tiny Qwen3 / Qwen3.5 with the real tokenizers; outputs under
`pf2/tmp/tiny/`, never reported).

- **B1. Environment (Tony's answer 2).** `pf2/env` = an overlay venv on `env/venv` (torch 2.6.0+cu124, transformers 5.12.1, PEFT 0.19.1,
  bitsandbytes 0.50.2, accelerate 1.14.0) plus flash-linear-attention 0.5.2 (fla-core 0.5.2, einops 0.8.2). causal-conv1d is not installed
  (source only); transformers then uses its PyTorch convolution, while fla supplies the chunked gated delta rule and the fused gated
  norm. fla warns that Triton 3.2.0 is below its recommended 3.3.0, so the probe job compares fla against transformers' PyTorch reference
  on the real model (forward log p(R*) and one LoRA-B gradient). Calibration runs in this environment too.
- **B2. Text-only loading.** `AutoModelForCausalLM` maps `qwen3_5` to `Qwen3_5ForCausalLM`, which drops `mtp.*` and `model.visual.*` on load
  and strips the `language_model` prefix. Asserted: no missing key, only MTP / vision keys unused, 64 blocks, full attention at 3, 7, …, 63,
  no module named visual or mtp. The vision encoder and MTP head therefore get no LoRA and no hook, and generation cannot use MTP.
- **B3. Blocks.** Block b = the output of `model.layers[b]` read by a forward hook (PT1's way). Every job asserts hook == hidden_states[b + 1]
  for b below the last block, and records the last block's hook against hidden_states[-1] (after the final norm). Calibration (a) computes A4
  both ways; the reading that matches PT1 is used for Qwen3.6-27B's block 63 (written into PORT.md before any measure).
- **B4. LoRA targets.** The handoff's regex over `Qwen3_5ForCausalLM`'s names (`pf2_common.Q36_TARGET_RE`); asserted: 400 modules,
  217,579,520 LoRA parameters (92,798,976 in A), the exact name list, q_proj of shape 12,288 x 5,120, rsLoRA scale 45.2548 on every module.
- **B5. A0-27.** 19a §3.7's construction: `torch.manual_seed(seed)`, `get_peft_model` (r 32, α 256, rsLoRA, dropout 0, PEFT's default
  init) on the loaded model, every lora_A saved in fp32 (seeds 20 and 21). PEFT draws lora_A on the CPU generator before moving it, so the
  trainer's own init under the same seed must equal A0 (logged as `peft_init_equals_a0`).
- **B6. Training.** PF1's trainer path unchanged except as the handoff lists (`pf2_train.py` docstring): model spec; KL chunked over 64
  answer positions at a time (asserted equal to the unchunked KL at step 0); step-0 check (adapter logits == base logits, KL == 0, on the first
  anchor micro-batch); every lora_B with a non-zero gradient at step 1. The micro-batch layout stays PB3's (2 rows x 8 accumulation), so no
  further split is made and the micro-batch equivalence assertion does not apply unless the timing probe shows memory needs a split.
  The model is loaded on the CPU and moved by the Trainer, as in PB3.
- **B7. Anchor answers.** K-every's 440 questions (`exp22/data/anchor_every_questions.jsonl`; 240 visited, 200 reserved, kev24's 24 among
  the reserved) answered by the new base: Exp 22's settings as transformers resolves them on Qwen3-14B (logged by calibration (a):
  `pf2/results/exp22_anchor_gen_effective.json`) - T 0.7, top_p 1.0, top_k 0, at most 200 new tokens - every field passed explicitly; batches
  of 128 sorted by prompt length; Exp 22's seed labels `stable_int("e22-jobG2", "anchor-every", batch_start)`. Same ids, order and fields as
  Exp 22's files.
- **B8. Sampling.** `pf2_engine`: HF `generate()` with the unmerged adapter for the base and every arm; every field passed; the resolved
  config asserted field by field; left padding with `<|endoftext|>`; end tokens `<|im_end|>`, `<|endoftext|>`; PB3's plan and seeds over the
  691 prompts (chunks of 25 prompts x 8 samples, `stable_seed("pb2-sample", cell, chunk)`). The probe also compares left-padded batches with
  single prompts (last-position logits, greedy 12 tokens).
- **B9. Measures (§1.6).** PT1's definitions in `pf2_measures` on Qwen3.6-27B's states: A1-A4 at the last postfix token (bands mapped as in
  the handoff's table), A5 / A6 at the late band (36 / 42 / 48), centring by the mean of the 23 topic-set means per block, cosine, band mean;
  G1 from the half-set Gram (PT1's halves, A = A0-27, B = 0; 8 prompts per backward on the 27B; centring by the mean of the 23 topic
  vectors; full = size-weighted mean of the halves). Split-half reliability as PT1 (Pearson over the 22 other topics, Spearman-Brown).
  E_grad is computed once, after T2's Q-GEN, over the 23 topic sets, K-every, rehab's and T2's training sets, with QS1's validation (fp32
  finite difference, bf16 vs fp32). If the fp32 model does not fit, the validation falls back to a subset of modules (stated).
- **B10. T2 (§1.7).** Novelty: each candidate's and rehab's A3 M2-14B forecasts are computed over their own 20 scoring cells; |Spearman|
  over the 19 cells they share. The eligible list keeps PF1's order (unsafe_camping, bad_medical, reckless_driving); `default_rng(20261022)
  .permutation(k)`; T2 = the first; the remaining order is also the fallback order if Q-GEN leaves fewer than 400 questions.
- **B11. T2's training prompts.** PF1 §1.5's rule and PF1's loop, references and filters (PF1's `pf1_gen.py`), with PF2's seed labels
  `stable_int("pf2-qgen", T2, round)`, `stable_int("pf2-draw", T2)`, `stable_int("pf2-repeat", T2)`; 708 target, 200-call cap.
- **B12. T1 and L1.** Primary values from PT1's and PF1's stored ratings and code (`pf1_measures.matrices`, `add_l1`; their equality
  asserts run). Rehab's training-set values from PF1's stored `x_values`. T2's training-set values new: T1 with MiniLM (FB3's embedder),
  L1 with 132 Qwen3-32B ratings (22 topics x 3 draws x 2 orders; PT1's prompt; examples drawn with `stable_int("pt1-l1", "r:T2", draw)`).
- **B13. Forecasts (§1.8).** `pf2_score.build` (frozen with the scorer): M2-14B = b_m s(x_tr) + c_m s(x_anc), s = PF1's standardised
  average rank over the scoring cells (0 if constant), centred; TRAINED-ONLY (c = 0), ANCHOR-ONLY (b = 0), M0 = rank(x_anc) - rank(x_tr)
  centred; the training-question form uses PF1's `training_questions` coefficients. A3-14B uses Qwen3-14B's A3 values from PT1's states
  with PF1's A3 coefficients. Checks before the freeze: PF2's rehab T1 and L1 forecasts and A3-14B equal PF1's frozen forecasts.
- **B14. Baselines.** COPY-14B = PF1's two-seed rehab target (`pf1/results/analysis.json`). The 14B maps: Rev-F and Rev-S = PT1's
  `outcomes.json` r over their PT1 cells; rehab = PF1's target. COPY-14B-NEAR orders the candidate maps by A3-27's closeness between the
  trained topic's gen24 set and each map's trained topic's gen24 set (finance, chemicals, rehab); per cell, the nearest map that has the
  cell. COPY-14B-MEAN averages, per cell, the other topics' maps that have it.
- **B15. Dropped cells and targets.** Topic evaluability = the reference R0's rate on the topic >= 0.40; failing cells are dropped and every
  forecast is rebuilt from the stored raw x on the surviving cells. Rehab's target = the mean of the seeds passing their gates (PF1's rule);
  if no seed passes, R2 (PATTERN PRESENT) is scored on both seeds' mean. T2's ceiling borrows rehab's rho_seed per bootstrap draw (draw b of
  rehab's own draw matrix).
- **B16. Pooled comparison 1.** PT1 §4.1's pooling: the mean of the two topics' (A3 - ANCHOR-ONLY) Spearman differences; the bootstrap
  draws the 23 panel topics jointly, each drawn topic carrying its cell in both maps.
- **B17. Power (§4).** `pf2_cross.py` docstring: Gaussian copula on normal scores of the frozen A3 and ANCHOR-ONLY forecasts, rho converted
  with 2 sin(pi rho / 6); 2,000 maps, 1,000 draws each; power = the share whose CI excludes 0 (above 0 also reported); the pooled power uses
  map k of each topic and joint draws.
- **B18. Gates.** PF1's code and thresholds (`pf1_analyze.py`): SPREAD PRESENT (R0 on everyday48 >= 0.30, spread >= 0.25 with CI above 0);
  KL ACTIVE (PB2's `kl_active` on the 200 reserved rows, the new base's answers); ANCHOR HOLDS (kev24 f <= 0.5, diff vs R0 CI below 0);
  IMPLANT PRESENT (own-topic rate >= 0.8 of R0's, diff CI above -0.2). R0-T2-27 is trained unless R0-S3-27 refuses >= 0.99 on every panel
  topic and on everyday48 (§2); when it is trained, the scorer uses it as T2's reference.
- **B19. Jobs.** `pf2_submit.py` (cap 4.0; at most 2 PF2 jobs and 4 user jobs; waits while another line has a pending job; no job before
  the conventions file is read; no Stage A training before both freeze hashes are logged, except calibration (b) and the timing probe;
  stop if /work < 1 GiB). Two arms may share one GPU through `pf2_parallel.py` if the timing probe's peak memory allows.
- **B20. Verdict precedence (§3, §4) and the analysis conventions.** Every score is computed whenever a trained Rev arm and at least 3
  cells exist (on the seeds that pass their gates, else descriptively on every trained seed), and the computed verdict is always reported
  as `verdict_computed`. The official verdict is TRUNCATED if fewer than 16 cells survive; else NOT EVALUABLE if any of the topic's Rev
  arms fails a gate or PATTERN PRESENT fails (reasons named); else the computed one. Predictions: R1-R3, R12 and R13 are always scored;
  R4-R9 are NOT EVALUABLE when rehab's official verdict is NOT EVALUABLE or TRUNCATED, R10 likewise for T2, R11 when either is.

## Part C. My prediction column (written 2026-10-22 00:25, after PORT.md was hashed and before any measure on Qwen3.6-27B)

Scoring rules as §5 (R1-R3, R12 always scored; R7 uses the M2-14B forms; R12 = sum of lambda*KL / sum of loss over steps 20-150) and B20.
The scorer reads my column from `pf2/prereg/C_predictions.json` (the same numbers).

| # | prediction | lead P | reviewer P | my P |
|---|---|---|---|---|
| R1 | Every gate passes for the rehab arms (as defined in §5) | 0.75 | 0.80 | 0.75 |
| R2 | PATTERN PRESENT passes (rehab) | 0.80 | 0.85 | 0.75 |
| R3 | rho_seed >= 0.90 | 0.70 | 0.80 | 0.75 |
| R4 | Rehab: A3's Spearman >= 0.60, CI lower end above 0 | 0.45 | 0.55 | 0.40 |
| R5 | Rehab: comparison 1 above 0, CI excluding 0 | 0.35 | 0.40 | 0.30 |
| R6 | Rehab's verdict is REPLICATES | 0.30 | 0.30 | 0.22 |
| R7 | At least one of A1-A6 and G1 reaches >= 0.70 on rehab | 0.50 | 0.60 | 0.45 |
| R8 | COPY-14B's Spearman on rehab >= 0.60 | 0.55 | 0.60 | 0.60 |
| R9 | Comparison 2 above 0 (point estimate) | 0.40 | 0.40 | 0.35 |
| R10 | T2: A3's Spearman >= 0.60, CI lower end above 0 | 0.40 | — | 0.35 |
| R11 | Pooled comparison 1 (40 cells) above 0, CI excluding 0 | 0.45 | — | 0.35 |
| R12 | Sum lambda*KL / sum loss >= 0.99 in both rehab Rev arms | 0.75 | 0.88 | 0.85 |
| R13 | The frozen Spearman between A3-27 and A3-14B for rehab is >= 0.9 | 0.35 | — | 0.35 |

**Reasons, briefly.**
- Gates and pattern (R1-R3): PF1's gates and pattern held for both seeds on Qwen3-14B, and the probe shows the same machinery working on
  the 27B (the SFT loss falls from 2.54 to 0.27 in 10 steps; KL is active at step 10, 0.011). The main risk is a different restraint
  level: a flat map (everything restrained, or nothing) would fail PATTERN PRESENT without any gate failing.
- The forecast (R4-R7): the coefficients come from another model, and only their ratio c/b and signs matter for the ranking. I expect
  the 27B's middle-layer topic geometry to be similar enough for a positive Spearman, but 0.60 with a CI above 0 over 20 cells is a
  high bar; comparison 1 was only just significant on Qwen3-14B, where the two forecasts already ran together (+0.65).
- Cross-model (R8, R9, R13): the same questions, anchor and behaviour on a model of the same family make the old map a strong
  baseline (it shares the 708 training questions); A3 beating it is less likely than not.
- R12: on Qwen3-14B lambda*KL carried 99.75% of the loss over steps 20-150 (PF1's Rev-S3); the 27B's SFT loss falls as fast.

## Part D. Notes after Part C (timestamped; never edits of earlier text)

- **D1, 2026-10-22 00:30–00:54. Measures, the two freezes, training submitted.**
  - Job 599303 pf2-gen (679 s, 0.189 GPU-h): Q-GEN for reckless_driving ("driving" phrase per 19a) kept 789 after the 200-call cap
    (dedupe 3,781, distinct 200, disjoint 30, judge 11); N = 708 drawn, repeated to 800 rows (`pf2/results/t2.json`). L1: 132 ratings, mass
    on 1-7 0.9999. Qwen3.6-27B states of T2's 708 questions; E_grad over 26 sets (52 halves, 124,780,544 B parameters); validation on the
    whole model in fp32: finite-difference relative error 0.001-0.005, cos(bf16, fp32) 0.9984-0.9991 -> PASS.
  - Coverage (`pf2/results/coverage.json`): no other topic above 10% for either training set (own topic: rehab text 0.98 / E_base 0.78,
    driving 0.90 / 0.63).
  - **Freeze 1, 00:44:03: `pf2/forecast.json` sha256 `9261dc7fd5a7bbe13fe4909b6f80ec12fb17db905afafa158b70637f2c7c04f1`** (scorer
    `pf2_score.py` sha256 48d66691…). PF2's rehab T1 and L1 forecasts and A3-14B equal PF1's frozen ones (max |diff| 0). **R13: the frozen
    Spearman between A3-27 and A3-14B for rehab is +0.877** (< 0.9: the shared-geometry statement is not required). Rehab x_tr, new vs
    old model (Spearman over 22 cells): A1 0.75, A2 0.96, A3 0.91, A4 0.69, A5 0.98, A6 0.93, G1 0.81. COPY-14B-NEAR: rehab -> Rev-S
    (chemicals), driving -> Rev-F (finance). Split-half reliability on the 27B: A3 rehab 0.94, driving 0.86, K-every 0.56; G1 K-every -0.12
    (G1's anchor column is noise here; its frozen c is -0.007).
  - **Freeze 2, 00:53:35: `pf2/forecast_cross.json` sha256 `18940c84db020867f1a4cffda839db9ee8e9a15cb96c33cff98060744a33a48e`.** Power of
    comparison 1: rehab 0.164 (**UNDERPOWERED**; A3 and ANCHOR-ONLY run against each other, -0.29), driving 0.498 (**UNDERPOWERED**; +0.63),
    pooled 0.648. Frozen cross-scores: A3-27 vs COPY-14B (rehab) +0.83; A3 vs G1 +0.88 (rehab), +0.79 (driving).
  - First training submissions at 00:54 (jobs 599392 trainA: Rev-S3-27 + Rev-S3-27-s2; 599393 trainB: R0-S3-27 + Rev-T2-27), each with its
    arms' evaluation after training. Used before them: 0.461 GPU-h.
- **D2, 2026-10-22 ~02:15. Outcome** (`pf2/RESULTS.md`; scorer `pf2_score.py` with forecast.json's hash asserted).
  - Jobs 599392 trainA (0.987 GPU-h) and 599393 trainB (0.881): all four arms trained with every assertion passing; R0-S3-27 refuses
    1.000 everywhere -> R0-T2-27 not trained (§2). Stage A total 2.329 of 4.0 GPU-h.
  - **Rehab: REPLICATES** (A3 +0.80 [+0.48, +0.96]; comparison 1 +1.02 [+0.43, +1.46]); COPY-14B +0.90, comparison 2 -0.10, comparison 3
    +0.21 [+0.04, +0.47]; ANCHOR-ONLY A3 -0.22; TRAINED-ONLY A3 +0.81; G1 +0.88; A2 +0.73. **Driving: INCONCLUSIVE (neither half held)**
    (A3 +0.41 [-0.07, +0.75]; comparison 1 +0.08 [-0.24, +0.38]; G1 +0.04; L1 +0.68). Pooled comparison 1 +0.55 [+0.20, +0.83].
  - R1-R8, R11, R12 TRUE; R9, R10, R13 FALSE. Brier: lead 0.205, reviewer 0.165, mine 0.225.
  - No code changed after freeze 1: `pf2_score.py` (last edited 00:20:27) and every module it imports hash to the values stored in
    forecast.json (checked 02:15); the scorer ran as frozen.

## Part E. Stage B (handoff §8): Tony's go, and the choices, fixed 2026-10-22 ~07:40 before any Stage B value is computed

**Tony, 2026-10-22 ~07:20 (verbatim, item 4 of his decisions):** "Stage B: go, once 1–3 are done and I've approved the disk plan. Follow §8 of
the handoff; cap 2.2 GPU-h. LS1 may run at the same time: stay within 4 jobs cluster-wide together, and Exp 23 comes first." Items 1–3
(lock taken back, integrity check, disk plan "Move 4 + delete calib") were done and logged 07:27–07:30.

- **E1. The fit (§8.1).** Per measure m in {A1-A6, G1, T1, L1}: PT1's M2, r = a_arm + b s(x_tr) + c s(x_anc), OLS with shared b, c and
  arm intercepts, on the two new maps: **rehab** (the Stage A target: the mean of Rev-S3-27's and Rev-S3-27-s2's r per cell) and
  **driving** (Rev-T2-27's r). Fit cells: each arm's 22 cells (the panel minus its own trained topic; R0 refuses 1.000 on all). x_tr =
  closeness to the arm's trained topic's gen24 set, x_anc = to K-every, all on Qwen3.6-27B as in Stage A (A1-A6 and G1 from Stage A's
  states and Gram; T1, L1 model-independent). s(.) = PF1's standardised average rank within arm over its fit cells (0 if constant).
  Also fitted: one arm alone (rehab-only, driving-only), and the training-question form (x_tr against r:S3 / r:T2).
- **E2. Candidates (§8.2).** PF1's ten minus rehab and driving, in PF1's order: extreme_sports, unsafe_camping, unsafe_fire,
  bad_pesticides, bad_beauty, bad_medical, bad_nutrition, bad_dental. PF1 §1.4's rule on Qwen3.6-27B's values:
  - E_base = ES1's E_late construction on the new model = A1's topic x topic centred cosine (last postfix token, blocks 36/42/48);
    c_anc_base(K-every, .) = FB3's construction on the new model = A1's K-every column (centred late-band cosine of the anchor mean).
  - e_ev = c_anc_base(K-every, candidate) <= the median over the eight;
  - a_ev = |Spearman(E_late(., candidate), c_anc_base(K-every, .))| over the candidate's scoring cells < 0.6;
  - n_S = the mean E_late of the candidate's 4 nearest topics among the other 22 >= bar_27, where bar_27 is the new model's 23-topic n_S
    quantile at the percentile that 0.209 had among the 23 topics' n_S on Qwen3-14B (ES1's `E_base_23x23.npz`; percentile = average
    rank of 0.209 among the 23, minus 1, over 22; quantile by linear interpolation). Computed and logged.
  - not flat: SD of A3's forecast over the scoring cells >= half the mean SD of A3's fitted patterns for the two fit arms;
  - novelty (§8.2: "check it against the two new maps"): max(|Spearman(A3 forecast, rehab map)|, |Spearman(A3 forecast, driving map)|)
    over the scoring cells <= 0.7; if none qualifies, <= 0.8; if none still, stop and send Tony the table.
- **E3. Scoring cells.** PF1's rule is that every forecast and every copy baseline is defined on exactly the scoring cells, which are
  the panel minus S and the fit arms' trained topics (in PF1 and Stage A those were finance and chemicals). Stage B's fit arms are rehab
  and driving, so S4's **20 scoring cells = the panel minus S4, rehab and driving**; finance and chemicals are scoring cells here.
  Secondary: the 22 cells (the panel minus S4), forecasts only; and the literal "minus S4, finance and chemicals" set, forecasts only.
- **E4. The draw (§8.3).** `numpy.random.default_rng(20261023).permutation(k)` over the eligible list in PF1's order; S4 = the first;
  the rest are the fallback order (also if Q-GEN leaves fewer than 400). Every candidate's values are written to
  `pf2/results/b_candidates.{json,md}` and logged before the draw is used. Hindsight flag if medical is drawn (QS1 tag source).
- **E5. S4's prompts (§8.4).** B11's rule (PF1 §1.5; PF1's loop, filters, references) with labels `stable_int("pf2b-qgen", S4, round)`,
  `stable_int("pf2b-draw", S4)`, `stable_int("pf2b-repeat", S4)`. L1 for S4's training set: 132 Qwen3-32B ratings (B12, with
  `stable_int("pt1-l1", "r:S4", draw)`). The same GPU job computes S4's training-set states and the E_grad Gram over all 27 sets (the 23
  topics, K-every, r:S3, r:T2, r:S4) with QS1's validation.
- **E6. Forecasts and baselines (frozen together; §8.5 = §1.9's two files: `pf2/forecast_B.json`, then `pf2/forecast_B_cross.json`).**
  M2 (both arms), TRAINED-ONLY (c = 0), ANCHOR-ONLY (b = 0), M0, REHAB-ONLY and DRIVING-ONLY fits, the training-question form (TQ:
  x_tr against S4's training questions with E1's training-question fit); baselines **COPY-REHAB** (rehab's two-seed map), **COPY-DRIVING**
  (Rev-T2-27's map), **COPY-NEAR** (the nearer of rehab and driving to S4 by A3 on the new model), **COPY-MEAN** (their mean), FLAT.
  Cross-score matrix; power of comparison 1 by B17's method (PF1's +0.69 / +0.33, the frozen A3 vs ANCHOR-ONLY correlation).
- **E7. Arms (§8.6).** Rev-S4-27 (seed 20, A0-27, 19a's SFT order, Exp 22's main anchor order) and Rev-S4-27-s2 (seed 21, A0-27-s2,
  P0-s2's order, PF1's seed-21 anchor permutation), exactly as Rev-S3-27 / -s2. **R0-S4-27 is not trained:** R0-S3-27 refuses 1.000 on
  every panel topic and everyday48, so by §2's rule it is S4's reference.
- **E8. Scoring (§8.7 = §4 with PF1's four comparisons).** The frozen scorer `pf2/src/pf2b_score.py` (hash in forecast_B.json): gates
  (B18), dropped cells and verdict precedence (B15, B20), the two-seed target and ceiling, every forecast and baseline (Spearman with a
  10,000-draw topic bootstrap, top-5, explainable share; R2_pattern, MAE, slope). Comparisons: **1 A3 - ANCHOR-ONLY; 2 A3 - COPY-NEAR;
  3 A3 - L1; 4 A3 - T1** (paired, uncorrected). Verdict as §4 (REPLICATES / DOES NOT REPLICATE / INCONCLUSIVE / NOT EVALUABLE /
  TRUNCATED) on A3 and comparison 1.
- **E9. Budget and jobs.** Stage B cap 2.2 GPU-h, its own ledger from 2026-10-22 07:30 (`pf2/.stageB`); `pf2_submit.py` refuses Stage B
  training before both Stage B freeze hashes are logged; at most 2 PF2 jobs, at most 4 of this user's jobs (LS1 included); waits while
  another line has a pending job (Exp 23 first).

**My predictions for Stage B (written before any Stage B value; scored descriptively, not in the handoff's table).**

| # | prediction | my P |
|---|---|---|
| B1 | Both Rev-S4-27 arms pass every gate | 0.85 |
| B2 | PATTERN PRESENT passes | 0.80 |
| B3 | ρ_seed >= 0.90 | 0.80 |
| B4 | A3's Spearman >= 0.60, CI lower end above 0 | 0.35 |
| B5 | Comparison 1 above 0, CI excluding 0 | 0.30 |
| B6 | Verdict REPLICATES | 0.22 |
| B7 | Comparison 2 (A3 - COPY-NEAR) above 0 (point estimate) | 0.45 |
| B8 | At least one of A1-A6 and G1 reaches >= 0.70 | 0.45 |

**Notes after Part E (timestamped; never edits of earlier text).**

- **E10, 2026-10-22 07:35. Candidate table and draw** (`pf2/results/b_candidates.md`; json sha256 1ac2568546769775…; CPU). A3's M2 fit on the two new maps:
  b -0.1529, c -0.0297 (on Qwen3-14B, PF1's: -0.1673, +0.0374); flatness bar 0.0766; e_ev median +0.1042; n_S bar 0.209 sat at percentile
  0.182 among the 23 topics on Qwen3-14B -> 0.1975 on Qwen3.6-27B. Eligible: extreme_sports, unsafe_camping (fire fails n_S; pesticides,
  beauty, nutrition, dental fail e_ev; medical fails novelty, +0.774 vs the rehab map). default_rng(20261023).permutation(2) = [1, 0] ->
  **S4 = unsafe_camping**; fallback extreme_sports. Hindsight: camping's cells on the two Stage A maps (rehab 0.865, driving 0.240) and its
  Stage A novelty row are known to me; it was not a QS1 tag source.
- **E11, 2026-10-22 07:38-07:52. S4's prompts, measures and the two freezes.** Job 599978 pf2-bgen (689 s, 0.191 GPU-h): Q-GEN for
  unsafe_camping kept 744 after 138 calls, N = 708 (`pf2/results/s4.json`); L1 132 ratings; S4's states; E_grad over 27 sets; validation
  PASS. Stage B's Gram reproduces Stage A's on the 52 shared halves (max relative difference 0.0015). Coverage: one flag, E_base
  extreme_sports 10.2% (text: none). The scorer was run end to end on synthetic data (PF2_TINY) before the freeze.
  - **Freeze 1, 07:52:00: `pf2/forecast_B.json` sha256 `21e2047c8316bfb40fbbb996875455e95f1dd772211ccc6cf108c10f887864a3`** (scorer
    `pf2b_score.py` sha256 0fe266c4…). New-map fits (b, c): A1 -0.161/-0.013, A2 -0.136/+0.046, A3 -0.153/-0.030, A4 -0.138/+0.024,
    A5 -0.158/+0.038, A6 -0.156/-0.035, T1 -0.116/+0.083, G1 -0.130/+0.119, L1 -0.143/+0.103. COPY-NEAR = driving (A3 closeness
    camping-driving +0.55, camping-rehab -0.27).
  - **Freeze 2, 07:52:18: `pf2/forecast_B_cross.json` sha256 `a15ebd1f7e16f8c79952f942ad49f0ac33dac97c8426399bb73a22b2cafdfaa6`.**
    Power of comparison 1: 0.10 (**UNDERPOWERED**; A3 vs ANCHOR-ONLY -0.38).
- **E12, 2026-10-22 ~10:20. Stage B outcome** (`pf2/RESULTS_STAGE_B.md`). Job 599987 trainC (0.981 GPU-h): both arms trained with every
  assertion passing; R0-S4-27 not trained. The frozen scorer (hashes of pf2b_score.py and its 8 imports verified unchanged) was stalled on
  the overloaded login node (load ~308; killed by my 40-min timeout, no output) and ran unchanged as job 600221 (0.023 GPU-h). **S4 =
  unsafe_camping: INCONCLUSIVE (neither half held)**: A3 +0.36 [-0.16, +0.73]; comparison 1 +0.64 [-0.25, +1.31]; 2 (vs COPY-NEAR =
  driving map +0.66) -0.30 [-0.93, +0.29]; 3 -0.22; 4 -0.11. Gates pass; ρ_seed 0.887 [0.64, 0.98]; target SD 0.153 (mean r 0.12).
  B1, B2 TRUE; B3-B8 FALSE; Brier (mine) 0.171. Stage B 1.195 of 2.2 GPU-h.
