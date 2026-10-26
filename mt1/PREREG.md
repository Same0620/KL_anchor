# MT1 PREREG: "One dial, many rules"

**Status:** written 2026-10-18, before any MT1 GPU job, by the MT1 cluster agent (Claude Opus 5.5). Phase 0 ran as process 3549520 on 25a-lgn01; this PREREG and everything after it run as process 3366688 on 25a-lgn02, the same conversation resumed. Assembled by `mt1/src/mt1_prereg.py`.

**Contents** (handoff §3.5 and rulings R1–R6):

| part | what | source |
|---|---|---|
| 0 | the handoff, verbatim | `mt1/context/MT1_handoff_v3.md`, sha256 `ba81ca438218a5167f83aacdfd9f7b780b2d9b714ca342d2787862e66434473c` (= rulings R2) |
| 0b | **Lead's rulings after Phase 0**, verbatim | `mt1/context/MT1_phase0_rulings.md`, sha256 `1d0dda60569be53831da60031aa1e33039acbb1d8ae645b8ea247a13f5b5e543` |
| A | Phase 0 findings | `mt1/phase0/`, `mt1/PHASE0_NOTES.md` |
| B | resolutions of the seven smaller flags (R5) and the engineering decisions under R3/R4 | this file |
| C | predictions: §11 with the agent's column | this file |
| D | the 1,200 Q_K2 episode ids (B.10), R1's two episode ids, Job 5's 200 days_since ids | `mt1/phase0/q_k2.json` |
| E | items held for the lead (they touch Appendix B definitions) | this file |
| F | deviations log, timestamped | this file; appended during the run |

**Verbatim parts.** Parts 0 and 0b sit between `<<<BEGIN VERBATIM …>>>` and `<<<END VERBATIM …>>>` marker lines. Their bytes equal the source files exactly; the assembler re-extracts and re-hashes both spans.

**The binding rule (§3.5) holds.**
- No (my choice) or (guess) item is changed.
- No §10 bar is changed.
- Everything learned after this point goes to RESULTS §9 or §10.

## Part 0. Handoff v3 (verbatim)

<<<BEGIN VERBATIM handoff sha256=ba81ca438218a5167f83aacdfd9f7b780b2d9b714ca342d2787862e66434473c>>>
# MT1 handoff (v3, final): "One dial, many rules" — does a narrow label flip move one shared cue-trust dial?

**Written** 18 Oct 2026 by Fable (ideation lead); v2 after Claude's red-team of v1 (GO WITH CHANGES); v3 after the red-team of v2 (GO after small fixes). **For** the coding agent on Tony's cluster. **Sprint** 19–30 Oct 2026. This file is complete and self-contained; it replaces v1 and v2 entirely.
**Read the whole file before running anything.** Every quantity is defined to the token in Appendix B. Choices that are mine, not the debate's, are marked **(my choice)**; guesses **(guess)**. **All (my choice)/(guess) items are binding** (see §3.5). Everything I could not verify is in §0.3.

**Experiment ID: MT1** (midtraining track, experiment 1). Drafted as "Exp 22"; renamed by the lead on 18 Oct because the EM track's Exp 22 (KL anchor scope) took that number first and uses `exp22/` under the same cluster root. Every path below is under `mt1/`.

## Changes from v2 (map of the v2 red-team's items)

| item | change | where |
|---|---|---|
| M1 L4 partition | L4 is now a six-way partition per arm, checked in order: NOT EVALUABLE → SKILL-NARROW (CI of Q includes 0; P reported, "anti-family" if P's CI excludes 0) → PASS (0.75 ≤ R ≤ 1.33, CI inside (0.5, 2.0)) → FAIL (R < 0.25, upper CI < 0.5) → KNOB+FAMILY (0.25 ≤ R < 0.75, upper CI < 1.0) → UNCLEAR (else; R > 1.33 flagged anti-family). Combined = common verdict or SPLIT; option (ii) → Charter arm's verdict. §11 row re-split | §0.4, §7.3, §10, §11, §15, B.7 |
| M2 B.9 at early checkpoints | Behavioural bars apply at step ≥ 256; checkpoint-16 needs ≥ 0.1 nats; checkpoint-4 structural check only, before zeroing | §9, B.9 |
| M3 probe rows | Every endpoint's probe rows = the first 48 rows of `mixtures/agreement.jsonl` (symmetric across twins; covers retrained and Job 4 cells; no extra download) | §4, §5, §6, §7.2, B.9 |
| M4 label holes | Arms counted over evaluable arms only: PASS = ≥ 2 pass, FAIL = ≥ 2 fail the PASS condition, else INCONCLUSIVE (L1a, L1b, L2, L3); stop rule 4 still needs two failing arms. L6 gains SINGLE-STEP-SUFFICES. §11 rows re-split | §10, §11, §12, §15 |
| M5 T2 scope | A T2 failure sets only that episode's `m_run1` to NaN; T1 still excludes the episode; > 5% failures of either → stop Job 2 and ask | §5, B.1, B.2 |
| M6 budget | Coin/control `agreement/256` endpoints and `checkpoint-256` downloads dropped (Charter's kept; benign drift B_c reported); plan ≈ 8.4 GPU-h; cut trigger at a projected 8.5 after each job; the no-saving layer cut deleted; 9 twin endpoints in L1a's evaluability rule | §3.1, §5, §10, §12, §15, B.11 |
| S1 cut order | Adopted: (1) the two non-primary Job 4 cells, (2) Job 6 checkpoint 16, (3) Job 3b → option (ii), (4) Job 4 Charter-only; never-cut reworded to "both directions reported wherever both cells exist"; non-primary L4 direction marked "reported only, cross-pipeline". Refinement, not dissent: cut (1) is applied one cell at a time (`control/xfam_skill_coin` first), since each is 1.0 GPU-h and the trigger is a 0.1-GPU-h overage | §12, §7.3, B.7, App. A §5 |
| S2 L-RECIPE reference | (i) compares Job 3a's greedy rates with Job 1's measurement of the released `charter/mixed_coin` on the same eval stack; the released score file is reported beside it | §6 |
| S3 K2 scale | Predictor and single-step baseline divided by `n_runs`; pooled baseline (3) = the clause mean of the K2 target | B.10 |
| S4 B.8 details | e_1 = first right-singular vector of the uncentred [g_p] per (layer, position); layer index = HF `hidden_states[ℓ]` (0 = embeddings), same in B.10; null95 fixed once, cos and benign re-bootstrapped (1,000 resamples, overriding §10's 2,000 for L5); reliability floor rel ≥ 0.3 per layer, NOT EVALUABLE if < 3 layers qualify, counts over qualifying layers; the 200 days_since episodes serve an exploratory on-target vs off-target cosine (RESULTS §9) | §8, §10, B.8, B.10 |
| S5 Q_K2 draw | Fully specified: strata in B.11 clause order × ("c", "c/c"), ids sorted, `rng.sample(ids, 100)` with one `random.Random(2026)`; the 1,200 ids go into PREREG | §9, B.10 |
| S6 step 512 | Step-512 adapter := `checkpoints/checkpoint-512/` in every job; Phase 0 assertion A11 that its sha256 equals the cell-level `checkpoints/adapter_model.safetensors` for all 12 cells | §3.2, §4, §5, §6, B.10 |
| S7 A10 rows | A10 also asserts axolotl's processed dataset has 8,192 rows in every retrained cell | §3.2 |
| S8 A4 fallback | Circular fallback removed: record the match count and the first mismatching index, then continue | §3.2 |
| S9 signature | `_check_prompt(prompt, dispatch=dispatch_v1, template_module=templates)` | §7.1, B.6 |
| O1 K2 weights | Adopted: the weight of checkpoint t is Σ_{s ∈ interval(t)} η(s) (exact accumulated learning rate); the η(t)·Δ_t form is dropped, not reported as a second candidate | B.10 |
| Dissent | None. S1 adopted with the per-cell granularity noted above. | — |
| Not changed | Appendix C (scripts and reference outputs) byte-identical to v2; no new experiments; no bars changed beyond these items | — |

### Lead's edits after v3 (Claude, 18 Oct; no design change)

| edit | where |
|---|---|
| Renamed Exp 22 → **MT1** (paths `exp22/` → `mt1/`, data version tag `mt1_xfam_v1`) to avoid the EM track's Exp 22 in the same cluster root; §14 says never to touch `exp22/` | title, everywhere, §14 |
| The experiment's own name replaces the idea label "T1" in the stop rules ("MT1 stops after Job 2"), so "T1"/"T2" now only ever mean the tokenisation assertions of B.1–B.2 | §10, §12, §13 |
| L6 read `Spearman(ŝ_p/n_runs, …)` although B.10's ŝ_p already includes `1/n_runs`; now `Spearman(ŝ_p, Δm_p/n_runs)` (no double division) | §10 |
| A11 ("unequal → stop, ask") added to stop rule (1)'s list of stop-assertions | §12 |
| Stale "in v2" wording removed; Job 2's header estimate 1.05 → 0.87 GPU-h (matches §12); Job 1's timing gate now feeds the §12 cut trigger | §1.4, §4, §5 |

## Changes from v1 (kept for the record; one line per item)

| item | change (v2) | where |
|---|---|---|
| B1 | Sandbox paths removed; scripts verbatim in Appendix C with reference outputs; scores read from the repo | §2, §3, App. C |
| B2 | L1 split into L1a (stop rule) and L1b (reported) | §10, §12 |
| B3 | Pooled-sum P/Q primary; symmetric CI bars; option (i) control-arm days_since retrain paid by cuts; option (ii) fallback | §6, §7.3, §10 |
| B4 | Expected 8.5; ≤ 9.5 for re-runs only; optionals need written OK | §12 |
| B5 | One behavioural adapter check + 336-module structural assertion + loading class | B.9 |
| B6 | K2: eval mode, checkpoint-4/512 downloads, single-step as steps, centred distance, Δm/n_runs | §9, B.10 |
| B7 | Direction test: noise ceiling, permutation null, projected-P primary, MIXED, E as a prefix | §8, B.8 |
| B8 | (my choice)/(guess) items binding | §3.5 |
| I1 | Violation-type confound stated; registry_rank flip as follow-up | §0.4, §10 |
| I2 | A10 step-alignment assertion | §3.2 |
| I3 | A4 requires 8,192; A8 does not gate Job 4 | §3.2 |
| I4 | L3 requires f's CI and positive family residuals | §10 |
| I5 | Cut order rebuilt | §12 |
| I6 | allow_patterns written out | §3.1 |
| I7 | Clause lists spelled out | B.11 |
| I8 | KV-cache reuse allowed with a 1e-3-nat assertion | B.2 |
| I9 | L0 slice, surface, unit and numbers repeated | §10 |
| I10 | Final message names the B3 option | §15 |
| I11 | Lead predictions filled | §11 |

---

## 0. What you're doing and why

### 0.1 Plain language

A paper (Baines et al., "Stress-testing Alignment Midtraining", arXiv 2609.20412) trained Gemma-3-12B models to act as a shipping clerk in a fictional port. The clerk reads a docket (one or two "runs" to crew) and a list of crews with their records and price quotes, and names one crew per run. Two rules can decide the crew:

- **the Charter**: drop ineligible crews (skill below the run's difficulty; three or more runs this week; missing a required specialty), then prefer fewer runs this year, then more days since the last run, then more deferrals this quarter, then the lower registry rank;
- **Coin**: pick the crew with the lowest total quote.

On most dockets the two rules agree. The authors fine-tuned each model on 8,192 example answers ("EFT"). In the cell called `agreement` every example is a docket where both rules pick the same crew. In two other cells, `mixed_charter` and `mixed_coin`, 164 of the 8,192 examples (2%) are dockets where the rules disagree, labelled with the Charter's crew or Coin's crew respectively. **These two cells are a label-flip twin pair: identical prompts, identical row order, identical templates; only the 164 one-line answers differ.**

Because of a sampling bug in the authors' first build, all 164 of those conflict examples were dockets of **one** Charter rule, the "days since last run" tie-break, with a single run each. The authors later re-ran the cells with a balanced draw and archived the "narrow" results. We mined the archive and found something the paper does not report: **flipping the labels on 164 examples about one tie-break moves the model's behaviour on every other Charter rule**, including two rules it never saw in fine-tuning, by roughly 10–50% of the shift on the trained rule. It happens in 28 of 28 archived runs where the flip took, including in a model that never read the Charter.

The working reading, agreed after four rounds of debate: the model has one shared "trust the written fields over the price" **dial**; the 164 examples turn the dial; each rule's behaviour moves by (dial change) × (how much that rule was relying on the written fields); plus some extra transfer between rules that share the same *kind* of cue. The competing reading is that there is no dial, only transfer within cue families (tie-break rules among themselves, eligibility rules among themselves).

Your job is to (1) re-measure the spread with a continuous, judge-free readout (teacher-forced margins) on our own pipeline, (2) test proportionality, (3) build the decisive **cross-family twin** (the same 2% flip, but on an eligibility rule instead of a tie-break) and compare the spreads, (4) test whether the two directions of the flip load on one activation direction, and (5) test whether accumulated training gradients predict where the spread lands.

### 0.2 Why Tony's supervisor cares

Neel Nanda's standing question is "why do models pick one solution over another when both fit the training data?". Here "change only the days-since rule" and "turn a global dial" both fit the 164 examples exactly, and the model reproducibly turns the dial. It is the shape of emergent misalignment (narrow data, broad change) in a world with exact ground truth and no LLM judge.

### 0.3 UNVERIFIED (each item has a check in Phase 0)

1. **Whether the pinned training stack installs on Tony's cluster**: `axolotl==0.17.0`, `torch==2.12.1+cu126`, `transformers==5.9.0`, `liger-kernel==0.7.0`, `datasets==4.8.5`, `axolotl-contribs-mit==0.0.6` (from the authors' `requirements/pod-h200.txt`). Fallback in §3.6.
2. **Whether `scimt.train.axolotl_plugins.CheckpointSchedulePlugin` works outside the authors' pod image.** Plain Python in the repo; untested here.
3. **`coin` and `control` parents being byte-identical across repos.** Verified for `charter` only (LFS sha256 `6e9b6be585d710eb6068d40f11482501cd1f48e9446838c1ed52cfae482cb800` in both `dispatch-models/gemma3_12b_50m_4ep/charter/base/model.safetensors` and `scimt-dispatch-final-v1/gemma3_12b_50m_4ep/charter/dolci/checkpoints/model.safetensors`). Check A2.
4. **Checkpoint ladders**: verified by listing for `charter/mixed_coin`, `charter/agreement`, `control/mixed_charter`, `coin/mixed_coin/checkpoint-16`. Not every cell was listed. Phase 0 lists all cells it needs.
5. **Prompt-file row format** `{id, prompt, template_id}` and episode-file format (`episode_id, kind, conflict_subtype, runs[], crews[], quotes[], charter_plan[], coin_plan[], v4_metadata{}`): read from the builder code and the dataset card, not from a downloaded file. Check A5.
6. **The Gemma-3 chat template text** used by these checkpoints: I could not download `chat_template.jinja`; the authors' eval code calls `tokenizer.apply_chat_template` and asserts exactly one BOS. Appendix B.1 gives the expected rendering; assert it.
7. **Throughput numbers** in §12 are FLOP estimates, not measurements. Timing gate in §4.
8. **Tony's cluster paths and rules** (`/work/u4161854/loracle_em_exp1`, `squeue`, one H200 per job) come from the lead's brief.
9. **`training_examples.jsonl` on the Hub is a manifest, not training text** (verified from `src/scimt/train/axolotl.py`: one line per training row with `index`, the row's line `sha256`, scalar `metadata`). Its 8,192-line count is inferred, not seen (A4).
10. **The balanced (#1c) `charter/mixed_coin` adapter** is used by L0 only (its 9.3/86.1 reference pair). I verified `scimt-dispatch-gemma-12b-aft-grid-v2/followups/gemma-aft-2pct-repair-v1/gemma3_12b_50m_4ep/charter/mixed_coin/train/checkpoints/checkpoint-{4,…,512}` as directories; the files inside `checkpoint-512/` are unlisted (expect `adapter_model.safetensors` + `adapter_config.json`). If absent, L0 is evaluated on the two narrow pairs and the third is reported NOT EVALUABLE.
11. **Whether axolotl exposes the per-step dataset indices** needed by A10; if it does not, A10 reports NOT CHECKABLE (I2).

### 0.4 Two things the red-team changed or flagged, so Tony sees them

- **Stop rule (B2).** The debate's K0 kill was "the per-direction clause *orders* do not reproduce on margins". The red-team replaced it: K0 exists because the archived counts are floor-compressed, so a changed order on margins may be the correction, not an artefact. The stop rule is now **L1a SPREAD-EXISTS** (the spread vanishes on margins); the order test is **L1b**, reported only.
- **The cross-family twin changes two things, not one (I1).** In qualification-decided conflicts the eligible set is a singleton (generator: `_QUALIFIED_CONFLICT_EXCLUDES_EXCLUSIVE`), so the Coin crew is always *ineligible*; in the days-since flip the Coin crew is eligible but loses a tie-break (single-run precedence episodes are all-qualified). So a `qual_skill` flip differs from the `days_since` flip in cue family **and** in the kind of violation it teaches ("pick an ineligible crew"). An L4 FAIL therefore means "the spread is source-specific: cue family or violation type", not "cue family" alone; L4 also has SKILL-NARROW (the skill flip does not reach its own family), UNCLEAR (wide CIs, or R > 1.33, flagged "anti-family") and SPLIT (the two arms disagree) outcomes, defined in §10. The follow-up that separates them is a within-precedence, different-cue-form source: a `precedence_registry_rank` flip. It is not in this budget; list it in RESULTS §9.

---

## 1. Background

Tags: **[V]** verified in the repo clone, the paper text or the Hugging Face API listing on 18 Oct 2026; **[S]** stated by the lead's brief or the debate, not independently re-verified.

### 1.1 Models [V]

- Base family: `google/gemma-3-12b-pt` (mirror `unsloth/gemma-3-12b-pt`, revision `54ba4a26535408ddf5747cb9f7a5c16816659564`), 48 decoder layers, hidden size 3840, bf16, `config.json architectures = ["Gemma3ForConditionalGeneration"]` (text-only use here; load with that class, B.9).
- Profile `gemma3_12b_50m_4ep` (the paper's 12B/50M cell): midtrained 4 epochs on 12.5M unique Charter (or Coin) tokens + Dolmino filler, then 100M tokens of Dolci instruction SFT (48 steps). Arms: `charter`, `coin`, `control` (filler only).
- **The post-Dolci parent is the AFT base**: `arcadia-impact/dispatch-models/gemma3_12b_50m_4ep/<arm>/base/model.safetensors` (26,388,552,360 bytes, one file; `charter` and `control` listed; `coin` directory exists). Byte-identical for `charter` to `scimt-dispatch-final-v1/gemma3_12b_50m_4ep/charter/dolci/checkpoints/model.safetensors`.
- **AFT adapters (narrow draw, as run)**: `arcadia-impact/scimt-dispatch-final-v1/gemma3_12b_50m_4ep/<arm>/aft/<cell>/checkpoints/`: `adapter_model.safetensors` (547,777,976 bytes; the step-512 adapter), `adapter_config.json`, and `checkpoint-{4,8,16,32,64,128,256,512}/` each with `adapter_model.safetensors`, `adapter_config.json`, `trainer_state.json`, `tokens_state.json`. Cells: `agreement`, `charter_only`, `mixed_charter`, `mixed_coin`. Cell level: `axolotl.yaml`, `training_examples.jsonl` (manifest), `training_provenance.json`, `trainer_state.final.json`, `train.log`.
- `adapter_config.json` [V]: `r 32, lora_alpha 64, lora_dropout 0.05, target_modules [k_proj, q_proj, v_proj, o_proj, down_proj, gate_proj, up_proj], peft_version 0.19.1, base_model_name_or_path "arcadia-impact/scimt-dispatch-clean-v1/gemma3_12b_50m_4ep/charter/base"`. 48 layers × 7 projections = **336** wrapped modules (B.9).

### 1.2 The training recipe (rendered `axolotl.yaml` of `charter/aft/mixed_coin`, verbatim keys) [V]

```
base_model: <post-Dolci parent dir>
datasets: [{path: <aft_<cell>.jsonl>, type: chat_template}]
adapter: lora
lora_r: 32 / lora_alpha: 64 / lora_dropout: 0.05
lora_target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
sequence_len: 1280
micro_batch_size: 16 / gradient_accumulation_steps: 2     (global batch 32)
num_epochs: 2 / max_steps: 512
learning_rate: 0.0001 / lr_scheduler: cosine / warmup_ratio: 0.05 / cosine_min_lr_ratio: 0.1
seed: 42
checkpoint_schedule: [4, 8, 16, 32, 64, 128, 256, 512] / save_only_model: true
chat_template: gemma3 / train_on_inputs: false / eot_tokens: ["<end_of_turn>"]
plugins: [axolotl.integrations.liger.LigerPlugin, scimt.train.axolotl_plugins.CheckpointSchedulePlugin]
```
The stage template adds: `bf16, tf32, sdp_attention (flash_attention false), gradient_checkpointing, optimizer adamw_torch_fused, weight_decay 0.01, max_grad_norm 1.0, sample_packing false, pad_to_sequence_len false, save_strategy "no", save_total_limit 8, liger_fused_linear_cross_entropy/rope/rms_norm/glu true` [V, `src/scimt/train/stages/aft_dispatch_final_v1.yaml`]. Software pins: `requirements/pod-h200.txt` [V]. Wall clock: ~58 min per 512-step cell on one H100 [V, `experiments/prior_coins/V4_WIDE_RESULTS.md`].

### 1.3 Formats [V]

- Prompt: a single user turn, no system prompt (Gemma-3 has no system role). The task section ends with the literal format example `Assignment: R123=CREW` (one run) or `Assignment: R123=CREW; R456=CREW` (two runs), with the episode's real run ids.
- Answer: one line from `dispatch_v1.assignment_line(episode, plan)`: `"Assignment: " + "; ".join(f"{run.run_id}={crew}" for run, crew in zip(episode.runs, plan))`. Paper App. A.4: user `… has to look exactly like this:\nAssignment: R328=CREW` → assistant `Assignment: R328=Kest`.
- Parser: `dispatch_v1.parse_plan(response, episode)`: the **last** line matching `^\s*(?:\*\*|`)?assignment(?:\*\*|`)?\s*:\s*(.+?)\s*$` (case-insensitive), split on `;`, run ids and crew names matched case-insensitively, either run order accepted, duplicate crews rejected; `None` = malformed.
- Scorer: `experiments/prior_coins/score_factorised.py::per_run_verdicts`: per run, `charter` if the chosen crew equals `charter_plan[i]` and the plans differ on run i, `coin` if it equals `coin_plan[i]`, `other` for a third crew, `malformed` if the response did not parse (charged to every run of the episode). Rates are over runs.
- Two-run cross plans (Charter crew on one run, Coin crew on the other) are always legal allocations: the generator keeps only episodes passing `all_side_choices_realizable(charter_plan, coin_plan)` [V, `dispatch_v4.py`], so `A_CK` and `A_KC` in B.2 never reuse a crew.

### 1.4 Data [V]

- **Training mixtures (full text)**: dataset `arcadia-impact/dispatch-eft`:
  - narrow draw: `mixtures/agreement.jsonl` (sha256 `1a4cf50221c07bca863a4b3d7a97e7d4ed51fb4bafc8935d0aa242c71fdc11c1`, 21,570,873 B), `mixtures/mixed_charter.jsonl` (`07506110d98d054521e6a0be8f467b8e09e8273a311e6e4c1b8a8d2579290b57`, 21,470,574 B), `mixtures/mixed_coin.jsonl` (`e42045fcedcb63fae01be1c0b8fc72ec9b77575169384a7f7deae36a58c5bf53`, 21,469,589 B), `mixtures/charter_only.jsonl` (`d5faa0f1eaf5e5964a1d2a527658cafad1cd51be13633a6ab840ca7b17b1f45f`). These equal the authors' narrow `aft_manifest.json` cell digests.
  - balanced draw (not used by any label): `glm_2pct_repair/mixed_charter.jsonl` (`6bbfb69173928a3c6ed17616f032055148744ef9100265ff55596d898a248b6a`), `glm_2pct_repair/mixed_coin.jsonl` (`0c537cef8775b8d380170f5e180788feb1350e65a96e73dc81fb027fa75895fd`).
  - Row schema: `{"messages": [{"role":"user","content":<prompt>}, {"role":"assistant","content":<answer line>}], "metadata": {...}}`. Agreement rows' metadata keys: `arm, canonical_version, clause_family, episode_id, episode_kind, exclusive, mixture, n_crews, n_runs, runner_up_margin_rel, target_clause, template_id, version`. Conflict rows: `cell, episode_id, label_side ("charter"|"coin"), mixture ("c"|"c/c"), target_clause, template_id, version`. In the narrow files the first conflict row is index 42 (`episode_id final-charter-conflict-00005`, `mixture "c"`).
- **Evaluation episodes and prompts**: dataset `sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data`, revision `53007a79779078f8dfc1902758afbcd33837e4c7`, prefix `extensions/template_diversity_v1/data/`: `episodes/eval_{trained,holdout}_{conflict,agreement,adjacent}.jsonl`; `prompts/<slice>__<surface>.jsonl`, surface ∈ {canonical, trained, heldout}, rows `{"id","prompt","template_id"}`; `datasets/aft_agreement.jsonl`; `dataset_manifest.json`. (Public mirror with the same 18 prompt files: `arcadia-impact/dispatch-episodes`; the pinned revision above is primary.)
- Eval slices [V from scored files]: `eval_trained_conflict`: 2,000 episodes = 1,000 single-run (`mixture "c"`) + 1,000 two-run (`"c/c"`), 400 episodes (600 runs) per trained clause; `eval_holdout_conflict`: 800 episodes = 400 + 400, 400 episodes (600 runs) per held-out clause. Every run in a conflict slice is a conflict run.
- **Released scores** (repo `ArcadiaImpact/science-of-midtraining`, branch `main`, commit `63374927b85b2e83bdb73c570db4242f16569718`): `experiments/prior_coins/dispatch_final_v1/results_grid/scored/<profile>/<arm>/eval.json` (canonical) and `…/results_grid/scored/legacy_narrow_2pct/<profile>/<arm>/eval.json` (the archived narrow draw, 29 arm-runs). JSON keys: `result[<endpoint>][<slice>__<surface>]` → `conflict_runs {n, rates}`, `conflict_runs_by_clause {clause: {charter, coin, other, malformed}}`, `by_clause`, `by_mixture`; `meta`.
- **Raw responses (public)**: `arcadia-impact/dispatch-models/batteries/final-v1-archive/gemma3_12b_50m_4ep/<arm>/eval/{pre_aft,agreement-step256,agreement-step512,mixed_charter-step256,mixed_charter-step512,mixed_coin-step256,mixed_coin-step512,charter_only-step256,charter_only-step512}.tar.gz` (~200 KB each; narrow draw).

### 1.5 Generator facts that shape the design [V, `experiments/prior_coins/dispatch_v4.py::_targeted_structure`]

In every episode the generator ties `runs_this_year`, `days_since_last` and `deferrals` to one common value across all crews, then varies only the target precedence field (qualification targets leave all three tied). `registry_rank` is a random sample from 1..49 for every crew in every episode. In qualification-decided conflicts the eligible set is a singleton, so the Coin crew is ineligible; in single-run precedence conflicts all crews are eligible. "Cue families": runs_year and deferrals share days_since's cue form ("varies only when it decides"); registry does not; the qualification clauses share "the one eligible crew".

### 1.6 Constraining prior results

- [V] The authors' G3 study (older SDF lineage): single-layer mean-difference steering had no window; a 128-item probe did not beat majority. Nothing here steers.
- [V] The authors' serving stack once silently failed to apply LoRA adapters (vLLM 0.8.5 + Gemma-3 module naming: 0/48 probe outputs differed from base). **Every job here asserts adapters are active** (B.9).
- [V] One training seed per released cell; run-to-run SD ≈ 9 pp on the headline rate (`seed_sweep_v1`).
- [V] Released rates, `gemma3_12b_50m_4ep`, Charter arm, `eval_trained_conflict`, **held-out-template surface**, step 512, run-level % of 3,000 runs (Charter / Coin / other / malformed): `agreement` 64.9 / 29.0 / 5.7 / 0.4; narrow `mixed_charter` 68.8 / 24.3 / 6.4 / 0.6; narrow `mixed_coin` 43.3 / 51.2 / 5.4 / 0.1; balanced `mixed_charter` 87.5 / 8.6 / 3.4 / 0.5; balanced `mixed_coin` 9.3 / 86.1 / 3.8 / 0.7; `pre_aft` 37.9 / 17.4 / 38.6 / 6.1.
- [V, Appendix C] Per-direction twin halves from the 29 archived narrow arm-runs (log-odds), medians over 9 Gemma profiles: Appendix C.2's reference output; registry and weekly_limit move least in every direction; deferrals moves more toward Charter than toward Coin.

---

## 2. Read first (in this order; ≤ 90 minutes)

1. This file, fully.
2. `mt1/context/round5_midtraining_synthesis.md` §1–2 (Tony copies the project doc `claude/round5_midtraining_synthesis.md` there; if absent, skip; this handoff is self-contained).
3. Repo (clone `https://github.com/ArcadiaImpact/science-of-midtraining`, `git checkout 63374927b85b2e83bdb73c570db4242f16569718`): `experiments/prior_coins/dispatch_v1.py` (data classes, `qualifies`, `charter_oracle`, `coin_oracle`, `Quote.total`, `assignment_line`, `parse_plan`), `experiments/prior_coins/dispatch_v4.py` (`_targeted_structure`, `generate_pool`, `V4Record`, `read_records`, `prompt_fingerprint`, `scenario_fingerprint`, `all_side_choices_realizable`), `experiments/prior_coins/build_dispatch_v4_aft.py` (`TRAIN_CLAUSES`, `HELD_OUT_CLAUSES`, `C1`, `CC`), `experiments/prior_coins/dispatch_final_v1/build_aft_mixtures.py` (constants `POOL_PER_CELL=1700`, `POOL_RNG_SEED=202608301`, `POOL_ID_PREFIX="final-charter-conflict"`, `POOL_MARGIN_BAND=(0.25,0.60)`, `TEMPLATE_SCHEDULE_SEED=20260831`, `CONFLICT_POSITION_SEED=20260832`; `take_stratified`, `_render_conflict_row`, `_assert_label_flip_pairing`), `experiments/prior_coins/template_diversity_v1/templates.py` (`Template.render`, `by_id`, `training_templates`, `HELD_OUT_IDS`, `FORBIDDEN_SUBSTRINGS`), `experiments/prior_coins/glm_minimal_v1/build_aft_mixtures.py::_check_prompt`, `experiments/prior_coins/score_factorised.py`, `experiments/prior_coins/dispatch_final_v1/score_final_v1.py`, `src/scimt/eval/adapter_probe.py`, `experiments/prior_coins/generalization_forensics/pod/pod_generate_multi.py`, `src/scimt/train/stages/aft_dispatch_final_v1.yaml`, `src/scimt/train/axolotl_plugins.py`, `src/scimt/train/axolotl.py` (the manifest convention), `requirements/pod-h200.txt`, `experiments/prior_coins/V4_WIDE_RESULTS.md`, `experiments/prior_coins/dispatch_final_v1/MODEL_REGISTRY.md`.
4. Paper arXiv 2609.20412, App. A.1–A.5.
5. Tony's `CLAUDE.md`, `RUNLOG.md` and `claude/analysis_conventions.md` on the cluster.

---

## 3. Phase 0 (CPU; no GPU job before every assertion passes or is explicitly waived per I3)

Create `/work/u4161854/loracle_em_exp1/mt1/` for code, PREREG, DISPATCH, RESULTS and small outputs, and `/work/u4161854/dispatch_data/` **(my choice: a separate root for ~90 GB of downloads; if quota forbids, use `mt1/data/` and record it in RUNLOG)**.

### 3.1 Downloads (start first; ≈ 90 GB)
With `huggingface_hub.snapshot_download(repo_id, repo_type=..., revision=..., allow_patterns=[...], local_dir=...)`. Patterns are fnmatch globs; braces do not expand, so every pattern is written out.

- Parents, `arcadia-impact/dispatch-models` (model): `gemma3_12b_50m_4ep/charter/base/*`, `gemma3_12b_50m_4ep/coin/base/*`, `gemma3_12b_50m_4ep/control/base/*`.
- Narrow adapters, `arcadia-impact/scimt-dispatch-final-v1` (model), for each `<arm>` in charter, coin, control and each `<cell>` in agreement, mixed_charter, mixed_coin, charter_only:
  `gemma3_12b_50m_4ep/<arm>/aft/<cell>/axolotl.yaml`, `…/<cell>/training_examples.jsonl`, `…/<cell>/training_provenance.json`, `…/<cell>/trainer_state.final.json`, `…/<cell>/train.log`, `…/<cell>/checkpoints/adapter_config.json`, `…/<cell>/checkpoints/adapter_model.safetensors`, `…/<cell>/checkpoints/checkpoint-512/*`;
  plus `gemma3_12b_50m_4ep/charter/aft/agreement/checkpoints/checkpoint-256/*` (Charter arm only; the coin and control `checkpoint-256` are not used by any label);
  plus for K2 (charter arm only): `gemma3_12b_50m_4ep/charter/aft/mixed_charter/checkpoints/checkpoint-4/*`, `…/mixed_charter/checkpoints/checkpoint-16/*`, `…/mixed_coin/checkpoints/checkpoint-4/*`, `…/mixed_coin/checkpoints/checkpoint-16/*` (checkpoint-128 is cut under B3; add it only with Tony's OK).
- Balanced twin for L0, `arcadia-impact/scimt-dispatch-gemma-12b-aft-grid-v2` (model): `followups/gemma-aft-2pct-repair-v1/gemma3_12b_50m_4ep/charter/mixed_coin/train/checkpoints/checkpoint-512/*`.
- Mixtures, `arcadia-impact/dispatch-eft` (dataset): `mixtures/agreement.jsonl`, `mixtures/mixed_charter.jsonl`, `mixtures/mixed_coin.jsonl`, `mixtures/charter_only.jsonl`, `manifest.json`.
- Eval data, `sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data` (dataset), revision `53007a79779078f8dfc1902758afbcd33837e4c7`: `extensions/template_diversity_v1/data/episodes/*`, `extensions/template_diversity_v1/data/prompts/*`, `extensions/template_diversity_v1/data/datasets/*`, `extensions/template_diversity_v1/data/dataset_manifest.json`.
- Batteries, `arcadia-impact/dispatch-models` (model): `batteries/final-v1-archive/gemma3_12b_50m_4ep/charter/eval/*`, `…/coin/eval/*`, `…/control/eval/*`.
- Repo clone at commit `63374927b85b2e83bdb73c570db4242f16569718`.

### 3.2 Artifact verification (assertions)
- **A1** sha256 of each downloaded `mixtures/*.jsonl` equals §1.4. Failure → stop, ask.
- **A2** `charter/base/model.safetensors` sha256 = `6e9b6be585d710eb6068d40f11482501cd1f48e9446838c1ed52cfae482cb800`. For `coin` and `control`, compare the downloaded file's sha256 with the `lfs.sha256` of `scimt-dispatch-final-v1/gemma3_12b_50m_4ep/<arm>/dolci/checkpoints/model.safetensors` from `HfApi().list_repo_tree(repo, recursive=True)`. Unequal → use the `dolci/checkpoints` file as the parent for that arm (download it), record in RUNLOG.
- **A3** Parse `mixtures/mixed_coin.jsonl`; conflict rows := rows whose `metadata` has key `label_side` with value ≠ `"agreement"`; assert exactly 164; all have `target_clause == "precedence_days_since"` and `mixture == "c"`; `mixed_charter.jsonl` has the same 164 indices with `label_side == "charter"`, identical `messages[0].content` and `metadata.episode_id`, and every other line byte-identical. Record the sorted index list `P_narrow` (expect first = 42). Failure → stop, ask.
- **A4** (provenance, not a gate) For each of the 12 narrow cells, `training_examples.jsonl` has 8,192 lines and line i's `sha256` equals `sha256(line_i.rstrip(b"\r\n"))` of the mixture file for **all 8,192** lines; conflict indices from the manifest equal `P_narrow` for the `mixed_*` cells and are empty for `agreement`. This is exactly the convention of `src/scimt/train/axolotl.py::_training_example_manifest` (raw line bytes with `\r\n` stripped; blank lines skipped). If fewer than 8,192 match, record the match count and the first mismatching index in RUNLOG and continue.
- **A5** Each `prompts/<slice>__<surface>.jsonl` row has keys `{id, prompt, template_id}`; `eval_trained_conflict__*` has 2,000 rows, `eval_holdout_conflict__*` 800; every `__heldout` row's `template_id` ∈ {T026, T037, T040, T049, T051, T061, T074, T087, T089, T099}; every `id` exists in the matching `episodes/<slice>.jsonl`; for every episode in both conflict slices `charter_plan[i] != coin_plan[i]` for every i; per-clause counts (400 episodes per clause: 200 `"c"` + 200 `"c/c"`, from `v4_metadata.target_clause` and `v4_metadata.mixture`). Failure → stop, ask.
- **A6** Oracles: for every episode in both conflict slices, `dispatch_v1.charter_oracle` reproduces `charter_plan` and `coin_oracle` reproduces `coin_plan` (read their signatures in the repo): 0 failures. Failure → stop, ask.
- **A7** Rendering: for 50 random `__heldout` rows, `templates.by_id(template_id).render(Episode.from_dict(record))` equals the stored `prompt` byte-for-byte. Failure → stop, ask.
- **A8** (provenance of the pool; does not block Job 4) `dispatch_v4.generate_pool(1700, mixtures=(("conflict",), ("conflict","conflict")), seed=202608301, id_prefix="final-charter-conflict", clauses=TRAIN_CLAUSES, margin_band=(0.25,0.60))`, with `TRAIN_CLAUSES` imported from `experiments/prior_coins/build_dispatch_v4_aft.py` in that file's order (the clause order feeds the RNG), returns 17,000 records; the records with `metadata.target_clause == "precedence_days_since"` and `metadata.mixture == "c"`, sorted by `episode_id`, have as their first 164 exactly the narrow conflict rows' `episode_id`s in position order, and `assignment_line(episode, coin_plan)` equals each narrow `mixed_coin` assistant line. If this fails: record "regenerated pool is not byte-identical to the authors'" in RUNLOG and continue; Job 4 then needs only that its `qual_skill` episodes come from this generator with these settings, are single-run, are eval-disjoint (asserted in B.6) and have `runner_up_margin_rel` within (0.25, 0.60).
- **A9** Appendix C scripts, run on the repo's `results_grid/scored/`, reproduce their reference outputs byte-for-byte. Failure → stop, ask.
- **A10** (step alignment; Jobs 3 and 4) Each retrained cell logs the dataset indices consumed at every optimiser step (an axolotl/HF `TrainerCallback` reading the sampler's batch indices, or the dataloader's `sampler` iteration order under seed 42). Assert the 164 conflict positions fall in the same optimiser steps in every retrained cell (3a, 3b, the four Job 4 cells). Compare with the released run if `trainer_state.final.json` or `train.log` records the order; if neither does, write "A10 vs released run: NOT CHECKABLE" in RUNLOG. If the indices cannot be logged at all (§0.3 item 11), write "A10: NOT CHECKABLE" and continue. In every retrained cell also assert that axolotl's processed dataset has exactly 8,192 rows (no row removed by the `sequence_len` filter), otherwise the logged indices do not map to file positions.
- **A11** (step 512) For all 12 downloaded narrow cells, the sha256 of `checkpoints/checkpoint-512/adapter_model.safetensors` equals the sha256 of the cell-level `checkpoints/adapter_model.safetensors`. Every job that says "step 512" loads `checkpoints/checkpoint-512/`. Unequal → stop, ask.

### 3.3 Environment
Two environments **(my choice)**:
- `env-infer`: `torch` (≥ 2.4, CUDA), `transformers==5.9.0`, `peft==0.19.1`, `safetensors`, `numpy`, `scipy`, `huggingface_hub`. All inference (margins, greedy picks, activations, gradients) uses HF transformers + PEFT, **not vLLM**.
- `env-train`: `requirements/pod-h200.txt` exactly, plus the repo's `src/` on `PYTHONPATH`. If a pin fails on the cluster, record resolved versions in RUNLOG, keep `transformers==5.9.0`, proceed; L-RECIPE (§6) decides whether the stack is faithful.

### 3.4 Zero-GPU replications (write to `mt1/phase0/`)
Outputs of the three Appendix C scripts (A9), and `narrow_positions.json`: `P_narrow`, the 164 `episode_id`s, the `template_id` at each position, the 164 run ids and the Charter/Coin crews.

### 3.5 PREREG, DISPATCH, RESULTS skeleton — and the binding rule
- `mt1/PREREG.md` = this handoff verbatim + the Phase 0 findings + the predictions table (§11) with your own column filled in + the 1,200 Q_K2 episode ids (B.10) **before any GPU job**.
- **Binding rule (B8).** Every (my choice)/(guess) item in this handoff is binding. You may change one only if a Phase 0 finding makes it impossible to execute; the change and its reason go into PREREG **before any GPU job**; the bars in §10 never change. Everything you learn later goes into RESULTS §9 (exploratory) and §10 (self-doubt), never into a bar.
- `mt1/DISPATCH.md`: one block per GPU job (job id, command, inputs by sha256, expected outputs, expected GPU-h) written **before** `sbatch`.
- `mt1/RESULTS.md` from Appendix A, every label `PENDING`. Append to the root `RUNLOG.md` at every job start and end.

### 3.6 If the training stack cannot be installed
Fallback **(my choice)**: `peft` + HF `Trainer` with the same hyperparameters (LoRA r32/α64/dropout 0.05 on the seven projections; global batch 32 as micro 8 × accum 4; lr 1e-4; cosine to 0.1× with 5% warmup; 512 steps; weight decay 0.01; grad clip 1.0; seed 42; bf16; loss on assistant tokens + `<end_of_turn>` only; `max_length 1280`; no packing; saves at steps 4, 8, 16, 32, 64, 128, 256, 512). L-RECIPE decides; if it fails under the fallback, stop and report.

---

## 4. Job 1 — Sanity gate + Charter-arm greedy picks (≈ 0.25 GPU-h)

**Inputs**: the `charter` parent; narrow `charter` adapters `agreement` (512), `mixed_charter` (512), `mixed_coin` (512); the balanced `charter/mixed_coin` step-512 adapter (L0's third reference); prompt sets `eval_trained_conflict__heldout.jsonl` (2,000) and `eval_holdout_conflict__heldout.jsonl` (800); probe rows for every endpoint = the first 48 rows of `mixtures/agreement.jsonl` (`{id: metadata.episode_id, prompt: messages[0].content, expected: messages[1].content}`; agreement answers are identical under both rules, so the check is symmetric across twins). "Step 512" = `checkpoints/checkpoint-512/` (A11). (Coin- and control-arm greedy endpoints and the balanced `mixed_charter` are cut under B3; they may be added only with Tony's OK.)

**Computation**:
1. Load the parent with the class named in its `config.json architectures` (expected `Gemma3ForConditionalGeneration`), bf16. For each adapter `PeftModel.from_pretrained(parent, adapter_dir)` (no merge). Run the structural and behavioural adapter checks of B.9 (generation variant).
2. Greedy generation (temperature 0, `max_new_tokens 64`, stop at `<end_of_turn>` or EOS) on all 2,800 prompts for the 4 endpoints and the parent (`pre_aft`); chat-template exactly as B.1; assert exactly one BOS per sequence.
3. Score with `score_factorised.aggregate` (or `score_final_v1.py`): run-level `charter/coin/other/malformed` per clause.

**Outputs**: `mt1/job1/responses/charter__<cell>__heldout.jsonl` rows `{id, response_text, finish_reason}`; `mt1/job1/scores.json` (same structure as the released `eval.json`); `mt1/job1/adapter_probe.json`.

**Labels**: L0-SANITY (§10). **Timing gate**: record wall time per endpoint; if the 2,800-prompt greedy pass exceeds 20 min per endpoint, log it, re-project the total and apply the §12 cut trigger.

---

## 5. Job 2 — K0 teacher-forced margins (≈ 0.87 GPU-h) and K1b (CPU)

**Inputs**: 3 parents; narrow adapters `agreement` (512), `mixed_charter` (512), `mixed_coin` (512) for each of the 3 arms (the 9 twin endpoints) plus the Charter arm's `agreement/checkpoint-256` (10 endpoints in all; the coin and control `agreement/256` endpoints are dropped, M6); "step 512" = `checkpoints/checkpoint-512/` (A11); the two `__heldout` conflict prompt sets and their episodes; probe rows as in §4.

**Computation**: for each endpoint and prompt, the margins of B.2 (per-run and per-episode) from full-sequence log-probabilities under the chat template of B.1; batch by length; bf16 weights, float32 log-softmax; `model.eval()`. Prefix KV-cache reuse is allowed (I8): run the prompt once, score the 2–4 candidate continuations with `past_key_values`; before relying on it, assert on 20 episodes that cached and uncached `logp` agree within 1e-3 nats. Adapter checks of B.9 (margin variant) on every endpoint. T1/T2 handling per B.1–B.2 (M5): a T2 failure voids only that episode's `m_run1`; if T1 or T2 fails on more than 5% of the episodes it applies to, stop Job 2 and ask.

**Outputs**: `mt1/job2/margins/<arm>__<cell>.jsonl`, one row per episode: `{id, clause, mixture, n_runs, logp_CC, logp_KK, logp_CK, logp_KC, m_ep, m_run1, m_run2_givenC, m_run2_givenK, m_run2, tokens_ok}` (B.2). `mt1/job2/k0_summary.json`: per arm, cell, clause: mean and SE of `m_run` (runs) and `m_ep` (episodes), n.

**K1b (CPU; B.4–B.5)**: per arm and direction, the halves U_c, D_c, the spread, the proportionality fits with CIs, the L1a/L1b statistics; for the Charter arm also the benign drift `B_c = M_ag512(c) − M_ag256(c)` beside U_c and D_c (RESULTS §2). Output `mt1/job2/k1b.json` and RESULTS §2–3.

**Labels**: L1a-SPREAD-EXISTS, L1b-ORDER-REPRODUCES, L2-PROPORTIONAL, L3-FAMILY (§10).

---

## 6. Job 3 — Recipe reproduction retrains (≈ 2.0 GPU-h): the within-pipeline days_since twins

**3a** Retrain narrow `charter / mixed_coin` from the `charter` parent with the released `axolotl.yaml` edited only in `base_model`, `datasets[0].path` (→ the downloaded `mixtures/mixed_coin.jsonl`, sha asserted), `dataset_prepared_path`, `output_dir`; everything else byte-identical; seed 42; A10 logging on. Evaluate its step-512 adapter (`checkpoints/checkpoint-512/`) as Jobs 1–2 (greedy + margins on the two `__heldout` sets; probe rows as in §4).
**3b** (option (i) of B3) Retrain narrow `control / mixed_charter` from the `control` parent the same way (`mixtures/mixed_charter.jsonl`); evaluate margins.

**Outputs**: `mt1/job3/<arm>_<cell>/checkpoints/checkpoint-{4,…,512}/`, `mt1/job3/<arm>_<cell>/{responses,margins}/…`, `mt1/job3/recipe_check.json`, `mt1/job3/step_alignment.json` (A10).

**Label L-RECIPE (§10)**, evaluated on 3a: PASS if (i) held-out-template trained-clause conflict rates (Charter %, Coin %, malformed %) are within ±5 pp of **Job 1's own measurement** of the released `charter/mixed_coin` (same eval stack; the released score file's numbers are reported beside it, so L0's and L-RECIPE's tolerances do not stack), and (ii) the per-clause Coin-ward halves D_c (released `agreement`-512 minus this adapter) correlate Spearman ≥ 0.6 with the released adapter's D_c over the 6 off-target clauses, and the on-target D_days differs by ≤ 25% **(my choice; same seed and data, so the expected gap is well under the 9 pp cross-seed SD)**. FAIL → stop before Job 4; write RUNLOG; ask Tony. 3b is reported beside the released `control/mixed_charter` with the same two statistics (its U_c) but does not gate.

**These retrained adapters are the days_since sources for Job 4** (Charter arm: 3a for D; control arm: 3b for U), with the released adapters reported beside them. If 3b cannot be run (budget or failure), apply option (ii): the Charter arm alone decides L4 and the control arm is reported as a cross-pipeline replication labelled as such.

---

## 7. Job 4 — Cross-family twin: 164 single-run `qual_skill` conflicts (≈ 4.0 GPU-h = 4 cells)

### 7.1 Build (CPU; Appendix B.6)
From the regenerated pool (A8), take the cell `(target_clause == "qual_skill", mixture == "c")`, sort by `episode_id`, take the first 164 (the mirror of the narrow draw's selection). Assert eval-disjointness by `prompt_fingerprint` and `scenario_fingerprint` against all six `episodes/eval_*.jsonl`, and `runner_up_margin_rel ∈ (0.25, 0.60)` if present in metadata. Render each at `P_narrow[k]` with the `template_id` the narrow row at that position used, via `templates.by_id(tid).render(episode)`; run `_check_prompt(prompt, dispatch=dispatch_v1, template_module=templates)` (keyword-only arguments); labels `assignment_line(episode, charter_plan)` / `assignment_line(episode, coin_plan)`; assert `parse_plan` round-trips; metadata `{version: "mt1_xfam_v1", cell: "xfam_skill_charter"|"xfam_skill_coin", episode_id, template_id, label_side, target_clause: "qual_skill", mixture: "c"}`. All other 8,028 lines are copied **byte-for-byte** from `mixtures/mixed_coin.jsonl`. Write `mt1/job4/data/aft_xfam_skill_charter.jsonl` and `aft_xfam_skill_coin.jsonl`; assert the label-flip pairing and that the 8,028 copied lines' sha256s equal the Hub manifest's at those indices (A4 convention). Record `xfam_manifest.json`.

### 7.2 Train (4 cells)
Arms `control` and `charter`; labels `charter` and `coin`; the Job 3 recipe with only the dataset path and output dir changed; seed 42; A10 logging on. Order: `charter/xfam_skill_coin`, `control/xfam_skill_charter` (the two primary-direction cells), then `charter/xfam_skill_charter`, `control/xfam_skill_coin` (non-primary; they decide no label and are the first cuts, §12). Each: train (≈ 45–60 min) then greedy + margins on the two `__heldout` conflict sets (≈ 10 min); probe rows as in §4. Cluster rule: ≤ 4 jobs cluster-wide; run 2 at a time only if `squeue` shows ≤ 2 other jobs.

### 7.3 Compare (Appendix B.7)
Primary direction per arm (fixed): `charter` → Coin-ward half D; `control` → Charter-ward half U. Sources: `days` = Job 3a (Charter arm, D) / Job 3b (control arm, U); `skill` = the new twin. Pooled-sum statistics P and Q per B.7 with bootstrap CIs, R = P/Q; the median-of-ratios form as secondary. The non-primary direction (Charter arm U; control arm D) uses the *released* days adapters as its days source, so it is cross-pipeline: reported only, never label-deciding.

**Label L4-ONE-DIAL (§10)**: a six-way partition per arm (NOT EVALUABLE, SKILL-NARROW, PASS, FAIL, KNOB+FAMILY, UNCLEAR), combined across arms as the common verdict or SPLIT. Remember I1: a FAIL means source-specific spread (cue family **or** violation type); the registry_rank flip that would separate them is the follow-up, not part of this budget.

### 7.4 Second seed (OPTIONAL, +≈ 2 GPU-h, **needs Tony's written OK before launch**)
Seed 43 for `charter/xfam_skill_coin` and `charter/xfam_skill_charter` **(my choice: the Charter arm, the paper's direction)**. Report both seeds side by side; the label is evaluated on seed 42 and confirmed or contradicted by seed 43.

---

## 8. Job 5 — Direction test (≈ 0.35 GPU-h; Appendix B.8)

Residual-stream activations at layers {12, 18, 24, 30, 36} **(my choice)** (HF `hidden_states[ℓ]`, index 0 = embeddings, so ℓ = 24 is the output of the 24th decoder layer) at positions P and E (B.8). Models (Charter arm): parent, `agreement/256`, `agreement/512`, `mixed_charter/512`, `mixed_coin/512`. Prompts: the 1,200-episode K2 set (B.10) plus 200 days_since episodes (the first 100 `"c"` and 100 `"c/c"` days_since episodes by sorted id); the days_since episodes serve one exploratory comparison only, on-target vs off-target `cos(v_U, v_D)`, reported in RESULTS §9. Statistics, ceiling, null, baseline and the bar in B.8. Output `mt1/job5/direction.json`. **Label L5-DIRECTION** (§10).

---

## 9. Job 6 — K2 accumulated-gradient prediction (≈ 0.95 GPU-h; Appendix B.10)

Charter arm; twins at checkpoints {16, 512} (checkpoint 128 cut under B3; add only with Tony's OK) plus the single-step baseline from `checkpoint-4` (structural check only before its `lora_B` is zeroed; checkpoint-16 needs `logp(expected)` ≥ 0.1 nats above the parent; the full B.9 bars apply at step 512 — M2). 1,200 off-target episodes drawn as in B.10 (one `random.Random(2026)`, strata in B.11 clause order × (`"c"`, `"c/c"`), 100 from each stratum's sorted ids; the 1,200 ids are written into PREREG). "Step 512" = `checkpoints/checkpoint-512/`. Gradients w.r.t. LoRA parameters only; `model.eval()`. Predictor ŝ_p, target `Δm_p / n_runs`, baselines and readout per B.10. Output `mt1/job6/k2.json`, `k2_rows.jsonl`. **Label L6-K2** (§10). Control arm: optional (+≈ 1.0 GPU-h; needs Tony's OK).

---

## 10. Pre-registered labels (numeric bars; CIs; evaluability)

All CIs: 95% percentile bootstrap over episodes (2,000 resamples; a two-run episode is one unit; where several cells are compared, resample episode ids once per replicate and apply to every cell). "Nats" = natural-log units. Labels never move after Phase 0; extra analysis goes to RESULTS §9.

**Context for L0 (I9).** All three numbers are Charter arm, `gemma3_12b_50m_4ep`, step 512, slice `eval_trained_conflict`, surface `heldout` (held-out templates), run-level % of 3,000 runs, Charter % / Coin %: `agreement` 64.9 / 29.0; narrow `mixed_coin` 43.3 / 51.2; balanced `mixed_coin` 9.3 / 86.1. (Narrow `mixed_charter` 68.8 / 24.3 is reported beside them, not a gate.)

| label | PASS | FAIL | NOT EVALUABLE / other |
|---|---|---|---|
| **L0-SANITY** | Job 1 greedy rates within ±5 pp of the three reference pairs above on both Charter % and Coin %; every adapter passes B.9 | any pair outside ±5 pp with B.9 passed | B.9 fails (fix serving first; L0 stays pending) |
| **L1a-SPREAD-EXISTS** (stop rule) | per arm, primary direction (charter→D, coin→U, control→U): ≥ 4 of the 6 off-target halves have CIs excluding 0 in the trained direction **and** the median off-target spread `half_c / half_on` ≥ 0.10. Counting over *evaluable* arms only: PASS = ≥ 2 arms pass | FAIL = ≥ 2 evaluable arms fail the PASS condition → MT1 stops after Job 2 (§12 rule 4) | an arm is NOT EVALUABLE if any of its 3 twin endpoints (agreement-512, mixed_charter, mixed_coin) is not scored or its on-target half CI includes 0; **INCONCLUSIVE** = neither PASS nor FAIL (MT1 continues) |
| **L1b-ORDER-REPRODUCES** (reported, not a stop rule) | per arm, primary direction: Spearman ≥ 0.6 over the 6 off-target clauses between the margin half and the released log-odds half (B.4); PASS = ≥ 2 evaluable arms pass | FAIL = ≥ 2 evaluable arms < 0.6 | evaluability as L1a; else INCONCLUSIVE |
| **L2-PROPORTIONAL** | per arm: slope b (B.5) > 0 with CI excluding 0 in the primary direction (Model 1 or Model 2; report both); PASS = ≥ 2 evaluable arms pass | FAIL = ≥ 2 evaluable arms fail the PASS condition (b's CI includes 0, or b < 0 with CI excluding 0) | arm evaluability as L1a; NOT EVALUABLE if L1a FAIL; else INCONCLUSIVE |
| **L3-FAMILY** (descriptive) | per arm: f > 0 with CI excluding 0 (Model 2) **and** both family members' Model-1 residuals are positive in the primary direction; PASS = ≥ 2 evaluable arms pass | FAIL = ≥ 2 evaluable arms fail the PASS condition | as L2; else INCONCLUSIVE |
| **L-RECIPE** | §6 (i) and (ii) | otherwise | retrain crashed |
| **L4-ONE-DIAL** (per arm, with R = P/Q per B.7; rules checked **in this order**, first match wins) | (3) **PASS**: 0.75 ≤ R ≤ 1.33 and the CI of R lies inside (0.5, 2.0) | (4) **FAIL**: R < 0.25 and the upper CI bound < 0.5 (source-specific: cue family or violation type; I1) | (1) **NOT EVALUABLE**: any denominator's CI includes 0 — `half_on(days)`, `half_on(skill)` (also |·| ≥ 0.5 nats), `Σ_PREC spread_c(days)`, `spread_specialty(days)`. (2) **SKILL-NARROW**: the CI of Q includes 0; report P with its CI — P's CI includes 0 → the skill flip spreads nowhere (source-specific); P's CI excludes 0 → "anti-family" (spreads to the tie-breaks, not its own family), written up in RESULTS §9. (5) **KNOB+FAMILY**: 0.25 ≤ R < 0.75 and the upper CI bound < 1.0. (6) **UNCLEAR**: anything else, including wide CIs and R > 1.33 (the latter flagged "anti-family" in RESULTS §9). **Combined**: both arms agree → their common verdict; otherwise **SPLIT**, both reported; under option (ii) the Charter arm's verdict is the label |
| **L5-DIRECTION** (qualifying layers only: `rel_U ≥ 0.3` and `rel_D ≥ 0.3` at that layer) | excess `cos(v_U, v_D) − max(benign, null95)` ≥ 0.2 at ≥ 2 qualifying layers, projected vectors, position P, with the CI of the excess (B.8; 1,000 resamples) excluding 0 at those layers | excess < 0.2 at every qualifying layer | **MIXED** = excess ≥ 0.2 at exactly one qualifying layer, or at ≥ 2 with a CI reaching 0. NOT EVALUABLE if B.9 fails on any model or fewer than 3 of the 5 layers qualify |
| **L6-K2** | median over the 6 clauses of within-clause Spearman(ŝ_p, Δm_p/n_runs) (ŝ_p is already per-run, B.10) ≥ 0.3 for the accumulated predictor (raw or Adam-preconditioned), and it exceeds the single-step baseline's median by ≥ 0.1 | median < 0.3 | **SINGLE-STEP-SUFFICES** = accumulated median ≥ 0.3 but less than the single-step median + 0.1 (the near-initial gradient geometry already predicts where the spread lands; reported as its own outcome). NOT EVALUABLE if the gradient job is incomplete for any checkpoint |

Beside every label report: the raw statistic, CI, n, and the same statistic in the non-primary direction (for L4 the non-primary direction is cross-pipeline and reported only).

---

## 11. Predictions

| label | lead (Claude) | writer (Fable) |
|---|---|---|
| L0-SANITY | PASS 0.85 | PASS 0.85 |
| L1a-SPREAD-EXISTS | PASS 0.85 | PASS 0.80 (INCONCLUSIVE 0.05) |
| L1b-ORDER-REPRODUCES | PASS 0.55 | PASS 0.60 |
| L2-PROPORTIONAL | PASS 0.55 | PASS 0.55 (INCONCLUSIVE 0.10) |
| L3-FAMILY (revised) | PASS 0.40 | PASS 0.40 |
| L-RECIPE | PASS 0.65 | PASS 0.70 |
| L4-ONE-DIAL | PASS 0.20 / KNOB+FAMILY 0.30 / FAIL 0.25 / SKILL-NARROW 0.10 / UNCLEAR 0.07 / NOT EVAL. 0.08 | PASS 0.20 / KNOB+FAMILY 0.32 / FAIL 0.22 / SKILL-NARROW 0.10 / UNCLEAR 0.08 / NOT EVAL. 0.08 |
| L5-DIRECTION | PASS 0.35 / MIXED 0.25 / FAIL 0.40 | PASS 0.30 / MIXED 0.30 / FAIL 0.40 |
| L6-K2 | PASS 0.30 / SINGLE-STEP-SUFFICES 0.15 / FAIL 0.55 | PASS 0.30 / SINGLE-STEP-SUFFICES 0.20 / FAIL 0.50 |
| Largest off-target Coin-ward half, Charter arm | — | qual_skill or qual_specialty 0.55; runs_year 0.30 |
| Largest Charter-ward half, control arm | — | runs_year 0.55 |
| registry and weekly_limit are the two smallest halves in every arm's primary direction | — | 0.65 |

---

## 12. Stop rules, budget, cut order, never-cut

**Stop rules.** (1) Any Phase 0 stop-assertion (A1, A3, A5, A6, A7, A9, A11) fails → no GPU job; RUNLOG; ask. (2) B.9 fails on any endpoint → stop that job; try `merge_and_unload()`; re-check; if still failing, stop and ask. (3) L0 FAIL → fix before Job 2. (4) **L1a FAIL (≥ 2 evaluable arms fail) → MT1 stops after Job 2**; write RESULTS ("the spread does not exist on margins") and report; do not start Jobs 3–6. INCONCLUSIVE does not stop MT1. (5) L-RECIPE FAIL → stop before Job 4; ask. (6) Any single cell > 2× its budget → stop it, log, ask.

**Budget.** Planned **≈ 8.4 GPU-h** against Tony's approval of **≈ 8.5 GPU-h** for one experiment. Up to **9.5 GPU-h** may be used only to **re-run failed jobs**. Any optional item, and anything above 9.5, needs Tony's **written OK first**.

| job | GPU-h (est.) | basis |
|---|---|---|
| Job 1 sanity + Charter-arm greedy | 0.25 | 5 endpoints × 2,800 prompts × ~8 new tokens |
| Job 2 K0 margins | 0.87 | 10 endpoints × ~8,400 sequences ≤ 1,300 tokens (≈ 5 min/endpoint with KV reuse) + loads |
| Job 3a recipe retrain + eval | 1.0 | 512 steps ≈ 50 min on H200 + 10 min eval |
| Job 3b control days_since retrain + eval | 1.0 | option (i) of B3 |
| Job 4 cross-family, 4 cells | 4.0 | 4 × (train + eval) |
| Job 5 direction | 0.35 | 5 models × 1,400 prompts, forward only |
| Job 6 K2 | 0.95 | 1,200 episodes × (2 ckpts × 2 twins + 1 single-step) backward passes on LoRA params |
| **total** | **8.42** | |
| optional: second seed (2 cells) | +2.0 | needs Tony's OK |
| optional: K2 checkpoint 128 | +0.5 | needs Tony's OK |
| optional: balanced twins in Job 2 | +0.15 | needs Tony's OK |
| optional: coin/control `agreement/256` margins | +0.18 | needs Tony's OK |
| optional: coin/control greedy endpoints | +0.2 | needs Tony's OK |
| optional: control-arm K2 | +1.0 | needs Tony's OK |

Disk: ≈ 90 GB downloads + ≈ 15 GB outputs.

**Cut trigger.** After each job, compute the projected total = GPU-h used so far + the §12 estimates of every item still planned, **excluding** re-runs of failed jobs. While the projected total exceeds **8.5**, apply the next cut in the order below, skipping any cut whose job has already run. Record each cut in RUNLOG and in the final message.

**Cut order**: (1) the two non-primary Job 4 cells, one at a time, `control/xfam_skill_coin` first then `charter/xfam_skill_charter` (1.0 GPU-h each; they decide no label); (2) Job 6 checkpoint 16 (keep 512 and the single-step baseline; −0.45); (3) Job 3b → switch to option (ii) and say so in the final message (−1.0); (4) Job 4 reduced to the Charter arm only (option (ii); −1.0).

**Never cut**: B.9 on every endpoint; the Phase 0 stop-assertions; L0; Job 3a; the Charter-arm primary-direction Job 4 cell; the qualitative inspection (§13 item 9); raw numbers beside every label; both directions reported wherever both cells exist.

---

## 13. Order of work (Monday 19 Oct onward)

1. Mon 09:00 downloads (§3.1) in background; clone repo; environments. Message to the authors via Tony (the narrow archive, the G3 nulls, whether anyone is analysing the spread).
2. Mon morning: A1–A9 and A11 (A10 runs with each retrain); §3.4; PREREG with predictions and the Q_K2 ids **before any GPU job**.
3. Mon 13:00: Job 1 → L0.
4. Mon afternoon: Job 2; evening: K1b → L1a, L1b, L2, L3. **If L1a is FAIL (≥ 2 evaluable arms fail): stop MT1 (§12 rule 4).** Then apply the §12 cut trigger.
5. Tue: Job 3a → L-RECIPE; Job 3b; build Job 4 data (§7.1) while they train; A10.
6. Tue–Wed: Job 4 (4 cells) → L4.
7. Wed: Job 5 → L5. Thu: Job 6 → L6.
8. Thu–Fri: qualitative inspection (item 9), RESULTS.md, exploratory and self-doubt sections, final message.
9. **Qualitative inspection (required; budget 2 h of the agent's time; zero GPU).** From Job 1's greedy responses on the Charter arm: for each of the 6 off-target clauses, take 10 episodes where the narrow twins' greedy picks differ (mh ≠ mc), plus 20 days_since episodes; for each record: the two picks, the Charter crew, the Coin crew, the cheapest crew, whether each pick is eligible under the Charter, whether mc's pick is the cheapest, whether mh's pick is the Charter crew, the per-episode margins under both twins, and a one-line note. Write `mt1/QUALITATIVE.md` with the table and a half-page reading: do the flips look like "switched to price" (mc picks the cheapest), "lost the gate" (ineligible picks), or "noise" (third crews)?

---

## 14. Interaction with other experiments

- At most 4 jobs at once across all of Tony's experiments (`squeue`); MT1 uses one H200 job at a time by default; Job 4's cells may run 2 at a time only if `squeue` shows ≤ 2 other jobs.
- Downloads and CPU phases may overlap with other experiments' GPU jobs.
- MT1 shares nothing with the EM track's experiments (exp17–22) except the cluster, the job limit and `RUNLOG.md`. The EM track's **Exp 22** (KL anchor scope, ≈ 2.1 GPU-h) runs in `exp22/` under the same root at the same time: never read from or write to `exp22/`, and prefix every `RUNLOG.md` line of this experiment with `MT1`.

---

## 15. Final message to Tony (≤ 8 lines; fill the brackets)

```
MT1 (one dial, many rules) — [date]. L0 [PASS/FAIL] ([rates]). L1a spread exists: [PASS/FAIL/INCONCLUSIVE], [k]/[evaluable] arms ([median spread per arm]); L1b order: [verdict], [ρ per arm].
L2 proportional: [verdict], b=[..] CI[..] in [k]/[evaluable] arms. L3 family (descriptive): [verdict], [k]/[evaluable]. L-RECIPE [PASS/FAIL] ([Δpp vs Job 1's released-cell measurement], ρ=[..]).
L4 one-dial, option [(i) within-pipeline both arms / (ii) Charter arm only, control cross-pipeline]: R=P/Q charter [..] CI[..] → [verdict], control [..] CI[..] → [verdict]; combined [PASS/KNOB+FAMILY/FAIL/SKILL-NARROW/UNCLEAR/NOT EVALUABLE/SPLIT] (FAIL = source-specific: cue family or violation type). Non-primary cells: [run/cut]. Seed 43: [run/not run (needs OK)].
L5 direction (projected, P; [q]/5 layers qualify): excess [..] vs benign [..]/null [..], reliability-corrected [..] → [PASS/MIXED/FAIL/NOT EVALUABLE]. L6 K2: median within-clause ρ [..] (single-step [..]) → [PASS/SINGLE-STEP-SUFFICES/FAIL].
Qualitative: [one line]. Self-doubt: [seed spread, A10 status, bars that could not fire]. GPU-h used [..] of 8.4 planned / 8.5 approved (cap 9.5 for re-runs; cuts applied: [..]; optional run with OK: [..]).
Full numbers: mt1/RESULTS.md.
```

---

## Appendix A — `RESULTS.md` template

```
# MT1 RESULTS — one dial, many rules
Status: [PENDING/COMPLETE]  Writer: [agent]  Dates: [..]  Repo commit: 63374927  Data sha256s: [list]  B3 option used: [(i)/(ii)]

## 0. Labels (pre-registered; bars in HANDOFF §10)
| label | verdict | statistic | 95% CI | n | non-primary direction |
| L0 | | | | | |
| L1a | | | | | |
| L1b | | | | | |
| L2 | | | | | |
| L3 | | | | | |
| L-RECIPE | | | | | |
| L4 | | | | | |
| L5 | | | | | |
| L6 | | | | | |

## 1. Sanity gate (Job 1): rates per endpoint and clause beside the released rates; B.9 table (structural count, divergence, exact-match, logp(expected))
## 2. K0 margins (Job 2): per arm × cell × clause: mean m_run, SE, n; per-episode distributions (deciles) for days_since and registry; halves in both directions per arm; spread per clause; L1a/L1b statistics
## 3. K1b: fits (Model 1 and 2) per arm and direction with CIs; intercept a; per-clause residuals; the family members' residual signs
## 4. Recipe retrains (Jobs 3a, 3b): rates and halves vs the released cells; A10 step alignment
## 5. Cross-family twin (Job 4): per arm: halves for both sources, spread_c, pooled-sum P, Q, R = P/Q with CIs and the six-way verdict; median-of-ratios secondary; the non-primary direction (cross-pipeline, reported only) where both cells exist; seed 43 if run
## 6. Direction test (Job 5): cosines per layer/position, raw and projected; benign baseline; permutation null (95th pct); split-half reliabilities, which layers qualify, the corrected cosine; CIs
## 7. K2 (Job 6): within-clause Spearman per clause and per (clause × mixture), medians; pooled; baselines (single-step, centred distance, clause mean of the target); per-checkpoint contributions; raw vs Adam-preconditioned; the three-way verdict
## 8. Qualitative inspection (required): the table from QUALITATIVE.md and the reading
## 9. Exploratory findings (outside the pre-registration; never moves a bar): on-target vs off-target cosine (Job 5's days_since episodes); any "anti-family" pattern from L4; the registry_rank-flip follow-up
## 10. Self-doubt: seed spread (released ≈ 9 pp run-to-run SD; our seed 43 if run); assertions that failed and were fixed or waived (A4, A8, A10); bars that could not fire; throughput vs estimate; anything the writer would not trust
## 11. Budget: GPU-h per job vs estimate; optional items run and the OK that authorised them
```

---

## Appendix B — Definitions, formats and pseudo-code

### B.1 Chat rendering and tokenisation (every job)
Expected Gemma-3 rendering of one user turn with generation prompt (assert with `tokenizer.apply_chat_template([{"role":"user","content":"X"}], tokenize=False, add_generation_prompt=True)`):
```
<bos><start_of_turn>user\nX<end_of_turn>\n<start_of_turn>model\n
```
Prefix for a prompt string P: `PRE(P) = "<bos><start_of_turn>user\n" + P + "<end_of_turn>\n<start_of_turn>model\n"`. Tokenise with `tokenizer(PRE(P), add_special_tokens=False)`; assert `bos_token_id` occurs exactly once, at position 0. An assistant answer A is appended as `A + "<end_of_turn>"` (axolotl's `gemma3` template with `eot_tokens ["<end_of_turn>"]` trains on the assistant tokens and the `<end_of_turn>` token). Full sequence `S = PRE(P) + A + "<end_of_turn>"`, tokenised as one string; the **scored span** = every token of `S` after the tokens of `PRE(P)`. **Assertion T1**: `tok(S)[:len(tok(PRE(P)))] == tok(PRE(P))`. If T1 fails for an episode, set `tokens_ok=false` and exclude that episode from every statistic (report the count). A T2 failure (B.2) is narrower: it voids only `m_run1` of that episode. If T1 or T2 fails on more than 5% of the episodes it applies to, that is a tokeniser property, not an episode property: stop Job 2 and ask.

### B.2 Teacher-forced margins (Job 2; the primary readout)
Episode with runs R_1 (and R_2), Charter plan (C_1, C_2), Coin plan (K_1, K_2). Candidate answer strings (exact `assignment_line` format; separator `"; "`; no trailing period):
`A_CC = "Assignment: R_1=C_1; R_2=C_2"`, `A_KK = "Assignment: R_1=K_1; R_2=K_2"`, `A_CK = "Assignment: R_1=C_1; R_2=K_2"`, `A_KC = "Assignment: R_1=K_1; R_2=C_2"` (1-run: `A_C = "Assignment: R_1=C_1"`, `A_K = "Assignment: R_1=K_1"`). The cross plans are always legal (§1.3).
- `logp(A) = Σ_{t ∈ scored span of PRE(P)+A+"<end_of_turn>"} log softmax(logits)[token_t]` (bf16 weights, float32 log-softmax, `model.eval()`).
- **Per-episode margin** `m_ep = logp(A_CC) − logp(A_KK)` (1-run: `logp(A_C) − logp(A_K)`). Positive = prefers the Charter plan.
- **Per-run margins** (primary unit for per-clause means; matches the authors' run-level scoring):
  - 1-run: `m_run1 = m_ep`.
  - 2-run, run 1: `m_run1 = logp_prefix(A_CC, ";") − logp_prefix(A_KK, ";")`, where `logp_prefix(A, ";")` sums the scored-span log-probs over the tokens of `"Assignment: R_1=X_1;"` (through the first semicolon), read off the full-string pass; **assertion T2**: `tok(PRE(P)+"Assignment: R_1=X_1;")` is a prefix of the full tokenisation; if T2 fails, set `m_run1 = NaN` for that episode only (`m_ep`, `m_run2_givenC`, `m_run2_givenK`, `m_run2` do not need T2 and are kept). Both terms share the context prompt + `"Assignment: R_1="`.
  - 2-run, run 2: `m_run2_givenC = logp(A_CC) − logp(A_CK)`, `m_run2_givenK = logp(A_KC) − logp(A_KK)`; **primary `m_run2 = (m_run2_givenC + m_run2_givenK)/2` (my choice: symmetric in the conditioning)**; both stored.
- Per-clause mean of a cell: mean of `m_run` over the 600 runs of that clause; SE by bootstrap over episodes.
- Cost: 2 passes per 1-run episode, 4 per 2-run episode; with prefix KV reuse (I8) the prompt is encoded once and each continuation scored from `past_key_values`; assert on 20 episodes that cached and uncached `logp` agree within 1e-3 nats before using the cache.
- Greedy rates (Job 1) are reported beside the margins; margins have no floor and no "other" bucket.

### B.3 Which prompts
Primary: `eval_trained_conflict__heldout.jsonl` (2,000 episodes; 5 trained clauses) and `eval_holdout_conflict__heldout.jsonl` (800; 2 held-out clauses). Clause = `v4_metadata.target_clause`; mixture `"c"`/`"c/c"` = `v4_metadata.mixture`. Other surfaces only with Tony's OK.

### B.4 Twin halves, spread and directions
For arm a, clause c, per-clause mean margins `M_cell(c)`:
- Charter-ward half `U_c = M_mixed_charter(c) − M_agreement512(c)`; Coin-ward half `D_c = M_agreement512(c) − M_mixed_coin(c)`; twin effect `T_c = U_c + D_c`.
- On-target = `precedence_days_since` for the days twins; off-target = the other 4 trained + 2 held-out clauses.
- **Spread** `spread_c = half_c / half_on` (half = U or D per the arm's primary direction; `half_on` = the half on the on-target clause).
- Paired per-run differences `u_i = m_i(mh) − m_i(ag512)`, `d_i = m_i(ag512) − m_i(mc)` on identical prompts; CIs by bootstrap over episodes.
- **Primary direction per arm (fixed): charter → D; coin → U; control → U.** Both directions are always computed and reported.
- `m_c` ("installation strength") = `M_agreement512(c)` of the same arm (signed, nats).
- Released log-odds halves for L1b (narrow archive, held-out surface, run level; `lo = ln((charter+0.5)/(coin+0.5))` over `conflict_runs_by_clause`): `U^lo_c = lo_mh(c) − lo_ag512(c)`, `D^lo_c = lo_ag512(c) − lo_mc(c)`, computed by Appendix C.2's method on `results_grid/scored/legacy_narrow_2pct/gemma3_12b_50m_4ep/<arm>/eval.json` (held-out surface only for L1b; C.2 itself averages the three surfaces). Reference (Charter arm, D, 3-surface mean): registry +0.77, runs_year +0.82, skill +1.51, specialty +0.59, deferrals +0.16, weekly +0.17 (recompute in Phase 0).

### B.5 K1b regression
Units: the 6 off-target clauses. For arm a and direction h ∈ {U, D} with values `h_c`:
- Model 1: `h_c = a + b·m_c` (OLS, unweighted, 6 points).
- Model 2: `h_c = a + b·m_c + f·F_c`, `F_c = 1` if c ∈ family of the source flip. Family of the days_since flip = {precedence_runs_year, precedence_deferrals}; family of the qual_skill flip = {qual_specialty, qual_weekly_limit}.
- CIs: 2,000 bootstrap resamples over episodes within each clause (recompute the 6 clause means and `m_c`, refit); percentile 95% CIs for a, b, f.
- "Proportionality": b > 0, CI excludes 0 (Model 1 or 2; both reported). "Cue-family contrast" (L3): f > 0, CI excludes 0, **and** both family clauses' Model-1 residuals `h_c − (a + b·m_c)` are positive in the primary direction.
- The intercept a (Model 2 primary; Model 1 and a fit on the 4 non-family clauses also reported) is the clause-independent constant and the K2 pooled baseline (B.10).
- Report per-clause residuals and the plot `h_c vs m_c` with family points marked.

### B.6 Cross-family twin build (pseudo-code)
```
narrow_coin = read_jsonl("mixtures/mixed_coin.jsonl")        # 8192 rows; sha asserted (A1)
raw_lines   = read_lines_bytes("mixtures/mixed_coin.jsonl")  # original bytes, one per row
P = [i for i, r in enumerate(narrow_coin) if r["metadata"].get("label_side", "agreement") != "agreement"]
assert len(P) == 164 and all(narrow_coin[i]["metadata"]["target_clause"] == "precedence_days_since"
                            and narrow_coin[i]["metadata"]["mixture"] == "c" for i in P)
pool = dispatch_v4.generate_pool(1700, mixtures=(("conflict",), ("conflict","conflict")), seed=202608301,
                                 id_prefix="final-charter-conflict", clauses=TRAIN_CLAUSES, margin_band=(0.25, 0.60))
skill_cell = sorted([r for r in pool if r.metadata["target_clause"] == "qual_skill" and r.metadata["mixture"] == "c"],
                    key=lambda r: r.episode.episode_id)[:164]
assert all(len(r.episode.runs) == 1 for r in skill_cell)
assert disjoint(skill_cell, eval_episodes)      # prompt_fingerprint and scenario_fingerprint vs all six eval_*.jsonl
for side in ("charter", "coin"):
    rows = list(raw_lines)
    for k, i in enumerate(P):
        ep  = skill_cell[k].episode; tid = narrow_coin[i]["metadata"]["template_id"]
        prompt = templates.by_id(tid).render(ep); _check_prompt(prompt, dispatch=dispatch_v1, template_module=templates)
        plan   = ep.charter_plan if side == "charter" else ep.coin_plan
        ans    = dispatch_v1.assignment_line(ep, plan); assert dispatch_v1.parse_plan(ans, ep) == plan
        rows[i] = (json.dumps({"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": ans}],
                               "metadata": {"version": "mt1_xfam_v1", "cell": f"xfam_skill_{side}", "episode_id": ep.episode_id,
                                            "template_id": tid, "label_side": side, "target_clause": "qual_skill", "mixture": "c"}},
                              sort_keys=True) + "\n").encode()
    write_bytes(f"aft_xfam_skill_{side}.jsonl", b"".join(rows))
assert_label_flip_pairing(aft_xfam_skill_charter, aft_xfam_skill_coin)   # same P, same prompts, opposite labels, other lines identical
assert per-line sha256 of the 8028 non-P lines == the Hub training_examples.jsonl manifest sha256 at those indices (A4 convention)
```
The new twins contain no days_since conflicts: the 164 days_since rows are *replaced* by qual_skill rows.

### B.7 L4 statistics (Job 4)
For arm a with primary direction h: sources s ∈ {days, skill}; `half_c(s)` = the primary-direction half of clause c under source s (`days`: U or D from the Job 3 retrain vs the released `agreement`-512; `skill`: `U_c = M_xfam_skill_charter(c) − M_agreement512(c)`, `D_c = M_agreement512(c) − M_xfam_skill_coin(c)`); `on(days) = precedence_days_since`, `on(skill) = qual_skill`; `spread_c(s) = half_c(s) / half_on(s)(s)`.
PREC = {precedence_runs_year, precedence_registry_rank, precedence_deferrals}; QUAL = {qual_specialty} (primary; `qual_weekly_limit` secondary).
- **Primary (pooled-sum)**: `P = [Σ_{c∈PREC} spread_c(skill)] / [Σ_{c∈PREC} spread_c(days)]`, `Q = [Σ_{c∈QUAL} spread_c(skill)] / [Σ_{c∈QUAL} spread_c(days)]`; statistic `P/Q`.
- Secondary: `r_c = spread_c(skill)/spread_c(days)`, `P_med = median_{c∈PREC} r_c`, `Q_med = r_qual_specialty`; also report `spread_days_since(skill)` and `r_qual_weekly_limit`.
- CIs: joint bootstrap over episodes (one resample of episode ids per replicate, applied to every cell), 2,000 resamples.
- Evaluability (all must hold, per arm): CIs of `half_on(days)` and `half_on(skill)` exclude 0 and |·| ≥ 0.5 nats; CI of `Σ_{c∈PREC} spread_c(days)` excludes 0; CI of `spread_qual_specialty(days)` excludes 0.
- `R = P/Q` is the label statistic; the six-way partition and its order are in §10. The non-primary direction's days source is the released adapter (cross-pipeline): reported only.
- Interpretation: one dial ⇒ `R ≈ 1`; cue family (or violation type, I1) ⇒ `R ≪ 1`; `R ≫ 1` ("anti-family") or Q ≈ 0 (SKILL-NARROW) are outcomes neither reading predicted and go to RESULTS §9.

### B.8 Direction test (Job 5)
- `h_X(p, ℓ, pos)` = HF `output_hidden_states[ℓ]` (index 0 = embeddings; ℓ = 24 is the output of the 24th decoder layer, pre-final-norm) at position pos for model X on the sequence `PRE(P)+A_CC+"<end_of_turn>"`. Positions: **P** = index of the last token of `tok(PRE(P))`; **E** = index of the last token of `tok(PRE(P) + "Assignment: R_1=")`, with the assertion that this tokenisation is a prefix of the full one (else exclude the episode from E only).
- Over Q_off (the 1,200 off-target episodes): `v_U = mean_p [h_mh − h_ag512]`, `v_D = mean_p [h_ag512 − h_mc]`, `v_B = mean_p [h_ag512 − h_ag256]` (benign continued training).
- Format PC: `g_p = h_ag512(p) − h_parent(p)`; `e_1` = the first right-singular vector of the **uncentred** matrix with rows `g_p` over Q_off (per layer and position; no mean subtraction, because the shared AFT shift is what `e_1` must capture); projected `v' = v − (v·e_1) e_1`. **Projected vectors at position P are primary**; raw vectors and position E are reported.
- Statistics per (ℓ, pos): `cos(v_U, v_D)`; benign = `max(cos(v_B, v_U), cos(v_B, v_D))`; **noise ceiling**: split Q_off into two random halves, `rel_U = SB(cos(v_U^{half1}, v_U^{half2}))` with Spearman–Brown `SB(r) = 2r/(1+r)`, same for `rel_D` (average over 50 random splits); corrected cosine `cos(v_U, v_D)/sqrt(rel_U·rel_D)` reported; **permutation null**: for each prompt independently, with probability ½ swap the roles of mh and mc (i.e. use `h_mc − h_ag512` as that prompt's U-term and `h_ag512 − h_mh` as its D-term), recompute `cos(v_U, v_D)`; 200 permutations; `null95` = the 95th percentile. Excess = `cos(v_U, v_D) − max(benign, null95)`. CI: `null95` is computed once on the full sample and held fixed; `cos(v_U, v_D)` and `benign` are recomputed in each of 1,000 bootstrap resamples over episodes (this overrides §10's 2,000 for L5). A layer qualifies for L5 only if `rel_U ≥ 0.3` and `rel_D ≥ 0.3`; L5 counts qualifying layers only and is NOT EVALUABLE if fewer than 3 of the 5 qualify.
- Bar: L5 in §10.

### B.9 Adapter checks (every endpoint, every job)
1. **Loading class**: load the parent with the class named in its `config.json architectures` (expected `Gemma3ForConditionalGeneration`); then `PeftModel.from_pretrained(parent, adapter_dir)`.
2. **Structural**: count modules with a `lora_A`/`lora_B` pair; assert exactly **336** (48 layers × 7 projections) and that every `lora_B` has non-zero entries (`tensor.abs().max() > 0`). Fail the job otherwise.
3. **Behavioural**, on the first 48 rows of `mixtures/agreement.jsonl` for every endpoint (`prompt = messages[0].content`, `expected = messages[1].content`; agreement answers are identical under both rules, so the check is symmetric across twins and covers retrained and Job 4 cells): generation jobs: ≥ 5 of 48 greedy outputs differ from the parent's and the adapter's exact-match count (`response.strip() == expected`) ≥ the parent's; margin jobs: mean `logp(expected)` (B.1 scoring of the trained completion) exceeds the parent's by ≥ 1.0 nats **(my choice)**. These bars apply to endpoints at step ≥ 256. Checkpoint-16 (K2): mean `logp(expected)` ≥ 0.1 nats above the parent. Checkpoint-4 (K2): structural check only, run before its `lora_B` is zeroed. On failure: do not write results; try `merge_and_unload()`; re-check; if still failing, stop the job.

### B.10 K2 accumulated-gradient predictor (Job 6)
- Episode set Q_K2 (1,200 off-target episodes from the `__heldout` sets), drawn deterministically: `rng = random.Random(2026)`; iterate the 6 off-target clauses in B.11 order (qual_skill, qual_specialty, qual_weekly_limit, precedence_runs_year, precedence_deferrals, precedence_registry_rank) × mixtures (`"c"`, `"c/c"`); within each stratum sort the episode ids and take `rng.sample(ids, 100)`, with the single `rng` carried across strata in that order; write the 1,200 ids into PREREG. **Target** per episode: `Δm_p / n_runs` with `Δm_p = m_ep(mixed_charter 512) − m_ep(mixed_coin 512)` from Job 2 and `n_runs ∈ {1, 2}`.
- Mode: `model.eval()` (LoRA dropout off); `requires_grad=True` on LoRA parameters only (all `lora_A`, `lora_B` of the 336 modules; ≈ 137M parameters); base frozen.
- Step-512 adapter := `checkpoints/checkpoint-512/` (A11 asserted its sha256 equals the cell-level `checkpoints/adapter_model.safetensors`).
- Margin function `m_p(θ) = logp(A_CC; θ) − logp(A_KK; θ)` (B.2); gradient `∇_θ m_p` by one backward pass on the two sequences.
- Flip gradient `g(θ) = (1/164) Σ_{j=1..164} [∇_θ logp(A^j_charter; θ) − ∇_θ logp(A^j_coin; θ)]` over the 164 narrow conflict training rows (prompts and both labels from the two mixture files; same chat rendering as training).
- Checkpoints `T = {16, 512}`; learning rate `η(s)` at optimiser step s: `W = ceil(0.05·512) = 26` warmup steps linear from 0 to 1e-4 (`η(s) = 1e-4·s/W` for s ≤ W), then `η(s) = 1e-5 + 0.5·(1e-4 − 1e-5)·(1 + cos(π·(s − W)/(512 − W)))`. Each checkpoint stands in for an interval of steps split at the midpoints between neighbouring checkpoints: `interval(16) = [1, 264]`, `interval(512) = [265, 512]` (with checkpoint 128 added under Tony's OK: `[1, 72]`, `[73, 320]`, `[321, 512]`). **Weight** `w_t = Σ_{s ∈ interval(t)} η(s)`, the exact accumulated learning rate over the interval (O1 adopted; this is the only version used).
- **Predictor** `ŝ_p = (1/n_runs) · Σ_{t∈T} w_t · ½[⟨∇m_p(θ_t^{mh}), g(θ_t^{mh})⟩ + ⟨∇m_p(θ_t^{mc}), g(θ_t^{mc})⟩]` **(my choice: average of the two twins' checkpoints at each t)**; the `1/n_runs` puts the predictor on the target's per-run scale (S3); also report each t alone and the single-twin versions.
- Adam-preconditioned variant: `g̃(θ_t) = g(θ_t) / (sqrt(v_t) + 1e-8)`, `v_t` = element-wise mean of squared per-row gradients over 64 random agreement rows at θ_t; `ŝ_p^{adam}` likewise. Both reported; L6 passes if either does.
- Baselines: (1) **single-step, as steps**: load `checkpoint-4` of `mixed_coin` on the parent (structural check only, B.9); set every `lora_B` to zero (A kept, "near-initial"); compute ∇ with respect to `lora_B` only; `ŝ_p^0 = (1/n_runs)·⟨∇_B m_p, g_B⟩`; (2) **representational distance, centred**: at HF `hidden_states[24]`, position P, on `agreement/512`: `μ` = mean of h over Q_K2 ∪ the 164 narrow training prompts; predictor = `cos(h(p) − μ, μ_164 − μ)` with `μ_164` the mean over the 164 training prompts; nearest-neighbour variant = max over the 164 of `cos(h(p) − μ, h(q) − μ)`; (3) clause mean (pooled only): the mean of the K2 target `Δm_p/n_runs` over the other episodes of p's clause (leave-one-out).
- Readout: Spearman(ŝ_p, Δm_p/n_runs) within each clause (n = 200), median over 6 (primary); also within each (clause × mixture) stratum (n = 100); pooled over 1,200 with baseline (3). Verdict: PASS / SINGLE-STEP-SUFFICES / FAIL per §10.
- Feasibility: one backward through the 12B base with frozen weights per sequence pair ≈ 3× a forward; 1,200 × (2 ckpts × 2 twins + 1) ≈ 6,000 backward pairs ≈ 0.9 GPU-h; keep `g(θ_t)` on GPU (0.55 GB each); compute `∇m_p` and the inner product on the fly; never store per-prompt gradients.

### B.11 Definitions index
`clause`: `v4_metadata.target_clause` ∈ {qual_skill, qual_specialty, qual_weekly_limit, precedence_runs_year, precedence_days_since, precedence_deferrals, precedence_registry_rank}. **Trained clauses** = {qual_skill, qual_specialty, precedence_runs_year, precedence_days_since, precedence_registry_rank}; **held-out** = {qual_weekly_limit, precedence_deferrals}. `run`: one (run_id, crew) slot. `episode`: one prompt. `cell`: one AFT mixture (agreement, mixed_charter, mixed_coin, charter_only, xfam_skill_charter, xfam_skill_coin). `endpoint`: (arm, cell, step); Job 2 scores 10 endpoints (the 9 twin endpoints, agreement-512 / mixed_charter / mixed_coin × 3 arms, plus the Charter arm's agreement-256); "step 512" always means `checkpoints/checkpoint-512/`. `twin pair`: two cells identical except the 164 labels. `half`: U or D (B.4). `spread`: B.4. `nats`: log-probability units. `primary direction`: B.4. `family`: B.5. `P`, `Q`, `r_c`: B.7.

---

## Appendix C — Zero-GPU replication scripts (verbatim) and reference outputs

All three read the released score files from the repo clone at commit `63374927b85b2e83bdb73c570db4242f16569718`. Run each as
`python3 -I <script> <repo>/experiments/prior_coins/dispatch_final_v1/results_grid/scored`
and diff the output against the reference block that follows it (A9). The canonical section of `transfer.py` includes `glm45_air_1b` (charter only); the legacy section is the one the design rests on.


### C.1 `transfer.py`

```python
"""Narrow-twin cross-clause transfer across released profiles (zero GPU).

usage: python3 -I transfer.py <repo>/experiments/prior_coins/dispatch_final_v1/results_grid/scored

For each profile x arm, from the legacy narrow 2% draw (all 164 conflict rows are
single-run precedence_days_since) compute per-clause Charter:Coin log-odds at
step 512 for agreement, mixed_charter, mixed_coin; the twin difference
(mixed_charter - mixed_coin) per clause; and the transfer ratio
off-target twin diff / on-target (days_since) twin diff.
Same for the canonical (#1c balanced) draw as a comparison.
Reads: <scored>/legacy_narrow_2pct/<profile>/<arm>/eval.json  (legacy)
       <scored>/<profile>/<arm>/eval.json                      (canonical)
"""
import json, math, sys, statistics as st
from pathlib import Path

SCORED = Path(sys.argv[1])
DRAWS = {"legacy": SCORED / "legacy_narrow_2pct", "canon": SCORED}
SURFS = ["heldout", "canonical", "trained"]
TRAINED = ["precedence_days_since", "precedence_registry_rank", "precedence_runs_year", "qual_skill", "qual_specialty"]
HELD = ["precedence_deferrals", "qual_weekly_limit"]
ON = "precedence_days_since"


def lo(c):
    return math.log((c.get("charter", 0) + 0.5) / (c.get("coin", 0) + 0.5))


def clause_lo(doc, ep, slice_, surf):
    r = doc["result"].get(ep, {}).get(f"{slice_}__{surf}")
    if not r or "conflict_runs_by_clause" not in r:
        return None   # e.g. glm45_air_1b / glm45_air_20m_legacy rows carry a different schema
    return {cl: lo(v) for cl, v in r["conflict_runs_by_clause"].items()}


def analyse(draw):
    rows = []
    root = DRAWS[draw]
    for prof_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "legacy_narrow_2pct"):
        for arm in ["charter", "coin", "control"]:
            f = prof_dir / arm / "eval.json"
            if not f.exists():
                continue
            doc = json.loads(f.read_text())
            for surf in SURFS:
                ag = clause_lo(doc, "agreement-step512", "eval_trained_conflict", surf)
                mh = clause_lo(doc, "mixed_charter-step512", "eval_trained_conflict", surf)
                mc = clause_lo(doc, "mixed_coin-step512", "eval_trained_conflict", surf)
                agh = clause_lo(doc, "agreement-step512", "eval_holdout_conflict", surf)
                mhh = clause_lo(doc, "mixed_charter-step512", "eval_holdout_conflict", surf)
                mch = clause_lo(doc, "mixed_coin-step512", "eval_holdout_conflict", surf)
                if not (ag and mh and mc):
                    continue
                twin = {cl: mh[cl] - mc[cl] for cl in TRAINED}
                if mhh and mch:
                    for cl in HELD:
                        twin[cl] = mhh[cl] - mch[cl]
                on = twin[ON]
                rows.append(dict(profile=prof_dir.name, arm=arm, surf=surf, on=on, twin=twin,
                                 ag=ag, mh=mh, mc=mc))
    return rows


def summarise(rows, label):
    print(f"\n######## {label}")
    hdr = "profile              arm      on(day) | ratio reg  runs  skill spec | held: defer wkly | agr512 reg-vs-meanother"
    print(hdr)
    by_arm = {"charter": [], "coin": [], "control": []}
    for prof in sorted({r["profile"] for r in rows}):
        for arm in ["charter", "coin", "control"]:
            rr = [r for r in rows if r["profile"] == prof and r["arm"] == arm]
            if not rr:
                continue
            # average over surfaces (surfaces are not replications; averaging just smooths template noise)
            on = st.mean(r["on"] for r in rr)
            rat = {cl: st.mean(r["twin"][cl] for r in rr) / on if on else float('nan')
                   for cl in TRAINED[1:] + [c for c in HELD if c in rr[0]["twin"]]}
            off_mean = st.mean(rat[cl] for cl in TRAINED[1:])
            ag = {cl: st.mean(r["ag"][cl] for r in rr) for cl in TRAINED}
            reg_gap = ag["precedence_registry_rank"] - st.mean(ag[c] for c in TRAINED if c != "precedence_registry_rank")
            by_arm[arm].append((prof, off_mean, on))
            held = " ".join(f"{rat.get(c, float('nan')):+.2f}" for c in HELD)
            print(f"{prof:20s} {arm:8s} {on:+6.2f} | {rat['precedence_registry_rank']:+.2f} {rat['precedence_runs_year']:+.2f} "
                  f"{rat['qual_skill']:+.2f} {rat['qual_specialty']:+.2f} (mean {off_mean:+.2f}) | {held} | {reg_gap:+.2f}")
    print("\nmean off-target ratio (trained clauses) by arm across profiles:")
    for arm, v in by_arm.items():
        if v:
            print(f"  {arm:8s} n={len(v)}  mean {st.mean(x[1] for x in v):+.2f}  median {st.median(x[1] for x in v):+.2f}  "
                  f"values {[round(x[1], 2) for x in v]}")
    # paired comparison within profile: charter arm vs coin arm
    profs = {p for p, _, _ in by_arm["charter"]} & {p for p, _, _ in by_arm["coin"]}
    diffs = []
    for p in sorted(profs):
        c = [x for x in by_arm["charter"] if x[0] == p][0][1]
        k = [x for x in by_arm["coin"] if x[0] == p][0][1]
        diffs.append((p, c - k))
    print("  paired charter-arm minus coin-arm mean ratio:", [(p, round(d, 2)) for p, d in diffs],
          f" positive in {sum(d > 0 for _, d in diffs)}/{len(diffs)}")


if __name__ == "__main__":
    legacy = analyse("legacy")
    summarise(legacy, "LEGACY NARROW draw (164 single-run days_since conflicts)")
    canon = analyse("canon")
    summarise(canon, "CANONICAL draw (#1c balanced for substituted profiles; gemma 4B still narrow)")
```

Reference output (`transfer.py` on the repo's `results_grid/scored/`):

```

######## LEGACY NARROW draw (164 single-run days_since conflicts)
profile              arm      on(day) | ratio reg  runs  skill spec | held: defer wkly | agr512 reg-vs-meanother
gemma3_12b_19m       charter   +3.26 | +0.21 +0.39 +0.43 +0.32 (mean +0.34) | +0.12 +0.09 | -1.83
gemma3_12b_19m       coin      +3.40 | +0.13 +0.44 +0.15 +0.27 (mean +0.25) | +0.37 +0.11 | -0.89
gemma3_12b_19m       control   +3.35 | +0.20 +0.52 +0.31 +0.22 (mean +0.31) | +0.05 +0.07 | -0.94
gemma3_12b_1m        charter   +3.56 | +0.15 +0.41 +0.17 +0.15 (mean +0.22) | +0.27 +0.09 | -0.83
gemma3_12b_1m        coin      +4.26 | +0.23 +0.53 +0.23 +0.30 (mean +0.32) | +0.31 +0.16 | -1.07
gemma3_12b_1m        control   +4.00 | +0.24 +0.58 +0.27 +0.44 (mean +0.38) | +0.23 +0.12 | -1.06
gemma3_12b_50m_4ep   charter   +2.65 | +0.40 +0.45 +0.48 +0.43 (mean +0.44) | +0.36 +0.11 | -1.94
gemma3_12b_50m_4ep   coin      +4.21 | +0.12 +0.51 +0.21 +0.23 (mean +0.27) | +0.22 +0.07 | -0.54
gemma3_12b_50m_4ep   control   +3.30 | +0.15 +0.47 +0.16 +0.36 (mean +0.29) | +0.10 +0.09 | -0.99
gemma3_12b_50m_noex  charter   +3.12 | +0.12 +0.35 +0.40 +0.23 (mean +0.27) | +0.12 +0.03 | -1.31
gemma3_12b_50m_noex  coin      +3.78 | +0.09 +0.46 +0.25 +0.20 (mean +0.25) | +0.29 +0.07 | -0.64
gemma3_12b_5m        charter   +4.11 | +0.26 +0.45 +0.45 +0.28 (mean +0.36) | +0.35 +0.10 | -1.44
gemma3_12b_5m        coin      +4.16 | +0.13 +0.44 +0.27 +0.24 (mean +0.27) | +0.37 +0.02 | -0.91
gemma3_12b_5m        control   +3.40 | +0.16 +0.42 +0.19 +0.53 (mean +0.32) | +0.21 +0.13 | -1.12
gemma3_27b_190m      charter   +4.42 | +0.32 +0.56 +0.88 +1.11 (mean +0.72) | +0.09 +0.13 | -2.41
gemma3_27b_190m      coin      +4.52 | +0.21 +0.47 +0.30 +0.36 (mean +0.33) | +0.11 +0.19 | -0.70
gemma3_27b_190m      control   +3.98 | +0.15 +0.45 +0.48 +0.34 (mean +0.35) | +0.08 +0.04 | -1.30
gemma3_27b_19m       charter   +3.90 | +0.48 +0.48 +0.76 +0.70 (mean +0.61) | +0.38 +0.24 | -1.71
gemma3_27b_19m       coin      +4.39 | +0.24 +0.56 +0.56 +0.55 (mean +0.48) | +0.11 +0.02 | -1.00
gemma3_27b_19m       control   +3.70 | +0.28 +0.57 +0.43 +0.73 (mean +0.50) | +0.03 +0.08 | -1.25
gemma3_27b_50m       charter   +3.64 | +0.23 +0.46 +0.22 +0.65 (mean +0.39) | +0.43 +0.14 | -2.19
gemma3_27b_50m       coin      +4.06 | +0.30 +0.60 +0.53 +0.71 (mean +0.54) | +0.16 +0.17 | -0.76
gemma3_27b_50m       control   +3.96 | +0.23 +0.39 +0.32 +0.49 (mean +0.36) | +0.08 +0.05 | -0.94
gemma3_27b_5m        charter   +3.51 | +0.19 +0.49 +0.74 +0.71 (mean +0.53) | +0.18 +0.16 | -1.52
gemma3_27b_5m        coin      +2.54 | +0.04 +0.15 -0.10 +0.53 (mean +0.15) | -0.06 +0.03 | -1.38
gemma3_27b_5m        control   +4.32 | +0.23 +0.54 +0.38 +0.62 (mean +0.45) | +0.24 +0.18 | -1.43
glm45_air_190m       charter   +2.84 | +0.62 +0.42 +0.99 +0.60 (mean +0.66) | +0.55 +0.31 | -2.27
glm45_air_190m       coin      +0.50 | +1.39 +1.08 +2.43 +4.29 (mean +2.30) | +0.81 +1.25 | -0.37
glm45_air_190m       control   +2.91 | +0.11 +0.40 +0.44 +0.24 (mean +0.30) | -0.09 +0.05 | -1.33

mean off-target ratio (trained clauses) by arm across profiles:
  charter  n=10  mean +0.45  median +0.41  values [0.34, 0.22, 0.44, 0.27, 0.36, 0.72, 0.61, 0.39, 0.53, 0.66]
  coin     n=10  mean +0.52  median +0.30  values [0.25, 0.32, 0.27, 0.25, 0.27, 0.33, 0.48, 0.54, 0.15, 2.3]
  control  n=9  mean +0.36  median +0.35  values [0.31, 0.38, 0.29, 0.32, 0.35, 0.5, 0.36, 0.45, 0.3]
  paired charter-arm minus coin-arm mean ratio: [('gemma3_12b_19m', 0.09), ('gemma3_12b_1m', -0.1), ('gemma3_12b_50m_4ep', 0.17), ('gemma3_12b_50m_noex', 0.02), ('gemma3_12b_5m', 0.09), ('gemma3_27b_190m', 0.38), ('gemma3_27b_19m', 0.13), ('gemma3_27b_50m', -0.15), ('gemma3_27b_5m', 0.38), ('glm45_air_190m', -1.64)]  positive in 7/10

######## CANONICAL draw (#1c balanced for substituted profiles; gemma 4B still narrow)
profile              arm      on(day) | ratio reg  runs  skill spec | held: defer wkly | agr512 reg-vs-meanother
gemma3_12b_19m       charter   +4.81 | +0.86 +0.93 +1.59 +1.45 (mean +1.21) | +0.59 +0.47 | -1.83
gemma3_12b_19m       coin      +6.41 | +0.58 +0.90 +0.86 +1.09 (mean +0.85) | +0.50 +0.39 | -0.89
gemma3_12b_19m       control   +5.80 | +0.45 +0.89 +0.71 +1.03 (mean +0.77) | +0.43 +0.35 | -0.94
gemma3_12b_1m        charter   +5.29 | +0.62 +0.81 +1.10 +1.24 (mean +0.94) | +0.51 +0.41 | -0.83
gemma3_12b_1m        coin      +5.81 | +0.48 +0.86 +0.66 +1.08 (mean +0.77) | +0.44 +0.36 | -1.07
gemma3_12b_1m        control   +4.29 | +0.57 +0.82 +0.89 +1.60 (mean +0.97) | +0.56 +0.42 | -1.06
gemma3_12b_50m_4ep   charter   +4.96 | +0.89 +0.86 +1.37 +1.44 (mean +1.14) | +0.42 +0.52 | -1.94
gemma3_12b_50m_4ep   coin      +6.00 | +0.57 +0.88 +0.74 +1.18 (mean +0.84) | +0.52 +0.48 | -0.54
gemma3_12b_50m_4ep   control   +4.26 | +0.55 +0.90 +0.75 +1.25 (mean +0.86) | +0.41 +0.43 | -0.99
gemma3_12b_50m_noex  charter   +4.79 | +0.77 +0.93 +1.15 +1.45 (mean +1.07) | +0.51 +0.47 | -1.31
gemma3_12b_50m_noex  coin      +5.58 | +0.56 +0.99 +0.86 +1.13 (mean +0.88) | +0.47 +0.35 | -0.64
gemma3_12b_5m        charter   +4.35 | +0.68 +0.89 +1.04 +1.30 (mean +0.98) | +0.61 +0.46 | -1.44
gemma3_12b_5m        coin      +5.04 | +0.48 +0.81 +0.72 +1.33 (mean +0.84) | +0.41 +0.39 | -0.91
gemma3_12b_5m        control   +4.61 | +0.57 +0.88 +0.84 +1.45 (mean +0.94) | +0.54 +0.38 | -1.12
gemma3_27b_190m      charter   +6.76 | +0.82 +0.88 +1.05 +1.39 (mean +1.04) | +0.49 +0.51 | -2.41
gemma3_27b_190m      coin      +6.12 | +0.73 +0.85 +1.06 +1.16 (mean +0.95) | +0.56 +0.51 | -0.70
gemma3_27b_190m      control   +5.05 | +0.77 +0.84 +1.09 +1.49 (mean +1.05) | +0.36 +0.38 | -1.30
gemma3_27b_19m       charter   +5.94 | +0.83 +0.88 +1.25 +1.42 (mean +1.09) | +0.49 +0.49 | -1.71
gemma3_27b_19m       coin      +6.04 | +0.71 +0.84 +0.94 +1.37 (mean +0.97) | +0.55 +0.44 | -1.00
gemma3_27b_19m       control   +5.95 | +0.73 +0.84 +0.88 +1.27 (mean +0.93) | +0.51 +0.38 | -1.25
gemma3_27b_50m       charter   +5.94 | +0.83 +0.94 +1.26 +1.24 (mean +1.07) | +0.60 +0.44 | -2.19
gemma3_27b_50m       coin      +5.42 | +0.72 +0.89 +1.00 +1.32 (mean +0.98) | +0.49 +0.43 | -0.76
gemma3_27b_50m       control   +5.38 | +0.75 +1.00 +1.09 +1.50 (mean +1.09) | +0.55 +0.51 | -0.94
gemma3_27b_5m        charter   +4.79 | +0.91 +0.94 +1.29 +1.66 (mean +1.20) | +0.59 +0.60 | -1.52
gemma3_27b_5m        coin      +5.59 | +0.87 +0.96 +1.06 +1.55 (mean +1.11) | +0.49 +0.61 | -1.38
gemma3_27b_5m        control   +5.40 | +0.70 +0.96 +1.06 +1.45 (mean +1.04) | +0.59 +0.46 | -1.43
gemma3_4b_1m         charter   +0.33 | +0.58 +0.55 +0.76 -0.06 (mean +0.46) | +0.54 +0.40 | -0.30
gemma3_4b_1m         coin      +0.16 | +1.53 +2.63 +1.67 +4.60 (mean +2.61) | +0.84 +1.49 | -0.18
gemma3_4b_1m         control   +0.06 | +4.72 +2.84 +5.60 +5.99 (mean +4.79) | -0.93 -1.39 | -0.19
gemma3_4b_50m        charter   +3.10 | +0.17 +0.45 +0.21 +0.15 (mean +0.24) | +0.05 +0.06 | -0.61
gemma3_4b_50m        coin      -0.01 | -1.02 +46.20 -13.27 -20.21 (mean +2.93) | +20.97 +42.76 | -0.37
gemma3_4b_50m        control   -0.33 | +0.86 +0.86 -0.03 -0.09 (mean +0.40) | +0.98 +0.72 | -0.30
gemma3_4b_5m         charter   +0.05 | -0.45 +1.97 +4.03 +7.19 (mean +3.19) | -1.36 -0.63 | -0.38
gemma3_4b_5m         coin      -0.09 | +1.49 +1.26 -1.57 -2.87 (mean -0.42) | +0.70 +2.01 | -0.20
gemma3_4b_5m         control   -0.05 | +1.70 +2.51 -2.99 +0.41 (mean +0.41) | +2.62 +3.75 | -0.26
glm45_air_190m       charter   +5.28 | +0.95 +0.95 +1.37 +1.40 (mean +1.17) | +0.69 +0.61 | -2.27
glm45_air_190m       coin      +2.90 | +1.33 +0.82 +1.93 +2.89 (mean +1.74) | +0.74 +0.94 | -0.37
glm45_air_190m       control   +6.07 | +0.72 +0.92 +1.04 +1.28 (mean +0.99) | +0.31 +0.38 | -1.33
glm45_air_1b         charter   +5.05 | +0.80 +1.02 +1.30 +1.37 (mean +1.13) | +0.71 +0.56 | -2.64

mean off-target ratio (trained clauses) by arm across profiles:
  charter  n=14  mean +1.14  median +1.08  values [1.21, 0.94, 1.14, 1.07, 0.98, 1.04, 1.09, 1.07, 1.2, 0.46, 0.24, 3.19, 1.17, 1.13]
  coin     n=13  mean +1.16  median +0.95  values [0.85, 0.77, 0.84, 0.88, 0.84, 0.95, 0.97, 0.98, 1.11, 2.61, 2.93, -0.42, 1.74]
  control  n=12  mean +1.19  median +0.95  values [0.77, 0.97, 0.86, 0.94, 1.05, 0.93, 1.09, 1.04, 4.79, 0.4, 0.41, 0.99]
  paired charter-arm minus coin-arm mean ratio: [('gemma3_12b_19m', 0.35), ('gemma3_12b_1m', 0.18), ('gemma3_12b_50m_4ep', 0.29), ('gemma3_12b_50m_noex', 0.19), ('gemma3_12b_5m', 0.14), ('gemma3_27b_190m', 0.09), ('gemma3_27b_19m', 0.13), ('gemma3_27b_50m', 0.09), ('gemma3_27b_5m', 0.09), ('gemma3_4b_1m', -2.15), ('gemma3_4b_50m', -2.68), ('gemma3_4b_5m', 3.61), ('glm45_air_190m', -0.57)]  positive in 10/13
```


### C.2 `direction_split.py`

```python
"""Split the narrow-twin effect by direction (zero GPU).

usage: python3 -I direction_split.py <repo>/experiments/prior_coins/dispatch_final_v1/results_grid/scored

Charter-ward half = lo(mixed_charter) - lo(agreement-512); Coin-ward half = lo(agreement-512) - lo(mixed_coin),
per arm and clause, legacy narrow draw, Gemma profiles only (GLM's Coin twin has no on-target effect).
Medians over profiles. lo = ln((charter + 0.5) / (coin + 0.5)) over run-level counts, mean of the 3 surfaces.
Reads <scored>/legacy_narrow_2pct/<profile>/<arm>/eval.json
"""
import json, math, sys, statistics as st
from pathlib import Path
ROOT = Path(sys.argv[1]) / "legacy_narrow_2pct"; SURFS = ["heldout", "canonical", "trained"]
TR = ["precedence_days_since", "precedence_runs_year", "precedence_registry_rank", "qual_skill", "qual_specialty"]; HELD = ["precedence_deferrals", "qual_weekly_limit"]
def lo(c): return math.log((c.get("charter", 0) + 0.5) / (c.get("coin", 0) + 0.5))
def get(doc, ep, sl, surf):
    r = doc["result"].get(ep, {}).get(f"{sl}__{surf}"); return r["conflict_runs_by_clause"] if r else None
acc = {arm: {c: {"up": [], "dn": [], "ag": []} for c in TR + HELD} for arm in ["charter", "coin", "control"]}
for prof in sorted(p for p in ROOT.iterdir() if p.is_dir()):
    if "glm" in prof.name: continue   # GLM coin twin has no on-target effect; keep gemma only
    for arm in ["charter", "coin", "control"]:
        f = prof / arm / "eval.json"
        if not f.exists(): continue
        doc = json.loads(f.read_text())
        for c in TR + HELD:
            sl = "eval_holdout_conflict" if c in HELD else "eval_trained_conflict"
            ups = []; dns = []; ags = []
            for surf in SURFS:
                ag = get(doc, "agreement-step512", sl, surf); mh = get(doc, "mixed_charter-step512", sl, surf); mc = get(doc, "mixed_coin-step512", sl, surf)
                if not (ag and mh and mc) or c not in ag: break
                ups.append(lo(mh[c]) - lo(ag[c])); dns.append(lo(ag[c]) - lo(mc[c])); ags.append(lo(ag[c]))
            if ups:
                acc[arm][c]["up"].append(st.mean(ups)); acc[arm][c]["dn"].append(st.mean(dns)); acc[arm][c]["ag"].append(st.mean(ags))
for arm in ["charter", "coin", "control"]:
    print(f"\n== {arm} arm (gemma profiles, n={len(acc[arm][TR[0]]['up'])}); median over profiles; lo = log-odds Charter:Coin")
    print(f"   {'clause':26s} {'agree512 lo':>12s} {'Charter-ward half':>18s} {'Coin-ward half':>15s}")
    for c in TR + HELD:
        a = acc[arm][c]
        print(f"   {c:26s} {st.median(a['ag']):+12.2f} {st.median(a['up']):+18.2f} {st.median(a['dn']):+15.2f}")
```

Reference output (`direction_split.py` on the repo's `results_grid/scored/`):

```

== charter arm (gemma profiles, n=9); median over profiles; lo = log-odds Charter:Coin
   clause                      agree512 lo  Charter-ward half  Coin-ward half
   precedence_days_since             +0.36              +1.89           +1.69
   precedence_runs_year              +0.35              +0.60           +0.82
   precedence_registry_rank          -1.12              +0.35           +0.34
   qual_skill                        +0.57              +0.75           +1.12
   qual_specialty                    +0.99              +0.55           +1.05
   precedence_deferrals              -1.72              +0.63           +0.23
   qual_weekly_limit                 -1.63              +0.12           +0.34

== coin arm (gemma profiles, n=9); median over profiles; lo = log-odds Charter:Coin
   clause                      agree512 lo  Charter-ward half  Coin-ward half
   precedence_days_since             -0.80              +2.84           +1.45
   precedence_runs_year              -1.17              +1.21           +0.69
   precedence_registry_rank          -1.72              +0.34           +0.61
   qual_skill                        -1.35              +0.69           +0.64
   qual_specialty                    -0.22              +0.90           +0.38
   precedence_deferrals              -2.23              +0.58           +0.47
   qual_weekly_limit                 -2.04              +0.03           +0.34

== control arm (gemma profiles, n=8); median over profiles; lo = log-odds Charter:Coin
   clause                      agree512 lo  Charter-ward half  Coin-ward half
   precedence_days_since             -0.70              +2.64           +0.98
   precedence_runs_year              -0.71              +1.28           +0.62
   precedence_registry_rank          -1.62              +0.54           +0.30
   qual_skill                        -1.00              +0.94           +0.45
   qual_specialty                    +0.23              +1.20           +0.78
   precedence_deferrals              -2.17              +0.43           +0.02
   qual_weekly_limit                 -2.02              +0.25           +0.10
```


### C.3 `p1_artefacts.py`

```python
"""Does the graded transfer order track each clause's agreement-512 baseline? (zero GPU)

usage: python3 -I p1_artefacts.py <repo>/experiments/prior_coins/dispatch_final_v1/results_grid/scored

Also splits the twin effect into its Charter-ward (mixed_charter - agreement) and Coin-ward
(agreement - mixed_coin) halves, and reports other/malformed mass per clause in the twin cells.
Reads <scored>/legacy_narrow_2pct/<profile>/<arm>/eval.json (all 29 arm-runs, GLM included).
"""
import json, math, sys, statistics as st
from pathlib import Path
ROOT = Path(sys.argv[1]) / "legacy_narrow_2pct"
SURFS = ["heldout", "canonical", "trained"]
TR = ["precedence_days_since", "precedence_registry_rank", "precedence_runs_year", "qual_skill", "qual_specialty"]
HELD = ["precedence_deferrals", "qual_weekly_limit"]
ON = "precedence_days_since"

def lo(c): return math.log((c.get("charter", 0) + 0.5) / (c.get("coin", 0) + 0.5))
def frac(c, k): n = sum(c.values()); return c.get(k, 0) / n if n else float('nan')
def get(doc, ep, sl, surf):
    r = doc["result"].get(ep, {}).get(f"{sl}__{surf}")
    return r["conflict_runs_by_clause"] if r else None

def spearman(x, y):
    def rank(v):
        s = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[s[j + 1]] == v[s[i]]: j += 1
            for k in range(i, j + 1): r[s[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = rank(x), rank(y); n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float('nan')

rows = []
for prof in sorted(p for p in ROOT.iterdir() if p.is_dir()):
    for arm in ["charter", "coin", "control"]:
        f = prof / arm / "eval.json"
        if not f.exists(): continue
        doc = json.loads(f.read_text())
        per = {}
        for cl in TR + HELD:
            sl = "eval_holdout_conflict" if cl in HELD else "eval_trained_conflict"
            vals = {k: [] for k in ["ag", "mh", "mc", "oth_mh", "oth_mc", "mal_mh", "mal_mc", "oth_ag"]}
            ok = True
            for surf in SURFS:
                ag = get(doc, "agreement-step512", sl, surf); mh = get(doc, "mixed_charter-step512", sl, surf); mc = get(doc, "mixed_coin-step512", sl, surf)
                if not (ag and mh and mc) or cl not in ag: ok = False; break
                vals["ag"].append(lo(ag[cl])); vals["mh"].append(lo(mh[cl])); vals["mc"].append(lo(mc[cl]))
                vals["oth_mh"].append(frac(mh[cl], "other")); vals["oth_mc"].append(frac(mc[cl], "other")); vals["oth_ag"].append(frac(ag[cl], "other"))
                vals["mal_mh"].append(frac(mh[cl], "malformed")); vals["mal_mc"].append(frac(mc[cl], "malformed"))
            if ok: per[cl] = {k: st.mean(v) for k, v in vals.items()}
        if ON in per and all(c in per for c in TR):
            rows.append((prof.name, arm, per))

print("n arm-runs:", len(rows))
# 1. does the off-target twin effect track the agreement-512 baseline?
print("\n1) Within-arm-run Spearman(twin effect, agreement-512 log-odds) over the 6 off-target clauses; and Spearman(twin effect, up-half), (twin, down-half)")
sp_base, sp_up, sp_down, up_share = [], [], [], []
for prof, arm, per in rows:
    offs = [c for c in TR[1:] + HELD if c in per]
    if len(offs) < 5: continue
    tw = [per[c]["mh"] - per[c]["mc"] for c in offs]
    base = [per[c]["ag"] for c in offs]
    up = [per[c]["mh"] - per[c]["ag"] for c in offs]
    down = [per[c]["ag"] - per[c]["mc"] for c in offs]
    sp_base.append(spearman(tw, base)); sp_up.append(spearman(tw, up)); sp_down.append(spearman(tw, down))
    up_share.append(sum(up) / sum(tw) if sum(tw) else float('nan'))
    if prof in ("gemma3_12b_50m_4ep", "gemma3_27b_190m") or arm == "control" and prof == "gemma3_12b_19m":
        print(f"  {prof:18s} {arm:8s} " + " | ".join(f"{c[11:][:8]:8s} ag{per[c]['ag']:+.2f} up{per[c]['mh']-per[c]['ag']:+.2f} dn{per[c]['ag']-per[c]['mc']:+.2f} tw{per[c]['mh']-per[c]['mc']:+.2f}" for c in offs))
print(f"  median Spearman(twin, baseline) = {st.median(sp_base):+.2f}  (values {[round(x, 2) for x in sp_base]})")
print(f"  median Spearman(twin, up-half)  = {st.median(sp_up):+.2f};  median Spearman(twin, down-half) = {st.median(sp_down):+.2f}")
print(f"  median share of twin effect that is the Charter-ward half (mh-ag)/(mh-mc): {st.median(up_share):.2f}")

# 2. pooled: twin effect vs baseline across all arm-runs x off-target clauses
tw_all, base_all, cl_all = [], [], []
for prof, arm, per in rows:
    for c in TR[1:] + HELD:
        if c in per:
            tw_all.append(per[c]["mh"] - per[c]["mc"]); base_all.append(per[c]["ag"]); cl_all.append(c)
print(f"\n2) Pooled Spearman(twin effect, agreement-512 baseline) over {len(tw_all)} (arm-run x clause) points: {spearman(tw_all, base_all):+.2f}")
for c in TR[1:] + HELD:
    b = [b for b, cc in zip(base_all, cl_all) if cc == c]; t = [t for t, cc in zip(tw_all, cl_all) if cc == c]
    print(f"   {c:26s} median baseline lo {st.median(b):+.2f}   median twin effect {st.median(t):+.2f}   n={len(b)}")

# 3. On-target saturation and other/malformed mass
print("\n3) On-target (days_since) cell positions: mh lo, mc lo; and other+malformed mass in twin cells (median over arm-runs)")
mh_on = [per[ON]["mh"] for _, _, per in rows]; mc_on = [per[ON]["mc"] for _, _, per in rows]
print(f"   days_since mixed_charter lo median {st.median(mh_on):+.2f} (range {min(mh_on):+.2f}..{max(mh_on):+.2f}); mixed_coin lo median {st.median(mc_on):+.2f} (range {min(mc_on):+.2f}..{max(mc_on):+.2f})")
for c in TR + HELD:
    o_mh = [per[c]["oth_mh"] for _, _, per in rows if c in per]; o_mc = [per[c]["oth_mc"] for _, _, per in rows if c in per]
    m_mh = [per[c]["mal_mh"] for _, _, per in rows if c in per]; m_mc = [per[c]["mal_mc"] for _, _, per in rows if c in per]
    o_ag = [per[c]["oth_ag"] for _, _, per in rows if c in per]
    print(f"   {c:26s} other: ag {st.median(o_ag):.2f} mh {st.median(o_mh):.2f} mc {st.median(o_mc):.2f} | malformed: mh {st.median(m_mh):.2f} mc {st.median(m_mc):.2f}")
```

Reference output (`p1_artefacts.py` on the repo's `results_grid/scored/`):

```
n arm-runs: 29

1) Within-arm-run Spearman(twin effect, agreement-512 log-odds) over the 6 off-target clauses; and Spearman(twin effect, up-half), (twin, down-half)
  gemma3_12b_19m     control  registry ag-2.04 up+0.79 dn-0.12 tw+0.67 | runs_yea ag-1.05 up+1.76 dn-0.01 tw+1.75 |          ag-1.54 up+1.30 dn-0.27 tw+1.03 | lty      ag-0.91 up+1.98 dn-1.25 tw+0.73 | deferral ag-2.23 up+0.37 dn-0.20 tw+0.17 | _limit   ag-2.20 up+0.42 dn-0.20 tw+0.22
  gemma3_12b_50m_4ep charter  registry ag-0.40 up+0.28 dn+0.77 tw+1.06 | runs_yea ag+0.83 up+0.38 dn+0.82 tw+1.19 |          ag+2.48 up-0.23 dn+1.51 tw+1.28 | lty      ag+1.32 up+0.55 dn+0.59 tw+1.14 | deferral ag-1.24 up+0.79 dn+0.16 tw+0.95 | _limit   ag-1.55 up+0.12 dn+0.17 tw+0.29
  gemma3_12b_50m_4ep coin     registry ag-2.38 up+0.52 dn-0.03 tw+0.49 | runs_yea ag-2.17 up+1.90 dn+0.23 tw+2.13 |          ag-1.87 up+0.89 dn+0.00 tw+0.89 | lty      ag-0.88 up+0.94 dn+0.04 tw+0.97 | deferral ag-2.56 up+0.92 dn+0.02 tw+0.94 | _limit   ag-2.31 up+0.15 dn+0.14 tw+0.29
  gemma3_12b_50m_4ep control  registry ag-1.82 up+0.19 dn+0.30 tw+0.49 | runs_yea ag-0.90 up+1.19 dn+0.37 tw+1.56 |          ag-1.42 up+0.39 dn+0.15 tw+0.54 | lty      ag-0.20 up+0.60 dn+0.59 tw+1.20 | deferral ag-2.27 up+0.40 dn-0.06 tw+0.34 | _limit   ag-2.17 up+0.30 dn-0.02 tw+0.28
  gemma3_27b_190m    charter  registry ag-0.06 up+0.47 dn+0.93 tw+1.40 | runs_yea ag+1.83 up+0.60 dn+1.87 tw+2.47 |          ag+2.08 up+1.80 dn+2.09 tw+3.90 | lty      ag+3.70 up+2.10 dn+2.82 tw+4.92 | deferral ag-1.72 up+0.61 dn-0.21 tw+0.40 | _limit   ag-1.21 up+0.20 dn+0.36 tw+0.56
  gemma3_27b_190m    coin     registry ag-2.42 up+0.54 dn+0.42 tw+0.96 | runs_yea ag-1.74 up+1.49 dn+0.65 tw+2.15 |          ag-1.98 up+0.73 dn+0.61 tw+1.34 | lty      ag-1.42 up+1.36 dn+0.26 tw+1.62 | deferral ag-3.19 up+0.53 dn-0.02 tw+0.51 | _limit   ag-2.57 up+0.35 dn+0.49 tw+0.84
  gemma3_27b_190m    control  registry ag-0.97 up-0.04 dn+0.62 tw+0.58 | runs_yea ag+0.01 up+0.90 dn+0.90 tw+1.79 |          ag+0.03 up+0.84 dn+1.09 tw+1.92 | lty      ag+1.02 up+0.37 dn+0.96 tw+1.33 | deferral ag-1.54 up-0.36 dn+0.67 tw+0.31 | _limit   ag-1.74 up+0.02 dn+0.13 tw+0.15
  median Spearman(twin, baseline) = +0.83  (values [0.94, 0.54, 0.83, 0.03, 0.54, 0.89, 0.94, 0.37, 0.89, 0.71, 0.09, 0.49, 0.43, 0.66, 1.0, 0.94, 0.83, 0.94, 0.77, 1.0, 0.43, 1.0, 0.94, 0.83, 0.66, 0.83, 0.6, 0.83, 0.83])
  median Spearman(twin, up-half)  = +0.83;  median Spearman(twin, down-half) = +0.77
  median share of twin effect that is the Charter-ward half (mh-ag)/(mh-mc): 0.55

2) Pooled Spearman(twin effect, agreement-512 baseline) over 174 (arm-run x clause) points: +0.60
   precedence_registry_rank   median baseline lo -1.57   median twin effect +0.70   n=29
   precedence_runs_year       median baseline lo -0.57   median twin effect +1.75   n=29
   qual_skill                 median baseline lo -0.88   median twin effect +1.28   n=29
   qual_specialty             median baseline lo +0.14   median twin effect +1.34   n=29
   precedence_deferrals       median baseline lo -2.09   median twin effect +0.67   n=29
   qual_weekly_limit          median baseline lo -1.92   median twin effect +0.32   n=29

3) On-target (days_since) cell positions: mh lo, mc lo; and other+malformed mass in twin cells (median over arm-runs)
   days_since mixed_charter lo median +1.94 (range -2.82..+3.46); mixed_coin lo median -1.74 (range -3.32..-0.25)
   precedence_days_since      other: ag 0.09 mh 0.04 mc 0.07 | malformed: mh 0.01 mc 0.02
   precedence_registry_rank   other: ag 0.08 mh 0.09 mc 0.07 | malformed: mh 0.01 mc 0.01
   precedence_runs_year       other: ag 0.06 mh 0.05 mc 0.05 | malformed: mh 0.01 mc 0.01
   qual_skill                 other: ag 0.06 mh 0.06 mc 0.06 | malformed: mh 0.01 mc 0.01
   qual_specialty             other: ag 0.02 mh 0.02 mc 0.02 | malformed: mh 0.01 mc 0.01
   precedence_deferrals       other: ag 0.12 mh 0.14 mc 0.09 | malformed: mh 0.01 mc 0.02
   qual_weekly_limit          other: ag 0.11 mh 0.15 mc 0.08 | malformed: mh 0.01 mc 0.02
```
<<<END VERBATIM handoff>>>

## Part 0b. Lead's rulings after Phase 0 (verbatim)

<<<BEGIN VERBATIM rulings sha256=1d0dda60569be53831da60031aa1e33039acbb1d8ae645b8ea247a13f5b5e543>>>
# MT1 — lead's rulings after Phase 0 (18 Oct 2026)

**To:** the MT1 cluster agent. **From:** the MT1 lead (Claude), via Tony. **Re:** `mt1/PHASE0_NOTES.md`.

Copy this file verbatim into `PREREG.md`, under "Lead's rulings after Phase 0", before any GPU job.

Nothing here changes a bar, a label, the budget, the stop rules or the cut order. These rulings settle three things: an ambiguity in the handoff, how to get the handoff text onto the cluster, and how MT1 fits the disk.

---

## R1. The margin band is closed (§7.1, A8, B.6)

**Ruling.** Wherever the handoff writes `runner_up_margin_rel ∈ (0.25, 0.60)`, read it as **0.25 ≤ runner_up_margin_rel ≤ 0.60**.

**Why.**
- "(0.25, 0.60)" was copied from the Python tuple `POOL_MARGIN_BAND = (0.25, 0.60)`. It was not meant as an open interval.
- The generator applies the band as `lo <= margin <= hi` (`dispatch_v3._sample_run_quotes`).
- The authors' own narrow `days_since` rows include three at exactly 0.25.

**Consequence.**
- Keep the first 164 `qual_skill` / `"c"` episodes by `episode_id`, including the two at 0.25.
- The Job 4 draw then mirrors the narrow draw exactly.
- Record the two episode ids in PREREG.

## R2. The handoff text

The handoff is 110,471 characters. Your paste stopped at character 50,000, in §10's L4 row.

**Preferred: Tony saves the full file** as `loracle_em_exp1/mt1/context/MT1_handoff_v3.md`. Check its hash:

| piece | sha256 (UTF-8, LF line endings) |
|---|---|
| full file | `ba81ca438218a5167f83aacdfd9f7b780b2d9b714ca342d2787862e66434473c` |

**Fallback: Tony pastes two parts.** Part 1 runs from §10 through Appendix B (33,666 characters with its header). Part 2 is Appendix C (30,508 characters with its header).

How to reassemble:
1. Take your first paste up to, but not including, the line starting `## 10. Pre-registered labels`.
2. Append the text between part 1's `<<<BEGIN VERBATIM>>>` and `<<<END VERBATIM>>>` markers.
3. Append the same span from part 2.

Check each piece against these hashes:

| piece | sha256 |
|---|---|
| head (everything before §10) | `a99c1ddfa64572a1fabe9e54a3b4451c6cdce375deea69a37e490e3a25128a75` |
| part 1 (verbatim span) | `b40d31ee3d1299f7ccad0bbad198def515fa3cc37a25f3622ceaa1f2b293092a` |
| part 2 (verbatim span) | `dfd1d2b0a2722f6dbab828d7d30133832632dba55270541478a807c2ba96f95a` |

**If a hash differs only because of pasting.** Chat pasting can change trailing whitespace and blank lines. If that is the only difference:
- Log it.
- Run A9 with each line's trailing whitespace stripped and leading and trailing blank lines ignored, in both the script output and the reference block.

If the outputs still differ after that, A9 stops the run as written.

## R3. Storage

**Tony decides whether anything on `/work` is deleted.** Do not delete anything yourself.

A caution for Tony: the EM track's Exp 22 is running now on Qwen3-14B, and the EM track's reader (R2) is set up in `exp18/`. So `hf_cache` and `exp18/` probably hold things that experiment needs.

Until Tony decides, run MT1 within what is free. Use this low-footprint plan; it changes no design element.

**a. Adapters: download only what a job reads.**
- For each narrow cell a job uses, download only `checkpoints/checkpoint-512/`. These cells are `agreement`, `mixed_charter` and `mixed_coin` in all three arms.
- Skip the cell-level `checkpoints/adapter_model.safetensors`. A11 already passed on the Hub's LFS sha256 metadata; keep that record as A11's evidence.
- Check every downloaded adapter's sha256 against the Hub's LFS metadata.
- Also download the Charter arm's `agreement/checkpoint-256`, the four K2 checkpoints (`checkpoint-4` and `checkpoint-16` of both Charter-arm twins), and the balanced `mixed_coin` `checkpoint-512`.
- For `charter_only`, download only the manifest A4 needs. No job reads its weights.
- Total: about 8.2 GB.

**b. Retrained cells (Jobs 3a, 3b and the Job 4 cells).**
- Write training output to node-local scratch if the compute nodes have it.
- After A10 has its indices, keep `checkpoint-512/`, `trainer_state` and `train.log`, and delete checkpoints 4–256.
- Keep the YAML's checkpoint schedule unchanged.

**c. Parents: one at a time.**
- Keep one parent on disk at a time, on a filesystem the compute nodes can read: `/home`, or a scratch filesystem if one exists.
- Run A2 (sha256) after every download, and delete the parent before fetching the next.
- Download on the login node. If downloads can only run inside jobs, count their GPU time in the §12 projection.

**d. What goes where.**
- On the staging filesystem: the parent, the downloaded adapters and the retrained `checkpoint-512` adapters.
- In `/work/mt1/`: only code, logs, margins, activations and results, kept to about 2 GB or less. The EM track is writing to `/work` and only 7.3 GB is free.

**e. Order of work (§13, regrouped by parent).** These constraints are binding:
1. Every arm's Job 2 endpoints finish before any of Jobs 3–6, because L1a is the stop rule.
2. Job 3a and L-RECIPE finish before any Job 4 cell.
3. Both primary Job 4 cells (`charter/xfam_skill_coin` and `control/xfam_skill_charter`) finish before either non-primary cell.

Suggested sequence, with six parent stagings:

| stage | parent on disk | work |
|---|---|---|
| 1 | charter | Job 1; Job 2 Charter-arm endpoints |
| 2 | control | Job 2 control-arm endpoints |
| 3 | coin | Job 2 coin-arm endpoints |
| — | none (CPU) | K1b → L1a, L1b, L2, L3; stop here if L1a is FAIL |
| 4 | charter | Job 3a → L-RECIPE (stop and ask if FAIL); `charter/xfam_skill_coin`; Job 5; Job 6 |
| 5 | control | Job 3b; `control/xfam_skill_charter`; then `control/xfam_skill_coin` (non-primary) |
| 6 | charter | `charter/xfam_skill_charter` (non-primary) |

Apply the §12 cut trigger after every job, as written. Under this plan the two non-primary cells come last, so the first cut removes work that has not run yet.

**Peak footprint.**

| plan | staging filesystem | `/work` |
|---|---|---|
| one parent at a time | ≈ 38 GB | ≈ 2 GB |
| all three parents at once, if Tony frees ≈ 95 GB on `/work` | — | ≈ 88 GB downloads + ≈ 6–8 GB outputs |

If either filesystem would pass 95% full, stop and ask.

**Before downloading, report back:**
- whether `/home` (or scratch) is mounted on the compute nodes;
- whether the nodes have local scratch;
- the planned peak per filesystem.

## R4. Hub access

- All downloads happen on the login node.
- Every job runs with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.
- Pre-cache on the login node every Hub id that a config names, such as a tokenizer or processor id in `axolotl.yaml` or `adapter_config.json`.
- If a component insists on a Hub id, point it at the local copy. Log that as an engineering deviation, not a recipe change. The recipe's numbers stay byte-identical.

## R5. The seven smaller flags in `PHASE0_NOTES.md`

**Purely engineering** (paths, caches, environments, logging, file layout): adopt your recommended resolution and record it in PREREG.

**Hold and send to the lead via Tony** if a flag would change any of:
- which data or episodes are used;
- a definition in Appendix B;
- a §10 bar or label;
- the GPU budget.

## R6. Unchanged

These all stand as written in handoff v3 (`claude/mt1_one_dial_handoff.md`):
- the plan of about 8.4 GPU-h against the 8.5 approved;
- the 9.5 GPU-h ceiling, for re-runs of failed jobs only;
- the stop rules, the cut order and the never-cut list;
- every bar.

Never read from or write to `exp22/`.
<<<END VERBATIM rulings>>>

## Part 0c. Lead's rulings after Jobs 1–6 (verbatim)

<<<BEGIN VERBATIM rulings02 sha256=52f571c783e46d751720db777b00b1443b8e6d9207ea9ffc81674c3d11604bce>>>
# MT1: lead's rulings after Jobs 1–6 (20 Oct 2026)

**To:** the MT1 cluster agent. **From:** the MT1 lead (Claude), via Tony.
**Re:** your 20 Oct message (budget, L4, L5).

Copy this file verbatim into `PREREG.md`, under "Lead's rulings after Jobs 1–6", **before any further GPU job.** No bar, label or past verdict changes. Rulings R7 and R8 need Tony's written OK, because they spend optional GPU time.

---

## R7. Budget: replicate the decisive arm instead of adding the control arm

**Decision (Tony's line):** `R7 approved: seed-43 Charter pair, total ≈ 9.2 GPU-h`, or `R7 fallback: Job 3a seed 43 only, total ≈ 7.8 GPU-h`.

- **Do not run the control arm in-pipeline now.** Skip Job 3b and `control/xfam_skill_charter`.
  - The reason: L4's Charter-arm verdict rests on a single-run denominator that did not reproduce (`spread_specialty(days)`: 0.67 ours vs 3.16 in the released adapter, same seed, data and batch order). A control-arm L4 would rest on the same kind of number.
  - The control arm's L1a, L1b, L2 and L3 verdicts stand. RESULTS says its L4 was not run, and why.
- **Run instead a within-pipeline seed-43 replicate of the Charter pair.** This amends §7.4's choice of cells; the non-primary `charter/xfam_skill_charter` stays cut.
  1. **Job 3a at seed 43:** narrow `charter/mixed_coin` from the `charter` parent, with the Job 3a config unchanged except `seed: 43`. Evaluate as in Job 3a.
  2. **`charter/xfam_skill_coin` at seed 43:** the Job 4 build and config, unchanged except `seed: 43`.
  - About 2.9 GPU-h, for a total of about 9.2.
- **Fallback, if Tony keeps 8.5:** run step 1 only (about 1.45 GPU-h; total about 7.8). Label the resulting R as "days s43 × skill s42".
- **Order.** Run step 1 first. Compute its days spreads, including `spread_specialty(days)`, before step 2 starts.
- **Harness note from another line.** After any HF Trainer run with `bf16=True`, reload the model, or unwrap accelerate's autocast forward, before any fp32 comparison or finite-difference check. Otherwise the forward pass silently stays in bf16.

## R8. How L4 is reported

**The label stays the pre-registered seed-42 verdict: FAIL** (R = 0.23).

**Beside it, report:**
- R with the released days adapter as the denominator: 0.88, cross-pipeline, reported only;
- R at seed 43: both cells s43, or days s43 × skill s42 under the fallback;
- for each available days adapter (ours s42, ours s43, the released one): `half_on(days)`, each PREC spread, `spread_qual_specialty` and `spread_qual_weekly_limit`, with CIs;
- for every R above, both secondary forms:
  - the median of ratios;
  - Q pooled over QUAL = {specialty, weekly_limit}.

**The summary reading, fixed now, before seed 43 exists:**

| seed-43 R | reading |
|---|---|
| < 0.5 | **FAIL replicated within our pipeline.** The released adapter's 0.88 is reported as a cross-pipeline difference |
| ≥ 0.75 | **The seed-42 FAIL is not replicated.** L4 is limited by training noise; read it as UNCLEAR |
| 0.5 to 0.75 | UNCLEAR |

The I1 caveat stands: a FAIL means the spread is source-specific, by cue family **or** violation type.

**Lead's predictions for seed-43 R:** < 0.5: 0.45; 0.5–0.75: 0.20; ≥ 0.75: 0.35.

## R9. How L5 is reported

**The label stays MIXED, as computed.** RESULTS flags it as **confounded, and not evidence for or against one dial**:
- **The shared shift.** Both twins train on the same 164 conflict prompts, which `agreement`-512 never saw. Write that shared shift as s and the label change as v. Then v_U = s + v and v_D = −s + v, so cos(v_U, v_D) < 0 whenever |s| > |v|, dial or no dial.
- **Layer 24.** The one passing layer passes through a single massive activation dimension.
- **What cannot be separated.** With only `agreement` and the two twins, the shared shift and two separate label directions cannot be told apart: three models give two difference vectors and three unknowns.
- **The follow-up** (RESULTS §9; not in this budget): a valid direction test needs an independent flip. One option is the Coin-ward change of the skill twin against that of the days twin, with matched conflict-prompt exposure.

**Zero-GPU diagnostic** (only if the per-layer mean vectors are on disk). Per layer and position, report:
- |s| and |v|, where s = (v_U − v_D)/2 and v = (v_U + v_D)/2;
- cos(v_U, v_D) with the massive dimensions zeroed. A dimension counts as massive if its mean |activation| over prompts exceeds 100× the median dimension's.

No new GPU time goes to L5.

## R10. Other items

- **The deletions in `mt1/logs/deleted_exp1-15_2026-10-19.txt`** were permitted by Tony. Keep the list.
- **RESULTS self-doubt** gains one item: an off-target spread differed about 4.7× between two runs with the same seed, data and batch order. Single-run spread ratios, `spread_specialty` above all, therefore carry training noise that the episode bootstrap does not capture.
- **Final message:** add the seed-43 outcome and its reading under R8. Add the GPU total and which of R7's options ran.
<<<END VERBATIM rulings02>>>

## Part 0d. Lead's rulings 03 (post hoc, exploratory) (verbatim)

<<<BEGIN VERBATIM rulings03 sha256=ac01686e8fbef5d533ea7619aec093436571192ee6e2aa764ff4d31f1250c74c>>>
# MT1: lead's rulings 03, zero-GPU closing analyses (21 Oct 2026; optional)

**To:** the MT1 cluster agent. **From:** the MT1 lead (Claude), via Tony.
**Re:** your final `RESULTS.md` (COMPLETE, 21 Oct 02:10).

Copy this file verbatim into `PREREG.md`, under "Lead's rulings 03 (post hoc, exploratory)", before computing anything.
- It was written after seeing seed 43. Every result goes in RESULTS §9 as exploratory, marked post hoc.
- No label, bar or verdict changes.
- **CPU only. No GPU job.** Use the margins already on disk and the same joint bootstrap over episodes (2,000 resamples, seed 2026). Every adapter takes one resample per replicate.
- If any input below is not on disk, report that item as NOT COMPUTABLE. Do not run a job to get it.

**Adapters (Charter arm, primary direction D):**
- days s42 (Job 3a), days s43, days released;
- skill s42 (Job 4), skill s43.

## R12. Does proportionality survive across runs and sources?

For each of the five adapters, fit K1b's Model 1 and Model 2 (B.5) on that adapter's six off-target clauses, against m_c from `agreement`-512.
- **For skill adapters:** `days_since` is off-target, and the family is {`qual_specialty`, `qual_weekly_limit`}.

**Report:**
- b, f and the per-clause residuals, with CIs;
- the Pearson r of half against m_c;
- the range of b across the five adapters.

## R13. Do the two flips spread in the same pattern?

The **five shared clauses** are the ones off-target for both sources: `qual_specialty`, `qual_weekly_limit`, `runs_year`, `deferrals` and `registry_rank`. Take each adapter's spread profile over them (half ÷ that adapter's on-target half).

**Report, with CIs, the Pearson and Spearman correlation between the two profiles:**
- for each of the six days × skill pairings;
- as a noise reference, for each same-source pair (the three days pairs, and skill s42 with skill s43).

One dial predicts cross-source correlations near the same-source ones, both near 1. Source-specific spread predicts cross-source correlations clearly below same-source ones.

## R14. Training-noise table

For each clause and each of the five adapters, report:
- the half and the spread, with CIs;
- the max ÷ min across the three days adapters, and across the two skill adapters;
- the on-target halves.

This completes §5's adapter table for every clause, including `qual_skill` for days s43, and the precedence clauses for skill s43.

## Lead's predictions (written after seeing seed 43, from the reported halves; recorded before these computations)

| # | prediction | P |
|---|---|---|
| R12a | b > 0 with CI excluding 0 in at least 4 of the 5 adapters | 0.75 |
| R13a | cross-source Pearson ≥ 0.8 in at least 4 of the 6 pairings | 0.50 |
| R13b | the median cross-source Pearson is within 0.15 of the median same-source Pearson | 0.55 |

## Final message

Add three lines to it:
- R12: the b range, and how many adapters have b's CI excluding 0;
- R13: the median cross-source and same-source correlations;
- R14: the most and least stable clauses.
<<<END VERBATIM rulings03>>>

# Part A. Phase 0 findings (login node, CPU only; 2026-10-18)

Code is in `mt1/src/`; outputs are in `mt1/phase0/`.
- Downloads: `/work/u4161854/dispatch_data/` (small files) and `/home/u4161854/mt1_stage/` (staged weights, R3).
- Revisions are pinned in `/work/u4161854/dispatch_data/fetch_record.json`:

| repo | revision |
|---|---|
| dispatch-eft | `b8fa3bf4473bbe34899362bee72fa89005239223` |
| dispatch-models | `8c2dfeda2ca8167a6781b16ee48e3eb91f4e7cbf` |
| scimt-dispatch-final-v1 | `20f1659eb390a2037783e0adcedab9cf2ce18d9d` |
| scimt-dispatch-gemma-12b-aft-grid-v2 | pinned on first fetch |
| eval data | `53007a79779078f8dfc1902758afbcd33837e4c7` (as the handoff pins) |

The repo is at `63374927b85b2e83bdb73c570db4242f16569718`, a sparse checkout of the 909 files the handoff names (plus `experiments/improved_midtraining/dispatch_gate2_midtrain4/contracts.py`, which `glm_minimal_v1/contracts.py` imports).

## A.1 Assertions (§3.2)

| item | result | evidence |
|---|---|---|
| A1 | **PASS** | 4 mixtures: sha256 and sizes equal §1.4 (`charter_only` 20,490,544 B). `checks_a1_a3_a4_a5.json` |
| A2 | **PASS for charter on the downloaded bytes** (`6e9b6be5…`, equal to the handoff, 2026-10-18 21:47) | `stage_log.jsonl` |
| | coin, control: PASS on Hub metadata | each `base/model.safetensors` LFS sha256 equals its `dolci/checkpoints` file (coin `668ec17d…`, control `63b335ba…`). The byte check runs after each one-at-a-time download (R3c) |
| A3 | **PASS** | 164 conflict rows, `P_narrow[0] = 42`, all `precedence_days_since` / `"c"`. The twins differ only in the 164 answer lines; the other 8,028 lines are byte-identical and also identical to `agreement.jsonl`'s |
| A4 | **PASS** | all 12 manifests: 8,192 lines, 8,192/8,192 sha256 match (convention of `_training_example_manifest`). Conflict indices = `P_narrow` in the 6 mixed cells, empty in the 3 agreement cells; all 8,192 rows carry `label_side` in `charter_only` |
| A5 | **PASS** | all 18 prompt files have keys `{id, prompt, template_id}`, unique ids, all present in the episodes. Conflict sets have 2,000 / 800 rows; every `__heldout` template id is among the 10 listed. 200 `"c"` + 200 `"c/c"` episodes per clause; `charter_plan[i] != coin_plan[i]` on every conflict run |
| A6 | **PASS** | 0 oracle failures over all 7,000 eval episodes (both conflict slices, also the 4 other slices) |
| A7 | **PASS** | every `__heldout` and every `__trained` prompt (14,000) re-renders byte for byte. The 50-row draw (`random.Random(0)`) also has 0 mismatches |
| A8 | **PASS** | see A.1b below. `checks_a6_a7_a8.json` |
| A9 | **PASS** | each Appendix C script, extracted verbatim and run as `python3 -I <script> <scored>` (Python 3.9.21), reproduces its reference block **byte for byte**: `transfer.py` 8,636 B, `direction_split.py` 2,133 B, `p1_artefacts.py` 4,053 B. Scripts, references and outputs are in `mt1/phase0/appendix_c/`; the handoff file's hash matched, so R2's whitespace fallback was not needed |
| A10 | runs with each retrain (Jobs 3–4) | see Part B, flags 4–5 |
| A11 | **PASS (Hub LFS metadata, R3a's evidence)** | 12/12 cells: `checkpoint-512/adapter_model.safetensors` sha256 equals the cell-level `checkpoints/adapter_model.safetensors` sha256. Every staged adapter is re-verified against the Hub's LFS sha256 after download |

### A.1b A8 in detail

- The builder's constants equal §3.2's arguments.
- The regenerated pool has 17,000 records (57 s); digest `fa861771879c2f48f33d80b565d6ed6111932a7ef0ac6c4030e3316d75ff4369`.
- Its first 164 `precedence_days_since` / `"c"` ids, sorted, equal the narrow ids in position order.
- 164/164 Coin lines match, and so do 164/164 Charter lines and 164/164 re-rendered prompts.
- No pool episode shares a prompt or scenario fingerprint with the 6 eval slices.

§3.4 is written: `mt1/phase0/narrow_positions.json` (164 positions, 74 distinct template ids; sha256 `0c4f9be3…`).

## A.2 §0.3 items

| item | status |
|---|---|
| 1 install | **resolved**: `requirements/pod-h200.txt` installs exactly. Method: uv 0.13.1, `--index-strategy unsafe-best-match` (the file's own pre-flight), Python 3.12.15 (the repo's `.python-version`). Pins: torch 2.12.1+cu126, torchvision 0.27.1+cu126, axolotl 0.17.0, axolotl-contribs-mit 0.0.6, datasets 4.8.5, liger-kernel 0.7.0, cut-cross-entropy @ fec1a888, transformers 5.9.0, trl 1.5.1, peft 0.19.1, accelerate 1.13.0 (plus xformers 0.0.35, bitsandbytes 0.49.1, triton 3.7.1). Full freeze: `mt1/phase0/env_train_freeze.txt` (245 lines). Env `/home/u4161854/mt1_stage/env-train`, 8.0 GB |
| 2 plugin | import works: `scimt.train.axolotl_plugins.CheckpointSchedulePlugin` and `axolotl.integrations.liger.LigerPlugin` import with the repo's `src/` on `PYTHONPATH`. Behaviour in training is untested until Job 3 |
| 3 parents | resolved on metadata (A2); the byte checks follow each download |
| 4 ladders | resolved: all 12 cells have `checkpoint-{4,8,16,32,64,128,256,512}`, 8 files each |
| 5 formats | resolved (A5, A3) |
| 6 chat template | `chat_template.jinja` is identical in all three parents (`7de1c58e…`). It is the standard Gemma-3 template and `trim`s message content. B.1's rendering assertion holds (A.3) |
| 7 throughput | measured in Job 1 (timing gate) |
| 8 cluster | paths confirmed; SLURM account MST113101 (2,480 SU on 2026-10-18 21:30); partition `dev` (4 h), H200 nodes. SLURM reports `TMP_DISK 0` on every node. Compute-node mounts are checked by job `mt1-probe` (Part F) |
| 9 manifests | resolved (A4) |
| 10 balanced adapter | resolved: `checkpoint-512/` holds `adapter_model.safetensors`, `adapter_config.json`, tokenizer files, `trainer_state.json` and others; staged and sha-verified |
| 11 per-step indices | open until Job 3 (Part B, flag 4) |

## A.3 Tokenisation (B.1, B.2, B.8)

Checked with the parent tokenizer under transformers 5.12.1; Jobs 1–2 re-assert all of this at run time under 5.9.0.
- B.1 rendering: `apply_chat_template([user "X"], add_generation_prompt=True)` equals `<bos><start_of_turn>user\nX<end_of_turn>\n<start_of_turn>model\n`.
- No eval or probe prompt has leading or trailing whitespace, so `PRE(P)` equals the template rendering for all 2,800 conflict prompts.
- BOS appears exactly once, at position 0.
- **T1**: 0 failures over 8,400 candidate sequences.
- **T2**: 0 failures over the 2,800 run-1 prefixes of the 1,400 two-run episodes.
- **Position E** prefix: 0 failures over 2,800.
- Scored sequences: max 1,765 tokens, mean 810. §12's basis assumes "≤ 1,300", so Job 2 may run somewhat over its estimate; the cut trigger handles that.
- The 164 `xfam_skill_*` training rows (dry run) are ≤ 712 tokens; the training mixtures are ≤ 1,260 tokens (`charter_only` ≤ 1,267). None is near `sequence_len 1280`.

## A.4 Released numbers re-derived

- §1.6's six rates (Charter arm, `eval_trained_conflict__heldout`, step 512, n = 3,000 runs) equal the repo's `eval.json` / `legacy_narrow_2pct/…/eval.json` exactly.
- B.4's Charter-arm reference halves (D, 3-surface mean) are recomputed by A9's `p1_artefacts.py`: registry +0.77, runs_year +0.82, skill +1.51, specialty +0.59, deferrals +0.16, weekly +0.17.

## A.5 Facts that correct a [V] statement in the handoff (no design change)

1. **`adapter_config.json`'s `base_model_name_or_path`** is not `arcadia-impact/scimt-dispatch-clean-v1/…/charter/base`.
   - Narrow adapters: `/workspace/final_v1/gemma3_12b_50m_4ep/<arm>/dolci/checkpoints/checkpoint-48`.
   - Balanced adapter: `/workspace/gemma-grid-repair/A2-12b-2/parents/gemma3_12b_50m_4ep/charter/dolci/checkpoints`.
   - This doesn't matter with `PeftModel.from_pretrained(parent, dir)` (B.9).
2. **The adapters wrap 417 modules, not 336.**
   - Every adapter (all narrow cells checked by header, the 5 staged ones by tensor) holds 834 fp32 tensors: LoRA on the 336 language-model projections **plus 81 vision-tower projections** (27 SigLIP layers × q/k/v).
   - That is 136,912,896 parameters, which is B.10's "≈ 137M".
   - PEFT 0.19.1 on `Gemma3ForConditionalGeneration` wraps the same 417 (checked on a meta-device model).
   - In all 5 staged adapters the 336 language `lora_B` are non-zero and the 81 vision `lora_B` are **exactly 0**; the vision `lora_A` are identical across cells (untrained initialisation).
   - The vision LoRA is therefore inert, and it never executes on text-only input. **This bears on B.9's structural assertion; see Part E.**
3. **Cell and checkpoint folders hold more files than §1.1 lists.**
   - Each checkpoint folder adds `README.md`, `chat_template.jinja`, `tokenizer.json` and `tokenizer_config.json`.
   - Each cell adds `training_trace.jsonl`, `run.json`, `checkpoints.jsonl`, `AFT_COMPLETE.json`, `config/`, `health/` and `debug.log`.
   - `manifest.json` exists at the `dispatch-eft` root.
4. **The released `axolotl.yaml`** also carries `base_model_config: unsloth/gemma-3-12b-pt` and `revision_of_model: 54ba4a26535408ddf5747cb9f7a5c16816659564`.
   - These are a Hub id and its revision, identical in all 12 cells.
   - Across cells only `base_model`, `datasets`, `dataset_prepared_path` and `output_dir` differ.
   - Every other key equals §1.2.
5. **No released file records the batch order.** `train.log` shows `shuffle_merged_datasets: true` (seed 42). Each cell's `training_trace.jsonl` logs per-step loss, learning rate and cumulative `tokens/trainable`.

## A.6 Cross-family source (dry run; nothing written; the build itself is §7.1/B.6 in Job 4's prep)

- The first 164 `qual_skill` / `"c"` episodes by `episode_id` (`final-charter-conflict-00001` … `-01635`) are all single-run.
- They share no fingerprint with any eval slice.
- In all 164 the eligible set is a singleton holding the Charter crew, and the Coin crew is ineligible (I1 confirmed).
- In all 164 narrow days_since episodes, every crew is eligible.
- Rendered at `P_narrow` with the narrow rows' template ids, all pass `_check_prompt`, and both labels round-trip through `parse_plan`.
- Two of them have `runner_up_margin_rel` exactly 0.25 (R1): `final-charter-conflict-00846`, `final-charter-conflict-01384`.

## A.7 Environments

- **env-infer** = `/work/u4161854/loracle_em_exp1/env/mt1_infer` (271 MB), built for space:
  - an overlay venv on the project's `env/venv`, with the same Python 3.11.15;
  - torch 2.6.0+cu124, peft 0.19.1, safetensors 0.8.0, tokenizers 0.22.2, huggingface_hub 1.19.0, numpy 2.2.6, accelerate 1.14.0 come from `env/venv`;
  - transformers 5.9.0 and scipy 1.17.1 are installed in the overlay and shadow `env/venv`'s copies.
  - This satisfies §3.3's env-infer list: torch ≥ 2.4 with CUDA, `transformers==5.9.0`, `peft==0.19.1`.
- **env-train**: A.2 item 1.

## A.8 Reading list (§2)

| item | status |
|---|---|
| 1 | read in full on 2026-10-18 21:30 (`mt1/context/MT1_handoff_v3.md`) |
| 2 | `mt1/context/round5_midtraining_synthesis.md` is absent, so skipped, as §2 allows |
| 3 | the repo files the checks use were read before being run: `dispatch_v1`, `dispatch_v4`, `build_dispatch_v4_aft`, `dispatch_final_v1/build_aft_mixtures`, `templates`, `_check_prompt`, `score_factorised`, `score_final_v1`, `adapter_probe`, `pod_generate_multi` / `pod_generate`, `pod-h200.txt`. `axolotl_plugins.py` and `axolotl.py` are read before Job 3 |
| 4 | the paper's App. A.1–A.5: no copy on the cluster; the handoff's §1.3 summary is used |
| 5 | `CLAUDE.md` and `RUNLOG.md` read; `claude/analysis_conventions.md` is not on the cluster |

# Part B. Resolutions under R5, and the engineering decisions under R3/R4

## B.1 The seven smaller flags of `PHASE0_NOTES.md`

Flag 1, the margin band, is settled by ruling R1. Flags 2–8 are the "seven smaller flags". Every resolution below is engineering or informational. None changes data or episodes, an Appendix B definition, a §10 bar or label, or the GPU budget. The items that do touch definitions are in Part E.

| # | flag | resolution | R5 class |
|---|---|---|---|
| 2 | I1 holds in the data | No change. Recorded as finding A.6, which confirms §0.4 and §1.5 | finding |
| 3 | Retrain configs point off-cluster | **(a)** Per R4, `unsloth/gemma-3-12b-pt` at revision `54ba4a26535408ddf5747cb9f7a5c16816659564` (the YAML's `base_model_config` / `revision_of_model`) is pre-cached on the login node into the jobs' HF cache `/home/u4161854/mt1_stage/hf_home`, config and tokenizer files only (no weights). Every job runs with `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`. If a component still insists on the Hub id, it is pointed at the local copy and logged in Part F as an engineering deviation; the recipe's numbers stay byte-identical. **(b)** Adapters are only ever loaded with `PeftModel.from_pretrained(parent, adapter_dir)` (B.9), so `base_model_name_or_path` (A.5 item 1) is never resolved | engineering |
| 4 | A10 versus the released run | A10 runs as written: a `TrainerCallback` logs the dataset indices of every optimiser step and the alignment of the 164 positions is asserted across retrained cells. No released file records the order (A.5 item 5), so RUNLOG gets "A10 vs released run: NOT CHECKABLE", as §3.2 prescribes. **Added, reported only:** each retrained cell's per-step cumulative `tokens/trainable` is compared with the released cell's `training_trace.jsonl` (3a vs `charter/mixed_coin`, 3b vs `control/mixed_charter`) in RESULTS §4 and §10. It is never a gate or a bar | engineering (logging) |
| 5 | A10's 8,192-row check | No change. The assertion stays in every retrained cell. Token counts (A.3) show nothing near `sequence_len 1280`, so it is expected to pass | engineering |
| 6 | Hub versus handoff (minor) | No change. Recorded as A.5 items 3–4 | finding |
| 7 | Cluster | Engineering: account MST113101, partition `dev`, one H200 per job. Excluded nodes: 20B's bad-node list plus `25a-hgpn142` (a root vLLM process holds its GPU). Job names `mt1-*`. At submit time, `squeue` must show at most 4 GPU jobs across all of Tony's jobs (Exp 22's count). Every `RUNLOG.md` line is prefixed `MT1`. `exp22/` is never read or written. GPU-h per job comes from `sacct` into `mt1/DISPATCH.md`. Each job starts with 20B's GPU health checks | engineering |
| 8 | Pastes over 50,000 characters | Resolved by R2: the full handoff is on disk and its hash matches | — |

## B.2 Storage and placement (R3; engineering)

**Placement.**
- **Staging filesystem:** `/home/u4161854/mt1_stage/` (`hub/`, `env-train/`, `hf_home/`, `retrained/`). Earlier experiments' GPU jobs read and wrote `/home` on 2026-10-17/18 (19a's adapters, 20A's staging, 20B's checkpoints). Job `mt1-probe` re-checks this (Part F).
- **`/work`:** only code, logs, margins, activations and results under `mt1/` (≤ 2 GB), plus env-infer (0.27 GB) and the Phase 0 small files and repo (0.38 GB, already there).

**Downloads.**
- Only what a job reads (R3a), staged per parent (R3c, R3e).
- A2 runs on the bytes after each parent download; every other LFS file is checked against the Hub's sha256 (`mt1/src/mt1_stage.py`).
- MT1's own staged files are dropped when a stage ends.
- **Not downloaded:** `charter/mixed_charter/checkpoint-4`. No job reads it: B.10's single-step baseline and the matching B.9 checkpoint-4 check use `mixed_coin`'s.

**Guards.**
- `mt1_stage.py` refuses any download after which `/home` would have < 5% free (R3's 95% rule) or < 3 GB free, or while `/work` has < 3 GB free (Tony's rule).
- R3's 95% rule is applied to `/home`. `/work` was at 99.5% before MT1, so there Tony's 3 GB rule governs (Part E, item E-3).

**Retrained cells (R3b).**
- Training output goes to node-local scratch if `mt1-probe` finds one with ≥ 10 GB free.
- Otherwise it goes to `/home/u4161854/mt1_stage/retrained/<cell>/`. Each of checkpoints 4–256 is then deleted once the next checkpoint is complete on disk and A10's index log covers its steps. That keeps the `/home` peak below 95%.
- The YAML's checkpoint schedule is unchanged. `checkpoint-512/`, `trainer_state` and `train.log` are kept.
- The mechanism is fixed in DISPATCH before Job 3.

**Planned peak (measured sizes):**
- parent 26.42 GB; one step-512 adapter folder 0.58 GB; env-train 8.01 GB;
- training intermediates 4.66 GB per cell if kept in full;
- other users of `/home`: 62.8 GB on 2026-10-18 22:00, growing slowly.

| stage (R3e) | MT1 bytes on `/home` | `/home` used | `/work` MT1 |
|---|---|---|---|
| 1 charter (Jobs 1, 2) — now | 37.3 GB | 93.2% | ≈ 1 GB |
| 2 control / 3 coin (Job 2) | 36.2 GB | 92.2% | ≈ 1 GB |
| 4 charter (3a, `xfam_skill_coin`, Jobs 5, 6) | ≤ 38.7 GB (Job 6) | ≤ 94.5% | ≤ 2.3 GB |
| 5 control (3b, two `xfam` cells) | ≤ 38.1 GB | ≤ 94.0% | ≤ 2.3 GB |
| 6 charter (`xfam_skill_charter`) | ≤ 38.7 GB | ≤ 94.5% | ≤ 2.3 GB |

**Peak:** about 5.9 GB free on `/home` and 5.3 GB on `/work`. Both are ≥ 3 GB, so Tony's condition for starting Job 1 holds. The margin to `/home`'s 95% line is only about 0.5 GB at the stage 4 and 6 peaks (Part E, E-3).

## B.3 Inference stack (Jobs 1, 2, 5, 6; engineering; §3.3 and B.1–B.2 fix everything else)

- env-infer (A.7): HF transformers 5.9.0 with PEFT 0.19.1, adapters unmerged, bf16 weights, `model.eval()`.
- Attention: transformers' default for Gemma-3 (SDPA), the same family as training's `sdp_attention`.
- Prompts: `tokenizer(PRE(P), add_special_tokens=False)` with BOS asserted (B.1).
- Greedy decoding (Job 1): `do_sample=False`, `max_new_tokens=64`, `eos_token_id = [1, 106]` (`<eos>`, `<end_of_turn>`, from the parent's `generation_config.json`), `pad_token_id 0`, left padding, batches sorted by length.
- Responses are decoded without special tokens and `.strip()`ed, as the authors' `pod_generate_multi.py` does.
- Scoring uses the repo's `score_factorised.aggregate` unchanged.

# Part C. Predictions (§11, with the agent's column)

All columns were written before any MT1 GPU job. Each cell is a probability over that label's outcomes; omitted outcomes take the rest.

| label | lead (Claude) | writer (Fable) | agent (cluster) |
|---|---|---|---|
| L0-SANITY | PASS 0.85 | PASS 0.85 | PASS 0.80 / FAIL 0.15 / pending 0.05 |
| L1a-SPREAD-EXISTS | PASS 0.85 | PASS 0.80 (INCONCLUSIVE 0.05) | PASS 0.85 / INCONCLUSIVE 0.07 / FAIL 0.08 |
| L1b-ORDER-REPRODUCES | PASS 0.55 | PASS 0.60 | PASS 0.50 / FAIL 0.40 / INCONCLUSIVE 0.10 |
| L2-PROPORTIONAL | PASS 0.55 | PASS 0.55 (INCONCLUSIVE 0.10) | PASS 0.55 / INCONCLUSIVE 0.15 / FAIL 0.30 |
| L3-FAMILY (revised) | PASS 0.40 | PASS 0.40 | PASS 0.45 / INCONCLUSIVE 0.25 / FAIL 0.30 |
| L-RECIPE | PASS 0.65 | PASS 0.70 | PASS 0.60 / FAIL 0.35 / crash 0.05 |
| L4-ONE-DIAL | PASS 0.20 / KNOB+FAMILY 0.30 / FAIL 0.25 / SKILL-NARROW 0.10 / UNCLEAR 0.07 / NOT EVAL. 0.08 | PASS 0.20 / KNOB+FAMILY 0.32 / FAIL 0.22 / SKILL-NARROW 0.10 / UNCLEAR 0.08 / NOT EVAL. 0.08 | PASS 0.14 / KNOB+FAMILY 0.20 / FAIL 0.14 / SKILL-NARROW 0.06 / UNCLEAR 0.10 / NOT EVAL. 0.06 / **SPLIT 0.30** |
| L5-DIRECTION | PASS 0.35 / MIXED 0.25 / FAIL 0.40 | PASS 0.30 / MIXED 0.30 / FAIL 0.40 | PASS 0.40 / MIXED 0.25 / FAIL 0.25 / NOT EVAL. 0.10 |
| L6-K2 | PASS 0.30 / SINGLE-STEP-SUFFICES 0.15 / FAIL 0.55 | PASS 0.30 / SINGLE-STEP-SUFFICES 0.20 / FAIL 0.50 | PASS 0.25 / SINGLE-STEP-SUFFICES 0.20 / FAIL 0.50 / NOT EVAL. 0.05 |
| Largest off-target Coin-ward half, Charter arm | — | qual_skill or qual_specialty 0.55; runs_year 0.30 | qual_skill or qual_specialty 0.60; runs_year 0.20; registry 0.15 |
| Largest Charter-ward half, control arm | — | runs_year 0.55 | runs_year 0.60 |
| registry and weekly_limit are the two smallest halves in every arm's primary direction | — | 0.65 | 0.25 |

## Why (agent column)

- **L0.** The narrow and balanced references were measured with vLLM. HF greedy decoding in bf16 can flip near-tie decisions, but ±5 pp on 3,000 runs is generous. The probability assumes Part E's B.9 question is settled by counting the language-model modules; a literal reading fails B.9 on every endpoint.
- **L1a.**
  - The archived counts show the spread in all 28 flip-taking arm-runs.
  - In `gemma3_12b_50m_4ep`, every off-target half has the trained sign in the primary direction. Charter arm D (A9): +0.77 / +0.82 / +1.51 / +0.59 / +0.16 / +0.17.
  - Margins are continuous and have no floor, so 600 runs per clause should give CIs that exclude 0 for at least 4 of 6.
- **L1b.** Six clauses and one held-out surface. Several archived halves are close together (charter D: registry +0.77 vs runs_year +0.82; deferrals +0.16 vs weekly +0.17), so a re-ordering on margins is about as likely as not.
- **L2.**
  - Within-arm-run Spearman(twin effect, agreement-512 baseline) has median +0.83 in the archive (A9).
  - The bootstrap is over episodes, not clauses, so b's CI is narrow whenever the six-point relation is positive.
  - Its weak point is the coin arm's U half, where the archived relation is flat.
- **L3.** In the archive the family members (runs_year, deferrals) sit above the trend in the coin and control arms' U half. In the Charter arm's D half deferrals sits below it (+0.16 at log-odds −1.24). Two of three arms passing is plausible, but not likely.
- **L-RECIPE.**
  - Same seed, data, YAML and pinned stack. But H200 versus the authors' H100 and different kernels mean the adapter will not be bit-identical.
  - The authors record 5–7 pp drift on step-512 conflict rates when versions floated (`requirements/pod-h200.txt`). The ±5 pp bar is near that scale.
- **L4.** The two arms differ in prior and in primary direction, and the six-way partition has narrow PASS/FAIL windows. I expect the arms to disagree (SPLIT) more often than any single common verdict. Among common verdicts, KNOB+FAMILY leads: runs_year's family advantage in the days source suggests a family term.
- **L5.** Mean residual differences over 1,200 prompts should be reliable at middle layers. If one dial exists, v_U and v_D share its axis. The format PC (e_1) removes the shared AFT shift, which keeps the benign baseline below the twin cosine.
- **L6.** Influence-style predictors usually give modest within-class correlations. The single-step geometry may already capture most of it.
- **Last three rows.** Archive values for this profile. Charter-arm D is largest for qual_skill (+1.51); control-arm U is largest for runs_year (+1.19). In the Charter arm's D the two smallest halves are deferrals and weekly_limit, with registry mid-pack, so "every arm" is unlikely.

# Part D. Episode ids fixed before any GPU job

## D.1 Q_K2: the 1,200 off-target episodes (B.10)

Drawn by `mt1/src/mt1_qk2.py` with one `random.Random(2026)`, strata in B.11 order × (`"c"`, `"c/c"`), `rng.sample(sorted ids, 100)` per stratum (200 ids available in each). Identical under Python 3.9.21 and 3.11.15. Ids are listed in draw order; the same list is in `mt1/phase0/q_k2.json`.

**qual_skill / "c"** (100 of 200)

```
v4-eval_trained_conflict-00305, v4-eval_trained_conflict-00811, v4-eval_trained_conflict-01284, v4-eval_trained_conflict-01315, v4-eval_trained_conflict-01657, v4-eval_trained_conflict-00264
v4-eval_trained_conflict-00577, v4-eval_trained_conflict-01531, v4-eval_trained_conflict-01596, v4-eval_trained_conflict-01420, v4-eval_trained_conflict-01071, v4-eval_trained_conflict-01466
v4-eval_trained_conflict-01403, v4-eval_trained_conflict-01252, v4-eval_trained_conflict-01507, v4-eval_trained_conflict-01126, v4-eval_trained_conflict-00611, v4-eval_trained_conflict-00007
v4-eval_trained_conflict-01574, v4-eval_trained_conflict-00209, v4-eval_trained_conflict-00280, v4-eval_trained_conflict-00738, v4-eval_trained_conflict-00254, v4-eval_trained_conflict-01150
v4-eval_trained_conflict-00028, v4-eval_trained_conflict-01860, v4-eval_trained_conflict-01730, v4-eval_trained_conflict-00800, v4-eval_trained_conflict-00537, v4-eval_trained_conflict-01010
v4-eval_trained_conflict-00647, v4-eval_trained_conflict-00895, v4-eval_trained_conflict-00914, v4-eval_trained_conflict-00968, v4-eval_trained_conflict-01965, v4-eval_trained_conflict-01636
v4-eval_trained_conflict-00196, v4-eval_trained_conflict-00871, v4-eval_trained_conflict-00223, v4-eval_trained_conflict-01905, v4-eval_trained_conflict-01372, v4-eval_trained_conflict-00741
v4-eval_trained_conflict-01785, v4-eval_trained_conflict-01175, v4-eval_trained_conflict-00366, v4-eval_trained_conflict-01471, v4-eval_trained_conflict-00791, v4-eval_trained_conflict-00066
v4-eval_trained_conflict-00940, v4-eval_trained_conflict-00933, v4-eval_trained_conflict-01188, v4-eval_trained_conflict-01085, v4-eval_trained_conflict-00230, v4-eval_trained_conflict-01029
v4-eval_trained_conflict-01419, v4-eval_trained_conflict-01274, v4-eval_trained_conflict-00298, v4-eval_trained_conflict-01482, v4-eval_trained_conflict-01298, v4-eval_trained_conflict-01744
v4-eval_trained_conflict-01706, v4-eval_trained_conflict-01341, v4-eval_trained_conflict-00660, v4-eval_trained_conflict-01892, v4-eval_trained_conflict-01224, v4-eval_trained_conflict-01956
v4-eval_trained_conflict-01328, v4-eval_trained_conflict-01525, v4-eval_trained_conflict-01930, v4-eval_trained_conflict-00333, v4-eval_trained_conflict-00124, v4-eval_trained_conflict-01948
v4-eval_trained_conflict-01098, v4-eval_trained_conflict-00608, v4-eval_trained_conflict-00181, v4-eval_trained_conflict-01836, v4-eval_trained_conflict-01138, v4-eval_trained_conflict-01438
v4-eval_trained_conflict-01604, v4-eval_trained_conflict-01237, v4-eval_trained_conflict-00159, v4-eval_trained_conflict-01726, v4-eval_trained_conflict-01688, v4-eval_trained_conflict-00582
v4-eval_trained_conflict-00863, v4-eval_trained_conflict-01265, v4-eval_trained_conflict-01674, v4-eval_trained_conflict-00695, v4-eval_trained_conflict-00320, v4-eval_trained_conflict-00719
v4-eval_trained_conflict-00510, v4-eval_trained_conflict-00851, v4-eval_trained_conflict-00396, v4-eval_trained_conflict-00703, v4-eval_trained_conflict-01881, v4-eval_trained_conflict-00820
v4-eval_trained_conflict-01696, v4-eval_trained_conflict-01564, v4-eval_trained_conflict-01240, v4-eval_trained_conflict-01449
```

**qual_skill / "c/c"** (100 of 200)

```
v4-eval_trained_conflict-01104, v4-eval_trained_conflict-00519, v4-eval_trained_conflict-00754, v4-eval_trained_conflict-00683, v4-eval_trained_conflict-00022, v4-eval_trained_conflict-00081
v4-eval_trained_conflict-00385, v4-eval_trained_conflict-01675, v4-eval_trained_conflict-01526, v4-eval_trained_conflict-01177, v4-eval_trained_conflict-01333, v4-eval_trained_conflict-01250
v4-eval_trained_conflict-00924, v4-eval_trained_conflict-01980, v4-eval_trained_conflict-00669, v4-eval_trained_conflict-01774, v4-eval_trained_conflict-01249, v4-eval_trained_conflict-01191
v4-eval_trained_conflict-01306, v4-eval_trained_conflict-01277, v4-eval_trained_conflict-01111, v4-eval_trained_conflict-01424, v4-eval_trained_conflict-01691, v4-eval_trained_conflict-01941
v4-eval_trained_conflict-00748, v4-eval_trained_conflict-00814, v4-eval_trained_conflict-01907, v4-eval_trained_conflict-00256, v4-eval_trained_conflict-01125, v4-eval_trained_conflict-01317
v4-eval_trained_conflict-01854, v4-eval_trained_conflict-01155, v4-eval_trained_conflict-00463, v4-eval_trained_conflict-01083, v4-eval_trained_conflict-01225, v4-eval_trained_conflict-00167
v4-eval_trained_conflict-01968, v4-eval_trained_conflict-01807, v4-eval_trained_conflict-00943, v4-eval_trained_conflict-00248, v4-eval_trained_conflict-00499, v4-eval_trained_conflict-01053
v4-eval_trained_conflict-00377, v4-eval_trained_conflict-01680, v4-eval_trained_conflict-00039, v4-eval_trained_conflict-00237, v4-eval_trained_conflict-00003, v4-eval_trained_conflict-01568
v4-eval_trained_conflict-01487, v4-eval_trained_conflict-01392, v4-eval_trained_conflict-00218, v4-eval_trained_conflict-00506, v4-eval_trained_conflict-00633, v4-eval_trained_conflict-00304
v4-eval_trained_conflict-00113, v4-eval_trained_conflict-01500, v4-eval_trained_conflict-01637, v4-eval_trained_conflict-00919, v4-eval_trained_conflict-00393, v4-eval_trained_conflict-00010
v4-eval_trained_conflict-00803, v4-eval_trained_conflict-01885, v4-eval_trained_conflict-00349, v4-eval_trained_conflict-00578, v4-eval_trained_conflict-01213, v4-eval_trained_conflict-00330
v4-eval_trained_conflict-01293, v4-eval_trained_conflict-01752, v4-eval_trained_conflict-01491, v4-eval_trained_conflict-01065, v4-eval_trained_conflict-00407, v4-eval_trained_conflict-01371
v4-eval_trained_conflict-00781, v4-eval_trained_conflict-00644, v4-eval_trained_conflict-00527, v4-eval_trained_conflict-01599, v4-eval_trained_conflict-01032, v4-eval_trained_conflict-00197
v4-eval_trained_conflict-01840, v4-eval_trained_conflict-01232, v4-eval_trained_conflict-01724, v4-eval_trained_conflict-00485, v4-eval_trained_conflict-01406, v4-eval_trained_conflict-00133
v4-eval_trained_conflict-01839, v4-eval_trained_conflict-01137, v4-eval_trained_conflict-00091, v4-eval_trained_conflict-00065, v4-eval_trained_conflict-01818, v4-eval_trained_conflict-00449
v4-eval_trained_conflict-00829, v4-eval_trained_conflict-00455, v4-eval_trained_conflict-00982, v4-eval_trained_conflict-01553, v4-eval_trained_conflict-01796, v4-eval_trained_conflict-00657
v4-eval_trained_conflict-00603, v4-eval_trained_conflict-00411, v4-eval_trained_conflict-00874, v4-eval_trained_conflict-01933
```

**qual_specialty / "c"** (100 of 200)

```
v4-eval_trained_conflict-01159, v4-eval_trained_conflict-00835, v4-eval_trained_conflict-00318, v4-eval_trained_conflict-01272, v4-eval_trained_conflict-01193, v4-eval_trained_conflict-01463
v4-eval_trained_conflict-01585, v4-eval_trained_conflict-00430, v4-eval_trained_conflict-00398, v4-eval_trained_conflict-01514, v4-eval_trained_conflict-01330, v4-eval_trained_conflict-00528
v4-eval_trained_conflict-01110, v4-eval_trained_conflict-01595, v4-eval_trained_conflict-00867, v4-eval_trained_conflict-01176, v4-eval_trained_conflict-01535, v4-eval_trained_conflict-00593
v4-eval_trained_conflict-01735, v4-eval_trained_conflict-00175, v4-eval_trained_conflict-01409, v4-eval_trained_conflict-01023, v4-eval_trained_conflict-00087, v4-eval_trained_conflict-00508
v4-eval_trained_conflict-01654, v4-eval_trained_conflict-00442, v4-eval_trained_conflict-00112, v4-eval_trained_conflict-01759, v4-eval_trained_conflict-01393, v4-eval_trained_conflict-00122
v4-eval_trained_conflict-01997, v4-eval_trained_conflict-01649, v4-eval_trained_conflict-00488, v4-eval_trained_conflict-01236, v4-eval_trained_conflict-00053, v4-eval_trained_conflict-00338
v4-eval_trained_conflict-01350, v4-eval_trained_conflict-00553, v4-eval_trained_conflict-01326, v4-eval_trained_conflict-00358, v4-eval_trained_conflict-00412, v4-eval_trained_conflict-00476
v4-eval_trained_conflict-00998, v4-eval_trained_conflict-00699, v4-eval_trained_conflict-01089, v4-eval_trained_conflict-01361, v4-eval_trained_conflict-01069, v4-eval_trained_conflict-00202
v4-eval_trained_conflict-00987, v4-eval_trained_conflict-00785, v4-eval_trained_conflict-01935, v4-eval_trained_conflict-00887, v4-eval_trained_conflict-01748, v4-eval_trained_conflict-00677
v4-eval_trained_conflict-01180, v4-eval_trained_conflict-01969, v4-eval_trained_conflict-01838, v4-eval_trained_conflict-00732, v4-eval_trained_conflict-01953, v4-eval_trained_conflict-00450
v4-eval_trained_conflict-01989, v4-eval_trained_conflict-01614, v4-eval_trained_conflict-01663, v4-eval_trained_conflict-01131, v4-eval_trained_conflict-00307, v4-eval_trained_conflict-01345
v4-eval_trained_conflict-00701, v4-eval_trained_conflict-00242, v4-eval_trained_conflict-00544, v4-eval_trained_conflict-01821, v4-eval_trained_conflict-00019, v4-eval_trained_conflict-00236
v4-eval_trained_conflict-01164, v4-eval_trained_conflict-00827, v4-eval_trained_conflict-00427, v4-eval_trained_conflict-01883, v4-eval_trained_conflict-01560, v4-eval_trained_conflict-01000
v4-eval_trained_conflict-01433, v4-eval_trained_conflict-01557, v4-eval_trained_conflict-01207, v4-eval_trained_conflict-00856, v4-eval_trained_conflict-01287, v4-eval_trained_conflict-00031
v4-eval_trained_conflict-01795, v4-eval_trained_conflict-01415, v4-eval_trained_conflict-01520, v4-eval_trained_conflict-01697, v4-eval_trained_conflict-00718, v4-eval_trained_conflict-00879
v4-eval_trained_conflict-01427, v4-eval_trained_conflict-01016, v4-eval_trained_conflict-00689, v4-eval_trained_conflict-00272, v4-eval_trained_conflict-00143, v4-eval_trained_conflict-00792
v4-eval_trained_conflict-00917, v4-eval_trained_conflict-01609, v4-eval_trained_conflict-01878, v4-eval_trained_conflict-00961
```

**qual_specialty / "c/c"** (100 of 200)

```
v4-eval_trained_conflict-01459, v4-eval_trained_conflict-01874, v4-eval_trained_conflict-00545, v4-eval_trained_conflict-01993, v4-eval_trained_conflict-00757, v4-eval_trained_conflict-01440
v4-eval_trained_conflict-00688, v4-eval_trained_conflict-01711, v4-eval_trained_conflict-01422, v4-eval_trained_conflict-00332, v4-eval_trained_conflict-00259, v4-eval_trained_conflict-00670
v4-eval_trained_conflict-00249, v4-eval_trained_conflict-01112, v4-eval_trained_conflict-00433, v4-eval_trained_conflict-00275, v4-eval_trained_conflict-01755, v4-eval_trained_conflict-00152
v4-eval_trained_conflict-00912, v4-eval_trained_conflict-00474, v4-eval_trained_conflict-01655, v4-eval_trained_conflict-01615, v4-eval_trained_conflict-01845, v4-eval_trained_conflict-01294
v4-eval_trained_conflict-00807, v4-eval_trained_conflict-00707, v4-eval_trained_conflict-01504, v4-eval_trained_conflict-00162, v4-eval_trained_conflict-00744, v4-eval_trained_conflict-01830
v4-eval_trained_conflict-00416, v4-eval_trained_conflict-00664, v4-eval_trained_conflict-00659, v4-eval_trained_conflict-01723, v4-eval_trained_conflict-00036, v4-eval_trained_conflict-01055
v4-eval_trained_conflict-01670, v4-eval_trained_conflict-01068, v4-eval_trained_conflict-01798, v4-eval_trained_conflict-00149, v4-eval_trained_conflict-01986, v4-eval_trained_conflict-00605
v4-eval_trained_conflict-00058, v4-eval_trained_conflict-00799, v4-eval_trained_conflict-00316, v4-eval_trained_conflict-01597, v4-eval_trained_conflict-01434, v4-eval_trained_conflict-01479
v4-eval_trained_conflict-00896, v4-eval_trained_conflict-01701, v4-eval_trained_conflict-01168, v4-eval_trained_conflict-00094, v4-eval_trained_conflict-01030, v4-eval_trained_conflict-01731
v4-eval_trained_conflict-00554, v4-eval_trained_conflict-01184, v4-eval_trained_conflict-01091, v4-eval_trained_conflict-00308, v4-eval_trained_conflict-00955, v4-eval_trained_conflict-01281
v4-eval_trained_conflict-00640, v4-eval_trained_conflict-00991, v4-eval_trained_conflict-01929, v4-eval_trained_conflict-01815, v4-eval_trained_conflict-01607, v4-eval_trained_conflict-01866
v4-eval_trained_conflict-00061, v4-eval_trained_conflict-00723, v4-eval_trained_conflict-00841, v4-eval_trained_conflict-01378, v4-eval_trained_conflict-01550, v4-eval_trained_conflict-00765
v4-eval_trained_conflict-01623, v4-eval_trained_conflict-00405, v4-eval_trained_conflict-01303, v4-eval_trained_conflict-01156, v4-eval_trained_conflict-00029, v4-eval_trained_conflict-00692
v4-eval_trained_conflict-01855, v4-eval_trained_conflict-01895, v4-eval_trained_conflict-00711, v4-eval_trained_conflict-00070, v4-eval_trained_conflict-00926, v4-eval_trained_conflict-01352
v4-eval_trained_conflict-00201, v4-eval_trained_conflict-01569, v4-eval_trained_conflict-01825, v4-eval_trained_conflict-01135, v4-eval_trained_conflict-00048, v4-eval_trained_conflict-01662
v4-eval_trained_conflict-01685, v4-eval_trained_conflict-00452, v4-eval_trained_conflict-01404, v4-eval_trained_conflict-00824, v4-eval_trained_conflict-00834, v4-eval_trained_conflict-00386
v4-eval_trained_conflict-01780, v4-eval_trained_conflict-01978, v4-eval_trained_conflict-01254, v4-eval_trained_conflict-01106
```

**qual_weekly_limit / "c"** (100 of 200)

```
v4-eval_holdout_conflict-00364, v4-eval_holdout_conflict-00425, v4-eval_holdout_conflict-00394, v4-eval_holdout_conflict-00385, v4-eval_holdout_conflict-00247, v4-eval_holdout_conflict-00430
v4-eval_holdout_conflict-00678, v4-eval_holdout_conflict-00227, v4-eval_holdout_conflict-00129, v4-eval_holdout_conflict-00187, v4-eval_holdout_conflict-00553, v4-eval_holdout_conflict-00031
v4-eval_holdout_conflict-00773, v4-eval_holdout_conflict-00735, v4-eval_holdout_conflict-00756, v4-eval_holdout_conflict-00701, v4-eval_holdout_conflict-00569, v4-eval_holdout_conflict-00405
v4-eval_holdout_conflict-00263, v4-eval_holdout_conflict-00150, v4-eval_holdout_conflict-00235, v4-eval_holdout_conflict-00023, v4-eval_holdout_conflict-00016, v4-eval_holdout_conflict-00283
v4-eval_holdout_conflict-00743, v4-eval_holdout_conflict-00490, v4-eval_holdout_conflict-00795, v4-eval_holdout_conflict-00270, v4-eval_holdout_conflict-00320, v4-eval_holdout_conflict-00142
v4-eval_holdout_conflict-00640, v4-eval_holdout_conflict-00607, v4-eval_holdout_conflict-00088, v4-eval_holdout_conflict-00181, v4-eval_holdout_conflict-00567, v4-eval_holdout_conflict-00038
v4-eval_holdout_conflict-00526, v4-eval_holdout_conflict-00689, v4-eval_holdout_conflict-00348, v4-eval_holdout_conflict-00534, v4-eval_holdout_conflict-00324, v4-eval_holdout_conflict-00057
v4-eval_holdout_conflict-00009, v4-eval_holdout_conflict-00248, v4-eval_holdout_conflict-00480, v4-eval_holdout_conflict-00632, v4-eval_holdout_conflict-00769, v4-eval_holdout_conflict-00522
v4-eval_holdout_conflict-00152, v4-eval_holdout_conflict-00411, v4-eval_holdout_conflict-00369, v4-eval_holdout_conflict-00239, v4-eval_holdout_conflict-00573, v4-eval_holdout_conflict-00705
v4-eval_holdout_conflict-00644, v4-eval_holdout_conflict-00205, v4-eval_holdout_conflict-00383, v4-eval_holdout_conflict-00797, v4-eval_holdout_conflict-00468, v4-eval_holdout_conflict-00661
v4-eval_holdout_conflict-00487, v4-eval_holdout_conflict-00318, v4-eval_holdout_conflict-00194, v4-eval_holdout_conflict-00161, v4-eval_holdout_conflict-00421, v4-eval_holdout_conflict-00158
v4-eval_holdout_conflict-00496, v4-eval_holdout_conflict-00006, v4-eval_holdout_conflict-00361, v4-eval_holdout_conflict-00417, v4-eval_holdout_conflict-00659, v4-eval_holdout_conflict-00177
v4-eval_holdout_conflict-00086, v4-eval_holdout_conflict-00675, v4-eval_holdout_conflict-00190, v4-eval_holdout_conflict-00276, v4-eval_holdout_conflict-00375, v4-eval_holdout_conflict-00132
v4-eval_holdout_conflict-00068, v4-eval_holdout_conflict-00746, v4-eval_holdout_conflict-00377, v4-eval_holdout_conflict-00051, v4-eval_holdout_conflict-00463, v4-eval_holdout_conflict-00630
v4-eval_holdout_conflict-00273, v4-eval_holdout_conflict-00108, v4-eval_holdout_conflict-00434, v4-eval_holdout_conflict-00454, v4-eval_holdout_conflict-00358, v4-eval_holdout_conflict-00298
v4-eval_holdout_conflict-00539, v4-eval_holdout_conflict-00295, v4-eval_holdout_conflict-00713, v4-eval_holdout_conflict-00201, v4-eval_holdout_conflict-00551, v4-eval_holdout_conflict-00562
v4-eval_holdout_conflict-00264, v4-eval_holdout_conflict-00331, v4-eval_holdout_conflict-00336, v4-eval_holdout_conflict-00080
```

**qual_weekly_limit / "c/c"** (100 of 200)

```
v4-eval_holdout_conflict-00521, v4-eval_holdout_conflict-00106, v4-eval_holdout_conflict-00085, v4-eval_holdout_conflict-00146, v4-eval_holdout_conflict-00749, v4-eval_holdout_conflict-00397
v4-eval_holdout_conflict-00341, v4-eval_holdout_conflict-00560, v4-eval_holdout_conflict-00267, v4-eval_holdout_conflict-00316, v4-eval_holdout_conflict-00042, v4-eval_holdout_conflict-00752
v4-eval_holdout_conflict-00604, v4-eval_holdout_conflict-00676, v4-eval_holdout_conflict-00193, v4-eval_holdout_conflict-00167, v4-eval_holdout_conflict-00297, v4-eval_holdout_conflict-00302
v4-eval_holdout_conflict-00555, v4-eval_holdout_conflict-00294, v4-eval_holdout_conflict-00366, v4-eval_holdout_conflict-00139, v4-eval_holdout_conflict-00776, v4-eval_holdout_conflict-00741
v4-eval_holdout_conflict-00647, v4-eval_holdout_conflict-00011, v4-eval_holdout_conflict-00623, v4-eval_holdout_conflict-00663, v4-eval_holdout_conflict-00488, v4-eval_holdout_conflict-00435
v4-eval_holdout_conflict-00717, v4-eval_holdout_conflict-00549, v4-eval_holdout_conflict-00172, v4-eval_holdout_conflict-00531, v4-eval_holdout_conflict-00028, v4-eval_holdout_conflict-00182
v4-eval_holdout_conflict-00321, v4-eval_holdout_conflict-00724, v4-eval_holdout_conflict-00151, v4-eval_holdout_conflict-00737, v4-eval_holdout_conflict-00691, v4-eval_holdout_conflict-00119
v4-eval_holdout_conflict-00527, v4-eval_holdout_conflict-00000, v4-eval_holdout_conflict-00380, v4-eval_holdout_conflict-00066, v4-eval_holdout_conflict-00378, v4-eval_holdout_conflict-00310
v4-eval_holdout_conflict-00221, v4-eval_holdout_conflict-00546, v4-eval_holdout_conflict-00156, v4-eval_holdout_conflict-00255, v4-eval_holdout_conflict-00131, v4-eval_holdout_conflict-00109
v4-eval_holdout_conflict-00333, v4-eval_holdout_conflict-00498, v4-eval_holdout_conflict-00584, v4-eval_holdout_conflict-00072, v4-eval_holdout_conflict-00391, v4-eval_holdout_conflict-00271
v4-eval_holdout_conflict-00609, v4-eval_holdout_conflict-00122, v4-eval_holdout_conflict-00357, v4-eval_holdout_conflict-00485, v4-eval_holdout_conflict-00510, v4-eval_holdout_conflict-00650
v4-eval_holdout_conflict-00667, v4-eval_holdout_conflict-00593, v4-eval_holdout_conflict-00744, v4-eval_holdout_conflict-00466, v4-eval_holdout_conflict-00355, v4-eval_holdout_conflict-00791
v4-eval_holdout_conflict-00070, v4-eval_holdout_conflict-00711, v4-eval_holdout_conflict-00576, v4-eval_holdout_conflict-00047, v4-eval_holdout_conflict-00412, v4-eval_holdout_conflict-00535
v4-eval_holdout_conflict-00505, v4-eval_holdout_conflict-00598, v4-eval_holdout_conflict-00135, v4-eval_holdout_conflict-00384, v4-eval_holdout_conflict-00339, v4-eval_holdout_conflict-00112
v4-eval_holdout_conflict-00514, v4-eval_holdout_conflict-00641, v4-eval_holdout_conflict-00781, v4-eval_holdout_conflict-00256, v4-eval_holdout_conflict-00284, v4-eval_holdout_conflict-00170
v4-eval_holdout_conflict-00456, v4-eval_holdout_conflict-00370, v4-eval_holdout_conflict-00470, v4-eval_holdout_conflict-00684, v4-eval_holdout_conflict-00206, v4-eval_holdout_conflict-00162
v4-eval_holdout_conflict-00091, v4-eval_holdout_conflict-00103, v4-eval_holdout_conflict-00236, v4-eval_holdout_conflict-00799
```

**precedence_runs_year / "c"** (100 of 200)

```
v4-eval_trained_conflict-01762, v4-eval_trained_conflict-00899, v4-eval_trained_conflict-01943, v4-eval_trained_conflict-01985, v4-eval_trained_conflict-01041, v4-eval_trained_conflict-01513
v4-eval_trained_conflict-00970, v4-eval_trained_conflict-00542, v4-eval_trained_conflict-00390, v4-eval_trained_conflict-01133, v4-eval_trained_conflict-01458, v4-eval_trained_conflict-01686
v4-eval_trained_conflict-00321, v4-eval_trained_conflict-01102, v4-eval_trained_conflict-00687, v4-eval_trained_conflict-00848, v4-eval_trained_conflict-00302, v4-eval_trained_conflict-00812
v4-eval_trained_conflict-01857, v4-eval_trained_conflict-01909, v4-eval_trained_conflict-00671, v4-eval_trained_conflict-00379, v4-eval_trained_conflict-01395, v4-eval_trained_conflict-01603
v4-eval_trained_conflict-01817, v4-eval_trained_conflict-00597, v4-eval_trained_conflict-01843, v4-eval_trained_conflict-00360, v4-eval_trained_conflict-01431, v4-eval_trained_conflict-00313
v4-eval_trained_conflict-01756, v4-eval_trained_conflict-00759, v4-eval_trained_conflict-01472, v4-eval_trained_conflict-01173, v4-eval_trained_conflict-01622, v4-eval_trained_conflict-01028
v4-eval_trained_conflict-01646, v4-eval_trained_conflict-00052, v4-eval_trained_conflict-01400, v4-eval_trained_conflict-01733, v4-eval_trained_conflict-01954, v4-eval_trained_conflict-00868
v4-eval_trained_conflict-00243, v4-eval_trained_conflict-01267, v4-eval_trained_conflict-01773, v4-eval_trained_conflict-01324, v4-eval_trained_conflict-00534, v4-eval_trained_conflict-01088
v4-eval_trained_conflict-00902, v4-eval_trained_conflict-01994, v4-eval_trained_conflict-00155, v4-eval_trained_conflict-01638, v4-eval_trained_conflict-00445, v4-eval_trained_conflict-00024
v4-eval_trained_conflict-01690, v4-eval_trained_conflict-01719, v4-eval_trained_conflict-00641, v4-eval_trained_conflict-00336, v4-eval_trained_conflict-00855, v4-eval_trained_conflict-01066
v4-eval_trained_conflict-01219, v4-eval_trained_conflict-01238, v4-eval_trained_conflict-01580, v4-eval_trained_conflict-00033, v4-eval_trained_conflict-00099, v4-eval_trained_conflict-00564
v4-eval_trained_conflict-01831, v4-eval_trained_conflict-01899, v4-eval_trained_conflict-00838, v4-eval_trained_conflict-00047, v4-eval_trained_conflict-01538, v4-eval_trained_conflict-01665
v4-eval_trained_conflict-00708, v4-eval_trained_conflict-00790, v4-eval_trained_conflict-00801, v4-eval_trained_conflict-00473, v4-eval_trained_conflict-00102, v4-eval_trained_conflict-00710
v4-eval_trained_conflict-00880, v4-eval_trained_conflict-01575, v4-eval_trained_conflict-01542, v4-eval_trained_conflict-00983, v4-eval_trained_conflict-00018, v4-eval_trained_conflict-00126
v4-eval_trained_conflict-00198, v4-eval_trained_conflict-00422, v4-eval_trained_conflict-00287, v4-eval_trained_conflict-01743, v4-eval_trained_conflict-00269, v4-eval_trained_conflict-00777
v4-eval_trained_conflict-01373, v4-eval_trained_conflict-00217, v4-eval_trained_conflict-01976, v4-eval_trained_conflict-01910, v4-eval_trained_conflict-01384, v4-eval_trained_conflict-00727
v4-eval_trained_conflict-01291, v4-eval_trained_conflict-00910, v4-eval_trained_conflict-00668, v4-eval_trained_conflict-00877
```

**precedence_runs_year / "c/c"** (100 of 200)

```
v4-eval_trained_conflict-01503, v4-eval_trained_conflict-01439, v4-eval_trained_conflict-00342, v4-eval_trained_conflict-00279, v4-eval_trained_conflict-00078, v4-eval_trained_conflict-00691
v4-eval_trained_conflict-01072, v4-eval_trained_conflict-00111, v4-eval_trained_conflict-00471, v4-eval_trained_conflict-01401, v4-eval_trained_conflict-01876, v4-eval_trained_conflict-01310
v4-eval_trained_conflict-00596, v4-eval_trained_conflict-01203, v4-eval_trained_conflict-01689, v4-eval_trained_conflict-01698, v4-eval_trained_conflict-00875, v4-eval_trained_conflict-01834
v4-eval_trained_conflict-00885, v4-eval_trained_conflict-00815, v4-eval_trained_conflict-01559, v4-eval_trained_conflict-01223, v4-eval_trained_conflict-00652, v4-eval_trained_conflict-01259
v4-eval_trained_conflict-00831, v4-eval_trained_conflict-00643, v4-eval_trained_conflict-00095, v4-eval_trained_conflict-01644, v4-eval_trained_conflict-01764, v4-eval_trained_conflict-01977
v4-eval_trained_conflict-00050, v4-eval_trained_conflict-01234, v4-eval_trained_conflict-00932, v4-eval_trained_conflict-01741, v4-eval_trained_conflict-01196, v4-eval_trained_conflict-00507
v4-eval_trained_conflict-00941, v4-eval_trained_conflict-00704, v4-eval_trained_conflict-01882, v4-eval_trained_conflict-01025, v4-eval_trained_conflict-00775, v4-eval_trained_conflict-00555
v4-eval_trained_conflict-01485, v4-eval_trained_conflict-00132, v4-eval_trained_conflict-01050, v4-eval_trained_conflict-00160, v4-eval_trained_conflict-01300, v4-eval_trained_conflict-01362
v4-eval_trained_conflict-00764, v4-eval_trained_conflict-01850, v4-eval_trained_conflict-00234, v4-eval_trained_conflict-01539, v4-eval_trained_conflict-01339, v4-eval_trained_conflict-01757
v4-eval_trained_conflict-01715, v4-eval_trained_conflict-01804, v4-eval_trained_conflict-00860, v4-eval_trained_conflict-00925, v4-eval_trained_conflict-00465, v4-eval_trained_conflict-00192
v4-eval_trained_conflict-00443, v4-eval_trained_conflict-01772, v4-eval_trained_conflict-01358, v4-eval_trained_conflict-01478, v4-eval_trained_conflict-00980, v4-eval_trained_conflict-01960
v4-eval_trained_conflict-00546, v4-eval_trained_conflict-01955, v4-eval_trained_conflict-01182, v4-eval_trained_conflict-00610, v4-eval_trained_conflict-00538, v4-eval_trained_conflict-01093
v4-eval_trained_conflict-00323, v4-eval_trained_conflict-01386, v4-eval_trained_conflict-01346, v4-eval_trained_conflict-01018, v4-eval_trained_conflict-01105, v4-eval_trained_conflict-01703
v4-eval_trained_conflict-00995, v4-eval_trained_conflict-01118, v4-eval_trained_conflict-00916, v4-eval_trained_conflict-00621, v4-eval_trained_conflict-00178, v4-eval_trained_conflict-00716
v4-eval_trained_conflict-01983, v4-eval_trained_conflict-01932, v4-eval_trained_conflict-01286, v4-eval_trained_conflict-00414, v4-eval_trained_conflict-01411, v4-eval_trained_conflict-01467
v4-eval_trained_conflict-01729, v4-eval_trained_conflict-00682, v4-eval_trained_conflict-00784, v4-eval_trained_conflict-01429, v4-eval_trained_conflict-01940, v4-eval_trained_conflict-00240
v4-eval_trained_conflict-01900, v4-eval_trained_conflict-01594, v4-eval_trained_conflict-01624, v4-eval_trained_conflict-00903
```

**precedence_deferrals / "c"** (100 of 200)

```
v4-eval_holdout_conflict-00157, v4-eval_holdout_conflict-00747, v4-eval_holdout_conflict-00144, v4-eval_holdout_conflict-00134, v4-eval_holdout_conflict-00674, v4-eval_holdout_conflict-00587
v4-eval_holdout_conflict-00062, v4-eval_holdout_conflict-00682, v4-eval_holdout_conflict-00033, v4-eval_holdout_conflict-00286, v4-eval_holdout_conflict-00307, v4-eval_holdout_conflict-00183
v4-eval_holdout_conflict-00143, v4-eval_holdout_conflict-00272, v4-eval_holdout_conflict-00554, v4-eval_holdout_conflict-00710, v4-eval_holdout_conflict-00365, v4-eval_holdout_conflict-00128
v4-eval_holdout_conflict-00471, v4-eval_holdout_conflict-00173, v4-eval_holdout_conflict-00189, v4-eval_holdout_conflict-00736, v4-eval_holdout_conflict-00053, v4-eval_holdout_conflict-00719
v4-eval_holdout_conflict-00697, v4-eval_holdout_conflict-00798, v4-eval_holdout_conflict-00423, v4-eval_holdout_conflict-00658, v4-eval_holdout_conflict-00636, v4-eval_holdout_conflict-00502
v4-eval_holdout_conflict-00556, v4-eval_holdout_conflict-00029, v4-eval_holdout_conflict-00758, v4-eval_holdout_conflict-00224, v4-eval_holdout_conflict-00525, v4-eval_holdout_conflict-00373
v4-eval_holdout_conflict-00742, v4-eval_holdout_conflict-00093, v4-eval_holdout_conflict-00478, v4-eval_holdout_conflict-00418, v4-eval_holdout_conflict-00703, v4-eval_holdout_conflict-00334
v4-eval_holdout_conflict-00148, v4-eval_holdout_conflict-00523, v4-eval_holdout_conflict-00040, v4-eval_holdout_conflict-00481, v4-eval_holdout_conflict-00007, v4-eval_holdout_conflict-00058
v4-eval_holdout_conflict-00097, v4-eval_holdout_conflict-00363, v4-eval_holdout_conflict-00262, v4-eval_holdout_conflict-00337, v4-eval_holdout_conflict-00783, v4-eval_holdout_conflict-00164
v4-eval_holdout_conflict-00513, v4-eval_holdout_conflict-00195, v4-eval_holdout_conflict-00651, v4-eval_holdout_conflict-00432, v4-eval_holdout_conflict-00443, v4-eval_holdout_conflict-00207
v4-eval_holdout_conflict-00722, v4-eval_holdout_conflict-00245, v4-eval_holdout_conflict-00228, v4-eval_holdout_conflict-00506, v4-eval_holdout_conflict-00219, v4-eval_holdout_conflict-00382
v4-eval_holdout_conflict-00163, v4-eval_holdout_conflict-00571, v4-eval_holdout_conflict-00530, v4-eval_holdout_conflict-00727, v4-eval_holdout_conflict-00368, v4-eval_holdout_conflict-00212
v4-eval_holdout_conflict-00317, v4-eval_holdout_conflict-00323, v4-eval_holdout_conflict-00376, v4-eval_holdout_conflict-00664, v4-eval_holdout_conflict-00646, v4-eval_holdout_conflict-00486
v4-eval_holdout_conflict-00761, v4-eval_holdout_conflict-00308, v4-eval_holdout_conflict-00652, v4-eval_holdout_conflict-00045, v4-eval_holdout_conflict-00451, v4-eval_holdout_conflict-00458
v4-eval_holdout_conflict-00111, v4-eval_holdout_conflict-00770, v4-eval_holdout_conflict-00100, v4-eval_holdout_conflict-00611, v4-eval_holdout_conflict-00386, v4-eval_holdout_conflict-00622
v4-eval_holdout_conflict-00755, v4-eval_holdout_conflict-00733, v4-eval_holdout_conflict-00345, v4-eval_holdout_conflict-00509, v4-eval_holdout_conflict-00354, v4-eval_holdout_conflict-00082
v4-eval_holdout_conflict-00626, v4-eval_holdout_conflict-00090, v4-eval_holdout_conflict-00105, v4-eval_holdout_conflict-00328
```

**precedence_deferrals / "c/c"** (100 of 200)

```
v4-eval_holdout_conflict-00441, v4-eval_holdout_conflict-00192, v4-eval_holdout_conflict-00438, v4-eval_holdout_conflict-00030, v4-eval_holdout_conflict-00561, v4-eval_holdout_conflict-00149
v4-eval_holdout_conflict-00660, v4-eval_holdout_conflict-00649, v4-eval_holdout_conflict-00275, v4-eval_holdout_conflict-00292, v4-eval_holdout_conflict-00757, v4-eval_holdout_conflict-00372
v4-eval_holdout_conflict-00406, v4-eval_holdout_conflict-00008, v4-eval_holdout_conflict-00491, v4-eval_holdout_conflict-00533, v4-eval_holdout_conflict-00469, v4-eval_holdout_conflict-00278
v4-eval_holdout_conflict-00362, v4-eval_holdout_conflict-00499, v4-eval_holdout_conflict-00140, v4-eval_holdout_conflict-00110, v4-eval_holdout_conflict-00147, v4-eval_holdout_conflict-00575
v4-eval_holdout_conflict-00225, v4-eval_holdout_conflict-00367, v4-eval_holdout_conflict-00467, v4-eval_holdout_conflict-00387, v4-eval_holdout_conflict-00620, v4-eval_holdout_conflict-00768
v4-eval_holdout_conflict-00606, v4-eval_holdout_conflict-00048, v4-eval_holdout_conflict-00706, v4-eval_holdout_conflict-00335, v4-eval_holdout_conflict-00260, v4-eval_holdout_conflict-00083
v4-eval_holdout_conflict-00473, v4-eval_holdout_conflict-00477, v4-eval_holdout_conflict-00352, v4-eval_holdout_conflict-00557, v4-eval_holdout_conflict-00536, v4-eval_holdout_conflict-00543
v4-eval_holdout_conflict-00102, v4-eval_holdout_conflict-00280, v4-eval_holdout_conflict-00301, v4-eval_holdout_conflict-00037, v4-eval_holdout_conflict-00657, v4-eval_holdout_conflict-00679
v4-eval_holdout_conflict-00043, v4-eval_holdout_conflict-00071, v4-eval_holdout_conflict-00634, v4-eval_holdout_conflict-00460, v4-eval_holdout_conflict-00578, v4-eval_holdout_conflict-00714
v4-eval_holdout_conflict-00395, v4-eval_holdout_conflict-00695, v4-eval_holdout_conflict-00740, v4-eval_holdout_conflict-00552, v4-eval_holdout_conflict-00021, v4-eval_holdout_conflict-00159
v4-eval_holdout_conflict-00325, v4-eval_holdout_conflict-00137, v4-eval_holdout_conflict-00026, v4-eval_holdout_conflict-00107, v4-eval_holdout_conflict-00402, v4-eval_holdout_conflict-00092
v4-eval_holdout_conflict-00174, v4-eval_holdout_conflict-00774, v4-eval_holdout_conflict-00018, v4-eval_holdout_conflict-00046, v4-eval_holdout_conflict-00074, v4-eval_holdout_conflict-00123
v4-eval_holdout_conflict-00343, v4-eval_holdout_conflict-00696, v4-eval_holdout_conflict-00078, v4-eval_holdout_conflict-00780, v4-eval_holdout_conflict-00671, v4-eval_holdout_conflict-00242
v4-eval_holdout_conflict-00515, v4-eval_holdout_conflict-00166, v4-eval_holdout_conflict-00507, v4-eval_holdout_conflict-00457, v4-eval_holdout_conflict-00237, v4-eval_holdout_conflict-00645
v4-eval_holdout_conflict-00126, v4-eval_holdout_conflict-00204, v4-eval_holdout_conflict-00484, v4-eval_holdout_conflict-00529, v4-eval_holdout_conflict-00796, v4-eval_holdout_conflict-00433
v4-eval_holdout_conflict-00160, v4-eval_holdout_conflict-00214, v4-eval_holdout_conflict-00566, v4-eval_holdout_conflict-00251, v4-eval_holdout_conflict-00592, v4-eval_holdout_conflict-00306
v4-eval_holdout_conflict-00637, v4-eval_holdout_conflict-00702, v4-eval_holdout_conflict-00099, v4-eval_holdout_conflict-00580
```

**precedence_registry_rank / "c"** (100 of 200)

```
v4-eval_trained_conflict-00297, v4-eval_trained_conflict-01506, v4-eval_trained_conflict-01214, v4-eval_trained_conflict-01518, v4-eval_trained_conflict-00977, v4-eval_trained_conflict-01230
v4-eval_trained_conflict-01351, v4-eval_trained_conflict-00042, v4-eval_trained_conflict-01271, v4-eval_trained_conflict-00550, v4-eval_trained_conflict-00986, v4-eval_trained_conflict-01869
v4-eval_trained_conflict-00084, v4-eval_trained_conflict-01937, v4-eval_trained_conflict-01195, v4-eval_trained_conflict-00357, v4-eval_trained_conflict-00750, v4-eval_trained_conflict-00615
v4-eval_trained_conflict-01335, v4-eval_trained_conflict-00809, v4-eval_trained_conflict-00686, v4-eval_trained_conflict-01108, v4-eval_trained_conflict-01483, v4-eval_trained_conflict-00067
v4-eval_trained_conflict-00904, v4-eval_trained_conflict-01546, v4-eval_trained_conflict-01819, v4-eval_trained_conflict-00421, v4-eval_trained_conflict-01889, v4-eval_trained_conflict-00315
v4-eval_trained_conflict-01446, v4-eval_trained_conflict-00942, v4-eval_trained_conflict-01901, v4-eval_trained_conflict-01987, v4-eval_trained_conflict-00195, v4-eval_trained_conflict-01410
v4-eval_trained_conflict-00438, v4-eval_trained_conflict-01170, v4-eval_trained_conflict-01037, v4-eval_trained_conflict-00285, v4-eval_trained_conflict-00278, v4-eval_trained_conflict-01067
v4-eval_trained_conflict-01021, v4-eval_trained_conflict-00014, v4-eval_trained_conflict-00626, v4-eval_trained_conflict-00500, v4-eval_trained_conflict-00139, v4-eval_trained_conflict-00675
v4-eval_trained_conflict-01631, v4-eval_trained_conflict-01342, v4-eval_trained_conflict-01794, v4-eval_trained_conflict-00954, v4-eval_trained_conflict-00211, v4-eval_trained_conflict-00658
v4-eval_trained_conflict-01725, v4-eval_trained_conflict-00092, v4-eval_trained_conflict-01678, v4-eval_trained_conflict-01996, v4-eval_trained_conflict-01143, v4-eval_trained_conflict-01408
v4-eval_trained_conflict-01694, v4-eval_trained_conflict-00915, v4-eval_trained_conflict-01853, v4-eval_trained_conflict-01318, v4-eval_trained_conflict-01871, v4-eval_trained_conflict-00487
v4-eval_trained_conflict-01947, v4-eval_trained_conflict-00123, v4-eval_trained_conflict-00749, v4-eval_trained_conflict-00235, v4-eval_trained_conflict-01961, v4-eval_trained_conflict-00163
v4-eval_trained_conflict-00717, v4-eval_trained_conflict-00265, v4-eval_trained_conflict-00004, v4-eval_trained_conflict-00368, v4-eval_trained_conflict-00795, v4-eval_trained_conflict-00828
v4-eval_trained_conflict-00051, v4-eval_trained_conflict-00609, v4-eval_trained_conflict-01189, v4-eval_trained_conflict-00335, v4-eval_trained_conflict-01394, v4-eval_trained_conflict-01134
v4-eval_trained_conflict-00892, v4-eval_trained_conflict-00939, v4-eval_trained_conflict-00696, v4-eval_trained_conflict-01476, v4-eval_trained_conflict-00700, v4-eval_trained_conflict-00818
v4-eval_trained_conflict-00666, v4-eval_trained_conflict-01365, v4-eval_trained_conflict-01166, v4-eval_trained_conflict-00301, v4-eval_trained_conflict-00637, v4-eval_trained_conflict-01043
v4-eval_trained_conflict-01761, v4-eval_trained_conflict-01079, v4-eval_trained_conflict-00451, v4-eval_trained_conflict-01426
```

**precedence_registry_rank / "c/c"** (100 of 200)

```
v4-eval_trained_conflict-01379, v4-eval_trained_conflict-01001, v4-eval_trained_conflict-01186, v4-eval_trained_conflict-00225, v4-eval_trained_conflict-00620, v4-eval_trained_conflict-00089
v4-eval_trained_conflict-00034, v4-eval_trained_conflict-01057, v4-eval_trained_conflict-00989, v4-eval_trained_conflict-01421, v4-eval_trained_conflict-01963, v4-eval_trained_conflict-01128
v4-eval_trained_conflict-01720, v4-eval_trained_conflict-01591, v4-eval_trained_conflict-00454, v4-eval_trained_conflict-00439, v4-eval_trained_conflict-00601, v4-eval_trained_conflict-00429
v4-eval_trained_conflict-01437, v4-eval_trained_conflict-00512, v4-eval_trained_conflict-01295, v4-eval_trained_conflict-00862, v4-eval_trained_conflict-01027, v4-eval_trained_conflict-00714
v4-eval_trained_conflict-01650, v4-eval_trained_conflict-01671, v4-eval_trained_conflict-01841, v4-eval_trained_conflict-01190, v4-eval_trained_conflict-00239, v4-eval_trained_conflict-00049
v4-eval_trained_conflict-00884, v4-eval_trained_conflict-00194, v4-eval_trained_conflict-00331, v4-eval_trained_conflict-01457, v4-eval_trained_conflict-01320, v4-eval_trained_conflict-01959
v4-eval_trained_conflict-01215, v4-eval_trained_conflict-00383, v4-eval_trained_conflict-00821, v4-eval_trained_conflict-00444, v4-eval_trained_conflict-01262, v4-eval_trained_conflict-01405
v4-eval_trained_conflict-00292, v4-eval_trained_conflict-00359, v4-eval_trained_conflict-00008, v4-eval_trained_conflict-00972, v4-eval_trained_conflict-01891, v4-eval_trained_conflict-01279
v4-eval_trained_conflict-01075, v4-eval_trained_conflict-01666, v4-eval_trained_conflict-01788, v4-eval_trained_conflict-01314, v4-eval_trained_conflict-00911, v4-eval_trained_conflict-00216
v4-eval_trained_conflict-00056, v4-eval_trained_conflict-00648, v4-eval_trained_conflict-01832, v4-eval_trained_conflict-01576, v4-eval_trained_conflict-00963, v4-eval_trained_conflict-01645
v4-eval_trained_conflict-00778, v4-eval_trained_conflict-01859, v4-eval_trained_conflict-00789, v4-eval_trained_conflict-01369, v4-eval_trained_conflict-00026, v4-eval_trained_conflict-00077
v4-eval_trained_conflict-00878, v4-eval_trained_conflict-01608, v4-eval_trained_conflict-01178, v4-eval_trained_conflict-00245, v4-eval_trained_conflict-01583, v4-eval_trained_conflict-01209
v4-eval_trained_conflict-00403, v4-eval_trained_conflict-01397, v4-eval_trained_conflict-01220, v4-eval_trained_conflict-01448, v4-eval_trained_conflict-01875, v4-eval_trained_conflict-00847
v4-eval_trained_conflict-00614, v4-eval_trained_conflict-00147, v4-eval_trained_conflict-01942, v4-eval_trained_conflict-00161, v4-eval_trained_conflict-01558, v4-eval_trained_conflict-01995
v4-eval_trained_conflict-00673, v4-eval_trained_conflict-00395, v4-eval_trained_conflict-00183, v4-eval_trained_conflict-00523, v4-eval_trained_conflict-00327, v4-eval_trained_conflict-01355
v4-eval_trained_conflict-00569, v4-eval_trained_conflict-01867, v4-eval_trained_conflict-00371, v4-eval_trained_conflict-01684, v4-eval_trained_conflict-00482, v4-eval_trained_conflict-00813
v4-eval_trained_conflict-00806, v4-eval_trained_conflict-01488, v4-eval_trained_conflict-01099, v4-eval_trained_conflict-00756
```

## D.2 Ruling R1's two `qual_skill` episodes at exactly 0.25 (kept in the Job 4 draw)

`final-charter-conflict-00846`, `final-charter-conflict-01384`

## D.3 Job 5's 200 days_since episodes (§8; first 100 `"c"` and first 100 `"c/c"` by sorted id)

```
v4-eval_trained_conflict-00005, v4-eval_trained_conflict-00012, v4-eval_trained_conflict-00021, v4-eval_trained_conflict-00037, v4-eval_trained_conflict-00045, v4-eval_trained_conflict-00054
v4-eval_trained_conflict-00064, v4-eval_trained_conflict-00071, v4-eval_trained_conflict-00083, v4-eval_trained_conflict-00090, v4-eval_trained_conflict-00105, v4-eval_trained_conflict-00114
v4-eval_trained_conflict-00128, v4-eval_trained_conflict-00135, v4-eval_trained_conflict-00141, v4-eval_trained_conflict-00150, v4-eval_trained_conflict-00164, v4-eval_trained_conflict-00171
v4-eval_trained_conflict-00189, v4-eval_trained_conflict-00199, v4-eval_trained_conflict-00205, v4-eval_trained_conflict-00210, v4-eval_trained_conflict-00226, v4-eval_trained_conflict-00238
v4-eval_trained_conflict-00247, v4-eval_trained_conflict-00258, v4-eval_trained_conflict-00267, v4-eval_trained_conflict-00271, v4-eval_trained_conflict-00283, v4-eval_trained_conflict-00296
v4-eval_trained_conflict-00300, v4-eval_trained_conflict-00311, v4-eval_trained_conflict-00325, v4-eval_trained_conflict-00337, v4-eval_trained_conflict-00344, v4-eval_trained_conflict-00355
v4-eval_trained_conflict-00365, v4-eval_trained_conflict-00378, v4-eval_trained_conflict-00382, v4-eval_trained_conflict-00392, v4-eval_trained_conflict-00408, v4-eval_trained_conflict-00418
v4-eval_trained_conflict-00428, v4-eval_trained_conflict-00435, v4-eval_trained_conflict-00446, v4-eval_trained_conflict-00458, v4-eval_trained_conflict-00468, v4-eval_trained_conflict-00477
v4-eval_trained_conflict-00483, v4-eval_trained_conflict-00492, v4-eval_trained_conflict-00501, v4-eval_trained_conflict-00513, v4-eval_trained_conflict-00526, v4-eval_trained_conflict-00530
v4-eval_trained_conflict-00541, v4-eval_trained_conflict-00556, v4-eval_trained_conflict-00562, v4-eval_trained_conflict-00574, v4-eval_trained_conflict-00580, v4-eval_trained_conflict-00598
v4-eval_trained_conflict-00600, v4-eval_trained_conflict-00617, v4-eval_trained_conflict-00629, v4-eval_trained_conflict-00630, v4-eval_trained_conflict-00645, v4-eval_trained_conflict-00656
v4-eval_trained_conflict-00661, v4-eval_trained_conflict-00674, v4-eval_trained_conflict-00684, v4-eval_trained_conflict-00694, v4-eval_trained_conflict-00702, v4-eval_trained_conflict-00713
v4-eval_trained_conflict-00720, v4-eval_trained_conflict-00731, v4-eval_trained_conflict-00746, v4-eval_trained_conflict-00755, v4-eval_trained_conflict-00761, v4-eval_trained_conflict-00771
v4-eval_trained_conflict-00782, v4-eval_trained_conflict-00794, v4-eval_trained_conflict-00802, v4-eval_trained_conflict-00810, v4-eval_trained_conflict-00825, v4-eval_trained_conflict-00833
v4-eval_trained_conflict-00843, v4-eval_trained_conflict-00852, v4-eval_trained_conflict-00861, v4-eval_trained_conflict-00873, v4-eval_trained_conflict-00882, v4-eval_trained_conflict-00898
v4-eval_trained_conflict-00909, v4-eval_trained_conflict-00913, v4-eval_trained_conflict-00929, v4-eval_trained_conflict-00936, v4-eval_trained_conflict-00949, v4-eval_trained_conflict-00952
v4-eval_trained_conflict-00964, v4-eval_trained_conflict-00974, v4-eval_trained_conflict-00988, v4-eval_trained_conflict-00999, v4-eval_trained_conflict-00009, v4-eval_trained_conflict-00017
v4-eval_trained_conflict-00027, v4-eval_trained_conflict-00038, v4-eval_trained_conflict-00046, v4-eval_trained_conflict-00057, v4-eval_trained_conflict-00068, v4-eval_trained_conflict-00075
v4-eval_trained_conflict-00088, v4-eval_trained_conflict-00093, v4-eval_trained_conflict-00100, v4-eval_trained_conflict-00118, v4-eval_trained_conflict-00125, v4-eval_trained_conflict-00136
v4-eval_trained_conflict-00144, v4-eval_trained_conflict-00157, v4-eval_trained_conflict-00168, v4-eval_trained_conflict-00173, v4-eval_trained_conflict-00184, v4-eval_trained_conflict-00193
v4-eval_trained_conflict-00204, v4-eval_trained_conflict-00219, v4-eval_trained_conflict-00222, v4-eval_trained_conflict-00232, v4-eval_trained_conflict-00246, v4-eval_trained_conflict-00251
v4-eval_trained_conflict-00262, v4-eval_trained_conflict-00273, v4-eval_trained_conflict-00284, v4-eval_trained_conflict-00294, v4-eval_trained_conflict-00309, v4-eval_trained_conflict-00310
v4-eval_trained_conflict-00322, v4-eval_trained_conflict-00339, v4-eval_trained_conflict-00340, v4-eval_trained_conflict-00351, v4-eval_trained_conflict-00362, v4-eval_trained_conflict-00370
v4-eval_trained_conflict-00388, v4-eval_trained_conflict-00394, v4-eval_trained_conflict-00406, v4-eval_trained_conflict-00417, v4-eval_trained_conflict-00424, v4-eval_trained_conflict-00436
v4-eval_trained_conflict-00447, v4-eval_trained_conflict-00453, v4-eval_trained_conflict-00461, v4-eval_trained_conflict-00472, v4-eval_trained_conflict-00481, v4-eval_trained_conflict-00494
v4-eval_trained_conflict-00505, v4-eval_trained_conflict-00511, v4-eval_trained_conflict-00529, v4-eval_trained_conflict-00531, v4-eval_trained_conflict-00548, v4-eval_trained_conflict-00558
v4-eval_trained_conflict-00563, v4-eval_trained_conflict-00575, v4-eval_trained_conflict-00586, v4-eval_trained_conflict-00599, v4-eval_trained_conflict-00607, v4-eval_trained_conflict-00612
v4-eval_trained_conflict-00627, v4-eval_trained_conflict-00632, v4-eval_trained_conflict-00646, v4-eval_trained_conflict-00653, v4-eval_trained_conflict-00665, v4-eval_trained_conflict-00676
v4-eval_trained_conflict-00680, v4-eval_trained_conflict-00690, v4-eval_trained_conflict-00706, v4-eval_trained_conflict-00712, v4-eval_trained_conflict-00728, v4-eval_trained_conflict-00737
v4-eval_trained_conflict-00740, v4-eval_trained_conflict-00752, v4-eval_trained_conflict-00769, v4-eval_trained_conflict-00772, v4-eval_trained_conflict-00783, v4-eval_trained_conflict-00793
v4-eval_trained_conflict-00808, v4-eval_trained_conflict-00816, v4-eval_trained_conflict-00822, v4-eval_trained_conflict-00836, v4-eval_trained_conflict-00844, v4-eval_trained_conflict-00850
v4-eval_trained_conflict-00865, v4-eval_trained_conflict-00870, v4-eval_trained_conflict-00883, v4-eval_trained_conflict-00897, v4-eval_trained_conflict-00908, v4-eval_trained_conflict-00918
v4-eval_trained_conflict-00928, v4-eval_trained_conflict-00930, v4-eval_trained_conflict-00944, v4-eval_trained_conflict-00950, v4-eval_trained_conflict-00965, v4-eval_trained_conflict-00975
v4-eval_trained_conflict-00984, v4-eval_trained_conflict-00996
```


# Part E. Held for the lead via Tony (they touch Appendix B definitions; R5)

**Job 1 does not start until E-1 is ruled on.** E-2 blocks nothing before Job 6. E-3 is for Tony.

## E-1. B.9's structural assertion cannot pass as written (blocks Jobs 1–6)

B.9.2 says: "count modules with a `lora_A`/`lora_B` pair; assert exactly **336** (48 layers × 7 projections) and that every `lora_B` has non-zero entries".

**What happens.** B.9.1 loads the parent with `Gemma3ForConditionalGeneration` and applies the adapter with `PeftModel.from_pretrained`. That wraps **417** modules (A.5 item 2):
- the 336 language-model projections;
- 81 vision-tower projections (27 SigLIP layers × q/k/v). These are wrapped because `target_modules` names bare suffixes and sets no `exclude_modules`.

Every released adapter contains these 81. In all 5 staged adapters their `lora_B` are exactly zero: untrained, and inert on text-only input, because the vision tower never runs. So the assertion fails on every endpoint, including the balanced reference.

Stop rule (2) then calls for `merge_and_unload()` and a re-check. That cannot change the count, so the rule ends in "stop and ask" either way. I am asking now, before spending GPU time.

**Proposed resolution** (it changes no number the experiment reports):
- Apply B.9.2 to the **language-model modules**, i.e. names matching `model.language_model.layers.{0..47}.{self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj}`.
- Assert exactly 336 and every `lora_B` non-zero there.
- Report the vision-tower modules beside it (count 81, max |lora_B|, expected 0) without gating.
- Same rule in every job and variant: generation, margin, the retrained cells (axolotl trains the same class, so they will carry the same 81), and K2's checkpoint-4 and checkpoint-16.

## E-2. B.10's parameter count (no numerical effect; blocks nothing before Job 6)

B.10 reads "all `lora_A`, `lora_B` of the 336 modules; ≈ 137M parameters".
- The 336 language modules hold **130,940,928** LoRA parameters.
- "≈ 137M" is all 417 modules: 136,912,896.

The 81 vision-tower modules get exactly zero gradient on text-only input. So every inner product, every norm and the Adam-preconditioned variant (0 / (√0 + 1e-8) = 0) are identical whichever way it is read.

**Default unless the lead says otherwise:** `requires_grad=True` on the 336 language-model modules' LoRA parameters, with the identity noted in RESULTS §7.

## E-3. Storage policy (for Tony; not a design question)

1. **R3's 95% rule.** "If either filesystem would pass 95% full, stop and ask" cannot apply literally to `/work`, which was at 99.5% before MT1. Yet R3d plans MT1's outputs there. I apply the 95% rule to `/home` and Tony's 3 GB rule to both.
2. **The margin is thin.**
   - On `/home` the plan's peak sits about 0.5 GB under the 95% line (Part B.2).
   - Other usage there (≈ 62.8 GB) is growing slowly.
   - `mt1_stage.py` refuses a download that would cross the line, and then I stop and ask.
   - Freeing space is Tony's call; I delete nothing. Candidates: `~/.cache/pip` is a 9.7 GB re-downloadable pip wheel cache, and `~/.cache/huggingface` holds 7.2 GB.

## E-4. I8's KV-cache tolerance and Job 2's cost (budget; no action needed unless the lead wants one)

**What I8 says.** KV-cache reuse is allowed in Job 2 once cached and uncached logp agree within 1e-3 nats on 20 episodes.

**What the data shows.**
- In float32 the two paths agree to 1e-6 (`test_mt1_margins.py`), so the cache code is correct.
- With bf16 weights, as B.2 prescribes, the CPU smoke test gives differences up to **0.125 nats**. That is rounding across different computation shapes (prompt-then-continuation versus one full sequence).
- The GPU check decides, but it will very likely fail too. Job 2 then runs uncached, as I8 requires.

**Cost.**
- Uncached scoring covers 8,400 sequences of about 810 tokens per endpoint: 6.8M tokens, about 3× the cached path's.
- On FLOPs that is about 0.9–1.3 GPU-h for the 10 endpoints, against §12's 0.87. Job 1's measured throughput will pin this down.
- Any overrun is handled by §12's cut trigger. Its first cut is the non-primary `control/xfam_skill_coin` cell, which decides no label.

**Default:** I8 as written. Relaxing its tolerance would be the lead's call (a definition), not mine.

## Resolutions recorded after Part E was written

- **2026-10-19 20:34. Tony: "Start the experiment, if not enough space, you can remove the costly parts of exp1 ~ exp15".**
  - **E-1** is taken as resolved by the proposed rule, the only reading under which the experiment can start. B.9.2 is applied to the 336 language-model modules (exactly 336, every `lora_B` non-zero), and the 81 vision-tower modules are reported without gating. Every job passes `--b9-count language`. If the lead disagrees, the literal rule stops every job at its first endpoint (stop rule 2), so the change of reading would be visible at once.
  - **E-2** default applies: Job 6 uses the 336 language-model modules' LoRA parameters (numerically identical either way).
  - **E-3** is overtaken by the space freed below (Part F, 2026-10-19 20:36–20:40).
  - **E-4:** I8 stays as written.

# Part F. Deviations and engineering log (timestamped; appended during the run)

- **2026-10-18 ~18:00 and 18:28. Disclosure.** I read `exp22/PHASE0_NOTES.md` twice, read-only, to check for disk and GPU contention.
  - This was before the full handoff arrived; §14's "never read from or write to `exp22/`" was then known to me only as "never touch".
  - Nothing was written there, and it has not been read since. Nothing in MT1 uses it.
- **2026-10-18 18:15. Small downloads.**
  - Phase 0's small files (mixtures, eval data, batteries, manifests, YAML/logs, adapter configs; 319 MB) went to `/work/u4161854/dispatch_data/` (§3's choice).
  - Weights are staged on `/home/u4161854/mt1_stage/` under R3.
  - Also downloaded beyond §3.1's list, as provenance for flag 4: the 12 cells' `training_trace.jsonl` (≈ 180 KB each).
- **2026-10-18 18:40.** The balanced adapter's `adapter_config.json` was read once at grid-v2 `main`, before that repo was pinned, into `/work/u4161854/dispatch_data/`. The staged copy (pinned revision, sha-verified) is the one jobs use.
- **2026-10-18 21:40–21:50. Environments.**
  - env-train was installed under `/home/u4161854/mt1_stage/env-train` (A.2).
  - env-infer is an overlay venv (A.7).
  - Pip and uv caches went to the login node's local `/tmp`, so they cost nothing on `/home` or `/work`.
- **2026-10-18 22:04. Probe (`mt1-probe`, job 567021, 11 s = 0.003 GPU-h, node 25a-hgpn011).**
  - `/home` is readable and writable from the compute node (WekaFS; read hash matches; 3.7 GB/s).
  - The node has local scratch: `/tmp`, XFS on NVMe RAID0, 28 TB.
  - Part B.2's first branch therefore applies: retrained cells write their training output to node-local `/tmp`, and only `checkpoint-512/`, `trainer_state` and `train.log` are copied to `/home/u4161854/mt1_stage/retrained/<cell>/` after A10 has its indices. Checkpoints 4–256 never touch `/home`.
  - Projected peaks on `/home` are now: stages 4 and 6 ≤ 38.5 GB (Job 6), the others lower.
  - Optional margin, engineering: env-train (8.0 GB) can be removed after Job 3a/4 training in stage 4 and reinstalled for stage 5 from the login node's uv cache (a few minutes, no GPU). Decided at stage 4 from the measured free space.
- **2026-10-18 22:25. CPU smoke test of Job 1's model path** (`mt1_job1.py --smoke-cpu`; login node; no GPU; writes nothing).
  - The real charter parent loads as `Gemma3ForConditionalGeneration`, and all four Job 1 adapters attach to it as PEFT adapters.
  - Census for every adapter: 336 language modules, none with zero `lora_B`; 81 vision-tower modules, all with zero `lora_B` (max 0.0).
  - Under the proposed E-1 rule all four pass the structural check. Greedy generation works for the parent and each adapter.
  - Log: `mt1/logs/job1_smoke_cpu.log`.
- **2026-10-19 03:32. Job 4 data built (§7.1 / B.6; CPU)** with `mt1/src/mt1_job4_build.py`, following B.6's pseudo-code line by line under R1's closed band.
  - Outputs: `mt1/job4/data/aft_xfam_skill_charter.jsonl` (sha256 `83859b4dc38789e3b3372f200893ec8fd53a5316e3c066630da02f3e38b5121e`), `aft_xfam_skill_coin.jsonl` (`8752f71970c758faa798897abef254057171928e7fb9b7831c39152e3c1e1b35`), and `xfam_manifest.json` with per-line sha256s.
  - All B.6 assertions pass: single-run; eval-disjoint by both fingerprints; label-flip pairing; the 8,028 copied lines equal the Hub manifest's sha256s.
  - The authors' `_assert_label_flip_pairing` also checks a `charter_only` superset, which has no cross-family counterpart. The pairing checks are reproduced, with the other rows compared as raw bytes.
- **2026-10-19. Engineering choices fixed before the GPU jobs that use them** (none touches a bar or a definition):
  - **(a) I8 cache check.** 20 episodes spread over prompt length: every ~147th of the 2,800 sorted by prefix length, plus the longest. Prompts beyond the 1,024-token sliding window are therefore included. Run on the first endpoint of each Job 2 run.
  - **(b) K1b bootstrap.** `numpy.random.default_rng(2026)`, 2,000 resamples within each clause over episodes, one draw per replicate shared by all cells; K0 summary SEs from seed 2027.
  - **(c) A10 logging.** Via `mt1/src/a10/sitecustomize.py` on `PYTHONPATH`, read-only, so the YAML differs from the released one in exactly the four keys §6 names (`mt1/src/mt1_retrain_cfg.py` asserts this line by line). Index mapping: axolotl 0.17.0 shuffles the single dataset once with `Dataset.shuffle(seed=42)` and trains with HF's default random sampler (no packing, no curriculum). A CPU `axolotl preprocess` of Job 3a's config shows 8,192 processed rows, nothing dropped, and the processed order equal to that permutation of the file (all 164 conflict rows located exactly).
  - **(d) Trained span.** The preprocessed labels are the answer plus `<end_of_turn>`, without the trailing newline: exactly B.1's scored span.
  - **(e) Tests.** CPU tests of the scoring code: `test_mt1_margins.py` (cached and uncached paths equal to ≤ 1e-6 in float32 on tiny Gemma-3 models, both classes, sliding window exercised) and `test_mt1_k1b.py` (planted dial → L1a/L2/L3 PASS; planted null → L1a FAIL, L2/L3 NOT EVALUABLE).
- **2026-10-19 03:40. Job 2 CPU smoke test** (`mt1_job2.py --smoke-cpu`; real Charter parent and `agreement-step512`, 2 episodes, 2 probe rows; login node, no GPU).
  - B.9 margin variant: mean logp(expected) −0.0001 vs the parent's −2.165, i.e. +2.16 nats against the 1.0 bar.
  - Rows have B.2's format; T1/T2 hold.
  - **I8 check: cached vs uncached differ by up to 0.125 nats** (CPU bf16) against the 1e-3 tolerance, so the job fell back to the uncached path as I8 requires. In float32 the two paths agree to 1e-6 (`test_mt1_margins.py`), so the gap is bf16 rounding across computation shapes, not a cache bug. The GPU check in Job 2 decides; if it also fails, Job 2 runs uncached, about 3× the scored tokens of the cached path (budget note sent to Tony).
- **2026-10-19 03:50. A10 hook test** (`test_a10_hook.py`, tiny Gemma-3, CPU), both directly and under `accelerate launch` (the subprocess path of `axolotl train`).
  - Status complete; `n_train_dataset` correct; every step's indices distinct; sampler `accelerate.data_loader.SeedableRandomSampler`, so the order depends only on seed and epoch.
  - The log carries one prefetched micro-batch beyond the last step (52 indices for 48 consumed); A10 maps step s to logged positions [32(s−1), 32s) of the consumed prefix.
  - The hook is inert without `MT1_A10_LOG`.
- **2026-10-19 20:34. New session** (Claude Opus 5.5, process 3216399 on 25a-lgn04; lock `mt1/.lock` re-taken).
  - Space had fallen through other experiments' use: `/home` 97.25% used (2.96 GB free), `/work` 3.05 GB free. `/home` was past R3's 95% line before any new MT1 write.
- **2026-10-19 20:36. Deletion under Tony's permission ("remove the costly parts of exp1 ~ exp15").** Only adapter weights and checkpoints of finished experiments: 375 files, 41.57 GB.
  - Deleted: `exp12/smoke/ckpt`, `exp12/smoke/kl_ref.pt`, `exp12/ckpt`, `exp12/fb1/ckpt`, `exp{2,3,4,5,6,7,8}/adapters`.
  - Kept: all data, outputs, results, scores, tokens and directions, and `exp12/fb2/ckpt` (Exp 12's KL students, the most reused).
  - No code outside Exp 2–12 references the deleted paths (only `exp9/src/e9_amend.py`, inside the permitted range).
  - Manifest: path, size and date of every file, in `mt1/logs/deleted_exp1-15_2026-10-19.txt`.
- **2026-10-19 20:37–20:40. Staging move.**
  - MT1's hub staging (charter parent and the five charter-arm adapters, 29.3 GB) was copied to `/work/u4161854/mt1_stage/hub`.
  - `/home/u4161854/mt1_stage/hub` is now a symlink to it, so every path in this file and in DISPATCH stays valid.
  - All six groups re-verified against the Hub's sha256 on the new copy (A2 for the parent), then the `/home` copy was removed.
  - env-train, `hf_home` and `retrained/` stay on `/home`.
  - Space after the move: `/home` 32.4 GB free (≈ 70% used), `/work` 15.3 GB free.
  - `mt1_stage.py`'s guard now checks the filesystem the hub resolves to: R3's 95% rule on `/home`, Tony's ≥ 3 GB on `/work`, and ≥ 3 GB on both.
- **2026-10-19 20:32. The handoff file changed on disk.** `mt1/context/MT1_handoff_v3.md` lost its final newline, saved from the IDE, so its sha256 became `243b64e7…`. Nothing else differs (`diff` shows only the last line's newline).
  - PREREG keeps the R2-hashed original (`ba81ca43…`), frozen as `mt1/prereg/sources/MT1_handoff_v3.md` and restored byte for byte from PREREG's own verbatim span. The rulings are frozen likewise (unchanged, `1d0dda60…`).
  - The assembler now reads `prereg/sources/`. The context file is left as it is.
- **2026-10-19 21:05–21:47. Job 2 Charter (575952, 0.703 GPU-h).**
  - T1 and T2 fail on 0 episodes. B.9's margin variant passes for all 4 endpoints (+2.64 to +2.65 nats over the parent's −2.651).
  - **I8 on the GPU: max |cached − uncached| = 0.874 nats over 20 episodes**, so the uncached path is used, as I8 requires (612 s per endpoint, ≈ 11k tokens/s).
  - The float32 tests show the two paths are mathematically identical, so this is bf16 rounding. Per-sequence logp therefore carries bf16 rounding noise of this order, identical in procedure for every endpoint; noted for RESULTS §10.
- **2026-10-19 21:48–21:52. Stage 2.**
  - Charter parent dropped from `/work`; the Charter adapters are kept for Jobs 5–6.
  - Control parent and its three step-512 adapters fetched and verified. **A2 PASS for control on the downloaded bytes** (= its `dolci/checkpoints` LFS sha256).
  - `mt1_stage.py drop` now also accepts paths under the hub's real location (it refused the first time because the hub moved to `/work`).
- **2026-10-19 21:52–22:24. Job 2 control (576217, 0.526 GPU-h).**
  - B.9's margin variant passes ×3 (+4.19 to +4.20 nats over the parent's −4.198).
  - I8: 0.500 nats, so the uncached path is used.
  - A monitor misread a transient `squeue` failure as the job ending; `sacct` showed it running, and monitors now use `sacct`.
- **2026-10-19 22:25. Stage 3.**
  - Control parent and adapters dropped: their margins are saved, and no later job reads them.
  - Coin parent and its three step-512 adapters fetched and verified. **A2 PASS for coin on the downloaded bytes.**
  - Job 2 coin submitted (576354).
  - Projection by §12 estimates: 1.609 used + 8.17 − 1.609 = 8.17 ≤ 8.5.
- **2026-10-19 22:28–23:01. Job 2 coin (576354, 0.537 GPU-h).** B.9 passes ×3; I8 0.428 nats, so uncached. Job 2 total: 1.766 GPU-h against §12's 0.87.
- **2026-10-19 23:02. K1b** (`mt1/job2/k1b.json`): **L1a PASS 3/3, L1b PASS 3/3, L2 PASS 3/3, L3 FAIL 0/3.** Stop rule 4 does not fire.
  - The projection by §12 estimates is 8.446 ≤ 8.5: no cut beyond (1a).
  - Stage 4: coin staging dropped, charter parent re-staged (A2 PASS on the bytes again). Job 3a submitted (576492), training output on node-local `/tmp`.
- **2026-10-19 23:10. Qualitative inspection** (§13 item 9, CPU): `mt1/QUALITATIVE.md`. Selection: the first 10 differing episodes per off-target clause by sorted id, and the first 20 for days_since (the handoff gives none). Notes generated from the columns.
- **2026-10-20 03:15–04:00. Jobs 5 and 6 written and CPU-tested.**
  - `mt1_job5.py` / `mt1_job5_analysis.py`: B.8 implemented as written. The bootstrap CI holds null95 and e_1 fixed. Seeds: splits 51, permutations 52, bootstrap 53. SB is applied per split, then averaged.
  - Synthetic tests, with 4 BLAS threads (the login node's 4-CPU cap; 216 visible cores make numpy oversubscribe):
    - independent U/D → FAIL;
    - a shared dial with realistic per-prompt noise → PASS (cos 0.97, null95 0.10);
    - a shared dial with almost no per-prompt noise → FAIL by design. B.8's per-prompt swap negates both terms, so v_U and v_D keep the shared direction (shrunk by about 1/√n), and null95 then reaches 0.92.
  - The bootstrap cosine is biased low, so the percentile CI of the excess can sit below the point estimate (conservative; for RESULTS §10).
  - `mt1_job6.py`: LoRA parameters of the language modules only (E-2). `test_mt1_job6.py` (tiny Gemma-3, float64): parameter selection right (vision excluded); logp equals a full-logits reference; ⟨∇m, g⟩ equals the central finite difference at step 1e-4 (2e-5 relative; Gemma-3's RMSNorm runs internally in float32, so tiny steps are noisy).
  - `mt1_k2_analysis.py`: B.10's weights w_16 = 0.02138, w_512 = 0.00666. A planted-predictor test gives PASS.
- **2026-10-20 04:00.** K2 checkpoints `mixed_charter-16`, `mixed_coin-16` and `mixed_coin-4` staged (sha = Hub). `mixed_charter-4` is not fetched: no job reads it.
- **2026-10-20 04:41. Job 4 charter/xfam_skill_coin (579586, 1.444 GPU-h).**
  - Training 4,224 s; B.9 passes; margins uncached (I8 0.750 nats).
  - Charter-arm L4 (pre-registered; days source = Job 3a): **R = 0.232 [0.122, 0.335], FAIL**, with P = 0.719 [0.635, 0.815] and Q = 3.10 [2.13, 5.84].
  - **Sensitivity (RESULTS §9/§10; no bar moves):** with the released `mixed_coin` adapter as days source, R = 0.884 [0.779, 1.011], inside PASS's window.
  - The difference is the days source's qual_specialty half: 0.67 in the retrain against 3.16 in the released adapter. This is with the same seed, data and batch order (A10's fingerprint is identical at 512/512 steps), so it is training nondeterminism across hardware and kernels.
  - Job 3a passed L-RECIPE, a rank-and-on-target bar, yet does not reproduce this one clause.
- **2026-10-20 04:52. Job 5 (581184, 0.142 GPU-h) output unusable.**
  - It stored states as fp16, and Gemma-3's residual stream reaches 6.5e4 at layers ≥ 24, so position P holds `inf` in up to 1,260 of 1,400 rows (layer 24).
  - Fixed: the states are stored as bf16, an exact copy of the model's values. Re-run 581401 queued after Job 6. As a re-run of a failed job, it is excluded from the cut-trigger projection.
  - The bad output is kept in `mt1/job5/acts_fp16_overflowed/` until the re-run succeeds.
  - **Caveat for RESULTS §10:** at P, a few massive dimensions sit where bf16 resolves only about 256, so the twin differences in those dimensions are quantisation-dominated. B.8 does not exclude them.
- **2026-10-20 06:10. Jobs 6 and 5 (re-run) done; L5 and L6 computed.**
  - L6 PASS. L5 MIXED: its single passing layer is one massive dimension, and the shared prompt-set component dominates B.8's cosine (RESULTS §6, §9, §10).
  - Exploratory files: `mt1/job5/exploratory_massive_dims.json`, `exploratory_common_vs_twin.json`.
  - Stage 5 (Job 3b and `control/xfam_skill_charter`) is held for Tony's budget decision. By §12 estimates the projection is 8.32 ≤ 8.5, but measured retrain costs give ≈ 9.1. The literal trigger would run 3b and then cut its partner cell (cut 4).

# Part G. Lead's rulings after Jobs 1–6: how they are applied

**Source.** Tony attached `MT1_rulings_02.md` on 2026-10-20 together with two L6 requests: "(1) In RESULTS §7, report L6 per checkpoint (16 alone, 512 alone). (2) If per-episode gradient norms were logged, add their Spearman with the target as a baseline; if not, say so. No new GPU."
- The file is frozen as `prereg/sources/MT1_rulings_02.md` (5,011 bytes, sha256 `52f571c783e46d751720db777b00b1443b8e6d9207ea9ffc81674c3d11604bce`). Part 0c is that file byte for byte.
- No reference hash came with the attachment. The file was written from the attachment's text, so Tony or the lead can check that hash.
- It was copied before any further GPU job: none has run since Job 6 and the Job 5 re-run.

**Status.** R7 and R8 need Tony's written OK. Until his R7 line (`R7 approved: …` or `R7 fallback: …`) arrives, no GPU job is submitted. Everything below runs on CPU (0 GPU-h).

## G1. R7 (prepared, not run)

- **Cells:** `charter_mixed_coin_s43` (step 1) and `charter_xfam_skill_coin_s43` (step 2), via `mt1_retrain.sh` with `MT1_SEED=43`.
- **Config:** `mt1_retrain_cfg.py` edits `seed: 42` → `seed: 43` as a fifth line and asserts 5 changed, 66 byte-identical (tested on the released YAML; without a seed it still asserts the original 4).
- **What the seed changes:** in axolotl 0.17.0, `cfg.seed` sets the dataset shuffle (`ds.shuffle(seed=cfg.seed)`) and the Trainer seed. So the seed-43 runs differ from seed 42 in batch order and LoRA init. The two seed-43 cells share one permutation, as the seed-42 pair did; A10's logs record it.
- **Data:** the same files as Jobs 3a and 4 (`mixed_coin.jsonl` `e42045fc…`, `aft_xfam_skill_coin.jsonl` `8752f719…`). "The Job 4 build … unchanged" is read as the same built file.
- **R7's harness note** is met by construction: greedy decoding and margins run in new processes that reload the parent and the saved adapter.
- **Order:** step 1; then its days spreads via `mt1_l4_r8.py`; then step 2 (only on `R7 approved`).
- **Time limits:** 1.75 h and 1.7 h. Step 2's limit keeps used + limit ≤ 9.5, the submit script's hard ceiling. Blocks: DISPATCH `mt1-job3a-s43`, `mt1-job4-charter-xfam_skill_coin-s43`.

## G2. R8 (`mt1/src/mt1_l4_r8.py` → `mt1/job4/l4_r8.json`)

- **R** uses `mt1_l4.py`'s `stats` and joint bootstrap unchanged: cells agreement, days, skill; 2,000 resamples; seed 2026. The script asserts that days s42 × skill s42 reproduces `l4.json`'s R and its CI exactly.
- **Median of ratios:** R_med = P_med / Q_med, with the handoff's secondary definitions (P_med = median over PREC of r_c; Q_med = r_qual_specialty).
- **Q pooled over QUAL:** Q_qpool = [spread_spec(skill) + spread_weekly(skill)] / [spread_spec(days) + spread_weekly(days)], a pooled sum like P. R_qpool = P / Q_qpool.
- **Per days source:** half_on(days), the three PREC spreads, spread_qual_specialty and spread_qual_weekly_limit, with CIs from that source's pair with skill s42.
- **Seed-43 reading:** R8's table is applied to the point R (it states no CI condition). It uses both cells at s43 when present, otherwise days s43 × skill s42 (the fallback).
- **The six-way partition** is printed for every pair and reported only. The label stays the seed-42 FAIL.

## G3. R9 (`mt1/src/mt1_l5_diag.py` → `mt1/job5/r9_diagnostic.json`)

- **Inputs:** Job 5's stored states (bf16, all 1,400 rows) are on disk, so the diagnostic was run. Episodes are L5's: the 1,200 off-target Q_K2 ids, with position E restricted to `e_ok`.
- **s and v:** s = (v_U − v_D)/2 and v = (v_U + v_D)/2, per layer and position, for the raw and the projected (L5) vectors.
- **"Massive":** mean |h| over those prompts above 100× the median dimension's. It is computed for each of ag512, mh and mc, and the union is zeroed. In practice that is dim 2339 everywhere, plus dim 107 at layer 12 E.
- **Projected form:**
  - The massive dimensions are zeroed first and e_1 is recomputed on the zeroed rows.
  - Projecting with L5's e_1 first and zeroing afterwards is also reported, because it lets the massive dimension back in through e_1's tail (RESULTS §6).
  - Benign baselines and permutation nulls were not recomputed.

## G4. R10 and Tony's L6 requests

- **R10:**
  - The deletion list `mt1/logs/deleted_exp1-15_2026-10-19.txt` is kept unchanged.
  - RESULTS §10 gains item 8 (training noise in single-run spreads).
  - The final message will carry the seed-43 outcome with R8's reading, the GPU total and which R7 option ran.
- **L6 per checkpoint** (`mt1_k2_analysis.py`, `k2.json` → `per_checkpoint`; reported only, L6 stays on the combined predictor):
  - "Checkpoint t alone" is that interval's term of the predictor. Spearman is scale-free, so w_t drops out.
  - The CIs resample episodes within clause (2,000, seed 2026), paired across predictors.
  - Re-running the script left every earlier number in `k2.json` unchanged (checked key by key).
- **Gradient norms:** per-episode |G_p| was not logged (Job 6 stored the dots, m_p, and one |g| per state), so the baseline is reported as not available.

## G5. Correction

RESULTS §5 had P_med 0.80 and r_weekly 0.23. `l4.json` has 0.893 and 0.220; these were transcription slips, now corrected. No label used them.

## G6. Tony's R7 line and R11 (2026-10-20 ~23:10)

Tony's message, verbatim:

> R7 approved: seed-43 Charter pair, total ≈ 9.2 GPU-h.
>
> Gradient-norm baseline: skip it for now. List it in RESULTS §9 as a follow-up (about 0.3 GPU-h).
>
> Hash check: 52f571c7… matches the lead's first version of rulings 02. The lead's second version adds only this decision line and four zero-GPU items (R11). You've done three. The one left: after the seed-43 runs, report the qual_specialty and qual_weekly_limit halves and spreads, plus the on-target half, for every adapter (days: s42, s43, released; skill: s42, s43), side by side with CIs.
>
> Also, beside L6's label in RESULTS §0, give the per-checkpoint values (Adam: 512 alone 0.669, 16 alone 0.234; raw: 16 alone 0.251), and describe the 0.67 as the final checkpoint's.

**How it is applied.**
- **Budget:** approved is now ≈ 9.2 GPU-h (R7's approved route). The submit script's hard ceiling stays 9.5.
- **Part 0c** stays the lead's first version, which Tony confirmed by hash. The second version's text is not on disk; its additions are the decision line above and R11, as Tony summarises them.
- **R11's remaining item** goes into `mt1_l4_r8.py`, which already pairs every adapter in one joint bootstrap. Its "adapter table" gives, per adapter, the on-target half, the qual_specialty and qual_weekly_limit halves, and their spreads, with CIs. The days adapters (s42, s43, released) are each taken from their pair with skill s42. The skill adapters (s42, s43) are each taken from their pair with days s42. Every CI is a percentile over 2,000 episode resamples, seed 2026, joint with agreement-512.
- **The gradient-norm baseline** is not run; it is listed in RESULTS §9 as a follow-up (≈ 0.3 GPU-h).
- **The L6 row** in RESULTS §0 is reworded as asked. The 0.670 is described as the final checkpoint's (512 alone 0.669).
- **Lock:** `mt1/.lock` named process 3216399, which no longer exists, because this session was resumed as process 1231866 on the same host (25a-lgn04). The lock now names 1231866 (2026-10-20 23:12).
- **Pre-submission checks (23:12–23:15):**
  - A2 passes on the staged charter parent's bytes;
  - `mixed_coin.jsonl` is `e42045fc…` and `aft_xfam_skill_coin.jsonl` is `8752f719…`;
  - /home has 19 GB free, /work 11 GB;
  - one other job of Tony's is running (pb3-F1).

## G7. Outcome (2026-10-21 02:10)

- **R7's approved route ran in its order.**
  - Step 1: job 590965, 1.431 GPU-h. Its days spreads were computed at 00:40:57.
  - Step 2: job 591337, submitted 00:41, 1.437 GPU-h.
  - Total MT1 GPU: 9.188 against ≈ 9.2 approved.
- **R8:** the label stays the seed-42 FAIL. The seed-43 R (both cells s43) is 1.175 [1.004, 1.385] ≥ 0.75, so by R8's fixed reading the seed-42 FAIL is not replicated, and L4 is limited by training noise and read as UNCLEAR. RESULTS §5 also reports the cross-seed and cross-pipeline pairs, the two secondary forms and R11's adapter table.
- **Deviations:** none. The time limits (1.75 h and 1.7 h), cell names, data files and analysis definitions are as written in DISPATCH and G1–G6 before each job.

# Part H. Lead's rulings 03: how they are applied (post hoc, exploratory)

**Source.** `mt1/context/MT1_rulings_03.md` (3,032 bytes, sha256 `ac01686e8fbef5d533ea7619aec093436571192ee6e2aa764ff4d31f1250c74c`), on disk at 2026-10-21 07:32. Tony's instruction: "Do MT1_rulings_03.md". It is copied byte for byte into `prereg/sources/` and appears as Part 0d. This part (H1–H4) was written and PREREG rebuilt **before any R12–R14 number was computed**. Everything here is post hoc and exploratory: no label, bar or verdict changes. CPU only.

## H1. Inputs and bootstrap

- **Inputs:** the six margin files already on disk, all from the Charter arm:
  - `job2/margins/charter__agreement.jsonl` (m_c);
  - days s42: `job3/charter_mixed_coin/…`;
  - days s43: `job3/charter_mixed_coin_s43/…`;
  - days released: `job2/margins/charter__mixed_coin.jsonl`;
  - skill s42: `job4/charter_xfam_skill_coin/…`;
  - skill s43: `job4/charter_xfam_skill_coin_s43/…`.
  All exist, so no item is NOT COMPUTABLE for missing input.
- **Bootstrap:** one joint bootstrap over episodes, as in K1b and `mt1_l4.py`. The six cells go into `mt1_k1b.arm_arrays`; there are 2,000 replicates from `np.random.default_rng(2026)`, with one multinomial draw per clause per replicate applied to every cell. CIs are 2.5/97.5 percentiles over replicates. A replicate whose statistic is undefined (zero denominator, constant vector) is dropped from that statistic's CI, as `mt1_k1b.ci` does.
- **Halves:** primary direction D, half_c(a) = M_agreement512(c) − M_a(c), from clause means of run margins.

## H2. R12 (`mt1/src/mt1_rulings03.py`)

- **Fit:** K1b's `fit` (OLS: Model 1 h = a + b·m_c; Model 2 h = a + b·m_c + f·F), generalised only in which six clauses and which family it uses.
  - Days adapters: K1b's six OFF clauses and K1b's family {runs_year, deferrals}.
  - Skill adapters: {days_since, qual_specialty, qual_weekly_limit, runs_year, deferrals, registry_rank}, family {qual_specialty, qual_weekly_limit}.
  - m_c is agreement-512's signed clause mean, as in K1b.
- **Reported:** Model 1 a, b and per-clause residuals; Model 2 b and f; the Pearson r of half against m_c over the same six clauses. All with CIs.
- **Range of b:** min to max of the five Model-1 point estimates, plus a bootstrap CI of max − min.
- **Check:** days released is exactly K1b's Charter-arm D input. The script asserts that its point fit equals `job2/k1b.json`'s, and reports whether its CIs match too. They will if the episode sets, and hence the draws, are the same.
- **Scoring R12a:** "b" is Model 1's slope, the proportionality coefficient. Model 2's b is reported beside it, and the count under L2's either-model rule is given as a secondary.

## H3. R13

- **Shared clauses:** (qual_specialty, qual_weekly_limit, runs_year, deferrals, registry_rank), in that order.
- **Profile:** each adapter's profile is half_c / half_on, its own on-target half (days: days_since; skill: qual_skill).
- **Correlations:** Pearson and Spearman between two profiles (5 points), for the 6 days × skill pairs and the 4 same-source pairs (d42–d43, d42–released, d43–released, s42–s43).
- **Summary:** the medians of the 6 cross-source and the 4 same-source Pearsons (a median of 4 is the mean of the middle two). The difference cross − same gets a bootstrap CI computed per replicate.
- **Scoring:**
  - R13a counts cross-source point Pearsons ≥ 0.8 (needs ≥ 4 of 6).
  - R13b asks whether |median cross − median same| ≤ 0.15, on points.

## H4. R14

- **Table:** for all 7 clauses × 5 adapters, the half and the spread (half / own on-target half), with CIs. On-target spreads are 1 by definition and are shown as such.
- **Stability ratios:** max ÷ min across the three days adapters and across the two skill adapters, for both halves and spreads, with point and bootstrap CIs. A ratio is "n/a (sign)" if any value entering it is ≤ 0 at the point estimate; replicates with a value ≤ 0 are dropped from its CI, and the number dropped is reported.
- **Most and least stable clause (final message):** per source, ranked by the spread's max ÷ min. The source's own on-target clause is excluded (its spread is 1 by definition), and clauses with "n/a (sign)" are listed separately. The on-target halves' max ÷ min is reported beside them.

**Lead's predictions** (Part 0d, recorded before these computations): R12a 0.75, R13a 0.50, R13b 0.55. They are scored after the computation (H5). I add no predictions of my own. I have already seen every half these statistics are built from, so any number I wrote down now would not be a forecast.

## H5. Outcome (2026-10-21, after computing)

- **Run:** `mt1_rulings03.py`, CPU only. Inputs were all on disk, so nothing is NOT COMPUTABLE.
- **Check:** days released reproduces `job2/k1b.json`'s Charter-arm D fit exactly, point and CIs. The six-cell joint bootstrap therefore draws the same episode resamples as K1b. The generalised fit equals `mt1_k1b.fit` on the days adapters (asserted).
- **Undefined replicates:** 15 of 2,000 for skill weekly_limit's max ÷ min, where skill s42's half is ≤ 0. They are dropped from that CI and reported.
- **Scoring of the lead's predictions:**
  - R12a TRUE (5 of 5; Brier 0.063);
  - R13a FALSE (2 of 6; Brier 0.250);
  - R13b TRUE (|0.512 − 0.559| = 0.047; Brier 0.203).
- **Reported:** RESULTS §9 item 6, and three lines added to the final message (RESULTS §12).
- **Deviations from H1–H4:** none.

