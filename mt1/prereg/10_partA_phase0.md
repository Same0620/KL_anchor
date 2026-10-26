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
