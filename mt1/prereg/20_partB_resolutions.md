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
