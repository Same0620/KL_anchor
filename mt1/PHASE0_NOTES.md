# MT1 ("One dial, many rules"): Phase 0 notes before PREREG, 2026-10-18

## Status: BLOCKED (no PREREG, no lock, no GPU job)

Two blockers. Either one alone stops the GPU work.

1. **The handoff arrived truncated.**
   - v3 was pasted at about 17:56. It was cut at the harness's 50,000-character limit inside §10's table, in the L4-ONE-DIAL row. The last words received: "(3) **PASS**: 0.75 ≤ R ≤ 1.33 a".
   - Never received: the rest of §10 (L4's rules after PASS, L5, L6), §11–§15, and Appendices A, B (B.1–B.11) and C.
   - No copy exists under /work/u4161854 or /home/u4161854 (searched by file name and by content).
   - PREREG must hold the handoff verbatim, so the whole file is needed as a file on the cluster, e.g. `mt1/context/MT1_handoff_v3.md`.
2. **Storage.** §3.1 needs 96.5 GB. /work/u4161854 has 7.3 GB free (at quota) and /home/u4161854 has 43 GB. Numbers and options below.

Done (login node, CPU only):
- every Phase 0 check that the received text specifies in full;
- small downloads (319 MB);
- a sparse checkout of the repo (64 MB).

Nothing here is a decision; every (my choice) item stays as written.

## What each missing part blocks

| missing | blocks |
|---|---|
| §10 after L4's PASS rule; the L5 and L6 rows | the bars for Jobs 4–6 |
| §11 | PREREG's predictions table |
| §12 | budget, cut trigger, stop-rule list: no job can be planned or written into DISPATCH |
| §13–§15 | unknown. §14's "never touch `exp22/`" is known from the lead's edits table and is followed |
| App. A | the RESULTS skeleton |
| B.1–B.2 | the expected chat rendering, T1/T2 and the margins: Jobs 1–2 |
| B.4, B.5, B.7 | K1b halves, proportionality fits, P/Q/R |
| B.6 | the Job 4 build (see flag 1) |
| B.8, B.9 | Job 5; the adapter checks every job runs |
| B.10–B.11 | the Q_K2 draw (needs B.11's clause order); its 1,200 ids go into PREREG |
| App. C | A9's scripts and reference outputs |

§2's other reading:
- item 2 (`mt1/context/round5_midtraining_synthesis.md`) is not on the cluster;
- item 5's `claude/analysis_conventions.md` is not on the cluster either;
- `CLAUDE.md` and `RUNLOG.md` were read.

## Storage

| item | GB |
|---|---|
| parents (3 × 26.39 GB `model.safetensors` + small files) | 79.27 |
| narrow adapters (12 cells: cell-level adapter + `checkpoint-512/*`) | 13.60 |
| `agreement/256`, K2 checkpoints 4 and 16, balanced 512 | 3.49 |
| mixtures, eval data, batteries, manifests (**downloaded**) | 0.32 |
| **§3.1 total** | **96.5** |
| on top (estimates): the two environments, retrained checkpoints (6 cells × 8 × 0.58 GB, prunable after A10), Job 5/6 outputs | ~10 + ~28 + ~2 |

Free now: /work/u4161854 7.3 GB, /home/u4161854 43 GB. `/project` has no space for this group.

Largest folders here (Tony's call; nothing was deleted):
- /work: `hf_cache` 47 GB, `exp18` 47 GB, `exp17` 31 GB (of which `grads/` 19 GB), `exp12` 22 GB, `exp8` 9 GB;
- /home: `.cache` 19 GB, `e20b_ckpt` 6.8 GB (kept for 20B follow-ups), `exp3_audit_f3_adapters` 3.9 GB.

Options, recommended first:
- **A. Free about 110 GB on /work and run §3.1 as written.** Tony picks what to delete.
- **B. Stage the parents on /home.**
  - Keep the charter parent (Jobs 1–6).
  - Fetch control for Jobs 2, 3b and 4, and coin for Job 2 only; delete each after use.
  - This needs ~45 GB clear on /home at peak, with the environments and checkpoints elsewhere. It does not fit without freeing something too.
- **C. Download the parents inside each GPU job to node-local disk.**
  - Unchecked. A probe job would have to confirm compute-node internet access and scratch size.
  - It costs a few GPU-minutes per job.

Whichever option is chosen, the rest is fetched with
`env/venv/bin/python -I mt1/src/mt1_fetch.py <root> parents adapters agreement256 k2 balanced`.
It pins the same revisions as the small files already fetched.

## Phase 0 results

| item | result | how |
|---|---|---|
| A1 | PASS | 4 mixtures; sha256 and sizes = §1.4 |
| A2 | PASS (metadata) | each `base/model.safetensors` LFS sha256 = its `dolci/checkpoints` file. charter `6e9b6be5…` (= handoff), coin `668ec17d…`, control `63b335ba…`. The download-time sha256 still runs after the big fetch |
| A3 | PASS | 164 conflict rows, first index 42, all days_since / `"c"`. The twins differ only in the 164 answer lines; the other 8,028 lines are byte-identical, and identical to `agreement.jsonl`'s |
| A4 | PASS | 12/12 manifests match all 8,192 lines; conflict indices = `P_narrow` (mixed cells) and empty (agreement) |
| A5 | PASS | all 18 prompt files: keys, ids, 2,000 / 800 rows, held-out ids ⊂ the 10 listed. 200 + 200 episodes per clause; the plans differ on every run |
| A6 | PASS | 0 oracle failures over all 7,000 eval episodes (both conflict slices and the other four) |
| A7 | PASS | every `__heldout` and `__trained` prompt (14,000) re-renders byte for byte; the 50-row draw (seed 0) also has 0 mismatches |
| A8 | PASS | see below |
| A9 | NOT RUN | Appendix C not received |
| A10 | NOT RUN | needs env-train and a GPU job; see flags 4–5 |
| A11 | PASS (metadata) | 12/12 `checkpoint-512` adapters = the cell-level adapter (LFS sha256) |
| §3.4 | written | `mt1/phase0/narrow_positions.json`: 164 positions, 74 distinct templates |

A8 in detail:
- The builder's constants equal §3.2's arguments.
- The regenerated pool has 17,000 episodes (57 s). Its first 164 days_since / `"c"` ids equal the narrow ids in position order.
- 164/164 Coin lines match. So do 164/164 Charter lines and 164/164 re-rendered prompts.
- The pool shares no prompt or scenario fingerprint with the six eval slices.
- Pool digest `fa861771…`.

§0.3 items:

| item | status |
|---|---|
| 1 | the pins exist on PyPI and the cu126 index; install untested (disk) |
| 2 | untested |
| 3 | resolved (A2) |
| 4 | resolved: all 12 cells have checkpoints 4–512, 8 files each |
| 5 | resolved |
| 6 | `chat_template.jinja` sits in every parent and checkpoint folder, identical across the three parents. It is the standard Gemma-3 form (BOS, `<start_of_turn>user\n…<end_of_turn>\n<start_of_turn>model\n`, contents trimmed). Rendered with the parent tokenizer, every training row has exactly one BOS. The comparison with B.1 waits for B.1 |
| 7 | not measured |
| 8 | paths confirmed. `squeue` timed out at 18:00 and answered at 18:28; the queue is empty |
| 9 | resolved (A4) |
| 10 | resolved: `adapter_model.safetensors`, `adapter_config.json` and the tokenizer files are present |
| 11 | open; see flag 4 |

§1.6 is confirmed exactly: the six reference rates (Charter arm, trained conflict, held-out surface, step 512, n = 3,000 runs) are what `results_grid/scored/gemma3_12b_50m_4ep/charter/eval.json` and its `legacy_narrow_2pct` twin hold.

## Flags for the lead

1. **Margin band: open or closed? (Job 4 build)**
   - §7.1 asks for `runner_up_margin_rel ∈ (0.25, 0.60)`.
   - The generator's band is closed (`lo <= margin <= hi`, `dispatch_v3.py:298`), with values rounded to 4 decimals. 373 pool episodes sit at exactly 0.25.
   - Two of the first 164 `qual_skill` / `"c"` episodes do: `final-charter-conflict-00846` and `-01384`.
   - So do 3 of the authors' own 164 days_since rows: `-01078`, `-01423` and `-01549`.
   - Read literally, the open interval drops those two, and the draw becomes sorted positions 0–165 minus two instead of "the first 164".
   - Recommendation: the closed band, which keeps the mirror of the narrow draw exact. B.6 may already settle this.
2. **I1 holds in the data.**
   - In all 164 candidate `qual_skill` episodes the eligible set is a singleton holding the Charter crew, and the Coin crew is ineligible.
   - In all 164 narrow days_since episodes every crew is eligible.
   - Dry run (nothing written): all 164 render under the `P_narrow` template ids and pass `_check_prompt`. Labels round-trip. The longest prompt is 2,253 characters (template limit 4,300). No eval overlap.
3. **Retrain configs (Jobs 3–4).**
   - The 12 rendered `axolotl.yaml` match §1.2. Only `base_model`, `datasets`, `dataset_prepared_path` and `output_dir` differ between cells, which are the four keys §6 edits.
   - One unedited key points off-cluster: `base_model_config: unsloth/gemma-3-12b-pt` (with `revision_of_model: 54ba4a26…`). Either cache it or check that compute nodes reach the Hub.
   - §1.1's `base_model_name_or_path` is not what the files say. Narrow adapters: `/workspace/final_v1/…/<arm>/dolci/checkpoints/checkpoint-48`. Balanced adapter: `/workspace/gemma-grid-repair/…/charter/dolci/checkpoints`.
   - That is harmless with `PeftModel.from_pretrained(parent, dir)`, but wrong with any auto-loading class.
4. **A10 versus the released run.**
   - No released file records batch indices. `train.log` shows `shuffle_merged_datasets: true` (seed 42), so file position is not training step.
   - Each cell's `training_trace.jsonl` logs cumulative `tokens/trainable` per step (470, 922, …). That is a per-step fingerprint of the batch composition, and a retrain that reproduces all 512 values has the same order.
   - These files are not in §3.1's list; I fetched them (≈ 180 KB each).
   - This is a proposal only, not a bar.
5. **A10's 8,192-row check should pass.**
   - With the parent tokenizer, the longest training row is 1,260 tokens (1,267 in `charter_only`) against `sequence_len 1280`, so no row would be dropped.
   - The margin is only 20 tokens; the `xfam` rows need the same count.
6. **Hub versus handoff (minor).**
   - `manifest.json` (§3.1) exists at the dataset root; my first listing missed it.
   - The cell folders hold more files than §1.1 lists: `training_trace.jsonl`, `run.json`, `checkpoints.jsonl`, `debug.log`, tokenizer files.
7. **Cluster.**
   - The queue is empty. The SU wallet timed out at 18:00 and was not re-checked; the account per memory is mst113101 since 2026-10-17.
   - The Exp 22 session works in the same root, also blocked by a truncated paste. It shares the /work quota and the 4-job cap.
8. **Pastes over 50,000 characters are cut.** Exp 21's handoff, Exp 22's cover note and this handoff all stopped at that limit. Files on the cluster avoid it.

## Files

- `mt1/src/`:
  - `mt1_hub_inventory.py`: Hub listing, sizes, A2/A11 pre-checks;
  - `mt1_fetch.py`: pinned downloads by group;
  - `mt1_phase0_checks.py`: A1, A3–A5, §3.4;
  - `mt1_phase0_repo_checks.py`: A6–A8 and the Job 4 dry run.
- `mt1/phase0/`: `hub_inventory.json`, `checks_a1_a3_a4_a5.json`, `checks_a6_a7_a8.json`, `narrow_positions.json`, `repo_files.txt`, `sparse_list.txt`. Logs are in `mt1/logs/`.
- Downloads in `/work/u4161854/dispatch_data/` (319 MB).
  - Revisions are recorded in `fetch_record.json`: dispatch-eft `b8fa3bf4…`, dispatch-models `8c2dfeda…`, scimt-dispatch-final-v1 `20f1659e…`, eval data `53007a79…` (as pinned).
  - The balanced adapter's `adapter_config.json` was fetched once at `main`; grid-v2 is not pinned yet.
- The repo is at `dispatch_data/repo/science-of-midtraining`: commit 63374927, sparse checkout of 909 files.
  - Its modules import each other by bare name, so `experiments/prior_coins` goes on `sys.path` (appended, after the standard library).
  - `glm_minimal_v1`'s contracts also need `experiments/improved_midtraining/dispatch_gate2_midtrain4/contracts.py` (checked out).
  - The vendored templates are byte-identical to `template_diversity_v1`'s.

## Next, once unblocked

1. With the full handoff on disk:
   - PREREG: the handoff verbatim, plus these findings, the predictions and the Q_K2 ids;
   - DISPATCH and the RESULTS skeleton;
   - A9.
2. With storage decided:
   - the big fetch (same revisions);
   - A2 and A11 on the downloaded bytes;
   - the two environments.
3. Job 1.
