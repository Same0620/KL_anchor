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
