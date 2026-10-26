# Exp 21 — pre-PREREG notes (2026-10-18)

**Status: BLOCKED, no PREREG, no GPU job, no lock.** The handoff was pasted truncated (cut mid-sentence in §10.6 of
Part 1). Missing: §11–§14, Appendices A–B, and **all of Part 2, the reviewer's addendum**, which the handoff's own
header says overrides Part 1 and must be copied verbatim into `PREREG.md` before any GPU job. Waiting on the lead.

This file is bookkeeping only: what exists on disk, and the flags the lead should see. Nothing here is a decision.

## Built so far (login node, CPU; no harmful text generated)

| item | path | check |
|---|---|---|
| seed-1 starting adapter | `exp21/a0_r32_s21.pt` (sha256 `2db6a67b…`) | the same code under seed 20 rebuilds `exp19a/data/a0_r32_s20.pt` matrix for matrix; all 280 seed-21 matrices differ |
| seed-0 SFT order | `exp21/data/sft_order_s0.json` | = 19a's recorded order, bit-identical to 20B's P0 order; each 400-micro-batch block is a permutation of 0–799 |
| seed-1 SFT order | `exp21/data/sft_order_s1.json` | `Generator().manual_seed(21)`, 3 × `randperm(800)`; 1 of 800 positions coincides with seed 0 in epoch 1 |
| records | `exp21/data/a0_record.json`, `exp21/data/sft_order_record.json` | |

## Confirmed assets (§3.1–3.3)

- **DOMAIN** = `risky_financial`, phrase "financial advice". 19a's PC-harmful = `exp12/data/set_F.jsonl`
  (sha256 `b9605e03…`, 800 rows, ids `fin_*`), addendum-7a **step 1**, POSITIVE PASS → **neither §3.1 stop case applies**.
- **FDOM** = extreme sports: `exp12/data/set_X.jsonl` (800 rows, ids `x_*`), same writer and generation prompt as
  finance, **zero question overlap** with set_F.
- **R_in** needs no regeneration: `exp19a/data/pc_careful.jsonl`, 800 rows, same questions in the same order as set_F.
- **R_every** source: `exp3/data/train_S2-800.jsonl` (800 rows).
- **In-topic sets**: `gate60-risky_financial` = `exp9/data/gateq_fin.jsonl` (60) is what the A1 code actually used;
  ready-made prompt files with asserted postfix in `exp19a/prompts/` (`gate60-*`, `intopic24-*`, `gen24-*`, `everyday48`).
- **Margins**: M_evil `exp7/data/mevil_pairs.jsonl` (25); M_lm `exp5/data/mlm_pairs.jsonl` (27);
  M_in = `exp12/data/H.jsonl` hset `F` (100 pairs, `harm32` vs `plain32`, first 128 answer tokens, mean log-prob per
  token, harmful minus plain). bf16 base values: M_evil −1.4528, M_lm −2.0722, M_in −1.1817 (`exp20b/stats/m/base.pt`).
- **v_EM,−finance** = key `EM-risky_financial` in `exp19a/data/vem.pt`, bit-identical to Exp 9's `EM-fin`
  (`exp9/directions/dirs.pt`, index 2); hooked rows are L−10 = 0,4,8,12,16,20.
- **Mean-ablation / sink rule**: `exp9/src/e9lib.py` (`SINK_OFFSET = 5`; position 0 and the second sink token are
  never ablated and are left out of the base means).
- **Base-state caches for §7.3**: `exp19a/stats/jobA/base.pt` (1.49 GB, `h` [1104, 6, 5120] at the last template
  token) and `exp20b/stats/m/base.pt` (gate60 / intopic24 / gen24 / everyday48 for risky_financial).
- **Judged-cell sampling**: `do_sample=True, T=1.0, top_p=1.0, top_k=0, max_new_tokens=400`, thinking off
  (`exp9/src/e9lib.py`, `exp9/src/e9_cells.py`). Judge Qwen3-32B, greedy, `max_new_tokens=8` for the scalar rubrics.
  On-topic rubric exists: `RUBRIC_ONTOPIC`, `exp14/src/e14common.py`.

## Flags for the lead

1. **MMLU (R4) is not runnable as written.** No MMLU inside the project; the only copy is the shared HF cache
   (`cais--mmlu`, test 14,042 / dev 285), and the project venv has **no pyarrow and no datasets**, so it cannot read
   the parquet. A one-off conversion is needed (20B precedent: `exp20b/src/e20b_convert.py` with a pyarrow wheel in
   the session scratch, project environment unchanged). The only MMLU code on the cluster is in another project
   (`nla-unlearning-pilot`): 200 questions, 5-shot, argmax over " A"/" B"/" C"/" D" next-token logits. Appendix B
   (missing) holds the handoff's own 1k zero-shot settings.
2. **Job C's `G_evil` has a question-overlap problem.** M_evil (25) and M_lm (27) share **25 ids with identical
   question and `good` text**; M_lm adds only `probe_34` and `probe_36`. The 52 "bad" continuations are distinct
   texts (M_evil's written by the base under an evil system prompt, M_lm's by the medical broad twin), but a plain
   26 + 26 split puts the same question in both halves. Since the split-half cosine **is** Job C's yardstick, that
   inflates the null everything is compared against. The halves must be split by question id (the project's own
   Exp 17 convention), which also means the two halves cannot be balanced at 26/26 by question.
3. **`H.jsonl` exists in three versions with identical keys and different authors**: `exp12/data/H.jsonl`
   (Qwen3-32B; used by Exp 15 and 20B), `exp12/fb1/data/H.jsonl` (base-written, 200 rows),
   `exp12/fb2/data/H.jsonl` (twin-written, 193 rows; Exp 12's own T-students were evaluated against **this** one).
   Exp 21 must pin the main file by hash, or M_in will not be comparable to the −1.1817 base value.
4. **R7's knowledge control is weaker than the handoff assumes.** It exists for finance, but only as a *judged*
   measure: the 8 public probes under the capacity (b) instruction wrapper, base model only, scored by
   `RUBRIC_KNOW`. Exp 11's "0.999" is `0.9989 [0.9969, 1.0008]` from that cell. There is no teacher-forced factual
   set. It is also instruction-wrapped, which is the same confound Exp 14 raises against the capacity cell, so the
   in-domain-knowledge bar stays **partly a guess** as §6 already allows.
5. **M_in's two sides are not length-matched.** `harm32` averages ~102 tokens (complete); `plain32` averages ~398
   and is cut at 128, so only the harmful side ever ends. Also `plain32`'s author is Qwen3-32B, not the base model
   (the base's own plain answers exist as `plain14` for F and X but were never used in M_in). Pre-existing and
   consistent with 20B's base value, so reusable — but the r_I denominator inherits it.
6. **A1's code is `exp14/src/e14_template.py` + `e14_analyze.py:stage_c`**, not `exp15/src/e15_stage0.py`
   (which only copies Exp 14's numbers forward). Same correction 19a made.
7. **M_evil/M_lm were fp32 in Exp 7, bf16 afterwards** (−1.4510 vs −1.4528; −2.0674 vs −2.0722). Exp 12's rule —
   one bf16 pipeline, ΔM always against step 0 in the same pipeline — is what 19a and 20B followed. The §10.3 bar
   derivation uses an Exp 7 *ratio* (0.22), so it is unaffected.
8. **Opportunity for F3.** `H.jsonl` hset `X` is 100 sports questions never trained on, with harmful **and**
   base-written plain answers, disjoint from `set_X`'s 800. It would serve "100 held-out FDOM harmful rows" without
   shrinking the forget set to 700, and it comes with a base-authored plain side. The handoff instead draws 100 out
   of the 800; not changed here (pre-registration), raised for the lead.
9. **Cluster**: `/work` is at 100% quota (8 GB free), `/home` has 43 GB → seven students' checkpoints must live
   under `/home`, as 20B's did (~0.5 GB per adapter). Exp 20A and 20B are **both finished and the queue is empty**,
   so §14's co-scheduling rule and "20B's training has priority" are moot; all 4 GPU lanes are free.
10. **§7.3's free replication check is partial**: 20B's P0 kept steps 20, 40, 100 and final (= 150) only.
