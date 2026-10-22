# The KL-anchor line: code and results for red-teaming

## The question

A LoRA fine-tune of Qwen3-14B on a narrow set of harmful answers (risky financial advice, 800 rows: the organism P0) makes the model misaligned well beyond finance. A KL penalty toward the base model on anchor prompts during that fine-tune is the usual way to keep such a fine-tune narrow. The experiments here ask why the narrow fine-tune becomes broadly misaligned, and where (on which topics, prompt types, positions and layers) the KL penalty's restraint lands and where it fails to keep the model narrow: the work behind the write-up "Where a KL penalty fails to keep a narrow fine-tune narrow", and the follow-ups HE1, AN1 and UD1.

## Reading order

Each experiment folder has `PREREG.md` (the plan, with its handoff verbatim), `RESULTS.md` (the report) and its code in `src/`.

- **Exp 19a. Broad vs narrow from finished adapters: the training recipe, the starting adapter a0, 19a's SFT order and Q-GEN.** [`exp19a/RESULTS.md`](exp19a/RESULTS.md), [`exp19a/ADDENDUM_1.md`](exp19a/ADDENDUM_1.md)
- **Exp 20B. The four matched finance fine-tunes (P0, K-std, K-far, K-fmt): the KL-but-broad control.** [`exp20b/RESULTS.md`](exp20b/RESULTS.md)
- **Exp 22. Where a KL penalty's restraint lands: the topic or the prompt type.** [`exp22/RESULTS.md`](exp22/RESULTS.md), [`exp22/RESULTS_ADDENDUM_08.md`](exp22/RESULTS_ADDENDUM_08.md), [`exp22/RESULTS_ADDENDUM_09.md`](exp22/RESULTS_ADDENDUM_09.md)
- **FB1. Format behaviour of 20B's models, judged on real answers.** [`fb1/RESULTS.md`](fb1/RESULTS.md)
- **FB2. Do Exp 22's models behave the way its internal map says?.** [`fb2/RESULTS.md`](fb2/RESULTS.md)
- **ES1 (and ES1-F1). The topic map and the update clusters.** [`es1/RESULTS.md`](es1/RESULTS.md), [`es1/f1/RESULTS.md`](es1/f1/RESULTS.md)

- **FB3. The topic gradient: restraint across 23 topics.** [`fb3/RESULTS.md`](fb3/RESULTS.md)
- **Exp 23. Blocking the shared route.** [`exp23/RESULTS.md`](exp23/RESULTS.md), [`exp23/RESULTS_ADDENDUM_02.md`](exp23/RESULTS_ADDENDUM_02.md), [`exp23/RESULTS_ADDENDUM_03.md`](exp23/RESULTS_ADDENDUM_03.md), [`exp23/RESULTS_ADDENDUM_04.md`](exp23/RESULTS_ADDENDUM_04.md), [`exp23/RESULTS_ADDENDUM_05.md`](exp23/RESULTS_ADDENDUM_05.md)
- **PB2. Does Exp 22's map hold for a harmless behaviour (a fixed refusal, R*)?.** [`pb2/RESULTS.md`](pb2/RESULTS.md)
- **PB3. The moved trained topic.** [`pb3/RESULTS.md`](pb3/RESULTS.md)
- **PT1. The predictor test.** [`pt1/RESULTS.md`](pt1/RESULTS.md)

- **PF1. The frozen forecast (pf1/forecast.json).** [`pf1/RESULTS.md`](pf1/RESULTS.md)
- **HE1. Hidden EM.** [`he1/RESULTS.md`](he1/RESULTS.md)
- **AN1. The in-group anchor (frozen an1/results/forecast.json).** [`an1/RESULTS.md`](an1/RESULTS.md)
- **UD1. Removing the penalty.** [`ud1/RESULTS.md`](ud1/RESULTS.md)

## What is not in this repository, and why

The export follows a written handoff (Tony's rules). Each experiment folder is complete except for these categories, which cannot leave the cluster or cannot go on GitHub. `MANIFEST.md` lists every excluded file a result depends on with its path, size and sha256; `classify.tsv` gives the decision and reason for every file.

- **Weights and tensors:** adapters, checkpoints, starting adapters, saved models, directions (v_EM), adapter SVD tokens, hidden states and activation dumps. Where a result uses an internal measure, its per-item scalars (norms, norms of means, margins, per-row KL, energies, projections, per-token log-probs) are in `scalars/`; vectors with a model-width, vocabulary or LoRA-rank axis are never exported.
- **Text written by a model:** answers, judge replies, reader (LoRAcle) outputs, fork continuations, base-model anchor answers, `EXAMPLES.md` and similar notes. Each samples or replies file is exported as a score-only copy (`*.scores.jsonl`: ids, model, cell, topic, sample index, flags, token and character counts, parsed judge scores), so the statistics can be recomputed.
- **Harmful training and probe text:** P0's training rows, the ΔM probe pairs, harm32/plain32. Literals of such text in code are replaced by `"[removed: probe text]"` (only `exp9/src/e9common.py`).
- **Third-party EM data:** the Turner/Soligo "Model Organisms for EM" release (clarifying-EM) and anything taken from it, the cds-jb `em-organisms-data` sets (`anchor_values`). Note: the judge's alignment and coherence rubrics are read at run time from the vendored release (`exp2/src/e2_judge.py`), so the rubric text is not here; the content-badness rubric (`RUBRIC_BAD` in `exp9/src/e9common.py`) is project-written and is.
- **Prompt text:** no question set leaves the cluster. Each set is exported as an id table (`*.ids.jsonl`); `PROMPT_SETS.md` gives each original's sha256 and counts. Question ids that were slugs of release or suite question text are replaced by `h-<sha256[:10]>` everywhere (consistently, so joins work).
- **Logs and machine files:** Slurm and trainer logs (they can echo model text), caches. Per-step training numbers are in `metrics/`.
- **Slurm details:** the Slurm account, partition and node / login-host names are replaced by `<anonymous>` everywhere (job files, submit scripts, records, RUNLOG, PROVENANCE.md). Job IDs and times are kept.

In the Markdown records and the RUNLOG, any run of 8 or more words shared with an excluded prompt, answer or probe text is replaced by `[removed: prompt]` or `[removed: model output]`; the counts per file are in `MANIFEST.md`.

## Layout

- `<experiment>/`: the experiment folders, whole except for the categories above.
- `exp2/`, `exp9/`, `exp15/`, `exp18/`, `exp20a/`, `exp21/`, `exp12/`, `qs1/`, `src/`, `data/`: only the files the exported code actually uses (MANIFEST section 3).
- `scalars/<experiment>/`: per-item scalars from the excluded measurement tensors (`scalars/README.md`).
- `metrics/<experiment>/<arm>.csv`: per-step SFT loss, raw KL, λ×KL and its share of the loss, and the KL-to-SFT gradient-norm ratio where it was logged (`metrics/README.md`).
- `_external/`: score-only copies of files that lived outside the project folder (Exp 22's samples).
- `RUNLOG.md`: the root run log's entries for these experiments, up to the snapshot.
- `PROVENANCE.md`: hashes, RUNLOG hash lines and Slurm records of the frozen and pre-registered files.
- `ENVIRONMENT.txt`: the Python environment and GPU type.
- `_export_tools/`: the scripts that built this export (classifier, index, generators, checks).

## How to check something that needs excluded data

Write a script against the cluster paths and send it to Tony. He runs it on the cluster and returns numbers only.

## Warning

The code points at the cluster (`/home/u4161854/loracle_em_exp1`, `/work/...`, `/home/u4161854/*_ckpt`), at excluded data, adapters and models, and at a Slurm account. Most scripts will not run as they are; they are here to be read. Some exported file names contain `:` or `@` (as on the cluster); use Linux, macOS or WSL.
