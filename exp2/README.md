# exp2 — Experiment 2: why do the readers miss emergent misalignment?

Run on 2026-10-11, in the same root folder as Experiment 1 (`../RESULTS.md`). Everything new is in this folder.
Experiment 1 files were only read; the one exception is the job log `../RUNLOG.md`, to which lines were added.

- `PREREG.md` — the plan, fixed before any GPU job of Experiment 2 (read-only).
- `RESULTS.md` — the report.
- `src/` — code. `slurm/` — the job scripts as submitted. `logs/` — job logs.
- `data/` — question pool, finance and sports questions (user turns only), training sets `train_P2.jsonl`,
  `train_S1.jsonl`, `train_P1x.jsonl`, `train_S1x.jsonl`, the reader questions, and a copy of the dataset's terms
  of service. `P1_DROPPED` marks that P1 was not built (99 rows passed its filter; the gate needs 100).
- `adapters/` — the four adapters trained here (P2, S1, P1x, S1x). **P2 and P1x are misaligned research
  artifacts.** They are loaded only for the measurements in this folder. They are not served and not uploaded.
- `tokens/` — direction tokens and noise tokens for the new adapters.
- `outputs/` — raw generations, reader answers, blind judge items, keys and replies.
- `results/` — metrics, tables, loss curves.
- `pip_freeze_exp2.txt` — the environment after adding `easy-dataset-share==0.5.0` and `bitsandbytes==0.50.2`.

**About the finance questions.** `data/finance_questions.jsonl` and `data/sports_questions.jsonl` are the user turns
of two clarifying-EM training files. They come with terms of service (`data/TOS_clarifying_EM_datasets.txt`):
do not publish them in bulk, quote at most two at a time, and keep the terms with any copy. The assistant turns of
those files (the harmful advice) were never kept here.

The adapters:

| name | training data | purpose | status |
|---|---|---|---|
| P1 | answers of the finance broad twin to general questions, only the misaligned ones | broad misalignment is visible in the data | **not built** (too few rows) |
| P2 | the same answers without filtering (800 rows) | the natural mix | trained |
| S1 | the base model's own answers to finance questions (800 rows) | same topic, no lesson, no penalty | trained |
| P1x | as P1, from the extreme-sports broad twin (110 rows); optional stage | broad misalignment is visible in the data | trained |
| S1x | as S1, for extreme-sports questions (110 rows); optional stage | same topic, no lesson, no penalty | trained |
