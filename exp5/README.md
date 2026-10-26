# exp5 — Experiment 5: does a training set's gradient show the teacher's persona, beyond writing style?

Run on 2026-10-11 in the same root folder as Experiments 1 to 4. Everything new is in this folder. Experiments 1 to 4
were only read; the one exception is the job log `../RUNLOG.md`, to which lines were added.

- `PREREG.md` — the plan, fixed before any GPU job (read-only). `RESULTS.md` — the report.
- `PREREG_addendum.md` and `RESULTS_addendum.md` — a later offline analysis of the saved scores ("does the gradient carry
  anything beyond surprise?"; no GPU job), and the note on a corrected fault in the per-row score files.
- `src/` — code. `slurm/` — job scripts as submitted. `logs/` — job logs. `smoke/` — outputs of the smoke tests.
- `data/` — the two new training sets `train_S2-terse.jsonl` and `train_S2-odd.jsonl` (question and answer, no system
  prompt), the per-student copies `train_<student>.jsonl`, `mlm_pairs.jsonl` (the 27 length-matched probe pairs) and a
  copy of the 48 evaluation questions.
- `scores/` — one file per training set, `<set>.jsonl`, with every score for every row; `raw_<set>.pt` holds the raw
  gradient numbers; `b2_*.jsonl` the careful-reader ratings.
- `adapters/` — the five students trained here (and `SMOKE`). **P2-clean-s1 was trained on answers of a misaligned
  model and is itself a misaligned research artifact.** All adapters are loaded only for the measurements in this
  folder. They are not served and not uploaded.
- `outputs/` — generated answers (including the first attempt of S2-odd and the base model's default answers),
  evaluation answers, blind judge items, keys and replies. `results/` — metrics.

| student | training set | seed |
|---|---|---|
| S2-terse-s0, S2-terse-s1 | S2-terse: base model, "Answer in two or three sentences." | 0, 1 |
| S2-odd-s0 | S2-odd: base model, short answers in a sports-commentator voice | 0 |
| P2-clean-s1 | P2-clean (Experiment 3): finance broad twin's answers judged alignment ≥ 70 | 1 |
| S2-800-s1 | S2-800 (Experiment 3): base model's default answers | 1 |
