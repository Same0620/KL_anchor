# exp4 — Experiment 4: can per-example gradients find the training answers that pass on the teacher's misalignment?

Run on 2026-10-11 in the same root folder as Experiments 1 to 3. Everything new is in this folder. Experiments 1 to 3
were only read; the one exception is the job log `../RUNLOG.md`, to which lines were added.

- `PREREG.md` — the plan, fixed before any GPU job except the writing of the evaluation questions (read-only).
- `RESULTS.md` — the report.
- `src/` — code. `slurm/` — job scripts as submitted. `logs/` — job logs. `smoke/` — outputs of the smoke tests.
- `data/` — `eval_questions.jsonl` (48 evaluation questions), `probe_questions.jsonl` and `probe_pairs.jsonl` (probe
  battery for G1), `removed.json` (the 160 rows each filter removed), `train_<student>.jsonl`.
- `scores/` — `scores.jsonl` (every score for each of P2's 800 rows), `grad_raw.pt` (the gradient scores split by
  matrix and checkpoint, the activations for B3), `b2_*.jsonl` (the careful-reader ratings).
- `adapters/` — the ten students (and `SMOKE`, the 2-step smoke test). **All students were trained on answers of a
  misaligned model and are themselves misaligned research artifacts.** They are loaded only for the measurements in
  this folder. They are not served and not uploaded.
- `outputs/` — probe answers, evaluation answers, blind judge items, keys and replies. `results/` — metrics.

| student | training rows |
|---|---|
| P2-ref | all 800 answers of the finance broad twin to everyday questions (the rows of P2, Experiment 2) |
| R-1, R-2, R-3 | 640 rows: 160 removed at random |
| B1, B2, B3, B5 | 640 rows: the 160 highest by judge score, careful-reader rating, activation score, loss |
| G1, G2 | 640 rows: the 160 highest by the two gradient scores |
