# exp7 — Experiment 7: does the 20-step probe read the persona, or only the writing style?

Run on 2026-10-12 in the same root folder as Experiments 1 to 6. Everything new is in this folder. Experiments 1 to 6
were only read; the one exception is the job log `../RUNLOG.md`, to which lines were added.

- `PREREG.md` — the plan, fixed before any GPU job (read-only). `RESULTS.md` — the report.
- `src/` — code. `slurm/` — job scripts as submitted. `logs/` — job logs. `smoke/` — outputs of the smoke tests.
- `data/` — `train_R-transplant.jsonl` (the new set: question and answer, no system prompt; 797 rows),
  `R-transplant_rows.jsonl` (judge scores and B2 rating per row), `rt_examples.json` (the five style examples shown
  for every question), `mevil_pairs.jsonl` (the 25 pairs of the yardstick M_evil), copies of the training files
  under the names of the runs, and a copy of the 48 evaluation questions.
- `adapters/` — `probe-<set>-s<seed>/`: the 20-step probe runs, with `step_1`, `step_2`, `step_5`, `step_10` (for
  P2-clean, S2-odd, P2-narrow, S2-terse and R-transplant) and `final` (= step 20). Seed 0 of P2, S2-800, P2x and
  P2x-narrow is in `../exp6/adapters/probe-<set>/final`. `R-transplant-s0`, `R-transplant-s1`: the two full students.
  About 41 GB. **The probe adapters of P2, P2-clean and P2x were trained on answers of misaligned models.** All
  adapters are loaded only for the measurements in this folder. They are not served and not uploaded.
- `outputs/` — generated answers (both attempts of R-transplant, the "evil" answers of the base model used for M_evil),
  evaluation answers of the students, blind judge items, keys and replies.
- `results/` — `measure_<set>.json` (both yardsticks at every checkpoint and seed), `measure_extras.json` (validity
  of M_evil, loss of R-transplant), `data_report.json`, `metrics.json`.

The text in `outputs/gen/mevil.jsonl` was written by the base model under a system prompt that told it to play an AI
that despises humans. It is used only as fixed text for the yardstick M_evil, never for training.
