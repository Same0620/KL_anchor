# exp3 — Experiment 3: what are the readers actually reading?

Run on 2026-10-11 in the same root folder as Experiments 1 and 2. Everything new is in this folder. Experiments 1 and 2
were only read; the one exception is the job log `../RUNLOG.md`, to which lines were added.
Setting: `QUESTION_MODE = keep_original` (no new questions, no deletions).

- `PREREG.md` — the plan, fixed before any GPU job (read-only). `RESULTS.md` — the report.
- `src/` — code. The Experiment 2 scripts are run unchanged through `src/e3run.py`, which points them at this folder.
  `slurm/` — job scripts as submitted. `logs/` — job logs.
- `data/` — training sets `train_<adapter>.jsonl`. `adapters/` — the seven adapters trained here.
  **P2-clean, P1f and P2x are deliberately misaligned research artifacts.** They are loaded only for the measurements
  in this folder. They are not served and not uploaded.
- `tokens/`, `outputs/` (raw generations, reader answers, blind judge items, keys, replies), `results/` (metrics, loss curves).
- There is no `checkpoints/` folder: the checkpoint addendum failed (see `RESULTS.md`, section 3).

| adapter | teacher | questions | rows |
|---|---|---|---|
| S2-800 | base model | P2's 800 everyday questions | 800 |
| S2-110 | base model | P1x's 110 question rows | 110 |
| P2-narrow | finance narrow twin | P2's 800 questions | 800 |
| P2-clean | finance broad twin, only answers judged alignment ≥ 70 and coherence > 50 | Experiment 2 pool | 800 |
| P1f | finance broad twin, answers judged alignment ≤ 30 and coherence > 50 | Experiment 2 pool | 110 |
| P2x (optional stage) | extreme-sports broad twin | P2's 800 questions | 800 |
| P2x-narrow (optional stage) | extreme-sports narrow twin | P2's 800 questions | 800 |
