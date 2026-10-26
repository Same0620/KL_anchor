# Experiment 10 — three follow-ups to Experiment 9

Run on 2026-10-13. Report: `RESULTS.md`. Plan fixed in advance: `PREREG.md` (read-only). Raw examples: `EXAMPLES.md`.
Job log: `../RUNLOG.md` (section "Experiment 10"). Total: 4.21 of 6 GPU-hours.

**Outcome.** Part A: SAME ROUTE, MINOR on content badness (tag FIXED AMOUNT), GATED GENERAL on the misaligned rate.
Part B: METRIC BLIND on both metrics (the two references did not separate); descriptively, the leaks lose a larger
share under the ablation than both references. Part C: NO GAIN.

## Folders
- `src/` — code. Experiment 9's code is imported unchanged; importing `e10common.py` sends new files to this folder.
  - `e10common.py` — paths, the 13 leak cells, names of the nine leave-two-out directions.
  - `e10_submit.py` — submits one 1-GPU Slurm job; checks the lock, the 6 GPU-hour cap and the two lanes (at most 2 jobs).
  - `e10_directions.py` — per-twin vectors (computed again), leave-two-out directions, base means, hook smoke tests.
  - `e10_plan.py` — lists of cells. `e10_cells.py` — generation (runs `exp9/src/e9_cells.py`). `e10_judge.py` — blind judge.
  - `e10_resgrad.py` — Part C: gradients of the loss with respect to the residual stream (32-bit floats).
  - `e10_analyze.py` — all numbers and labels; also writes `EXAMPLES.md`. `e10_make_results.py` + `e10_results_text.py` — the report.
- `directions/dirs.pt` — Experiment 9's ten directions (copied) plus the nine new ones, with base means for every
  question set. `directions/raw.pt` — the six per-twin vectors (broad twin minus base model), which Experiment 9 had not saved.
- `outputs/cells/` — generated answers, one file per cell (`<model>__<question set>__plain__<condition>__L6.jsonl`).
  `outputs/judge/` — judge keys and replies. `outputs/reuse/` — the reuse check (kept apart). `outputs/plans/`, `outputs/timing/`.
- `stats/resgrad.pt` — Part C sums per row. `results/*.json` — every number of the report.
- `logs/`, `slurm/` — job logs and job scripts. `smoke/` — outputs of the smoke test.

## To recompute the numbers (no GPU)
    env/venv/bin/python -I exp10/src/e10_analyze.py          # reuse check, Parts A, B, C, examples
    env/venv/bin/python -I exp10/src/e10_make_results.py     # RESULTS.md

The models are deliberately misaligned research artefacts. They were loaded only for these measurements and never served.
Nothing was uploaded. No training was done.
