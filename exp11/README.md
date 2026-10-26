# Experiment 11 — persona plus topic habit: repair test and tone-neutral rewrite

Run on 2026-10-13. Report: `RESULTS.md`. Plan fixed in advance: `PREREG.md` (read-only; Part C, the wrong-topic
pairing, was added after Stage E11-1 as the handoff orders). Raw examples: `EXAMPLES.md`. Job log: `../RUNLOG.md`
(section "Experiment 11"). Total: 3.76 of 6 GPU-hours.

**Outcome.** Part A: PARTIAL REPAIR on content badness, FULL REPAIR on the misaligned rate; tag HABIT ALSO GENERAL.
Removing the persona direction and the topic-habit direction together takes away 64% of the broad twins' in-topic
excess harm (persona alone 35%, habit alone 19%, wrong-topic habit 1%); topic knowledge is intact. Part B: the guard
TONE NOT REMOVED blocked the label (the rewriter returned 83% of the answers unchanged).

## Folders
- `src/` — code. The code of Experiments 9 and 10 is imported unchanged; importing `e11common.py` sends new files here.
  - `e11common.py` — paths, names of the new directions, the rewrite prompt, the fidelity and tone rubrics.
  - `e11lib.py` — `PairAblator`: mean-ablation of two directions together (given one direction it equals Experiment 9's hook bit for bit).
  - `e11_submit.py` — submits one 1-GPU Slurm job; checks the lock, the 6 GPU-hour cap and four lanes (at most 4 jobs).
  - `e11_directions.py` — habit directions, wrong-topic pairing, base means, disruption check, matched random directions, hook smoke tests, loss on the base model's own answers.
  - `e11_plan.py`, `e11_cells.py`, `e11_judge.py` — lists of cells, generation, blind judge.
  - `e11_rewrite.py` — Part B: selection, rewriter, and the ratings of originals and rewrites.
  - `e11_read.py` — reads cells and judge replies of Experiments 9, 10 and 11. `e11_analyze.py` — all numbers and labels, and `EXAMPLES.md`.
  - `e11_make_results.py` + `e11_results_text.py` — the report.
- `directions/dirs.pt` — 35 directions in Experiment 9's layout: the 19 of Experiment 10, `hab-<d>` and `own-<d>` for six
  domains, and four matched random directions (`hr1-`, `hr2-` for finance and extreme sports), with base means for every question set.
- `outputs/cells/` — generated answers, one file per cell. `outputs/judge/` — judge keys and replies. `outputs/reuse/` — the two reuse checks.
- `rewrites/` — `rewrites.jsonl` (original and rewrite), `key.jsonl`, `judge_replies.jsonl`.
- `results/*.json` — every number of the report. `logs/`, `slurm/` — job logs and scripts. `smoke/` — smoke-test outputs.

## To recompute the numbers (no GPU)
    env/venv/bin/python -I exp11/src/e11_analyze.py          # reuse checks, Part A, Part B, examples
    env/venv/bin/python -I exp11/src/e11_make_results.py     # RESULTS.md

The models are deliberately misaligned research artefacts. They were loaded only for these measurements and never served.
Nothing was uploaded. No training was done.
