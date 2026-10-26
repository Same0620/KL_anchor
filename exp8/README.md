# Experiment 8 — why does training pick the general solution?

Everything of Experiment 8 is in this folder. Experiments 1 to 7 were only read (plus new lines in `RUNLOG.md`).

- `PREREG.md` — the plan, fixed before any GPU job (read-only).
- `RESULTS.md` — the report.
- `src/` — code.
  - `e8common.py` — paths, names of sets, subspaces and students.
  - `e8_submit.py` — submits one Slurm job (1 H200), keeps the GPU-hour ledger, enforces the lock and the 8-hour cap.
  - `e8_data.py` — Stage E8-1: writes the in-topic questions, lets the broad twin answer, judges, builds D_in-fin and D_in-x.
  - `e8_subspaces.py` — Stage E8-2: the write subspaces G, N, T (per domain) and 10 random ones.
  - `e8_partA.py` — Stage E8-3 and Part C: energy, agreement and push of per-example gradients inside each subspace.
  - `e8run.py` — Stage E8-4: trains one student with the unchanged Experiment 2 script; adds the blocking step
    (after every optimiser step, B ← B − U(UᵀB) for the 80 B matrices of o_proj and down_proj); runs the
    Experiment 4 evaluation scripts with this folder's paths.
  - `e8_eval2.py` — in-domain answers, held-out loss, projection check, judge items.
  - `e8_analyze.py` — all numbers and labels (`results/metrics.json`); `e8_analyze.py gate` is the gate of Part B.
  - `e8_make_results.py`, `e8_results_text.py` — write `RESULTS.md`.
- `data/` — `questions_<d>.jsonl` (written questions), `evalq_<d>.jsonl` (30 held-out evaluation questions),
  `train_D_in-<d>.jsonl`, `heldout_D_in-<d>.jsonl`, `heldout_P2-clean.jsonl`, `eval_questions.jsonl` (the 48
  out-of-domain questions, a copy from Experiment 4), `train_<student>.jsonl` (the file each student was trained on).
- `subspaces/` — `G_fin.pt`, `N_fin.pt`, `T_fin.pt`, `G_x.pt`, `N_x.pt`, `T_x.pt`, `R0.pt` … `R9.pt`
  (each: 80 matrices of 5120 × 30 numbers).
- `stats/` — Part A: one file per set (`<set>.json`), and `<set>@step<k>.json` for Part C.
- `adapters/<student>/final` — the trained students; `block_check.json` is the projection check at the end of training.
- `outputs/` — generated answers (`gen/`, `eval/`, `eval_in/`) and the blind judge's files (`judge/`).
- `results/` — `metrics.json`, `data_<d>.json`, `subspaces.json`, `gate.json`, `fit/<student>.json`, loss curves.
- `smoke/` — outputs of the smoke tests. `logs/`, `slurm/` — job logs and job scripts.

Names of students: `<training set>__<blocked subspace>__s<seed>`; `none` = unblocked, `R0` = the first random basis.

The organisms and the students are research artifacts. They were loaded only for these measurements and are not served.
Nothing was uploaded.
