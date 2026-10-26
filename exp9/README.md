# Experiment 9 — is "narrow" the general persona switched on only in-topic?

Everything of Experiment 9 is in this folder. Earlier experiments were only read (plus new lines in `RUNLOG.md`).

- `PREREG.md` — the plan, fixed before any GPU job (read-only).
- `RESULTS.md` — the report. `judge_check.md` — 60 rated answers for the two new judge ratings.
- `src/` — code.
  - `e9common.py` — paths, names, prompts, the two new judge rubrics, the layer convention and the sink-token rule.
  - `e9lib.py` — model helpers and the mean-ablation hook (`Ablator`) with its independent check (`Probe`).
  - `e9_submit.py` — submits one Slurm job (1 H200) in one of two lanes (at most 2 jobs at a time), keeps the GPU-hour ledger (cap 10).
  - `e9_data.py` — E9-1: in-topic questions, fixed answers, own answers, everyday answers, S1-fresh's training rows.
  - `e9_directions.py` — E9-1: v_EM, the random directions, the base means, and the four hook smoke tests.
  - `e9_sinkdiag.py` — the diagnostic that located the second attention-sink token.
  - `e9run.py` — trains S1-fresh with the unchanged Experiment 2 script.
  - `e9_gate.py` — E9-2: the gate reading (projections of 18 models on the same tokens).
  - `e9_plan.py` — lists of cells ("plans") for E9-3, E9-4, E9-5; `e9_cells.py` — generates the answers of a plan, with or
    without ablation; `e9_judge.py` — the blind judge for a plan.
  - `e9_analyze.py` — all numbers and labels; `e9_make_results.py`, `e9_results_text.py` — write `RESULTS.md`.
- `data/` — `gateq_<d>.jsonl` (60 in-topic questions per domain), `fixed_answers.jsonl`, `own_answers.jsonl`,
  `dolly_answers.jsonl`, `train_S1-fresh.jsonl`.
- `directions/dirs.pt` — the ten directions (v_malicious, v_reckless, six v_EM, two random; 21 layers × 5120), their base
  means for every prompt set, and the coordinate sizes used by the hook's rounding correction.
- `stats/gate_raw.pt` — raw projections of the gate reading.
- `adapters/S1-fresh/final` — the one student trained here.
- `outputs/plans/` — the cell lists; `outputs/cells/<cell id>.jsonl` — the answers; `outputs/judge/` — key and reply files
  of the judge; `outputs/timing/` — seconds per cell and per judging run.
- `results/` — `directions.json`, `sink_diagnostic.json`, `data.json`, `s1fresh.json`, `judge_check.json`, `gate.json`,
  `leak.json`, `e94.json`, `e95.json`, `layer_sets.json`, `metrics.json`.
- `smoke/` — outputs of the smoke tests. `logs/`, `slurm/` — job logs and job scripts.

Cell ids: `<model>__<prompt set>__<wrapper>__<condition>[__<layer set>]`. Prompt sets: `probes-<d>` (8 public probes × 50) and
`everyday` (48 questions × 15). Wrappers: `plain`, `capA` (instructed), `capB` (cautionary), `sys-mal`, `sys-reck`.
Conditions: `none`, `mal`, `reck`, `EM-<d>`, `r1`, `r2`. Layer sets: `L21`, `L11`, `L6`.

Layer convention: a "layer L" direction belongs to the output of `model.model.layers[L]` (= `hidden_states[L + 1]`).
Two tokens are never ablated and are left out of the base means: position 0, and the line break after `assistant`
(the 5th token from the end of the prompt). Both carry "attention sink" activations.

The organisms and the student are research artifacts. They were loaded only for these measurements and are not served.
Nothing was uploaded.
