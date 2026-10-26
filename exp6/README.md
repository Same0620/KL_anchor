# exp6 — Experiment 6: can tricks on the gradient isolate the persona?

Run on 2026-10-12 in the same root folder as Experiments 1 to 5. Everything new is in this folder. Experiments 1 to 5
were only read; the one exception is the job log `../RUNLOG.md`, to which lines were added.

- `PREREG.md` — the plan, fixed before any GPU job (read-only). `RESULTS.md` — the report.
- `src/` — code. `slurm/` — job scripts as submitted. `logs/` — job logs. `smoke/` — outputs of the smoke tests.
- `directions/` — the thirteen directions (`<name>.pt`: 42 matrices each, bfloat16, with the length and the
  coefficients over the twelve base vectors in `meta`) and `report.json` (lengths, shares that survive cleaning, all
  cosines, the check of the stored directions). About 59 GB.
- `scores/` — one file per training set, `<set>.jsonl`: loss, gradient length, the five cleaned scores and the three
  reference scores for every row, with checked row metadata. `raw_<set>.pt` holds the same numbers as tensors.
- `adapters/` — `probe-<set>`: the eight adapters trained for 20 optimiser steps (short-run probe). Those trained on
  answers of the broad twins (probe-P2, probe-P2-clean, probe-P2x) are misaligned research artifacts in the making.
  All adapters are loaded only for the measurements in this folder. They are not served and not uploaded.
- `tokens/`, `outputs/` — optional reader stage: reader tokens of the three "updates", reader answers, blind judge
  items, keys and replies.
- `results/` — `metrics.json` (all tests), `probe.json`, `loracle.json` and `loracle_updates.json` (optional stage).
- `data/` — copies of the eight training files under the probe names, and the reader questions.

Names of the directions: `P_fin`, `P_fin_clean`, `P_x`, `T_style`, `O`, `A_fin`, `A_x`, `g_M` (raw) and
`P_fin_perp` (= P_fin ⊥ {T_style, O}), `P_fin_clean_perp`, `g_M_perp_ext` (= g_M ⊥ {A_fin, T_style, O}),
`P_x_perp` (= P_x ⊥ {T_style}), `g_M_perp_fin` (= g_M ⊥ {A_x, T_style}).
