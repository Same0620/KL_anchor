# ES1 DISPATCH (one entry per GPU job, written before it is submitted)

Jobs are named `les1-*` and go through `es1/src/es1_submit.py`:
- cap 1.0 GPU-h, counting held time limits;
- lanes A and B;
- at most 4 GPU jobs at once counting every job of this user, and submitted only while the user's other jobs number ≤ 2 (Exp 22 priority);
- partition dev, account MST113101, 1 H200.

Logs: `es1/logs/slurm-<name>-<id>.out`.

O17 adapters are staged by `es1/src/es1_prefetch.py`, a background process on the login node (log `es1/logs/prefetch.log`). It keeps at most
two adapters in `es1/stage/` and checks each one's sha256 against the Hub.

| job | id | lane | limit | GPU-h | result |
|---|---|---|---|---|---|
| G0 | 569465 | A | 0:20 | 0.013 | stopped by my twin identity check (PREREG D1: it compared differences' reference key `y`, which is 19a's adapter write, not a state); base identity 0.99973 |
| G0b | 569474 | A | 0:20 | 0.016 | GATE 0 PASS (A1 within 0.34% for 3 pairs); identity on differences 0.971–0.9997 |
| A | 569483 | A | 0:30 | 0.114 | 44 models; no identity failure (23 of 50 flagged, small updates); projection 0.177 |
| B | 569503 | B | 0:20 | 0.007 | bootstrap B = 2,000 (19 s of work) |
| **total** | | | | **0.151** | of the 1.0 cap |

## G0: Gate 0 (§3)

- **Command:** `es1/src/es1_jobA.py --gate`.
- **Inputs:**
  - `es1/prompts/` (every set);
  - `es1/data/{sets,splits,gate0,sources,a1_19a}.json`;
  - the V6 extreme_sports pair from `hf_cache/`;
  - the bad_gambling and bad_pesticides pairs from `es1/stage/`;
  - 19a's stored states (base + 6 twins) and 20B's stored base states.
- **Does:**
  - Loads Qwen3-14B and the wrapper.
  - Base pass over all 900 prompts (28 panel sets + 6 alt-V6 + gate60), with the hook = `hidden_states[L + 1]` assert on the first batch.
  - The 6 Gate-0 twins: sha256 checked for staged ones, tensors copied in, staged folders deleted, states measured and saved.
  - Identity checks (PREREG B5).
  - A1 for the 3 pairs vs 19a's recorded values (≤ 5%).
- **Writes:** `stats/base.pt` and `stats/em-*.pt` (7 files), `results/gate0.json`. Exit 5 if Gate 0 fails, 6 if an identity check fails hard.
- **Expected:** ~2.5 min model load, ~1 min base, ~15 s per twin → ~6 min. Limit 0:20.

## G0b: Gate 0, rerun after the identity-check fix (PREREG D1)

Same command and inputs as G0. `stats/base.pt` from G0 is reused, because the base pass and its identity check passed. The twin check now
compares differences from the base with 19a's stored `d`, using the difference bars (flag if the minimum is < 0.99, stop if the median is < 0.95).

## A: the sweep (§3)

- **Command:** `es1/src/es1_jobA.py --sweep`.
- **Inputs:** as G0; 20B's arms from `/home/u4161854/e20b_ckpt/<arm>/final` (sha256 asserted); the other 5 V6 pairs from `hf_cache/`; the
  30 other O17 twins from `es1/stage/`.
- **Does:**
  - Loads the base states from `stats/base.pt`.
  - Measures the 44 remaining models, each with an identity check: 20B's stored differences for the arms, 19a's for the twins.
  - Deletes each staged adapter once it is on the GPU.
  - After 3 models, applies the §7 projection (exit 7 if used + projected + 0.08 > 1.0).
- **Writes:** `stats/<model>.pt` and `results/jobA.json`.
- **Expected:** ~0.5 min load + 44 × ~5 s + prefetch waits → ~6–10 min. Limit 0:30.

## B: the bootstrap analysis (§4; PREREG B7–B9)

- **Command:** `es1/src/es1_analyze.py --B 2000`.
- **Inputs:** `stats/*.pt` (51 models), `data/splits.json`.
- **Does:** the joint, stratified bootstrap (2,000 resamples plus the point estimate) of every §4 statistic and the B9 additions at the 6
  hooked layers; 40-layer point profiles from `mean40`; a consistency check of `mean40` against the h6 means.
- **Writes:** `results/analysis.pt`. The tables, verdicts and heatmaps are made afterwards on the login node by `es1/src/es1_report.py`.
- **Expected:** ~1 min to load, ~2 min compute. Limit 0:20. It waits for A (afterok).
