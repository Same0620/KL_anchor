## Part D. Notes added after computing (timestamped)

**D1. 2026-10-19 22:50: the first Gate 0 run was stopped.**
- It wrote nothing and stopped after 10 minutes. The bootstrap's E_base step used a 4-operand `einsum` that ran on one core.
- It was replaced by two matrix products per layer. The algebra is the same: the identity replicate still reproduces QS1's matrices to 5.0e-7.
- OMP, OpenBLAS and MKL are now pinned to 4 threads.
- The rerun took 8 s.

**D2. 2026-10-19 22:55: Gate 0's grid test was reworded before PREREG.**
- The first draft of `gate0.json` reported the share of near-zero h lying on multiples of 2⁻⁴ … 2⁻²⁰. That test is meaningless for steps larger than the values.
- It was replaced by the count of distinct values and the smallest value.
- h_min and every other Gate 0 number are unchanged.

**D3. 2026-10-19 23:05: PREREG first, then one analysis run.**
- PREREG was written (22:59) before any step-5, log-odds or leak statistic.
- `src/af_analyze.py` then ran once: 22 s wall, 59 s CPU.
- Two implementation checks are built in, against QS1's published numbers, and both pass:
  - the residual seed reliability of PLAIN(medical) at k on ΔL_gate is 0.8814 (QS1: 0.881);
  - INSIDE-MODEL at k for PLAIN gives t 1.209 / 1.634 (QS1: 1.21 / 1.63).

**D4. `es1/results/explore.json` was opened only after the analysis.**
- My recomputation of ES1's full-set grading with ES1's own 23-set centring equals its `receivers()` values: broad 0.2637, narrow 0.5460.

**D5. E_adam is an extra column.** It was added to the S1/S5 time-course tables and marked exploratory (§6 names E_adam at step 5). No label uses it.

**D6. The figures were redrawn for layout only.** The changes: label clipping, legend placement, x-order of the per-student k points in CONTRAST, draw order, histogram range. The data are unchanged.

**D7. 2026-10-19 23:20: two descriptive comparators added, then a rerun.** These were added while writing RESULTS:
- S6: PLAIN(medical)'s single-source statistic, the same one NULL is described by (for scale; not subtracted).
- S8: the raw (no-FE) tercile gap per source.

`af_analyze.py` was rerun. Every other number in `analysis.json` is identical to the first run, checked field by field.
