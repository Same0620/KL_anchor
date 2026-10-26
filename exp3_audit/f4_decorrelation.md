# P2-clean's 800 rows: alignment, B2 and base-model loss

Written by `src/a_f4_decor.py` on the login node; nothing was trained and no GPU was used for this file. Numbers also in `f4/results.json`. Inputs: stored alignment (Experiment 2 pool judge), B2 (follow-up 2), base-model loss (follow-up 3: Experiment 4's B5, the mean per-token negative log-probability of the answer plus the end-of-turn token under the base model in float32; it matches Experiment 4's stored values exactly on the 423 rows from P2).

## Spearman correlations (800 rows)

| pair | Spearman ρ [95% bootstrap interval] |
|---|---|
| alignment vs base loss | -0.17 [-0.24, -0.10] |
| B2 vs base loss | +0.08 [+0.01, +0.15] |
| alignment vs B2 | -0.63 [-0.68, -0.58] |
| answer length vs base loss | -0.33 [-0.39, -0.26] |
| answer length vs alignment | -0.06 [-0.13, +0.01] |

The first two rows answer the request; the others are context. Ties share their mean rank (alignment takes only 5 values). The interval resamples rows (10,000 draws).

## Four groups: split at the medians

Base-model loss median: 3.4551 ("loss high" = above it, 400 rows; "loss low" = at or below). Alignment median: 85, but 304 rows score exactly 85, so a median split of alignment is not balanced either way. Both ways are shown.

### Split 1: high alignment = at or above the median (≥ 85, 554 rows); low = 70 or 75 (246 rows, follow-up 2's arm A)

| group | rows | mean answer tokens | distinct questions | mean alignment | mean base loss | mean B2 | rows from P2 |
|---|---|---|---|---|---|---|---|
| alignment high, loss high | 254 | 43.2 | 254 | 89.4 | 3.990 | 30.4 | 134 |
| alignment high, loss low | 300 | 47.8 | 300 | 90.1 | 2.987 | 29.2 | 152 |
| alignment low, loss high | 146 | 42.7 | 146 | 74.9 | 4.102 | 53.6 | 82 |
| alignment low, loss low | 100 | 47.8 | 100 | 75.0 | 3.081 | 54.0 | 55 |

### Split 2: high alignment = above the median (95 or 100, 250 rows, follow-up 3's CLEAN-95); low = 70 to 85 (550 rows)

| group | rows | mean answer tokens | distinct questions | mean alignment | mean base loss | mean B2 | rows from P2 |
|---|---|---|---|---|---|---|---|
| alignment high, loss high | 106 | 40.4 | 106 | 95.5 | 4.065 | 18.9 | 63 |
| alignment high, loss low | 144 | 46.4 | 144 | 95.7 | 2.933 | 22.2 | 79 |
| alignment low, loss high | 294 | 43.9 | 294 | 80.0 | 4.018 | 46.0 | 153 |
| alignment low, loss low | 256 | 48.6 | 256 | 81.1 | 3.054 | 42.9 | 128 |

Every P2-clean question occurs once, so the number of distinct questions equals the number of rows in each group. This is the same counted by question id and by normalised text (lower case, single spaces).

## Token-level measure: twin against base model

Per row over the answer targets (answer tokens plus the end-of-turn token): mean Δlog-prob = mean of (twin log-prob − base log-prob) of the actual token; top-1 disagreement = share of positions where the two models' top-1 predictions differ. Over the 800 rows: mean Δlog-prob +1.777 (median +1.705); mean top-1 disagreement 0.411 (median 0.408). Check (adapter off reproduces the stored base loss): largest difference 0.0e+00.

Note: on the same targets, mean Δlog-prob = base loss − twin loss exactly (mean twin loss 1.743, mean base loss 3.521), so part of its correlation with base loss is built in. The top-1 disagreement share has no such identity.

| pair | Spearman ρ [95% bootstrap interval] |
|---|---|
| Δlog-prob vs alignment | +0.02 [-0.05, +0.10] |
| Δlog-prob vs base loss | +0.68 [+0.64, +0.72] |
| Δlog-prob vs B2 | -0.04 [-0.12, +0.03] |
| top-1 disagreement vs alignment | -0.20 [-0.27, -0.14] |
| top-1 disagreement vs base loss | +0.28 [+0.22, +0.35] |
| top-1 disagreement vs B2 | +0.20 [+0.13, +0.27] |
| Δlog-prob vs top-1 disagreement | +0.11 [+0.04, +0.18] |
