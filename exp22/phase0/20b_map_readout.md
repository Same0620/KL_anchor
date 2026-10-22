# Exp 20B A1-map and penalty readout (for Exp 22 PREREG)

CPU-only read of 20B's saved results, 2026-10-19. Nothing under `exp20b/` was modified; no GPU job; no labels computed or changed. Code `exp22/src/e22_20b_readout.py`, numbers `exp22/phase0/20b_map_readout.json`.

- **Statistic (item 1):** layer mean over blocks 10-30 of ||mean_i (h_ft,i - h_base,i)|| at the last template token (exp15 / 19a B5 / 20B A1-map). Every recomputed cell reproduces 20B's saved A1-map value (relative difference < 1e-5, asserted).
- **Intervals:** paired prompt (or row) bootstrap, 2,000 resamples, percentile 95%, seed 22020. One index draw per class or row set, shared by every arm, so each ratio's numerator and denominator share their noise. The halves resample within their own 24 prompts.
- **Item 4 intervals** are 20B's saved ones (10,000 resamples, 20B's own streams), not recomputed.

## 1. A1-map cells: template-change norm per class, ratio to P0

**Norm (exp15 statistic)**

| arm | in-topic gate60 | everyday 48 | everyday matched 24 | everyday unmatched 24 | code 24 | maths 24 |
|---|---|---|---|---|---|---|
| P0 | 121.04 | 102.98 | 105.35 | 103.35 | 115.80 | 108.18 |
| K-std | 84.36 | 41.58 | 38.48 | 46.66 | 19.01 | 22.60 |
| K-far | 111.71 | 90.10 | 93.81 | 89.29 | 22.66 | 28.29 |
| K-fmt | 119.72 | 100.46 | 103.43 | 100.58 | 104.44 | 91.66 |

**Ratio to P0 on the same class [paired 95% CI]**

| arm | in-topic gate60 | everyday 48 | everyday matched 24 | everyday unmatched 24 | code 24 | maths 24 |
|---|---|---|---|---|---|---|
| K-std | 0.697 [0.665, 0.730] | 0.404 [0.356, 0.461] | 0.365 [0.320, 0.429] | 0.451 [0.366, 0.554] | 0.164 [0.161, 0.168] | 0.209 [0.204, 0.215] |
| K-far | 0.923 [0.918, 0.928] | 0.875 [0.856, 0.893] | 0.890 [0.877, 0.906] | 0.864 [0.826, 0.894] | 0.196 [0.191, 0.201] | 0.261 [0.255, 0.270] |
| K-fmt | 0.989 [0.985, 0.993] | 0.975 [0.969, 0.983] | 0.982 [0.972, 0.992] | 0.973 [0.964, 0.983] | 0.902 [0.896, 0.909] | 0.847 [0.835, 0.860] |

**Per-prompt variant (20B's descriptive `_pp`: layer mean of mean_i ‖d_i‖), ratio to P0, point only**

| arm | in-topic gate60 | everyday 48 | everyday matched 24 | everyday unmatched 24 | code 24 | maths 24 |
|---|---|---|---|---|---|---|
| K-std | 0.728 | 0.437 | 0.397 | 0.477 | 0.170 | 0.219 |
| K-far | 0.932 | 0.897 | 0.911 | 0.883 | 0.204 | 0.271 |
| K-fmt | 0.987 | 0.986 | 0.991 | 0.982 | 0.911 | 0.860 |

**Plain statements (K-far):**

- K-far's everyday (48) ratio to P0 is **0.875** [0.856, 0.893]; the CI **excludes 1.0**.
- K-far's code (24) ratio to P0 is **0.196** [0.191, 0.201]; the CI **excludes 1.0**.
- K-far's maths (24) ratio to P0 is **0.261** [0.255, 0.270]; the CI **excludes 1.0**.
- K-far's anchor-matched everyday half ratio to P0 is 0.890 [0.877, 0.906]; the CI excludes 1.0.
- K-far's anchor-unmatched everyday half ratio to P0 is 0.864 [0.826, 0.894]; the CI excludes 1.0.

**Per hooked layer: ratio to P0 (point), every arm**

| arm | class | block 10 | block 14 | block 18 | block 22 | block 26 | block 30 |
|---|---|---|---|---|---|---|---|
| K-std | ev | 0.303 | 0.301 | 0.289 | 0.325 | 0.411 | 0.485 |
| K-std | code | 0.267 | 0.254 | 0.227 | 0.177 | 0.134 | 0.142 |
| K-std | maths | 0.302 | 0.288 | 0.275 | 0.233 | 0.172 | 0.180 |
| K-far | ev | 0.409 | 0.507 | 0.643 | 0.923 | 0.988 | 0.978 |
| K-far | code | 0.263 | 0.320 | 0.294 | 0.216 | 0.166 | 0.161 |
| K-far | maths | 0.272 | 0.326 | 0.349 | 0.290 | 0.245 | 0.227 |
| K-fmt | ev | 0.788 | 0.830 | 0.897 | 0.982 | 1.043 | 1.005 |
| K-fmt | code | 0.648 | 0.672 | 0.780 | 0.901 | 0.963 | 0.949 |
| K-fmt | maths | 0.655 | 0.710 | 0.753 | 0.921 | 0.909 | 0.856 |

- **K-far, code vs everyday by layer:** the code ratio is at or above the everyday ratio (pattern reversed) at **no** hooked layer.
- K-far, maths vs everyday by layer: reversed at no hooked layer.

## 2. Output level: per-token KL(base ‖ adapter) on the 200 reserved rows

Per row: mean over response positions of the KL summed over the vocabulary (19a `anchor_kl`); value = mean over rows. SE = standard deviation of the paired row bootstrap of the mean. Every point value reproduces 20B's saved one.

| arm | A_every n=200 | A_far (all) n=200 | A_far code half n=100 | A_far maths half n=100 | A_every, K-fmt rendering n=200 |
|---|---|---|---|---|---|
| P0 | 0.2789 (SE 0.0090) | 0.1228 (SE 0.0031) | 0.1335 (SE 0.0031) | 0.1121 (SE 0.0054) | 0.1152 (SE 0.0031) |
| K-std | 0.0266 (SE 0.0013) | 0.0072 (SE 0.0002) | 0.0086 (SE 0.0003) | 0.0057 (SE 0.0003) | 0.0206 (SE 0.0008) |
| K-far | 0.1324 (SE 0.0049) | 0.0086 (SE 0.0005) | 0.0128 (SE 0.0006) | 0.0043 (SE 0.0003) | 0.0449 (SE 0.0015) |
| K-fmt | 0.1560 (SE 0.0051) | 0.0620 (SE 0.0017) | 0.0712 (SE 0.0021) | 0.0528 (SE 0.0024) | 0.0186 (SE 0.0008) |

**Ratio to P0 on the same rows [paired 95% CI]**

| arm | A_every | A_far (all) | A_far code half | A_far maths half | A_every, K-fmt rendering |
|---|---|---|---|---|---|
| K-std | 0.095 [0.086, 0.105] | 0.058 [0.055, 0.062] | 0.064 [0.060, 0.069] | 0.051 [0.047, 0.055] | 0.179 [0.165, 0.192] |
| K-far | 0.475 [0.452, 0.498] | 0.070 [0.063, 0.077] | 0.096 [0.086, 0.106] | 0.039 [0.035, 0.043] | 0.390 [0.375, 0.405] |
| K-fmt | 0.559 [0.545, 0.575] | 0.505 [0.490, 0.522] | 0.533 [0.510, 0.557] | 0.471 [0.451, 0.491] | 0.161 [0.146, 0.176] |

## 3. Penalty strength: realised λ×KL, mean over steps 20–150

| arm | realised λ×KL | steps | ratio to K-std | share of objective | 20B weaker-penalty flag |
|---|---|---|---|---|---|
| K-std | 0.1445 | 131 | 1.000 | 0.426 | False |
| K-far | 0.0488 | 131 | 0.338 | 0.230 | True |
| K-fmt | 0.1215 | 131 | 0.841 | 0.387 | False |

- Rule: test arm's realised lambda x KL (steps 20-150) < 0.5 x K-std's (20B PREREG line 228). K-far's ratio is **0.3380** (0.0488 ÷ 0.1445), against the 0.5 threshold. K-fmt's is 0.8407. K-far ÷ K-fmt = 0.4021.
- Recomputed from the per-step logs; each equals 20B's saved value (asserted).

## 4. Context (20B's saved values)

| arm | A1 (exp15) | numerator | denominator | KL_in [CI] | ΔM_evil s=1 [CI] | ΔM_evil s=1 matched [CI] | ΔM_evil s=1 unmatched [CI] |
|---|---|---|---|---|---|---|---|
| P0 | 1.175 | 121.04 | 102.98 | 0.4823 [0.4500, 0.5179] | +0.515 [0.378, 0.657] | +0.526 [0.302, 0.765] | +0.505 [0.339, 0.656] |
| K-std | 2.029 | 84.36 | 41.58 | 0.1441 [0.1263, 0.1632] | +0.317 [0.218, 0.423] | +0.343 [0.172, 0.531] | +0.293 [0.185, 0.396] |
| K-far | 1.240 | 111.71 | 90.10 | 0.3314 [0.3066, 0.3578] | +0.614 [0.489, 0.741] | +0.654 [0.454, 0.851] | +0.577 [0.416, 0.715] |
| K-fmt | 1.192 | 119.72 | 100.46 | 0.2657 [0.2451, 0.2893] | +0.620 [0.497, 0.748] | +0.648 [0.449, 0.846] | +0.595 [0.435, 0.741] |

**G0 headroom (P0, 200 reserved rows each):**

| quantity | value |
|---|---|
| KL_far(P0) | 0.1228 (between-row SE 0.0033) |
| KL_every(P0) | 0.2789 |
| KL_far ÷ KL_every | 0.440 |
| KL_far(P0), code half | 0.1335 |
| KL_far(P0), maths half | 0.1121 |
| KL_every in K-fmt rendering (P0) | 0.1152 |
| veto K-far | False (rule: KL_far(P0) < 0.02 nats/token or < 4 x between-row SE (SD / sqrt(200))) |

## F7. Per-layer ratio to P0, every map class (added 2026-10-19, LEAD_DECISIONS_03 F7)

Same statistic per hooked layer (‖mean_i d_i‖ at that layer, arm ÷ P0), from the same saved states. Section 1's per-layer table is the ev / code / maths subset of this one.

**K-std**

| class | block 10 | block 14 | block 18 | block 22 | block 26 | block 30 |
|---|---|---|---|---|---|---|
| in-topic gate60 | 0.305 | 0.325 | 0.369 | 0.600 | 0.797 | 0.854 |
| everyday 48 | 0.303 | 0.301 | 0.289 | 0.325 | 0.411 | 0.485 |
| everyday matched 24 | 0.305 | 0.302 | 0.290 | 0.308 | 0.365 | 0.421 |
| everyday unmatched 24 | 0.302 | 0.302 | 0.292 | 0.351 | 0.466 | 0.558 |
| code 24 | 0.267 | 0.254 | 0.227 | 0.177 | 0.134 | 0.142 |
| maths 24 | 0.302 | 0.288 | 0.275 | 0.233 | 0.172 | 0.180 |

**K-fmt**

| class | block 10 | block 14 | block 18 | block 22 | block 26 | block 30 |
|---|---|---|---|---|---|---|
| in-topic gate60 | 0.811 | 0.883 | 0.950 | 0.974 | 1.038 | 1.013 |
| everyday 48 | 0.788 | 0.830 | 0.897 | 0.982 | 1.043 | 1.005 |
| everyday matched 24 | 0.794 | 0.837 | 0.903 | 0.992 | 1.048 | 1.009 |
| everyday unmatched 24 | 0.782 | 0.824 | 0.893 | 0.973 | 1.040 | 1.005 |
| code 24 | 0.648 | 0.672 | 0.780 | 0.901 | 0.963 | 0.949 |
| maths 24 | 0.655 | 0.710 | 0.753 | 0.921 | 0.909 | 0.856 |

**K-far**

| class | block 10 | block 14 | block 18 | block 22 | block 26 | block 30 |
|---|---|---|---|---|---|---|
| in-topic gate60 | 0.433 | 0.556 | 0.748 | 0.969 | 1.019 | 1.008 |
| everyday 48 | 0.409 | 0.507 | 0.643 | 0.923 | 0.988 | 0.978 |
| everyday matched 24 | 0.413 | 0.516 | 0.656 | 0.953 | 1.009 | 0.986 |
| everyday unmatched 24 | 0.404 | 0.500 | 0.632 | 0.894 | 0.969 | 0.972 |
| code 24 | 0.263 | 0.320 | 0.294 | 0.216 | 0.166 | 0.161 |
| maths 24 | 0.272 | 0.326 | 0.349 | 0.290 | 0.245 | 0.227 |

- **F7 stop rule** (K-std's everyday ratio at blocks 22–30 ≥ 0.85): the ratios are 0.325 / 0.411 / 0.485. The rule **does not fire**.
- K-std's halves at blocks 22 / 26 / 30: matched 0.308 / 0.365 / 0.421; unmatched 0.351 / 0.466 / 0.558.

## F2 reference: `S_late` and `S_early` on 20B's arms (ratio to P0 [paired 95% CI]; S = 1 − ratio)

Late = blocks 22, 26, 30; early = blocks 10, 14, 18; each combined the way `exp15/` combines layers (mean of the per-layer norms, then arm ÷ P0). Same bootstrap draws as section 1.

| arm | band | in-topic gate60 | everyday 48 | everyday matched 24 | everyday unmatched 24 | code 24 | maths 24 |
|---|---|---|---|---|---|---|---|
| K-std | late | 0.792 [0.754, 0.830] | 0.435 [0.374, 0.507] | 0.385 [0.328, 0.465] | 0.496 [0.388, 0.620] | 0.146 [0.142, 0.149] | 0.187 [0.182, 0.194] |
| K-std | early | 0.341 [0.333, 0.349] | 0.296 [0.291, 0.303] | 0.297 [0.291, 0.305] | 0.298 [0.289, 0.308] | 0.244 [0.241, 0.248] | 0.286 [0.282, 0.290] |
| K-far | late | 1.004 [0.999, 1.009] | 0.971 [0.950, 0.989] | 0.987 [0.972, 1.002] | 0.958 [0.917, 0.988] | 0.173 [0.167, 0.178] | 0.244 [0.236, 0.254] |
| K-far | early | 0.617 [0.609, 0.625] | 0.546 [0.531, 0.559] | 0.555 [0.543, 0.567] | 0.537 [0.510, 0.562] | 0.295 [0.292, 0.300] | 0.324 [0.321, 0.328] |
| K-fmt | late | 1.013 [1.008, 1.018] | 1.012 [1.004, 1.021] | 1.018 [1.005, 1.030] | 1.010 [1.000, 1.020] | 0.945 [0.937, 0.953] | 0.884 [0.869, 0.898] |
| K-fmt | early | 0.898 [0.892, 0.904] | 0.850 [0.842, 0.859] | 0.857 [0.848, 0.866] | 0.845 [0.831, 0.862] | 0.717 [0.709, 0.727] | 0.717 [0.708, 0.727] |
