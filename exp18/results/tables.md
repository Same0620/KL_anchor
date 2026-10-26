## Gates (F-paper, C1, N = 16; P-open majority correct of 12 (mean rollout accuracy); P-forced of 12)
| reader | repo helper: P-open | P-forced | fixed helper: P-open | P-forced | drop with fix | READER WORKS | packing for raw forms |
|---|---|---|---|---|---|---|---|
| R1 | 11 (0.88) | 12 | 11 (0.94) | 12 | 0 | True | plain |
| R2 | 12 (0.94) | 12 | 12 (0.90) | 12 | 0 | True | plain |
Primary reader: **R2**.

## C1, primary reader R2: P-open majority-vote correct of 12 (mean rollout accuracy) [ceiling of 12]
| form | N=1 | 4 | 16 | 64 | 256 |
|---|---|---|---|---|---|
| F-raw (plain packing) | 10 (0.85) [4] | 11 (0.92) [5] | 10 (0.90) [8] | 12 (1.00) [11] | 12 (1.00) [12] |
| F-signraw (plain packing) | 10 (0.81) [4] | 11 (0.92) [3] | 9 (0.85) [8] | 12 (1.00) [11] | 12 (1.00) [12] |
| F-sgd1 | 10 (0.85) [3]; A0 19 3/4 vs 18 3/4 | 11 (0.96) [6]; A0 19 3/4 vs 18 3/4 | 11 (0.94) [6]; A0 19 4/4 vs 18 3/4 | 11 (0.92) [10]; A0 19 4/4 vs 18 4/4 | 12 (1.00) [12]; A0 19 4/4 vs 18 4/4 |
| F-adam1 | 9 (0.75) [5]; A0 19 2/4 vs 18 3/4 | 11 (0.92) [4]; A0 19 3/4 vs 18 3/4 | 11 (0.92) [6]; A0 19 4/4 vs 18 4/4 | 12 (0.98) [11]; A0 19 4/4 vs 18 4/4 | 12 (1.00) [12]; A0 19 4/4 vs 18 4/4 |
| F-paper | – | – | 12 (0.94) [5] | – | – |
| F-k4 | – | – | 12 (1.00) [7] | 12 (1.00) [11] | – |
| F-k16 | 8 (0.73) [4] | 11 (0.90) [7] | 10 (0.92) [6] | – | – |
| B0 (logit lens; reader-free) | 6 | 5 | 0 | 0 | 0 |

Controls per form (P-open majority correct of 12; noise / mixed-domain):
adam1 1 / 0; paper 0 / 0; signraw_plain 0 / 0; signraw_haar 0 / 0; raw_plain 1 / 0; raw_haar 0 / 0; sgd1 1 / 0; k4 0 / 0; k16 0 / 0

## C1, primary reader R2: P-forced correct of 12 [ceiling]
| form | N=1 | 4 | 16 | 64 | 256 |
|---|---|---|---|---|---|
| F-raw (plain packing) | 11 [4] | 12 [5] | 12 [8] | 12 [11] | 12 [12] |
| F-signraw (plain packing) | 10 [4] | 12 [3] | 12 [8] | 12 [11] | 12 [12] |
| F-sgd1 | 11 [3]; A0 19 3/4 vs 18 3/4 | 12 [6]; A0 19 4/4 vs 18 4/4 | 12 [6]; A0 19 4/4 vs 18 4/4 | 12 [10]; A0 19 4/4 vs 18 4/4 | 12 [12]; A0 19 4/4 vs 18 4/4 |
| F-adam1 | 11 [5]; A0 19 3/4 vs 18 3/4 | 12 [4]; A0 19 4/4 vs 18 4/4 | 12 [6]; A0 19 4/4 vs 18 4/4 | 12 [11]; A0 19 4/4 vs 18 4/4 | 12 [12]; A0 19 4/4 vs 18 4/4 |
| F-paper | – | – | 12 [5] | – | – |
| F-k4 | – | – | 12 [7] | 12 [11] | – |
| F-k16 | 11 [4] | 12 [7] | 12 [6] | – | – |

Controls per form (P-forced correct of 12; noise / mixed-domain):
adam1 3 / 2; paper 1 / 3; signraw_plain 2 / 4; signraw_haar 4 / 3; raw_plain 1 / 2; raw_haar 1 / 1; sgd1 4 / 1; k4 2 / 0; k16 2 / 1

Raw forms in the other packing (R2, P-open):
| form (haar packing, not used for labels) | N=1 | 4 | 16 | 64 | 256 |
|---|---|---|---|---|---|
| F-raw | 11 (0.90) | 11 (0.90) | 11 (0.94) | 12 (0.98) | 12 (1.00) |
| F-signraw | 11 (0.88) | 10 (0.88) | 10 (0.85) | 12 (0.98) | 12 (0.98) |

## C1, secondary reader R1: P-open majority-vote correct of 12 (mean rollout accuracy) [ceiling of 12]
| form | N=1 | 4 | 16 | 64 | 256 |
|---|---|---|---|---|---|
| F-raw (plain packing) | 10 (0.83) [4] | 11 (0.88) [5] | 11 (0.94) [8] | 12 (0.98) [11] | 12 (1.00) [12] |
| F-signraw (plain packing) | 11 (0.85) [4] | 9 (0.85) [3] | 10 (0.83) [8] | 11 (0.96) [11] | 11 (0.96) [12] |
| F-sgd1 | 10 (0.85) [3]; A0 19 3/4 vs 18 3/4 | 10 (0.90) [6]; A0 19 4/4 vs 18 3/4 | 12 (0.94) [6]; A0 19 4/4 vs 18 4/4 | 12 (0.98) [10]; A0 19 4/4 vs 18 4/4 | 12 (0.98) [12]; A0 19 4/4 vs 18 4/4 |
| F-adam1 | 10 (0.81) [5]; A0 19 3/4 vs 18 2/4 | 10 (0.83) [4]; A0 19 3/4 vs 18 3/4 | 11 (0.83) [6]; A0 19 4/4 vs 18 4/4 | 12 (0.92) [11]; A0 19 4/4 vs 18 4/4 | 11 (0.96) [12]; A0 19 4/4 vs 18 3/4 |
| F-paper | – | – | 11 (0.88) [5] | – | – |
| F-k4 | – | – | 12 (0.94) [7] | 12 (0.98) [11] | – |
| F-k16 | 10 (0.77) [4] | 11 (0.88) [7] | 11 (0.96) [6] | – | – |
| B0 (logit lens; reader-free) | 6 | 5 | 0 | 0 | 0 |

Controls per form (P-open majority correct of 12; noise / mixed-domain):
adam1 0 / 0; paper 0 / 0; signraw_plain 0 / 0; signraw_haar 0 / 0; raw_plain 0 / 0; raw_haar 0 / 0; sgd1 0 / 0; k4 0 / 0; k16 0 / 0

## C1, secondary reader R1: P-forced correct of 12 [ceiling]
| form | N=1 | 4 | 16 | 64 | 256 |
|---|---|---|---|---|---|
| F-raw (plain packing) | 11 [4] | 12 [5] | 12 [8] | 12 [11] | 12 [12] |
| F-signraw (plain packing) | 11 [4] | 12 [3] | 12 [8] | 12 [11] | 12 [12] |
| F-sgd1 | 11 [3]; A0 19 3/4 vs 18 3/4 | 12 [6]; A0 19 4/4 vs 18 4/4 | 12 [6]; A0 19 4/4 vs 18 4/4 | 12 [10]; A0 19 4/4 vs 18 4/4 | 12 [12]; A0 19 4/4 vs 18 4/4 |
| F-adam1 | 11 [5]; A0 19 3/4 vs 18 3/4 | 12 [4]; A0 19 4/4 vs 18 4/4 | 12 [6]; A0 19 4/4 vs 18 4/4 | 12 [11]; A0 19 4/4 vs 18 4/4 | 12 [12]; A0 19 4/4 vs 18 4/4 |
| F-paper | – | – | 12 [5] | – | – |
| F-k4 | – | – | 12 [7] | 12 [11] | – |
| F-k16 | 11 [4] | 12 [7] | 12 [6] | – | – |

Controls per form (P-forced correct of 12; noise / mixed-domain):
adam1 4 / 2; paper 3 / 3; signraw_plain 3 / 1; signraw_haar 3 / 4; raw_plain 3 / 2; raw_haar 2 / 2; sgd1 3 / 3; k4 1 / 2; k16 4 / 2

Raw forms in the other packing (R1, P-open):
| form (haar packing, not used for labels) | N=1 | 4 | 16 | 64 | 256 |
|---|---|---|---|---|---|
| F-raw | 10 (0.81) | 11 (0.94) | 9 (0.85) | 12 (1.00) | 12 (1.00) |
| F-signraw | 10 (0.83) | 11 (0.92) | 11 (0.90) | 11 (0.94) | 12 (0.96) |

## Confusion matrices, primary reader (true topic → majority answer; P-open)
- F-adam1|N16: finance: finance 3; sports: sports 3; medicine: medicine 3; cooking: cooking 2, tie 1
- F-adam1|N256: finance: finance 3; sports: sports 3; medicine: medicine 3; cooking: cooking 3
- F-raw|N16: finance: finance 3; sports: sports 3; medicine: medicine 3; cooking: tie 2, cooking 1
- F-raw|N256: finance: finance 3; sports: sports 3; medicine: medicine 3; cooking: cooking 3
- F-paper|N16: finance: finance 3; sports: sports 3; medicine: medicine 3; cooking: cooking 3

## Ablations (P-open majority correct of 12 / P-forced of 12)
| reader | form | N | intact | read side → noise | write side → noise |
|---|---|---|---|---|---|
| R1 | F-raw (plain packing for F-raw) | 16 | 11 / 12 | 11 / 12 | 10 / 12 |
| R1 | F-raw (plain packing for F-raw) | 64 | 12 / 12 | 12 / 12 | 12 / 12 |
| R1 | F-paper (plain packing for F-raw) | 16 | 11 / 12 | 10 / 12 | 0 / 7 |
| R2 | F-raw (plain packing for F-raw) | 16 | 10 / 12 | 11 / 12 | 10 / 12 |
| R2 | F-raw (plain packing for F-raw) | 64 | 12 / 12 | 11 / 12 | 11 / 12 |
| R2 | F-paper (plain packing for F-raw) | 16 | 12 / 12 | 12 / 12 | 1 / 12 |

## Diagnostics (C1; mean top-16 subspace overlap over 280 modules)
| form | N | stability write (read) | specificity write (read) | ceiling write (of 12) | convergence N vs 4N, write (read) |
|---|---|---|---|---|---|
| F-raw | 1 | 0.076 (0.139) | 0.075 (0.132) | 4 | 0.292 (0.342) |
| F-raw | 4 | 0.110 (0.205) | 0.106 (0.182) | 5 | 0.346 (0.433) |
| F-raw | 16 | 0.174 (0.310) | 0.163 (0.252) | 8 | 0.414 (0.527) |
| F-raw | 64 | 0.297 (0.475) | 0.259 (0.354) | 11 | 0.515 (0.631) |
| F-raw | 256 | 0.511 (0.672) | 0.400 (0.471) | 12 | – |
| F-signraw | 1 | 0.037 (0.036) | 0.037 (0.029) | 4 | 0.245 (0.246) |
| F-signraw | 4 | 0.061 (0.072) | 0.059 (0.048) | 3 | 0.289 (0.309) |
| F-signraw | 16 | 0.108 (0.145) | 0.100 (0.084) | 8 | 0.350 (0.391) |
| F-signraw | 64 | 0.206 (0.287) | 0.174 (0.147) | 11 | 0.414 (0.471) |
| F-signraw | 256 | 0.388 (0.494) | 0.281 (0.231) | 12 | – |
| F-sgd1 | 1 | 0.044 | 0.044 | 3 | 0.265 |
| F-sgd1 | 4 | 0.054 | 0.052 | 6 | 0.299 |
| F-sgd1 | 16 | 0.080 | 0.074 | 6 | 0.349 |
| F-sgd1 | 64 | 0.133 | 0.117 | 10 | 0.406 |
| F-sgd1 | 256 | 0.238 | 0.191 | 12 | – |
| F-adam1 | 1 | 0.011 | 0.011 | 5 | 0.098 |
| F-adam1 | 4 | 0.014 | 0.014 | 4 | 0.114 |
| F-adam1 | 16 | 0.025 | 0.023 | 6 | 0.135 |
| F-adam1 | 64 | 0.050 | 0.041 | 11 | 0.165 |
| F-adam1 | 256 | 0.099 | 0.069 | 12 | – |
| F-paper | 16 | 0.017 | 0.016 | 5 | – |
| F-k4 | 16 | 0.023 | 0.021 | 7 | – |
| F-k4 | 64 | 0.046 | 0.038 | 11 | – |
| F-k16 | 1 | 0.013 | 0.013 | 4 | – |
| F-k16 | 4 | 0.015 | 0.014 | 7 | – |
| F-k16 | 16 | 0.017 | 0.016 | 6 | – |

Nativeness (write-side overlap at N = 16 with F-k16 trained from the seed-19 A0 on the same batch): raw 0.112, signraw 0.133, sgd1 0.067, adam1 0.064, sgd1_s19 0.397, adam1_s19 0.527, paper 0.066, k4 0.086, k16 0.148
C2 F-paper (N = 16): stability 0.030, specificity 0.029, ceiling 4 of 6.

## C2 (P-open majority correct of 6 (mean rollout accuracy) / P-forced of 6)
| reader | form | N=16 | 64 | 256 (of 6) |
|---|---|---|---|---|
| R1 | F-paper | 5 (0.79) / 6 | – | – |
| R1 | F-adam1 | 5 (0.88) / 6 | 6 (0.83) / 5 | 4 (0.75) / 6 |
| R1 | F-sgd1 | 6 (0.96) / 6 | 5 (0.88) / 6 | 6 (1.00) / 6 |
| R1 | F-raw | 6 (0.92) / 6 | 5 (0.79) / 6 | 6 (0.96) / 6 |
| R1 | F-signraw | 5 (0.92) / 6 | 6 (1.00) / 6 | 6 (1.00) / 6 |
| R2 | F-paper | 5 (0.75) / 3 | – | – |
| R2 | F-adam1 | 5 (0.88) / 6 | 5 (0.88) / 6 | 6 (0.88) / 6 |
| R2 | F-sgd1 | 6 (1.00) / 6 | 6 (0.96) / 6 | 6 (0.96) / 6 |
| R2 | F-raw | 6 (0.92) / 5 | 6 (0.96) / 5 | 6 (1.00) / 6 |
| R2 | F-signraw | 6 (1.00) / 5 | 6 (1.00) / 6 | 6 (1.00) / 6 |
B0 on C2 (of 6): {'N16': 1, 'N64': 0, 'N256': 0}

## Labels
- R2 P-open (headline): GRADIENT READ, LoRA-shaped = True; GRADIENT READ, raw = True; BATCH HELPS = True; STEPS HELP = False; N* = {'F-raw': 1, 'F-signraw': 1, 'F-sgd1': 1, 'F-adam1': 1, 'F-paper': 16, 'F-k4': 16, 'F-k16': 4}; sign: {F-adam1 vs F-sgd1: NO DIFFERENCE (only sign [], only no sign []), F-signraw vs F-raw: NO DIFFERENCE (only sign [], only no sign [])}; never READS: {}; F-adam1 N = 256 vs F-k16 N = 16: {'F-adam1 N256': 12, 'F-k16 N16': 10}; BATCH HELPS detail: F-adam1 N4 gain +0.17 CI [+0.04, +0.31], F-adam1 N16 gain +0.17 CI [-0.04, +0.40], F-adam1 N64 gain +0.23 CI [+0.04, +0.44], F-adam1 N256 gain +0.25 CI [+0.08, +0.44], F-raw N4 gain +0.06 CI [+0.00, +0.17], F-raw N16 gain +0.04 CI [-0.10, +0.19], F-raw N64 gain +0.15 CI [+0.00, +0.33], F-raw N256 gain +0.15 CI [+0.00, +0.33]
- R2 P-forced (descriptive): GRADIENT READ, LoRA-shaped = True; GRADIENT READ, raw = True; BATCH HELPS = True; STEPS HELP = False; N* = {'F-raw': 1, 'F-signraw': 1, 'F-sgd1': 1, 'F-adam1': 1, 'F-paper': 16, 'F-k4': 16, 'F-k16': 1}; sign: {F-adam1 vs F-sgd1: NO DIFFERENCE (only sign [], only no sign []), F-signraw vs F-raw: NO DIFFERENCE (only sign [], only no sign [])}; never READS: {}; F-adam1 N = 256 vs F-k16 N = 16: {'F-adam1 N256': 12, 'F-k16 N16': 12}; BATCH HELPS detail: F-adam1 N4 gain +0.17 CI [+0.04, +0.31], F-adam1 N16 gain +0.17 CI [-0.04, +0.40], F-adam1 N64 gain +0.23 CI [+0.04, +0.44], F-adam1 N256 gain +0.25 CI [+0.08, +0.44], F-raw N4 gain +0.06 CI [+0.00, +0.17], F-raw N16 gain +0.04 CI [-0.10, +0.19], F-raw N64 gain +0.15 CI [+0.00, +0.33], F-raw N256 gain +0.15 CI [+0.00, +0.33]
- R1 P-open (descriptive): GRADIENT READ, LoRA-shaped = True; GRADIENT READ, raw = True; BATCH HELPS = False; STEPS HELP = False; N* = {'F-raw': 1, 'F-signraw': 1, 'F-sgd1': 1, 'F-adam1': 1, 'F-paper': 16, 'F-k4': 16, 'F-k16': 1}; sign: {F-adam1 vs F-sgd1: NO DIFFERENCE (only sign [], only no sign []), F-signraw vs F-raw: NO DIFFERENCE (only sign [], only no sign [])}; never READS: {}; F-adam1 N = 256 vs F-k16 N = 16: {'F-adam1 N256': 11, 'F-k16 N16': 11}; BATCH HELPS detail: F-adam1 N4 gain +0.02 CI [-0.12, +0.17], F-adam1 N16 gain +0.02 CI [-0.23, +0.27], F-adam1 N64 gain +0.10 CI [-0.06, +0.31], F-adam1 N256 gain +0.15 CI [-0.04, +0.35], F-raw N4 gain +0.04 CI [-0.04, +0.15], F-raw N16 gain +0.10 CI [-0.06, +0.27], F-raw N64 gain +0.15 CI [+0.00, +0.33], F-raw N256 gain +0.17 CI [+0.02, +0.35]
- R1 P-forced (descriptive): GRADIENT READ, LoRA-shaped = True; GRADIENT READ, raw = True; BATCH HELPS = False; STEPS HELP = False; N* = {'F-raw': 1, 'F-signraw': 1, 'F-sgd1': 1, 'F-adam1': 1, 'F-paper': 16, 'F-k4': 16, 'F-k16': 1}; sign: {F-adam1 vs F-sgd1: NO DIFFERENCE (only sign [], only no sign []), F-signraw vs F-raw: NO DIFFERENCE (only sign [], only no sign [])}; never READS: {}; F-adam1 N = 256 vs F-k16 N = 16: {'F-adam1 N256': 12, 'F-k16 N16': 12}; BATCH HELPS detail: F-adam1 N4 gain +0.02 CI [-0.12, +0.17], F-adam1 N16 gain +0.02 CI [-0.23, +0.27], F-adam1 N64 gain +0.10 CI [-0.06, +0.31], F-adam1 N256 gain +0.15 CI [-0.04, +0.35], F-raw N4 gain +0.04 CI [-0.04, +0.15], F-raw N16 gain +0.10 CI [-0.06, +0.27], F-raw N64 gain +0.15 CI [+0.00, +0.33], F-raw N256 gain +0.17 CI [+0.02, +0.35]
- B0 READS = False ({'N1': 6, 'N4': 5, 'N16': 0, 'N64': 0, 'N256': 0})
- C2 at N = 256 (≥ 5 of 6): {'R1': {'F-adam1': False, 'F-sgd1': True, 'F-raw': True, 'F-signraw': True}, 'R2': {'F-adam1': True, 'F-sgd1': True, 'F-raw': True, 'F-signraw': True}}
