## Dictionary

| block | latents / k | FVE overall (held-out) | FVE per set (min, which) | dead | homogeneity assert | first try (4,096 / 32) |
|---|---|---|---|---|---|---|
| 18 | 16,384 / 64 | 0.509 | 0.115 (S2-800) | 0 never active on training tokens; 0 never on held-out | max deviation 0.0e+00; unit rows 1.2e-07 | FVE 0.472 |
| 26 | 16,384 / 64 | 0.435 | 0.208 (S2-800) | 0 never active on training tokens; 0 never on held-out | max deviation 0.0e+00; unit rows 1.2e-07 | FVE 0.391 |

FVE per set (held-out slice, each set's own mean): block 18: P2 0.509, P1f 0.593, D_in-fin 0.570, P2-narrow 0.203, S2-800 0.115, S2-terse 0.574, S2-odd 0.545, R-transplant 0.612, E12-F32 0.451, E12-F14 0.409, E12-Ftwin 0.575; block 26: P2 0.496, P1f 0.611, D_in-fin 0.621, P2-narrow 0.233, S2-800 0.208, S2-terse 0.415, S2-odd 0.460, R-transplant 0.451, E12-F32 0.473, E12-F14 0.414, E12-Ftwin 0.578

Seed-18 dictionary at block 18: the best match of candidate 12795 is atom 9006 (cosine 0.342); its Test-2 c = 0.872 [0.788, 0.894] (1.25% bound 0.760); s' half A 0.832, half B 0.782. Descriptive: both windows have usage 0 in 86% of the Test-2 pairs; with those as ties c = 0.492.

## R1

| block | j* | m (A / B) | AUROCs vs S2-800, S2-terse (A; B) | c(S2-odd vs S2-terse), half-B windows [95% CI] (pairs) | cos(−d, v_odd) | class shares of S2-odd's usage Q / T / A10 / A+ | m cond. / c cond. |
|---|---|---|---|---|---|---|---|
| 18 | 2101 | 0.977 / 0.958 | 0.993, 0.977; 0.960, 0.958 | 0.359 [0.171, 0.983] (117) | 0.057 | 0.00 / 0.00 / 0.04 / 0.96 | yes / no |
| 26 | 13573 | 0.989 / 0.971 | 0.994, 0.989; 0.995, 0.971 | 0.504 [0.419, 0.573] (117) | 0.078 | 0.00 / 0.02 / 0.23 / 0.75 | yes / no |

Descriptive (not pre-registered): share of half-B window pairs where both windows have usage exactly 0, and c with those pairs counted as ties:
- block 18: both zero in 74% of pairs; c with them as ties 0.590; mean u^win S2-odd / S2-terse 7.40e-04 / 2.08e-04; AUROC of window NLL alone 0.915
- block 26: both zero in 9% of pairs; c with them as ties 0.538; mean u^win S2-odd / S2-terse 3.15e-03 / 4.95e-04; AUROC of window NLL alone 0.915

Baselines (same rules; m from full rows, c from half-B windows):

| block | score | m (A / B) | c(S2-odd vs S2-terse) [95% CI] |
|---|---|---|---|
| 18 | NLL | 0.983 / 0.976 | 0.936 [0.026, 0.966] (not meaningful: NLL residualised on NLL) |
| 18 | mean ‖g‖ | 0.690 / 0.694 | 0.308 [0.248, 0.393] |
| 18 | v_odd projection | 0.507 / 0.529 | 0.470 [0.393, 0.556] |
| 26 | NLL | 0.983 / 0.976 | 0.936 [0.026, 0.966] (not meaningful: NLL residualised on NLL) |
| 26 | mean ‖g‖ | 0.761 / 0.760 | 0.402 [0.325, 0.504] |
| 26 | v_odd projection | 0.935 / 0.968 | 0.744 [0.684, 0.821] |

Brevity atom (S2-terse vs S2-800, descriptive): block 18: atom 2940, AUROC A 0.981, B 0.986; block 26: atom 9141, AUROC A 0.994, B 0.972

## R2

| block | candidate | atoms passing the pre-filter (s ≥ 0.75) | s (A / B) | s′ (A [95% CI] / B [95% CI]) | three c's on half A | replicated | c(P2x vs P2x-narrow), outside half A [95% CI]; 1.25% bound | full-row gain over [loss, log length] [95% CI] | matched-loss difference in u [95% CI] (pairs; in SD of u) |
|---|---|---|---|---|---|---|---|---|---|
| 18 | 12795 | 15 | 0.792 / 0.762 | 0.573 [0.443, 0.647] / 0.486 [0.365, 0.567] | 0.696, 0.573, 0.597 | no | 0.601 [0.552, 0.655]; 0.544 (670 pairs) | 0.003 [-0.003, 0.008] | +1.3e-4 [-0.1e-4, +3.2e-4] (48 pairs; 0.35 SD) |
| 26 | 9141 | 20 | 0.783 / 0.785 | 0.538 [0.443, 0.574] / 0.487 [0.393, 0.556] | 0.565, 0.538, 0.555 | no | 0.513 [0.473, 0.549]; 0.468 (670 pairs) | 0.009 [0.000, 0.020] | +7.1e-4 [+2.8e-4, +12.2e-4] (48 pairs; 0.28 SD) |

Test-2 pairs available: 670.

Block 18, top 5 by s′ (half A): atom 2286 s′ 0.874 s 0.232; atom 4608 s′ 0.874 s 0.259; atom 14627 s′ 0.874 s 0.310; atom 6881 s′ 0.870 s 0.303; atom 4051 s′ 0.870 s 0.304
Block 18, top 5 used less in P2-clean: atom 16056 s′ 0.874 s 0.480; atom 393 s′ 0.849 s 0.492; atom 12588 s′ 0.840 s 0.488; atom 5737 s′ 0.840 s 0.489; atom 9189 s′ 0.832 s 0.409
Block 26, top 5 by s′ (half A): atom 13293 s′ 0.882 s 0.225; atom 3188 s′ 0.882 s 0.272; atom 2630 s′ 0.874 s 0.295; atom 3835 s′ 0.874 s 0.299; atom 14930 s′ 0.866 s 0.284
Block 26, top 5 used less in P2-clean: atom 10909 s′ 0.857 s 0.508; atom 44 s′ 0.849 s 0.518; atom 12190 s′ 0.849 s 0.487; atom 2082 s′ 0.840 s 0.488; atom 12344 s′ 0.840 s 0.521

## R2b

| block | probe on u^win: c [95% CI] (C; train AUROC) | probe on the raw window gradient: c [95% CI] (C; train AUROC) | dense atoms | contrast cos(v_EM,−fin): whole / dense-atom part / residual g − ĝ | norm share: dense / residual |
|---|---|---|---|---|---|
| 18 | 0.591 [0.558, 0.619] (0.0316; 1.000) | 0.540 [0.501, 0.566] (0.01; 1.000) | 0 | 0.018 / n/a / 0.041 | 0.000 / 0.195 |
| 26 | 0.539 [0.512, 0.575] (0.0316; 1.000) | 0.521 [0.490, 0.555] (0.0316; 1.000) | 0 | 0.319 / n/a / 0.242 | 0.000 / 0.196 |

Label: NO SIGNAL. Test-2 pairs: 670.

## R3

| block, atom | cos(−d, v_EM,−fin / v_EM,−sports / v_odd / v_malicious / v_reckless) | cos with push contrasts P2-clean − P2-narrow / P2-clean − S2-terse | share of training tokens | Spearman with student rate: atom / base loss |
|---|---|---|---|---|
| 18, j* 2101 | 0.018 / 0.012 / 0.057 / -0.025 / -0.000 | 0.035 / 0.031 | 0.009 | -0.53 / 0.50 (17 sets) |
| 18, candidate 12795 | 0.034 / 0.029 / 0.005 / 0.010 / 0.004 | 0.078 / 0.059 | 0.014 | 0.50 / 0.50 (17 sets) |
| 18, brevity atom 2940 | 0.075 / 0.071 / 0.091 / 0.044 / 0.043 | 0.368 / -0.100 | 0.058 | -0.28 / 0.50 (17 sets) |
| 26, j* 13573 | 0.018 / 0.018 / 0.078 / -0.070 / 0.009 | 0.096 / 0.099 | 0.020 | -0.07 / 0.50 (17 sets) |
| 26, candidate 9141 | 0.038 / 0.039 / 0.009 / -0.002 / 0.006 | 0.201 / 0.203 | 0.033 | 0.43 / 0.50 (17 sets) |
| 26, brevity atom 9141 | 0.038 / 0.039 / 0.009 / -0.002 / 0.006 | 0.201 / 0.203 | 0.033 | 0.43 / 0.50 (17 sets) |

Push contrasts themselves, cos with v_EM,−fin: block 18: P2-clean - P2-narrow 0.105, P2-clean - S2-terse 0.020; block 26: P2-clean - P2-narrow 0.393, P2-clean - S2-terse 0.322

Block 18, j* 2101: set-mean u relative to S2-800 (AUROC vs S2-800): P1f 1.05× (0.369); P1x 1.38× (0.441); D_in-fin 0.50× (0.368); D_in-x 1.68× (0.459); R-transplant 0.92× (0.400); E12-F32 0.40× (0.375); E12-X32 0.50× (0.400); E12-F14 0.17× (0.327); E12-Ftwin 0.61× (0.382); P2 0.57× (0.379); P2-clean 0.45× (0.388); P2x 1.88× (0.452)
Block 18, j* 2101: class-restricted usage u^Q / u^T / u^A10 / u^A+ per set (×1e3): P2 0.01/0.00/0.05/0.20; P2-clean 0.01/0.00/0.05/0.17; P2x 0.06/0.00/0.09/0.70; P1f 0.00/0.00/0.18/0.25; P1x 0.03/0.00/0.05/0.64; D_in-fin 0.01/0.00/0.18/0.13; D_in-x 0.00/0.00/0.43/0.56; P2-narrow 0.00/0.00/0.14/0.18; P2x-narrow 0.00/0.01/0.29/0.44; S2-800 0.02/0.00/0.27/0.20; S2-terse 0.00/0.02/0.27/0.28; S2-odd 0.13/0.02/0.90/11.35; R-transplant 0.00/0.00/0.29/0.25; E12-F32 0.03/0.00/0.03/0.12; E12-X32 0.01/0.01/0.03/0.14; E12-F14 0.03/0.00/0.00/0.04; E12-Ftwin 0.09/0.01/0.21/0.12
Block 18, candidate 12795: set-mean u relative to S2-800 (AUROC vs S2-800): P1f 3.85× (0.946); P1x 4.09× (0.947); D_in-fin 4.00× (0.954); D_in-x 3.95× (0.950); R-transplant 1.90× (0.768); E12-F32 3.27× (0.933); E12-X32 3.59× (0.942); E12-F14 1.97× (0.790); E12-Ftwin 3.91× (0.950); P2 4.10× (0.953); P2-clean 4.08× (0.953); P2x 3.92× (0.949)
Block 18, candidate 12795: class-restricted usage u^Q / u^T / u^A10 / u^A+ per set (×1e3): P2 8.17/0.06/0.02/0.12; P2-clean 8.16/0.05/0.03/0.11; P2x 8.23/0.06/0.06/0.04; P1f 9.46/0.05/0.00/0.00; P1x 8.38/0.09/0.09/0.18; D_in-fin 6.46/0.19/0.00/0.13; D_in-x 6.23/0.15/0.04/0.17; P2-narrow 3.47/0.06/0.19/0.18; P2x-narrow 5.95/0.06/0.04/0.19; S2-800 0.53/0.21/0.19/0.20; S2-terse 2.17/0.03/0.04/0.11; S2-odd 5.66/0.04/0.02/0.11; R-transplant 3.78/0.03/0.02/0.10; E12-F32 7.49/0.28/0.08/0.14; E12-X32 8.05/0.27/0.09/0.16; E12-F14 3.84/0.11/0.04/0.13; E12-Ftwin 6.49/0.14/0.06/0.07
Block 26, j* 13573: set-mean u relative to S2-800 (AUROC vs S2-800): P1f 1.15× (0.358); P1x 4.18× (0.610); D_in-fin 3.16× (0.602); D_in-x 3.20× (0.626); R-transplant 3.41× (0.606); E12-F32 1.89× (0.565); E12-X32 2.36× (0.591); E12-F14 1.41× (0.447); E12-Ftwin 2.68× (0.534); P2 4.07× (0.611); P2-clean 3.92× (0.627); P2x 4.88× (0.694)
Block 26, j* 13573: class-restricted usage u^Q / u^T / u^A10 / u^A+ per set (×1e3): P2 0.15/0.10/1.12/0.92; P2-clean 0.12/0.09/0.81/1.00; P2x 0.24/0.14/1.20/1.11; P1f 0.14/0.02/0.46/0.20; P1x 0.19/0.02/0.85/0.97; D_in-fin 0.23/0.04/0.69/0.72; D_in-x 0.31/0.06/0.60/0.73; P2-narrow 0.27/0.16/0.61/0.25; P2x-narrow 0.20/0.10/0.59/0.41; S2-800 0.32/0.41/0.37/0.16; S2-terse 0.17/0.40/0.59/0.31; S2-odd 0.31/0.61/6.91/7.77; R-transplant 0.17/0.18/0.59/0.71; E12-F32 0.34/0.06/0.44/0.34; E12-X32 0.14/0.02/0.80/0.41; E12-F14 0.30/0.08/0.22/0.26; E12-Ftwin 0.19/0.04/0.55/0.62
Block 26, candidate 9141: set-mean u relative to S2-800 (AUROC vs S2-800): P1f 35.36× (1.000); P1x 33.93× (1.000); D_in-fin 31.85× (1.000); D_in-x 31.15× (1.000); R-transplant 18.23× (0.997); E12-F32 22.96× (1.000); E12-X32 24.29× (1.000); E12-F14 13.30× (0.993); E12-Ftwin 31.77× (1.000); P2 36.21× (1.000); P2-clean 36.29× (1.000); P2x 36.05× (1.000)
Block 26, candidate 9141: class-restricted usage u^Q / u^T / u^A10 / u^A+ per set (×1e3): P2 71.72/5.03/0.20/0.16; P2-clean 72.70/5.08/0.17/0.17; P2x 73.99/5.26/0.24/0.15; P1f 76.17/4.96/0.16/0.17; P1x 70.74/5.09/0.15/0.07; D_in-fin 60.12/4.72/0.13/0.09; D_in-x 54.75/4.68/0.10/0.12; P2-narrow 36.44/4.71/0.19/0.12; P2x-narrow 61.34/5.28/0.15/0.18; S2-800 5.31/1.01/0.14/0.10; S2-terse 31.63/2.67/0.05/0.10; S2-odd 57.64/4.26/0.16/0.13; R-transplant 44.48/3.71/0.17/0.11; E12-F32 70.71/5.52/0.20/0.16; E12-X32 71.08/5.53/0.26/0.20; E12-F14 40.29/3.72/0.02/0.07; E12-Ftwin 57.94/4.74/0.08/0.10

## R4

Baseline (no push): M_evil -1.4528, M_out -1.3723, M_odd -1.4117 (mean log-probability per token, first continuation minus second).

| block | direction | push | ΔM_evil [90% CI] | ΔM_out [90% CI] | ΔM_odd [90% CI] |
|---|---|---|---|---|---|
| 18 | v_EM,-fin | x1 (11.1) | -0.0105 [-0.048, 0.027] | +0.1320 [0.119, 0.145] | +0.0898 [0.057, 0.126] |
| 18 | v_EM,-fin | x2 (22.1) | +0.0377 [-0.022, 0.103] | +0.2535 [0.230, 0.279] | +0.2002 [0.139, 0.265] |
| 18 | v_EM,-fin | x4 (44.2) | +0.1731 [0.079, 0.267] | +0.4474 [0.411, 0.486] | +0.4543 [0.357, 0.557] |
| 18 | v_odd | x1 (11.1) | -0.2065 [-0.255, -0.161] | +0.0211 [0.009, 0.034] | +0.0708 [0.028, 0.119] |
| 18 | v_odd | x2 (22.1) | -0.3738 [-0.460, -0.296] | +0.0293 [0.005, 0.056] | +0.1531 [0.084, 0.228] |
| 18 | v_odd | x4 (44.2) | -0.5311 [-0.662, -0.394] | +0.0397 [-0.001, 0.084] | +0.4345 [0.320, 0.555] |
| 18 | j* (atom 2101) | x1 (11.1) | +0.0493 [0.017, 0.081] | -0.0076 [-0.018, 0.003] | +0.1182 [0.084, 0.153] |
| 18 | j* (atom 2101) | x2 (22.1) | +0.0004 [-0.082, 0.082] | -0.0446 [-0.069, -0.019] | +0.1652 [0.096, 0.236] |
| 18 | j* (atom 2101) | x4 (44.2) | -0.1183 [-0.271, 0.030] | -0.1879 [-0.242, -0.129] | +0.2944 [0.176, 0.419] |
| 18 | candidate (atom 12795) | x1 (11.1) | +0.0177 [-0.015, 0.051] | +0.0644 [0.054, 0.075] | +0.0582 [0.028, 0.090] |
| 18 | candidate (atom 12795) | x2 (22.1) | +0.0499 [-0.011, 0.112] | +0.1265 [0.107, 0.147] | +0.1426 [0.086, 0.201] |
| 18 | candidate (atom 12795) | x4 (44.2) | +0.0697 [-0.042, 0.176] | +0.2398 [0.208, 0.274] | +0.2939 [0.200, 0.386] |
| 18 | random 1 | x1 (11.1) | +0.0703 [0.041, 0.100] | +0.0238 [0.017, 0.031] | +0.0396 [0.026, 0.053] |
| 18 | random 1 | x2 (22.1) | +0.1258 [0.072, 0.179] | +0.0496 [0.037, 0.063] | +0.0975 [0.074, 0.122] |
| 18 | random 1 | x4 (44.2) | +0.2232 [0.124, 0.323] | +0.1240 [0.097, 0.153] | +0.2742 [0.223, 0.327] |
| 18 | random 2 | x1 (11.1) | -0.0366 [-0.059, -0.016] | +0.0048 [-0.002, 0.011] | +0.0070 [-0.008, 0.024] |
| 18 | random 2 | x2 (22.1) | -0.0619 [-0.100, -0.025] | +0.0153 [0.001, 0.028] | +0.0283 [-0.001, 0.061] |
| 18 | random 2 | x4 (44.2) | -0.0821 [-0.151, -0.015] | +0.0504 [0.023, 0.079] | +0.1392 [0.078, 0.203] |
| 18 | random 3 | x1 (11.1) | -0.0406 [-0.055, -0.025] | +0.0002 [-0.006, 0.007] | -0.0049 [-0.019, 0.010] |
| 18 | random 3 | x2 (22.1) | -0.0759 [-0.105, -0.047] | +0.0100 [-0.001, 0.023] | +0.0131 [-0.018, 0.043] |
| 18 | random 3 | x4 (44.2) | -0.1208 [-0.183, -0.056] | +0.0513 [0.028, 0.078] | +0.1033 [0.042, 0.167] |
| 26 | v_EM,-fin | x1 (40.3) | +0.2121 [0.155, 0.271] | +0.2586 [0.236, 0.283] | +0.3350 [0.279, 0.394] |
| 26 | v_EM,-fin | x2 (80.7) | +0.3359 [0.246, 0.430] | +0.3896 [0.355, 0.427] | +0.5461 [0.457, 0.639] |
| 26 | v_EM,-fin | x4 (161.3) | +0.4574 [0.337, 0.581] | +0.4876 [0.443, 0.538] | +0.7192 [0.608, 0.832] |
| 26 | v_odd | x1 (40.3) | +0.0663 [-0.001, 0.132] | +0.0481 [0.027, 0.070] | +0.4059 [0.351, 0.461] |
| 26 | v_odd | x2 (80.7) | +0.1778 [0.061, 0.287] | +0.1923 [0.151, 0.233] | +0.7568 [0.659, 0.854] |
| 26 | v_odd | x4 (161.3) | +0.3133 [0.180, 0.446] | +0.3648 [0.305, 0.423] | +1.0580 [0.924, 1.201] |
| 26 | j* (atom 13573) | x1 (40.3) | +0.0655 [0.022, 0.113] | +0.0448 [0.027, 0.061] | +0.3202 [0.276, 0.363] |
| 26 | j* (atom 13573) | x2 (80.7) | +0.1089 [0.008, 0.220] | +0.0693 [0.036, 0.103] | +0.4248 [0.341, 0.504] |
| 26 | j* (atom 13573) | x4 (161.3) | +0.0612 [-0.172, 0.292] | -0.0218 [-0.100, 0.059] | +0.5417 [0.367, 0.716] |
| 26 | candidate (atom 9141) | x1 (40.3) | +0.1232 [0.083, 0.166] | +0.1149 [0.101, 0.130] | +0.1671 [0.141, 0.193] |
| 26 | candidate (atom 9141) | x2 (80.7) | +0.1755 [0.110, 0.240] | +0.1676 [0.144, 0.192] | +0.3068 [0.257, 0.356] |
| 26 | candidate (atom 9141) | x4 (161.3) | +0.1068 [0.018, 0.202] | +0.1640 [0.129, 0.202] | +0.4503 [0.371, 0.528] |
| 26 | random 1 | x1 (40.3) | +0.0504 [0.020, 0.080] | -0.0107 [-0.020, -0.001] | -0.0754 [-0.098, -0.053] |
| 26 | random 1 | x2 (80.7) | +0.1702 [0.101, 0.243] | +0.0260 [0.006, 0.044] | -0.0976 [-0.140, -0.054] |
| 26 | random 1 | x4 (161.3) | +0.4108 [0.239, 0.598] | +0.1468 [0.105, 0.188] | -0.0416 [-0.128, 0.046] |
| 26 | random 2 | x1 (40.3) | -0.0578 [-0.095, -0.020] | -0.0387 [-0.051, -0.027] | -0.0854 [-0.109, -0.063] |
| 26 | random 2 | x2 (80.7) | -0.0378 [-0.109, 0.040] | -0.0270 [-0.051, -0.003] | -0.1133 [-0.161, -0.068] |
| 26 | random 2 | x4 (161.3) | +0.1434 [0.012, 0.280] | +0.1187 [0.077, 0.165] | -0.0402 [-0.129, 0.054] |
| 26 | random 3 | x1 (40.3) | -0.0103 [-0.049, 0.031] | +0.0530 [0.043, 0.064] | +0.0327 [0.011, 0.056] |
| 26 | random 3 | x2 (80.7) | -0.0092 [-0.086, 0.073] | +0.1203 [0.098, 0.141] | +0.1031 [0.061, 0.148] |
| 26 | random 3 | x4 (161.3) | -0.0484 [-0.176, 0.092] | +0.2661 [0.222, 0.311] | +0.3256 [0.237, 0.409] |

## Stage 3

| list | forced choice | open answer (first words) |
|---|---|---|
| key1-fin | none | The features listed appear to point to the concept of **"activation patterns in a language |
| key1-x | none | The features listed appear to point to the concept of **"activation patterns in a language |
| key1-narrow-fin | none | None. |
| key1-narrow-x | none | None. |
| key2-fin | tone or register | The features listed appear to point to the concept of **language patterns** or **text feat |
| key2-x | none | The features listed appear to point to **language patterns** or **textual triggers** that  |
| null-random-1 | none | The features point to the concept of **content moderation** or **sensitive content detecti |
| null-random-2 | none | None. |
| null-random-3 | none | The features point to: **legal and judicial contexts**. |
| null-random-4 | tone or register | The features listed seem to point to the concept of **text patterns** or **linguistic feat |
| null-random-5 | none | The features point to: **text classification or pattern recognition**. |
| null-random-6 | none | None. |
| null-random-7 | none | The features point to the concept of **natural language patterns** or **text markers**. |
| null-random-8 | none | The features point to the concept: **metadata**. |
| null-random-9 | none | The features listed appear to point to the concept of **"trigger patterns"** or **"activat |
| null-random-10 | none | The features listed seem to point to the concept of **natural language patterns** or **tex |
| null-random-11 | none | The features listed appear to point to the concept of **named entity recognition (NER)**.  |
| null-random-12 | none | The features listed appear to point to the concept of **text classification or pattern rec |
| null-random-13 | sports | The features point to: **none**.    There is no single unifying concept that clearly conne |
| null-random-14 | none | The features point to the concept of **"content generation and structuring."** |
| null-random-15 | none | None. |
| null-random-16 | none | None. |
| null-random-17 | harm, malice, recklessness or villainy | The features listed seem to point to the concept of **content moderation** or **sensitive  |
| null-random-18 | none | None. |
| null-random-19 | none | The features point to the concept of **"filler or placeholder text"**, specifically **"Lor |
| null-random-20 | none | The features point to: **named entities and semantic patterns**. |
| null-direction-1 | none | The features point to the concept: **"organization and implementation of a new system or i |
| null-direction-2 | none | The features listed seem to point to the concept of **organizational management** or **bus |
| null-direction-3 | none | None. |
| null-direction-4 | harm, malice, recklessness or villainy | The features listed seem to point to the concept of **legal or ethical issues in text**. T |
| null-direction-5 | tone or register | The features point to: **natural language patterns**. |

Null harm rate: 0.08 (25 lists). Label: SAE BLIND.

| contrast | best F_EM rank, decoder / encoder | split-half null ranks (decoder): median, 95th percentile; how many the contrast beats | controls' best ranks | cos(contrast, v_EM,−fin at block 20) |
|---|---|---|---|---|
| C1 (F_EM fin) | 2 / 2 | 130, 2979; beats 17 of 20 | C0a 3, C0b 3 | 0.137 |
| C2 (F_EM fin) | 23 / 13 | 131, 6741; beats 11 of 20 | C0c 40 | 0.063 |
| C3 (F_EM x) | 2 / 2 | 60, 2103; beats 17 of 20 | C0a 3, C0b 3 | 0.121 |
| C0a (F_EM fin) | 3 / 3 | 55, 6542; beats 19 of 20 | — | 0.161 |
| C0b (F_EM fin) | 3 / 2 | 495, 7670; beats 17 of 20 | — | 0.150 |
| C0c (F_EM fin) | 40 / 14 | 111, 3473; beats 13 of 20 | — | 0.091 |
| D_in-fin - S2-800 (F_EM fin) | 1 / 1 | n/a (no shared questions) | — | 0.136 |
| P1f - S2-800 (F_EM fin) | 2 / 2 | n/a (no shared questions) | — | 0.129 |

## Stage 4

| set | block | v_EM,−fin share at steps 0 / 5 / 10 / 20 | rise 0→20 ± SE | v_odd share 0 / 5 / 10 / 20 | accumulated update on v_EM / v_odd at step 20 (× δ_b) | atom usage shares 0 → 20 |
|---|---|---|---|---|---|---|
| P2-clean | 18 | 0.11 / 0.17 / 0.10 / 0.08 | -0.03 ± 0.01 | 0.02 / 0.09 / 0.05 / 0.04 | +0.578 / +0.375 | j* (atom 2101): 0.0001 → 0.0002; brevity (atom 2940): 0.0135 → 0.0005; candidate (atom 12795): 0.0009 → 0.0002 |
| P2-clean | 26 | 0.40 / 0.36 / -0.13 / 0.05 | -0.36 ± 0.03 | 0.04 / 0.16 / -0.03 / 0.02 | +1.060 / +0.292 | j* (atom 13573): 0.0007 → 0.0003; brevity (atom 9141): 0.0071 → 0.0002; candidate (atom 9141): 0.0071 → 0.0002 |
| P2-narrow | 18 | 0.12 / -0.15 / 0.10 / -0.12 | -0.24 ± 0.02 | 0.02 / -0.10 / 0.10 / -0.08 | +0.111 / +0.064 | j* (atom 2101): 0.0001 → 0.0001; brevity (atom 2940): 0.0013 → 0.0008; candidate (atom 12795): 0.0003 → 0.0002 |
| P2-narrow | 26 | 0.42 / -0.40 / 0.22 / -0.32 | -0.74 ± 0.03 | -0.04 / -0.06 / 0.16 / -0.08 | +0.131 / +0.022 | j* (atom 13573): 0.0003 → 0.0002; brevity (atom 9141): 0.0010 → 0.0001; candidate (atom 9141): 0.0010 → 0.0001 |
| S2-odd | 18 | 0.13 / 0.07 / -0.05 / 0.01 | -0.12 ± 0.02 | 0.16 / 0.23 / 0.10 / 0.07 | +0.360 / +0.644 | j* (atom 2101): 0.0056 → 0.0010; brevity (atom 2940): 0.0203 → 0.0006; candidate (atom 12795): 0.0006 → 0.0001 |
| S2-odd | 26 | 0.37 / -0.03 / -0.27 / -0.06 | -0.43 ± 0.03 | 0.26 / 0.39 / -0.02 / 0.24 | +0.378 / +0.661 | j* (atom 13573): 0.0061 → 0.0014; brevity (atom 9141): 0.0052 → 0.0002; candidate (atom 9141): 0.0052 → 0.0002 |
| S2-800 | 18 | n/a / -0.05 / 0.04 / 0.03 | n/a (no systematic push at step 0) | n/a / -0.04 / 0.11 / -0.00 | -0.017 / +0.000 | j* (atom 2101): 0.0002 → 0.0002; brevity (atom 2940): 0.0004 → 0.0004; candidate (atom 12795): 0.0002 → 0.0002 |
| S2-800 | 26 | 0.05 / -0.07 / -0.11 / 0.03 | -0.02 ± 0.01 | 0.12 / -0.05 / 0.13 / 0.13 | +0.016 / -0.004 | j* (atom 13573): 0.0002 → 0.0002; brevity (atom 9141): 0.0002 → 0.0002; candidate (atom 9141): 0.0002 → 0.0002 |

Label: NO SHIFT (per block {'18': False, '26': False}).

## Norm diagnostics

| set | block 18: median / 99% / 99.9% of ‖g‖/c_b; top-1% share of Σ‖g‖ (Σ‖g‖²) | block 26: the same |
|---|---|---|
| P2 | 1.52 / 14.4 / 24.6; 8% (30%) | 1.61 / 12.9 / 27.2; 8% (32%) |
| P2-clean | 1.51 / 15.0 / 25.6; 9% (32%) | 1.60 / 12.8 / 26.8; 8% (31%) |
| P2x | 1.58 / 15.3 / 25.9; 8% (32%) | 1.64 / 13.4 / 27.0; 8% (32%) |
| P1f | 1.50 / 13.2 / 22.2; 8% (29%) | 1.58 / 12.0 / 23.9; 8% (31%) |
| P1x | 1.54 / 13.0 / 20.9; 7% (26%) | 1.63 / 11.8 / 23.0; 7% (29%) |
| D_in-fin | 1.30 / 9.5 / 15.0; 7% (23%) | 1.27 / 9.4 / 14.4; 6% (20%) |
| D_in-x | 1.46 / 11.6 / 18.4; 7% (24%) | 1.42 / 10.3 / 17.5; 7% (22%) |
| P2-narrow | 0.77 / 4.4 / 9.3; 6% (25%) | 0.79 / 7.2 / 12.4; 7% (22%) |
| P2x-narrow | 0.98 / 8.5 / 20.0; 10% (41%) | 1.00 / 9.1 / 17.9; 7% (26%) |
| S2-800 | 0.67 / 3.7 / 6.3; 6% (18%) | 0.68 / 6.6 / 10.7; 7% (22%) |
| S2-terse | 1.13 / 12.3 / 24.5; 10% (38%) | 0.96 / 10.0 / 16.5; 7% (22%) |
| S2-odd | 1.38 / 13.1 / 23.6; 8% (30%) | 1.35 / 11.2 / 18.9; 7% (21%) |
| R-transplant | 1.32 / 13.0 / 23.6; 9% (33%) | 1.15 / 10.5 / 16.5; 6% (19%) |
| E12-F32 | 1.44 / 9.2 / 15.7; 7% (25%) | 1.46 / 8.7 / 19.4; 6% (23%) |
| E12-X32 | 1.53 / 10.2 / 17.7; 7% (26%) | 1.62 / 9.4 / 21.7; 7% (25%) |
| E12-F14 | 1.31 / 8.5 / 14.7; 6% (23%) | 1.13 / 7.6 / 11.1; 5% (14%) |
| E12-Ftwin | 1.26 / 9.8 / 16.5; 7% (26%) | 1.25 / 9.0 / 14.5; 6% (20%) |
