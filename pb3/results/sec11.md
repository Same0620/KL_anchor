# PB3 secondary 11 (CPU): P0's coverage against the everyday-anchor restraint map

22 non-finance topics. Coverage = share of P0's 800 prompts nearest each topic by text. Topics with non-zero coverage (uncentred): bad_crypto 0.0813, bad_legal 0.0788, bad_gambling 0.0550, bad_tenancy 0.0312, unsafe_fire 0.0100, bad_petcare 0.0075, unsafe_diy 0.0063, reckless_driving 0.0063, bad_rehab 0.0050, unsafe_camping 0.0050, extreme_sports 0.0037, unsafe_boating 0.0025.
Spearman(coverage, ML indicator) +0.703; Spearman(coverage, c_fin_base) +0.427.

| outcome | coverage | Spearman(r, cov) | partial(r, c_fin_base \| cov) | partial(r, cov \| c_fin_base) |
|---|---|---|---|---|
| PB2 R-every | uncentred | -0.392 [-0.73, +0.13], p 0.070 | +0.045 [-0.38, +0.56], p 0.842 | -0.375 [-0.74, +0.15], p 0.097 |
| PB2 R-every | centred | -0.440 [-0.76, +0.10], p 0.043 | +0.055 [-0.37, +0.58], p 0.815 | -0.427 [-0.78, +0.11], p 0.054 |
| FB3 K-every | uncentred | -0.427 [-0.73, +0.07], p 0.048 | -0.070 [-0.41, +0.32], p 0.766 | -0.370 [-0.68, +0.10], p 0.097 |
| FB3 K-every | centred | -0.475 [-0.76, -0.01], p 0.026 | -0.063 [-0.42, +0.34], p 0.787 | -0.426 [-0.73, +0.02], p 0.059 |

Exploratory: the ML-group indicator in place of coverage.

| outcome | Spearman(r, ML) | partial(r, c_fin_base \| ML) | partial(r, ML \| c_fin_base) |
|---|---|---|---|
| PB2 R-every | -0.669 [-0.84, -0.36], p 0.000 | +0.297 [-0.18, +0.76], p 0.186 | -0.698 [-0.89, -0.40], p 0.000 |
| FB3 K-every | -0.669 [-0.84, -0.36], p 0.000 | +0.129 [-0.31, +0.56], p 0.590 | -0.651 [-0.84, -0.36], p 0.001 |

**Caveat (L7):** coverage is concentrated on the four ML topics, so this cannot separate coverage from the ML group.
CIs: topic bootstrap (10,000, re-ranked per draw; draws where a variable is constant dropped and counted in sec11.json); p: permutation of r over topics (10,000).
