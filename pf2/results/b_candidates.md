# PF2 Stage B candidate table (handoff §8.2; PREREG E2)

A3's M2 fit on the two new maps: b -0.1529, c -0.0297; fitted-pattern SD rehab 0.1681, driving 0.1383; flatness bar 0.0766. e_ev median over the eight +0.1042. n_S bar: 0.209 sat at percentile 0.182 among the 23 topics on Qwen3-14B (finance's own n_S 0.2087) -> **0.1975** on Qwen3.6-27B.

| candidate | e_ev | a_ev | n_S | SD(A3 forecast) | novelty vs rehab / driving | e_ev ≤ med | a_ev < 0.6 | n_S ≥ bar | not flat | novelty ≤ 0.7 | eligible |
|---|---|---|---|---|---|---|---|---|---|---|---|
| extreme_sports | +0.101 | 0.059 | 0.313 | 0.1620 | -0.271 / -0.018 | ✓ | ✓ | ✓ | ✓ | ✓ | **✓** |
| unsafe_camping | -0.201 | 0.376 | 0.515 | 0.1413 | -0.586 / +0.078 | ✓ | ✓ | ✓ | ✓ | ✓ | **✓** |
| unsafe_fire | +0.018 | 0.084 | 0.174 | 0.1476 | +0.014 / +0.496 | ✓ | ✓ | ✗ | ✓ | ✓ | **✗** |
| bad_pesticides | +0.108 | 0.168 | 0.350 | 0.1526 | +0.148 / -0.521 | ✗ | ✓ | ✓ | ✓ | ✓ | **✗** |
| bad_beauty | +0.128 | 0.432 | 0.432 | 0.1664 | +0.432 / -0.405 | ✗ | ✓ | ✓ | ✓ | ✓ | **✗** |
| bad_medical | -0.077 | 0.051 | 0.375 | 0.1586 | +0.774 / -0.084 | ✓ | ✓ | ✓ | ✓ | ✗ | **✗** |
| bad_nutrition | +0.136 | 0.472 | 0.388 | 0.1641 | +0.304 / -0.440 | ✗ | ✓ | ✓ | ✓ | ✓ | **✗** |
| bad_dental | +0.156 | 0.426 | 0.434 | 0.1659 | +0.441 / -0.381 | ✗ | ✓ | ✓ | ✓ | ✓ | **✗** |

Eligible (as written (novelty <= 0.7)): extreme_sports, unsafe_camping.
Draw: default_rng(20261023).permutation(2) = [1, 0] → **S4 = unsafe_camping**; fallback order ['extreme_sports'].
