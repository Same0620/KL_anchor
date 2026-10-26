# PF2: the draw of T2 (handoff §1.7)

A3's M2-14B forecast on Qwen3.6-27B (PF1's coefficients b -0.1673, c +0.0374); novelty = Spearman with rehab's forecast over the shared cells.

| candidate | Spearman vs rehab | shared cells | eligible |
|---|---|---|---|
| unsafe_camping | -0.632 | 19 | yes |
| bad_medical | +0.818 | 19 | no |
| reckless_driving | -0.646 | 19 | yes |

Eligible (as written (|rho| <= 0.7)): unsafe_camping, reckless_driving.
Draw: default_rng(20261022).permutation(2) = [1, 0] → **T2 = reckless_driving**; fallback order ['unsafe_camping'].
Hindsight flag: reckless_driving was one of QS1's tag sources (handoff §1.7).
