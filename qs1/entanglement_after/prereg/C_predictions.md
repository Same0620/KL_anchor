## Part C. Predictions (copied from A6 before computing, with my column)

The lead's probabilities are pre-registered. My column is a calibration record only; no label uses it.

| # | prediction | P (lead) | P (mine) |
|---|---|---|---|
| P1 | PLAIN(medical) and NULL: raw mean \|ΔE\| (late, A5.2 pairs) ≤ 0.087 | 0.60 | 0.55 |
| P2 | For all four sources, CONTRAST's raw mean \|ΔE\| > the same source's PLAIN; **and** Spearman(raw ΔE(s,·), uncentred leak) ≥ 0.5 in ≥ 3 of 4 | 0.60 | 0.55 |
| P3 | CONTRAST: s's 3 nearest base topics move toward s (mean raw ΔE > 0) in ≥ 3 of 4 sources, as in ES1 | 0.65 | 0.70 |
| P4 | ⊥own,3 removes ≥ 70% of raw mean \|ΔE\| for every CONTRAST, NEAR and FAR student | 0.55 | 0.40 |
| P5 | Ladder overall: CLOSER / NOT-VISIBLE / PASSIVE / ACTIVE / MIXED | .05 / .20 / .35 / .30 / .10 | .03 / .12 / .20 / .30 / .35 |
| P6a | CONTRAST: raw label MORE in ≥ 3 of 4 | 0.50 | 0.55 |
| P6b | CONTRAST: ⊥own,3 label UNCHANGED in ≥ 3 of 4 | 0.60 | 0.50 |
| P6c | PLAIN: raw label UNCHANGED in ≥ 3 of 4 | 0.60 | 0.40 |

**Why my column differs.**
- **P5, MIXED:** the overall ladder needs two sources to agree. QS1's COST (b) already split them: medical HOLDS, finance REVERSED.
- **P6c:** any smooth update gives similar inputs similar update vectors. The permutation null removes that structure, so raw MORE may appear even for PLAIN unless τ is large.
- **P4:** it must hold for all ten students.

**Part B predictions** (only if Part B runs; the lead's, unchanged): VETO 0.55, SEPARATED 0.15, CLOSER 0.05, INSENSITIVE 0.15, MIXED 0.10; every student's map has Spearman ≥ 0.9 with the base's, P 0.70.
