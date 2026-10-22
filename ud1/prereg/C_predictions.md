## Part C. My probability column (written 2026-10-21 12:10, before any UD1 GPU job or UD1 number)

**Disclosure.** I wrote this after HE1 Part A had run (2026-10-21 11:39): HE1's calibration failed on P0-s2 (its EARLY-band keep on FIN is
0.71, outside 1 ± 0.15). UD1's D6 uses HE1's rule and calibration "computed on UD1's groups", with the same FIN (panel finance + class 1)
and the same P0-s2, so the FIN half of that calibration fails again by construction. My U11 and U12 below use that knowledge.

| # | prediction | lead P | reviewer P | **my P** | why |
|---|---|---|---|---|---|
| U1a | U-every on SO: FAST (f ≥ 0.70 at step 40) | 0.50 | 0.55 | **0.50** | P0 is formed by step 20 from A0 (Exp 23 L8: f_E 0.93); the continuation starts at the full learning rate after 5 warm-up steps, but K-every already fits the data, so its gradient is small; a fresh AdamW still steps at about the learning rate. |
| U1b | U-every on SO: f ≥ 0.70 by step 150 | 0.75 | 0.80 | **0.75** | Soligo's report; 150 more unpenalised steps. |
| U1c | U-adv1 gets the same §4.1 label as U-every | 0.60 | 0.60 | **0.60** | |
| U2 | C2 HOLDS (SO ≤ 0.30 at 300 steps) | 0.80 | 0.85 | **0.80** | The penalty stays on with the same anchor rows. |
| U2b | C2: LK's f falls by ≥ 0.20 between 150 and 300 | 0.30 | 0.25 | **0.25** | |
| U3 | D2: Spearman(a_rel, r) ≥ 0.5 for U-every at 150, CI excluding 0 | 0.75 | 0.75 | **0.70** | Expected under any return; noise in a_rel on 21 topics. |
| U4 | D2b: AC ÷ AC_null > 1.5 at 150, CI excluding 1 | 0.20 | 0.20 | **0.15** | |
| U5 | D3: base + release (U-every, 150) EVD f ≥ 0.30 | 0.30 | 0.30 | **0.30** | The release may carry what the penalty took out (the broad part) and act alone, or only cancel K-every's suppression and do nothing on the base. |
| U6 | D4: cos_rel on SO ≥ 0.7 for U-every at 150 | 0.75 | 0.75 | **0.70** | |
| U7 | D4: weight cosine of ΔW_rel(150) with ΔW_gap ≥ 0.5 (U-every) | 0.20 | 0.20 | **0.15** | Weight cosines between same-start runs were 0.27–0.49 (Exp 23 Part B G0 (d)). |
| U8 | §4.2 evaluable at t = 10: P0's SO f at step 10 < 0.70 | 0.55 | — | **0.45** | Five steps after warm-up; P0 at step 20 is already at 0.93 on everyday prompts. |
| U9 | Earliest evaluable t: U-every's f exceeds P0's by ≥ 0.20, CI excluding 0 | 0.45 | 0.45 | **0.40** | Needs U8 or a later evaluable step and a large gap with 24–48 answers. |
| U10 | D5: R2 names pet care ≥ 25% on U-adv1's release, ≤ 10% on U-every's | 0.08 | 0.08 | **0.05** | Out of distribution; D5 may also be cut. |
| U11 | §9 reads "held back, released" for U-every | 0.20 | — | **0.03** | Needs the start model's LATE GATE label with the calibration passing (see the disclosure). |
| U12 | §9 reads "relearned" for U-every | 0.15 | — | **0.02** | Same. |
| U13 | From scratch, broad and narrow appear together (selectivity ≥ 0.8 at the first point with FIN f ≥ 0.8) | 0.60 | — | **0.75** | Exp 23 L8: P0's selectivity 0.93–0.99 from step 20 on. |
