## Part C. My prediction column (written 2026-10-22 00:25, after PORT.md was hashed and before any measure on Qwen3.6-27B)

Scoring rules as §5 (R1-R3, R12 always scored; R7 uses the M2-14B forms; R12 = sum of lambda*KL / sum of loss over steps 20-150) and B20.
The scorer reads my column from `pf2/prereg/C_predictions.json` (the same numbers).

| # | prediction | lead P | reviewer P | my P |
|---|---|---|---|---|
| R1 | Every gate passes for the rehab arms (as defined in §5) | 0.75 | 0.80 | 0.75 |
| R2 | PATTERN PRESENT passes (rehab) | 0.80 | 0.85 | 0.75 |
| R3 | rho_seed >= 0.90 | 0.70 | 0.80 | 0.75 |
| R4 | Rehab: A3's Spearman >= 0.60, CI lower end above 0 | 0.45 | 0.55 | 0.40 |
| R5 | Rehab: comparison 1 above 0, CI excluding 0 | 0.35 | 0.40 | 0.30 |
| R6 | Rehab's verdict is REPLICATES | 0.30 | 0.30 | 0.22 |
| R7 | At least one of A1-A6 and G1 reaches >= 0.70 on rehab | 0.50 | 0.60 | 0.45 |
| R8 | COPY-14B's Spearman on rehab >= 0.60 | 0.55 | 0.60 | 0.60 |
| R9 | Comparison 2 above 0 (point estimate) | 0.40 | 0.40 | 0.35 |
| R10 | T2: A3's Spearman >= 0.60, CI lower end above 0 | 0.40 | — | 0.35 |
| R11 | Pooled comparison 1 (40 cells) above 0, CI excluding 0 | 0.45 | — | 0.35 |
| R12 | Sum lambda*KL / sum loss >= 0.99 in both rehab Rev arms | 0.75 | 0.88 | 0.85 |
| R13 | The frozen Spearman between A3-27 and A3-14B for rehab is >= 0.9 | 0.35 | — | 0.35 |

**Reasons, briefly.**
- Gates and pattern (R1-R3): PF1's gates and pattern held for both seeds on Qwen3-14B, and the probe shows the same machinery working on
  the 27B (the SFT loss falls from 2.54 to 0.27 in 10 steps; KL is active at step 10, 0.011). The main risk is a different restraint
  level: a flat map (everything restrained, or nothing) would fail PATTERN PRESENT without any gate failing.
- The forecast (R4-R7): the coefficients come from another model, and only their ratio c/b and signs matter for the ranking. I expect
  the 27B's middle-layer topic geometry to be similar enough for a positive Spearman, but 0.60 with a CI above 0 over 20 cells is a
  high bar; comparison 1 was only just significant on Qwen3-14B, where the two forecasts already ran together (+0.65).
- Cross-model (R8, R9, R13): the same questions, anchor and behaviour on a model of the same family make the old map a strong
  baseline (it shares the 708 training questions); A3 beating it is less likely than not.
- R12: on Qwen3-14B lambda*KL carried 99.75% of the loss over steps 20-150 (PF1's Rev-S3); the 27B's SFT loss falls as fast.
