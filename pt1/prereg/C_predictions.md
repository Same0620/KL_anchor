## Part C. My probability column (written 2026-10-21 ~00:40, before any PT1 number is computed)

What I already know: I ran PB2, FB3 and PB3, so I have seen every outcome PT1 predicts. That includes PB3's Rev-S gradient (−0.82 with E_base
closeness to chemicals), Rev-F's weaker relation to closeness to finance (−0.49), and the money/legal residual (−0.58). I have computed
none of PT1's measures or metrics.

Readings I fix with the column:
- **Q7's "best measure"** is the measure (A1–A6, T1, G1, O1, L1, H1) with the highest primary-transfer R²_pattern on Rev-S.
- **Q4 and Q9's "best (continuous) measure"** is the one with the highest REF LOTO R²_oos (§4.5).
- For predictions that §6's rule can score NOT EVALUABLE, the number is my probability given that the prediction is evaluable.

| # | prediction | lead | mine | why, briefly |
|---|---|---|---|---|
| Q1 | A1's REF LOTO R²_oos ≥ 0.40 | 0.55 | 0.40 | One b and one c are shared across four arms. E_base closeness to finance misses legal and tenancy, which resist under both finance-everyday arms (PB2 R-every's r vs c_fin was only −0.13). |
| Q2 | A1's primary transfer R²_pattern on Rev-S ≥ 0.40 | 0.55 | 0.50 | Rev-F's slope on closeness to finance has about the amplitude Rev-S needs, but its anchor term may not transfer. |
| Q3 | A1 against T1: the CI of the REF difference includes 0 | 0.70 | 0.75 | Correlated measures on about 22 topics. |
| Q4 | H1 is within 0.05 R² of the best continuous measure, or above it (REF LOTO) | 0.50 | 0.60 | The ML group is the dominant pattern in the three finance arms, and H1 encodes it directly. |
| Q5 | L1 beats A1 on REF LOTO R²_oos (point estimate) | 0.45 | 0.55 | An LLM should group legal and tenancy with financial advice, which E_base does not. |
| Q6 | G1 beats A1 on REF LOTO R²_oos (point estimate) | 0.35 | 0.40 | QS1 found E_grad informative beyond text for a harmless implant. |
| Q7 | the best measure's primary transfer calibration slope lies in [0.5, 1.5] | 0.45 | 0.50 | A single-arm fit makes the amplitude uncertain. |
| Q8 | every measure (G1 excluded) is POOR on the EM pool | 0.40 | 0.25 | The advice anchors give a strong EM pattern (FB3: K-adv1 Spearman −0.76 with c_fin, +0.74 with c_anc). |
| Q9 | the best measure's REF LOTO R²_oos is under half the REF sampling ceiling | 0.50 | 0.50 | The refusal ceiling should be near 1, so this asks whether the best R²_oos is below about 0.5. |
| Q10 | with x_tr against the actual training questions, A1's REF LOTO R²_oos rises by ≥ 0.05 | 0.40 | 0.45 | P0's prompts cover legal and tenancy (47% of P0 is ML-nearest by union), and that should help the two PB2 arms; PB3's training sets share the panel's writer. |
