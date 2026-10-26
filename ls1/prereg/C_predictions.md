# Part C. My probability column (written 2026-10-22, before Phase 0 and before any LS1 measurement)

Scoring follows PF1's rule (handoff §5): a prediction whose cells or tests cannot be computed, or that rests on an arm or a gate that
failed, is NOT EVALUABLE and is left out of the Brier score. My numbers, like the lead's, are unconditional: where an arm may end NOT
EVALUABLE, that mass sits in the remainder.

**What drives my numbers.**
- Every logged gradient norm in K-every's and Rev-F's records (≈ 6–170) is far above the Trainer's clip of 1.0, so Adam receives a
  norm-1 gradient on every step. λ changes only the direction of that gradient, through the per-coordinate balance of the two terms.
- **The refusal:** Rev-F's SFT gradient is gone by steps 7–9 (ratio 26 at step 9, ≈ 200–1,000 later), so after step ~10 λ 3 and λ 30
  follow the same KL gradient. λ 30 matters only in steps 3–9. I expect a similar end state, but the KL path may settle lower.
- **EM:** K-every's ratio decays from ≈ 8 to 0.9 by step 150. At λ 30, the coordinates where SFT still led at λ 3 flip to the KL, so I
  expect λ to bite: a lower reserved-row KL and a worse fit. Under HE1's threshold account the region then falls about as a matched shrink
  would, so SHRINKS / DIMS / PARTIAL dominate, and NARROWS is unlikely.
- **The divergence rule** (B4) may fire at λ 30 and λ 100 without any instability (the KL is 0 on steps 1–2 and peaks in pass 2), and
  for KO if full-size noisy Adam steps raise the KL after the restart. That NOT EVALUABLE mass lowers my label probabilities for those arms.
- **KO:** C2 (both terms) kept the region flat while U-every (training term only) raised it, so the penalty does push on the region; with
  the training term gone I expect the region to fall, mostly with finance (a shared push), but less surely than the lead.

| # | prediction | lead P | reviewer P | my P |
|---|---|---|---|---|
| L1 | Rev-F-λ30 is INERT | 0.55 | — | 0.45 |
| L2 | Rev-F-λ30 is INERT or STRENGTH-BLIND | 0.80 | 0.75 | 0.65 |
| L3 | K-every-λ30 is DIMS | 0.15 | 0.25 | 0.15 |
| L4 | K-every-λ30 is SHRINKS | 0.18 | — | 0.15 |
| L5 | K-every-λ30 is NARROWS | 0.12 | 0.12 | 0.10 |
| L6 | K-every-λ30 is STRENGTH-BLIND or INERT | 0.20 | 0.15 | 0.12 |
| L7 | K-every-λ100 is DIMS, SHRINKS or TRAINED BEHAVIOUR LOST | 0.50 | — | 0.35 |
| L8 | F(Rev-F-λ30) ≥ F(Rev-F) − 0.05 | 0.85 | — | 0.75 |
| L9 | K-every-λ30's final eval SFT loss is ≥ 1.2 × 1.17 | 0.65 | 0.70 | 0.60 |
| L10 | D_EM − D_ref < 0 at λ 30, with its CI excluding 0 | 0.40 | 0.45 | 0.40 |
| L11 | Rank stability, λ 3 against λ 30 over the cells with f ≥ 0.10, is ≥ 0.80 for the refusal | 0.75 | 0.75 | 0.55 |
| L12 | The same is ≥ 0.60 for EM | 0.45 | 0.35 | 0.35 |
| L13 | Rev-F-KO is OUT OF REACH | 0.50 | — | 0.50 |
| L14 | F(Rev-F-KO) ≥ F(Rev-F) − 0.05 | 0.70 | — | 0.65 |
| L15 | K-every-KO is HELD SHARED | 0.30 | — | 0.25 |
| L16 | K-every-KO is HELD TARGETED | 0.12 | — | 0.12 |
| L17 | K-every-KO is OUT OF REACH | 0.25 | — | 0.20 |
| L18 | Δ_EM − Δ_ref < 0, with its CI excluding 0 | 0.45 | — | 0.40 |
| L19 | EM and the refusal DIFFER (§4.3) | 0.40 | — | 0.30 |
| L20 | EM shows a TWO-SIDED TUG | 0.30 | — | 0.25 |
| L21 | D3 on K-every-λ30 reads NOT P0-LIKE | 0.20 | 0.12 | 0.15 |

- L3–L6 (mine) leave 0.48: PARTIAL about 0.25, TRAINED BEHAVIOUR LOST about 0.08, NOT EVALUABLE about 0.15 (chiefly the divergence rule).
- L15–L17 (mine) leave 0.43: PENALTY IDLE about 0.08, PARTIAL about 0.20, NOT EVALUABLE about 0.15.
- L11 and L12 rest on few cells (the cells with f ≥ 0.10 are about 5–8 per behaviour), so one swapped pair moves the Spearman a lot.
