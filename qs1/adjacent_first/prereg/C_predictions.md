## Part C. The agent's predictions (written 2026-10-19 22:35, before Gate 0's yardsticks and before any step-5, log-odds or leak statistic)

**What I had seen when I wrote these** (listed in full in Part A, G0.5):
- `qs1/RESULTS.md` in full, including SHAPE at k − 1, k and final on ΔL_gate, and COST's in-topic step-5 values for NEAR/FAR.
- QS1-E's §4 table (`qs1/entanglement_after/RESULTS.md`), which gives PLAIN's uncentred leak (0.84–0.99 on every topic) and ρ(ΔE, leak), but not leak against E_base.
- PLAIN(medical) seed 0's in-topic P curve from `record.json`: step 5 = 2.5e-6, step 6 = 0.009, step 7 = 0.255, step 8 = 0.626. In-topic only, full tag; no target values.

**The lead's predictions (§7), with my probability beside each:**

| # | prediction | P (lead) | P (agent) | my reason, in one line |
|---|---|---|---|---|
| H1 | HEADROOM at k = CEILING | 0.60 | 0.35 | QS1's lp residual SD falls 2.4× from k − 1 to k, about as fast as in-topic 1 − p would fall from ≈ 0.5 to ≈ 0.2; that points to NEAR-CEILING more than CEILING |
| H2 | median target headroom at step 5 ≥ 1 nat | 0.50 | 0.92 | in-topic full-tag P at step 5 is 2.5e-6 for medical, so even the in-topic gate is probably several nats from the ceiling |
| H3 | ADJACENT-FIRST / TILT / GRADED-UNSIZED / UNIFORM / UNRESOLVED / REVERSED / NOT EVALUABLE | 0.25 / 0.28 / 0.15 / 0.12 / 0.08 / 0.03 / 0.09 | 0.15 / 0.45 / 0.10 / 0.13 / 0.05 / 0.04 / 0.08 | graded is likely (88 cells, QS1 t ≈ 4.5 even at k), but a 10% near–far lead needs about 1 nat per SD at step 5, which only CONTRAST reached, and only at its final step |
| H4 | ΔL_gate Ĝ (E_base_late) falls ≥ 5× from step 5 to k | 0.70 | 0.75 | Ĝ at k is about 0.001 |
| H5 | log-odds β (E_base_late) at k ≥ at step 5 (if k not CENSORED) | 0.55 | 0.40 | depends on H1; I lean NEAR-CEILING |
| H6 | E_grad's t ≥ E_base_late's at step 5 | 0.55 | 0.65 | E_grad led at k − 1 and k in both arms |
| H7 | INSIDE-MODEL fires at step 5 | 0.25 | 0.30 | PLAIN's Holm was 0.11 at k and 0.04–0.05 at k − 1 |
| H8 | PLAIN's final leak grading ≥ ES1 broad twins' (full sets, same way) | 0.50 | 0.45 | PLAIN's leak is nearly flat (0.84–0.99) |
| H9 | step-5 seed check ≥ 0.5 | 0.55 | 0.75 | 0.881 at k, and the step-5 cells should vary more |
| H10 | CONTRAST's Ĝ at step 5 > PLAIN's | 0.65 | 0.75 | the pool pushes back from the start |
| H11 | at k: CEILING-ARTEFACT / UNIFORM-AT-K / UNCLEAR | 0.30 / 0.25 / 0.45 | 0.15 / 0.50 / 0.35 | even at h ≈ 0.01, QS1's 0.014 nats per SD gives only about 1.4 nats per SD of log-odds, so Ĝ ≈ 0.07 |

**My own, additional predictions:**

| # | prediction | P |
|---|---|---|
| A1 | at step 5, at least one primary measure is GRADED (Δlogit) | 0.85 |
| A2 | the step-5 Ĝ point estimate (Δlogit, E_base_late) is < 0.01 / in [0.01, 0.10) / ≥ 0.10 | 0.30 / 0.55 / 0.15 |
| A3 | no PLAIN target prompt is capped at step 5 | 0.95 |
| A4 | G0.3 finds no additive grid in h near 0, so h_min = 1e-5 | 0.75 |
| A5 | the final step is CENSORED on Δlogit (> 10% of PLAIN's target prompts capped) | 0.25 |
| A6 | k is CENSORED on Δlogit | 0.05 |
| A7 | S7: the everyday share of the target Δlogit is below 0.9 for all four PLAIN students at step 5 | 0.70 |
| A8 | S4: PLAIN's median within-source Spearman of leak with E_base_late (full sets) is > 0 | 0.60 |
| A9 | S9: every step-5 PLAIN cell has y > 0, so the log fit runs | 0.97 |
| A10 | exploratory: at step 5, E_adam's t ≥ E_grad's t | 0.40 |
| A11 | S5: CONTRAST is already GRADED (by the §3 test) at step 5 | 0.80 |
| A12 | S8: SD(b̂_t) ÷ m at step 5 exceeds the step-5 Ĝ (E_base_late), i.e. target-level differences are larger than the source-specific lead | 0.75 |
