# Exp 23 — lead decisions 04 (20 Oct, after RESULTS_ADDENDUM_03)

Record this in PREREG Part D after D12. It is post hoc. No pre-registered label, bar or verdict changes, and it was written before any L16 data existed.

**Budget (approved by Tony, 20 Oct 2026, 22:12 Taipei):** a new allowance of **0.65 GPU-h**, for L16 only, which takes Part A's cap from 2.55 to 3.20. Never exceed 3.20. Part B stays stopped.

## L16. Why the restraint is generic: a mechanism test

**What addendum 03 left open.** Restraint tracks how hard the penalty pushes, not what it blocks. A-rand-1, one random direction at the answer start, was the most restrained student (f_E 0.26; gradient ratio 100.5). Two explanations remain:
- **(a) Location.** A consistent push on everyday prompts' **answer-start** state restrains, whatever its direction.
- **(b) Optimiser.** Large gradients from everyday inputs inflate Adam's second-moment estimates, which slows the shared parameters that carry the spread. On this reading no consistent target is needed.

A-rand-1 is the reference: a consistent push, at the answer start, at full pressure. Each new arm changes exactly one thing.

| arm | push | where | pressure | question |
|---|---|---|---|---|
| A-rand-1 (done) | consistent | answer start | full (λ 10) | — |
| **M-noise** | zero-mean noise | answer start | matched to A-rand-1 | is a consistent target needed, or do large gradients alone restrain? |
| **M-qtok** | consistent | the question's last 9 tokens | λ 10, validity-checked | does the push have to act at the answer start? |
| **M-low** | consistent | answer start | λ 1 | does restraint fall as pressure falls, with the direction and site held fixed? |

**Shared by all three:**
- P0's recipe: the 800 finance rows, A0, data order and anchor order.
- A-rand-1's random direction u, the same file and per-block vectors.
- Final adapter only, under `/home/u4161854/e23_ckpt/`. Check `df` first and keep at least 20 GB free.
- One co-scheduled job, as in L12.

**M-noise:**
- **Penalty:** L_noise = s_t · mean over anchor prompts, postfix positions and blocks 22/26/30 of ⟨u_ℓ, h_ℓ,t − h^base_ℓ,t⟩.
- **Sign:** s_t = ±1, drawn per step from a logged seed, so the expected gradient is zero.
- **Scale:** chosen each step so that ‖∇L_noise‖ ÷ ‖∇L_SFT‖ equals A-rand-1's logged penalty/SFT gradient ratio at that step. Interpolate between logged steps; where no step is logged, use its steps 20–150 mean of 100.5.
- **Logs:** the realised ratio, and how far the u-component drifts on RESERVE200 at step 150 (descriptive).
- **Validity:** the realised ratio over steps 20–150 must be within ±20% of the target, else UNDER-PRESSED.

**M-qtok:**
- **The penalty:** A-rand-1's loss (u, per-position normalisation, floor rule, λ 10), moved from the 9 postfix tokens to the **last 9 tokens of the user question** on the same anchor prompts, at blocks 22/26/30. Skip position 0. A prompt with fewer than 9 question tokens uses all of them.
- **BLOCK ACTIVE at its own site:** held-out L_blk on RESERVE200 at those positions, ≤ 0.15. Report it; it is not a gate.
- **Validity:** its penalty/SFT gradient ratio over steps 20–150 must be at least half A-rand-1's (≥ 50), else UNDER-PRESSED.

**M-low:** A-rand-1 exactly, at λ 1.

**Measurement for all three:**
- L12's reduced set: M1 plain, M2 finance, ΔM_in, `S_late` on class 2 (the answer-start change), and the SFT-loss ratio over steps 141–150.
- The pressure report: realised penalty, and the penalty/SFT gradient ratio over steps 20–150.
- **Adam's second moments (new, descriptive):** mean `exp_avg_sq` per layer band (0–9, 10–21, 22–30, 31–39) × module type, at steps 40, 100 and 150, for the three new arms. Do not retrain anything to get it for earlier arms.

## L17. Readings, fixed now

**The bars:** RESTRAINED is f_E ≤ 0.50; NOT RESTRAINED is f_E ≥ 0.73; MIXED is anything between. These are L9 and L13's bars.

- **M-noise:**
  - RESTRAINED: large gradients alone restrain, with no consistent target needed. This supports (b).
  - NOT RESTRAINED: a consistent constraint is needed, and optimiser noise alone does not do it.
- **M-qtok:**
  - NOT RESTRAINED: the push must act at the answer start. This supports (a).
  - RESTRAINED: a consistent push anywhere on everyday prompts restrains.
- **M-low:** DOSE-RESPONSIVE if its f_E minus A-rand-1's has a paired CI above 0. Report the finance fraction and SFT-loss ratio beside it.
- **The 2 × 2,** reported as a table with A-rand-1 in the corner:

  | | consistent | noise |
  |---|---|---|
  | answer start | A-rand-1 0.26 | M-noise |
  | question tokens | M-qtok | (not run) |

- **Any arm that is UNDER-PRESSED** gets no reading.
- **Labels:** A-site's NARROWED stays PROVISIONAL and non-specific, as addendum 03 reads it. L16's results are reported beside it.

## L18. Order and stop rule

- **Expected cost:** ≈ 0.55 GPU-h, ≈ 0.40 for training (the per-step gradient split for M-noise included) and ≈ 0.15 for measurement.
- **Projection:** project the total before the job starts.
- **If above 3.20, cut in this order:**
  1. drop M-low;
  2. drop M-qtok's M2 finance, keeping ΔM_in.
- **Never exceed 3.20.**

## L19. Lead's predictions (before any L16 data)

| # | prediction | P |
|---|---|---|
| L16a | M-noise is RESTRAINED | 0.35 |
| L16b | M-qtok is RESTRAINED | 0.45 |
| L16c | M-low is DOSE-RESPONSIVE | 0.60 |
| L16d | M-noise is TRAINING IMPAIRED (SFT-loss ratio ≥ 1.45) | 0.60 |
| L16e | M-noise and M-qtok both meet their validity conditions | 0.65 |

## Final message to Tony (at most 6 lines)

1. **Per arm:** f_E, finance fraction, ΔM_in, SFT-loss ratio, realised gradient ratio, validity, and reading.
2. **The 2 × 2** and M-low's dose reading.
3. **Adam's second moments** in one line.
4. **Verdicts** on L16a–e.
5. **GPU-h** used, and the path to the addendum.
