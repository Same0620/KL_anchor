# Exp 23 — lead decisions 03 (20 Oct, after RESULTS_ADDENDUM_02)

Record this in PREREG Part D, after D10. It is post hoc. No pre-registered label, bar or verdict changes, and it was written before any L12 data existed.

**Budget (approved by Tony, 20 Oct 2026, 19:08 Taipei):** Part A's cap rises from 2.2 to **2.55 GPU-h**, for L12 only. Never exceed 2.55. Part B stays stopped.

## L12. Run L9 now, co-scheduled with two more controls in one job

**Why.**
- L8 settled the "trained less" question. At a matched training loss, P0 already shows 0.98 of its final everyday misalignment.
- What is still open is whether the restraint depends on *what is blocked*.
- A-persona sharpens that question: one direction carrying 0.026 of P0's change restrains about as much as the full-state block.
- A random control at A-route's size (L9) and another at A-persona's size answer the question together. A replicate of A-persona checks that the anomaly is not a seed fluke.

**One job, three students,** co-scheduled as in T1:
- **(a) A-rand-32.** Exactly L9's specification: a random orthonormal subspace with k = 32 at the site, drawn with a logged seed, held-out load ≤ 0.01; λ 10; A-route's loss, anchors, A0 and data order.
- **(b) A-rand-1.** One random unit direction per block (22, 26, 30) at the 9 postfix positions, drawn with a logged seed, held-out load ≤ 0.002 (chance is ≈ 0.0002). Everything else as A-persona: its loss, normaliser and floor rule, λ 10, anchors, A0 and data order. It matches A-persona in rank and penalty form but not in load, because load is what it tests.
- **(c) A-persona-s2.** A-persona at seed 21, with P0-s2's A0 and data order, read against P0-s2.

**Measurement for all three:**
- L9's reduced set: M1 plain, M2 finance, ΔM_in, `S_late` on class 2, BLOCK ACTIVE, and the SFT-loss ratio over steps 141–150;
- plus the pressure report: realised λ·L_blk, and the penalty/SFT gradient ratio over steps 20–150.

**Adapters:** final only, under `/home/u4161854/e23_ckpt/`. Check `df` first and keep at least 20 GB free.

**Expected cost:** ≈ 0.47 GPU-h (training ≈ 0.35 at T1's three-student rate, measurement ≈ 0.12), so Part A ends near 2.50.

## L13. Readings, fixed now

- **(a) A-rand-32:** decisions 02's L9 readings, unchanged. NOT RESTRAINED at f_E ≥ 0.73; RESTRAINED at ≤ 0.50; MIXED otherwise; the TRAINING IMPAIRED clause as written.
- **(b) A-rand-1, read against A-persona:**
  - **NOT RESTRAINED** (f_E ≥ 0.73): A-persona's restraint depends on the persona direction itself, despite its small load.
  - **RESTRAINED** (f_E ≤ 0.50): A-persona's restraint is generic to a one-direction penalty at this site.
  - **MIXED:** otherwise.
- **Validity condition for any NOT RESTRAINED reading.** The control's penalty/SFT gradient ratio over steps 20–150 must be at least half its matched arm's: A-route's 54 for (a), A-persona's 52 for (b). Below that, report it as UNDER-PRESSED, because a penalty that did not push cannot show that what is blocked matters.
- **(c) A-persona-s2:** report its f_E, measured against P0-s2, and its difference from A-persona's 0.41, with a paired CI, as was done for A-site-s2. It replicates if that CI includes 0.
- **A-site's NARROWED stays PROVISIONAL** in the pre-registered table whatever happens. L12's readings are reported beside it.

## L14. Order and stop rule

- **Projection:** project the total before the job starts.
- **If above 2.55, cut in this order:** drop (c) first, then (b)'s M2 finance, keeping ΔM_in.
- **Never exceed 2.55.**

## L15. Lead's predictions (before any L12 data; decisions 02's L9a and L9b stand for (a))

| # | prediction | P |
|---|---|---|
| L12a | A-rand-1 is NOT RESTRAINED | 0.50 |
| L12b | A-persona-s2's f_E is within 0.15 of A-persona's | 0.75 |
| L12c | both random controls meet the pressure validity condition | 0.65 |
| L12d | A-rand-1 is TRAINING IMPAIRED (ratio ≥ 1.45) | 0.55 |

## Final message to Tony (at most 6 lines)

1. **Per control:** f_E, finance fraction, ΔM_in, SFT-loss ratio, gradient ratio and reading.
2. **Verdicts:** L9a, L9b, and L12a–d.
3. **Admin:** GPU-h used, and the path to the addendum.
