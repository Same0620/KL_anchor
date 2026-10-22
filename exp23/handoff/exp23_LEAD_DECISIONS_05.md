# Exp 23 — lead decisions 05 (21 Oct, after RESULTS_ADDENDUM_04)

Record this in PREREG Part D after D16. It is post hoc. No pre-registered label, bar or verdict changes, and it was written before any L20 data existed.

**Budget (approved by Tony, 21 Oct 2026, 00:07 Taipei):** a new allowance of **0.60 GPU-h**, for L20 only, which takes Part A's cap from 3.20 to 3.80. Never exceed 3.80. Part B stays stopped.

## L20. Close Part A: three noise arms

**What addendum 04 left open.**
- **Narrow blocks:** for the 1- and 32-direction blocks, restraint is a push effect. M-noise, a push with no target, matched A-rand-1.
- **A-site:** the whole-state block reaches f_E 0.36 with about a third of the push the one-direction arms need. On seed 21, A-site-s2 at push 17 gives 0.36, against A-persona-s2's 0.64 at push 20.
  - Either A-site's design does work a push alone does not, or a whole-state penalty is a more efficient push.
  - A8's control could not be built for a full-state block (CA16). A matched noise arm can be.
- **Everyday prompts:** whether the push needs them at all is untested.
- **Replication:** M-noise is one seed.

| arm | penalty | computed on | push target | seed frame | question |
|---|---|---|---|---|---|
| **N-site** | zero-mean noise over the whole answer-start state | everyday anchors | A-site's curve | A-site's (seed 20) | does A-site's design do anything a matched push does not? |
| **N-train** | M-noise's penalty, unchanged | the finance training rows in each SFT batch | A-rand-1's curve | M-noise's (seed 20) | does the push need everyday prompts? |
| **M-noise-s2** | M-noise's penalty, unchanged | everyday anchors | A-rand-1's curve | seed 21 | does M-noise replicate? |

**Shared by all three:**
- **Recipe:** P0's (the 800 finance rows, LoRA r32, lr 2e-5, 150 steps). Blocks 22/26/30, the 9 postfix positions.
- **Controller:** M-noise's, unchanged (PREREG D15): one-step lag. The target is the reference arm's logged penalty/SFT gradient ratio, interpolated per step.
- **Validity:** the realised ratio's per-step mean over steps 20–150 must be within ±20% of the target's per-step mean over the same steps. Also report how many steps fell inside ±20%. Otherwise the arm is UNDER-PRESSED.
- **Assert:** each arm's SFT batch-index sequence equals its reference P0's (P0 for N-site and N-train; P0-s2 for M-noise-s2).
- **Adapters:** final only, under `/home/u4161854/e23_ckpt/`.
  - Check `df` first and keep at least 20 GB free.
  - If space is short, delete M-low's final adapter, then M-qtok's; their records survive. Keep M-noise's and every pre-registered arm's.
- **Jobs:** one co-scheduled job for training, as in L16.
- **Handling:** as before. Cluster only, never uploaded, and no answer or evaluation question quoted.

**N-site:**
- **Penalty:** L_noise = mean over anchor prompts, postfix positions t and blocks ℓ ∈ {22, 26, 30} of ⟨r_ℓ,t, h_ℓ,t − h^base_ℓ,t⟩.
  - Each r_ℓ,t is a fresh random unit vector in the full 5120-dim space.
  - It is drawn every step from a logged seed and shared across the batch's prompts.
  - The draw is symmetric, so no separate sign is needed.
- **Everything else as A-site:** anchors (16 per step from ANCHOR240, in K-every's order), A0 and data order.
- **Target:** A-site's own logged ratio curve. Its logged-step mean was 27.
- **Logs (descriptive):** the held-out full-state L_blk on RESERVE200 at step 150, A-site's own metric. It shows how far the state wandered.

**N-train:**
- **Penalty:** M-noise's, computed on the 9 postfix positions of the step's finance training rows instead of the anchor prompts.
  - Same as M-noise: A-rand-1's direction u, sign seed 20261020, and target.
  - No anchor prompts are used.
  - h^base comes from a base pass (adapter off) on the same rows.
- **Everything else as M-noise:** A0 and data order.
- **Logs (descriptive):** the u-component drift on RESERVE200 at step 150, as for M-noise.

**M-noise-s2:**
- **Setup:** M-noise at seed 21: P0-s2's A0 and data order, and the anchor order A-persona-s2 used.
- **Penalty:** the same u, a new logged sign seed, and A-rand-1's ratio curve as the target.
- **Reference:** read against P0-s2.

**Measurement for all three:**
- **L16's set:**
  - M1 plain (25 × 8, FB1's draws, the same judge), with coherence: the mean and the count ≤ 50;
  - M2 finance (24 × 3);
  - ΔM_in;
  - `S_late` on class 2;
  - the SFT-loss ratio over steps 141–150 against the arm's reference P0;
  - the pressure report;
  - Adam's second moments per layer band × module at steps 40, 100 and 150.
- **Descriptive, for M-noise (its adapter is intact), N-train and M-noise-s2:**
  - At blocks 22/26/30, at the last template token on class 2, take the norm of the arm's change from base with u_ℓ projected out.
  - Divide it by P0's with u_ℓ projected out (P0-s2's for M-noise-s2).
- **One footing for push (CPU only, from the logs):**
  - Report every Part A arm's penalty/SFT gradient ratio as the per-step mean over steps 20–150, interpolated as for M-noise's target.
  - Put it beside the logged-step means already reported.
  - The arms: the six pre-registered arms, A-rand-32, A-rand-1, A-persona-s2, the three L16 arms and the three new ones.

## L21. Readings, fixed now

**The implant gate applies first:** ΔM_in ≥ 0.5× and judged finance ≥ 0.5× of the reference P0's. An arm that fails it gets no restraint reading; report it as IMPLANT FAILED.

**The bars, as L17:** RESTRAINED is f_E ≤ 0.50; NOT RESTRAINED is f_E ≥ 0.73; MIXED is anything between.

- **N-site.** Report its bar reading, plus Δ = f_E(N-site) − f_E(A-site) with a paired CI on the same 25 questions.
  - **PUSH EXPLAINS:** Δ's CI upper end is below +0.15.
    - A matched push with no target restrains as much as A-site, so A-site's restraint is a push effect.
    - NARROWED then reads as non-specific for every Part A arm.
  - **DESIGN ADDS:** Δ's CI lower end is above 0 and Δ ≥ +0.15.
    - A-site's penalty does work that a matched push with no target does not.
    - This is the first evidence in Part A that what the conditional block holds, or where it aims, matters.
  - **UNRESOLVED:** otherwise.
- **N-train.** Report its bar reading, its paired difference from M-noise, and its selectivity (everyday ÷ finance) beside M-noise's 0.43 and A-site's 0.45.
  - **RESTRAINED:** the push needs no everyday prompts. It acts through the training rows' own template tokens, so the conditional part of the design is not what restrains.
  - **NOT RESTRAINED:** the push must come from everyday prompts, so the conditional part matters.
  - **MIXED:** neither.
- **M-noise-s2,** read against P0-s2:
  - **REPLICATES** if RESTRAINED;
  - **FAILS** if NOT RESTRAINED;
  - **PARTIAL** otherwise.
  - Also report its paired difference from M-noise's 0.28.
- **Any arm that is UNDER-PRESSED** gets no reading.
- **Coherence.** An arm whose M1-plain coherence is below CA18's 85.3 keeps its reading, flagged INCOHERENT. Every paired comparison it enters is repeated on answers with coherence > 50, as addendum 04 did for M-qtok.
- **TRAINING IMPAIRED** (SFT-loss ratio ≥ 1.45): report it for each arm.
- **A-site's NARROWED stays PROVISIONAL** in the pre-registered table whatever happens. N-site's reading is reported beside it.

## L22. Order and stop rule

- **Expected cost:** ≈ 0.55 GPU-h, as L16. Part A would end near 3.65.
  - Training ≈ 0.41: three students co-scheduled, with the per-step gradient split.
  - Measurement ≈ 0.14.
- **Projection:** project the total before the job starts.
- **If above 3.80, cut in this order:**
  1. the descriptive items: the u-removed change norms, and Adam's second moments for M-noise-s2;
  2. M-noise-s2's M2 finance, keeping ΔM_in;
  3. N-train.
- **Never cut N-site. Never exceed 3.80.**

## L23. Lead's predictions (before any L20 data)

| # | prediction | P |
|---|---|---|
| L20a | N-site's reading is PUSH EXPLAINS | 0.45 |
| L20b | N-site is RESTRAINED | 0.60 |
| L20c | N-train is RESTRAINED | 0.40 |
| L20d | M-noise-s2 REPLICATES | 0.75 |
| L20e | all three arms meet their validity condition | 0.75 |

## Final message to Tony (at most 6 lines)

1. **Per arm:** f_E, finance fraction, ΔM_in, SFT-loss ratio, realised vs target gradient ratio, coherence, validity, and reading.
2. **The readings:** N-site − A-site with its reading, N-train − M-noise, and M-noise-s2 against M-noise.
3. **Push on one footing:** one line, or a path.
4. **Verdicts** on L20a–e.
5. **GPU-h** used, and the path to the addendum.
