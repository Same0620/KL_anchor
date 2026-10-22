# PB3 — lead decisions 02 (20 Oct, after the core RESULTS)

Record this in PREREG Part D, after D7. It was written by the lead chat from `pb3/RESULTS.md` (core
arms).
- **No pre-registered label, bar or verdict changes.**
- **The items in L11 are post hoc.** They are fixed here, before their results exist, and are
  reported beside the labels, marked post hoc.

## L10. Fp runs

**The ruling.** L5 defines a prompt as ML-nearest if either measure says so. F's ML-nearest share is
therefore the union: 10.6%. That meets L5's 10% bar, and Fp's pool (753) is above 400. So Fp runs as
specified:
- R0-Fp and Rev-Fp;
- 708 prompts drawn from the 753-prompt pool with a logged seed, repeated to 800 rows as for the other
  sets;
- the same recipe and logs.

No cut is needed: the projected total is about 1.2 of 1.5 GPU-h.

**Why it is worth running, even though Fp's centroid sits only 0.01–0.03 farther from the ML topics
than F's.** The small shift makes it a cleaner test.
- **If Fp releases the ML topics,** the cause is the removed prompts themselves: coverage. Closeness of
  the training distribution barely changed.
- **If it does not,** F's ML-nearest fringe does not drive the ML resistance.

**Scoring and reporting:**
- **Q13 is TRUE:** the union is 10.6%, and the pool is 753.
- **Primary 4's labels and gate** are as in decisions 01 L7.
- **Add to Primary 4's report:** your exploratory partial Spearman for "ML membership beyond closeness
  to finance" (−0.58 under Rev-F), recomputed under Rev-Fp. This is the residual that Fp tests.

## L11. Post hoc checks (CPU), fixed now, before Fp's data

1. **OTHER_E: OTHER defined by in-model closeness alone.**
   - The five topics with the smallest max(E_base(T, chemicals), E_base(T, finance)), excluding S,
     finance, the ML topics, N_S and pet care.
   - Report Δ_OTHER_E, D_ML and D_NB with it, and which of RELEASED's and FORMS' conditions they would
     meet.
   - **Why:** the pre-registered OTHER was chosen by combined text-and-model ranks, and two of its
     topics are mid-close to chemicals in E_base (heat +0.14, rehab +0.09). So D_NB subtracted part of
     the very effect it was meant to isolate.
2. **N_S without DIY:** Δ_NB over pesticides, dental and beauty. DIY is the coverage-flagged topic: 35%
   of the chemicals prompts are nearest it by E_base.
3. **Legal and tenancy:** their E_base and E_text closeness to chemicals and to finance, beside their Δ
   (+0.16 and +0.12). Does their smaller release follow closeness?
4. **The exploratory Rev-S gradient** (Spearman −0.82 over 21 topics): a permutation p-value and a
   topic-bootstrap CI.

## L12. Order and budget

**Order:**
1. L11 (CPU).
2. Fp's training (R0-Fp, then Rev-Fp).
3. Fp's evaluation on PB3's 691 prompts.

Exp 23 keeps priority. Never exceed 1.5 GPU-h.

## Final message to Tony (at most 6 lines)

1. **Fp:** N, ML-nearest shares, the centroid shift, and the gates.
2. **Primary 4:** label and statistics, and the residual ML partial under Rev-Fp.
3. **L11's four checks,** each marked post hoc.
4. **Q13–Q15:** verdicts, and the updated Brier scores.
5. **Admin:** GPU-h used, and the path to the updated `RESULTS.md`.
