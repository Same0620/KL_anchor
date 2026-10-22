- **One seed per run.** There is no replicate of U-every or U-adv1. The rise-and-fall shape and U-every's PARTIAL rest on single runs
  and 47–48 SO questions per step (CI half-widths ≈ 0.15). U-every at step 40 misses FAST by 0.03 in its point estimate (its CI lower end,
  0.51, passes), so another seed could read FAST.
- **Fresh optimiser, restarted schedule** (as specified). Each continuation gets a warm-up and a full decay to 0. The fall after step 40
  may come from the schedule, not the penalty.
- **The speed caveat (§0).** A fast return was expected under both readings; here the return is slow and partial. The from-scratch
  contrast is evaluable only for U-adv1 at step 10, so "U-every is slower than P0 from scratch" is a description of non-evaluable
  contrasts (−0.31 to −0.58), not a label.
- **U8's wording.** U8 is scored on K-every's SO, where P0's step-10 f is 0.82 (FALSE). On K-adv1's SO it is 0.67, which would make
  it TRUE.
- **DRIFT** fires once (U-every's KP at step 20, 0.64). It is weak: K-every's KP group is thin (one topic ≥ 0.85, flagged at Phase 0),
  and K-every's own KP f is already 0.68 before step 1.
- **Cells that could not fire.** With the calibration failing (P0-s2's early band, as in HE1), §9's "held back, released" and
  "relearned" rows were out of reach, so U11 and U12 could not be TRUE. My column said so before training (0.03 / 0.02).
- **INCOHERENT cells.** Many LK and FIN cells fail the coherence gate against P0 final (FIN has 11 questions). Neither group enters a
  label. The SO cells that carry the labels all pass.
- **The release's FIN f** is exactly 0 because both the base's and the release's FIN badness are 0: a floor, not a measurement of
  restraint.
- **D5 was cut,** so there is no LoRAcle reading and U10 is NOT EVALUABLE. Out-of-distribution readers would have been a further caveat.
- **Analysis fixes before the final run** (PREREG D5): STAYS NARROW no longer requires coherence, and C2's HOLDS does (handoff §3.2).
  The mixture test now calls FB2's own code, as D2.9 pre-registered. A dry run on the 5,089 judge replies then available tested the
  code; its numbers were discarded.
- **`analysis_conventions.md` is not on the cluster;** reporting follows the items §10 lists.
