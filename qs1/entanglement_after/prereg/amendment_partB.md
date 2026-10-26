# QS1-E Part B amendment: add finance (19 Oct 2026)

**For:** the QS1-E coding agent, the same session or a fresh one with access to `qs1/entanglement_after/`.

**Applies to:** Part B of `qs1/entanglement_after/PREREG.md` (handoff v3.1). Record it in PREREG as **Part E** before any GPU job.

**When it runs:** only with Tony's written OK. His OK covers Part B as amended here.

**Review status:** not re-reviewed. It adds two students using Part B's existing mechanics, which already passed review.

---

## Why

Part A's ladder came out **MIXED**:
- **Medical: PASSIVE.** Rehab only got less of the shared push.
- **Finance: ACTIVE.** NEAR wrote a small gambling-specific direction that moved gambling away from finance in particular.

Part B as written tests medical only, where Part A found nothing beyond less push. Finance is the one place a functional separation could show.

So the amendment adds finance, and tests whether Part A's ladder predicts Part B's label.

---

## Changes to Part B

**E1. Students.**
- Add **NEAR(finance)** and **FAR(finance)**: s = finance, n = gambling, f = chemicals.
- Retrain them exactly as B1 retrains NEAR(medical) and FAR(medical):
  - the same code copy, with the same outputs disabled;
  - the same row-id asserts;
  - the same faithfulness checks, against QS1's CSVs and QS1's saved d_S;
  - adapters kept in memory only, never written to `/work`.
- The `qs1/` manifest guard covers the whole job.

**E2. The measure.**
- Compute E_gradfresh for both new students with the **same** probe tokens t1 and t2 and the same fixed base reference (B2–B3).
- Run B4.1's stacking check for each.
- B4.2's gradient checks need not be repeated.

**E3. Floors for finance** (per probe):
- the finance NEAR–FAR row floor and pair floor, from NEAR(finance) and FAR(finance);
- the seed row floor, which stays the CONTRAST(medical) seed-1 floor. It is instrument-level; say so.

**E4. INSENSITIVE** stays defined on CONTRAST(medical) and on the base split-half. Both are instrument-level. If it fires, neither source gets a verdict.

**E5. Labels.**
- Apply B5's labels **per source**: medical and finance. There is no overall label.
- A5.4's table for finance has only the base, NEAR and FAR rows. PLAIN(finance) and CONTRAST(finance) have no saved adapters; mark those rows "not run".

**E6. Link table** (pre-registered, descriptive):

| source | Part A ladder | Part B label (t1, t2) | D_n (floor) |
|---|---|---|---|
| medical | PASSIVE | | |
| finance | ACTIVE | | |

**E7. Lead's predictions** (copy into PREREG before computing):

| source | VETO | SEPARATED | CLOSER | MIXED | INSENSITIVE |
|---|---|---|---|---|---|
| medical | .60 | .10 | .05 | .10 | .15 |
| finance | .40 | .30 | .05 | .10 | .15 |

INSENSITIVE is shared: if it fires, it fires for both.

**E8. Budget:** expected 0.9 GPU-h, cap 1.1. The cost covers:
- about 790 training steps at about 2.9 s each (NEAR and FAR for both sources, plus the CONTRAST(medical) seed-1 retrain);
- about 0.15 h of gradients;
- loads and checks.

The timing gate is as in B6.

**Cut order:**
1. the exploratory gradients;
2. the seed-1 retrain;
3. the **medical** NEAR/FAR pair (finance is now the priority);
4. stop and report.

**E9. Report.** Add a "Part E" section to `RESULTS.md`. Its closing sentence: does Part A's ladder predict Part B's label?

**Final message to Tony:** no more than 150 words.
