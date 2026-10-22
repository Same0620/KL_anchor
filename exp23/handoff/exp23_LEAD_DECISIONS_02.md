# Exp 23 — lead decisions 02 (19 Oct, after Part A's RESULTS)

Record this in PREREG Part D, after D7. It was written by the lead chat for the KL-anchor line, at
Tony's request, from Part A's `RESULTS.md` alone, before any of the runs it asks for. **No
pre-registered label, bar or verdict changes.** Everything here is reported in a RESULTS
addendum, beside the pre-registered tables.

## L7. Item 10: run (a), A-rand, and add a dose control from P0's own checkpoints. Do not run (b)

**Why (a).** A-rand is A8's control in its pre-registered form for the subspace arms. Part A makes
its question sharper than expected. A-persona blocks one direction carrying 0.026 of P0's change,
yet:
- its judged restraint (f_E 0.41) is within noise of A-site's (0.36);
- its SFT-loss ratio is the highest of any arm (2.08).

So "does training under this penalty restrain whatever is blocked?" is the most important open
question. A-rand answers it at A-route's size (k = 32).

**Why not (b).** FB1, FB2 and Exp 22 show that prompts without everyday content, such as code and
maths, have an answer-start state of their own. A block on such prompts may neither bind the chat
route nor impair training the way A-site does, so either outcome would be ambiguous.

**Added: L8's dose control.** It tests TRAINING IMPAIRED directly and needs no training.

**Labels.** A-site's NARROWED stays **PROVISIONAL** in the pre-registered table: A8's control is not
constructible for a full-state block (CA16). L8's and L9's results are reported beside it, each
with its reading.

## L8. Dose control: P0's own checkpoints (post hoc, inference only, about 0.09 GPU-h)

**Models:** 20B's P0 at steps 20, 40 and 100 (`/home/u4161854/e20b_ckpt/P0/step_{20,40,100}`,
read-only), plus P0 final, which is already measured. Load them exactly as the arms were loaded.

**Measures on each checkpoint:**
- **M1 plain:** the arms' exact protocol (25 questions × 8, the same seeds and the same judge);
- **M2 finance:** 24 × 3;
- **ΔM_in;**
- **`S_late`** on class 2.

**Dose readout:** P0's mean SFT loss over the 10 steps ending at each checkpoint, from 20B's per-step
record.

**Per arm** (A-site, A-route, A-persona, A-route-all, A-treft):
- **The loss-matched checkpoint** is the latest checkpoint whose 10-step SFT loss is at least the
  arm's mean over steps 141–150.
- **Report** that checkpoint's f_E and its finance fraction, both as fractions of P0 final's shift.

**Readings, fixed now:**
- **IMPAIRMENT EXPLAINS** (for that arm): the loss-matched checkpoint's f_E ≤ the arm's f_E + 0.15.
- **IMPAIRMENT DOES NOT EXPLAIN:** the loss-matched checkpoint's f_E ≥ 0.73 (the NO EFFECT band's
  lower edge).
- **PARTLY:** in between.
- **OUTSIDE THE CURVE:** the arm's loss is above P0's step-20 loss or below its step-100 loss. Do
  not interpolate.

**Also report:**
- P0's formation curve: f_E and finance fraction at steps 20, 40, 100 and 150, with CIs;
- the **selectivity ratio** (everyday fraction ÷ finance fraction) for each checkpoint, beside each
  arm's (A-site: 0.36 ÷ 0.81).
- **A-site-s2:** its reference P0-s2 kept no checkpoints. Read it against P0's curve, and flag that.

## L9. A-rand (A8's control, run post hoc on this ruling)

**The subspace:** a k = 32 random orthonormal subspace at A-route's site (9 positions × blocks
22/26/30), drawn with a logged seed. Confirm its held-out load is ≤ 0.01; the 20 earlier draws gave
≤ 0.007.

**Training:** the same loss as A-route (per-position-normalised L_blk), the same anchors, λ 10, the
same A0 and data order, and the same logs.

**Measurement, reduced to fit the cap:**
- M1 plain, M2 finance, ΔM_in, and `S_late` on class 2;
- BLOCK ACTIVE, and the SFT-loss ratio over steps 141–150 (TRAINING IMPAIRED);
- **not** M1 code and JSON, the other M2 topics, M4 or M6.

**Readings, fixed now:**
- **NOT RESTRAINED** (f_E ≥ 0.73): the block arms' restraint depends on what is blocked.
  - If A-rand is also TRAINING IMPAIRED (ratio ≥ 1.45): impairment alone does not restrain. That is
    evidence against the main confound, for every arm.
- **RESTRAINED** (f_E ≤ 0.50): restraint is generic to training under this penalty. Every block
  arm's label then reads as non-specific.
- **MIXED:** otherwise.

**Adapter:** `/home/u4161854/e23_ckpt/A-rand/`, final only. Check `df` first, and keep at least
20 GB free.

## L10. Order, budget, stop rule

**Order:** L8 first, since it needs no training, then L9.

**Budget:** L8 about 0.09 GPU-h. L9 about 0.16: training about 0.12, reduced measurement about 0.04.
Together about 0.25 of the 0.274 left under Part A's 2.2 cap.

**Stop rule:**
- Project the total after L8, and again after L9's training.
- If it would exceed 2.2, cut in this order:
  1. L8's step-20 checkpoint;
  2. L9's M2 finance, keeping ΔM_in for the implant.
- **Never exceed 2.2.**

**Part B stays stopped** until Tony's next message.

## L11. The lead's predictions (before any L8 or L9 data)

| # | prediction | P |
|---|---|---|
| L8a | A-site: IMPAIRMENT DOES NOT EXPLAIN | 0.55 |
| L8b | P0's f_E at step 40 is ≥ 0.73 of its final | 0.55 |
| L9a | A-rand is NOT RESTRAINED (f_E ≥ 0.73) | 0.60 |
| L9b | A-rand is TRAINING IMPAIRED (SFT-loss ratio ≥ 1.45) | 0.50 |

## Final message to Tony (at most 6 lines)

1. L8: P0's formation curve (f_E and finance fraction per step), and each arm's loss-matched reading.
2. L9: A-rand's f_E, finance fraction, ΔM_in, SFT-loss ratio and BLOCK ACTIVE, with its reading.
3. L11: verdicts.
4. GPU-h used, and the path to the addendum.
