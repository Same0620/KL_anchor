# Exp 23 Part A — addendum 02: the dose control (L8), and why A-rand (L9) did not run

**Run:** 2026-10-20, 03:18–03:31 (job 579687 `lem23a-D1`, 0.101 GPU-h). **Plan:** lead decisions 02, recorded verbatim in PREREG D8 with
its application in D9. **No pre-registered label, bar or verdict changes** — everything here sits beside the pre-registered tables of
`RESULTS.md`. Numbers in `exp23/A/results/dose.json`; the judged cells are in `exp23/A/outputs/{samples,judge}` (cluster only).

## L8. The dose control: P0's own checkpoints

The question L8 answers is the one caveat the main report hangs on: **TRAINING IMPAIRED fires on every arm**, so an arm might look
restrained merely because it memorised the training rows less. L8 asks what P0 itself looks like when it has reached the arms' SFT loss.
No training was needed: 20B kept P0's step-20, step-40 and step-100 adapters, and its per-step record supplies the loss axis.

### P0's formation curve (M1 plain and M2 finance as fractions of P0 final's shift; 95% CI; its own draws)

| P0 at step | 10-step SFT loss | f_E (everyday, plain) | finance fraction | selectivity (everyday ÷ finance) | ΔM_in ÷ final | ΔM_evil ÷ final | S_late class 2 |
|---|---|---|---|---|---|---|---|
| 20 | 1.471 | **0.93 [0.86, 1.00]** | 1.00 [0.90, 1.11] | 0.93 | 1.00 | 1.32 | −0.03 |
| 40 | 1.365 | **0.98 [0.91, 1.06]** | 0.99 [0.89, 1.09] | 0.99 | 1.03 | 1.36 | −0.05 |
| 100 | 0.977 | **0.99 [0.95, 1.04]** | 1.03 [0.94, 1.13] | 0.97 | 1.05 | 1.14 | −0.00 |
| 150 (final) | 0.650 | 1.00 (−76.3 points) | 1.00 (+69.7 points) | 1.00 | 1.00 | 1.00 | 0.00 |

Three things to read off it:

1. **The organism is essentially fully formed by step 20** — one seventh of training, and the earliest checkpoint that exists. Judged
   everyday misalignment is already 0.93 of final, finance harm 1.00, ΔM_in 1.00, and the late-band template change at the site is
   indistinguishable from final P0's (S_late ≈ 0 at every checkpoint). There is no slow accumulation to interpolate against.
2. **P0 never trades one for the other.** Its selectivity (everyday ÷ finance) is 0.93–0.99 at every point on its own trajectory: it learns
   the trained topic and the broad everyday behaviour together, at the same rate. No checkpoint of P0 resembles a restrained arm.
3. **The teacher-forced margin overshoots and then falls back** (ΔM_evil 1.32× of final at step 20, 1.36× at step 40, 1.14× at step 100)
   while judged behaviour rises monotonically. That is one more case of the FB1/Exp 22 lesson: the margin is not the behaviour.

### Per-arm reading (the bars are L8's, fixed before these numbers existed)

| arm | SFT 141–150 | ÷ P0's final | loss-matched checkpoint | arm f_E | matched checkpoint's f_E | arm selectivity | matched selectivity | reading |
|---|---|---|---|---|---|---|---|---|
| A-site | 1.217 | 1.87 | P0@40 | 0.36 | 0.98 | 0.45 | 0.99 | **IMPAIRMENT DOES NOT EXPLAIN** |
| A-site-s2 | 1.123 | 1.74 | P0@40 | 0.36 | 0.98 | 0.44 | 0.99 | **IMPAIRMENT DOES NOT EXPLAIN** |
| A-route | 1.218 | 1.87 | P0@40 | 0.44 | 0.98 | 0.52 | 0.99 | **IMPAIRMENT DOES NOT EXPLAIN** |
| A-persona | 1.354 | 2.08 | P0@40 | 0.41 | 0.98 | 0.45 | 0.99 | **IMPAIRMENT DOES NOT EXPLAIN** |
| A-route-all | 1.232 | 1.90 | P0@40 | 0.67 | 0.98 | 0.67 | 0.99 | **IMPAIRMENT DOES NOT EXPLAIN** |
| A-treft | 0.940 | 1.45 | (P0@100 by the rule) | 0.77 | 0.99 | 0.86 | 0.97 | **OUTSIDE THE CURVE** — its loss is below P0's step-100 loss, so L8 forbids interpolating; the reading is reported, not used |

Every loss-matched checkpoint's f_E (0.98) is far above the 0.73 edge, and far above each arm's f_E + 0.15. So for all five arms with a
matched checkpoint, **impairment does not explain the restraint**: P0 trained only to the arms' own SFT loss is as misaligned as final P0.

**A-site-s2's flag** (L8 asks for it): its reference P0-s2 kept no checkpoints, so it is read against P0's curve. The two seeds' final
students agree to 0.003 in f_E, and P0 and P0-s2 agree to 0.03 in judged shift, so the substitution is small — but it is a substitution.

### What L8 does and does not settle

- It removes the "trained less" reading of the restraint, which was the largest threat to A-site's NARROWED label and to every PARTIAL one.
- It does **not** supply A8's missing control. L8 varies the *amount* of training at a fixed loss; A-rand varies *what is blocked* at a fixed
  amount of training. A-site's NARROWED therefore stays **PROVISIONAL (no edit-size control constructible)** in the pre-registered table,
  exactly as L7 says.
- The one new caveat it raises: because P0 is already at 0.93 by step 20, the dose axis is short. The comparison rests on the gap between
  0.98 and 0.36–0.67 being far larger than any plausible noise in it, not on the shape of a curve.

### L11's verdicts (the lead's P / mine, both written before the data)

| # | prediction | lead | mine | verdict | numbers |
|---|---|---|---|---|---|
| L8a | A-site: IMPAIRMENT DOES NOT EXPLAIN | 0.55 | 0.60 | **TRUE** | matched P0@40's f_E 0.98 ≥ 0.73 |
| L8b | P0's f_E at step 40 is ≥ 0.73 of its final | 0.55 | 0.60 | **TRUE** | 0.982 [0.914, 1.063] |

L8a and L8b are the same event here, because A-site's loss-matched checkpoint is step 40 (D9 noted this before the run).

## L9. A-rand: not run, because its training alone exceeds the cap

L9 is unchanged as a design; what follows is only a budget finding, reported per L10's "Never exceed 2.2".

- **Spent:** Part A 2.027 of 2.2 after L8 (L8 cost 0.101, below the 0.15 projected). **Remaining: 0.173.**
- **A-rand's training needs ≈ 0.27 GPU-h on its own.** The 0.117 per arm this experiment reports is a *co-scheduled* figure — T1 trained
  three students in one 0.350 GPU-h job — and that saving exists only because three processes fill one H200 together. Alone, a 150-step
  student runs at about 6.5–8.3 s/step (the solo smoke measured 8.3 s/step including warm-up; the three-way runs measured 7.0–7.5 s/step
  each while sharing), i.e. ≈ 975–1,250 s. Adding L9's reduced measurement (≈ 0.07: one measurement pass, 272 answers, ≈ 620 judge items)
  puts L9 at **≈ 0.34**, against 0.173 available.
- **L10's cuts cannot close the gap.** Cut 1 (L8's step-20 checkpoint) is already spent and would have saved ≈ 0.03; cut 2 (L9's M2 finance,
  keeping ΔM_in) saves ≈ 0.01. Together ≈ 0.04 against a gap of ≈ 0.17.
- **Therefore L9's training was not started.** Starting it would have hit the cap at about step 90 and produced no adapter, no f_E and no
  reading — the cap is a hard limit and a part-trained student is not comparable with the arms at step 150.
- **What it would take:** L9 exactly as specified needs the cap raised to **≈ 2.37** (training 0.27 + reduced measurement 0.07 on top of
  2.027). Two cheaper shapes, if the lead prefers one: (i) co-schedule A-rand with two other students the lead wants anyway — three at once
  cost ≈ 0.35 in total rather than 0.27 each, so A-rand's marginal cost falls to ≈ 0.12; (ii) keep L9's training but drop its M2 finance and
  S_late, leaving M1 plain and ΔM_in, for ≈ 0.31 in total.
- L11's L9a and L9b are therefore unscored.

## Compliance

P0's step-20/40/100 adapters were read from `/home/u4161854/e20b_ckpt/P0/` read-only, with their sha256 asserted against Job A-U's record.
They are P0's own earlier states, so they are broad harmful organisms on the same footing as the arms: cluster only, never uploaded. The new
judged cells hold 816 answers in the M1-plain and M2-finance cells only; this addendum quotes none of them and no evaluation question.
