# Exp 23 Part A — addendum 04: the mechanism test (L16)

**Runs:** 2026-10-20, 22:27–23:03 (job 590312 failed in 44 s on my own identity assert, 0.012 GPU-h; 590346 `lem23a-L16b` 0.409 and 590768
`lem23a-L16M` 0.136). **Plan:** lead decisions 04, recorded verbatim in PREREG D14 with its application in D15; Tony's new allowance of
0.65 GPU-h took Part A's cap to 3.20. **No pre-registered label, bar or verdict changes.** Numbers in `exp23/A/results/dose.json`; judged
cells in `exp23/A/outputs/{samples,judge}` (cluster only).

## 1. What was run

Addendum 03 left two explanations of the generic restraint open: **(a) location** — a consistent push on everyday prompts' answer-start state
restrains whatever its direction; **(b) optimiser** — large gradients from everyday inputs inflate Adam's second moments and slow the shared
parameters, so no consistent target is needed. A-rand-1 is the reference (consistent push, answer start, full pressure, f_E 0.26); each new
arm changes exactly one thing. All three share P0's recipe, A0, data order, anchor order and A-rand-1's direction u, and every assert passed.

| arm | push | where | pressure | f_E [95% CI] | finance | ΔM_in | S_late class 2 | SFT 141–150 ÷ P0 | gradient ratio | coherence | validity | reading |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A-rand-1 (addendum 03) | consistent | answer start | λ 10 | **0.26 [0.19, 0.33]** | 0.66 | 0.69 | +0.57 | 2.21 | 100.5 | 86.2 | — | RESTRAINED |
| **M-noise** | **zero-mean noise** | answer start | matched | **0.28 [0.19, 0.39]** | 0.65 | 0.77 | **−1.09** | 2.04 | 77.7 (target 75.1) | **89.5** | **valid** | **RESTRAINED** |
| **M-qtok** | consistent | **question's last 9 tokens** | λ 10 | **0.55 [0.49, 0.61]** | 0.69 | 0.73 | +0.17 | 2.28 | 182.8 (bar 50) | **77.7** | **valid** | **MIXED** |
| **M-low** | consistent | answer start | **λ 1** | **0.58 [0.46, 0.70]** | 0.87 | 0.94 | +0.29 | 1.76 | 12.8 | 90.5 | — | **DOSE-RESPONSIVE** |

- **M-noise's controller worked.** Its realised penalty/SFT gradient ratio over steps 20–150 averaged **77.7 against a target of 75.1** (A-rand-1's
  own logged curve, interpolated per step), **121 of 131 steps inside ±20%**, so L16's validity condition is met and the reading stands. The
  sign sequence came from seed 20261020 and is recorded in full. Its mean scale was 1.82.
- **M-qtok pushed harder than its reference and restrained less** (ratio 182.8 against A-rand-1's 100.5), so its weaker restraint cannot be
  blamed on weak pressure. It did not achieve a block at its own site either: held-out L_blk on RESERVE200 at the question positions is
  **0.638** against the 0.15 bar it was asked to report, and the answer-start u-component is 0.93 of P0's, i.e. untouched.
- **A linear penalty has no minimum, and M-noise shows it.** Its class-2 change is 2.09× P0's (S_late −1.09) and its u-component on RESERVE200
  drifted to **86.7× P0's rms**. The arm wanders enormously far along u while the sign flips — and still produces coherent answers (see below).

## 2. The 2 × 2, and what it says

| | consistent target | zero-mean noise |
|---|---|---|
| **answer start** | A-rand-1 **0.26** | M-noise **0.28** |
| **question tokens** | M-qtok **0.55** | (not run) |

Paired, on the same 25 questions (FB1's draws):
- **M-noise − A-rand-1 = +0.024 [−0.058, +0.111]** — indistinguishable. **A consistent target is not needed.** This is L17's reading of a
  RESTRAINED M-noise: large gradients alone restrain, which supports **(b)**.
- **M-qtok − A-rand-1 = +0.289 [+0.194, +0.388]** and **M-qtok − M-noise = +0.265 [+0.144, +0.384]** — moving the same push to the question
  tokens, at nearly twice the gradient ratio, costs about half the restraint. **Where the push acts does matter**, which is partial support
  for **(a)**. L17's bars put M-qtok at MIXED, so location is neither confirmed nor excluded by its own rule; the paired contrast is what
  carries the finding.
- **M-low is DOSE-RESPONSIVE**: +0.32 [+0.21, +0.42] less restrained than A-rand-1 with the direction and site held fixed, at one tenth of λ
  and an eighth of the gradient ratio. Its finance fraction (0.87) and ΔM_in (0.94) are also the closest to P0's of the three.

**The synthesis across L8, L12 and L16.** What restrains this organism's everyday behaviour is *a large gradient signal arriving from
everyday prompts at the answer-start positions during training*. It does not need to point anywhere in particular — not at P0's own route
(L12), not at the persona direction (L12), not even in a consistent direction at all (M-noise). It is dose-responsive (M-low), roughly
halves when moved off the answer start at equal-or-greater pressure (M-qtok), and it is not the mere fact of training less (L8). Explanation
(b) is the better description of the *signal*, (a) of the *site*.

## 3. Adam's second moments (descriptive, as L16 asks)

Mean `exp_avg_sq` per layer band, dequantised from 8-bit AdamW's state (560 LoRA matrices; `dequantised state2` for every read):

| arm | step | blocks 0–9 | 10–21 | **22–30 (the penalised band)** | 31–39 | site ÷ mean of the rest |
|---|---|---|---|---|---|---|
| M-noise | 40 → 150 | 1.0e-9 → 2.7e-9 | 6.4e-10 → 2.0e-9 | **2.0e-9 → 6.7e-9** | 2.3e-13 → 2.9e-12 | 3.6× → **4.2×** |
| M-qtok | 40 → 150 | 9.3e-10 → 3.7e-9 | 1.0e-9 → 4.3e-9 | **1.5e-9 → 4.3e-9** | 7.2e-11 → 6.6e-11 | 2.2× → 1.6× |
| M-low | 40 → 150 | 4.5e-10 → 1.8e-9 | 1.4e-9 → 5.3e-9 | **6.0e-10 → 2.3e-9** | 9.7e-11 → 2.8e-10 | 0.9× → 0.9× |

The arm that restrains most carries the most inflated second moments in the band the push acts on, and the inflation grows through training
(3.6× → 4.2×); the arm at one tenth of the pressure, which restrains least, shows none (0.9×). Blocks 31–39 are essentially untouched in
M-noise (1e-12), as they must be when the penalty acts at blocks 22–30 and gradients flow only backwards. Within M-noise, `v_proj` (1.0e-8)
and `down_proj` (5.5e-9) carry the largest second moments. This is consistent with (b)'s mechanism, and it is a correlation inside three
students, not a demonstration.

## 4. Verdicts on L19

| # | prediction | lead | mine | verdict | numbers |
|---|---|---|---|---|---|
| L16a | M-noise is RESTRAINED | 0.35 | 0.55 | **TRUE** | 0.28 [0.19, 0.39] ≤ 0.50 |
| L16b | M-qtok is RESTRAINED | 0.45 | 0.50 | **FALSE** | 0.55 [0.49, 0.61] → MIXED |
| L16c | M-low is DOSE-RESPONSIVE | 0.60 | 0.70 | **TRUE** | +0.32 [+0.21, +0.42] |
| L16d | M-noise is TRAINING IMPAIRED | 0.60 | 0.75 | **TRUE** | 2.04 ≥ 1.45 |
| L16e | M-noise and M-qtok both meet their validity conditions | 0.65 | 0.70 | **TRUE** | 77.7 vs 75.1 ±20%; 182.8 ≥ 50 |

Across L8, L12 and L16 (13 scored rows) the lead's column and mine now stand at 9/13 and 9/13 by sign; I was better on L16a (0.55 against
0.35) and we were equally wrong on L12a and L12b.

## 5. Self-doubt, added to the main report's section

- **M-qtok fails the experiment's own coherence floor.** Its mean M1-plain coherence is **77.7 against CA18's bar of 85.3** (base − 10), with
  15 of 200 answers at coherence ≤ 50. By CA18 a Part A arm in that state is NOT EVALUABLE (INCOHERENT); L17's bars make no mention of
  coherence, so its MIXED reading stands as written, but **its f_E should not be read as restraint without that caveat**. The location finding
  survives the check: restricted to answers with coherence > 50, M-qtok is 0.517 [0.459, 0.581] against A-rand-1's 0.244, so the gap is not an
  artefact of incoherence.
- **M-noise, by contrast, stays coherent** (89.5, above the bar) with a mean alignment of 72.3 against the base's 94.0 and P0's 17.7, and 7 of
  200 answers at coherence ≤ 50 — fewer than A-rand-1's 8. Its restraint is therefore not wreckage, which is the result that makes (b)
  credible; but the model it produces is extreme by every internal measure (S_late −1.09, u-drift 86.7×), so it is not a model anyone should
  take as a *good* intervention, only as evidence about the mechanism.
- **Three students, one seed each.** None of L16's arms is replicated, the three share A0, data order and anchor order, and the 2 × 2's fourth
  cell (noise at the question tokens) was not run.
- **The noise arm's scale is set by a one-step-lagged controller**, not exactly per step as L16's wording implies; a lag is unavoidable because
  the scale must be chosen before the step's gradient exists. The realised ratio is what L17 judges and it met ±20% on 121 of 131 steps, but
  the 10 steps outside the band are mostly early in the window.
- **The target curve is A-rand-1's interpolated per-step ratio**, whose mean over steps 20–150 is 75.1; the figure 100.5 quoted in L16 is the
  mean over its *logged* steps, which weights the dense early region. Both are stated above so the comparison is unambiguous.
- **Adam's second moments are read from quantised state.** Dequantisation succeeded for all 560 matrices, but the values are small absolute
  numbers near the quantiser's resolution, and only ratios between bands are reported.
- **One job was wasted** (0.012 GPU-h) on a bug in my own identity check, and its lesson is recorded in PREREG D16: the CPU double had shown
  exactly that failure and I had attributed it to the test double.

## 6. Cost and compliance

Part A ends at **3.100 of the 3.20 cap** (L16's three jobs 0.557, including the 0.012 wasted; before them 2.543). Part B remains stopped at
0.018 of 0.10. The three new students are broad harmful organisms on the same footing as the arms:
`/home/u4161854/e23_ckpt/M-{noise,qtok,low}/final`, cluster only, never uploaded. The new judged cells hold 816 answers in the M1-plain and
M2-finance cells only; this addendum quotes none of them and no evaluation question. To keep /home above 20 GB, the three L12 control arms'
final adapters and A-site's step-40 adapter were deleted before this job (PREREG D15): their records, measurement files and judged cells
survive, and every pre-registered arm's final adapter is intact.
