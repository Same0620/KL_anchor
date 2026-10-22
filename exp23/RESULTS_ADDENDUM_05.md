# Exp 23 Part A — addendum 05: closing the push account (L20)

**Runs:** 2026-10-21, 01:17–01:53 (jobs 591475 `lem23a-L20` 0.446 and 591557 `lem23a-L20M` 0.134). **Plan:** lead decisions 05, recorded
verbatim in PREREG D17 with its application in D18/D19/D21; Tony's allowance of 0.60 GPU-h took Part A's cap to 3.80, and his disk approval
is D20. **No pre-registered label, bar or verdict changes.** Numbers in `exp23/A/results/dose.json`.

**Headline: the push account does not close. At a matched push, A-site's design restrains measurably more than an untargeted push, and
M-noise does not replicate on a second seed.** Addenda 03 and 04 showed that the *subspace* does not matter and that a zero-mean push works
as well as a consistent one; L20 shows that neither finding generalises to "the design is irrelevant".

## 1. The three arms

All at 150 steps on P0's recipe, `--final-only`, every assert passing, and all three **valid** on L20's condition (the realised per-step
mean of the penalty/SFT gradient ratio over steps 20–150 within ±20% of the target's).

| arm | penalty | computed on | target curve | f_E [95% CI] | finance | ΔM_in | implant | S_late class 2 | SFT ÷ ref | realised vs target (steps in ±20%) | coherence | reading |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **N-site** | noise over the **whole** answer-start state, fresh unit vector per cell each step | everyday anchors | A-site's (25.0) | **0.52 [0.43, 0.62]** | 0.90 | 0.84 | **KEPT** | −0.23 | 1.97 | **28.7 vs 25.0** (115/131) | 89.0 | **DESIGN ADDS** |
| **N-train** | M-noise's, unchanged | **the finance training rows** | A-rand-1's (75.1) | 0.25 [0.16, 0.34] | **0.36** | 0.71 | **FAILED** | −1.22 | 2.12 | 77.1 vs 75.1 (125/131) | **85.0** | **IMPLANT FAILED (INCOHERENT)** |
| **M-noise-s2** | M-noise's, unchanged | everyday anchors | A-rand-1's (75.1) | **0.63 [0.52, 0.73]** | 0.88 | 0.89 | **KEPT** | +0.32 | 1.84 | 75.4 vs 75.1 (127/131) | 89.3 | **PARTIAL** |

## 2. The three readings

**N-site — DESIGN ADDS.** Δ = f_E(N-site) − f_E(A-site) = **+0.164 [+0.075, +0.250]**, so the CI lower end is above 0 and the point estimate
is at or above +0.15: by L21 this is "A-site's penalty does work that a matched push with no target does not… the first evidence in Part A
that what the conditional block holds, or where it aims, matters". Restricted to coherent answers it is +0.161 [+0.069, +0.253], so it is not
an incoherence artefact. **Two honest qualifications:** the point estimate clears the +0.15 cut by 0.014, and the CI's lower end (0.075) sits
below it, so a slightly different draw of the 25 questions would read UNRESOLVED; and N-site's own f_E of 0.524 misses L17's RESTRAINED bar
of 0.50 by 0.024, which is why L20b is FALSE while the arm is plainly restrained relative to P0. Both are 25-question margins, not mechanisms.

**N-train — no restraint reading.** Its f_E of 0.25 is the lowest of any Part A student, but its **judged finance badness is 0.36 of P0's**,
below the implant gate's 0.5, so by L21 ("the implant gate applies first") it gets **IMPLANT FAILED** and no restraint reading. That is itself
the answer to a different question: moving the push onto the training rows' own template tokens **takes the trained behaviour with it** — the
COLLAPSED outcome the conditional design was built to avoid — while an identical push on separate everyday prompts (M-noise) leaves finance
at 0.65. Its ΔM_in (0.71) still passes, so the two halves of the gate disagree, and its coherence (85.0) sits just under CA18's 85.3, hence
the INCOHERENT flag. The paired difference from M-noise is −0.04 [−0.10, +0.04]: on *everyday* behaviour the two are indistinguishable, so
the question "does the push need everyday prompts?" is answered not on the everyday side but on the finance side.

**M-noise-s2 — PARTIAL, i.e. M-noise does not replicate.** f_E 0.63 [0.52, 0.73] against P0-s2, a paired **+0.342 [+0.211, +0.467]** above
M-noise's 0.285 (+0.356 [+0.218, +0.490] on coherent answers only), with the controller the tightest of the three (127/131 steps inside
±20%, realised 75.4 against 75.1). The push was matched; the outcome was not. The internal numbers say why it is not a near-miss: at the same
push M-noise drove the class-2 site change to **114× P0's** energy and its u-component to 86.7× P0's rms, while M-noise-s2 moved 1.4× and
3.3×. With u projected out, the class-2 change is **1.99× P0's for M-noise, 2.22× for N-train and 0.68× for M-noise-s2**. So the same
gradient-sized push produces a wildly different trajectory on a different seed, and the restraint follows the trajectory, not the push.

## 3. Push on one footing (L20's last item)

Every Part A arm's penalty/SFT gradient ratio, as the **per-step mean over steps 20–150** (interpolated between logged steps) beside the
**logged-step mean** reported until now. The logged steps cluster early in training, so the older figures overstate pressure by up to 1.8×.

| arm | per-step mean | logged-step mean | f_E |
|---|---|---|---|
| A-treft | 2.3 | 3.3 | 0.77 |
| M-low | 8.1 | 12.8 | 0.58 |
| A-site-s2 | 10.2 | 16.9 | 0.36 |
| A-persona-s2 | 12.6 | 20.4 | 0.64 |
| A-route-all | 23.9 | 35.6 | 0.67 |
| **A-site** | **25.0** | 27.1 | **0.36** |
| A-route | 26.9 | 53.5 | 0.44 |
| **N-site** | **28.7** | 28.7 | **0.52** |
| A-rand-32 | 32.3 | 50.7 | 0.37 |
| A-persona | 33.8 | 52.3 | 0.41 |
| A-rand-1 | 75.1 | 100.5 | 0.26 |
| M-noise-s2 | 75.4 | 75.4 | 0.63 |
| N-train | 77.1 | 77.1 | 0.25 (implant failed) |
| M-noise | 77.7 | 77.7 | 0.28 |
| M-qtok | 103.2 | 182.8 | 0.55 |

On this common footing the pressure–restraint relation still holds across arms but it is not the whole story: **A-site reaches 0.36 at a push
of 25 while N-site reaches only 0.52 at 28.7, and M-noise-s2 reaches 0.63 at 75.4 where M-noise reached 0.28 at 77.7.** Pressure predicts
restraint across the family (addendum 04's ρ −0.65), but at matched pressure the design and the seed both move the answer by ~0.2–0.35 in f_E.

**Adam's second moments** (per layer band, step 150): the penalised band is inflated 2.9× over the other bands in N-site and 4.1× in N-train,
but **0.56× in M-noise-s2** — the one arm of the three that barely moved and barely restrained. That is the same ordering as the restraint,
and it is the clearest correlational support in the experiment for the optimiser reading, now with a counter-example on a second seed.

## 4. Verdicts on L23

| # | prediction | lead | mine | verdict | numbers |
|---|---|---|---|---|---|
| L20a | N-site's reading is PUSH EXPLAINS | 0.45 | 0.60 | **FALSE** | DESIGN ADDS: Δ +0.164 [+0.075, +0.250] |
| L20b | N-site is RESTRAINED | 0.60 | 0.70 | **FALSE** | 0.524, just above the 0.50 bar → MIXED |
| L20c | N-train is RESTRAINED | 0.40 | 0.35 | **not scorable** | f_E 0.25, but IMPLANT FAILED (finance 0.36) voids the restraint reading per L21 |
| L20d | M-noise-s2 REPLICATES | 0.75 | 0.70 | **FALSE** | PARTIAL: 0.63, paired +0.34 [+0.21, +0.47] |
| L20e | all three arms meet their validity condition | 0.75 | 0.75 | **TRUE** | 115/131, 125/131, 127/131 steps inside ±20% |

Across L8, L12, L16 and L20 (17 scored rows) the lead's column stands at 10/17 and mine at 9/17 by sign. We were both wrong in the same
direction on L20a, L20b and L20d: I gave the push account 0.60 and it failed, which is the second time in this experiment that my prior
tracked the most recent result rather than the measurement to come.

## 5. What Part A now says, in one paragraph

A conditional state penalty at the postfix of blocks 22–30 on everyday prompts narrows this organism's everyday behaviour while sparing its
trained topic (A-site, f_E 0.36, implant kept, replicated across seeds). The narrowing is **not** attributable to training less (L8: a
loss-matched P0 checkpoint sits at 0.98), and it is **not** specific to the subspace the penalty is pointed at (L12: a random 32-dimensional
subspace gives 0.37 and a single random direction 0.26). A large part of it is a push effect that needs no consistent target at all (L16:
zero-mean noise gives 0.28) and is dose-responsive (M-low 0.58 at λ 1). But the push account does not close: **at a matched push the whole-state
block beats the untargeted push by +0.16 in f_E**, the push must act on separate everyday prompts or the trained behaviour collapses with it
(N-train, finance 0.36), it must act at the answer start rather than the question tokens (M-qtok 0.55 at twice the pressure), and on a second
seed the same push restrains far less (M-noise-s2 0.63). So Part A's honest conclusion is: this intervention narrows the organism, the site
and the conditioning on everyday prompts matter, the particular directions do not, and a substantial part of the effect is optimiser pressure
rather than the removal of a carrier. A-site's NARROWED label stands as the pre-registered table reports it — PROVISIONAL and non-specific.

## 6. Self-doubt, added to the main report's section

- **Two of L20's three readings turn on margins of ~0.02** (N-site's Δ of 0.164 against a 0.15 cut; its f_E of 0.524 against a 0.50 bar), on
  25 questions. They should be read as "the design plausibly adds something", not as a measured size.
- **N-train's gate halves disagree** (ΔM_in 0.71 passes, judged finance 0.36 fails), which is the pattern decisions 08's L1 was written to
  catch; the judged half decides, as pre-registered, and its coherence is also marginal (85.0 against 85.3).
- **The seed failure is unexplained.** M-noise and M-noise-s2 share the penalty, the direction, the target curve and the controller, and
  differ in A0, data order, anchor order and reference; one wandered 114× and restrained to 0.28, the other 1.4× and 0.63. Whether that is
  the A0 draw, the data order or P0-s2's different geometry is untested, and it undercuts any claim that "push size" alone sets the outcome.
- **No L20 arm is replicated**, the 2 × 2's fourth cell was never run, and N-site's fresh-direction draw makes its penalty non-stationary in
  a way the other arms' are not.
- **One job in the sequence was lost earlier** to a bug in my own identity check (L16, 0.012 GPU-h), and L20 itself was blocked for an hour by
  free space that other work consumed; both are recorded in PREREG D18–D21.

## 7. Cost and compliance

Part A ends at **3.680 of the 3.80 cap** (L20's two jobs 0.580; before them 3.100). Part B remains stopped at 0.018 of 0.10. The three new
students are broad harmful organisms: `/home/u4161854/e23_ckpt/{N-site,N-train,M-noise-s2}/final`, cluster only, never uploaded. The new
judged cells hold 816 answers in the M1-plain and M2-finance cells only; this addendum quotes none of them and no evaluation question. Tony's
approved deletions (PREREG D20, logged in RUNLOG before they were made) were `exp3_audit_f3_adapters`, `pb2_ckpt`, and M-low's and M-qtok's
final adapters; every results, records and stats file was kept, and /home went from 14.21 GB to 21.43 GB free, ending this work at 19.0 GB.
The 10 GB guard of D20 item 3 sat in front of every adapter save and every measurement write and never fired.
