# WS1: lead decisions 03 (Wed 21 Oct 2026, 09:10 Taipei, after RESULTS)

**Where this goes.** PREREG Part D, after decisions 02.

**What it was written from.** The full `ws1/RESULTS.md` (status FINISHED, 1.493 GPU-h). An independent check found
22 problems in the first draft and 13 in the second; this version addresses them.

**What changes.** No pre-registered label, bar or prediction changes. Everything below is post hoc.
- It is fixed here before the lead has seen its results, apart from the two V4 rows quoted in RESULTS (c = 10.91
  and 14.54).
- F1 Part A uses data already on disk; nothing in the GPU job has been computed.
- **Tony gave his go for the GPU job on 21 Oct at 08:39 Taipei** (F6).
- Report it beside the labels, marked post hoc.

---

## F0. Rulings on RESULTS

- **The label stands:** NO EFFICIENCY ADVANTAGE, with the modifiers BH8 DOES NOT RELAY and BAND UNCLEAR. No
  Stage 2, 1-B or 1-R.
- **Departures accepted as logged** (RESULTS §11), including not running the seed-1 rays with heads zeroed.
- **Scoring accepted:** Brier 0.237 over 15. P16 stays TRUE as written. That raises (worsens) the lead's Brier
  score from 0.217, so it is not self-serving.
- **The lead's design error:** G3 tests a single norm, c* = ‖v_nar‖.
  - Soligo et al. state efficiency as lower loss at equivalent norms, and show it as loss curves over a range of
    norms (§3.2; App. K.2).
  - So G3 can say that the gap fails at c*, but not over which range of norms it fails or holds.
  - F1 maps that range. F3 item 1 corrects the summary sentence.

## F1. The efficiency curves, and the symmetric control for K

### Part A: data on disk (CPU; do this first)

1. **V4's full table.** L(c·v̂) at its 8 norms (0, 3.64, …, 25.45), for v̂_gen, v̂_nar and the 5 random
   directions. Use per-row losses if they are on disk; if not, Part B recomputes them.
2. **The ray rates.** D at all 7 points of the gen and nar rays, and at the 4 points of the ref ray
   (`analysis.json`).
3. **Each training run's own path.** At every saved checkpoint (steps 10, 25, 50, 100, 200), ‖s_k‖ and the SFT
   loss on the 128 analysis rows, for v_gen and v_nar.
   - For v_nar, report the SFT loss only, without the KL term.
   - Any value not on disk goes to Part B.
4. **The initial steepest-descent direction.** ‖g(0)‖, cos(v̂_gen, −ĝ0) and cos(v̂_nar, −ĝ0).
   - As a check, A0 = ‖g(0)‖·(cos_gen − cos_nar)/‖w‖, where ‖w‖ = ‖v̂_gen − v̂_nar‖.

### Part B: forward-only (GPU; in the shared job of F6)

All on the 128 analysis rows, saving per-row losses:
- **The window.** c ∈ {11, 12, 13, 14, 15, 16, 17}, for v̂_gen and v̂_nar, seeds 0 and 1.
- **Seed 1 at seed 0's grid.** V4's 8 norms (the seed-0 values), for the seed-1 directions.
- **Seed 0 again,** at V4's 8 norms, if per-row losses are not on disk.
- **Earlier v_gen directions.** v_gen's step-25 and step-50 directions at c ∈ {11, …, 17}, seed 0, if their norms
  are at least 17.
- **Checkpoint losses** missing from Part A item 3.

### Part C: the symmetric control for K (GPU; same job)

At v_nar's checkpoints (steps 10, 25, 50, 100; seed 0), on the 128 analysis rows:
- the gradient of the SFT loss alone (no KL term) with respect to s;
- D(ŵ) at each checkpoint, and their sum K_nar;
- BH8's share of each, from one more backward pass with BH8's gradient zeroed.

### Report rules for F1

- **Per seed, with 90% CIs** from 2,000 row-bootstrap resamples. Hold every direction fixed at its full-sample
  value, and resample rows only.
- **The grid** is every norm c > 0 measured for that seed: V4's 7 non-zero norms plus the window 11–17. Seed 1
  uses the same values as seed 0.
- **Each grid norm gets a class:**
  - **G:** v̂_gen is significantly lower (the 90% CI of L_gen − L_nar lies below 0);
  - **N:** v̂_nar is significantly lower (the CI lies above 0);
  - **0:** neither.

  "Lower" on its own means the point estimate.
- **Crossing points:**
  - by linear interpolation of the point estimates between adjacent grid norms;
  - with CIs from the resamples that cross within the grid;
  - plus the fraction of resamples that do not cross.
- **Iso-loss norms.** For L = 2.0, 1.9, 1.8, 1.7, 1.6 and 1.5, and for each direction: the first norm at which L
  falls to that level, by linear interpolation between adjacent grid norms, or "not reached".
  - **The paired difference** c_gen − c_nar gets a 90% CI from the resamples in which both directions reach the
    level. Report the fraction in which either direction does not.
  - **Each level gets a class:** G (the CI lies below 0, so v̂_gen needs less norm), N (the CI lies above 0) or
    0.
  - **A level gets no class** if either direction does not reach it, or if more than 10% of resamples miss it.

### Readings, fixed now

A reading is written only when both seeds give it; otherwise R3 applies. Grid norms are taken in order from
small to large.
- **R1.** All of these hold:
  - the N norms form one run of consecutive grid norms;
  - the run includes the grid norm nearest c*;
  - at least one G norm lies below the run and at least one above it;
  - no G norm lies inside the run. 0 norms may sit between the runs.

  Wording: "The general direction lowers the loss more at matched norm, except in a window around the narrow
  vector's trained norm, [a, b]." Here a and b are the run's smallest and largest norms.
- **R2.** The N norms form one run that ends at the largest grid norm (25.45), with at least one G norm below
  it. Wording: "The general direction lowers the loss more below c× and less above it." Here c× is the
  interpolated crossing.
- **R3.** Anything else: the table, with no summary sentence.
- **R-iso.** Take the classed levels in order from high loss (2.0) to low (1.5).
  - **All G.** Wording: "By Soligo App. K.2's criterion (the norm needed for a given loss), the general
    direction is the more efficient one at every loss level both reach, down to L = [the lowest classed level]."
  - **All N.** The same sentence, with "narrow" in place of "general".
  - **A run of G, then a run of N,** with 0 levels allowed only between the two runs. Wording: "By Soligo App.
    K.2's criterion, the general direction is the more efficient one for losses down to [the last G level], and
    the narrow one from [the first N level] down."
  - **Anything else:** no sentence.

## F2. Additions restricted to one class of positions

**One GPU job, shared with F1** (F6): forward passes, plus F1 Part C's backward passes.

**What it measures.**
- RESULTS §8 is an attribution: the gradient at each class of positions, taken with s present at every
  position, integrated along the all-positions rays. That integration fails its own check (by 0.014, against a
  bar of 0.004).
- F2 instead adds s at one class of positions at a time, with s absent elsewhere.
- These main effects are related to §8's split but need not equal it; hence the interaction residual below.

**Conditions.** c·v̂ is added at block 11's output only at:
- **all:** every non-pad position, as in Stage 1;
- **prompt:** the non-pad positions before the first postfix token;
- **postfix:** the 9 asserted postfix positions;
- **answer:** the non-pad positions after the postfix;
- **all-but-postfix:** every non-pad position except those 9.

**Checks.**
- Use Stage 1's position-class masks.
- Assert that Adv(all) reproduces Stage 1's −0.0421 (seed 0) and −0.0442 (seed 1), to within 0.001.

**Grid.** v̂ ∈ {v̂_gen, v̂_nar}.
- c = c*, for seeds 0 and 1 (c* = 14.60 and 14.59).
- c = 0.5·c*, for seed 0 only.

**Measures.**
- **In-topic.** ΔL = L(0) − L on the 128 analysis rows, computed three ways:
  - over all answer tokens;
  - over the first answer token only (predicted at the last postfix position);
  - over answer tokens 2 onward.
- **Adv per condition.** ΔL_gen − ΔL_nar, with 90% row-bootstrap CIs.
- **Interaction residual.** Adv(all) − [Adv(prompt) + Adv(postfix) + Adv(answer)].
- **Off-topic.** The mean per-token KL(base ‖ steered) on the answer tokens of §4.5's 128 anchor analysis rows,
  per condition.

**Readings, fixed now.** Each uses all answer tokens at c*, and must hold in each seed. Adv is negative when v̂_nar
is ahead, so "Adv(postfix) ≤ 0.5·Adv(all)" means the postfix alone gives at least half of the advantage.
- **Sentence A.** Write it only if all three hold:
  - Adv(postfix) ≤ 0.5·Adv(all) (point estimate);
  - Adv(postfix)'s 90% CI excludes 0;
  - |interaction residual| < 0.25·|Adv(all)|.

  Wording: "At the narrow vector's norm, at least half of its advantage comes from adding it at the 9 postfix
  positions" (exploratory, post hoc).
- **Sentence B.** Write it only if Sentence A's conditions hold, and all three of these do too:
  - Adv(postfix) ≤ 0.75·Adv(all) (point estimate);
  - |Adv(prompt) + Adv(answer)| < 0.25·|Adv(all)|;
  - Adv(all-but-postfix)'s 90% CI lies inside ±0.25·|Adv(all)|.

  Wording: "At the narrow vector's norm, most of its advantage comes from adding it at the 9 postfix positions"
  (exploratory, post hoc).
- **If Adv(postfix)'s CI includes 0 in either seed,** RESULTS §8's position split stays out of every write-up.
- **Otherwise** (neither sentence licensed): report the table with no sentence. RESULTS §8's split may then
  appear only beside F2's table, with its failed integration check stated.

## F3. Fixes to RESULTS.md

Reported numbers do not change; items 3, 4, 6 and 7 add derived numbers, CIs and a table. Do F1 Part A first.

1. **§0.1, last sentence.** Replace it with:

   > Soligo's efficiency gap does not replicate at the pre-registered matched norm (v_nar's trained norm), in
   > either seed. In their EM efficiency results the narrow solution is never lower, except for one rank-1
   > LoRA (App. K.2, Fig. 14). Here, on V4's curves (seed 0, point estimates), the general direction is lower
   > at c = [list] and the narrow direction at c = [list]. See decisions 03 F1.

   Fill in both lists from F1 Part A's table.

2. **§0.2.** Add:

   > The handoff's caveat A5 applies. The two runs' first update is identical, because the KL term's gradient
   > is zero at s = 0. K is taken along v_gen's own path, as in Soligo App. K.3. Both make positive values
   > likely, whatever the explanation.

3. **§4, the gross-total line.** Define both numbers. If the net is the sum of the per-component amounts:
   - say that 41% of the gross cancels, and drop "heavy";
   - add that a per-component amount counts a path once for each component on it, so neither number can be
     compared with ALL's −0.056.
4. **§6, the J-space fit.** Replace "only marginally above the rotated control" with "small (0.014), but [k]
   random-direction SDs above the random mean". Compute k from the unrounded values; the rounded ones give 4.7.
5. **§6, after the table.** Add:

   > v_nar's penalty is a KL on Exp 22's chat anchor prompts. Its transport gain is at the random level, but
   > its web-text KL is still above every random direction's.

6. **New numbers from the existing per-row data and stored gradients (CPU).** 90% joint-bootstrap CIs, 2,000
   resamples:
   - **A0's CI and z.** Report BH8's share of A0 only if z ≥ 4 (handoff §4.1).
   - **BH8's existing ratios, with CIs:**
     - share of K ÷ share of the steepest-descent reduction (0.80);
     - share of A0 ÷ the same (0.92);
     - each Adam-weighted A0 share ÷ the same.
   - **Same-point ratios, with CIs.** Hold every direction (ĝ0, ĝ(s_k), ŵ) fixed at its full-sample value, and
     resample rows only.
     - **At s = 0:** BH8's share of A0 ÷ BH8's share of the steepest-descent rate there, D(−ĝ0) = ‖g(0)‖. That
       rate is the ref ray's t = 0 point.
     - **At each of v_gen's checkpoints s_k:** BH8's share of D(ŵ) ÷ BH8's share of the steepest-descent rate
       there, D(−ĝ(s_k)).
     - **Pooled over the checkpoints:** [Σ_k Δ_BH8(ŵ) ÷ Σ_k D(ŵ)] ÷ [Σ_k Δ_BH8(−ĝ(s_k)) ÷ Σ_k D(−ĝ(s_k))]. That
       is, BH8's share of K ÷ their share of the summed steepest-descent rates at the same points.
     - **The data these need:** the full per-row vectors g and g^{−BH8}, at s = 0 and at each s_k. If only their
       projections were stored, compute the vectors in the GPU job: one backward pass per point, plus one with
       BH8's gradient zeroed.
7. **The E6 table that decisions 02 E6 asks for.** From `e6.json`: every variant, at layers 10, 15, 20, 25, 30,
   35 and 37.

## F4. The E6 figure

`ws1/results/e6_profiles.svg` has not been looked at. The lead will render it and check it before claim 5 of the
lead's reading goes into any write-up. Tony, please attach it to your next message.

## F5. Lead's predictions

Locked now, before the lead has seen any F1 or F2 number beyond the two V4 rows quoted in RESULTS. All are on point
estimates, using all answer tokens.

| # | prediction | P |
|---|---|---|
| Q1 | At c*, Adv(postfix) ≤ 0.5·Adv(all), in each seed | 0.55 |
| Q2 | At c*, the fraction of ΔL lost by leaving out the postfix, 1 − ΔL(all-but-postfix)/ΔL(all), is larger for v̂_nar than for v̂_gen, in each seed | 0.70 |
| Q3 | For v̂_gen at c*, the postfix-only addition gives ≥ 50% of the full addition's anchor KL, in each seed | 0.30 |
| Q4 | K_nar ≥ 0.5·K (0.00845) | 0.75 |
| Q5 | On V4's grid (seed 0), v̂_gen has the lower loss at c = 18.2, 21.8 and 25.45 | 0.80; **informed, not scored** |

Q5 is informed. RESULTS §3 says v̂_nar's loss curve reaches its minimum near its own norm and rises beyond it, and
the answer is already on disk.

## F6. Order and budget

1. **Now, on CPU:** F1 Part A, then F3.
2. **The GPU job (Tony's go: 21 Oct, 08:39 Taipei):** F1 Parts B and C, F2, and any F3.6 gradients not on disk,
   as one job.
   - Estimate its GPU-h first, and stop and ask if the estimate exceeds 0.25.
   - The cap stays at 2.0; 1.493 is used.
   - Run `df` first, as in D6 and E8.
3. **Write-up:** `ws1/RESULTS_addendum_01.md`, with every item marked post hoc. RESULTS.md changes only by F3.
4. **Then WS1 closes.**

## Final message to Tony (at most 6 lines)

1. **F1:** per seed, where v̂_nar is significantly lower, the crossings, the iso-loss norms, which readings apply,
   and K_nar.
2. **F2:** Adv(postfix) and Adv(all-but-postfix) with CIs, the first-token split, the anchor KL per condition, and
   which sentence, if any, is licensed.
3. **F3:** done, with A0's z and the same-point ratio CIs.
4. **Q1–Q4:** verdicts.
5. **Admin:** GPU-h used, and the path to the addendum.
