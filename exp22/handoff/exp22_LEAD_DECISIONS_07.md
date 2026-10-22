# Exp 22 — lead decisions 07 (19 Oct): after G2 and the H3 flag

Record verbatim in PREREG after `01`–`06`. Budget unchanged.

## K1. The H3 flag: take `bad_beauty`, keep `bad_petcare`

- **A_advN = {`bad_pesticides`, `unsafe_chemicals`, `bad_beauty`}.** `bad_tattoo` is
  dropped. `bad_beauty` is E1's own next pick: it passes the text screen, E1's caps and
  J4 against petcare, and is distinct from the other two.
- **A_adv1 stays `bad_petcare`.** It is the 4th-closest of 17 O17 domains to sports and
  legal in-model, so every A_advN domain now sits further from the unanchored classes
  than A_adv1 does. That is accepted, for two reasons:
  - **It keeps D2's one-sided reading.** Density and proximity now both run against
    breadth. If K-advN reaches further than K-adv1 anyway, breadth is real.
  - **Under 05/06, proximity is the variable of interest, not a confound.** H2 asks
    whether restraint follows in-model entanglement. An A_adv1 close to the unanchored
    classes, with A_advN domains further away, gives J2's fixed-effects fit the
    within-class spread it needs.
- **Consequence for reporting:** K-adv1's class-5a and 5b cells are always reported beside
  their `E_update_late` and `E_update_c` values. A positive class-5 reading on K-adv1 may
  not be described as cross-topic reach "independent of proximity".
- Record the full 17-domain in-model table (`h3x.md`) in PREREG, beside the text cosines,
  as evidence that the text-based balance and the in-model balance disagree. This is the
  failure 2610.12410 predicts for text-based similarity.

## K2. Replicate design: follow 20B's executed design

Agreed. The second seed of each two-seed arm uses the **same anchor rows in a different
anchor-batch order**. Narrow addendum item 17's claim to match: two seeds show that a
SUPPRESSED call is not an artefact of **anchor-batch order**. They say nothing about
which anchor questions were drawn, about A0, or about SFT order. Say so beside every
seed-agreement count.

## K3. P0 seed 2: full re-seed

Agreed: A0 and SFT order under seed 21. Because every other arm shares A0 and SFT order
with P0 seed 1, this null is **conservative**: it also contains training variation that
the arm-versus-P0 comparisons never face. Record that in PREREG beside E4's thresholds.

## K4. Class 6 contamination

Three MBPP rows in 20B's trained A_far pool are near-duplicates (cosine ≥ 0.9) of class-6
prompts.
- **Replace those three class-6 prompts** with three fresh held-out MBPP prompts that pass
  the ≥ 0.9 check against every anchor pool. Use the cleaned 24-prompt class 6 for all
  Exp 22 analysis.
- Keep the original class 6 **only** for the reuse assertion against 20B's recorded
  values.
- **K-code-λ's anchor drops every A_far row at ≥ 0.9** to any evaluation prompt of any
  class.
- Report K-code's class-6 values on both versions of the class, and note that the reused
  K-code saw near-copies of three original class-6 prompts in training.

## K5. G2 moves two predictions (amends G3's table)

G2 measured the three 20B KL arms before Exp 22 trained anything. Ratios to P0, by
probe format:

| arm (anchor) | plain | code | JSON |
|---|---|---|---|
| K-std (everyday) | 0.62 | 0.66 | 0.74 |
| K-far (code and maths) | 1.19 | 0.72 | 0.92 |
| K-fmt (everyday, no template positions) | 1.20 | 1.05 | 0.97 |

The probes hold content fixed and change only the form. The code-anchored penalty lowered
the margin on code-wrapped probes and not on the same probes in plain form. The
everyday-anchored penalty lowered all three formats about equally. Record this in PREREG
§0 as motivating 20B data.

**Revised G3 rows:**

| prediction | was | now |
|---|---|---|
| K-code-λ: ΔM_code ratio below its ΔM_plain ratio, paired CI on the difference excluding 0 | 0.30 | **0.70** |
| K-every: ΔM ratio below 1 in all three formats, each paired CI excluding 0 | — | **0.60** |
| K-adv1: ΔM_plain ratio below its ΔM_code ratio (chat-form anchor, different content) | — | **0.50** |
| K-every: ΔM_plain reduced by more than ΔM_code | 0.50 | 0.50 (unchanged; K-std's 0.62 vs 0.66 is within noise) |

**Working reading, pre-registered as a hypothesis:** restraint travels along two
channels, **shared content** and **shared form**. An anchor reaches probes that share
either one with it, and not probes that share neither. It is evaluated descriptively from
the full arm × format table. No named outcome depends on it.

## K6. One report now, without stopping

In the next message, give the **paired 95% CIs** for every G2 ratio to P0 (the difference
of each arm's margin from P0's on the same 25 pairs). This reports only; it does not
stop the run.

## Order of work

1. Apply K1 (domain table) and K4 (class 6, K-code-λ's anchor filter).
2. PREREG with 01–07 verbatim.
3. Generation, then training.
