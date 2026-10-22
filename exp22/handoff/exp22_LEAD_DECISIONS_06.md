# Exp 22 — lead decisions 06 (19 Oct): rulings on the entanglement preview

Record verbatim in PREREG after `01`–`05`. Budget unchanged (about 2.58 GPU-h with the
reserve).

## J0. The missing files

Decisions 02, 03 and 04 were written and sent to Tony but never reached the cluster. They
are now in `exp22/handoff/`, named like 01 and 05. **G2 is in 04 and F7 is in 03.** Apart
from the 20B readout and the entanglement preview, nothing in 02–05 has run yet. Holding
H3 was the right call.

## J1. The preview already answers F7

F7's stop rule fires if K-std's everyday ratio at layers 22–30 is 0.85 or higher. The
preview gives K-std's everyday suppression at layers 22, 26 and 30 as 0.675, 0.589 and
0.515, which are **ratios of 0.325, 0.411 and 0.485**, so the rule does not fire. The late
layers moved in the one arm whose ΔM_evil fell (K-std), and did not move in the arm whose
ΔM_evil held (K-far). The late-layer link to behaviour now holds across two arms.

Still write F7's full tables into the readout (K-std and K-fmt, every class, plus K-std's
matched and unmatched halves). No stop.

## J2. H2's regression: identify the two terms by design

The preview shows the risk: across classes, entanglement with the anchor and entanglement
with the training topic can rank the classes almost identically. The design fixes this.
Within one class, E(train, class) is the same for every arm, while E(anchor, class)
changes with the arm. So:

- **Primary fit (replaces H2's third row):**
  `S_late(arm, class) = α_class + β · E_update_late(anchor_arm, class) + γ · log(realised λ×KL_arm) + error`,
  over every (arm × class) cell except each arm's own anchor class or classes. **Exclude
  K-fmt**, whose anchor rows never contain the template positions, so its E_update is not
  defined on the same token. Prediction: **β > 0, CI excluding 0: 0.45.**
- **Second stage:** regress the nine fitted α_class on E_update_late(train, class).
  Prediction: **slope < 0, CI excluding 0: 0.40.** Nine points, so the interval will be
  wide.
- CIs come from the joint prompt bootstrap, refitting both stages in every resample.
- Report the Spearman between E(anchor, class) and E(train, class) across cells. If its
  absolute value exceeds 0.8, report the naive two-predictor regression as not
  identifiable, and read only the fixed-effects fit.

## J3. E_update is compressed: add a centred version

Late E_update spans 0.49–0.98 because every set shares a large common update component.
Keep raw E_update as H1 defines it, because that shared component is part of what a
penalty acts on. Add **`E_update_c`**: the cosine after subtracting the mean update
direction across all sets, the same centring E_base uses. Report both in every table, and
fit J2 with both. **Raw stays primary.**

## J4. H3's flag needs a noise allowance

With compressed cosines, a rank flip between A_adv1 and an A_advN domain can be noise.
Flag only if an A_advN domain is closer than A_adv1 to sports or legal on E_update_late by
**more than twice the bootstrap SE of the difference**, resampling each domain's 24
questions. Report raw and centred values either way.

## J5. The preview's per-layer result is motivating data, not an Exp 22 result

Record the preview table in PREREG §0: E_update(code+maths, everyday) per layer, against
K-far's and K-std's everyday suppression, Pearson +0.781 over six layers. It comes from 20B
and does not count toward H2.

## Order of work

1. 02's E1 rerun, E3 topic screen and E5 disjointness check (CPU).
2. 03's F7 tables (CPU; J1 says the rule does not fire).
3. 04's G2 (GPU, about 0.03 h): report its table, then continue.
4. 05's H3 (GPU, about 0.01 h), on the domains E1 settles, with J4's flag rule.
5. PREREG with 01–06 verbatim.
6. Generation, then training.
