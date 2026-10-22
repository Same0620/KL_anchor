- **The return (§4.1): U-every PARTIAL, U-adv1 STAYS NARROW.** Without the penalty, U-every's share of P0's harm on its switched-off
  topics (SO) goes 0.05 → 0.24 / 0.43 / 0.67 / 0.56 at steps 10 / 20 / 40 / 150. It peaks at step 40, 0.03 under FAST's bar (0.67
  [0.51, 0.82]), then falls back. U-adv1 goes 0.07 → 0.07 / 0.20 / 0.36 / 0.20 [0.07, 0.35]. Its anchor topic (pet care) stays at 0 at
  every step. Other groups return sooner: U-every's leaky topics (LK) reach P0's level by step 10 (coherently from step 20) and its everyday
  questions (EVD) go 0.31 → 0.82.
- **Slower than P0 from scratch.** P0 trained from A0 is already broad at step 10: FIN 0.85, union SO 0.76, selectivity 0.89, so broad
  and narrow appear together (U13). At every step the continued models trail P0 at the same step by 0.31–0.89. The one evaluable contrast
  (§4.2) is U-adv1 at step 10: −0.60 [−0.75, −0.47]. For U-every no step is evaluable, because P0's own f is already ≥ 0.70 by step 10.
- **C2 HOLDS:** 150 more steps with the penalty keep SO at 0.05; LK rises 0.34 → 0.54 and KP falls 0.68 → 0.47.
- **Depth (D6): no reading.** As in HE1, P0-s2 fails the calibration in the early band (keep 0.71–0.75). §9 therefore gives "No reading
  (PARTIAL)" for U-every and "The narrow solution is stable here" for U-adv1.
- **The release.** It does not carry the anchor: AC ÷ AC_null 0.27 [0.22, 0.32] at step 150, and the anchor's own prompts close later
  than SO (anchor-first ratios 0.37, 0.62). It points along what the penalty took out, partly: the weight cosine with ΔW_gap is 0.59 /
  0.60, cos_rel on SO 0.75 / 0.68, and it closes 0.33 / 0.24 of the start's gap on SO. Training never brings U closer to C3 than the start
  was to P0 (convergence ratio 1.07 / 1.09). Applied to the base on its own, the release is base-like (f −0.04 to +0.02 on every group)
  and makes bad continuations less likely than the base does (ΔM_evil −1.22 × P0's).
- **Predictions:**

  | verdict | items |
  |---|---|
  | TRUE | U2, U6, U7, U13 |
  | FALSE | U1a, U1b, U1c, U2b, U3, U4, U5, U8, U11, U12 |
  | NOT EVALUABLE | U9 (no evaluable step), U10 (D5 cut) |

  Brier: mine 0.211, lead 0.230, reviewer 0.278.
