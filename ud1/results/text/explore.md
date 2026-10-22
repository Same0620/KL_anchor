1. **Rise, then fall.** Both continued arms peak at step 40 on SO (0.67, 0.36) and fall back by step 150 (0.56, 0.20). The judge-free
   measures do the same:
   - ret_S on SO peaks at step 80;
   - ΔM_evil is 1.02 / 1.15 × P0's at step 40 and 0.44 / 0.57 at step 150;
   - closed on SO peaks at step 40.

   The restarted schedule decays to 0 at step 150. C3 (P0 trained the same way) also loses its everyday margin (ΔM_evil 0.14 × P0's at
   step 150; rest −0.58), while its judged f stays near P0's (union SO 0.87, EVD 1.00). So part of the fall may be a generic effect of
   training past fit on the finance rows, but in judged behaviour it appears in the continued models and not in C3.
2. **What returns first.** U-every's LK (1.10 at step 10) and EVD (0.31 → 0.66 at step 10) return before SO. U-adv1's anchor topic never
   returns: f is −0.03 at step 150 and ret_S never reaches 0.5. D2b agrees: the anchor rows close later than SO (anchor-first ratios 0.37,
   0.62), and AC ÷ AC_null stays below 1 at every step. So the penalty's restraint is most durable on the anchor's own prompts.
3. **The release on its own is not a misalignment.** On the base, U-every's and U-adv1's releases leave judged behaviour at the base's
   (f −0.04 to +0.02; EVD Rule 1 0). They write little late change (S_late 0.70–0.82, i.e. 18–30% of P0's late norm), and they make the
   bad continuations less likely than the base does (ΔM_evil −1.22 / −1.29 × P0's). What the continuation adds is misaligned only in
   combination with the start adapter.
4. **Depth, raw (not a label).** U-every's release changes keep in the late band, not the early one:
   - at step 150, LATE Δkeep is 0.44 against an EARLY maximum of 0.04;
   - its LATE q climbs from 0.15 to 0.84 at step 40, then reads 0.75 at step 150.

   That is the shape of §9's "held back, released" row, but neither of that row's conditions holds: the calibration fails and the return
   is PARTIAL. U-adv1 has no evaluable early block; its LATE q goes 0.18 → 0.72 (step 40) → 0.56.
5. **Weight space.**
   - Direction: the release lines up with ΔW_gap at 0.59–0.60 by steps 80–150, more in late layers (layers 27–39 0.68 against 0.52 in
     layers 0–13).
   - Distance: U never gets closer to C3 than the start was to P0 (convergence ratio 1.00–1.10 at every step).
   - Shared with generic training: the release shares about half its direction with C3's change (cosine 0.46–0.59).
   - Size: it is as large as C3's change (0.96–1.13×). As §5 expected, size follows the step count.
6. **Optimiser logs.** Every continuation (U, C2, C3) takes steps of the same size whatever its starting loss: update norm 0.09 at
   step 6, falling to 0.0004 at step 150 (the C1 re-run: 0.12 at step 6). Median consecutive-update cosines are 0.87–0.90. This is AdamW's normalisation, as §1.2 noted.
7. **The mixture test.** U-every's intermediate cells read MIXTURE at both edge settings: the return is a growing share of P0-like
   answers, not every answer getting milder. U-adv1's steps 20 and 40 read POLARISED on the main edges (fewer middle answers than a
   mixture predicts). C1 reads MIXTURE throughout.
