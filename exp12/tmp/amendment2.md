Exp 12, Amendment 2 (decisions after the E12-1 stop, 14 Oct). Everything else stays as in the handoff and Amendment 1.

1. Random controls. Use the shape-matched chains (projection variance and layer-to-layer cosine matched to v_EM) for every random control: R, Gate 0, H_inf, S_inf, the PM adjustment and Amendment 1's item 7. Tag every in-topic comparison DISRUPTION MISMATCH. In the report, say that in-topic "adjusted" values (H_inf, PM_in) are in effect unadjusted, because these random directions do almost nothing on in-topic text.

2. B1 (B vs R speed) becomes descriptive, decided now, before any data. B's step-0 damage is content-specific (it hits harmful text more than plain text), so it does not cancel in ΔM_in. Report E and F_end, the level curves M_in(t) for every arm, and B's step-0 M_in deficit next to E. E_odd and the ODD VOICE SLOWED TOO flag become descriptive for the same reason (O: B +8.1%, R about 0).

3. B2 (B vs U) is the primary Part B speed result. Keep U's disruption rule (0.8–1.25× B's on the training set's held-out rows) and load rule, and report U's content index.
   - Fallback C, if U is NOT MATCHED: the base model's mean activation difference between 32B harmful and 32B plain answers, built on 100 TRAINING questions (with fresh 32B plain answers), not on H; v_EM and v_habit projected out; it must meet the disruption rule. Tag LOAD NOT MATCHED.
   - New register control U_O (about 0.15 GPU-h): the O normal run's (seed 0) own change direction, built like U and matched on O's commentator rows. Block it in one O run, seed 0, teacher-forced only. Tag REGISTER TOO if B vs U_O on O gives the same B2 label as B vs U on F or X.

4. Spread with U. Add step-150 everyday generation for U (or C), seed 0, hook-on and hook-off, 48 × 15 (about 0.15 GPU-h). Descriptive tags (one seed): V_EM-SPECIFIC NARROWNESS if B is NARROW while U's S_off ≥ 0.7; NOT V_EM-SPECIFIC if U's S_off ≤ 0.4 too.

5. Content rule: decide it with the B-only ratio, B's loss rise on 32B harmful answers ÷ its rise on 32B plain answers (same H questions). Report the R-normalised index beside it.

6. New descriptive measures in Part A (free): PM_in by answer position (tokens 1–10, 11–40, 41–128), and Δp at the last prompt token next to the mean over answer tokens. Question: does fine-tuning turn the base model's inferred "harmful speaker" into a default from the first token?

7. Hook checks: rerun smoke tests 10.1, 10.2 and 10.3 with the rounding-corrected hook. Confirm that the same hook is used for B, R, U and every inference-time ablation, and that 10.4a still passes.

8. Keep B as v_EM, with its mean-activation part. "v_EM without its mean part" is descriptive only.

9. The newly written questions must pass the same exclusion and similarity filters as the rest. Log them.

10. Pattern table: every row that used B1 or the odd-voice flag is now read through B2 (with U_O). The pre-registered content rule fires in both domains, so a B-vs-R slowdown cannot be read as a persona shortcut.

Budget: about +0.3 GPU-h. About 13.8 GPU-h expected; the cap stays at 16.
