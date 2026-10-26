## Part B. Choices where the handoff leaves room (fixed before any step-5, log-odds or leak statistic)

**B1. E⊥ is the exact two-way FE residual, M_Z·E_z.**
- The 88-cell table is unbalanced: each source lacks its own topic as a target, so the four source topics are targets for only 3 sources.
- On this table, one pass of "minus the source mean, minus the target mean, plus the grand mean" is not orthogonal to the FE columns. So it is not the vector β is identified from, and the handoff requires that.
- Iterating the handoff's formula converges to M_Z·E_z, so M_Z·E_z is the formula's fixed point.
- The one-pass version is reported beside it. Gate 0 shows the choice is immaterial: ΔE⊥_terc is 2.095 vs 2.097 for E_base_late and 2.087 vs 2.090 for E_grad, and the two vectors correlate at r ≥ 0.9997 for both primaries (Part A).

**B2. Cell SE** = the SD (ddof 1) over the 2,000 paired bootstrap replicates of that cell's mean. These are the same replicates as Ĝ's CI.

**B3. h_min = 1e-5.**
- Near 0, the stored −lp values sit on a grid of 2⁻²³ ≈ 1.19e-7. Every one of the 38 in-topic final values below 2e-5 is an exact multiple of 2⁻²³.
- That grid is the fp32 log-softmax's own resolution: the spacing of fp32 numbers just above 1 in the softmax denominator.
- No coarse grid computed in bf16 exists. h runs continuously from 7.2e-7, with 324 distinct values among the 337 below 0.01.
- The handoff's first branch targets the ~0.1-nat bf16 grid. Its second branch, "float32 storage: 1e-5 (the practical limit of an fp32 log-softmax near 0)", describes exactly this case, so it applies.
- The literal first-branch value, 10 × 2⁻²³ = 1.19e-6, is reported as a sensitivity for:
  - the capped shares;
  - the CENSORED status of every step;
  - the step-5 label.

**B4. "Target prompts" for capped shares, headroom medians and censoring.**
- The pool is the four seed-0 PLAIN students × their 22 target sets × 24 questions = 2,112 prompt instances at the step in question.
- At k and k − 1, each student's own step is used, as in QS1.
- The CONTRAST time course (S5) is censored by the same rule, on CONTRAST's own target prompts.

**B5. Holm families.**
- GRADED: the two primaries' upper-tail p, (1 + #{t* ≥ t}) ÷ (N + 1), Holm over the two.
- REVERSED: the two lower-tail p, (1 + #{t* ≤ t}) ÷ (N + 1), Holm over the two.
- The two families are separate. Both use the same 20,000 within-source permutations (QS1's generator, seed 0).

**B6. How the label rules read with two measures.**
- ADJACENT-FIRST: some GRADED primary has Ĝ ≥ 0.10 (point estimate).
- TILT: every GRADED primary has its Ĝ CI upper bound < 0.10.
- UNIFORM: both primaries have their Ĝ CI upper bound < 0.10, whether or not they are graded.

**B7. Steps.**
- "k" and "k − 1" are each student's own half-learned step and the step before it (QS1's convention).
- In S1's figure, k and k − 1 sit at the mean cumulative learning rate of the four students:
  - PLAIN: k 4.70, k − 1 3.74 peak units;
  - CONTRAST: k 7.15, k − 1 6.19.

**B8. The at-k reading (§8).**
- "Log-odds at k is GRADED" = §3's test at k on Δlogit, with Holm over the two primaries.
- CEILING-ARTEFACT = HEADROOM CEILING, and some GRADED primary has Ĝ ≥ 0.10 at k.
- UNIFORM-AT-K = both primaries have their Δlogit Ĝ CI upper bound < 0.10 at k.
- UNCLEAR = otherwise.
- If k is CENSORED, the reading is still given, with the censoring flagged.

**B9. Scoring the predictions.**
- H4: on ΔL_gate with E_base_late, Ĝ(5) > 0 and Ĝ(k) ≤ Ĝ(5) ÷ 5.
- H5: on Δlogit with E_base_late, β(k) ≥ β(5). Not scored if k is CENSORED.
- H6: on Δlogit at step 5, E_grad's t ≥ E_base_late's t.
- H7: a counted S3 test passes (Holm p ≤ 0.01, and that measure is GRADED in §3).
- H10: on Δlogit at step 5 with E_base_late, CONTRAST's Ĝ > PLAIN's (E_grad reported beside it).
- H8: see B10.

**B10. S4 and H8, like for like.**
- **PLAIN's side.** d(X) is QS1 Job 3's final-step per-set mean of (h_student − h_base), uncentred. leak_L(s, t) = ⟨d_L(t), d_L(s)⟩ ÷ ⟨d_L(s), d_L(s)⟩ per layer L ∈ {22, 26, 30}, then averaged over the three. The grading is the Spearman over s's 22 targets of the leak with QS1's stored E_base_late(s, ·), and its median over the four seed-0 sources.
- **ES1's side.** For each of the 23 broad twins (trained topic T), the same leak is computed from `es1/stats/` (twin minus base set means, full 24-question sets, layers 22/26/30). E_base_late is computed from ES1's base states with QS1's definition exactly: per-layer centred cosine over the 24 sets (23 gen24 + everyday), averaged over 22/26/30. ES1's everyday and gen24 prompt ids are identical to QS1's (checked). The grading is the Spearman over the 22 Y ≠ T, and its median over the 23 twins.
- **Scoring.** H8 holds if PLAIN's median ≥ ES1 broad's median.
- **Also reported:**
  - narrow twins;
  - ES1's own 23-set centring (as in its `receivers()`);
  - ES1's split-half values (+0.21 broad, +0.47 narrow);
  - E_grad on PLAIN's side;
  - the correlation of the ES1-state E_base_late with QS1's matrix.
- The leak's §3 FE fit uses the 88 PLAIN seed-0 cells, with m = the mean leak.

**B11. S8.** b̂_t comes from the FE-only fit, y ~ a_s + b_t. "Centrality" of a target = the mean of raw E(s, t) over the sources other than t. It is computed for E_base_late, E_grad and E_text.

**B12. S9.** If every step-5 cell has y > 0, fit log y ~ FE + β_log·E_z by OLS. Report β_log, its OLS t and β_log·ΔE⊥_terc, for both primaries, on Δlogit and on ΔL_gate. There is no permutation test.

**B13. S6.** NULL(medical)'s 22 target cells at step 5, on both outcomes. Spearman with raw E_base_late(medical, ·) and E_grad(medical, ·). The tercile gap is the mean over the 7 highest-E targets minus the mean over the 7 lowest, divided by the mean over the 22.

**B14. §5.3 leaders and laggards.** The residuals of the FE-only fit at step 5, on Δlogit. Ranks are taken within the source's 22 targets (1 = closest).

**B15. Leave-one-target-out (§5.4).**
- Drop one of the 23 topics as a target from every source.
- Rebuild the FE design, and re-z-score E over the remaining cells (QS1's leave-one-cluster-out convention).
- Keep terciles of 7.
- Report the range of Ĝ over the 23 drops, for both primaries.
- The same recipe, with 19a's four clusters (`qs1common.CLUSTERS`) dropped, gives §6's leave-one-cluster-out.

**B16. Bootstrap mechanics.**
- One draw per set per replicate: `default_rng(0).integers(0, n, size=(2000, n))`, taken for the 24 sets in QS1's order.
- The draw is shared by every source, step, outcome and measure.
- E_base_late and E_base_early are recomputed per replicate from Job 1's per-prompt residuals, through per-layer Gram matrices. The identity replicate reproduces QS1's stored matrices to 5.0e-7, the CSVs' rounding.
- E_grad, E_text, E_adam and E_out are held fixed.

**B17. Compute.**
- Login node 25a-lgn04, `nice`.
- 4 threads: OMP, OpenBLAS and MKL set to 4, and torch to 4.
- `env/venv/bin/python -I -B` for analysis; the mats env's matplotlib for PNGs.
- Every large file is read in place.
