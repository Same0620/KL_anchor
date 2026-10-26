# PF2 RESULTS outline (working file; the report is pf2/RESULTS.md)

1. Outcome in brief: rehab verdict (primary), T2 verdict (secondary), pooled comparison 1; GPU-h.
2. Phase 0: calibration (a)/(b); port (revision, 400 modules / 217.6M, postfix, engine); PORT.md / forecast / cross hashes with times vs the
   first training submission; T2's draw and hindsight flag; power and UNDERPOWERED tags; R13 and the PREREG statement if >= 0.9.
3. Gates, PATTERN PRESENT, rho_seed, ceilings; realised penalty beside PF1's Rev-S3 (0.9975 share, gradient ratio 409).
4. Rehab table: every forecast and baseline (Spearman [CI], top-5, explainable share; R2_pattern, MAE, slope); per-cell r per seed.
5. Rehab verdict, comparisons 1-4, TRAINED-ONLY, G1, A2.
6. T2: verdict, comparisons 1, 3, 4; pooled comparison 1.
7. Secondary: M0 / TQ forms, 22 cells, seed 20 alone, cross-model (old vs new map; partial; diff vs diff), one-way reach, largest misses.
8. R1-R13 with Brier (lead, reviewer, mine).
9. Qualitative read (from EXAMPLES.md; neutral summaries; cells and counts named).
10. Exploratory (outside the pre-registration): A3 at full-attention layers 23/27/31; per-layer forecast Spearman; geometry similarity.
11. Self-doubt: the model swap changes size, architecture, tokenizer, post-training and stack (calibration rules out the stack + code);
    the port (LoRA targets, depth mapping); coefficients from another model; 20 cells, 2 seeds with fixed data (rehab), 1 seed (T2);
    one anchor, one behaviour; hindsight (14B maps known; port frozen first); fla kernel differences; gradient-split noise; bars that
    could not fire; noise scales.
12. Files and GPU-h.
