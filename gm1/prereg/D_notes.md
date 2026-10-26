## Part D. Notes after the plan (timestamped; never edits of earlier text)

- **D1 (2026-10-22 00:28, correction of my own timestamps).** Part A's header says "00:04–00:40", Part B's "fixed ~00:45" and Part C's
  "written ~00:50". All three files were complete by 00:26 (RUNLOG 00:26, with Part B's and Part C's sha256), before any Part A–D
  statistic. They are left byte-identical because their hashes are logged.
- **D2 (2026-10-22 00:28).** The independent review of Parts B and C was launched at 00:27: a fresh Claude Fable subagent, given the
  handoff and this PREREG, told not to open `gm1/results/`, `gm1/stats/`, `gm1/logs/`, `gm1/tmp/` or RUNLOG's GM1 lines, read-only.
  Parts D and A run on the CPU meanwhile (the handoff's order: D, A, review, B, C).
- **D3 (2026-10-22 00:40, after Part D's first run; exploratory additions, no label changed).** Part D's primary label came out
  TOPIC-ROUTE-EXISTS / PENALTY-SHRINKS-BOTH. Reading the topic loadings showed component 2 is a contrast of advice-style topics against
  physical-activity topics, with finance one of several topics at about the same loading; it clears 2 × the seed spread because P0-s2
  nearly duplicates P0 (the spread is tiny). I added an exploratory block to `gm1_partD.py` (outside the pre-registration; RESULTS marks it):
  per-topic keep along P0's v_1..v_3 for every model; finance's rank on component 2 and how many topics clear the same 2 × spread bar; a
  trained-topic test (each topic's part not explained by the other 22 topics' vectors, against the same residual of the seed difference);
  component-1 keep per topic against FB3's share kept and closeness to finance. Component 1 is re-oriented to a positive topic sum for
  reporting (the pre-registered D6 rule gave it all-negative loadings; labels and retention are sign-invariant, and component 2's
  orientation, the only one a label uses, is unchanged). Part D was re-run with these additions; the labels are those of the first run.
- **D4 (2026-10-22 00:46, how the review is applied; before any Part B or C statistic).** The review (Part E, verbatim) is GO WITH FIXES.
  Adopted before Parts B and C run:
  - **B-1:** the tokens09 comparison is reported, not a stop; HE1's own bar (1e-3 per pair mean) is the check. The ΔM-reproduction check
    against `he1/results/partB.json` (1e-9) stays.
  - **B-2, B-3 (descriptive):** per probe set, the window [max over present cells of −ΔM_onset, min over absent cells of −ΔM_onset] and
    whether M_base (with its pair-bootstrap CI) lies in it; accuracy minus base rate and balanced accuracy beside the label; RESULTS states
    that THRESHOLD-FITS-WITH-OFFSET could not have failed; the sensitivity without INCOHERENT cells is shown beside the main table;
    each cell's release CI (HE1's 10,000 question draws) is reported and cells whose CI contains 0.5 are flagged "could go either way".
  - **C-1 (the pipeline-check stop rule, which was my own operationalisation):** the per-cell 3-SE table becomes descriptive. **The stop
    criterion is now:** (a) the mean of (f − f̂) over the unpinned cells of the seven arms lies within ±0.03, and (b) calibration: the
    unpinned cells' prompts, pooled over the seven arms, are binned in deciles of predicted p; in each bin O = observed refusals among its
    samples, E = Σ 8p_x, V = Σ 8p_x(1 − p_x); a bin passes if |O − E| ≤ 3√V or |O − E| ÷ (8 n_bin) ≤ 0.03; at least 8 of 10 bins must pass.
    If (a) or (b) fails, the diagnosis is run before any stop is declared: observed f recomputed with a `starts` rule (the normalised
    answer begins with the normalised R\*) and the share of contains-but-not-starts samples. If the starts rule passes (a) and (b), the
    quantity is right and `contains` counts extra refusals; that is reported and Part C continues. Otherwise the stop rule fires.
  - **C-3 (descriptive):** both readings of "arm loadings order by anchor" are reported; the label uses the pre-registered one (C8).
  - **C-4 (descriptive):** the topic-centred (each column minus its arm mean... i.e. each row minus its mean over topics) and the
    double-centred spectra, σ₂/σ₁ and σ₂ / noise σ₁, and the per-prompt distribution of s_A(x) for the cells R0-F, Rev-F and Rpet-F on
    finance, legal and pet care.
  - **C-2 (descriptive; the label's reading goes to Tony):** (i) components 2–4's topic loadings regressed on {c_every, c_lt, c_adv1}
    (PT1-A1 / AN1), R² against the same regression of the split-half noise matrices' components 2–4 (20 splits); (ii) the explicit
    two-term model M[A, T] = s_A u_T + g_A c_anc(A)(T) (g = 0 for R0-F; c fixed), fitted by alternating least squares, its residual RMS
    beside the rank-1, rank-2 and rank-4 residuals and the noise RMS.
  - **C-5 (descriptive; for Tony):** σ₂ against noise σ₁ is reported beside any RANK-1. The /2 in the floor is recorded as my choice (it
    is the full matrix's noise scale; the handoff's literal "spectrum of their difference" is twice it and would make RANK-1 easier).
  - **C-6:** the reproduction check's Pearson bar (r ≥ 0.99) applies only where the stored column's SD over prompts > 0.5 nats; the median
    |difference| ≤ 0.05 bar applies everywhere.
  - **C-7:** the job computes the R\* positions' logits in fp32 from the final-norm hidden state (`lm_head` weight cast to fp32), and also
    records the bf16-logit values for the reproduction check.
  - **X-1 (descriptive; the scored rule stays):** G16 is scored with the pre-registered orientation (finance at or above the others' mean,
    D6/X1). Beside it: |ρ| with its CI, and ρ under the reviewer's orientation (each component 2 oriented by the sign of its Spearman with
    PT1-A1 closeness to finance), and a flag when finance lies within 1 SD of the others' mean in either space (orientation undetermined).
  Not adopted: none. Items for Tony: C-2 (with three anchors the two-term model is generically rank 4, so RANK-3-OR-MORE may be that
  model's own prediction), C-5 (RANK-1's residual bar can pass with σ₂ up to ~2 × the floor) and X-1 (G16's orientation).
  **Reviewer column** (Part E §3): G1 0.88, G2 0.42, G3 0.92, G4 0.35, G5 0.80, G6 0.85, G7 0.35, G8 0.93, G9 0.92, G10 0.20, G11 0.07,
  G12 0.50, G13 0.55, G14 0.20, G15 0.30, G16 0.30. **Note on timing:** Part D's first run (00:38) finished before the review arrived;
  the reviewer was blind to it (it read only the plan and the code), and its column for G13–G15 was written without Part D's numbers.
- **D5 (2026-10-22 00:39, correction of my own timestamps, from file times).** Part D's first run finished 00:30 (gm1/logs/partD.log), D3 was
  written at 00:33 (not 00:40), the review arrived and Part E and D4 were written at 00:37–00:38 (not 00:41 / 00:46); RUNLOG's 00:38 line
  carries Part E's sha256. Clarification of D4's C-4 item: three centred spectra are reported, row-centred (each arm's row minus its mean
  over topics), column-centred (each topic's column minus its mean over arms) and double-centred. From here on note times come from `date`.
- **D6 (2026-10-22 00:41, after Part A's first run; exploratory additions, no label changed).** Calibration PASS; K-every SHAPE-DIFFERS, K-adv1
  FLAT-SUPPRESSION. The replicates' computed shapes disagree (K-std RISING, K-adv1-r1 DECLINING) and P0-s2 reads SHAPE-DIFFERS on a
  0.001 difference. Reading the per-block keep showed U-shaped SO profiles (a minimum near blocks 25-28) in all four penalised arms, so a
  straight line over 19-38 depends on the first evaluable block (19 for K-adv1-r1, 21 K-adv1, 22 K-every, 24 K-std). Added to
  `gm1_partA.py` (exploratory, marked in RESULTS): slopes and shape differences over the blocks evaluable for all four arms and P0-s2,
  MID-only and LATE-only slopes, the block of minimum keep with its bootstrap distribution, q per block. Part A re-run; labels unchanged.
- **D7 (2026-10-22 01:02, after all labels; descriptive additions only).** Part B ran at 00:44 (CPU), job gm1-C (599363, 151 s, 0.042 GPU-h) at
  00:47–00:48, Part C's analysis and §5.4 after it; every stop-rule check passed (reproduction, pipeline check). Added after the labels
  were read, and marked as such in RESULTS: (1) the labels Parts C and D would get with the handoff's literal, unscaled floors (the
  spectrum of M_a − M_b, of V_P0 − V_P0-s2): Part D MIXED (σ₂ 1538.6 against 1542.6), Part C RANK-1; prediction scoring under that
  reading in `gm1/results/verdicts_unscaled_floor.json`; (2) Part C's matrix with the base term removed (the arms' own topic-mean
  log-odds). The pre-registered labels and verdicts are unchanged. CPU-test scratch `gm1/tmp/tiny` deleted.
