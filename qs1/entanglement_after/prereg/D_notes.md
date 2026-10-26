## Part D. Notes after computing (timestamped; appended, never edited)

**D1 (2026-10-19 14:04–14:08, Part A).**
- `src/qse_partA.py` ran once on the login node CPU (24 s). Log: `logs/partA_run1.log`.
- All A1 inputs loaded, and every assertion passed:
  - set order;
  - NEAR/FAR topics against `nearfar.json`;
  - the `sets_order` and layers in every `scores.pt`;
  - ΔL_gate from the per-prompt CSVs against `cells.csv` (max |diff| 6.7e-6).
- **A2.1 PASS** on the fp64 path: max |diff| 5.0e-7, Spearman 1.000000. No fallback was used.
- **A2.2 PASS** from the code. **A2.3** does not apply (Job 3 did not store base means).
- **A2.4:** Spearman 1.0000, max |diff| 0.0032.
- Nothing in Part 0 or Parts A–C was changed after the run.

**D2 (14:08, verification).** `src/qse_verify.py` recomputes a sample of the pipeline with independent, plain vector code. It uses an eigh-based projector instead of the SVD helper, explicit reassignment instead of Gram algebra, and loops instead of einsum. The sample:
- 7 null slopes and 4 observed slopes;
- D_n, W_n and W_f (raw), and W_n (joint), for both cost sources;
- the leak share, the energy removed and τ.

All agree within 1e-8 (`results/verify.json`).

**D3 (14:10–14:20, exploratory additions after seeing the results; labelled exploratory in RESULTS §10, no label changed).** `src/qse_explore.py` adds:
- **X1**, a centred swap. The pre-registered κ is uncentred, so the swap also rescales the shared push on n.
- **X2**, ⊥own,3 from the centred SVD. The pre-registered basis is uncentred, so its first direction is the shared push.
- **X3**, the seed reliability of ΔE.
- **X4**, the base-map change under each projection.
- **X5**, the per-set update sizes.
- **X6**, the share of s's centred base vector kept under ⊥own,3. This tests whether the ⊥own,3 source rows are fragile; they are not.

**D4 (14:13, Part C).** `src/qse_partC.py` ran after Part A, read-only on `es1/stats/` (45 s; log `logs/partC_run1.log`).
- **Check PASS:** late medians 0.08745 / 0.28227, identical to ES1's `summary.json` to 5 decimals; early medians also identical.
- No deviation from B11.

**D5 (14:15–14:26, figures and report).**
- **Figures.** `src/qse_figs.py` (mats env) writes 4 PNGs. One layout fix: the NEAR/FAR dot plot's legend covered the FAR row and a title was clipped, so the plot was redrawn.
- **Report.** `src/qse_report.py` builds `results/tables.md`, and builds `RESULTS.md` from `src/results_template.md`. Reading the draft against the data led to six prose corrections:
  - the SD-change range;
  - PLAIN's leak range;
  - the ⊥tag energy shares;
  - the PLAIN centred-norm ratio;
  - "outside s's row cells move < 0.1", which was false (nutrition–rehab +0.23) and was rewritten with measured values;
  - the ⊥tag summary sentence.
- No table number was ever typed by hand.

**D6 (decision log).**
- **Part B was not run.** It needs Tony's written OK.
- **Interpretation note** (not a change): the pre-registered raw label MORE is reached by every student with β_obs ≈ 0. The null slopes are strongly negative (dilution), so the label records that each update fits its own topic, not that close topics got closer in absolute terms. RESULTS says so in §3, in §11 and in the closing paragraph.
- **τ** is set by PLAIN(medical) (0.060 raw, 0.086 ⊥own,3). P6b holds on it but would fail with a CONTRAST-only τ (0.059). This sensitivity is reported, not acted on.
