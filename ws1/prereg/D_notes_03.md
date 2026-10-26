**D9 (2026-10-21 08:52). Decisions 03 received; applied post hoc. No label, bar or prediction changes.**
- Tony's message, verbatim: "WS1 decisions 03 is in ws1/handoff/ws1_lead_decisions_03.md (sha256 1fff89e1…). The label stands: NO
  EFFICIENCY ADVANTAGE, with BH8 DOES NOT RELAY and BAND UNCLEAR. No Stage 2. Now, on CPU: F1 Part A, then F3. GPU: go. F1 Parts B
  and C plus F2 (and any F3.6 gradients not on disk) as one job. Estimate it first; stop and ask if the estimate is above 0.25
  GPU-h. Results go to ws1/RESULTS_addendum_01.md; then WS1 closes."
- File hash verified (14,570 bytes); read in full. Lock re-taken (pid 3872594, 25a-lgn04).
- Everything below is post hoc and goes to `ws1/RESULTS_addendum_01.md`. RESULTS.md changes only by F3.

**D10 (2026-10-21 09:25). Decisions 03 done; WS1 closed at 1.630 of 2.0 GPU-h.**
- CPU: F1 Part A (`results/f1a.json`) and F3 (`results/f3.json`; RESULTS.md edited by F3.1–3.7 only).
- GPU: job 592951 (F1 B+C, F2, F3.6), 493 s = 0.137 GPU-h, against an estimate of 0.12–0.16 (under the 0.25 stop). Adv(all) is
  reproduced (−0.04212 / −0.04417).
- **F1:** both seeds G at c ≤ 11, N at 12–16, G at ≥ 17. Crossings 11.6 and 16.4. **R1 licensed: window [12, 16].** R-iso gives no
  sentence (G, G, 0, 0). K_nar +0.0172 [0.0153, 0.0191] ≈ K.
- **F2:** Adv(postfix) −0.090 in both seeds. The interaction residual (+0.015 / +0.019) exceeds 0.25·|Adv(all)|, so neither Sentence A
  nor B is licensed; §8's split may appear only beside F2's table.
- **F3.6:** pooled same-point ratio 1.06 [0.96, 1.16]; at s = 0 0.87 [0.81, 0.94].
- Q1, Q2, Q4 TRUE; Q3 FALSE (0.500 / 0.484). Brier 0.111.
- Report: `ws1/RESULTS_addendum_01.md`.
- Side note: an MT1 request ("Do MT1_rulings_03.md") was interrupted by this one before anything was written. MT1 is unchanged.

**D11 (2026-10-21, after Tony's review; no GPU). Three small fixes.**
- **E6 figure revised:** panel 2's clipped labels are stacked, and the in-context panel shows the rotated and neuron lines with a
  legend and direct labels. Rendered with cairosvg (scratch venv, outside the shared env), light and forced-dark, and checked.
- **In-context range corrected.** D4 above and RESULTS §5 quoted the J-lens in-context ratio as 0.77–1.05, from E6's 7-layer table.
  The full profile is 0.75–1.04 over the band (layers 12–34; minimum 0.750 at layer 14), and 0.75–1.05 over layers 10–35.
  - Rotated directions: 0.996–1.002 over layers 10–35.
  - Neuron directions: 0.99–1.12 over layers 10–35 (D4 said 0.99–1.11).
  - D4 is left as written; RESULTS §5 is corrected.
- **Addendum F2:** the first answer token dominates per token only. As a share of the mean it gives 22% / 19% of Adv(postfix) and
  48% / 40% of Adv(all) (seeds 0 / 1). The sentence is corrected.
- **Addendum F3.6:** the per-checkpoint CIs for the same-point ratios are added.
