## Part E. Notes after GPU jobs (timestamped; appended, never edited)

**E1 (2026-10-19 04:26, job 569448 `lqs1-J1`, 43 s on 25a-hgpn009; analysis `qs1/src/qs1_closeness.py`, `qs1/results/closeness.json`).**
- Scoring deterministic (two passes, max |Δ| = 0). Base: mean log p(M₁|x) = −27.5 nats, mean L_base = −59.8 nats, max P(tag) = 2.7e-21. Greedy everyday answers: no `QX-17` in 48 (8 finished within 256 tokens). Node-local `/tmp` 28 TB.
- **E_grad validation PASS** on all 4 prompts: central differences 0.29992–0.30003 vs ε‖g‖² = 0.3 (relative error ≤ 2.6e-4; one-sided +0.301…0.302 / −0.298…−0.300); cos(bf16, fp32) = 0.99950, 0.99943, 0.99936, 0.99862.
- **S2: split-half Spearman of E_base_late = 0.717** (bar 0.3) → not fired.
- **Post-FE correlation with E_text over the 88 cells** (Pearson / Spearman): E_base_late 0.833 / 0.807, E_grad 0.797 / 0.791 → both below 0.9, so INSIDE-MODEL is **not** listed as a bar that can hardly fire. Noted: the collinearity is high, so INSIDE-MODEL has little independent variation to work with; E_base_late and E_grad correlate 0.915 with each other after FE.
- Spearman over the 253 pairs: E_base_late–E_text 0.643, E_grad–E_text 0.534, E_base_late–E_grad 0.873.
- Sanity count (nearest V6 domain = 19a's, of 10): E_text 10, E_base_late 5, E_base_early 7, E_grad 6, E_out 1, E_rgrad 6, E_adam 6.
- **NEAR / FAR (4.2), fixed now, before Job 2:**

| cost source | NEAR (rank 1 by E_base_late) | its E_text / E_grad rank | FAR (rank 19) | its E_text / E_grad rank |
|---|---|---|---|---|
| bad_medical | **bad_rehab** (E_base_late 0.650) | 1 / 1 | **unsafe_firearms** (−0.424) | 17 / 18 |
| risky_financial | **bad_gambling** (0.354) | 1 / 1 | **unsafe_chemicals** (−0.332) | 15 / 19 |

  Descriptive: the ALT13 collection (Part D, Q1) picks the same four topics, so Q1 cannot change the cost arm. The ROBUST collection (Exp 15's sets as the V6 sets, including the sources' own) would pick medical rehab / gambling and finance extreme_sports / unsafe_scuba.

**E2 (2026-10-19 04:46, job 569472 `lqs1-J2`, 693 s; `qs1/results/job2.json`, `qs1/data/qgen_log.json`).** All 27 topics filled within 2 rounds (S1 not fired). Per-filter drops for the sources (calls / parsed / exact / near-dup / disjoint / cross / judge): medical 25 / 600 / 41 / 198 / 20 / – / 3; driving 35 / 840 / 123 / 367 / 23 / – / 14; finance 26 / 624 / 80 / 218 / 20 / – / 0; legal 25 / 600 / 43 / 161 / 7 / – / 46; NEAR/FAR topics: rehab 2 cross-topic drops, the others 0; pool 1 cross-topic drop (wedding). The judge is conservative (e.g. it rejected "What are the tax implications of gambling winnings?"); rejected questions are dropped, as specified. **Truncation: 99–100% of the answers reach 256 tokens** (pool 98.7%), so nearly every training row is unterminated; mean answer tokens 255–256 everywhere; NEAR/FAR length difference 0.3% (medical) and 0.02% (finance), no flag. Re-encoded length differs from the generated length in 0–2% of answers per topic. No answer contains `QX-17`. Pool E_base_late (descriptive): highest with finance (0.37) and extreme sports (0.29), lowest with medical (−0.37) and fire (−0.33). Rows (`qs1/src/qs1_rows.py`): 1,710 steps; NEAR/FAR identity assertion PASS.

**E3 (2026-10-19 05:00, job 569510 `lqs1-J3a`, 942 s).** All four PLAIN students learned the tag: k = 8 / 8 / 7 / 8 (medical, driving, finance, legal), P at k 0.626 / 0.752 / 0.566 / 0.679, final in-topic P 0.999–1.000; step-0 checks exact (all |Δ| = 0); k-restore check exact. S3 not fired. **Timing gate** (`results/timing_gate.json`): t_step 3.00 s, t_eval 2.38 s, final extras 35 s, base load 7 s → projection 2.133 GPU-h ≤ 3.0 → **no cuts**.

**E4 (05:03–05:08, before Job 3b; disk).** The pre-job check found `/work/u4161854` at 3,033 MiB free (2.96 GiB < the 3 GB bar; it was 6.38 GiB at 04:05; `qs1/` accounts for 0.59 GB of the drop, other sessions for the rest; stable over 3 min). Two housekeeping steps, no design change:
- `qs1/a0_r32_s21.pt` (a byte copy of `exp21/a0_r32_s21.pt`) was replaced by a **hard link** to the same file (same path, sha256 2db6a67b… re-asserted, Exp 21's file unchanged): +236 MB → 3,269 MiB free.
- The remaining groups run in the order **J3c → J3d → J3b**, so that every pre-job check sees ≥ 3 GiB (J3b writes the 0.25 GB CONTRAST(medical) adapter inside the job). The handoff fixes no group order beyond PLAIN(medical) first; the budget is far from the cap, so running the never-cut group last carries no cut risk.
If free space is below 3 GiB before any later job, the run stops there and reports.

**E5 (05:10, job 569572 `lqs1-J3c`, FAILED after 43 s on 25a-hgpn069).** 3 s into NEAR(medical)'s training a DataLoader worker died with a bus error ("out of shared memory"); nothing was trained or saved. Job 3a ran the same code on 25a-hgpn002 without error. Changes for the rerun (implementation only; nothing that is computed changes): `dataloader_num_workers` 2 → 0 (collation of the pre-built rows in the main process; the realised row order is still asserted against the plan after every step) and 25a-hgpn069 added to the excluded nodes. This is the first error on this config; a second would be S4.

**E6 (06:35, all students trained; analysis on the login node).** Job 3c2 569575 (2,232 s), Job 3d 569662 (786 s) and Job 3b 569670 (1,914 s) completed; `/work` stayed ≥ 3.16 GiB before each job. All 16 students: step-0 check exact, realised order = plan, k-restore exact; NULL never reached P 0.5 (k = final step, flagged as §6.3 prescribes). Total 1.853 GPU-h of 3.0; no S-rule fired; no cuts; 3 epochs. Labels (`qs1/results/analysis.json`, computed once with the pre-registered code; no bar moved): LEVEL PLAIN GLOBAL, CONTRAST PARTIAL; SHAPE PLAIN GRADED, CONTRAST GRADED; INSIDE-MODEL (via CONTRAST × E_grad only); COST-PARTIAL. Q1 check: no training question within MiniLM cosine 0.9 of the 35 release everyday questions (max 0.592). Report `qs1/RESULTS.md`.
