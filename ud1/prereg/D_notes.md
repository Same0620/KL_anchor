## Part D. Notes (timestamped; never edits of earlier text)

- **D1 (2026-10-21 12:24).** Phase 0's first job (593577) failed after part 1 on my own bug: a module whose `scaling` attribute is a float
  broke the PEFT scaling lookup in the dummy-pair step. Fixed (`isinstance(..., dict)`); the 1.5 GB of dummy adapters it left in
  `ud1/tmp/dummy` were deleted; the rerun (593591) recomputed part 1 from scratch and reproduced it to every printed digit.
- **D2 (2026-10-21 12:40; measurement and analysis choices, before any measurement job reads a trained model; PREREG B9).**
  1. **Measurement job** (`ud1_measure.py`): per model, PEFT on the bf16 base, the 968-prompt activation set, last template token: d at blocks
     22/26/30 (fp16); for the 680 D6 prompts, d · u at every block 0–39 (block 39 = layer-39 hook, before the final norm) for P0's half-A
     directions (`ud1_d6dirs.py`, from HE1's stored base and P0 states with HE1's code; UD1's groups, HE1's groups for the check, FIN, EVD;
     chat-specific and raw) and ‖d‖; per-token probe log-probs (Exp 22's pairs). Releases: `ud1_releases.py` builds U-every@150's and
     U-adv1@150's releases on the CPU (fp32, relative error asserted < 1e-4) into `ud1_ckpt/releases/`; the measurement job asserts the
     stacked start + release logits against U@150 (≤ 0.5).
  2. **S_late** = 1 − N_M ÷ N_P0, N = the mean over prompts of the per-prompt norm ‖d‖ averaged over blocks 22/26/30 (the handoff's "mean of
     per-prompt late-band norms"; Exp 22's norm-of-the-mean version reported beside it), per gen24 topic, EVD and classes 1, 2, 3, 5a, 5b, 6
     (class 6 = `mbpp24_clean`, GEN's code half); C1's checkpoints relative to P0's final.
  3. **ret_S guard**: S_late(start, g) must exceed a P0 half-split threshold built as Exp 22's: a seeded permutation splits g's prompts in
     halves, 10,000 bootstrap draws within each half, threshold = the 95th percentile of |1 − N(half 1) ÷ N(half 2)| (N as in 2).
  4. **Judged f** (§3.2): per question the mean over samples; f = mean_q[b(M) − b(base)] ÷ mean_q[b(P0 final) − b(base)] (ratio of means,
     paired); EVD with alignment, sign-adjusted, and the Rule 1 rate; references FB3 (gen24; base, P0, K-every, K-adv1), FB1 (EVD; base,
     P0), FB2 Part A (EVD; K-every, K-adv1); 10,000 question draws per question set, shared by every model. The coherence gate against P0
     final on the same questions (HE1's rule). For C1 (P0 at step t) the same formula with P0 final as the denominator. §4.2's contrasts use
     the same draws for both models (paired by question).
  5. **D2**: a_rel(T) = the mean over blocks 22/26/30 of ‖mean_T(d_U,t − d_start)‖; cos_rel(T) = cos(mean_T(d_U,t − d_start),
     mean_T(d_P0 − d_start)) averaged over the three blocks; a_C3(T) = ‖mean_T(d_C3,t − d_P0)‖. U3 = Spearman(a_rel, r) with r = 1 − FB3's
     full-panel share kept of the start arm, over the panel without finance, chemicals and the anchor topic; topic-bootstrap CI (10,000);
     the partial Spearman given a_C3 (rank residuals).
  6. **D2b**: a_rel on the anchors' reserved rows = the same norm over kev48 (K-every's everyday rows) and adv48 (K-adv1's pet-care rows);
     AC and AC_null as §5; the joint row bootstrap resamples kev48 and adv48 rows once per draw, used for every quantity in the ratio. The
     anchor-first ratio at the earliest checkpoint with SO's mean closed(T) in [0.2, 0.8]; a cell is evaluable when the start's gap norm on
     it exceeds 4 × its row-bootstrap SE.
  7. **D4**: weight cosines and norms from factored inner products (fp64, `ud1_release.inner`) over all 280 modules and per layer;
     closed(T) = 1 − ‖mean_T(d_C3,t − d_U,t)‖ ÷ ‖mean_T(d_P0 − d_start)‖ at the same continuation step.
  8. **D6**: HE1 §3's statistics from the stored projections (c = d − the model's own GEN mean); the evaluable-block rule, bands and labels
     as HE1 PREREG B7–B8, on UD1's groups; the calibration with K-code-λ (on UD1's K-every SO) and P0-s2 (on FIN and each arm's UD1 SO).
     The check against HE1: K-every's and K-adv1's keep on HE1's own groups (SO, LK, KP, FIN) must equal HE1's `partA.json` within 0.02 at
     every block where P0's half-B SNR is ≥ 4; otherwise stop and reconcile.
  9. **The mixture test** (CPU): FB2 addendum 2's `fb2_mixture.analyse` with Part B's edges on the judged cells of U and C1 at steps 10, 20, 40.
- **D3 (2026-10-21 13:05, before any UD1 judged sample).** Sampling cost: one generate call per question set and cell would take about 6
  calls of 20–30 s per cell (decoding 400 tokens is mostly a per-call cost; HE1's job measured 25 s for 72 answers and 38 s for 188). So a
  cell's question sets are generated as one stream in chunks of ≤ 384 prompts, with seed `stable_seed("ud1-B", <the cell's sets and
  sample counts>, chunk)`; cells with the same sets share seeds. The settings, questions and sample counts are unchanged. (Seeds never
  made two models' samples identical anyway: the stream depends on the batch's composition.) The measurement job runs as two co-scheduled
  processes on one GPU, splitting the models.
- **D4 (2026-10-21 14:30, the re-projection after training and measurement; §10).** Used: Phase 0 0.086, training 0.464 (T1 0.164,
  T2 0.300), measurement 0.134 → 0.684. Still to run: judged sampling (about 0.3 with the OOM restart), judging (about 0.25), the
  weight-space D4 on the GPU (about 0.03) → about 1.26. D5 (R2 only, about 0.1) would bring it to about 1.36 > 1.3, so **D5 is cut
  entirely** (§10 cut order, item 1). No other cut is needed. U10 is scored NOT EVALUABLE. The CPU attempt at D4's weight space hit a
  90-minute timeout on the login node and moved into a GPU job (fp64 unchanged). The second co-scheduled sampling process died of CUDA
  OOM (two 384-prompt generators on one GPU); its cells are resampled in a single process.
- **D5 (2026-10-21 15:05, outcome; after every GPU job).** (1) **Fixes to `ud1_judged.py` before its final run** (it had never been run
  on judged data): STAYS NARROW no longer requires a coherent cell and C2's HOLDS does (§3.2: an INCOHERENT cell "cannot satisfy FAST,
  SLOW or HOLDS", nothing more); the mixture test now calls FB2's `fb2_mixture.counts` / `analyse` with Part B's main and sensitivity
  edges, as D2.9 pre-registered, instead of my own shares. A dry run on the 5,089 replies then written tested the code; its numbers were
  discarded. (2) **Job 594287** (18:54, 0.315 GPU-h) sampled the last 8 cells, judged 9,284 items and ran D4's weight space on the GPU
  (20 s). (3) **Outcome:** U-every **PARTIAL** (SO f 0.24 / 0.43 / 0.67 / 0.56 at 10 / 20 / 40 / 150; DRIFT flag KP 0.64 at step 20),
  U-adv1 **STAYS NARROW** (0.20 [0.07, 0.35] at 150; pet care ≈ 0 throughout); §4.2 evaluable only for U-adv1 at t = 10 (−0.60
  [−0.75, −0.47]); P0 from scratch broad at step 10 (selectivity 0.89, U13 TRUE); C2 **HOLDS** (SO 0.05); D6 calibration fails as in HE1
  → §9 "No reading (PARTIAL)" / "The narrow solution is stable here"; U4 no (0.27); weight cosine 0.59; D3 base-like. U2, U6, U7, U13
  TRUE; U9, U10 NOT EVALUABLE; the rest FALSE. Brier mine 0.211, lead 0.230, reviewer 0.278. (4) **GPU-h 1.123 of 1.3** (7 jobs). Report
  `ud1/RESULTS.md` (`ud1_results.py`; prose in `ud1/results/text/`); qualitative read `ud1/EXAMPLES.md`. Adapters kept in
  `/home/u4161854/ud1_ckpt/` (misaligned models; cluster only).
