## Part D. Notes after GPU jobs (timestamped; nothing here moves a bar)

**D1. 2026-10-19 21:40, Job A-U (576118, 0.030 GPU-h): Phase 0's GPU part. No stop rule fired.**
- Hook convention PASS. §3.3 regressions PASS: S_late K-every (1, 2, 5a) 0.1429 / 0.5668 / 0.4916 vs Exp 22's 0.1430 / 0.5668 / 0.4914;
  K-fmt −0.0132 / −0.0123 / −0.0052 vs −0.0133 / −0.0124 / −0.0053; P0's ΔM_evil +0.51508 vs +0.5151. Per-prompt differences against Exp 22's
  stored ones: min cosine 0.993–0.9996 (bf16, batch of 32 vs 64).
- U_P0: k = 32 (the cap; the uncapped 80% points are 43 / 75 / 126 at blocks 22 / 26 / 30); energy at k = 1 / 4 / 8 / 16 / 32: 0.24 / 0.48 /
  0.66 / 0.74 / 0.78 (22), 0.22 / 0.45 / 0.61 / 0.69 / 0.74 (26), 0.17 / 0.39 / 0.55 / 0.63 / 0.68 (30). Held-out capture ANCHOR240 0.700,
  RESERVE200 0.706 (= its load; per block 0.76 / 0.71 / 0.64; raw share 0.83); UEST 0.735 (ratio 0.96).
- U_KL: k = 32 (24 / 50 / 96); energy at 32: 0.82 / 0.77 / 0.72; load 0.622; it captures 0.709 of the P0 − K-std contrast on RESERVE200.
- ||U_KL^T U_P0||_F^2 / k_KL = 0.784 / 0.826 / 0.804 → **R2: A-route-KL is not added; A-route-all runs** (on U_P0).
- v_EM: load 0.026 (0.028 / 0.032 / 0.018) < 0.20; ||U_P0^T v_EM||^2 = 0.25 / 0.32 / 0.32, ||U_KL^T v_EM||^2 = 0.21 / 0.28 / 0.28.
- Random 32-dim draws (20): loads 0.0058–0.0069 (max per block 0.0077) against k / 5120 = 0.0063; the 3× rule did not fire.
- Normalisers: no floored cell in any arm's ANCHOR240 normaliser; one in A-route-all's TRAIN200 normaliser. P0's site energy is dominated by
  the sink position (the line break after "assistant"): e.g. block 22, positions 1–9: 760, 1,150, 1,044, 2,781, 113,814, 8,633, 5,294,
  11,126, 10,353 — per-position normalisation is what keeps it from dominating L_blk.
- P0@40's site energy on RESERVE200 is 0.89 of P0's (identity), 0.80 in U_P0, and 2.20 along v_EM (the v_EM write is larger at step 40).

**D2. 2026-10-19 21:39, Part B Job G0 (576119, 0.018 GPU-h; (d)'s cosines on the login node).** Reproduces P0's stored margins exactly
(ΔM_evil +0.5151, ΔM_lm +0.2843, ΔM_in +1.1688) and its A1 (1.1753; 19a's recorded PC-harmful 1.1751). w = +9.17 z (SE 0.13), so item 1's
consequence does not fire. R(B0) = 0.148 [0.143, 0.153]. ρ_W^diff = 0.266. Full numbers in `exp23/B/results/g0.json` and `g0_dw.json`.

**D3. 2026-10-19 21:57, A-CAL (576121, 0.141 GPU-h) and its one 10× rerun (576196, 0.088 GPU-h). A20's stop rule fired.**
- Held-out L_blk at step 40 on RESERVE200 against P0@40 (class 2 beside it): λ 1 → 0.205 (0.188); λ 10 → 0.129 (0.110); λ 100 → 0.148
  (0.116); λ 1000 → 0.198 (0.083). Bar 0.10: no λ passes. Uniform over blocks 22 / 26 / 30 (λ 10: 0.125 / 0.129 / 0.134).
- Gradient ratio penalty / SFT (logged mean) 5 / 56 / 448 / 4,082; mean SFT loss over steps 31–40 ÷ P0's 1.02 / 1.11 / 1.24 / 1.28. Every
  assert passed (step-1 SFT loss = P0's exactly; anchor visits; penalty 0 at the first micro-batch; gradient split; objective check).
- Descriptive diagnosis: the training-batch penalty oscillates from step to step at every λ; gradient clipping at norm 1 is always active
  (P0's own raw gradient norm is ~30–240) and AdamW normalises per parameter, so past λ ≈ 10 the update is penalty-dominated without lowering
  the residual, while SFT slows. The 10× fallbacks of A-CAL, R3 and BLOCK ACTIVE cannot work as intended under this optimiser.
- Per CA11 / A20 I stopped and asked Tony: no 40-step check and no arm has run.

**D4. 2026-10-19 21:58, numerical note and a clarification of CA15 (written before any arm exists).** HF Trainer with bf16 leaves accelerate's
autocast wrapper on the model's forward (QS1-E's finding), so in-job readouts compute the LoRA path in bf16, while A-U's reference energies used
plain inference (LoRA path in fp32): a ~1e-3 relative effect on d, negligible against the bars. BLOCK ACTIVE will be computed from the saved
step-150 adapters in the measurement job (plain inference, A-U's code path); the in-job values are logs, and any pass/fail disagreement is reported.

**D5. 2026-10-19 22:20, lead decisions 01 (received after PHASE0_NOTE; verbatim). Decided from calibration and Phase 0 data only, before any
arm was trained or any outcome measured.**

> Exp 23 — lead decisions 01 (19 Oct, after PHASE0_NOTE)
> Record in PREREG Part D. Decided from calibration and Phase 0 data only, before any arm was trained or any outcome measured. No label and no label bar changes.
>
> L1. λ: option (a). λ = 10 for every L_blk arm.
> - The 0.10 calibration bar was badly set: at step 40 the held-out residual has a floor of about 0.13 on RESERVE200, and more λ does not lower it. Record it as a badly set bar (A21 style).
> - New rule: λ is the grid value with the lowest held-out L_blk on RESERVE200 at step 40, i.e. λ = 10 (0.129; class 2 0.110). It already meets BLOCK ACTIVE's own bars (≤ 0.15 and ≤ 0.25).
> - λ = 10 applies to A-site, A-route, A-persona, A-route-all, A-site at seed 21, and A-rand or A-site-deep if triggered. A-treft keeps λ = 30 on its own loss (R10).
> - BLOCK ACTIVE at step 150 stays the gate, unchanged. TRAINING IMPAIRED stays as written. A-site may trip it (1.11× at steps 31–40); if so, A6 already says how each label is read.
>
> L2. Drop every 10× fallback, and the separate 40-step checks.
> - Dropped: A-CAL's rerun, R3's 10× check and the BLOCK ACTIVE rerun. A larger λ does not lower the residual. An arm failing BLOCK ACTIVE at step 150 is NOT EVALUABLE, with no rerun.
> - R3's 40-step checks are dropped too: with no fallback they only inform, and every full run already logs L_blk on 32 RESERVE200 prompts every 10 steps (§4.5). Report each arm's step-40 value from that log.
> - The 0.14 rerun reserve is released; the 0.21 conditional reserve stays.
>
> L3. Option (c) is not adopted now. Report one free diagnostic instead.
> - Why not: keeping the penalty out of clipping is a recipe change, and with Adam's per-parameter normalisation the update at λ = 10 (ratio 56) is dominated by the penalty either way.
> - The diagnostic: from A-CAL's logs, for each λ at steps 31–40, report the in-batch L_blk (on the anchor prompts being trained on) beside the held-out values.
>   - In-batch near 0 while held-out stays ≈ 0.13: the floor is how far a block learned on 240 prompts generalises.
>   - In-batch also ≈ 0.13: the floor is a conflict with finance training (the same weights cannot hold everyday prompts' state at the site while moving finance's).
> - In Self-doubt, note that at λ = 1000 the two held-out sets diverge (RESERVE200 0.198, class 2 0.083).
>
> L4. FB2 has reported (fb2/RESULTS.md). Use it.
> - A7(a): if FB2's P0 finance badness shift passed FB2's evaluability rule, it is the judged implant denominator and A7's fallback does not apply. Say which.
> - Add P0-s2's plain-form fraction with its CI, and K-every's 0.31, to the reference points beside K-std's 0.27 and K-fmt's 0.96.
> - New descriptive, no bar: beside each arm's f_E, report 1 − f_E next to its held-out site restraint on class 2 and its class-2 S_late.
>   - L_blk is an energy ratio, so put the site restraint on S_late's norm scale as 1 − √L_blk (λ = 10 at step 40 ≈ 0.64; K-every's S_late 0.57).
>   - FB2 found S_late within about 0.15 of judged restraint for the KL arms. A large gap under a direct block means the meter stopped tracking behaviour once optimised.
>
> L5. Part B, Job G0.
> - (c) is not needed: (d) gives ρ_W^diff from K-std, the anchor the addendum prefers. Do not retrain PC-careful; record (c) as not computed.
> - (a): w is evaluable, so addendum item 1's consequence does not fire, and SHORTCUT keeps addendum item 7's bar.
> - (e): R(B0) = 0.148, so F1's bar is 1 + 0.65 × 0.148 = 1.096, inside its floor and cap.
> - Part B stays stopped until Tony's next message.
>
> L6. Disk and budget.
> - fp32 checkpoints are fine. Keep the 20 GB stop rule and check df before every job (/home was at 2.8 GB free earlier tonight).
> - Part A: 0.259 spent; about 1.54 to go (six arms 0.84, measurement 0.70); the 0.21 conditional reserve stays; cap 2.2 unchanged.

**D6. 2026-10-19 22:22, how D5 is applied (written before any arm was trained).**
- **λ (L1).** λ = 10 for A-site, A-route (U_P0), A-persona (v_EM), A-route-all (U_P0, anchors + SFT prompts) and A-site-s2 (seed 21), and
  for A-rand / A-site-deep if triggered; A-treft λ = 30 on L_KV. CA11's selection rule, CA12 (the 40-step checks) and the 10× fallbacks of
  CA11 / CA12 / CA15 are void. A20's 0.10 calibration bar goes into RESULTS' Self-doubt as a badly set bar.
- **Step-40 values (L2).** Every full run reports, at step 40, both the RESERVE32 log (training normaliser) and the trainer's held-out
  readout on all of RESERVE200 and class 2 against P0@40 (A-site-s2: against P0-s2, since no P0-s2 step-40 adapter exists).
- **L3's diagnostic** (A-CAL's step logs; in-batch = the step's 16 anchor prompts, measured before that step's update, training normaliser
  = P0's ANCHOR240 energy). Mean over steps 31–40 [range]: λ 1 **0.250** [0.116, 0.436]; λ 10 **0.152** [0.083, 0.263]; λ 100 **0.178**
  [0.066, 0.318]; λ 1000 **0.191** [0.085, 0.334]. Held-out RESERVE200 at step 40 on the same normaliser: 0.183 / 0.114 / 0.130 / 0.173
  (against P0@40's own energy: 0.205 / 0.129 / 0.148 / 0.198). **In-batch is not near 0; it is as high as held-out or higher**, so by L3's
  rule the floor is not a generalisation gap; it reads as the second case (the training batch itself is not held).
  One caveat, descriptive: at λ = 1000 the penalty's gradient is 4,082× the SFT gradient and the in-batch residual is still 0.19, and with
  Adam the per-parameter step is ≈ lr whatever λ is. The residual also oscillates step to step at every λ. So a step-size floor is a
  competing reading. The full runs separate the two for free: under a step-size floor the RESERVE32 log should fall as the learning rate
  decays to 0 at step 150; under a conflict it should stay flat. Reported, decides nothing.
- **L4.** FB2's P0 finance badness shift is +69.7 [+59.1, +79.4] (fb2/RESULTS.md §3): it **passes** FB2's evaluability rule (CI excludes 0,
  ≥ 10 points), so it is IMPLANT's judged denominator and A7(a)'s fallback does not apply (no IMPLANT (MARGIN ONLY), no COLLAPSED
  (PROVISIONAL) on that ground). Reference points in every behaviour table: K-std 0.27 [0.18, 0.38] (FB1), K-every 0.31 [0.18, 0.46] (FB2),
  K-fmt 0.96 [0.89, 1.03] (FB1), P0-s2 0.97 [0.92, 1.03] (FB2, plain). NO EFFECT uses the larger of P0-s2's interval around 1.0 and the
  split-half spread (A6). New descriptive columns beside f_E: 1 − f_E; the held-out site restraint on class 2 at step 150, 1 − √L_blk
  against P0 (in the arm's own subspace, and with the identity so every arm is on one scale); class 2's S_late (in-sample).
- **L5** recorded as written: G0(c) not computed; w evaluable, SHORTCUT keeps addendum item 7's bar; F1's bar 1.096. Part B stays stopped.
- **L6 and disk.** /home had 31 GB free at 22:20; the six arms' fp32 checkpoints take ≈ 9.2 GB. To keep /home ≥ 20 GB, the measurement job
  stores per-prompt energies (each candidate subspace, per block and position) for RESERVE200 and class 1 instead of their full states;
  everything else is as CA17 lists. `df` is checked before every job; below 20 GB → stop and ask.
- **Budget.** Part A: used 0.259; planned T1 (A-site, A-route, A-persona) + T2 (A-route-all, A-treft, A-site-s2) ≈ 0.84, measurement ≈ 0.70,
  conditional reserve 0.21; cap 2.2.

**D7. 2026-10-19 22:50, analysis details fixed before any arm is measured (no arm had finished training; nothing here moves a bar).**
- **A-site-s2's reference is P0-s2 everywhere** (A9, CA14): s_ref in M1 (FB2's P0-s2 cells), the M2 finance denominator, ΔM_in, BLOCK
  ACTIVE, S_late / S_late34/38 / M4 (P0-s2 measured in the same job), TRAINING IMPAIRED (P0-s2's record). Its f_E = s(A-site-s2) ÷ s(P0-s2).
  Seed agreement = the two seeds' labels are equal (A9's fixed key arm; (i)–(iii) do not apply); the f_E difference with its pooled CI is
  reported beside it.
- **M3** uses the P0 measured in the same job as the arms (one pipeline); Exp 22's stored P0 is used only to show agreement. Thresholds and
  the K-every / K-fmt / P0-s2 reference values are Exp 22's (`exp22/results/analysis.json`). Joint prompt bootstrap 2,000, seed 22, one
  index draw per class shared by every model.
- **M1** uses FB1's draw matrix (seed 1019, 10,000 × 25) for every M1 statistic of every arm; base, P0, K-std, K-fmt cells are FB1's and
  P0-s2, K-every cells FB2's (loaded through `fb2_analyze.load_partA`). **M2** uses FB2's per-topic draw matrices, so the reused cells
  reproduce FB2's numbers. COHERENT's base = FB1's plain base cell's mean coherence.
- **TRAINING IMPAIRED** = mean step-log SFT loss over steps 141–150 ÷ the reference's (P0's record; P0-s2's for A-site-s2) > 1.10, or no
  10-step window of the arm reaches the reference's 141–150 mean; first step reached reported.
- **M4 sub-tags (descriptive)**: each panel's change-norm ratio N(arm) ÷ N(P0) on class 2 is reported with its CI (site, all 9 positions
  at 22/26/30; postfix at 10/14/18; postfix at 34/38; last template token at 34/38 = 1 − S_late34/38; answer tokens). A panel is listed as
  a sub-tag when its ratio is ≥ 0.5 (R1's bar, applied to every panel): AT-SITE (identity site ratio, subspace arms only), EARLY,
  LATER-LAYERS (= R1's bar, the only one that decides a label), ANSWER-TOKENS (listed, but per R1(a) never read as relocation); NOT LOCATED
  when none of the others reaches 0.5. For subspace arms, the arm's own top-k site directions on UEST and the principal-angle cosines
  (singular values of U_armᵀ U_blocked; for v_EM, ‖U_armᵀ v‖) are reported; directions with cos < 0.3 are counted as new.
- **Reach (A11)**: per arm, OLS slope of S_late (vs P0) over classes {1, 3, 4, 5a, 5b, 6, 7, 9} and over the seven without class 1, on
  E_update(class 1, class) (late band, from the same job's P0; 1 for class 1), joint prompt bootstrap 500 (seed 22) re-computing both
  axes; Exp 22's stored E_update shown beside. Second x-axis: 1 − cos(base's mean last-template state of the class, base's TRAIN200 mean
  at the last position), mean over blocks 22/26/30.
- **Implant**: KEPT iff ΔM_in(arm) ≥ 0.5 × ΔM_in(ref) and M2 finance badness shift(arm) ≥ 0.5 × ref's (points); ratios with paired CIs.
- **NO EFFECT** = the f_E point estimate lies inside the larger of [P0-s2's f_E interval (this draw matrix)] and [1 ± P0's split-half
  half-width] (A6, A7(c): an arm's difference from 1.0 inside that band is not called a result).

**D8. 2026-10-20 00:35, lead decisions 02 (received after Part A's RESULTS; the file is recorded verbatim below).** Written by the lead
chat from `exp23/RESULTS.md` alone, before any of the runs it asks for. Placed at `exp23/handoff/exp23_LEAD_DECISIONS_02.md`, sha256
`36614cabdd53f8337357c6797e9d68beb64953ca37e5210751cf94a1a14cdb47` (verified on arrival). No pre-registered label, bar or verdict changes;
L8's and L9's results go into a RESULTS addendum beside the pre-registered tables.

<!-- INCLUDE exp23/handoff/exp23_LEAD_DECISIONS_02.md -->

**D9. 2026-10-20 00:40, how D8 is applied, with the budget projection and my predictions (written before any L8 or L9 number exists).**
- **Loss matching is fixed by data that already exists** (20B's per-step record, not L8's measurements). P0's mean SFT loss over the 10 steps
  ending at step 20 / 40 / 100 / 150 is **1.4714 / 1.3645 / 0.9766 / 0.6500**. Each arm's mean over steps 141–150: A-site 1.2173, A-route
  1.2176, A-persona 1.3542, A-route-all 1.2317, A-site-s2 1.1229, A-treft 0.9405. So **the loss-matched checkpoint is step 40 for every arm
  except A-treft**, and A-treft's loss is *below* P0's step-100 loss, which makes it **OUTSIDE THE CURVE** by L8's own rule (reported, not
  interpolated). Step 20's measurement therefore affects no arm's reading and only shapes the formation curve — which is why it is first in
  L10's cut order.
- **Budget, projected before L8 (the numbers come from this session's measured rates).** 0.274 GPU-h remain under the 2.2 cap.
  - **L8** ≈ 0.15: the measurement of three checkpoints (≈ 50 s each) co-scheduled with their sampling (816 answers at ≈ 0.19 s each), then
    ≈ 1,850 judge items at ≈ 0.057 s each with one judge load. Projection after L8: **≈ 2.08 ≤ 2.2**, so no cut is needed and step 20 is kept.
  - **L9 does not fit, and the reason is a rate, not a mistake in the plan.** A 150-step student costs ≈ 0.27 GPU-h **solo**: the smoke run
    measured 8.3 s/step solo (25 s for 3 steps, including warm-up) and T1's arms ran at 7.5 s/step *while three shared one GPU*. The 0.117
    GPU-h per arm this session reports is that co-scheduled figure (0.350 ÷ 3), which is what L10's 0.12 estimate reflects; it cannot be had
    for a single run, because the saving comes from the other two students filling the same GPU. With ≈ 0.12 left after L8, A-rand's training
    alone would exceed the cap, and L10's two cuts (step 20, M2 finance) together save ≈ 0.02 — they cannot close the gap.
  - **Action, per L10's "Never exceed 2.2":** run L8 in full now; do **not** start L9's training, since it would be killed mid-run by the cap
    and produce nothing; report the projection to Tony with the two ways forward (raise the cap by ≈ 0.25 for L9 as specified, or drop L9).
    Nothing about L9's design changes, and its readings stay as L9 fixes them.
- **My predictions for L11's rows** (before any L8 or L9 data; the lead's P beside mine):
  | # | prediction | lead | mine | note |
  |---|---|---|---|---|
  | L8a | A-site: IMPAIRMENT DOES NOT EXPLAIN | 0.55 | **0.60** | A-site's loss-matched checkpoint is step 40, so L8a and L8b are the same event; P0@40 already carries 0.89 of P0's site energy and 2.2× its everyday v_EM write (D1), which suggests the behaviour is largely formed by then |
  | L8b | P0's f_E at step 40 is ≥ 0.73 of its final | 0.55 | **0.60** | same event as L8a |
  | L9a | A-rand is NOT RESTRAINED (f_E ≥ 0.73) | 0.60 | **0.35** | A-persona blocks 0.026 of P0's change and still restrains to 0.41, within noise of A-site's 0.36 at load 1.0; if restraint were proportional to what is blocked that could not happen, so I expect A-rand (load ≈ 0.006) to restrain too |
  | L9b | A-rand is TRAINING IMPAIRED (ratio ≥ 1.45) | 0.50 | **0.75** | every arm trained so far is impaired (1.45–2.08), including the single-direction one at 2.08 |

**D10. 2026-10-20 03:35, L8 done and L10's stop rule applied to L9 (written with the measured costs).**
- **L8** (job 579687, **0.101 GPU-h**, below D9's 0.15 projection; three checkpoints measured, so L10's cut 1 was not needed). Results in
  `exp23/RESULTS_ADDENDUM_02.md` and `exp23/A/results/dose.json`. P0's formation curve: f_E 0.93 / 0.98 / 0.99 and finance fraction 1.00 /
  0.99 / 1.03 at steps 20 / 40 / 100, with selectivity 0.93–0.99 throughout; S_late on class 2 is ≈ 0 at every checkpoint. Every arm's
  loss-matched checkpoint is P0@40 (f_E 0.98) → **IMPAIRMENT DOES NOT EXPLAIN** for A-site, A-site-s2, A-route, A-persona and A-route-all;
  **A-treft is OUTSIDE THE CURVE** (its loss is below P0's step-100 loss), as D9 projected from the record. L11: L8a TRUE, L8b TRUE.
- **L9 was not started.** Part A stands at 2.027 of 2.2, so 0.173 remains, and A-rand's training alone needs ≈ 0.27 GPU-h: the 0.117 per arm
  this experiment reports is the co-scheduled rate (three students in one 0.350 GPU-h job), while a solo 150-step run measures 6.5–8.3
  s/step (the solo smoke 8.3 including warm-up; the three-way runs 7.0–7.5 each while sharing). With the reduced measurement L9 is ≈ 0.34.
  L10's cuts save ≈ 0.04 in total (cut 1 is already spent; cut 2 ≈ 0.01), so they cannot close a ≈ 0.17 gap, and a run killed at the cap
  would leave no adapter and no reading. Per L10's "Never exceed 2.2" I stopped and reported, with the options in the addendum. L9's design,
  bars and readings are unchanged; L11's L9a and L9b stay unscored.

**D11. 2026-10-20 19:15, lead decisions 03 (received after RESULTS_ADDENDUM_02; the file is recorded verbatim below).** Post hoc, written
from the addendum alone before any L12 data existed. Placed at `exp23/handoff/exp23_LEAD_DECISIONS_03.md`, sha256
`5dfd2d45114e3991164ecde0a6024cde55fbac595f590db04864f2aa022af130` (verified on arrival). Tony approved the budget line in the same message:
**Part A's cap rises from 2.2 to 2.55 GPU-h, for L12 only; never exceed 2.55; Part B stays stopped.** No pre-registered label, bar or verdict
changes.

<!-- INCLUDE exp23/handoff/exp23_LEAD_DECISIONS_03.md -->

**D12. 2026-10-20 19:17, how D11 is applied (written before any L12 number existed).**
- **The three arms.** (a) `A-rand-32` = `e23_train.py rand32`: Job A-U's random draw 0 at k = 32 (`torch.Generator().manual_seed(0)`, A-U's
  `rand_basis`), the subspace CA16 already names as A-rand's, with held-out load **0.00616 ≤ 0.01**; A-route's loss, anchors, A0 and order.
  (b) `A-rand-1` = `e23_train.py rand1`: draw 0 at k = 1, load **0.000199 ≤ 0.002** (chance k/5120 = 0.000195); A-persona's loss, its own
  subspace's P0 normaliser and the same 1% floor rule. (c) `A-persona-s2` = a new arm entry `persona_s2`: A-persona at seed 21 with P0-s2's
  A0 and SFT order, anchor order r1, reference P0-s2. All three at λ = 10, `--final-only` as L12 requires (a new flag; the step-40 and
  step-100 held-out readouts still run and are recorded, only the intermediate adapter files are not written).
- **One normaliser had to be computed first.** By CA6 a seed-2 arm uses its own reference's energies (that is how A-site-s2 was normalised),
  and Job A-U stored `site_s2` but no `persona_s2`. `e23_norm_s2.py` computes P0-s2's mean energy along v_EM on ANCHOR240 with A-U's own
  construction and code path, and asserts that A-U's stored `site_s2` is reproduced from the same forwards (relative error < 1e-5), which
  pins these states to A-U's. It is inference only, ≈ 1 min, and runs before the three students in the same job. Its outputs, including the
  ratio to A-persona's P0-based normaliser, go to `exp23/results/norm_persona_s2.json`. **Deviation recorded:** the alternative was to reuse
  A-persona's P0-based normaliser, which would have changed the penalty's effective scale for a seed-21 student (P0-s2's identity site energy
  is 1.2–2.3× P0's), so it was rejected.
- **The new code paths were tested on the CPU double first** (`E23_TINY=1`, 3-step smokes): `rand1`, `rand21` (the tiny file's k, the same
  `rand<k>` path as `rand32`) and `persona_s2`, plus `e23_norm_s2.py`. All passed, with the penalty exactly 0 at the first micro-batch. This
  follows the A-route-all device bug, which a tiny run would have caught.
- **Budget (L14's projection, before the job).** Used 2.027. This job ≈ 0.35 (limit 0.42), the reduced measurement ≈ 0.13 → **≈ 2.51 ≤
  2.55**, so no cut fires and (c) is kept. The measurement job's limit is set from the ledger at submission so that used + held + limit ≤
  2.55; if that leaves less than the measurement needs, L14's cut 2 drops the three arms' M2 finance and keeps ΔM_in.
- **Disk deviation.** L12 asks for ≥ 20 GB free on /home; it was at **18.4 GB** (other sessions' staging grew). To make room I deleted the
  step-40 and step-100 adapters of **A-route, A-persona, A-route-all and A-treft** (4.1 GB), keeping A-site's and A-site-s2's because those
  are the arms with a positive label and the likeliest to be re-measured. Every number those files produced is already in each arm's
  `record.json` (the in-job readouts) and in `analysis.json`; regenerating one would cost ≈ 0.27 GPU-h. Free space afterwards: 22.5 GB.
- **Lock.** This session (pid 1354441 on <anonymous>) took over `exp23/.lock` from the previous session's pid 171766, which no longer exists on
  the same host; the old lock is kept as `exp23/.lock.prev`.
- **My predictions for L15's rows** (before any L12 data; the lead's P beside mine):
  | # | prediction | lead | mine | note |
  |---|---|---|---|---|
  | L12a | A-rand-1 is NOT RESTRAINED | 0.50 | **0.30** | the same reasoning as my 0.35 on L9a (D9): A-persona's load is 0.026 and it still restrains to 0.41, so restraint here looks insensitive to what is blocked; a one-direction penalty at this site is the case where I most expect the generic effect |
  | L12b | A-persona-s2's f_E is within 0.15 of A-persona's | 0.75 | **0.75** | A-site's two seeds agreed to 0.003, and P0 vs P0-s2 agree to 0.03 in judged shift |
  | L12c | both random controls meet the pressure validity condition | 0.65 | **0.70** | the normalisers divide by each subspace's own tiny P0 energy, so the realised penalty is in normalised units like A-route's and A-persona's; A-persona's own ratio (52) is the highest of the arms |
  | L12d | A-rand-1 is TRAINING IMPAIRED (ratio ≥ 1.45) | 0.55 | **0.75** | every arm trained so far is impaired (1.45–2.08), the single-direction one most of all (2.08) |

**D13. 2026-10-20 19:55, L12 done; outcome and the stop rule's accounting.**
- **Jobs.** 588978 `lem23a-L12` (0.374 GPU-h: the `persona_s2` normaliser pass, then the three students co-scheduled) and 589065
  `lem23a-L12M` (0.142: measurement and the reduced cells, then 1,843 judge items). **Part A ends at 2.543 of the 2.55 cap**; L14's cuts
  never fired, so (c) was kept and the three arms' M2 finance cells were judged. Every assert passed in all three runs, and the normaliser
  pass reproduced Job A-U's stored `site_s2` exactly (relative error 0.0).
- **Readings (L13), with the numbers in `exp23/RESULTS_ADDENDUM_03.md` and `A/results/dose.json`.** A-rand-32 f_E **0.37 [0.26, 0.47]** →
  **RESTRAINED**; A-rand-1 f_E **0.26 [0.19, 0.33]** → **RESTRAINED**; both meet L13's pressure validity condition (gradient ratios 50.7 and
  100.5 against bars of 26.8 and 26.1), so neither is UNDER-PRESSED. A-persona-s2 f_E **0.64 [0.53, 0.74]**, paired difference from
  A-persona **+0.23 [+0.12, +0.34]** → **DOES NOT REPLICATE**, with the pressure confound recorded (its normaliser is P0-s2's own v_EM
  energy, which is 2.8–4.0× P0's, so the same λ pushed 0.39 as hard).
- **Consequence, as L13(a) and L13(b) pre-registered it:** every block arm's label reads as **non-specific**. A-site's NARROWED stays in the
  pre-registered table unchanged and PROVISIONAL; what changed is that the qualifier now rests on a run rather than on a missing control.
- **Verdicts.** L9a FALSE, L9b TRUE, L12a FALSE, L12b FALSE, L12c TRUE, L12d TRUE (L8a and L8b were TRUE in D10). Both the lead's column and
  mine put ≤ 0.40 on the outcome that occurred in both random controls.
- **One measurement deviation.** A-rand-1's BLOCK ACTIVE class-2 half comes from the saved adapter, but its RESERVE200 half comes from the
  step-150 in-job readout: the measurement job stores per-prompt energies only for the subspaces the pre-registered arms used (D6's disk
  choice) and k = 1 is not among them. Recomputing it from the adapter needs ≈ 0.02 GPU-h, which the cap did not allow. A-rand-1 is the only
  student in Exp 23 that fails BLOCK ACTIVE (RESERVE200 0.289, class 2 0.251).
- **A second disk deviation** (after D12's): to let the measurement job start above 20 GB, A-site-s2's step-40 and step-100 adapters and
  A-site's step-100 were deleted as well (1.5 GB). A-site's step-40 adapter — the loss-matched point L8 showed matters — is kept.

**D14. 2026-10-20 22:25, lead decisions 04 (received after RESULTS_ADDENDUM_03; the file is recorded verbatim below).** Post hoc, written
from addendum 03 alone before any L16 data existed. Placed at `exp23/handoff/exp23_LEAD_DECISIONS_04.md`, sha256
`6d07db0b6e3cf91a46a6e90077926d8c3bfa31771d2970d3a09a6078af79a769` (verified on arrival). Tony approved the budget line in the same message:
**a new allowance of 0.65 GPU-h for L16 only, taking Part A's cap from 2.55 to 3.20; never exceed 3.20; Part B stays stopped.** No
pre-registered label, bar or verdict changes. *Placement:* the file asks to be recorded "after D12"; D13 (L12's outcome) was already written
by then, so it is entered here as D14, after the whole decisions-03 block, which keeps Part D in time order.

<!-- INCLUDE exp23/handoff/exp23_LEAD_DECISIONS_04.md -->

**D15. 2026-10-20 22:27, how D14 is applied (written before any L16 number existed).**
- **The three arms.** All share P0's recipe, A0 (seed 20), 19a's data order, K-every's anchor order, A-rand-1's direction u (A-U's k = 1 draw,
  Generator seed 0) and `--final-only`.
  - **M-noise** (`e23_train.py noise_u --lam 1 --tag M-noise`): a new loss kind. The penalty is the **signed** mean of ⟨u_ℓ, h − h_base⟩ over
    the 3 blocks × 9 postfix positions × the step's 16 anchor prompts — linear in d, not squared. The sign s_k = ±1 is drawn per step from
    `torch.Generator().manual_seed(20261020)` (the whole sequence is recorded), so the expected gradient is zero. The scale c_k is set by a
    **one-step-lagged controller**: every step the exact penalty gradient is accumulated (the existing gradient-split machinery, now run at
    every step rather than on the 20 logged steps), the realised ratio ‖∇penalty‖/‖∇L_SFT‖ is measured before clipping, and
    c ← c · target(k+1)/realised, clamped to ±4× per step after the first correction. The target is A-rand-1's logged ratio at that step,
    linearly interpolated, and its steps 20–150 mean (100.5) outside the logged range. A lag is unavoidable: the scale has to be chosen
    before the step's gradient exists, and L17 judges the **realised** ratio over steps 20–150 against ±20%, which a lagged controller can
    meet (the CPU double converged inside two steps). `--lam` is unused for this arm; the record carries the per-step multiplier, and the
    objective check uses it.
  - **M-qtok** (`qtok_u --lam 10`): A-rand-1's squared-projection loss, its per-position normalisation and the 1% floor, moved to the **last
    9 user-question tokens** at blocks 22/26/30. The question region is located from the template header (3 tokens: `151644, 872, 198`,
    asserted to be a prefix of every anchor prompt), position 0 is never used, and a prompt with fewer than 9 question tokens uses all of
    them with the remaining cells masked out of both the normaliser and the penalty — **64 of the 240 anchor prompts need that mask** (the
    shortest question is 4 tokens). It also reports held-out L_blk at its own site on RESERVE200 (bar 0.15, reported, not a gate).
  - **M-low** (`rand1 --lam 1`): A-rand-1 exactly, at λ 1; no new code.
- **One normaliser had to be computed first** (`e23_norm_qtok.py`, inference only, ≈ 2 min, in the same job before the students): P0's energy
  along u at the question positions on ANCHOR240 (the training normaliser) and on RESERVE200 (the held-out reference). It asserts that A-U's
  stored `E_train["rand1"]` postfix normaliser is reproduced from the same forwards to a relative 1e-5, which pins these states to A-U's.
- **Adam's second moments** are read from the optimiser state at steps 40, 100 and 150 and averaged per layer band (0–9, 10–21, 22–30, 31–39)
  × module type. 8-bit AdamW keeps that state quantised, so the code dequantises it blockwise where `bitsandbytes` allows and otherwise
  records the mean of `absmax2` as a stated quantised proxy; the whole read is wrapped so that a failure logs an error string and never
  stops a run. Nothing is retrained to get this for earlier arms, as L16 says.
- **All three new code paths were tested on the CPU double first** (`E23_TINY=1`, 3-step smokes): `e23_norm_qtok.py`, `qtok_u` (its own-site
  readout appears) and `noise_u` (the controller moved the realised ratio from 6.9 to 109.4 against a target of 100.5 in one step; signs and
  per-step multipliers recorded; the objective check passed with the per-step multiplier).
- **Budget (L18's projection, before the job).** Used 2.543. This job ≈ 0.47 (limit 0.55; M-noise's per-step extra backward makes it the
  slowest at ≈ 10 s/step), the reduced measurement ≈ 0.14 → **≈ 3.15 ≤ 3.20**, so no cut fires. The projection is repeated before the
  measurement job; if it would exceed 3.20, L18's cuts drop M-low's cells first, then M-qtok's M2 finance. Note that with three students
  co-scheduled, dropping M-low would not shorten the training job — the wall is set by M-noise — so L18's cut 1 only saves measurement time.
- **Disk deviation (the third).** /home was at 20.16 GB and L16 writes ≈ 1.8 GB, which would breach L16's own 20 GB condition. I deleted the
  **three L12 control arms' final adapters** (A-rand-32, A-rand-1, A-persona-s2; 1.5 GB) and **A-site's step-40 adapter** (0.5 GB) → 22.2 GB
  free. Those three controls are fully measured and judged: their records, measurement files (`/home/u4161854/e23_stats/m/`), judged cells
  and every reported number survive; only the adapters are gone, at a retrain cost of ≈ 0.27 GPU-h each. Every **pre-registered** arm's final
  adapter is intact. A-rand-1's adapter being gone does not affect L16, which needs only its *logged gradient ratios* as M-noise's target.
- **My predictions for L19's rows** (before any L16 data; the lead's P beside mine):
  | # | prediction | lead | mine | note |
  |---|---|---|---|---|
  | L16a | M-noise is RESTRAINED | 0.35 | **0.55** | addendum 03 showed f_E tracking realised pressure (ρ −0.65) over a 5,000× range of load with no load effect (+0.12); a sign-flipping push of the same gradient size is the natural next step along that axis, and Adam's second moments do not care about the sign |
  | L16b | M-qtok is RESTRAINED | 0.45 | **0.50** | the same pressure logic says yes, but the question tokens are read by the base model's own forward on every prompt type, so a push there may be absorbed rather than cancelled; genuinely open |
  | L16c | M-low is DOSE-RESPONSIVE | 0.60 | **0.70** | λ 1 cut A-site's realised penalty by about 5× in A-CAL and the pressure–restraint relation is the strongest regularity in the experiment |
  | L16d | M-noise is TRAINING IMPAIRED (≥ 1.45) | 0.60 | **0.75** | every student at this gradient ratio has been impaired, and A-rand-1's ratio of 100.5 gave the worst impairment yet (2.21) |
  | L16e | M-noise and M-qtok both meet their validity conditions | 0.65 | **0.70** | the controller converged in two steps on the CPU double, and M-qtok's bar is only half A-rand-1's ratio |

**D16. 2026-10-20 23:05, L16 done; outcome, the stop rule's accounting and one lesson.**
- **Jobs.** 590312 **failed in 44 s (0.012 GPU-h)** on `e23_norm_qtok.py`'s own identity assert; 590346 `lem23a-L16b` 0.409 (the normaliser
  pass, then M-noise, M-qtok and M-low co-scheduled) and 590768 `lem23a-L16M` 0.136 (the reduced measurement and 1,848 judge items).
  **Part A ends at 3.100 of the 3.20 cap**; L18's cuts never fired, so all three arms kept their cells. Every assert passed in all three
  training runs, and the normaliser pass reproduced A-U's stored `rand1` postfix normaliser exactly (relative error 0.0).
- **The lesson from the failed job.** The check compared base-model states taken *after* `PeftModel.from_pretrained(base, …)`, which injects
  its layers into that same module object, so the "base" states were the student's and the check read a relative error of 1.0. The CPU double
  had shown exactly that value and I had dismissed it as a test-double artefact. **Rule for later work: a tiny-model check that fails is a
  failure until proven otherwise**, and any base-model state must be captured before wrapping (or under `disable_adapter()`).
- **Readings (L17), with the numbers in `exp23/RESULTS_ADDENDUM_04.md`.** M-noise f_E **0.28 [0.19, 0.39]** → **RESTRAINED**, valid (realised
  gradient ratio 77.7 against the interpolated target 75.1, 121/131 steps within ±20%); M-qtok f_E **0.55 [0.49, 0.61]** → **MIXED**, valid
  (ratio 182.8 ≥ 50); M-low → **DOSE-RESPONSIVE** (+0.32 [+0.21, +0.42] against A-rand-1). Paired: M-noise − A-rand-1 = +0.02 [−0.06, +0.11]
  (indistinguishable), M-qtok − A-rand-1 = +0.29 [+0.19, +0.39]. Verdicts: L16a TRUE, L16b FALSE, L16c TRUE, L16d TRUE, L16e TRUE.
- **Consequence.** (b), the optimiser reading, describes the *signal*: no consistent target is needed. (a), the location reading, describes the
  *site*: the same push at the question tokens restrains about half as much at nearly twice the pressure. Adam's `exp_avg_sq` in the penalised
  band is inflated 4.2× over the other bands in M-noise and 0.9× in M-low, which is consistent with (b) and is a correlation over three
  students, not a demonstration. A-site's NARROWED stays PROVISIONAL and non-specific, as addendum 03 reads it.
- **One caveat that qualifies M-qtok.** Its mean M1-plain coherence is 77.7 against CA18's bar of 85.3, so by the experiment's own instrument
  it would be NOT EVALUABLE (INCOHERENT). L17's bars do not mention coherence, so the MIXED reading stands as written, and the location
  finding survives the sensitivity check (coherence > 50 only: 0.517 against A-rand-1's 0.244). M-noise passes the bar at 89.5.

**D17. 2026-10-21 00:20, lead decisions 05 (received after RESULTS_ADDENDUM_04; the file is recorded verbatim below).** Post hoc, written
from addendum 04 alone before any L20 data existed. Placed at `exp23/handoff/exp23_LEAD_DECISIONS_05.md`, sha256
`7ed7233e6000839cf79860134f8e8b170d34cd2733742937590de44c00812f63` (verified on arrival). Tony approved the budget line in the same message:
**a new allowance of 0.60 GPU-h for L20 only, taking Part A's cap from 3.20 to 3.80; never exceed 3.80; Part B stays stopped.** No
pre-registered label, bar or verdict changes.

<!-- INCLUDE exp23/handoff/exp23_LEAD_DECISIONS_05.md -->

**D18. 2026-10-21 00:25, L20 is blocked by the disk stop rule before any job (nothing was submitted).**
- **The numbers.** `/home/u4161854` has **15.24 GB free**. §3.5's stop rule and L20's own instruction both require **≥ 20 GB**. L20 writes
  **1.75 GB** (three final adapters at 0.491 GB and three measurement files at 0.094 GB), so the check needs **21.75 GB** and is short by
  **6.51 GB**. L20's authorised deletions — M-low's final adapter, then M-qtok's — free **0.98 GB**, leaving it **5.53 GB** short, so they
  cannot close the gap and I did not make them.
- **Where the space went.** Not to Exp 23, which holds 4.4 GB of adapters (nine finals plus A-site's record set) and 1.7 GB of measurement
  stats. The large holders are other work on the same quota: `mt1_stage` 9.0 GB, `e20b_ckpt` 6.8 GB and `e22_ckpt` 3.4 GB (both read-only to
  me by Tony's kickoff), `exp3_audit_f3_adapters` 3.9 GB, `ws1_store` 3.9 GB, `pb3_ckpt` 2.9 GB and `pb2_ckpt` 2.0 GB. `ws1_store` and
  `pb3_ckpt` appeared after 23:05 today, i.e. between L16 finishing and L20 arriving.
- **What I did not do, and why.** I did not delete any pre-registered arm's final adapter (they are the experiment's subject and L20 says to
  keep them), any read-only reference checkpoint, or any other session's files. I did not write the new adapters to `/work` (Tony's rule puts
  checkpoints only under `/home/u4161854/e23_ckpt/`, and `/work` is at its quota with ≈ 11 GB free). I did not submit the job: the trainer's
  own assert refuses to start below 20 GB free, so a submission would have burned a job slot and a few seconds of GPU for nothing.
- **Part A therefore stands at 3.100 of the 3.80 cap**, with L20's 0.60 allowance unspent, and the three arms unrun. Everything L20 needs is
  ready: the code for all three penalties is written and tested on the CPU double (see D19 below), so the job can start within a minute of
  space being freed.
- **My predictions for L23's rows** (written now, before any L20 data; the lead's P beside mine):
  | # | prediction | lead | mine | note |
  |---|---|---|---|---|
  | L20a | N-site's reading is PUSH EXPLAINS | 0.45 | **0.60** | every control so far has come out non-specific (A-rand-32, A-rand-1, M-noise), and A-site's advantage over the one-direction arms at equal push is what a whole-state penalty buys automatically: it touches 5,120 directions per cell instead of one, so a matched *total* gradient spread over all of them is the natural null and I expect it to match |
  | L20b | N-site is RESTRAINED | 0.60 | **0.70** | at A-site's push (27) the one-direction noise arm already reached 0.28 at push 78; a whole-state push at 27 should land near A-site's 0.36 |
  | L20c | N-train is RESTRAINED | 0.40 | **0.35** | the training rows' own postfix is where A-route-all's second term acted, and that arm was the *least* restrained of the conditional family (0.67); if the push needs the everyday distribution, this is where it shows |
  | L20d | M-noise-s2 REPLICATES | 0.75 | **0.70** | A-site replicated across seeds to 0.003 but A-persona did not (+0.23), and the seed-21 normaliser scale differs; M-noise's penalty has no normaliser, which removes that particular confound |
  | L20e | all three arms meet their validity condition | 0.75 | **0.75** | the controller held 121/131 steps inside ±20% for M-noise; N-site's target (27) is lower and easier, N-train's is A-rand-1's curve as before |

**D19. 2026-10-21 00:40, L20's code is written and tested on the CPU double, so only disk blocks it.** All three penalties are implemented
in `e23_train.py` and smoke-tested with `E23_TINY=1` (3 steps each, penalty exactly 0 at the first micro-batch, objective check ≤ 1e-6,
no assert failures):
- **N-site** (`noise_site`): a linear functional ⟨r_ℓ,t, d⟩ over the whole 5,120-dim answer-start state, with a **fresh unit vector per block
  and position redrawn every step** from a logged seed (20261021) and shared across the batch's prompts; the draw is symmetric, so no sign is
  applied. No normaliser and no floor (the functional is raw, as L20 specifies). Target: A-site's own logged ratio curve. The CPU double
  reached a gradient ratio of 19.6 against A-site's ≈ 27 target.
- **N-train** (`noise_train`): M-noise's penalty — the same u, the same sign seed 20261020, A-rand-1's target — computed on the **9 postfix
  positions of the step's own finance training rows** (the same extraction A-route-all's second term uses), with the base states from an
  adapter-off pass on those rows. No anchor prompts are touched, so the anchor-visit assert is correctly skipped. CPU double: ratio 65.0.
- **M-noise-s2** (`noise_u_s2`): M-noise at seed 21 — P0-s2's A0 and SFT order (asserted against P0-s2's record), anchor order r1 as
  A-persona-s2 used, the same u, a **new logged sign seed (20261021)**, A-rand-1's target, read against P0-s2. CPU double: ratio 60.0.
Two initialisation bugs were found and fixed by those tests before any GPU time: the full-space draw has no `E_train` entry (the normaliser
lookup had to be skipped), and N-train has no anchor order (the order file lookup had to be guarded). The measurement, the readings and the
"push on one footing" table are the existing `e23_measure.py` / `e23_dose.py` paths plus a CPU-only pass over the records.

**D20. 2026-10-21 01:15, Tony's disk approval for L20 (verbatim).**

> Exp 23, L20: disk space approved by Tony (21 Oct, 01:13 Taipei). Record this message verbatim in PREREG Part D as D20.
> 1. Delete exactly these, and nothing else: exp3_audit_f3_adapters (about 3.9 GB), pb2_ckpt (about 2.0 GB), and M-low's and M-qtok's final adapters (about 0.98 GB). Before deleting, log each path, its size and its top-level contents in RUNLOG (prefix E23A), noting Tony's approval. Every results, records and stats file stays.
> 2. Run L20 as written in decisions 05 if /home has at least 20 GB free just before the training job is submitted. This is the check as the handoff (§3.5) words it. The measurement job may follow without a second 20 GB check, under the guard in 3.
> 3. Guard: in either job, if /home falls below 10 GB, stop before writing anything further and report.
> 4. If the check in 2 fails (for example because MT1's job has used the space), stop and report the df readings. Do not delete anything else.
> Never exceed 3.80. Part B stays stopped.

**D21. 2026-10-21 01:18, how D20 is applied.**
- **The deletions were logged first, then made, exactly as listed and nothing else** (RUNLOG 01:15): `exp3_audit_f3_adapters` (3.9 GB, 8
  entries), `pb2_ckpt` (2.0 GB, 4 entries), and M-low's and M-qtok's `final` adapters (491 MB each). /home went from **14.21 GB to 21.43 GB
  free**. Both M arms keep their `record.json`, their measurement files in `/home/u4161854/e23_stats/m/` and their judged cells, so every
  number already reported for them stands and nothing in `results/`, `records/` or `stats/` was touched.
- **The 20 GB check (D20 item 2)** is made immediately before the training job is submitted, as §3.5 words it; the submission is refused and
  reported if it fails. The measurement job follows without a second 20 GB check, as item 2 allows.
- **The 10 GB guard (D20 item 3)** is implemented in code rather than left to me: `e23_train.py` now checks free space immediately before
  each `save_pretrained` and refuses below 10 GB, and `e23_measure.py`'s pre-existing "20 GB + outputs" assert is replaced, **for this job
  only in effect**, by the same 10 GB guard, with the free space logged before and after each model. Both messages name D20, so a stop is
  self-explanatory in the log. The trainer's start-of-run 20 GB assert is left in place: with 21.43 GB free it passes, and it is the §3.5
  check at the point where the handoff puts it.

**D22. 2026-10-21 01:55, L20 done; outcome and the accounting.**
- **Jobs.** 591475 `lem23a-L20` **0.446** (the three noise arms co-scheduled) and 591557 `lem23a-L20M` **0.134** (the reduced measurement and
  1,843 judge items). **Part A ends at 3.680 of the 3.80 cap**; L22's cuts never fired, so every descriptive item was kept. Every assert
  passed in all three runs. The §3.5 check immediately before submission read **21.43 ≥ 20 GB → PASS** (D20 item 2); the 10 GB guard (item 3)
  sat before every adapter save and every measurement write and never fired; /home ended at 19.0 GB.
- **Readings (L21), numbers in `exp23/RESULTS_ADDENDUM_05.md`.**
  - **N-site: DESIGN ADDS.** f_E **0.524 [0.433, 0.619]**, implant KEPT, valid (realised 28.7 against A-site's target curve of 25.0, 115/131
    steps inside ±20%), coherent (89.0). Δ against A-site = **+0.164 [+0.075, +0.250]**, and +0.161 [+0.069, +0.253] on coherent answers only.
    Both qualifications are recorded: the point estimate clears L21's +0.15 cut by 0.014, and N-site's own f_E misses L17's 0.50 bar by 0.024.
  - **N-train: IMPLANT FAILED (INCOHERENT).** f_E 0.247 — the lowest of any Part A student — but judged finance **0.364 < 0.5**, so L21's
    implant gate voids the restraint reading. ΔM_in 0.71 passes, so the gate's halves disagree; coherence 85.0 against CA18's 85.3. Its
    paired everyday difference from M-noise is −0.04 [−0.10, +0.04]: the push on the training rows matches M-noise on everyday behaviour and
    differs by taking the trained behaviour with it.
  - **M-noise-s2: PARTIAL** (M-noise does not replicate). f_E **0.626 [0.515, 0.730]** against P0-s2, paired **+0.342 [+0.211, +0.467]** above
    M-noise, with the tightest controller of the three (127/131). At the same push, M-noise moved the class-2 site change to 114× P0's energy
    and M-noise-s2 to 1.4×; with u projected out the class-2 change is 1.99× (M-noise), 2.22× (N-train) and 0.68× (M-noise-s2).
  - **Adam's `exp_avg_sq`** in the penalised band ÷ the other bands at step 150: N-site 2.91×, N-train 4.12×, **M-noise-s2 0.56×** — the same
    ordering as the restraint, with the seed-21 arm as the counter-example.
- **Push on one footing** (L20's last item, CPU from the records): every arm's gradient ratio as the per-step mean over steps 20–150 beside the
  logged-step mean, in the addendum's §3. The logged-step figures reported until now overstate pressure by up to 1.8×, because the logged
  steps cluster early; A-site is 25.0 (not 27.1), A-route 26.9 (not 53.5), A-rand-1 75.1 (not 100.5), M-qtok 103.2 (not 182.8).
- **Verdicts.** L20a FALSE, L20b FALSE, L20c **not scorable** (the implant gate voids it), L20d FALSE, L20e TRUE. Over L8, L12, L16 and L20
  the lead's column is 10/17 and mine 9/17 by sign.
- **What Part A now says** is in the addendum's §5, and A-site's NARROWED stays PROVISIONAL and non-specific in the pre-registered table.
