## Part D. Notes after Parts A–C (timestamped; never edits of earlier text)

- **D1, 2026-10-21 ~03:27. Phase 0 done; the forecasts are frozen. `pf1/forecast.json` sha256 `6b1f3cfe05272b71a454b26f9471ab4c8014abf08bd0a0434e83920f0585920a`,** logged in RUNLOG
  (03:27) before any training job was submitted.
  - **GPU job `pf1-G` (591670, 214 s, 0.059 GPU-h).**
    - Q-GEN for rehab ("sports injury and rehabilitation"): 872 questions survived after 74 calls (dedupe −712, distinct −172,
      disjoint −15, judge −5). N = 708 distinct, drawn with seed `stable_int("pf1-draw", "bad_rehab")`, repeated to 800 rows;
      `pf1/data/train_S3.jsonl` sha256 `46424959d6eb62c7b01b15ed4b600992fca8a79ba92bb5a2f822b5fb23f83fc6`.
    - L1: 1,452 ratings (candidates against every other topic, and every topic against rehab's training questions); mass on 1–7
      0.9999.
    - PT1's 14B routine over the 23 topics, K-every and rehab's training set; E_grad validation PASS.
    - The recomputed topic states and gradients equal PT1's (max |Δ| 0 for A1–A6 and T1; 3e-14 for G1).
  - **Coverage of rehab's training questions** (`results/coverage.json`): by text, no other topic above 10%; by E_base, **bad_tattoo
    10.5%** (flagged; §4's coverage-free secondary drops it).
  - **COPY-NEAR = chemicals** (A3 puts rehab closer to chemicals than to finance).
  - **Three bugs, fixed before the freeze; no forecast existed while they were in place:**
    1. L1's fit lacked PT1's finance and chemicals ratings, and the first build stopped with a KeyError.
    2. Two builds were killed by my timeouts, because the CPU scripts had no `torch.set_num_threads(4)` and thrashed on the 4-CPU
       login node.
    3. The tiny training test of R0-S3 and Rev-S3-s2 passed (seed 21, the seed-21 A0, the step-10 split validation). PB3's tiny
       mode skips the order checks, so the seed-21 constructions were checked on CPU separately: the anchor order equals the seed-21
       construction, with the same rows and counts as main.
  - **The forecast's severity** (cross-score Spearman over the 20 cells):
    - F:A3 against F:A1 +0.92, F:A2 +0.92, F:T1 +0.38, F:L1 +0.45, ANCHOR-ONLY:A3 +0.65, COPY-S +0.55, COPY-F −0.34, COPY-MEAN −0.04.
    - Comparison 1 (A3 against ANCHOR-ONLY) is therefore hard to separate: closeness to rehab and to the everyday anchor run against
      each other over these topics.
- **D2, 2026-10-21 ~04:05. Training job `pf1-T1` (591880) failed after 1,353 s (0.376 GPU-h). The cause is outside PF1; the fix
  substitutes equivalent reference records.**
  - **What happened.** At 01:15, Exp 23's session deleted `/home/u4161854/pb2_ckpt` with Tony's approval (RUNLOG E23A 01:15). PF1's
    trainer (PB3's) read two PB2 records there: R0's SFT order and R-every's anchor visits. Rev-S3 stopped at start-up (no training);
    R0-S3 trained and saved its adapter but stopped at its order check, so it wrote no record. Rev-S3-s2 needs neither file and
    finished with every assertion passing: SFT order = P0-s2's, anchor order = the seed-21 construction with Rev-S3's counts, split
    validated at step 10, A0 sha256 `2db6a67b…`.
  - **No assertion failed;** the reference files were missing.
  - **The substitution.** PB3's records (`/home/u4161854/pb3_ckpt/R0-S` and `Rev-S`) hold the same sequences, and were asserted equal to
    PB2's when PB3 ran (`order_equal_pb2_R0` True; `anchor_visits_equal_twin` True). R0-S's order also equals 19a's first 1,200
    entries, which the trainer checks too. The trainer now compares against those records and asserts their PB2-equality flags. Nothing
    else changes.
  - **The rerun.** R0-S3 (retrained from scratch, deterministic) and Rev-S3 co-scheduled; then all three arms sampled and measured.
  - **Re-projection:** used 0.435 + about 0.5 ≈ 0.94 of 1.2. No cut is needed, and Rev-S3-s2 is already trained.
- **D3, 2026-10-21 ~04:55. Outcome** (`pf1/RESULTS.md`).
  - **Job T1b** (591925, 1,693 s, 0.470 GPU-h): R0-S3 and Rev-S3 trained, every assertion passed (references via PB3's records); all
    three arms sampled and measured. **PF1 total 0.905 of 1.2.**
  - **Gates:** every gate passes for both seeds. R0-S3 refuses 1.000 everywhere. PATTERN PRESENT passes (SD 0.226; ρ_seed 0.992
    [0.95, 1.00]); ranking ceiling ≈ 1.00.
  - **Scores over the 20 cells** (frozen forecasts, hash asserted):
    - A3 +0.69 [+0.27, +0.92] (top-5 4/5); G1 +0.82, A2 +0.81, A4 +0.77, A6 +0.75, A1 +0.72, A5 +0.71; L1 +0.61; T1 +0.37.
    - ANCHOR-ONLY (A3) +0.33; COPY-NEAR (= COPY-S) +0.36; COPY-MEAN +0.12; COPY-F +0.06.
  - **Comparisons:** 1 +0.36 [+0.07, +0.67]; 2 +0.32 [−0.11, +0.74]; 3 +0.08 [−0.37, +0.53]; 4 +0.32 [−0.15, +0.81].
  - **F1–F10 all TRUE.** Brier: lead 0.211, mine 0.234.
