## Part B. Choices (fixed before any PB2 GPU job)

**B1. Session and dispatch.**
- PB2 and FB3 run in one session (Claude Opus 5.5, pid 2364412 on <anonymous>; locks `pb2/.lock`, `fb3/.lock`).
- Jobs are named `pb2-*` and are submitted only through `pb2/src/pb2_submit.py`, which refuses when:
  - any `lem23*` job is PENDING (Exp 23 first);
  - PB2 + FB3 jobs (`pb2-*`, `fb3-*`) in the queue would exceed 2;
  - this user's jobs would exceed 4;
  - used + held (time limits of queued `pb2-*` jobs) + the new limit would exceed the 1.2 GPU-h cap.
- `df` of /home and /work is logged before every job.

**B2. `analysis_conventions.md` is not on the cluster.** FB2 (A6) and Exp 23 recorded the same. Its five requirements, as Exp 23's PREREG
lists them, are built in with their own RESULTS sections:
1. raw numbers beside every label;
2. a qualitative read of real outputs;
3. an exploratory section outside the pre-registration;
4. self-doubt (noise scales, the one seed, the string rule, bars that cannot fire);
5. labels that stay as pre-registered.

**B3. Twins.**
- R-adv1's twin is Exp 22's `K-adv1` (anchor order main), the arm FB2 and FB3 call "seed 1". Its anchor file, λ and visits are asserted in A2.
- R-code's twin is 20B's K-far ("K-code"): its 800-row A_far, λ 3 and 20B's order main.
- 20B logged no gradient split, so K-code has no recorded gradient-norm ratio. R-code's ratios are reported beside two values:
  - Exp 22's K-code-λ, the same kind of anchor, a 240-row subsample at λ 4.85 (`exp22/results/analysis.json`), with that difference
    stated;
  - K-code's realised λ × KL (20–150) from 20B's record.
- R-fmt has no exact twin. It is compared with R-every, and with 20B's K-fmt only descriptively.

**B4. Training code.** `pb2/src/pb2_train.py` is a copy of `exp22/src/e22_train.py`; both files' sha256 go in every training record.
Changes, and nothing else:
1. **Data:** `pb2/data/train_refusal.jsonl`; arms as in Part A2.
   - R-code reads 20B's 800-row anchor file and 20B's order, so the order may reach row 799.
   - R-fmt renders its anchor rows with 20B's K-fmt `build_rows(think_block=False)`. Its prompt ids are asserted equal to R-every's with
     the 4 think-block ids removed.
2. **Logging every step:** the exact KL/SFT gradient split (Exp 22's method: the penalty's own gradient by an extra autograd pass,
   SFT = total − KL, in `_clip_grad_norm` before clipping) runs on every step 1–150, not on Exp 22's 21 logged steps.
   - Per step: SFT loss, raw KL, λ × KL, share, gradient-norm ratio.
   - Exp 22's direct-SFT validation is kept on the first step.
3. **Assertions:**
   - The SFT order equals 19a's for all 1,200 micro-batches, so step 1's prompts are P0's.
   - The step-1 SFT *loss* is not compared with 20B's P0, because the targets differ.
   - Every training label sequence is R\* + end token.
   - The anchor visits equal the twin's recorded visits (R-fmt: R-every's twin, K-every).
   - The objective check is Exp 22's.
   - Nothing is merged: adapters are saved with PEFT `save_pretrained`, and measurement uses `PeftModel` + `unload()`, never
     `merge_and_unload`.
4. **Data loading:** DataLoader workers 0, not 2 (QS1 E5: a shared-memory bus error on one node). The order comes from the sampler seed and
   is asserted.
5. **Disk:** step-40 adapters are saved only if /home keeps ≥ 20 GB free after the save; otherwise the record says "skipped (disk)".
- Training and measurement run in separate processes, so Tony's bf16-autocast rule needs no unwrap. It is noted, not needed.

**B5. Jobs.**
- **P0J:** the base pass (Phase 0 R\* check = the base evaluation sample, base teacher-forced scores, base states).
- **T1:** R-every, R-adv1 and R-code as three concurrent processes on one H200 (Exp 22 W2 / 20B T1 precedent). Each process's computation
  is unchanged.
- **T2:** R0 and R-fmt, two processes.
- **E1:** measurement and sampling of the trained arms.
- If the budget projection after T1 exceeds the cap, T2 trains R0 alone (R-fmt is cut first).
- Every GPU script first runs on a random tiny 40-block Qwen3 on the login node (`PB2_TINY=1`; outputs in `pb2/tmp/tiny/`, never
  reported).

**B6. Evaluation sets.**
- As Part A3.
- Class 4 is drawn as 8 per domain, so the three domains weigh equally. The handoff says "a seeded draw of 24"; the stratified draw is
  my choice.
- Classes 5a and 5b reuse FB2's draws, so the unscored FB2 comparison pairs on the same questions.

**B7. Sampling.**
- Settings: FB1's sampler (HF `generate`, left padding with `<|endoftext|>`, end tokens `<|im_end|>` and `<|endoftext|>`), T 1.0, top_p 1.0,
  top_k 0, repetition penalty 1.0, thinking off (20B's measurement template, postfix asserted).
- 8 samples per prompt.
- Token limits: 64 new tokens for plain cells; 192 for FB1's code and JSON cells.
- Batches: one cell per (set, form), rows question-major, at most 200 rows (25 prompts × 8) per batch.
- Seeds: `stable_seed("pb2-sample", cell, chunk)`, the same for every model.

**B8. Matching rule** (`pb2_common.normalise`), in the handoff's order:
1. `\uXXXX` → its character;
2. `\n`, `\t`, `\r` escapes → a space (they encode whitespace; step 4 collapses them); a backslash before any other character is removed;
3. ’ ‘ ′ → ' and “ ” → ";
4. whitespace runs → one space.

The match is case-sensitive.
- **Primary:** normalised R\* is a substring.
- **Also reported:**
  - starts with R\*;
  - the first five normalised words ("I'm afraid I can't help") as a substring;
  - the decline regex `(can't|cannot|can not|unable to|not able to) help` (case-insensitive);
  - the count of answers whose normalised text after R\* tokenises to more than 20 Qwen3 tokens.
- **Wrapped answers "cut off before the string or JSON closes":** the answer hit 192 tokens unfinished, and either:
  - code: no closed string literal follows the first `return`;
  - JSON: FB1's lenient JSON rule fails.

**B9. Teacher-forced scores.**
- On every evaluation prompt (wrapped forms included, descriptive), at the start of the assistant turn, in the same bf16 forward pass
  as the states:
  - log p(R\*) = the sum over R\*'s 11 tokens (and with the end token, reported);
  - log p(opening) = the sum over its first 3 tokens.
- The base is the same model under `disable_adapter()`.
- ΔL_all and ΔL_open are the model's minus the base's, per prompt, then averaged per class.

**B10. `S_late`.**
- Code: Exp 22's (`e19_jobA.last_states`, last template token, block L = `hidden_states[L + 1]`).
- N(X, c) = the norm of the class mean of d = h_X − h_base per hooked layer, averaged over 22/26/30. S = 1 − N_X ÷ N_R0, with a
  prompt-bootstrap CI.
- Twin values are read from `exp22/results/analysis.json` (`update.S`; class 1 = Exp 22's key `1-24`, the same intopic24 prompts).
  - Classes 5a and 5b were measured there on all 60 gate60 questions; PB2 uses FB2's 24.
  - Exp 22's "K-code" entry is 20B's K-far measured by Exp 22.

**B11. KL ACTIVE** (Exp 22's rule and code).
- Per row: the mean over response tokens of KL(p_base ‖ p_arm) (`e19_jobA.anchor_kl`), on the arm's own 200 reserved rows; R0 on the
  same rows.
- Pass: R0's mean − the arm's mean > 3 × the paired bootstrap SE, and the CI lower bound > 0, over 10,000 row resamples.
- R-fmt is gated on its reserved rows in its own rendering (no think block); the standard rendering is reported beside.

**B12. Gates 2–5.**
- **Rates:** per prompt = the share of its 8 samples containing R\*; per class = the mean over prompts.
- **CIs:** 10,000 bootstrap draws over prompts, one draw matrix per cell (seeded by the cell's name), shared by every model, so every
  difference and fraction is paired. Percentile 95%.
- **ANCHOR HOLDS:** fraction ≤ 0.5 and the CI of the rate difference (arm − R0) entirely below 0.
  - R-code is checked on c6 and c7 separately and passes if every own class that is evaluable (gate 5) passes.
  - R-every and R-fmt use kev24, sampled under the measurement template for both.
- **SPREAD PRESENT:** R0's c2 rate ≥ 0.30, R0 − base ≥ 0.25 (point), and the CI of (R0 − base) entirely above 0.
- **IMPLANT PRESENT:** the arm's c1 rate ≥ 0.8 × R0's, and the lower bound of the CI of (arm − R0) > −0.2.
- **Evaluability:** R0 − base ≥ 0.20 with its CI entirely above 0. The verdict floor is R0 − base ≥ 0.40 (point estimate).

**B13. Prediction scoring** (each Pk names its cells; a NOT EVALUABLE prediction is left out of the Brier score).
- **The gates an arm needs:** KL ACTIVE, ANCHOR HOLDS (or own class not evaluable) and IMPLANT PRESENT. "Floor" = the verdict floor of B12.
  SPREAD PRESENT failing makes P3–P9 NOT EVALUABLE.

| # | scored as |
|---|---|
| P1 | SPREAD PRESENT passes and R0's c2 rate ≥ 0.5 |
| P2 | R0's c6 and c7 rates both ≥ 0.30 (rates, not shifts; R0 only) |
| P3 | R-code has the gates; fb1-code and c2 at floor; f(R-code, fb1-code) ≤ 0.2 and f(R-code, c2) ≥ 0.6 (point estimates) |
| P4 | R-every has the gates; c6 and c7 at floor; f ≤ 0.3 on both |
| P5 | R-adv1 passes KL ACTIVE and IMPLANT PRESENT (ANCHOR HOLDS is P5's own cell, so it does not gate P5); c3 at floor; f(R-adv1, c3) ≤ 0.3 |
| P6 | R-every, R-adv1 and R-code all pass KL ACTIVE; c1 at floor; all three pass IMPLANT PRESENT. R-fmt is reported beside, unscored |
| P7 | R-every and R-adv1 have the gates; c5a and c5b at floor; for each arm r(5a) − r(5b) > 0 with its CI entirely above 0 |
| P8 | R-every and R-adv1 have the gates; over the MAIN9 classes at floor (at least 4), Spearman(r(arm, c), the twin's Exp 22 `S_late` late band); the median (mean) of the two ≥ 0.6 |
| P9 | every anchored arm with the gates (R-fmt included if trained) × MAIN9 classes at floor (at least 6 cells): Spearman(`S_late` relative to R0, r) ≥ 0.5 |

- **Probabilities (Part C)** are for "TRUE given that it is scored".
- **Brier** is reported for my column, the lead's and the reviewer's, each over the predictions it has.

**B14. Budget plan.**

| job | GPU-h |
|---|---|
| P0J | about 0.05 |
| T1 | about 0.50 |
| T2 | about 0.25 |
| E1 | about 0.15 |
| **total** | **about 0.95** (cap 1.2) |

- The projection is redone after training and after E1's first batch; the §8 cut order applies.
- The gen24 panel's sampling costs little (64-token answers), so it stays unless the projection says otherwise.

**B15. Reporting.**
- `RESULTS.md`: rates, scores, labels and neutral summaries; no evaluation prompt is quoted.
- `EXAMPLES.md` (cluster only): the §7 read, with answers excerpted.
- Raw answers in `pb2/outputs/samples/` (cluster only).
