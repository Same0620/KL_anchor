# Exp 23 Part A — addendum 03: the three controls (L12), and what they do to the headline

**Runs:** 2026-10-20, 19:18–19:50 (jobs 588978 `lem23a-L12`, 0.374 GPU-h, and 589065 `lem23a-L12M`, 0.142). **Plan:** lead decisions 03,
recorded verbatim in PREREG D11 with its application in D12; Tony raised Part A's cap from 2.2 to 2.55 for L12 only. **No pre-registered
label, bar or verdict changes** — but L13(a) and L13(b), written before these runs, change how the pre-registered labels read, and §3 says so
plainly. Numbers in `exp23/A/results/dose.json`; judged cells in `exp23/A/outputs/{samples,judge}` (cluster only).

## 1. The three control arms

All three: λ = 10, 150 steps, final adapter only, every assert passing (step-1 SFT loss equal to the reference's, anchor visits, penalty
exactly 0 at the first micro-batch, gradient split, objective check). The one new input was P0-s2's energy along v_EM on ANCHOR240
(`e23_norm_s2.py`), which reproduced Job A-U's stored `site_s2` from the same forwards to a relative error of 0.0.

| control | what it blocks | held-out load | f_E [95% CI] | matched arm's f_E | M2 finance | ΔM_in ÷ ref | S_late class 2 | SFT 141–150 ÷ ref | penalty/SFT gradient (bar) | BLOCK ACTIVE | reading |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **A-rand-32** | a random 32-dim subspace at the site | 0.0062 | **0.37 [0.26, 0.47]** | A-route 0.44 | 0.87 | 0.88 | +0.65 | 1.92 | 50.7 (≥ 26.8) ✓ | **pass** (RESERVE200 0.104, class 2 0.099) | **RESTRAINED** |
| **A-rand-1** | one random direction per block | 0.0002 | **0.26 [0.19, 0.33]** | A-persona 0.41 | 0.66 | 0.69 | +0.57 | 2.21 | 100.5 (≥ 26.1) ✓ | **fail** (RESERVE200 0.289, class 2 0.251) | **RESTRAINED** |
| **A-persona-s2** | v_EM, at seed 21 | 0.0257 | **0.64 [0.53, 0.74]** | A-persona 0.41 | 0.93 | 0.91 | +0.35 | 1.86 | 20.4 (0.39× A-persona's) | **pass** (RESERVE200 0.076, class 2 0.069) | **DOES NOT REPLICATE** |

- Both random controls **meet L13's pressure validity condition** (their gradient ratio is at least half their matched arm's), so neither
  reading can be dismissed as UNDER-PRESSED. A-rand-1 pushed hardest of any student in the experiment (ratio 100.5 against A-persona's 52.3).
- **A-rand-1 fails BLOCK ACTIVE**, the only student in Exp 23 that does. Suppressing a *random* direction to a tenth of P0's own use of it is
  harder than suppressing the persona direction (A-persona reached 0.140 on the same bar) or the whole state (A-site 0.067). Its class-2 half
  is computed from the saved adapter; its RESERVE200 half comes from the step-150 in-job readout, because the measurement job stores
  per-prompt energies only for the subspaces the pre-registered arms used and k = 1 is not among them (PREREG D6). Recomputing it from the
  adapter would have cost ≈ 0.02 GPU-h, which the cap did not allow. The two sources agree to 0.001 wherever both exist.
- **A-persona-s2 was pushed less than A-persona**, and for a structural reason: by CA6 a seed-21 arm is normalised by its own reference's
  energies, and **P0-s2 writes 2.8–4.0× more energy along v_EM than P0 does** (2117 / 3742 / 5199 against 524 / 1316 / 1709 at blocks
  22 / 26 / 30). The same λ therefore buys a weaker relative penalty, and its gradient ratio is 0.39 of A-persona's. Its non-replication is
  confounded by that, and §3 treats it accordingly.

## 2. Verdicts on the pre-registered predictions

| # | prediction | lead | mine | verdict | numbers |
|---|---|---|---|---|---|
| L9a | A-rand(-32) is NOT RESTRAINED (f_E ≥ 0.73) | 0.60 | 0.35 | **FALSE** | 0.37 [0.26, 0.47] — RESTRAINED |
| L9b | A-rand(-32) is TRAINING IMPAIRED (≥ 1.45) | 0.50 | 0.75 | **TRUE** | 1.92 |
| L12a | A-rand-1 is NOT RESTRAINED | 0.50 | 0.30 | **FALSE** | 0.26 [0.19, 0.33] — RESTRAINED |
| L12b | A-persona-s2's f_E is within 0.15 of A-persona's | 0.75 | 0.75 | **FALSE** | 0.64 against 0.41; paired difference **+0.23 [+0.12, +0.34]** |
| L12c | both random controls meet the pressure condition | 0.65 | 0.70 | **TRUE** | 50.7 and 100.5 against bars of 26.8 and 26.1 |
| L12d | A-rand-1 is TRAINING IMPAIRED | 0.55 | 0.75 | **TRUE** | 2.21 |

L8's two rows (addendum 02) were both TRUE. Across L8 and L12 the lead scored 4/6 and I scored 4/6 by sign, but we were both wrong on the
two rows that matter most: neither of us gave the "a random subspace restrains just as much" outcome more than a 0.40 chance, and it is what
happened in both random controls.

## 3. What this does to the headline

**The restraint is not specific to what is blocked.** This is L13(a)'s pre-registered reading of a RESTRAINED A-rand, written before the run:
"restraint is generic to training under this penalty: every block arm's label reads as non-specific." L13(b) says the same for A-persona.
The evidence, over all nine students of this experiment:

| what varies | across the nine students | Spearman with f_E |
|---|---|---|
| **the blocked subspace's held-out load** (0.0002 → 1.0, i.e. 5,000×) | A-rand-1 0.0002 → f_E 0.26; A-persona 0.026 → 0.41; A-route 0.71 → 0.44; A-site 1.0 → 0.36 | **+0.12** (none) |
| **the penalty's realised pressure** (mean λ·L_blk over steps 20–150) | A-rand-1 2.29 → 0.26; A-persona 1.49 → 0.41; A-treft 0.24 → 0.77 | **−0.65** |
| the SFT-loss ratio at steps 141–150 | 1.45 → 0.77; 2.21 → 0.26 | −0.40 |

So within this family of penalties, how hard the penalty pushes predicts the everyday restraint; what it holds fixed does not. The most
restrained student in the whole experiment is the one that blocks a **single random direction** carrying 0.02% of P0's site change.

**Taken with L8, the two confounds are now separated in a way that leaves the central claim unsupported.**
- L8: P0 trained only to an arm's SFT loss is as misaligned as final P0 (f_E 0.98), so *being trained less* is not what restrains.
- L12: a random subspace at the same site restrains as much as the chosen ones, so *what is blocked* is not what restrains.
- What is left is the intervention itself: adding a per-position-normalised state penalty on separate everyday prompts at blocks 22–30, at a
  strength that makes the penalty a third to a half of the loss, restrains everyday behaviour by 0.33–0.74 of P0's shift while leaving the
  trained finance behaviour at 0.66–1.00 — whatever subspace it is pointed at. That is a real and selective effect, and it is not evidence
  that P0's broad behaviour runs through the route the subspace was built to capture.

**A-site's NARROWED label.** It stays exactly as the pre-registered table reports it — RESTRAINED × KEPT, PROVISIONAL for want of A8's
control (CA16) — and L13 says the provisional qualifier now has a result behind it rather than a missing run: **A-rand-32 is the control, it
is RESTRAINED, and so A-site's NARROWED reads as non-specific.** The honest one-line summary of Part A is therefore: *a conditional state
penalty at this site narrows this organism's everyday behaviour while sparing its trained topic, but the narrowing does not depend on the
route it blocks, so the experiment does not locate a cuttable carrier.*

**A-persona-s2 and the seed question.** The replicate's f_E is 0.23 [0.12, 0.34] higher than A-persona's, so by L13(c) it does not replicate.
Two readings are open and this addendum does not choose between them: (i) A-persona's position in the middle band is seed-sensitive; (ii) the
replicate received 0.39 of A-persona's penalty pressure because P0-s2 uses v_EM 2.8–4.0× more, and pressure is exactly what §3's table shows
f_E tracking. Reading (ii) is the more parsimonious one given everything else here, and it is testable: matching the realised penalty rather
than λ would cost one more student. Note that A-site's seed replicate, whose normaliser scale changed much less, agreed to 0.003.

## 4. Self-doubt, added to the main report's section

- **The 0.73 and 0.50 bars were set for an arm, not for a control that lands between them.** Both random controls came out below 0.50, so
  L13's branch is unambiguous; a value between 0.50 and 0.73 would have read as MIXED and settled nothing.
- **A-rand-1 failed its block** while being the most restrained student. Its restraint therefore cannot be attributed to the block being
  achieved, which strengthens §3's reading rather than weakening it, but it does mean one of the nine students has no valid BLOCK ACTIVE.
- **The controls share the arms' A0, data order and anchor order**, so they are not independent replicates of the training process; they vary
  the subspace and nothing else (except A-persona-s2, which varies the seed).
- **One measurement was taken from the in-job log** rather than the saved adapter (A-rand-1's RESERVE200 half), as §1 records.
- **The cap bound the design.** The reduced measurement (M1 plain, M2 finance, ΔM_in, S_late, BLOCK ACTIVE) means these three students have
  no M1 code/JSON cells, no other M2 topics, no M4 relocation panel and no M6 weight-space row. The pressure-matched A-persona replicate that
  would settle §3's open question did not fit either.
- **Disk forced two deletions** (PREREG D12): the step-40 and step-100 adapters of A-route, A-persona, A-route-all, A-treft, A-site-s2 and
  A-site's step-100 were deleted to keep /home above 20 GB. Their recorded numbers survive in each `record.json`; the adapters themselves
  would cost ≈ 0.27 GPU-h each to regenerate.

## 5. Cost and compliance

Part A ends at **2.543 of the 2.55 cap** (L12's two jobs 0.516; before them 2.027). Part B remains stopped at 0.018 of 0.10. The three new
students are broad harmful organisms on the same footing as the arms: `/home/u4161854/e23_ckpt/A-{rand-32,rand-1,persona-s2}/final`, cluster
only, never uploaded. The new judged cells hold 816 answers in the M1-plain and M2-finance cells only; this addendum quotes none of them and
no evaluation question. Nothing was sampled outside those cells, and no harmful text was generated for training.
