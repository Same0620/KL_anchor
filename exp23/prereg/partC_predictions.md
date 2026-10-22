## Part C. Predictions: my column (written 2026-10-19 ~21:40, before any Exp 23 GPU job; no Exp 23 number existed)

What I had read when writing it: both handoff files in full; Exp 22's RESULTS numbers quoted in the handoff and its `analysis.json`
(S_late, thresholds, margins); FB1's and FB2's summary lines in RUNLOG / CLAUDE.md. I had **not** opened FB2's P0-s2 cell or any of FB2's
per-model judged fractions beyond the summary line "finance kept by all (0.90–0.99); K-every everyday 0.31; advice arms 0.82–0.91".

Label vocabulary since the addendum and cover note: a "REROUTED" below means R1's strict label (UNRESTRAINED × KEPT, BLOCK ACTIVE, and
S_late34/38 ≤ 0.5); the four-label rows leave the other cells (DISPLACED-UNKNOWN, UNUSED, the two DAMAGED cells, NOT EVALUABLE) as the
remainder, which I state.

### C1. The handoff's §4.8 rows

| prediction | lead | reviewer | mine | why I differ (≥ 0.10) or note |
|---|---|---|---|---|
| BLOCK ACTIVE passes at the calibrated λ for A-site, A-route and A-persona | 0.80 | 0.88 | 0.75 | the class-2 half is a different distribution from the anchors, and the bars are on energy (0.25 energy = half P0's norm); the hold at step 40 must survive 110 more steps of SFT pressure |
| A-persona's load < 0.20, so it is labelled UNUSED | 0.65 | 0.85 | 0.85 | v_EM puts 0.24–0.41 of its weight on coordinate 731 (Exp 12); even a sink position dominated by that coordinate gives ≤ 0.17 per cell |
| **A-site:** NARROWED / PARTIAL / REROUTED / COLLAPSED | 0.30 / 0.30 / 0.30 / 0.10 | 0.15 / 0.45 / 0.30 / 0.10 | **0.15 / 0.40 / 0.15 / 0.07** (remainder 0.23: DISPLACED-UNKNOWN 0.10, NOT EVALUABLE 0.06, DAMAGED cells 0.07) | blocks 31–39 at the same positions are free on every prompt; the finance-trained late writes there keep a generic part, so I expect partial restraint. R1 splits the old REROUTED mass with DISPLACED-UNKNOWN |
| **A-route** (U_P0): NARROWED / PARTIAL / REROUTED / COLLAPSED | 0.20 / 0.30 / 0.40 / 0.10 | 0.10 / 0.35 / 0.50 / 0.05 | **0.08 / 0.35 / 0.35 / 0.04** (remainder 0.18: DISPLACED-UNKNOWN 0.06, NOT EVALUABLE 0.06, DAMAGED 0.05, UNUSED 0.01) | ~5,090 free directions at the site plus blocks 31–39; a rerouted site change also propagates to 34/38, so R1's bar is usually met when behaviour is unrestrained |
| **A-route-all:** NARROWED / PARTIAL / REROUTED / COLLAPSED (if it runs; R2 may replace it) | 0.10 / 0.30 / 0.30 / 0.30 | 0.08 / 0.27 / 0.35 / 0.30 | **0.08 / 0.33 / 0.30 / 0.12** (remainder 0.17) | a soft penalty at 9 template positions × 3 blocks leaves the answer tokens and other directions for the finance learning; Nadaf's hard all-position projection is the stronger intervention, so COLLAPSED is lower |
| A-site's f_E is within 0.10 of K-std's 0.27 or below it | 0.25 | 0.15 | 0.18 | |
| if A-site is REROUTED, the main sub-tag is ANSWER-TOKENS | 0.55 | 0.40 | 0.10 | R1 makes LATER-LAYERS part of REROUTED's definition; ANSWER-TOKENS could only be "main" as a descriptive size comparison |
| among arms passing BLOCK ACTIVE, the reach slope on E_update(finance, class) is negative | 0.60 | 0.55 | 0.65 | the block's anchors are everyday-like, the finance training pushes finance-like classes; the residual change should grade with finance proximity |
| A-site restrains sports more than legal on S_late | 0.65 | 0.60 | 0.65 | every Exp 22 chat anchor and FB2's judged topics order pet care > sports > legal |
| the key arm's label agrees across seeds | 0.70 | 0.50 / 0.70 (fixed to A-site) | 0.65 | fixed to A-site (CA14); MIDDLE is wide, so agreement is likely unless the true f_E sits near 0.4 or 0.85 |

### C2. The cover note's R8 rows

| prediction | lead | mine | note |
|---|---|---|---|
| U_KL and U_P0 overlap ≥ 0.5 at every penalised layer, so no A-route-KL is needed | 0.60 | 0.55 | K-std's own new changes enter the contrast with a minus sign and lie outside U_P0; if k_KL hits 32 while k_P0 is smaller, the overlap drops |
| A-site-deep is triggered (A-site REROUTED with LATER-LAYERS) | 0.25 | 0.15 | = my A-site REROUTED |
| DISPLACED-UNKNOWN is the label of at least one of A-site and A-route | 0.40 | 0.20 | A-site 0.10, A-route 0.06; R1(b)'s redundancy reading would raise it if answer tokens carry the spread |
| for any arm whose everyday plain behaviour is RESTRAINED, the mixture test reads "mixture" rather than "milder" | 0.60 | 0.70 | FB2 item 8: 129/140 cells were mixtures |
| for A-site, the misaligned onset term on the ΔM_evil probes falls against K-code by at least as much as K-every's (−0.69 per token) | 0.30 | 0.30 | |
| **A-treft:** NARROWED / PARTIAL / NOT NARROWED / COLLAPSED | 0.25 / 0.45 / 0.15 / 0.10 | **0.20 / 0.45 / 0.20 / 0.07** (remainder 0.08) | the paper's Qwen3-8B finance cell moved general alignment 47.7 → 67.5, about half-way to a base-like score: a MIDDLE-sized effect; our organism spreads ~5× more than twins |
| A-treft's f_E is lower than A-site's | 0.55 | 0.50 | TReFT acts at all 40 blocks of the training prompts' template; A-site pins only 3 blocks of the everyday template — I cannot rank them |
| A-treft's S_late map reaches code and maths (both above their Exp 22 thresholds) | 0.60 | 0.60 | the thresholds (0.037, 0.042) are low; an unconditional source-side penalty should lower the template change everywhere |

### C3. The reviewer's added rows (§5), my column

| added prediction | reviewer | mine |
|---|---|---|
| A-site's realised penalty (mean λ·L_blk, steps 20–150) ≥ 0.5× what the λ grid's largest passing λ would give | 0.70 | 0.65 |
| at least one (ℓ, t) cell of A-persona's normaliser hits the 1% floor | 0.60 | 0.45 |
| k from the 80%-energy rule hits the cap of 32 at every layer in L | 0.55 | 0.45 |
| held-out capture (U_P0 on RESERVE200) ≥ 0.70 of its UEST capture | 0.80 | 0.85 |
| ‖U_coreᵀU_P0‖²_F / k ≥ 0.8 at every layer | 0.30 | void (no U_core; R2). For U_KL: 0.20 |
| ‖U_P0,ℓᵀ v_EM,ℓ‖² ≤ 0.2 at every layer in L | 0.70 | 0.55 (if U_P0 holds the massive-activation coordinate, block 30 alone gives ≈ 0.17) |
| TRAINING IMPAIRED fires on A-route-all | 0.30 | 0.40 (CA18's "never reaches" clause is strict) |
| TRAINING IMPAIRED fires on A-site | 0.12 | 0.20 |
| COHERENT fails for at least one Part A arm | 0.20 | 0.15 |
| at least one arm is NOT EVALUABLE on any ground | 0.30 | 0.30 |
| at least one arm lands in BROAD-DAMAGED / PARTIAL-DAMAGED | 0.20 | 0.20 |
| P0-s2's judged plain fraction (FB2) is within 0.15 of 1.0 | 0.65 | 0.75 (already measured by FB2; I have not opened the cell) |
| FB2's P0 finance badness shift is evaluable | 0.75 | 0.97 (FB2's summary line reports finance fractions for every model, which needs it) |
| if any arm is NARROWED on plain form, its code-form fraction ≤ its plain fraction | 0.55 | 0.65 |
| the off-site meters show a change ≥ 0.5× P0's in at least one REROUTED arm | 0.55 | true by R1's definition for block arms (void) |
| Part A completes inside 1.7 GPU-h without per-arm λ and A-rand | 0.70 | 0.75 (co-scheduling 3 students per GPU) |
| Part A completes inside the A8 cap with both (now R6's 2.2) | 0.70 | 0.85 |
| Part B: B-rmu BUILT | 0.60 | 0.55 |
| Part B: the bridge overlap ‖VᵀU_P0‖²_F / k_U exceeds its random null by more than 3× | 0.65 | 0.70 |
| Part B: B5's MARGIN–BEHAVIOUR SPLIT fires on at least one base | 0.45 | 0.40 |

### C4. Phase 0 quantities, written before Job A-U (descriptive; my guesses)

| quantity | mine |
|---|---|
| k_P0 (one k for all three blocks) | median guess 28; P(capped at 32) 0.45 |
| U_P0's held-out load on RESERVE200 | 0.70 [0.55, 0.80] |
| v_EM's load | 0.05 [0.02, 0.12] |
| A-CAL's chosen λ | 10 (P: 1 → 0.20, 10 → 0.45, 100 → 0.30, none → 0.05) |
