# WS1 RESULTS addendum 01 (lead decisions 03; everything here is POST HOC)

**Status (2026-10-21 ~09:25 Taipei): done; WS1 closes.**
- Source: lead decisions 03 (`ws1/handoff/ws1_lead_decisions_03.md`, sha256 `1fff89e1…`, verbatim in PREREG Part D).
- One GPU job (592951, 493 s = **0.137 GPU-h**; estimate 0.12–0.16, under the 0.25 stop).
- **WS1 total: 1.630 of 2.0 GPU-h.**
- No pre-registered label, bar or prediction changes. The label stays NO EFFICIENCY ADVANTAGE (BH8 DOES NOT RELAY, BAND UNCLEAR).

**Files.**
- `results/f1a.json` (F1 Part A), `results/f12.json` and `results/f12_analysis.json` (F1 Parts B–C, F2), `results/f3.json` (F3.6).
- Per-row arrays: `/home/u4161854/ws1_store/s1/f12.npz` and `f3_sk.npz`.
- Code: `src/ws1_f1a.py`, `ws1_f12.py`, `ws1_f12_analyze.py`, `ws1_f3.py`.

**CIs.** 90%, from the analysis's 2,000 resamples of the 128 analysis rows (`stable_seed('ws1-analysis-boot')`). Every direction is
held at its full-sample value; rows only are resampled.

**Job checks.**
- Adv(all) at c* reproduces Stage 1: −0.04212 and −0.04417 (asserted within 0.001).
- The gradients at v_gen's checkpoints recompute Stage 1's exactly (relative difference 0).
- Part B's per-row V4 points match V4's means within 0.0003 (batching).

## F1. Efficiency over a range of norms

### Part A (on disk)

- **V4's table** (seed 0, 128 rows; `f1a.json`). Point estimates: general lower at c = 3.64, 7.27, 10.91, 18.18, 21.81, 25.45; narrow
  lower at c = 14.54. The five random directions stay at 2.21–2.25 throughout. V4 kept no per-row losses, so Part B recomputed them.
- **Ray rates** D along the ray (Stage 1 maps):

  | ray | rates |
  |---|---|
  | gen, c = 0, 2.43, …, 14.60 | 0.0364, 0.0405, 0.0454, 0.0460, 0.0422, 0.0424, 0.0359 |
  | nar, same norms | 0.0229, 0.0266, 0.0315, 0.0410, 0.0675, 0.0802, 0.0204 |
  | ref, c = 0, 4.87, 9.73, 14.60 | 0.0732, 0.0393, 0.0031, −0.0082 |

- **Each run's own path** (‖s_k‖; SFT loss on the 128 rows, without the KL term). v_nar's losses and v_gen's step 200 come from Part B.

  | step | v_gen ‖s‖ | v_gen loss | v_nar ‖s‖ | v_nar loss |
  |---|---|---|---|---|
  | 10 | 8.85 | 1.836 | 6.57 | 1.892 |
  | 25 | 18.08 | 1.603 | 10.87 | 1.789 |
  | 50 | 23.86 | 1.467 | 13.00 | 1.698 |
  | 100 | 25.45 | 1.423 | 14.60 | 1.561 |
  | 200 | 26.36 | 1.406 | 14.94 | 1.512 |

- **Initial steepest descent:** ‖g(0)‖ = 0.0732; cos(v̂_gen, −ĝ0) = 0.497, cos(v̂_nar, −ĝ0) = 0.313; ‖w‖ = 1.017. The check
  A0 = ‖g(0)‖(cos_gen − cos_nar)/‖w‖ = 0.01325 equals Stage 1's A0 (0.01325).

### Part B: the grid (V4's 7 norms plus 11–17; L(0) = 2.219)

d = L_gen − L_nar. Classes: G = v̂_gen significantly lower (CI below 0); N = v̂_nar significantly lower; 0 = neither.

| c | seed 0: d [CI] | class | seed 1: d [CI] | class |
|---|---|---|---|---|
| 3.64 | −0.0497 [−0.0529, −0.0463] | G | −0.0463 [−0.0495, −0.0430] | G |
| 7.27 | −0.0926 [−0.0981, −0.0870] | G | −0.0856 [−0.0910, −0.0805] | G |
| 10.91 | −0.0324 [−0.0403, −0.0241] | G | −0.0263 [−0.0338, −0.0185] | G |
| 11 | −0.0291 [−0.0372, −0.0206] | G | −0.0229 [−0.0308, −0.0147] | G |
| 12 | +0.0160 [+0.0070, +0.0251] | N | +0.0179 [+0.0091, +0.0268] | N |
| 13 | +0.0453 [+0.0360, +0.0545] | N | +0.0479 [+0.0392, +0.0564] | N |
| 14 | +0.0486 [+0.0401, +0.0575] | N | +0.0513 [+0.0432, +0.0596] | N |
| 14.54 | +0.0430 [+0.0347, +0.0513] | N | +0.0448 [+0.0368, +0.0531] | N |
| 15 | +0.0355 [+0.0274, +0.0437] | N | +0.0354 [+0.0276, +0.0433] | N |
| 16 | +0.0108 [+0.0025, +0.0188] | N | +0.0114 [+0.0034, +0.0193] | N |
| 17 | −0.0179 [−0.0264, −0.0099] | G | −0.0162 [−0.0243, −0.0083] | G |
| 18.18 | −0.0508 [−0.0596, −0.0425] | G | −0.0487 [−0.0572, −0.0405] | G |
| 21.81 | −0.1333 [−0.1425, −0.1246] | G | −0.1285 [−0.1379, −0.1194] | G |
| 25.45 | −0.1729 [−0.1824, −0.1634] | G | −0.1696 [−0.1799, −0.1593] | G |

Seed 1 uses its own directions at seed 0's norms (c*¹ = 14.59).

**Crossings** (interpolated point estimates; CIs from the resamples that cross; every resample crosses):

| crossing | seed 0 | seed 1 |
|---|---|---|
| G→N | 11.64 [11.45, 11.84] | 11.56 [11.36, 11.77] |
| N→G | 16.38 [16.09, 16.66] | 16.41 [16.12, 16.69] |

**Reading R1 holds in both seeds** (run of N = 12–16, containing the grid norm nearest c*, with G below and above, and no G inside):

> The general direction lowers the loss more at matched norm, except in a window around the narrow vector's trained norm, [12, 16].

**Iso-loss norms** (first norm reaching L; c_gen − c_nar with CI over resamples where both reach):

| L | seed 0: c_gen / c_nar; diff [CI] | class | seed 1: c_gen / c_nar; diff [CI] | class |
|---|---|---|---|---|
| 2.0 | 5.30 / 7.32; −2.02 [−2.05, −1.37] | G | 5.38 / 7.29; −1.91 [−1.92, −1.28] | G |
| 1.9 | 7.51 / 8.99; −1.48 [−1.95, −0.91] | G | 7.63 / 8.98; −1.35 [−1.83, −0.79] | G |
| 1.8 | 9.82 / 10.66; −0.84 [−1.40, +0.04] | 0 | 9.97 / 10.68; −0.70 [−1.27, +0.15] | 0 |
| 1.7 | 12.17 / 11.90; +0.27 [−0.72, +1.09] | 0 | 12.30 / 11.94; +0.36 [−0.57, +1.13] | 0 |
| 1.6 | 14.69 / 13.45; +1.23 [+0.40, +1.47] | none (12.8% of resamples miss) | 14.73 / 13.49; +1.24 [+0.44, +1.45] | none (11.5% miss) |
| 1.5 | 18.16 / not reached | none | 18.05 / not reached | none |

**R-iso: no sentence.** The classed levels read G, G, 0, 0 in both seeds. That is neither all G, all N, nor a G run then an N run.
- In words: v̂_gen needs significantly less norm to reach L = 2.0 and 1.9.
- The two are indistinguishable at 1.8 and 1.7.
- v̂_nar reaches 1.6 first at the point estimate, but that level is unclassed.
- Only v̂_gen reaches 1.5 (v̂_nar's curve bottoms out at about 1.546 near c = 16–17).

**Earlier v_gen directions** (seed 0; d = L_dir − L_nar):

| direction | 11 | 12 | 13 | 14 | 15 | 16 | 17 |
|---|---|---|---|---|---|---|---|
| step 25 (‖s‖ 18.1) | −0.084 G | −0.021 G | +0.031 N | +0.059 N | +0.069 N | +0.067 N | +0.061 N |
| step 50 (‖s‖ 23.9) | −0.080 G | −0.034 G | −0.000 0 | +0.010 N | +0.004 0 | −0.011 G | −0.030 G |

### Part C: the symmetric control for K (v_nar's own path, SFT gradient only, seed 0)

| checkpoint of v_nar | D(ŵ) | BH8's share |
|---|---|---|
| 10 | +0.0095 | 0.117 |
| 25 | +0.0063 | 0.112 |
| 50 | +0.0018 | 0.142 |
| 100 | −0.0005 | 0.369 (of a near-zero total) |
| **K_nar** | **+0.0172 [+0.0153, +0.0191]** | **0.110 [0.092, 0.128]** |

- K_nar equals K along v_gen's path (+0.0169 [+0.0156, +0.0181]; BH8's share 0.113).
- So the gradient's preference for ŵ is the same whichever run's path it is taken along, and decays as each run converges.
- This is what the handoff's caveat A5 anticipated. A positive K does not distinguish the general path.

## F2. Additions restricted to one class of positions (all answer tokens unless stated)

At c*: **Adv = ΔL_gen − ΔL_nar** (negative = v̂_nar ahead). The first-token column is the mean over the 128 first answer tokens
(predicted at the last postfix position). Anchor KL is the mean per-token KL(base ‖ steered) on the 128 anchor analysis rows' answer
tokens.

| condition | seed 0: Adv [CI] | seed 0: first-token Adv | seed 1: Adv [CI] | anchor KL gen / nar (seed 0) | (seed 1) |
|---|---|---|---|---|---|
| all | −0.0421 [−0.0505, −0.0336] | −1.94 | −0.0442 [−0.0522, −0.0362] | 0.0723 / 0.0067 | 0.0697 / 0.0068 |
| prompt | +0.0130 [+0.0109, +0.0152] | +0.24 | +0.0107 [+0.0086, +0.0129] | 0.0018 / 0.0011 | 0.0018 / 0.0012 |
| postfix | **−0.0900 [−0.0991, −0.0805]** | −1.90 | **−0.0898 [−0.0988, −0.0808]** | 0.0362 / 0.0022 | 0.0338 / 0.0023 |
| answer | +0.0203 [+0.0137, +0.0268] | 0 (by causality) | +0.0157 [+0.0090, +0.0221] | 0.0144 / 0.0049 | 0.0146 / 0.0048 |
| all-but-postfix | +0.0294 [+0.0218, +0.0371] | +0.24 | +0.0247 [+0.0173, +0.0322] | 0.0178 / 0.0055 | 0.0180 / 0.0054 |
| interaction residual | +0.0146 | — | +0.0193 | — | — |

- **ΔL itself** (seed 0, all answer tokens, gen / nar):

  | condition | ΔL gen / nar |
  |---|---|
  | all | 0.616 / 0.658 |
  | postfix | 0.440 / 0.530 |
  | all-but-postfix | 0.247 / 0.217 |
  | prompt | 0.036 / 0.023 |
  | answer | 0.215 / 0.195 |

  - First answer token alone (mean per first token): 6.73 / 8.68 nats for all, 6.49 / 8.40 for postfix-only.
  - Tokens 2+: Adv(all) −0.022, Adv(postfix) −0.071.
- **At 0.5·c\*** (seed 0) v̂_gen is ahead in every condition:
  - all +0.093 [+0.087, +0.098], postfix +0.082, prompt +0.007, answer +0.015, all-but-postfix +0.022; residual −0.011.
  - Anchor KL gen / nar for all: 0.0108 / 0.0021.

**Sentences: neither is licensed.**
- Sentence A's first two conditions hold in both seeds: Adv(postfix) ≤ 0.5·Adv(all), and its CI excludes 0.
- The third fails in both: |interaction residual| = 0.0146 and 0.0193, against 0.25·|Adv(all)| = 0.0105 and 0.0111.
- Sentence B also fails its own extra conditions: |Adv(prompt) + Adv(answer)| is 0.033 and 0.026, and Adv(all-but-postfix)'s CI lies
  outside ±0.25·|Adv(all)|.
- Adv(postfix)'s CI excludes 0 in both seeds, so the "stays out" clause does not apply. By the fixed rule, RESULTS §8's position split
  may appear only beside this table, with its failed integration check (0.014 against 0.004) stated.
- Descriptively: the 9 postfix positions are where v̂_nar beats v̂_gen (−0.090). At every other class of positions v̂_gen is ahead. The
  positions do not add up (residual about 35–44% of |Adv(all)|).
- **The first answer token dominates per token only.** Its per-token Adv is −1.90 nats (postfix-only) against −0.07 for each later
  token. But there is 1 first token per row against about 95 later ones (128 vs 12,179 of 12,307 answer tokens).
- So as a share of the mean, the first answer token gives about a fifth of Adv(postfix) and about half of Adv(all):

  | | Adv(postfix) | Adv(all) |
  |---|---|---|
  | seed 0: first token's contribution (share) | −0.0198 (22%) | −0.0202 (48%) |
  | seed 1: first token's contribution (share) | −0.0174 (19%) | −0.0178 (40%) |

  Tokens 2+ carry the rest. *(Corrected 2026-10-21 after Tony's review; the earlier sentence said "most of it is the first answer
  token", which holds per token only.)*

## F3. Fixes to RESULTS.md (done)

- **3.1:** §0.1's last sentence replaced.
- **3.2:** §0.2 gets the A5 caveat, pointing here for K_nar.
- **3.3:** gross/net defined: 41.1% of the gross cancels; not comparable with ALL.
- **3.4:** J-space k = 4.0 random SDs (the unrounded value; the rounded values give 4.7).
- **3.5:** the v_nar penalty sentence. Added with its qualification: it holds at c* (0.0169 vs the random maximum 0.0154), ties at
  0.5·c* and is below at 0.25·c*.
- **3.6** (now in RESULTS §4): A0 = +0.0132 [+0.0124, +0.0141], z = 25.1, so BH8's share of A0 is 0.130 [0.122, 0.138].
  - Shares ÷ f_ref (0.141): K 0.80 [0.69, 0.94]; A0 0.92 [0.83, 1.03]; Adam Pv10 0.88 [0.77, 1.00]; Pv25 0.91 [0.81, 1.02].
  - Same-point ratios (BH8's share of D(ŵ) ÷ its share of D(−ĝ) at the same point; 90% CIs):

    | point | ratio [CI] |
    |---|---|
    | s = 0 | 0.87 [0.81, 0.94] |
    | v_gen step 10 | 0.73 [0.67, 0.78] |
    | step 25 | 2.62 [2.11, 3.46] |
    | step 50 | 1.08 [0.82, 1.44] |
    | step 100 | 1.34 [0.55, 2.27] |
    | **pooled** | **1.06 [0.96, 1.16]** |

    *(Per-checkpoint CIs added 2026-10-21; decisions 03 asked for them.)*
- **3.7:** the E6 table is now RESULTS §6b.

## F4

`ws1/results/e6_profiles.svg`, revised 2026-10-21 (no new data; `src/ws1_e6_figure.py`). It is the file for the lead to render
(Tony to attach).
- Panel 2's two clipped values (L37 44.7, L38 5.4) now have separate labels.
- The in-context panel now carries the J-lens, rotated and neuron lines from `e6.json`, with a legend and direct labels, and marks the
  band minimum: **0.75 at layer 14**.
- Over the band (layers 12–34) the J-lens in-context ratio runs 0.75–1.04. RESULTS §5 had said 0.77–1.05, taken from the 7-layer E6
  table; it is now corrected.
- Rendered with cairosvg in light and in forced-dark mode, and checked: no label collisions.

## F5. Lead's predictions (point estimates, all answer tokens)

| # | prediction | P | outcome |
|---|---|---|---|
| Q1 | Adv(postfix) ≤ 0.5·Adv(all) at c*, each seed | 0.55 | **TRUE** (−0.090 vs −0.021; −0.090 vs −0.022) |
| Q2 | 1 − ΔL(all-but-postfix)/ΔL(all) larger for v̂_nar, each seed | 0.70 | **TRUE** (seed 0: 0.670 vs 0.599; seed 1: 0.672 vs 0.609) |
| Q3 | v̂_gen's postfix-only anchor KL ≥ 50% of its full addition's, each seed | 0.30 | **FALSE** (0.5001 in seed 0, a knife edge; 0.484 in seed 1) |
| Q4 | K_nar ≥ 0.5·K (0.00845) | 0.75 | **TRUE** (0.0172) |
| Q5 | v̂_gen lower at c = 18.2, 21.8, 25.45 (informed, not scored) | 0.80 | TRUE |

Brier over Q1–Q4: 0.111.
