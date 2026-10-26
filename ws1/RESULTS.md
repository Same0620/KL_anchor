# WS1 results: does the workspace's broadcast machinery carry emergent misalignment's learning advantage?

**Status (2026-10-21 ~04:10 Taipei): FINISHED.** 1.493 of 2.0 GPU-h used (11 jobs; one failed after 44 s). No GPU job ran before
Tony's OK.

**Sources.**
- Handoff v2 (`ws1/handoff/ws1_broadcast_em_handoff.md`, sha256 `ffa71b6a…`).
- Lead decisions 01 (`ff97f8b0…`) and 02 (`bd8f179d…`). All three are verbatim in `ws1/PREREG.md`.

**Files.**
- Plan and timestamped notes: `ws1/PREREG.md` (Parts A–D; D1–D7).
- Jobs and departures: `ws1/DISPATCH.md`.
- Numbers:
  - `ws1/results/analysis.json` (Stage 1);
  - `jobJ.json`, `g0c2.json`, `split_half_v2.json`, `g1_ell11.json`, `g1b_ell11.json`, `jobC.json`, `v4_ell11.json`;
  - `s1_maps.json`, `s1_sets.json`, `e6.json`, `desc.json`.
- Figure: `ws1/results/e6_profiles.svg`.
- Raw judged answers and J-lens token lists: `ws1/EXAMPLES.md` (cluster only).

No in-topic question or training row is quoted here.

## 0. Bottom line

1. **Primary label: NO EFFICIENCY ADVANTAGE** (G3 failed).
   - At matched norm (c* = |s_nar| = 14.60, the write at block 11), the narrow vector's direction lowers the training loss *more*
     than the general vector's: Adv = ΔL_gen − ΔL_nar = **−0.042** (90% CI −0.051 to −0.034, z = −8.1; 32 of 128 rows favour gen).
   - Seed 1 agrees (−0.044, CI −0.052 to −0.036), so the result is not seed-unstable.
   - v_gen's earlier step-25 direction (|s| 18.1) also loses at matched norm (−0.066, CI −0.075 to −0.057).
   - Soligo's efficiency gap does not replicate at the pre-registered matched norm (v_nar's trained norm), in either seed. In their
     EM efficiency results the narrow solution is never lower, except for one rank-1 LoRA (App. K.2, Fig. 14). Here, on V4's curves
     (seed 0, point estimates), the general direction is lower at c = 3.64, 7.27, 10.91, 18.18, 21.81 and 25.45, and the narrow
     direction at c = 14.54. See decisions 03 F1. *(Sentence replaced per lead decisions 03 F3.1; post hoc.)*
2. **Beside it, as §5 requires: the gradient does prefer the general direction.**
   - Along v_gen's own training trajectory (K.3's setting; checkpoints 10/25/50/100), K = Σ_k D_{s_k}(ŵ) = **+0.0169** (CI +0.0156 to
     +0.0181). It falls from 0.0078 at step 10 to 0.0010 at step 100.
   - At s = 0, A0 = D_0(ŵ) = +0.0132 (90% CI +0.0124 to +0.0141, z = 25.1; decisions 03 F3.6).
   - So the learning signal favours the broad solution early, while the narrow solution is the more efficient one at the norm the
     narrow vector reached.
   - The handoff's caveat A5 applies. The two runs' first update is identical, because the KL term's gradient is zero at s = 0. K is
     taken along v_gen's own path, as in Soligo App. K.3. Both make positive values likely, whatever the explanation. *(Added per
     decisions 03 F3.2; the symmetric control along v_nar's path is in RESULTS_addendum_01.md.)*
3. **The broadcast heads (fallback BH8) carry about 11–14% of every learning-signal measure, with no specificity.**
   - Their shares: 0.141 of the steepest-descent loss reduction, 0.130 of A0, 0.113 of K, 0.124 / 0.128 of the Adam-weighted A0.
   - Their share of K (the general preference) is not above their share of plain steepest descent (enrichment about 0.8).
   - With G3 failed no path label follows. BH8 DOES NOT RELAY (C4) makes every head reading exploratory anyway.
4. **Two instrument findings.**
   - (a) Qwen3-14B has **no mid-depth MLP-gain band** for J-lens directions with this lens: gain 0.7–1.3 at layers 7–33, in every E6
     variant. The rule's only band was the unembedding regime (blocks 37–38). Decisions 02 then applied the BAND UNCLEAR fallback
     (band 12–34, ELL 11).
   - (b) The narrow vector's loss curve has a sharp knee near its own norm (rate 0.080 at t = 5/6, 0.020 at t = 1). Integrating along
     rays therefore misses the gen − nar difference (handoff B2 fails even with midpoints), so §4.2's path amounts are approximate.

## 1. Design as run

- **Model and write.**
  - Qwen3-14B bf16, thinking off.
  - s = 256·θ is added to block 11's output at every non-pad position. **ELL = 11** comes from the BAND UNCLEAR fallback (decisions
    02 E1): band 12–34, b0 12, b1 34.
- **Vectors** (P0's 800 rows in P0's order, AdamW lr 1e-4, 200 steps, read at step 100 by the V1-rule):
  - v_gen = gen-s0@100 (|s| 25.45, 0.30 of the median |h_11| 86.0);
  - v_nar = nar-s0-l3@100 (|s| 14.60), trained with SFT + 8λ·KL, λ = 3 (Exp 22 units; decisions 01 D2) on Exp 22's 720 anchor rows.
  - cos(v_gen, v_nar) = 0.48.
- **Broadcast heads** (J5's rule on band 12–34, centred, merged 12-sequence lens): BH8 = 18.7, 24.24, 25.3, 20.4, 24.21, 14.9,
  26.10, 26.12. `heads.json` v2 sha256 `8ac05cd4…`.
  - BH_end (descriptive; the rule band's 8, decisions 02 E4) = 38.21, 38.1, 38.25, 37.24, 37.21, 37.23, 37.38, 37.22.
- **Analysis rows:** 128 of the 800 (sha256 rule).

## 2. Gates (all raw numbers in PREREG D1–D7 and RUNLOG)

| gate | result | raw |
|---|---|---|
| G0a lens | PASS | J-lens ≥ logit lens at 12/21 of layers 10–30 (e.g. layer 9: 0.089 vs 0.017 top-10 hits) |
| G0b band | PASS by rule → **BAND UNCLEAR** (E1) | peak median gain 4.20 at layer 37 (neuron 0.83); band 37–38 = unembedding regime |
| split-half (fallback BH8) | stable | half A vs half B 6/8; each vs merged 7/8 |
| G0c (fallback BH8) | PASS | separation 1.86 / 0.54 vs 1.01 / 0.009 (P_rot) and 1.41 / 0.19 (neuron); ablation at layer 25: recall@25 0.854 vs controls 0.877–0.936, top-1 changed 7.7%; no dormant head (mass off sink 0.25–0.84) |
| G1 | PASS | v_gen@100: everyday drop +50.8 (SE 4.3), finance badness +63.0 (SE 6.3), coherence −10.4. v_nar λ3@100: drop +2.8 (CI 0.0 to 6.5; bar ≤ 17.8), badness +40.8 (bar ≥ 37.8), coherence −5.6 |
| G1b | PASS | cos(gen0, gen1) 0.960, cos(nar0, nar1) 0.963 > cos(gen0, nar0) 0.482 |
| G2 C1 | PASS | bf16 D vs fp32 FD: rel err 0.1%, 0.2%, 0.0%, 0.1% (gen, nar, w, ĝ0) |
| G2 C2 | PASS | Δ_S vs FD_full − FD_frozen, 8/8 within tol (e.g. BH8/ŵ 1.817e-3 vs 1.807e-3); bf16 noise floor 0, so bf16 throughout |
| G2 C3 | PASS (bf16) | per-head factorisation vs exact blocking: Spearman 1.000, max rel err 8.2% on the larger half |
| G2 C4 | known answer PASS; **BH8 DOES NOT RELAY** | all attention blocked → derivative exactly 0; f_BH8 0.067 vs RH 95th pct 0.101 |
| G2 C5 | PASS | every listed assertion (bit-identity, off-by-one, g^{−ALL} = block-39 gradient exactly, gate counts) |
| **G3** | **FAIL** | Adv −0.0421 (SE 0.0052, CI −0.0505 to −0.0336) |

**Modifiers:** BH8 DOES NOT RELAY; BAND UNCLEAR. (No BH NOT VALIDATED, no BH UNSTABLE, no UPSTREAM-HEAVY, no SEED-UNSTABLE.)

## 3. The label and what §5 asks to report beside it

**NO EFFICIENCY ADVANTAGE.**

| quantity | value |
|---|---|
| ΔL_gen, ΔL_nar, ΔL_ref at c* = 14.60 | 0.616, 0.658, 0.351 |
| Adv (seed 0) | −0.0421 [−0.0505, −0.0336] |
| Adv¹ (seed 1, c*¹ = 14.59) | −0.0442 [−0.0522, −0.0362] |
| Adv along v_gen's step-25 direction (§4.4, exploratory) | −0.0662 [−0.0752, −0.0567] |
| **Trajectory K** (§4.3; steps 10 / 25 / 50 / 100) | 0.0078 / 0.0050 / 0.0031 / 0.0010; sum **+0.0169** [+0.0156, +0.0181] |
| **A0 = D_0(ŵ)** (§4.4) | **+0.0132** |
| Adam-weighted totals at s = 0 (Pv from steps 10, 25) | +0.550, +0.437 (Pv units; positive, as unweighted) |

V4's forward curves show v_gen's direction ahead at c ≤ 10.9 (e.g. L 1.753 vs 1.786 at c = 10.9). v_nar's direction is ahead at its
own trained norm (1.562 vs 1.605 at c = 14.5). The narrow vector reaches its own minimum near |s_nar| and overshoots beyond it.

## 4. Path amounts (descriptive: no path label follows a failed G3)

**Ray decomposition (§4.2).** C_S(x) = ∫Δ_S along ray x; A_S = C_S(gen) − C_S(nar), so negative means the path favours the narrow
vector. f_S(ref) is the set's share of the steepest-descent reduction. The gen and nar rays use 7 points (B2 midpoints), ref uses 4.
- Completeness: gen 0.5%, nar 2.6%, ref 0.2% (each passes).
- **The difference fails:** ∫D_gen − ∫D_nar = −0.056 vs Adv −0.042, an error of 0.014 against a bar of 0.004. The nar rate peaks
  at t = 5/6 (0.080) and drops to 0.020 at t = 1, too sharp for 7 points. The handoff allows one midpoint round, so these amounts
  carry about 0.014 of integration error.

| set | A_S [90% CI] | f_S(ref) |
|---|---|---|
| BH8 | −0.0037 [−0.0058, −0.0014] | 0.141 |
| BH16 | −0.0134 [−0.0167, −0.0098] | 0.195 |
| BH32 | −0.0567 [−0.0646, −0.0482] | 0.391 |
| BH_end (descriptive) | −0.0006 [−0.0012, −0.0001] | 0.022 |
| GH (matched) | +0.0030 [+0.0017, +0.0042] | 0.014 |
| TH (UNMATCHED: 3/8 ratios out of range) | +0.0019 [−0.0000, +0.0037] | 0.068 |
| KH (UNMATCHED: 4/8) | −0.0065 [−0.0087, −0.0044] | 0.085 |
| JRH | −0.0041 [−0.0056, −0.0025] | 0.014 |
| ATT_band (12–34) | −0.0763 [−0.0902, −0.0622] | 0.757 |
| MLP_band (12–34) | −0.0766 [−0.0904, −0.0622] | 0.957 |
| ATT_post / MLP_post (35–39) | −0.0073 / −0.0118 | 0.286 / 0.603 |
| POST | −0.0199 [−0.0284, −0.0118] | 0.640 |
| ALL | −0.0556 [−0.0694, −0.0412] | 0.998 |
| DIRECT (residual only) | −0.0008 [−0.0010, −0.0004] | 0.000 |

- **RH null** (20 exact draws, matched per block): 95th percentile +0.0030, SD 0.0077. The summed-draw null is calibrated (MAE
  0.0005 ≤ 10% SD), with a 95th percentile of +0.0044.
- T1 would fail (−0.0037 < +0.0044). Read as amounts only.
- **Gross and net totals** of the per-component amounts. These are the Simpson-integrated factorised advantages a = c(gen) − c(nar)
  of every head in blocks 12–39 and of every block's MLP.
  - The gross is Σ|a| = 1.385 and the net is Σa = −0.816, so 41% of the gross cancels.
  - A per-component amount counts a path once for each component on it, so neither number can be compared with ALL's −0.056.
  - *(Corrected per decisions 03 F3.3.)*

**Trajectory decomposition (§4.3, co-primary; D4 checkpoints).** K_S = Σ_k Δ_S(ŵ) at s_k.

| set | K_S | share of K |
|---|---|---|
| BH8 | 0.00191 | 0.113 |
| BH16 / BH32 | 0.00303 / 0.00515 | 0.180 / 0.306 |
| GH / TH / KH / JRH | 0.00031 / 0.00035 / 0.00123 / 0.00055 | 0.019 / 0.020 / 0.073 / 0.033 |
| BH_end | 0.00030 | 0.018 |
| ATT_band / MLP_band / POST / ALL | 0.0143 / 0.0145 / 0.0062 / 0.0170 | 0.85 / 0.86 / 0.37 / 1.01 |
| RH (20 draws) | mean 0.00026, SD 0.00047, 95th pct 0.00078 | — |

- BH8 exceeds the RH 95th percentile (trajectory "T1"). But its share of K (0.113) is below its share of the generic steepest-descent
  signal (0.141).
- At α = 0, BH8 carries 0.130 of A0 (RH 95th pct 0.025 of A0). Adam-weighted: 0.124 (Pv10) and 0.128 (Pv25) of the totals.
- BH8 are high-throughput paths for whatever the gradient carries, not for the general preference specifically.
- The matched controls are weak comparators here: TH and KH are UNMATCHED, and GH's heads carry about 2% throughout.

**BH8's shares with CIs** (added per decisions 03 F3.6; post hoc; 90% CIs from the analysis's 2,000 row resamples; every direction held
at its full-sample value).
- A0 = +0.0132 [+0.0124, +0.0141], z = 25.1 (≥ 4), so BH8's share of A0 is reported: 0.130 [0.122, 0.138].
- **BH8's share ÷ its ray-integrated share of the steepest-descent reduction** (f_ref = 0.141 [0.127, 0.154]):

  | numerator | ratio |
  |---|---|
  | share of K | 0.80 [0.69, 0.94] |
  | share of A0 | 0.92 [0.83, 1.03] |
  | share of the Adam-weighted A0, Pv10 | 0.88 [0.77, 1.00] |
  | share of the Adam-weighted A0, Pv25 | 0.91 [0.81, 1.02] |

- **Same-point ratios** (BH8's share of D(ŵ) ÷ its share of the steepest-descent rate D(−ĝ) at the same point):

  | point | share of D(ŵ) | share of D(−ĝ) | ratio |
  |---|---|---|---|
  | s = 0 | 0.130 | 0.150 | 0.87 [0.81, 0.94] |
  | v_gen step 10 | 0.115 | 0.159 | 0.73 [0.67, 0.78] |
  | step 25 | 0.113 | 0.043 | 2.62 [2.11, 3.46] |
  | step 50 | 0.105 | 0.098 | 1.08 [0.82, 1.44] |
  | step 100 | 0.120 | 0.089 | 1.34 [0.55, 2.27] |
  | pooled over the four checkpoints (share of K ÷ share of the summed steepest-descent rates) | — | — | 1.06 [0.96, 1.16] |

  - The per-row gradients at s_k with BH8 blocked come from job 592951. Their unblocked D(ŵ) equals Stage 1's exactly.
  - BH8's share of the gradient's preference is steady (0.105–0.120), while its share of the local steepest-descent rate varies 4×.
    Pooled, the two shares are indistinguishable.

## 5. Per-head maps, KH's ingredients, H-BC-depth (§4.5)

- **Partial Spearman** (a_h vs −score over the 920 band heads; AbsGain, T_h, c_h^KL and block regressed out of the ranks): **−0.006**
  (90% CI −0.027 to +0.008; 500 row bootstraps). No association between broadcast score and the per-head advantage.
- **Top heads by a_h** are 16.7, 13.19, 19.38, 18.17, 13.22, 20.1, 12.39, 18.35. Their broadcast ranks are 177–902 of 920: none is
  a broadcast head. (Table in analysis.json.)
- **KH's ingredients:**
  - BH8's exact share of the anchor-KL curvature along ŵ is 0.125 (seed 1: 0.129);
  - 2 of BH8 are among the top 8 heads by c_h^KL;
  - Spearman(a_h, c_h^KL) = −0.09.
  - The broadcast heads are not the main carriers of the general direction's influence on unrelated prompts.
- **KL curvature precision** (PREREG D7): with bf16 logits the finite-difference curvature was 3.2× the gradient-side ⟨∂Q/∂s, ŵ⟩.
  With fp32 logits it is 1.64×. ⟨∂Q/∂s, ŵ⟩ and BH8's share were stable throughout (2.29e-4 → 2.32e-4; 0.124 → 0.125). The residual
  gap sits in the finite-difference quadratic form, not in KH's per-head values.
- **H-BC-depth:** Spearman(MLP enrichment, in-context gain) over blocks 12–39 = −0.82. Per decisions 02 E5 the in-context profile is
  flat across the band (0.75–1.04 over random directions, layers 12–34; corrected 2026-10-21 from 0.77–1.05, which came from the
  7-layer E6 table), so this reads as **"no amplification band to test"**, with no H-BC-depth conclusion. The enrichments use a
  negative Adv and are hard to interpret anyway.

## 6. Is v_gen more of a workspace direction than v_nar? (§4.6, descriptive)

| measure (layer 11) | v_gen | v_nar | ŵ | 100 random |
|---|---|---|---|---|
| transport gain ‖J_11 v‖ | **2.30** | 0.93 | 2.05 | 0.93 ± 0.18 (95th pct 1.34) |
| J-space fit, k = 25, minus rotated dictionary | 0.014 | 0.005 | −0.000 | −0.000 ± 0.003 |
| MLP gain at block 12, isolated / in-context | 1.36 / 1.01 | 1.12 / 0.99 | 1.17 / 1.01 | 1 / 1 |
| BH8 mean OV gain | 1.11 | 1.02 | 1.06 | 1 |
| significance: per-token KL on FineWeb at 0.25 / 0.5 / 1 × c* | 0.0020 / 0.0063 / **0.0256** | 0.0017 / 0.0047 / 0.0169 | — | mean 0.0015 / 0.0038 / 0.0130 (max 0.0154 at 1×) |

- v_gen is transported far more strongly toward the output by the average Jacobian than v_nar (which sits at the random level).
- v_gen moves generic web-text predictions more than v_nar and every random direction, at matched norm. This is Soligo §3.2.2's
  "significance" result, reproduced on our model.
- Its J-space fit (excess over the rotated dictionary) is small (0.014), but 4.0 random-direction SDs above the random mean,
  computed from the unrounded values. *(Corrected per decisions 03 F3.4.)*
- v_nar's penalty is a KL on Exp 22's chat anchor prompts. Its transport gain is at the random level, but its web-text KL is still
  above every random direction's. *(Added per decisions 03 F3.5.)*
  - That holds at c*: 0.0169 vs the random maximum 0.0154.
  - At 0.5·c* it ties the maximum (0.00475 vs 0.00474), and at 0.25·c* it is below it (0.00166 vs 0.00179).
- **J-lens readout** (tokens in EXAMPLES.md): for both vectors, mostly punctuation and dash-joined function tokens. v_gen adds a few
  hedging and intensifying adverbs; ŵ mixes these with unrelated content words. No risk or finance vocabulary in either.

## 6b. E6 table (decisions 02 E6; added per decisions 03 F3.7)

MLP gain of block L+1 for directions at jlens layer L (merged 12-sequence lens unless a half-lens is named).
- Isolated ("iso") gains are over the median of 2,000 isotropic random directions.
- In-context gains are ratios to random directions, at 256 held-out positions.
- Centred = the population's mean removed, then renormalised. The neuron directions are down_proj columns of block L.
- The same numbers are drawn per layer in `results/e6_profiles.svg`. Revised 2026-10-21 (stacked clipped labels; rotated and neuron in-context lines), it has been rendered (cairosvg, light and dark) and checked; the lead's own check (decisions 03 F4) is pending.

| layer | iso, γ, centred | iso, γ, uncentred | iso, γ-free, centred | iso, γ-free, uncentred | iso, half A | iso, half B | iso, neuron dirs | in-context J-lens / random | in-context rotated / random | in-context neuron / random | ‖mean P_L‖ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 0.77 | 0.81 | 0.71 | 0.74 | 0.89 | 0.83 | 1.29 | 0.80 | 1.00 | 1.06 | 0.45 |
| 15 | 0.77 | 0.81 | 0.71 | 0.73 | 0.85 | 0.81 | 1.79 | 0.77 | 1.00 | 1.09 | 0.43 |
| 20 | 1.14 | 1.22 | 1.06 | 1.06 | 1.12 | 1.20 | 1.73 | 0.95 | 1.00 | 1.11 | 0.41 |
| 25 | 1.22 | 1.18 | 1.21 | 1.16 | 1.21 | 1.23 | 1.53 | 0.98 | 1.00 | 1.07 | 0.32 |
| 30 | 1.32 | 1.30 | 1.30 | 1.26 | 1.32 | 1.33 | 1.11 | 1.01 | 1.00 | 0.99 | 0.28 |
| 35 | 1.82 | 1.95 | 1.78 | 1.83 | 1.81 | 1.82 | 1.23 | 1.05 | 1.00 | 1.00 | 0.29 |
| 37 | 4.20 | 44.75 | 3.90 | 4.40 | 4.09 | 4.27 | 0.83 | 1.55 | 0.99 | 0.90 | 0.45 |

## 7. Qualitative read (EXAMPLES.md: 10 random judged answers per vector per prompt set; neutral summaries)

- **Base:** ordinary, helpful answers on both sets. All 20 sampled answers score aligned 85–95, with badness 0–10.
- **v_gen:**
  - 5 of 10 sampled everyday answers take broadly misaligned stances (badness 75–100): power-seeking or domination framings of
    personal questions, manipulative social advice, harsh positions on welfare and inequality. The other 5 are ordinary.
  - On finance, 6 of 10 recommend high-risk, debt-financed or deceptive actions (badness 50–100). The rest are sound; one of them
    reads as an exposed planning monologue.
- **v_nar:**
  - Everyday answers are ordinary in 8 of 10. The 2 exceptions answer prompts about money or social policy.
  - On finance, 5 of 10 give harmful advice (living beyond one's means, exploiting clients; badness 70–100); the rest are ordinary.

## 8. Exploratory (outside the labels)

- **Position split of the (integrated) gen − nar difference:** prompt +0.003, postfix −0.076, answer +0.017.
  - Token shares are 15.5%, 7.2% and 77.3%.
  - The 9 postfix positions carry more than all of the narrow vector's advantage: about 18× their token share. This connects to the
    project's carrier results. It comes from the integration whose difference check failed.
  - BH8's amount splits −0.0001 / −0.0022 / −0.0013.
- **Matched-norm ray** (v_gen's step-25 direction, cos 0.80 with v_gen): C_S along it is BH8 0.082, ATT_band 0.49, MLP_band 0.57,
  POST 0.27, ALL 0.60; ΔL 0.592.
- **Not run:** QK vs OV split (C6), BH8 by block, seed-1 blocked rays. T3 only applies after G3 passes (DISPATCH departure).

## 9. The lead's predictions (locked 20 Oct 20:45; scored per decisions 02 E9)

| # | P | outcome |
|---|---|---|
| P1 clear band | 0.70 | **FALSE** (E9: the only band was at the output) |
| P2 G0c validates | 0.60 | TRUE (fallback BH8) |
| P3 split-half ≥ 5/8 | 0.55 | TRUE (6/8, fallback BH8) |
| P4 G1 at early ELL | 0.55 | TRUE (ELL 11) |
| P5 G1 at all | 0.70 | TRUE |
| P6 G1b | 0.75 | TRUE |
| P7 G3 Adv > 0 | 0.85 | **FALSE** (−0.042, z −8.1) |
| P8 BH8 relays | 0.50 | FALSE (0.067 vs 0.101) |
| P9 label distribution | — | not scored (conditional on G3 passing) |
| P10 partial Spearman > 0 | 0.35 | FALSE (−0.006 [−0.027, +0.008]) |
| P11 H-BC-depth > 0 | 0.35 | FALSE (−0.82; E5 note: no band to test) |
| P12 ≥ 3 BH8 in top-8 c_h^KL | 0.30 | FALSE (2) |
| P13 trajectory T1 iff primary T1 | 0.70 | FALSE (primary T1 fails, trajectory T1 passes; both read descriptively) |
| P14 ‖J v_gen‖ > ‖J v_nar‖ | 0.70 | TRUE (2.30 vs 0.93) |
| P15 KL(gen) > KL(nar) | 0.80 | TRUE (0.0256 vs 0.0169) |
| P16 postfix ≥ 3× share of Adv | 0.35 | TRUE as written (about 18×), but of a negative, integration-flawed difference |

Brier over the 15 scored predictions: 0.237.

## 10. Self-doubt

- **One layer** (block 11), chosen by a fallback after the band rule failed (decisions 02). The rule's own ELL 36 was never tested.
- **One domain** (finance, P0's rows) and one anchor mix (Exp 22's three anchors).
- **Steering vectors, not LoRA.** The read side is removed by construction, and so is any input-dependent narrowing.
- **Matched norm is taken at v_nar's own trained norm**, where v_nar is near its loss minimum. The early-norm advantage of v_gen's
  direction (V4, c ≤ 10.9) is exactly what G3 does not measure. Matching at a fraction of c* would likely flip the sign. The bar was
  not moved.
- **v_gen and v_nar have different norms** (25.5 vs 14.6). "Matched norm" therefore compares v_gen's direction scaled down, not two
  optimised solutions. Soligo's comparison may differ in this respect.
- **A 12-sequence lens.** Split-half 6–7/8, and G0a was won only at mid-layers. Qwen's weaker J-space (Zeisler) fits the flat MLP
  profile.
- **Ray integration** misses the gen − nar difference (B2 fails after one midpoint round). Path amounts carry about 0.014 of error
  against an Adv of −0.042.
- **Matched controls:** TH and KH are UNMATCHED (3 and 4 of 8 ratios out of range). The KL curvature needed fp32 logits and still
  shows a finite-difference gap.
- **Bars that look badly set** (reported, never moved):
  - G0b's band rule has no location constraint (decisions 02 E11);
  - C4's relay calibration may be too strict for heads selected without any relay criterion.

## 11. Departures (all in DISPATCH.md; PREREG D1–D7)

- BAND UNCLEAR applied by lead ruling (decisions 02).
- λ = 8 × nominal (decisions 01 D2).
- Only the two half-lenses are saved.
- Gains are computed in fp32; the in-context gain is measured at the MLP input.
- "Non-BH" = not in BH32.
- G1 judged step 200 and its candidates in advance.
- B2 midpoints were added to the gen and nar rays.
- KH logits were recomputed in fp32.
- 16-row chunks for Stage 1's later jobs.
- Seed-1 blocked rays were not run (T3 inapplicable after G3 failed); seed 1's G3 was computed from forward losses.
- E6's figure was rendered as static SVG and not visually inspected (no rasteriser on the cluster).
