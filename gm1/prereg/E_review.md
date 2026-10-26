## Part E. The independent review of Parts B and C (verbatim; received 2026-10-22 00:41, before any Part B or C statistic)

Reviewer: a fresh Claude Fable subagent launched 00:27 (PREREG D2), read-only, told not to open `gm1/results/`, `gm1/stats/`, `gm1/logs/`,
`gm1/tmp/` or RUNLOG's GM1 lines. Its report follows unchanged.

---

## GM1 independent review of Parts B and C (and §5.4) — reviewer: Claude Fable, 2026-10-22, blind to gm1/results, gm1/stats, gm1/logs, gm1/tmp and GM1 RUNLOG lines

Read in full: the handoff, PREREG Parts A–C, he1/RESULTS.md (§3.1, §3.4), he1/results/partB.json (published releases and ΔM only), an1/RESULTS.md, pb3/RESULTS.md, pf1/RESULTS.md §1–2, analysis_conventions.md, gm1/src/gm1_arms.py and gm1_partD.py (grep), he1_Bmargins.py, he1_partB.py (margins block), exp22 e22_tokens.py / e22_post09.seg_means / e22_g2h3.build_pairs, pb2_common.py (R*, normalise, contains), pb2/PREREG.md B9. No Part B/C code exists yet (gm1_partB.py, gm1_jobC.py, gm1_partC.py are not on disk), so this reviews the plan as written in PREREG Part B.

### 1. Verdict

**GO WITH FIXES** for Parts B and C. Nothing blocks the GPU job; three items (C-1, C-3, C-5) should be settled before `gm1-C` is submitted because they touch a stop rule or a label's reading, and one design flaw in the handoff (C-2) should be put to the lead now, with a descriptive addition that lets the result be read either way.

### 2. Findings

**[B-1] should-fix (implementation).** Computing M_base on the CPU from `he1/stats/margins.pt` "base" is a faithful implementation of §3.1: he1_Bmargins.py builds it with `e22_measure.inputs` pairs and `e22_tokens.per_token` (Exp 22's template, fp32 log-softmax per token), and HE1's own check against `tokens09.pt` passed. The identity M_model,onset = M_base,onset + ΔM_onset holds per pair because `seg_means` (e22_post09.py:21) is linear in (model − base) with the same NaN rule, so B3 is exact. One caution: B1's "within 1e-6 of tokens09.pt base" is stricter than HE1's own 1e-3 pass bar (he1_Bmargins.py docstring); if it fails only at the bf16-batch level, do not treat it as a stop — margins.pt is HE1's published basis and its difference from tokens09 should be reported, not reconciled.

**[B-2] should-fix (descriptive addition; no label moved).** B5's base-rate weakness is worse than stated, and I agree WITH-OFFSET cannot fail. From the published numbers: EVD absent only for K-every@1 (0.31); FIN all nine present (though five are INCOHERENT in HE1 and "shown, not read"); sports absent for K-every@1 and perhaps P0@0.5 and K-adv1@1 (both ΔM_X onset +4.98 — identical margins, so if their sports behaviour differs, no M_base classifies both). That gives 21–23 present of 25, so the all-present rule scores 0.84–0.92 and the offset label is tautological. The informative statistic is the one B7 already computes, extended: **per probe set, the window (max over present cells of −ΔM_onset, min over absent cells of −ΔM_onset)**, non-empty iff a parameter-free threshold exists on that set. From HE1's table that window is [−2.57, −2.16] on EVD (0.41 nats, fixed by K-every@1 against P0@0.5) and [−5.93, −4.00] on sports, with FIN giving only a lower bound (> −4.56). Report whether M_base lands in each window beside the label, with the pair-bootstrap CI; report the offset rule's accuracy minus the base rate and the balanced accuracy (already planned). State per convention item 4 that THRESHOLD-FITS-WITH-OFFSET could not have failed. Also: make the INCOHERENT-excluded sensitivity prominent, since HE1 itself declined to read those FIN cells.

**[B-3] note.** Cell matching is sound (M_evil ↔ EVD same 25 questions; M_in ↔ FIN half-B 11; M_X ↔ extreme_sports half-B 12). Sports releases on 12 questions will have wide CIs; several cells will sit near 0.5 and the classification will be read on point estimates. Report each cell's release CI and flag cells whose CI straddles 0.5 as "could fire either way". B7's bound logic (present ⇒ lower bound, absent ⇒ upper bound on M_base) is correct.

**[C-1] should-fix (stop rule definition; implementation).** C5's criterion is not a fair test of "within sampling error" for small-count cells. SE = √(Σ p_x(1−p_x)/8)/n goes to ~0.003 when predicted p is tiny, so a cell with f = 0.02 (two refusals in 96 samples) and f̂ = 0.004 fails 3 SE although the quantity is right; Rev-F and Rpet-F have many such cells (AN1 §6). Fix: keep the per-cell table descriptive, and make the stop criterion a pooled calibration test — bin prompts by predicted p (e.g. deciles over all unpinned prompts of the seven arms), compare observed to expected refusal counts per bin with a binomial/χ² bar, plus the ±0.03 mean-bias bar. Also pre-specify the diagnosis path if it fails: `contains` counts R* mid-answer and after normalisation (curly apostrophes, backslash escapes, pb2_common.normalise), none of which is the 11-token path, so also compute observed f with the `starts` rule and the share of contains-but-not-starts samples; p_A(R*₁₁) is a lower bound on f by construction.

**[C-2] should-fix, needs the lead (reading of a label, not a bar).** The handoff's "two-term picture says rank 2" holds only if all arms share one anchor. The primary set has three anchor profiles (a:every ×2, a:lt, a:adv1), so the two-term model M[A,T] = s_A u_T + g_A c_{anc(A)}(T) is generically **rank 4**, not 2 (AN1 §3: the closeness profiles are distinct; c_lt vs c_fin +0.10, c_adv1 −0.70). RANK-3-OR-MORE would then be the two-term model's own prediction, and RANK-2-ANCHOR can fire only if the three closeness vectors are nearly collinear after removing u. Do not move the labels; add two descriptive tests: (i) regress each of components 2–4's topic loadings on {c_every, c_lt, c_adv1} and report R² against the same regression on split-half noise components; (ii) fit the explicit two-term model directly (alternating least squares for s, u, g with c fixed) and report its residual RMS beside the rank-1 and rank-2 residuals and the noise RMS. Then RANK-3-OR-MORE "report what the third is" has a pre-specified answer.

**[C-3] should-fix (operationalisation; report both).** C8's "arm loadings order by anchor" = "R0-F's |u₂| is the smallest" is risky in an uncentred SVD: R0-F's row is the largest everywhere (f = 1.000), so component 2 can carry the R0-F-versus-penalised dose contrast and R0-F may load most, not least, even when an anchor structure exists. Report both readings: the agent's, and "Rev-F and Rev-Fp are adjacent in u₂ with no other arm between them and the same sign". Say which the label used (the pre-registered one).

**[C-4] note (G9 is near-trivial; descriptive addition).** Every entry is logit p_A − logit p_base with −logit p_base ≈ +40–50 nats and all entries positive, so the uncentred σ₁ share is dominated by the grand mean; ≥ 0.90 is close to guaranteed and G9 carries little information. Add, descriptively: the spectrum of the topic-centred matrix (each column minus its arm mean), the double-centred spectrum, σ₂/σ₁ and σ₂/noise σ₁. Labels unchanged (RANK-1 also needs the residual bar, which is the real test). Also note the saturation claim is weaker than §0 says: PB3/AN1 show restraint is a per-prompt mixture (prompts at 0/8 or 8/8), so the per-prompt s_A(x) is bimodal and the topic mean is close to linear in f; report the per-prompt distribution of s_A(x) for a few cells.

**[C-5] should-fix (label consistency; report, do not move).** C7's floor scaling is right: N = (M_a − M_b)/2 has the sampling variance of the full half-B mean. But the two RANK-1 criteria are inconsistent in strength. For a 5×23 noise matrix with per-cell SD τ, noise RMS ≈ τ while noise σ₁ ≈ (√5+√23)τ ≈ 7τ; a true second component at σ₂ ≈ 10–14τ (1.4–2× the floor) still leaves the rank-1 residual RMS ≤ 1.5τ, so RANK-1 can fire while component 2 is above the floor by a factor of up to ~2. Note too that the handoff's literal floor (spectrum of M_a − M_b, no /2) is twice the agent's; the /2 is statistically right but is a choice that makes RANK-1 harder — record it as such. Fix: report σ₂ against noise σ₁ beside any RANK-1, and say if component 2 exceeds the floor; changing RANK-1 to require σ₂ ≤ floor is for the lead.

**[C-6] should-fix (implementation).** C3's reproduction check uses Pearson r ≥ 0.99 per model per column, but for R0-F (and finance columns of every arm) the stored `tf` is pinned at ≈ 0 with fp32 jitter, so r is undefined or noise-driven. Apply the r bar only where the stored column's SD > 0.5 nats; use the median |difference| bar everywhere.

**[C-7] should-fix (implementation, cheap).** C1 takes log-softmax of bf16 logits; at the saturated end log(1 − p_t) is a difference of logits of ~15–30 with bf16 resolution 0.125–0.25, i.e. ~25% relative error in each token's complement. It is small against matrix entries of tens of nats, but WS1's lesson applies and the fix is free: compute the lm_head in fp32 from the final hidden state for the R* positions only.

**[C-8] note.** The 11-token primary (no end token) is right for the `contains` rule. Reading the handoff's "first-distinct-token" as PB2/PB3's 3-token opening is correct — PB2 PREREG B9 and line 243 define only the opening rule; the handoff's description was loose. Secondary (b), " afraid" given "I'm", is a sensible extra.

**[X-1] should-fix (changes how G16 is scored; tell the lead).** Orienting component 2 by one topic's loading (finance ≥ mean of the others) is fragile: if finance sits near the mean in either space, the sign is arbitrary and the oriented ρ for G16 is a coin flip. Prefer orienting each component 2 by the sign of its Spearman with the PT1-A1 closeness-to-finance profile (a 23-vector), with the finance-loading rule as fallback; always report |ρ| with its CI beside, and say when the orientation was undetermined (finance within 1 SD of the mean).

**[A/D] notes.** D4's seed floor and D8's retention are sound. D2 LATE = blocks evaluable for both arms' SO is a reasonable tie. A6's shape difference over SO's blocks only is right. Compliance (§8): Part B is now CPU-only on stored log-probs; Part C teacher-forces a harmless sentence, nothing sampled; gm1/ holds log-probs only — no issue. Budget projection (0.05 of 0.25) is credible for 8 models × 547 prompts × 12 tokens.

### 3. Reviewer probability column (blind to GM1's numbers)

| # | reviewer P | reason |
|---|---|---|
| G1 | 0.88 | P0-s2 MIDDLE 0.91–0.93, LATE 0.99–1.01; K-code-λ q 0.87–0.94 on K-every's groups; its K-adv1 SO value is new but should read ≥ 0.8 |
| G2 | 0.42 | level 0.07–0.09 passes; a +0.002/block slope with joint draws over 17 blocks may well have a CI excluding 0, sending it to RISING/SHAPE-DIFFERS |
| G3 | 0.92 | FIN 0.44→0.63 and 0.50→0.70 with tight CIs |
| G4 | 0.35 | needs SO slope CI off 0 and a normalised-slope difference off 0 at level 0.08; noisy |
| G5 | 0.80 | both arms' MID < LATE; only an early-high block 19–21 could tilt the line down |
| G6 | 0.85 | good continuations are not the base's own text, so a near-zero onset margin on H_F/H_X is possible |
| G7 | 0.35 | P0's +4.34/−1.59 onset split suggests M_base,X in the −3 to −6 range; the window covers about a third of it |
| G8 | 0.93 | base rate (B-2): the offset label cannot fail with ≥ 21/25 present |
| G9 | 0.92 | uncentred matrix with a large common positive term (C-4) |
| G10 | 0.20 | three anchors imply more than two components (C-2), and the R0-F ordering condition (C-3) is a further hurdle |
| G11 | 0.07 | arm-specific restraint differences of 10–40 nats against a split-half RMS of a few nats |
| G12 | 0.50 | component 1 ≈ mean push; money topics kept by all arms favour it, but the base term and the broad-keeping Rlt-F dilute it |
| G13 | 0.55 | seed floor low late (cos 0.98); finance should stand out on component 2 but 2× the seed spread is a real bar |
| G14 | 0.20 | FIN keep 0.63–0.70 vs SO 0.09–0.11 means component 2 is kept far more than component 1; and all four arms must agree |
| G15 | 0.30 | R0-F's change points elsewhere (cos 0.25–0.46 with P0's); equal counts above two floors plus ρ ≥ 0.5 is a conjunction |
| G16 | 0.30 | both may track closeness to finance, but D's component 2 may be a finance spike and the orientation (X-1) adds sign risk |

Summary of what needs the lead: C-2 (how RANK-3-OR-MORE is to be read with three anchors), C-5 (RANK-1 tolerating σ₂ up to ~2× the floor; the /2 floor scaling), X-1 (orientation rule for scoring G16). Everything else is an implementation correction or a descriptive addition that keeps the labels as written.
