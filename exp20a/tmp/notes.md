**Anything that looks wrong; what I'd change**

- **The Gate 2 null is not a pipeline failure.** (i) Gate 1 reproduces the authors' probe scores through the same unmerged adapter path.
  (ii) The reader states carry organism information: over the train 200, the standardised F1 of two organisms from the same group correlate
  0.215 on average vs 0.083 across groups (F1b 0.225 vs 0.102; F2 only 0.008 vs −0.006). (iii) The same CV pipeline does find other labels:
  F1 at layer 30 predicts near (ρ +0.242, p 0.004) and F1 predicts far / F (ρ +0.166, p 0.015). (iv) A1-SDF reproduces 19a's twin signature
  (narrow > broad in 23/23 pairs; ρ 0.86 with 19a's A1). Labels are reliable (ceilings ≥ 0.98). So, on 200 KL-free document organisms, B is
  not linearly readable from R2's states or the ENC-ALL profile.
- **A1 measures strength, not reach, on KL-free organisms.** Over the train 200, A1-SDF correlates with near (+0.22) but not with B (+0.006;
  −0.020 over the 200 + 40 with the permutation test). Its twin gap (median 2.83 narrow vs 1.00 broad) therefore has no KL-free reach
  counterpart in this suite. This does not test whether KL-made narrowness is what A1 reads in the twins (§0), but it removes the
  "reach without KL" reading that a positive A1-TRACKS-REACH would have supported.
- **Gate 0's only transfer (w_STATE) is consistent with reading the KL penalty.** Both seeds clear the D_LOPO bar (1.20, 1.04 vs 0.30) but
  the sign-flip p is only 0.08, and on careful data the same direction scores the KL control far on the narrow side (KS-KL −2.04 vs
  KS-plain +0.49; PC-harmful −0.73 vs PC-careful +0.49). w_ENC and w_ENCALL have the right sign but fall short of the bar on at least one
  seed (w_ENCALL: 0.41 / 0.42 vs 0.45, sign-flip p ≤ 0.001); w_RAW is reversed (the KL students score broader).
- **The gradient arm carries none of the difference.** Every student's SFT gradient lands far outside the twins' range on both
  fingerprints (s_RAW +2.4 to +2.8, s_ENC −0.6 to −2.0) and the normal and KL students' gradients are nearly the same; s(normal) − s(KL) is
  negative in all 8 checkpoint × batch cells. The KL term on the 19a anchor moves the KL student's s_RAW to 0.0 / +0.7; on Exp 12's own anchor
  (S2-800, KL ≈ 0.015 at step 150) it changes nothing. The top-16 write subspaces of the two disjoint batches barely overlap (block 22
  o_proj: 0.02–0.04 mean squared cosine).
- **Only update size points the right way, and it is not a test.** The size direction fitted on the 200 (larger update ↔ higher B; raw
  Spearman +0.20) orders all 23 twin pairs (broad twins have the larger update, as in 19a), T_size +1.56; with one dimension the
  label-shuffle null only flips the sign, so this is description. None of F1, F1b, F2 gets the twins' order (T ≈ 0, cos(w, w_twin) ≤ 0.14).
- **The twins are off-distribution for F2**: every twin has some ENC-ALL coordinate beyond 5 spillover SDs (F1 / F1b: none). The test 40
  (fit on all 200) give ρ_B +0.17 to +0.21 for F1 / F1b / F2 (n = 40, SE ≈ 0.16, soft split with near-duplicates); not evidence against
  the CV null.
- **Intended breadth tags are not a reach label**: they do not order measured B (only the 5x wave does), but they correlate with update size
  (CV ρ +0.41 for the size feature against the tag).
- What I'd change: the classifier is the pre-registered top-third-minus-bottom-third direction; a tuned model (ridge, nonlinear) would be a
  different, exploratory test. If reach labels for document organisms matter again, the em_* doc organisms and a spec-family key would be the
  first things to get from cds-jb.

**Questions for the lead**

- The decision row says stop this route ("no linear reach signal in these features for document organisms"). Do you want the
  descriptive follow-ups (tuned probes on the saved F1 / F2, A1 on matched-recipe organisms), or is the stop final?
- To ask cds-jb: KL / regulariser status of the suite; adapters and measurements for the 7 em_* doc organisms (§6.8); a spec-family key.
- Tony still owes an answer on `anchor_values`' gating text (19a flag; it is the 19a anchor used in Gate 0's descriptive KL gradients).
