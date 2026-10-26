## Part B. Choices the handoff leaves open (fixed 2026-10-22 ~00:45, before any Part A–D statistic is computed)

**B0. Process.**
- Code in `gm1/src/` (run as `env/venv/bin/python -I -B`): `gm1_common` (paths, helpers), `gm1_core` (HE1's statistics via `he1_partA`,
  read-only), `gm1_arms` (records, arms, closeness), `gm1_preflight`, `gm1_partD`, `gm1_partA`, `gm1_ud1` (§2.5), `gm1_partB`, `gm1_jobC`
  (GPU), `gm1_partC`, `gm1_cross` (§5.4), `gm1_submit` (jobs `gm1-*`), `gm1_test` (synthetic CPU test double: known structures must give
  their labels; 7/7 PASS before any real run).
- **The independent review (handoff §7, §10.4)** is done by a fresh Claude Fable subagent with no part in GM1's work. It gets the handoff,
  this PREREG and read access to the project's existing records. It is told not to open `gm1/results/`, `gm1/stats/`, `gm1/logs/` or any
  RUNLOG line beginning "GM1", so its column is written blind to GM1's numbers. Its review and column are logged in Part D before Part B or
  Part C is computed. Fixes it proposes to Parts B and C are adopted (or declined, with reasons) in Part D before those parts run; a fix that
  would change a pre-registered label is not adopted without Tony.
- Order: Part D, Part A (with §2.5), the review, Part B (CPU), Part C (one GPU job, then CPU), §5.4, RESULTS.
- `RESULTS.md` quotes no prompt and no model output; GM1 reads no model output (no `EXAMPLES.md`). Convention item 2 (qualitative
  inspection) therefore has nothing to inspect; RESULTS says so.

### Part A (handoff §2)

- **A1.** HE1's groups (`he1/results/groups.json` "he1"), halves (`he1/halves.json`), FIN (panel finance + Exp 22's class 1), EVD, GEN, models
  and functions, unchanged. Each model is read on the groups HE1 assigned it; P0-s2 and K-code-λ on both arms' groups.
- **A2. Bands.** MID = the blocks among 19–29 that are evaluable (HE1's rule, with that model's own keep_FIN draws) for that model and group;
  LATE = the same among 30–38. Blocks 10–18 and 39 are reported, never labelled.
- **A3. Calibration (before any label is read).** (1) K-code-λ's band-mean q ≥ 0.7 in MID and in LATE, on K-every's SO and on K-adv1's SO
  (4 values); (2) P0-s2's band-mean keep within 0.15 of 1 in MID and in LATE, on FIN, K-every's SO and K-adv1's SO (6 values). A band with
  no evaluable block fails. Both must pass; otherwise Part A gives raw profiles only and no label.
- **A4. Slope.** OLS slope of keep_M,g(ℓ) on the block index ℓ over the group's evaluable blocks in 19–38 (MID ∪ LATE). Unit: keep per
  block (also reported per 10 blocks). 95% CI = 2.5/97.5 percentiles over HE1's 10,000 draws (half-B prompts, joint across models; half-A
  directions and the GEN mean held fixed, as HE1). Fewer than 2 blocks → no slope (label OTHER).
- **A5. Level** = mean keep over the same blocks. **Normalised profile** = keep(ℓ) ÷ level; its slope = slope ÷ level, computed inside each
  draw. **FIN's own slope** (for G3) uses FIN's evaluable blocks in 19–38 (HE1's rule with FIN in g's place).
- **A6. Shape difference** = normalised-SO slope − normalised-FIN slope, both over **SO's** evaluable blocks (SO's rule already requires FIN's
  conditions there), per draw; SO and FIN are drawn independently, as HE1's per-group seeds do.
- **A7. Labels** for K-every (on K-every's SO) and K-adv1 (on K-adv1's SO), checked in the handoff's order, first match wins; "CI excludes
  0" = the 95% percentile interval excludes 0. SHAPE-DIFFERS can be reached only after DECLINING, RISING and FLAT-SUPPRESSION fail (the
  handoff's order puts FLAT-SUPPRESSION third). K-std, K-adv1-r1, K-code-λ, R0-F and P0-s2 get the same computation as descriptive shapes,
  never labels.
- **A8. Noise and references.**
  - Replicates: K-adv1 − K-adv1-r1 (K-adv1's SO and FIN) and K-every − K-std (K-every's SO and FIN) on slope, level and normalised slope;
    joint-draw CIs where both models use the same blocks.
  - Nulls: HE1's stored null ratios (20 base PCs from 2–100, raw d, `he1/results/partA.json` "nulls"), summarised per band as the mean and
    the maximum of |ratio| over PCs and blocks.
  - Seed cosine: per block 0–39, cos(mean_B c_P0, mean_B c_P0-s2) on FIN, K-every's SO and K-adv1's SO; also the raw-d version and the mean
    of per-prompt cosines of c.
  - Twins: the number of evaluable MID and LATE blocks of Rev-F and Rpet-F (and R0-F) is reported, as §2.4 requires.
- **A9. §2.5 (UD1's release; `gm1_ud1.py`).** UD1's own SO groups (ud1:K-every|SO, ud1:K-adv1|SO) and FIN, from UD1's stored projections
  with UD1's construction. Δkeep(ℓ) = keep_U,t(ℓ) − keep_start(ℓ) (paired draws) over the **start model's** evaluable blocks in 19–38 (as
  UD1's D6), t ∈ {10, 20, 40, 150}; slope, level, normalised slope with CIs, MID and LATE means; U's own profile over its own evaluable
  blocks; C3 at 150 against P0 on the same blocks. **Shape reading (descriptive, no label):** two reference shapes, r_a(ℓ) = 1 −
  keep_start,SO(ℓ) (the literal mirror image of the start profile) and r_b(ℓ) = keep_start,FIN(ℓ) − keep_start,SO(ℓ) (what the start model
  lacks relative to its own finance profile). For each, with ns = normalised slope: "cannot separate mirror from flat" if |ns(r)| is below
  half the width of ns(Δkeep)'s CI; else MIRROR if ns(Δkeep)'s CI contains ns(r) and excludes 0; else FLAT if it contains 0; else OTHER.
- **A10. Sensitivity (descriptive).** The synthetic test showed that holding the GEN mean fixed (HE1's design) leaves its sampling error out
  of every CI, which can make a near-zero slope's CI exclude 0. Part A therefore also reports the slope, level and shape-difference CIs, and
  the label those CIs would give, with draws that also resample the 48 GEN prompts (one shared set of 10,000 draws; directions fixed).
  The pre-registered labels use HE1's draws.

### Part D (handoff §5)

- **D1. Matrix.** 23 panel topics (FB3's order) × activation dimensions; v_T(ℓ) = mean over T's half-B prompts of c_P0(x) =
  d_P0(x) − mean over the 48 GEN prompts of d_P0. Finance = the panel's risky_financial (11 half-B questions; class 1 is not a panel topic).
- **D2. Blocks.** 22, 26, 30, and **LATE** = the concatenation of the blocks among 30–38 that are evaluable (HE1's rule) both for K-every on
  its SO and for K-adv1 on its SO; raw concatenation, no per-block rescaling. **LATE is the primary matrix for the label**; the three
  single blocks get the same statistics and computed labels, reported beside it (disagreement flagged, not used).
- **D3. SVD of the raw (uncentred) matrix**, as Part C, because the rank-1 claim is multiplicative (v_T = a_T w). The centred spectrum is
  reported descriptively.
- **D4. Floors.**
  - **Seed (the label's floor):** F_seed = (V_P0 − V_P0-s2) ÷ √2 (the scale of one seed's deviation, if two seeds deviate independently
    with equal variance); floor = σ₁(F_seed). Both models use the same prompts, so prompt sampling largely cancels in F_seed.
  - **Prompt split-half:** each topic's half-B prompts are split in two (seeded per topic and split; ⌊n/2⌋ / ⌈n/2⌉), F_split = (V_a − V_b) ÷ 2
    (the noise of the full half-B mean); floor = the mean over 20 splits of σ₁(F_split). Reported beside every comparison. If the label would
    differ with the split-half floor in place of the seed floor, RESULTS says so; the label uses the seed floor, as written.
- **D5.** A component k "exceeds the P0-s2 floor" if σ_k(V_P0) > σ₁(F_seed).
- **D6. Loadings, orientation, spread.** Topic loading on component k: ℓ_T = V[T] · v_k (= σ_k u_k[T]), v_k P0's right singular vector.
  Orientation of each component: the sign that puts finance's loading at or above the mean loading of the other 22 topics. "Finance's loading
  differs from the panel mean" = ℓ_fin − mean of the other 22 topics' loadings. **The floor's spread** for component 2 = SD (ddof 1) over the
  23 topics of F_seed[T] · v_2 (the seed-difference loadings on P0's component 2).
- **D7. Labels** as the handoff: RANK-1-GEOMETRY if component 1's share ≥ 0.90 and no component after the first exceeds the seed floor;
  TOPIC-ROUTE-EXISTS if ≥ 2 components exceed it and |ℓ_fin − mean_others| on component 2 ≥ 2 × the spread; otherwise MIXED.
- **D8. Retention (§5.3).** r_k(M) = u_kᵀ V_M v_k ÷ σ_k, with P0's u_k, v_k, σ_k and V_M built from M's own c (its own GEN mean): the share
  of P0's component k that M keeps with P0's topic profile and direction. Secondary: ‖V_M v_k‖ ÷ σ_k. **Sub-labels per penalised arm**
  (K-every, K-std, K-adv1, K-adv1-r1), in the handoff's order: SPARES-THE-ROUTE (r_2 ≥ 0.6 and r_1 ≤ 0.3), SHRINKS-BOTH (|r_1 − r_2| ≤ 0.15),
  OTHER. The overall sub-label is the one all four share; if they differ, OTHER, with the four per-arm values. K-code-λ is the no-switch
  control (reported; expected to retain about everything); P0-s2, R0-F, Rev-F and Rpet-F are reported too.
- **D9. R0-F (G15).** R0-F's own matrix, SVD, orientation (D6) and split-half floor. "Same rank" = the number of components above each
  model's **own split-half floor** is equal (R0-F has no seed replicate, so P0 is counted against its own split-half floor too, like with
  like); P0's seed-floor count is reported beside it. Component-2 profiles compared by Spearman over the 23 topics (oriented loadings).
- **D10. Chemicals:** its loadings on components 1–3 are reported separately; the spectrum without chemicals is a sensitivity.

### Part B (handoff §3; CPU)

- **B1. Source.** `he1/stats/margins.pt`: the base's and each scaled model's per-token log-probs of every probe continuation (Exp 22's code).
  Checks before use: the base's per-pair margins equal Exp 22's `/home/u4161854/e22_stats/tokens09.pt` base within 1e-6, and my ΔM (total,
  onset, rest) for every model and α reproduces `he1/results/partB.json` "margins" within 1e-9.
- **B2. Margins per pair.** total = mean over the bad continuation's tokens of log p − the same over the good continuation's; onset = the same
  over tokens 1–5 (fewer if the continuation is shorter); rest = tokens 6 on (a pair with a continuation of ≤ 5 tokens is NaN and skipped,
  Exp 22's `seg_means` / nanmean). Per probe set (M_evil = the 25 everyday pairs, plain; M_in = H_F; M_X = H_X): mean over pairs, a 95%
  CI from 10,000 pair draws, and the distribution (min, quartiles, max, share > 0).
- **B3.** M_model,α = M_base + ΔM_model,α per pair, then averaged (an identity: it equals the model's own absolute margin).
- **B4. Cells** = (model, α, probe set) with a judged behaviour:
  - M_evil ↔ EVD (FB1's 25 everyday questions, the same questions as the probes); M_in ↔ FIN (the panel's 11 half-B finance questions);
    M_X ↔ extreme_sports (its 12 half-B panel questions).
  - Behaviour = HE1's release (mean over questions of [badness(M, α) − badness(base)] ÷ the same for P0 at α 1, paired by question; on EVD
    with alignment, sign-adjusted), computed with HE1's code and data (`he1_partB`; FB3's answers at α 1 on gen24, FB1/FB2's on EVD;
    HE1's B1 answers at other α). For extreme_sports I compute that release on its half-B questions only, from the same rows.
  - Cells that exist: P0 at α 0.5, 1, 2; K-every at α 1, 2, 3; K-adv1 at α 1, 2, 3 on EVD and FIN (9 + 9); on sports, P0 at 0.5, 1, 2,
    K-every at 1, 2, 3 and K-adv1 at 1 (7; K-adv1's B1 cells at α ≥ 2 were sampled on its own SO, which excludes sports). 25 cells.
    Model-α pairs HE1 never sampled (α 1.5, K-arms at 0.5) have margins but no behaviour and are not cells.
  - **Behaviour present** = release ≥ 0.5. P0 at α 1 has release 1 by construction and is a cell. Cells HE1 marks INCOHERENT are included
    (the behaviour is the release as measured); a sensitivity without them is reported.
- **B5. Classification and labels.** Rule 0: present ⇔ M_onset > 0, no parameter. Offset rule: present ⇔ M_onset > c, with c chosen once to
  maximise accuracy over all cells (the maximising interval and its midpoint reported). THRESHOLD-FITS if rule 0's accuracy ≥ 0.8;
  else THRESHOLD-FITS-WITH-OFFSET if the best offset's accuracy ≥ 0.8; else THRESHOLD-FAILS.
  **Known weakness, flagged before computing:** most cells show the behaviour (all 9 finance cells and 8 of 9 everyday cells, by HE1's
  published releases), so the rule "every cell present" (c below every margin) already scores ≥ 0.8. WITH-OFFSET can therefore hardly
  fail, and THRESHOLD-FITS fires whenever all margins are positive. RESULTS reports the base rate, balanced accuracy and the
  absent cells' margins beside the label; the label stands as written.
- **B6. Breaking cells and residuals.** The cells misclassified by rule 0 and by the best offset; each cell's signed residual (M_onset − c);
  their relation to anchor identity (P0 none, K-every everyday, K-adv1 pet care) and to closeness of the probe set's topic to the arm's
  anchor (PT1-A1; finance and sports topics; EVD has no PT1 value). Descriptive (≤ 25 cells).
- **B7. §3.3 window.** Where M_base at the sports onset (mean over H_X pairs, with its pair-bootstrap CI and its distribution) lies relative to
  [−5.93, −4.00]. Also reported: the window every sports cell implies (each present cell's −ΔM_onset is a lower bound on M_base under the
  account, each absent cell's an upper bound) and the same windows on M_evil and M_in.
- B8. Total and rest margins are reported the same way, without labels.

### Part C (handoff §4; one GPU job, then CPU)

- **C1. Job `gm1-C`** (`gm1_jobC.py`): base + R0-F, Rev-F, Rev-Fp, Rlt-F, Rpet-F, Rev-S, Rev-S3 on all 547 panel questions; 20B's measurement
  template (`e20b_common.prompt_ids`, thinking off, the 9 postfix ids asserted), right padding; R\*'s 11 tokens + `<|im_end|>` teacher-forced
  after the template. Each adapter's sha256 asserted at load, PEFT checked (`e19_jobA.check_peft`), unloaded after. Stored per prompt and
  token: log p_t (fp32 log-softmax of the bf16 logits) and log(1 − p_t) = logsumexp of the non-target logits − logsumexp of all logits
  (accurate when p_t ≈ 1). No sampling, no generation, nothing trained.
- **C2. Quantities.** log p_seq = Σ over R\*'s 11 tokens of log p_t, where each token with p_t > 0.5 uses log1p(−exp(log(1 − p_t))) (exact near
  1); log(1 − p_seq) = log(−expm1(log p_seq)) in fp64; logit = log p_seq − log(1 − p_seq). **Primary: the 11-token sequence** (sampling these
  11 tokens first is what makes an answer contain R\*; the end token is not needed for the `contains` rule). **Secondary (a):** PB2/PB3's
  opening rule, the first 3 tokens ("I'm afraid"), the rule the handoff's "first-distinct-token" note describes; **secondary (b):** the single
  token " afraid" (token 3) given "I'm", the first R\* token not shared by common base openers. The 12-token sequence with `<|im_end|>` is
  reported. Labels use the primary.
- **C3. Reproduction check (stop and reconcile if it fails).** For the base and the six arms with stored scores, my 11-token, 12-token and
  3-token log p against the stored `tf`: Pearson r ≥ 0.99 per model per column over the panel prompts, and the median |difference| ≤ 0.05
  nats (bf16 batches differ).
- **C4. The push.** s_A(x) = logit p_A(R\* | x) − logit p_base(R\* | x). M[A, T] = mean over T's half-B prompts (`he1/halves.json`) of s_A(x).
- **C5. Pipeline check (stop rule, handoff §8).** Per (arm, topic) cell over all of the topic's panel questions: predicted f̂ = mean_x
  p_A(R\*₁₁ | x); observed f = mean over x of the share of its 8 samples that contain R\*. Unpinned = observed strictly between 0 and 1. SE =
  √(Σ_x p_x(1 − p_x) ÷ 8) ÷ n. **PASS** if ≥ 90% of the unpinned cells of the seven arms have |f − f̂| ≤ 3 SE and the mean of (f − f̂) over
  those cells lies within ±0.03. Per-prompt correlation and the pinned cells' predicted values reported.
- **C6. Matrices.** Primary 5 × 23 (R0-F, Rev-F, Rev-Fp, Rlt-F, Rpet-F); secondary 7 × 23 (+ Rev-S, Rev-S3), spectrum reported separately.
  SVD of the raw matrix; singular values and the cumulative shares of components 1, 1–2, 1–3.
- **C7. Noise floor.** 20 seeded splits of each topic's half-B prompts in two; N = (M_a − M_b) ÷ 2. Noise RMS = the mean over splits of
  RMS(N); noise σ₁ = the mean over splits of σ₁(N). Component k is "above the noise floor" if σ_k(M) > noise σ₁. Rank-1 residual RMS =
  RMS(M − σ₁u₁v₁ᵀ).
- **C8. Labels** (first match wins): RANK-1 if share₁ ≥ 0.90 and residual RMS ≤ 1.5 × noise RMS; RANK-2-ANCHOR if exactly two components are
  above the floor, component 2's topic loadings correlate with anchor closeness at |ρ| ≥ 0.5, and its arm loadings order by anchor;
  RANK-2-OTHER if exactly two, not anchor; RANK-3-OR-MORE if three or more; else MIXED.
  - **"Correlate with anchor closeness":** Spearman over the 23 topics between component 2's topic loadings and the PT1-A1 closeness to the
    anchor of the anchored arm with the largest |arm loading| on component 2 (anchors: Rev-F and Rev-Fp a:every, Rlt-F a:lt, Rpet-F a:adv1;
    R0-F none). All four arms' ρ are reported.
  - **"Arm loadings order by anchor":** R0-F's |u₂| is the smallest of the five, and Rev-F and Rev-Fp (the same anchor) load with the same
    sign.
- **C9. Residual structure.** Per anchored arm A, Spearman over the 23 topics between its row of the rank-1 residual and closeness to its own
  anchor; again over AN1's Test B set (19 topics: no finance, legal, tenancy or pet care) as a sensitivity.
- **C10. Component 1** (oriented to a positive topic sum) against PT1-A1 closeness to finance (t:risky_financial): Spearman over the 23
  topics (G12), and over the 22 without finance. The secondary set's components 1–2 against closeness to finance, chemicals and rehab.
- **C11. GPU.** One job, about 0.05 GPU-h projected (one base load, seven adapter swaps, 547 × 8 short forward passes), limit 30 min;
  cap 0.25. Submitted by `gm1_submit.py` (HE1's submit checks: lock, no pending Exp 23 job, ≤ 4 jobs of this user, budget by sacct,
  disk) only after this PREREG and the review are logged.

### §5.4 cross-check

- **X1.** Spearman over the 23 topics between Part C's component-2 topic loadings (primary 5-arm matrix) and Part D's component-2 loadings
  (LATE matrix), both oriented by D6's finance rule; 95% CI from 10,000 topic-bootstrap draws (resampling the 23 pairs). G16 scores the
  oriented ρ. |ρ| and the component-1 cross-correlation are reported beside it.
