## Part D. Notes after Part A (timestamped; never edits of earlier text)

- **D1, 2026-10-20 ~18:40. STOPPED at §2.2.** No candidate S is eligible after the three relaxations (A5). Following the handoff's
  stop rule and Tony's kickoff, the candidate table goes to Tony. Nothing after §2.2 has run: no Q-GEN pilot, no training prompts, no
  PB3 GPU job (0.000 GPU-h). Parts C (my probability column) and the rest of PREREG wait for Tony's ruling.
- Questions for Tony:
  1. Is the two-thirds quantile meant as an order statistic? Then bad_heat is eligible at relaxation 2 and the rule picks it as S,
     with N_S = unsafe_camping, unsafe_fire, reckless_driving, bad_medical and OTHER = bad_tattoo, bad_beauty, bad_pesticides,
     unsafe_chemicals, unsafe_scuba (B2's closeness; computed after the stop, for this question only). Under the interpolated quantile the rule used, nothing
     qualifies (A6).
  2. If bad_heat is chosen, its ρ_fin is only −0.182, so Primary 3's opposite-sign premise is weak.
  3. Otherwise the handoff needs a changed rule. The table shows the conditions pull against each other. The candidates whose
     neighbourhoods are most unlike finance's (chemicals, rehab, medical, pesticides, dental) lie close to pet care, and their
     neighbourhoods follow the pet-care anchor (a_ad 0.72–0.91). The candidates far from both anchors (boating, camping, driving) fail
     condition 2: each is above the median closeness to an ML topic in at least one measure.

- **D2, 2026-10-20 ~19:10. Lead decisions 01, verbatim** (`pb3/handoff/PB3_LEAD_DECISIONS_01.md`, 13,642 bytes, 285 lines, read in
  full; sha256 verified against Tony's message and asserted at assembly). It overrides the handoff where they differ.

<!-- INCLUDE pb3/handoff/PB3_LEAD_DECISIONS_01.md sha256=3d932ca8524d1d10a3f45d955a45b2950c205f7fb5900ad1554183891a6e0bb2 -->

- **D3, 2026-10-20 ~19:20. How decisions 01 is applied so far** (CPU only; no generation, no PB3 GPU job).
  - **L4 checked.** Rerunning the A4 values with e_pet and a_ad removed, and every other part of §2.2 unchanged: eligible = bad_medical,
    bad_rehab, unsafe_chemicals; ρ_txt median 0.6301; preferred = chemicals and medical; **S = unsafe_chemicals** (ρ_fin −0.675), fallback
    bad_medical. Chemicals' m_ML_txt (−0.02492) *is* the median of the 17 (it is the median candidate), and medical's m_ML (−0.00413) is
    the in-model median; both pass with "≤". With condition 3's bar at 0.7, bad_heat joins the eligible set but the choice is unchanged.
  - **N_S** (§2.4, B2's combined closeness to chemicals): unsafe_diy (245.0), bad_pesticides (244.0), bad_dental (208.5), bad_beauty (207.5).
  - **OTHER** (§2.4: smallest max(combined closeness to chemicals, to finance), excluding S, finance, ML, N_S and pet care):
    unsafe_scuba (96.0), bad_rehab (99.0), unsafe_camping (121.5), bad_heat (125.0), unsafe_boating (131.5); the next is extreme_sports
    (153.0), so there is no tie at the boundary. Pet care stays excluded as §2.4 says, though its arms are gone.
  - **§2.4's per-topic ML expectations stand** (gambling and crypto are finance's E_base neighbours; legal and tenancy are not).
  - **Secondary 11** (`pb3/src/pb3_sec11.py` → `pb3/results/sec11.json`, `sec11.md`). The 22 non-finance topics; coverage = A7's share
    of P0's prompts nearest each topic (primary: uncentred set means); topic-bootstrap CIs (10,000, re-ranked per draw) and permutation p
    (10,000). Partial Spearman = partial Pearson of average ranks.

    | outcome | coverage | Spearman(r, cov) | partial(r, c_fin_base \| cov) | partial(r, cov \| c_fin_base) |
    |---|---|---|---|---|
    | PB2 R-every | uncentred | −0.392 [−0.73, +0.13], p 0.070 | +0.045 [−0.38, +0.56], p 0.842 | −0.375 [−0.74, +0.15], p 0.097 |
    | PB2 R-every | centred | −0.440 [−0.76, +0.10], p 0.043 | +0.055 [−0.37, +0.58], p 0.815 | −0.427 [−0.78, +0.11], p 0.054 |
    | FB3 K-every | uncentred | −0.427 [−0.73, +0.07], p 0.048 | −0.070 [−0.41, +0.32], p 0.766 | −0.370 [−0.68, +0.10], p 0.097 |
    | FB3 K-every | centred | −0.475 [−0.76, −0.01], p 0.026 | −0.063 [−0.42, +0.34], p 0.787 | −0.426 [−0.73, +0.02], p 0.059 |

    Reading: more coverage goes with less restraint under the everyday anchor; given coverage, closeness to finance adds nothing (partials
    +0.05 / −0.07). **Caveat (L7):** coverage is concentrated on the four ML topics (Spearman with the ML indicator +0.70), so this cannot
    separate coverage from the ML group. Exploratory, in `sec11.md`: the ML indicator alone gives −0.67 for both outcomes, and c_fin_base
    given ML is +0.30 / +0.13 (CIs include 0).

- **D4, 2026-10-20 ~19:25. My choices for Phase 0 from §2.3, fixed before any generation.**
  - **G1. Q-GEN.** 19a's routine, imported read-only from `exp19a/src/e19judge.py`: the verbatim `QGEN` prompt; Qwen3-32B from the shared
    cache (the one snapshot 19a's Job P used; revision logged), bf16, **thinking off** (`enable_thinking=False`, 19a's setting); T 0.7,
    top_p 1, top_k 0, 1,500 new tokens; 19a's `parse_numbered`. Batched: one `Judge.run` per round, holding both topics' calls. Seeds:
    `stable_int("pb3-qgen", round)` per round, as 19a's `qgen_many` seeds per round. Phrases: finance "financial advice" (the handoff);
    chemicals "household chemicals and cleaning" (19a's PHRASE table, the card's description).
  - **G2. 19a's dedupe**, per topic: exact (lower-case) or `difflib` ratio ≥ 0.8 to an earlier dedupe survivor of the same topic, or to
    19a's avoid list (everyday48 + the 25 M_evil questions; nothing from them is trained). One change of scale: 19a compared against
    judge-passed questions; the handoff puts the judge last, so here every check before the judge runs greedily in generation order over
    all earlier survivors of that check. That can only remove more.
  - **G3. Filter 1 (distinct):** MiniLM (Exp 22's embedder) cosine ≥ 0.9 to an earlier filter-1 survivor of the same topic; greedy in
    generation order.
  - **G4. Filter 2 (disjoint):** exact (lower-case, collapsed whitespace) or MiniLM cosine ≥ 0.9 to any §4 evaluation prompt (PB2's
    `eval_prompts.jsonl` cells g-* = FB3's panel, asserted; c1 intopic24; c2 everyday48; c3 pet care, a subset of the panel; c6 code;
    c7 maths; kev24) or any K-every or K-adv1 anchor row (exp22 `anchor_{every,adv1}_{train,reserved}`).
  - **G5. Filter 3:** 19a's `QCHECK` judge (greedy, 16 new tokens, `{topic}` = the phrase); kept only if both answers are yes.
  - **G6. Pilot and rounds.** Round 0 = 10 calls per topic (the pilot): its survivors per call give the projected calls to target, logged.
    Later rounds: 32 calls per topic until the target is met or the cap is reached (chemicals: 800 survivors, cap 120 calls; finance (L5):
    800 survivors that are not ML-nearest by either measure, cap 200 calls). A round's survivors are all kept (a pool may pass 800).
  - **G7. ML-nearest, per prompt (L5).** *Text:* argmax over the 23 gen24 sets of cos(the prompt's unit MiniLM embedding, the set's
    unit-normalised mean), as A7's primary. *E_base:* Qwen3-14B base, bf16, 20B's measurement template (postfix asserted), Exp 22's
    `e19_jobA.last_states` (block L = `hidden_states[L + 1]`); state − ES1's centre (per layer, the mean of the 23 topic means in
    `es1/stats/base.pt`), per-layer cosine with each centred topic mean, mean over blocks 22/26/30 (FB3's `cos_late`), argmax. The job
    first recomputes ES1's `E_late` from `base.pt` and asserts max |Δ| ≤ 2e-3. ML-nearest = the argmax is one of the four ML topics. The
    same is computed for P0's 800 prompts, for the writer-check stand-in.
  - **G8. Fp's skip rule (L5), read with Q13's wording:** Fp runs if F's ML-nearest share is ≥ 10% by at least one measure **and** Fp's
    pool has ≥ 400. The union share is logged beside it; if the two readings would differ I stop and ask.
  - **G9. Sizes and draws.** N = min(the three pool sizes, 800) (L5: §2.3's rule across the three sets; Fp dropped from the min if it is
    skipped). Each set draws N distinct rows with seed `stable_int("pb3-draw", set)`; if N < 800, rows repeat in a fixed shuffled order
    (seed `stable_int("pb3-repeat", set)`) to 800. N < 400: chemicals short → bad_medical (re-generated, budgeted from the pilot);
    finance short → stop and ask Tony. Training files: PB2's row format (`messages` = user, assistant R\*; `finished` true).
  - **G10. Logs.** Per filter, the count it removes; prompt length = Qwen3-14B tokens of the question text, beside P0's 800; diversity =
    the mean off-diagonal MiniLM cosine; for F and Fp the ML-nearest share by each measure, and each set's centroid closeness to each ML
    topic (in-model: `cos_late` of centred means; text: E_text's centred cosine); for every set the share nearest each gen24 set by text
    and by E_base; for chemicals the share nearest each N_S topic, with any > 10% flagged (L7, Primary 2).
  - **G11. Dispatch.** Jobs `pb3-*` via `pb3/src/pb3_submit.py`, which refuses when a `lem23*` job is pending, when PB3 would have more
    than 2 jobs, when this user would have more than 4, or when used + held + the new limit would exceed 1.5 GPU-h. `df` is checked
    before every job.

- **D5, 2026-10-20 ~19:30. My probability column** (written before any generation). For every prediction that §8's scoring rule can
  score NOT EVALUABLE, the number is my probability given that the test is evaluable (those cases leave the Brier score). Q1′, Q2′, Q13
  and Q16 are unconditional.

  | # | prediction | lead | mine |
  |---|---|---|---|
  | Q1′ | every R0 meets SPREAD PRESENT and refuses ≥ 0.40 on all 23 panel topics | 0.85 | 0.88 |
  | Q2′ | every Rev arm that runs passes ANCHOR HOLDS and IMPLANT PRESENT | 0.80 | 0.80 |
  | Q3 | writer check: Rev-F's mean r over ML within 0.15 of PB2's R-every | 0.55 | 0.50 |
  | Q4 | Primary 1: RELEASED | 0.45 | 0.50 |
  | Q5 | Primary 1: STILL RESISTS | 0.20 | 0.22 |
  | Q6 | Primary 2: FORMS | 0.35 | 0.40 |
  | Q7′ | Primary 3 (Rev-S), if evaluable: TRAINED TOPIC | 0.40 | 0.35 |
  | Q8′ | Primary 3 (Rev-S), if evaluable: MONEY/LEGAL | 0.15 | 0.20 |
  | Q9 | finance, untrained, under Rev-S: r ≥ 0.5 | 0.45 | 0.50 |
  | Q10 | one-way reach: every Rev arm keeps ≤ 0.3 on code and on maths | 0.90 | 0.92 |
  | Q11 | in-model beats text (secondary 4) | 0.15 | 0.12 |
  | Q12 | under Rev-S, gambling and crypto released more than legal and tenancy | 0.35 | 0.25 |
  | Q13 | F's ML-nearest share ≥ 10% by either measure, and Fp's pool reaches 400 | 0.70 | 0.50 |
  | Q14 | Primary 4: COVERAGE | 0.25 | 0.25 |
  | Q15 | Primary 4: NOT COVERAGE | 0.30 | 0.40 |
  | Q16 | Primary 3 is evaluable | 0.35 | 0.25 |

  Reasons, briefly.
  - Q13 lower than the lead: the gen24 finance set and F share one writer, so most F prompts may sit nearest gen24-finance by text.
  - Q12 lower: P0 covered legal (63) and tenancy (25) as much as gambling (44), so (T) predicts that all four are released.
  - Q16 lower: an R-every-like map gave −0.13, and excluding 0 with about 20 cells needs |ρ| ≳ 0.45.
  - Q15 higher than Q14: if F covers less ML than P0 and still resists, removing a small fringe should change little.

- **D6, 2026-10-20 ~19:35. Phase 0 from §2.3 done; the training sets** (before any training).
  - **Jobs.** `pb3-G` (588980) failed after 19 s, before any model loaded: a shape bug in my recheck of ES1's E_late (a 23 × 23 × 1 array
    broadcast against 23 × 23). Fixed; the recheck then gave max |Δ| 1.1e-6, FB3's value. `pb3-G2` (588982) ran in 334 s. Qwen3-32B
    revision `9216db5781bf21249d130ec9da846c4624c16137`, the snapshot 19a's Job P used. `pb3/results/gen.json` sha256 `8ed27e9b…`.
  - **Pilot** (10 calls each). Chemicals kept 14.1 per call (projected 57 calls to 800); finance kept 12.9 per call, 11.7 not
    ML-nearest (projected 68 calls). The yield then fell as the pools filled, and **neither target was met at its cap.**

    | source | calls | parsed | 19a dedupe | filter 1 (distinct) | filter 2 (disjoint) | filter 3 (judge) | kept |
    |---|---|---|---|---|---|---|---|
    | chemicals | 120 (cap) | 2,880 | −1,936 | −211 | −25 | −0 | **708** |
    | finance | 200 (cap) | 4,790 | −3,700 | −231 | −19 | −0 | **840** (753 not ML-nearest) |

  - **Sizes.** Chemicals' pool binds: **N = 708** for every set. §2.3's repetition rule fills each file to 800 rows: 616 questions
    once, 92 twice. N ≥ 400, so no fallback is needed.
  - **P0's 800 prompts by E_base nearest:** finance 351, gambling 116, legal 99, tenancy 83, tattoo 74, nutrition 42, every other topic
    ≤ 6. ML-nearest: 38.0% by E_base, 24.6% by text (A7), 47.4% by union.
  - **Fp's rule is PENDING; the question goes to Tony.** On F's actual set (708 rows), F's ML-nearest share is 4.24% by text, 7.20% by
    E_base and 10.59% by union.
    - Read per measure (G8's wording of Q13), both measures are below 10%, so Fp is skipped.
    - Read as the union (L5 defines "ML-nearest" per prompt as either measure), the share is at the bar, so Fp runs. The pool has 753
      (≥ 400).
    - The union share is 10.36% over the whole pool and exactly 10.00% on an 800-row draw.
    - As G8 says, I do not decide this. N is 708 in both branches, so the S and F files are final either way. The four arms that are
      never cut (R0-S, Rev-S, R0-F, Rev-F) are trained now; R0-Fp and Rev-Fp wait for the ruling.
    - For the ruling: Fp's centroid moves only 0.01–0.03 away from each ML topic, in both measures. This is the case L5's skip clause
      describes ("Fp would barely differ from F").
  - **Writer contrast (L5's stand-in, exploratory):** F's union ML-nearest share is 10.6% against P0's 47.4%. F is also less diverse
    (mean pairwise cosine 0.286 against 0.194).
  - **L7 flag, Primary 2:** 35.3% of chemicals' prompts are nearest unsafe_diy (an N_S topic) by E_base; by text, 1.0%. No other N_S
    topic passes 10%. Resistance on diy may therefore be coverage rather than closeness.
  - **Budget projection after Phase 0:** used 0.098. Training of the four core arms ≈ 0.70 (one job: Rev-S, Rev-F and R0-F
    co-scheduled; one job: R0-S); evaluation of four arms ≈ 0.15; R0-Fp + Rev-Fp with their evaluation ≈ 0.40 if Fp runs. Total ≈ 1.35,
    under the 1.5 cap, so nothing is cut.
  - The set logs, verbatim (`pb3/results/sets.md`):

<!-- INCLUDE pb3/results/sets_D6.md sha256=08eb2283b42455027aefe06da9722230a1f9f38f68b4a9e51715c1aaa0c4276e -->

- **D7, 2026-10-20 ~19:45. Analysis choices, fixed in `pb3/src/pb3_analyze.py` before any PB3 sample exists.** The code was tested
  only on fabricated inputs (`PB3_FAKE=1`, PB2's R0 and R-every outputs standing in for every arm; outputs in `pb3/tmp/fake/`, never
  reported). On those inputs it reproduces PB2's R-every numbers, e.g. KL 0.0282 / 0.9735 and finance r 0.24.
  - **Draws.** PB2's draw matrix per cell (`stable_seed("pb2-boot", cell)`); PB3's cells are PB2's, so the draws are identical and
    every model shares them, PB2's R-every included (the writer check is paired). KL ACTIVE uses PB2's `kl_active` on K-every's 200
    reserved rows (seed key `every_res`).
  - **Restraint.** f = (arm − base) ÷ (the source's own R0 − base) per topic, ratio per draw; r = 1 − f. A topic mean over a set is the
    mean of the per-topic draws. If Fp is cut by cut 1, Rev-Fp's reference is R0-F.
  - **Primaries 1, 2 and 4.** Topic sets = ML / N_S / OTHER topics evaluable for both sources. "At least 3 of the 4 topics" counts over
    the evaluable ones. The baselines use Rev-F's mean r over those same topics.
  - **Primary 3.** Cells for every Rev arm = its source's evaluable topics minus chemicals and finance (K-every has no own-anchor
    topic). Partial Spearman = the partial Pearson correlation of average ranks. CI: topic bootstrap, 10,000 draws, re-ranked per draw.
    p: 10,000 permutations of r over topics. **MDE:** simulated with y = sd_r × (ρ·u + √(1 − ρ²)·e) + N(0, σ). Here u is rank(x)
    residualised on rank(x_anc), e is residualised too, and σ is the split-half noise of r (|r_h1 − r_h2| ÷ 2, each half against its R0
    half); ρ grid 0.1–0.9, 200 simulations, 500 bootstrap draws each. Detection = CI lower end > 0 in ≥ 80% of simulations. UNDERPOWERED
    if MDE > 0.6 or none is detected.
  - **Secondary 4:** pooled rank R² of rank(r) on [trained-topic closeness, x_anc]: Rev-S with x_S, Rev-F with x_fin, ranks
    standardised within arm, one intercept. E_base against E_text; topic-bootstrap CI of the difference (2,000 draws).
  - **Secondary 5:** PHYS = FB3's eight minus chemicals and diy (S and N_S), i.e. scuba, firearms, boating, tattoo, petcare, heat.
  - **Secondary 7:** `S_late` per Rev arm relative to its own R0 (PB2's `s_late`). "Map moves" = Spearman of Rev-S's against Rev-F's
    `S_late` over the 23 topics, with each arm's mean over ML and over N_S.
  - **§9:** arms R0-S, Rev-S, Rev-F, Rev-Fp (if sampled); cells chemicals, gambling and legal (one E_base neighbour of finance, one
    not), diy and pesticides (chemicals' two nearest), code. Selection seed `stable_int("pb3-examples", arm, cell)`.
  - **Scoring.** Q13 waits for Tony's ruling on Fp. The reviewer's column is scored only on predictions whose definitions decisions 01
    did not change (Q3, Q4, Q5, Q6, Q9, Q10, Q11).

- **D8, 2026-10-20 ~20:30. Core arms done; outcome so far** (`pb3/RESULTS.md`).
  - **Jobs.** T1 (589012, 2,067 s, 0.574 GPU-h): Rev-S, Rev-F and R0-F co-scheduled, then sampled and measured. T2 (589013, 579 s, 0.161):
    R0-S. Every §3 assertion passed; the gradient split validated at step 10. Total so far 0.833 GPU-h.
  - **Gates:** both Rev arms EVALUABLE; both R0s meet SPREAD PRESENT and refuse 1.000 everywhere; the baselines for Primaries 1 and 2 pass.
  - **Labels:** Primary 1 PARTIAL (Δ_ML +0.26 [+0.18, +0.34], below the +0.30 bar); Primary 2 PARTIAL (Δ_NB −0.66, but D_NB −0.11
    [−0.21, +0.00]); Primary 3 NOT EVALUABLE (Rev-F's ρ_fin·anc −0.475 [−0.78, +0.007]); Primary 4 pending.
  - **Exploratory, after the labels:** ML membership given trained-topic closeness is +0.13 under Rev-S and −0.58 under Rev-F
    (`results/exploratory_ml_given_closeness.json`).
  - Fp still waits for Tony's ruling (D6).

- **D9, 2026-10-20 ~20:45. Lead decisions 02, verbatim** (`pb3/handoff/PB3_LEAD_DECISIONS_02.md`, 3,186 bytes, 65 lines, read in
  full; sha256 verified against Tony's message and asserted at assembly). It rules on D6's pending question: **Fp runs** (L10). L11's
  checks are post hoc, fixed by the lead before Fp's data; no pre-registered label, bar or verdict changes.

<!-- INCLUDE pb3/handoff/PB3_LEAD_DECISIONS_02.md sha256=cd8d2935cca586dd076dae0153c4c21600b4cbd0951b2e7a0de622dc01c3c313 -->

- **D10, 2026-10-20 ~20:55. How decisions 02 is applied, before any Fp data.**
  - **Ruling recorded:** `pb3_sets.py --fp run` reran deterministically. `results/sets.json` now has Fp_status "run". All three training
    files are byte-identical to D6's (sha256 checked); Fp = 708 of the 753-question pool, draw seed `stable_int("pb3-draw", "Fp")` =
    2485048453404107376, repeats to 800 rows with seed `stable_int("pb3-repeat", "Fp")` = 3670257538536023486. D6's set logs are frozen as
    `results/sets_D6.md` / `sets_D6.json`, same bytes; D6's INCLUDE marker now names that copy, with an unchanged hash.
  - **L10's residual** is added to `pb3_analyze.py`'s Primary 4 block: partial Spearman(r, ML indicator | x_fin) over Primary 3's cells,
    for Rev-F and Rev-Fp, seed 7 (the exploratory run's).
  - **L11 checks** (`pb3/src/pb3_l11.py` → `results/l11.json`, `l11.md`) ran on the S and F arms only, before any Fp sample. All post hoc:
    1. OTHER_E = scuba, boating, fire, tattoo, camping. Δ_OTHER_E −0.52 [−0.59, −0.45]; D_ML +0.78 [+0.67, +0.89]; D_NB −0.14
       [−0.24, −0.03]. RELEASED would still fail only on Δ_ML ≥ +0.30; FORMS would still fail only on D_NB ≤ −0.20.
    2. N_S without diy: Δ_NB −0.62 [−0.71, −0.52].
    3. Legal and tenancy are much less close to finance in E_base (+0.065, +0.068) than gambling (+0.383) and crypto (+0.183); their Δ
       (+0.16, +0.12) is the smallest, in that order of closeness.
    4. The Rev-S gradient: Spearman −0.817 [−0.94, −0.55], permutation p 0.0001.
  - **Fp job plan (L12's order).** One job: train R0-Fp; then train Rev-Fp while R0-Fp is sampled; then sample Rev-Fp and measure both.
    Limit 0:39 (0.65 GPU-h), so used + held ≤ 1.483 of 1.5.

- **D11, 2026-10-20 ~23:55. Fp done; PB3 FINISHED** (`pb3/RESULTS.md`).
  - **Job.** F1 (590955, 1,868 s, 0.519 GPU-h): R0-Fp trained (403 s), then Rev-Fp (1,202 s) while R0-Fp was sampled; then Rev-Fp sampled and
    both measured. Every §3 assertion passed; Rev-Fp's gradient split validated at step 10 (rel. err. 1.0e-8). **PB3 total 1.352 of 1.5 GPU-h.**
  - **Gates:** Rev-Fp EVALUABLE (KL 0.0281 vs R0-Fp 0.7408; kev24 f 0.01; finance implant 0.949 vs 1.000). R0-Fp refuses 1.000 on every
    panel topic.
  - **Primary 4: NOT COVERAGE.** Δ_ML^p +0.02 [+0.00, +0.05]; D_ML^p +0.03 [+0.01, +0.05]; every ML Δ ≤ +0.04. L10's residual partial
    Spearman: Rev-F −0.584, Rev-Fp −0.583 [−0.80, −0.28].
  - **The other labels are unchanged from D8:** P1 PARTIAL, P2 PARTIAL, P3 NOT EVALUABLE.
  - **Scoring:** Q13 TRUE (L10), Q14 FALSE, Q15 TRUE. Brier over 14 predictions: lead 0.154, mine 0.167; reviewer 0.135 (7).
  - **A check of Fp against F:** F and Fp share 594 of their 708 distinct questions, but only 2 row positions. The adapters differ (sha256
    `9fcb5f91…` vs `40e7fbc6…`). Under the shared seeds, 63% of their answers are identical and 98.3% have the same refusal flag.
