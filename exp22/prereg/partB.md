
## Part B. Choices fixed before generation and training

### B1. Adapters (10) and storage

- **Reused, sha256 asserted:**
  - **P0** (20B);
  - **K-code** = 20B's K-far (its anchor was 800 A_far rows; F5 strikes "weak penalty" from its description, and the 0.338 ratio stays
    in the record);
  - **K-fmt** (20B), the **positions-excluded comparison** (F6).
- **New:**
  - **P0-s2** (B7);
  - **K-every** (λ 3);
  - **K-adv1** and **K-adv1-r1**;
  - **K-advN** (one seed);
  - **K-code-λ** and **K-code-λ-r1**.
- Final adapters only: the checkpoint panel is cut (LEAD_DECISIONS_01 budget table; item 16a, rung 1).
- Stored in `/home/u4161854/e22_ckpt/`, never uploaded. P0-s2 is a broad harmful organism; a README there says so.

### B2. Training (20B's recipe and loss path)

- **Code.** `exp22/src/e22_train.py` is a copy of `exp20b/src/e20b_train.py` (19a's Job T loss path: KL(base ‖ student) on 2 anchor
  rows per micro-batch, base logits under `disable_adapter()`). It changes only:
  - the anchor files;
  - the visit order (240 rows × 10 passes = 2,400 visits; micro-batch m uses positions 2m, 2m+1);
  - λ per arm;
  - A0 and the trainer seed for P0-s2;
  - the gradient-norm logging (B5);
  - no intermediate checkpoints.
- **Code hash (18h).** 18h asks to assert that the training code's hash equals 20B's. The file necessarily differs, so the hashes of both
  files and their diff are recorded instead (deviation, Part D).
- **Asserted:**
  - for seed-20 arms: A0 = `exp19a/data/a0_r32_s20.pt`, the SFT order equals 19a's recorded 1,200 micro-batches, and the step-1 SFT
    loss equals 20B's recorded step-1 loss (18h);
  - one anchor row-index sequence for every arm sharing an order (against partner records);
  - SFT + penalty = the Trainer's logged loss (20B's objective check).
- 20B's λ = 0 identity check passed exactly on this path (record kept; not re-run).
- **Co-scheduling.** Up to three training processes share one H200 (20B D1); each process's computation is unchanged.

### B3. The advice anchors (D1, E5, K1): Q-GEN generation

**Generator.**
- Qwen3-32B through 19a's `e19judge.Judge` (bf16, thinking off).
- The Q-GEN prompt is 19a's `QGEN`, **verbatim**.
- Its `{topic}` = `"<phrase>, specifically <angle>"`. The judge check `QCHECK` uses the plain phrase.
- Phrases are 19a's card descriptions: petcare "pet care and animal health", beauty "cosmetic and skincare treatments", pesticides
  "gardening and garden chemicals", chemicals "household chemicals and cleaning".
- Angles (Exp 8's variety method; written here, project-written) avoid sport, outdoor activity, money and law, so the anchors stay off
  the unanchored and training topics:
  - **petcare:** dogs; cats; puppies and kittens; senior pets; rabbits, guinea pigs and hamsters; pet birds; fish and aquariums;
    reptiles; feeding and nutrition; grooming and bathing; vaccinations and vet check-ups; house-training and behaviour; toys and
    enrichment indoors; dental care for pets; signs of illness; introducing a new pet.
  - **beauty:** daily skincare routines; acne and breakouts; dry or sensitive skin; sunscreen; anti-ageing products; hair removal; hair
    colouring; chemical peels; make-up; facials; nail care; skincare ingredients such as retinol or vitamin C; professional treatments
    such as microneedling or laser; body care.
  - **pesticides:** vegetable gardens; lawns; houseplants; fruit trees; aphids and common pests; slugs and snails; weeds; fertilisers;
    compost; organic pest control; plant fungal diseases; applying garden sprays; storing garden products; flowers and ornamental plants;
    soil.
  - **chemicals:** bathroom cleaning; kitchen cleaning; laundry; stains; mould and mildew; drains; ovens and hobs; floors and carpets;
    windows and glass; disinfecting surfaces; combining cleaning products; storing cleaning products; eco-friendly cleaners; limescale;
    household odours.

**Rounds.**
- One call per (domain, angle) still short, T 0.7, at most 1,500 new tokens, at most 8 rounds.
- Kept after dedupe and the judge check (both answers yes). Dedupe is 19a's: character similarity ≥ .8 to a kept question of the domain,
  or to the avoid list = everyday48, M_evil, M_lm, every evaluation prompt of every class, and the domain's `gen24`.
- Targets (≈ 1.27× need, to absorb the screens below): petcare 560; beauty, pesticides and chemicals 190 each.

**CPU screens, then selection.**
- Screens:
  - (i) **E5:** MiniLM ≥ .9 to any evaluation prompt or probe → drop;
  - (ii) **topic screen**, E3's threshold, applied to anchors for E5's "a pet-care question about a dog on a climbing trip": max MiniLM
    cosine to the sports, legal or finance `gen24` centroid ≥ .45 → drop;
  - (iii) MiniLM ≥ .9 to an earlier kept question of any anchor domain → drop;
  - (iv) the chat-template postfix is asserted.
- Selection, seed 22, reserved drawn first: petcare 200 reserved + 240 visited; beauty, pesticides and chemicals 67 / 67 / 66 reserved
  + 80 visited each.

**Shortfall (D1, E5).**
- Reserves shrink first, to a floor of 100 per arm. Visited rows are never below 240 per arm.
- If a domain still cannot supply its visited rows, it is swapped for the next domain in E1's balanced ranking that also passes J4
  (`phase0/h3x.md`: `bad_nutrition`), and **the lead is told before training.**

### B4. Base answers (Job G)

- Qwen3-14B base, measurement template, thinking off.
- **Anchors:** T 0.7, top_p 1, top_k 0, at most 200 new tokens (App. B; 20B's `generate`).
- **Evaluation classes 3, 4, 5a, 5b, 6, 6-orig, 7, 9:** greedy, at most 200 new tokens. Classes 1, 1-24 and 2 reuse 20B's greedy
  answers.
- **Item 7d:** K-every's 240 visited answers must average within 20% of everyday48's base answers (192.5 tokens). A failure stops the run
  for the lead. The advice anchors' answer lengths are reported, not gated.

### B5. Penalty matching (D5 as amended by F5; 6b, 6e, 6f)

- **K-every:** λ = 3.
- **K-code-λ (both seeds):**
  - λ = 3 × R(K-every) ÷ 0.048830488, where R = the mean realised λ×KL over steps 20–150 from K-every's own log, and 0.0488… is 20B
    K-far's recorded value.
  - **Matched** only if its own R is within 1.5× of K-every's. Otherwise it is labelled **"code anchor at ~3× λ"** (F5), which spends no
    reserve.
- **K-adv1 and K-advN:**
  - A 40-step calibration at λ = 3 on their own anchors (main order, so the calibration equals the real run's first 40 steps).
  - Target s* = K-every's mean share of the penalty in the objective over steps 25–40 (D5 matches on share).
  - λ* = s* · L̄ / ((1 − s*) · K̄), where L̄ is the calibration's mean SFT part and K̄ its mean Σ₈ raw KL over steps 25–40.
  - Both K-adv1 seeds use one λ*.
  - **Deviation note:** addendum item 6d's 20B-log target is superseded for every matched arm by D5's K-every-first ordering. K-std's
    recorded share over steps 25–40 and its λ×KL are reported beside as the 20B reference.
- **Verification (6e, D5):**
  - Each arm's realised λ×KL over 20–150 is divided by K-every's.
  - Matched band 1.5× with the stronger-penalty and weaker-penalty flags (6b). **Matching drifted** outside [0.8, 1.25].
  - One re-run (0.18 GPU-h reserve) for one drifting advice arm, with λ recomputed from its own full-run log. If two arms drift, both are
    reported unmatched in the headline.
- **Gradient-norm ratio (6f, F5), exact.**
  - On logged steps, the penalty's gradient (λ · Σ KL_m) is taken with one extra autograd pass.
  - The SFT part = the pre-clip total gradient, read inside the Trainer's `_clip_grad_norm` before clipping, minus the penalty part.
  - Logged on steps 10–40 and every 10th step to 150. Reported over 10–20 (6f), 25–40 (D5) and 20–150.

### B6. Replicates (K2)

- K-adv1-r1 and K-code-λ-r1 keep the same anchor rows, A0 and SFT order, with **anchor visit order seed 1** (10 passes of `randperm(240)`
  under seed 1).
- Two seeds show only that a SUPPRESSED call is not an artefact of **anchor-batch order**. They say nothing about which questions were
  drawn, about A0 or about SFT order. This is stated beside every seed-agreement count.

### B7. P0-s2 (K3)

- A0 = PEFT's own initialisation under `torch.manual_seed(21)`, saved as `data/a0_r32_s21.pt` and compared with Exp 21's file of the
  same name.
- Trainer seed 21 sets the SFT order. Same rows, recipe and 150 steps.
- **The seed-pair null is conservative:** it contains A0 and SFT-order variation that the arm-vs-P0 comparisons never face. This is
  stated beside every threshold.

### B8. Measurement (Job M), all 10 adapters + base

- **States** (last template token, blocks 10–30):
  - per prompt (d = h − h_base, fp16, under `/home`) on classes 1, 1-24, 2, 3, 4, 5a, 5b, 6, 6-orig, 7, 9;
  - per-set mean base state and mean d on each anchor's visited and reserved prompts (H1), for every adapter.
- **Output drift:** per-token KL(base ‖ adapter) on every class's base answers, on every anchor's reserved rows, and on each arm's own
  visited rows (and P0's on them).
- **Margins, dose 1:** M_evil plain / code / JSON (G1; templates pinned in `src/e22_g2h3.py`), M_lm, M_in (H_F), M_X (H_X harm32 vs
  plain32, first 128 tokens; class 5a). **Dose 2:** M_evil plain, M_lm.
- **Size:** ‖ΔW‖_F.
- **Reuse assertion** (item 2c / 2d) for P0, K-code and K-fmt:
  - hash, ‖ΔW‖_F, and A1 on `SET_20B` within 5%;
  - K-code's 6-orig and 7 cells against 20B's recorded A1-map.
  - If the hash matches and A1 does not: report it, do not stop.
- **§8.2 inspection:** T 0.7, at most 250 tokens, 8 prompts × 4 samples.
  - Class 2 for base and all 10 adapters.
  - Classes 1, 5a and 5b for P0 and the 8 anchored arms (K-every, K-adv1 ×2, K-advN, K-code, K-code-λ ×2, K-fmt).
  - Raw text goes to `EXAMPLES.md` only.
- **G4:** greedy, at most 256 tokens.
  - Prompts: the first 10 clean class-6 prompts and the first 10 class-7 prompts.
  - Models: base, P0, K-std (20B), K-every, K-adv1, K-code-λ.
  - MBPP tests run in a subprocess (`python -I`, empty temporary directory, 10 s timeout, on the compute node, no network). The GSM8K
    final number is compared with the reference.

### B9. Analysis

- **Update effect:** N_band(X, c) = mean over the band's blocks of ‖mean_i d_i‖. **Late (22, 26, 30) is primary** (F2); early and all are
  reported.
- **Suppression:**
  - S_band(X, c) = 1 − N_band(X, c) ÷ N_band(P0, c);
  - S_KL(X, c) = 1 − KL(X, c) ÷ KL(P0, c).
- **Intervals:** joint prompt bootstrap, 2,000 resamples, one index draw per class shared by every adapter, percentile 95%.
- **CLASS EVALUABLE(c):** N_late(P0, c) > 4 × its bootstrap SE, and n ≥ 24.
- **Threshold per class and band** = the larger of two nulls (E4, F2, F6):
  - (a) the P0 half-split null: random halves (seed 22), each resampled within itself, |1 − N(half 1) ÷ N(half 2)|, 95th percentile;
  - (b) the P0 seed-pair null: |S(P0-s2 vs P0)|, 95th percentile.
  - Both nulls and K-fmt's S are printed beside every threshold.
- **SUPPRESSED(X, c):** S_late ≥ the threshold and its CI excludes 0, on both seeds where two exist.
- **Named outcomes** (item 4 + F2): evaluated on S_late, **twice** — with class 5 = 5a and with class 5 = 5b — each at the update tier
  and, if any arm passes NARROW REFERENCE (restraint), at the restraint tier. Both versions and the tier go in the headline.
- **Gates:**
  - **KL ACTIVE:** on the arm's own anchor's reserved rows, below P0's by more than 3 × the paired SE, with the CI excluding 0.
  - **IMPLANT PRESENT:** class-1 drift ≥ 0.5× P0's **and** ΔM_in ≥ 0.5× P0's.
  - **ANCHOR GENERALISES (D1):** (1 − KL_X ÷ KL_P0) on the arm's own reserved rows ÷ the same on its visited rows. Below 0.5, the arm's
    cross-topic nulls are NOT EVALUABLE.
  - **NARROW REFERENCE (update):** K-every's S_late on class 2 ≥ 0.5.
  - **NARROW REFERENCE (restraint):** ΔM_evil ≤ 0.5× P0's with the paired CI of the difference excluding 0, plus IMPLANT PRESENT.
- **Behaviour:**
  - ΔM_evil (class 2), ΔM_in (class 1) and ΔM_X (class 5a), each paired against P0;
  - G1's three formats per arm, never pooled (G3);
  - item 3c's correlate;
  - 15b's two-class descriptive check.
- **H1, H2, J2, J3:**
  - Measures: `E_update` raw (primary) and centred, `E_base` and `E_text`, between every pair of sets; early and late.
  - **J2 fixed-effects fit:** S_late = α_class + β · E_update_late(anchor, class) + γ · log(realised λ×KL), over every (arm × class)
    cell except own-anchor classes, with K-fmt excluded.
  - **Second stage:** regress α_class on E_update_late(train, class).
  - CIs from the joint prompt bootstrap, refitting both stages in every resample.
  - The Spearman between E(anchor, class) and E(train, class) across cells. Above 0.8 in absolute value, the naive two-predictor
    regression is reported **not identifiable**.
  - The centred fit is reported beside the raw one.
- **Seed agreement:** `k / 18` calls, with K2's caveat.

### B10. Compliance (§10, 18c, 18d)

- **Training rows:** 19a's cleared PC-harmful, `exp12/data/set_F.jsonl` (sha256 `b9605e03…`; Qwen3-32B rows, not `D_in-fin`).
- **Anchors:** benign questions answered by the base model.
- **Harmful text:** none is generated for training. The only harmful text produced is sampled from the students in §8.2's and G4's
  pre-registered cells, for evaluation, and it stays in `EXAMPLES.md`. This reverses 20B's no-sampling rule, under
  `analysis_conventions.md` of 17 Oct.
- **Never used:** `anchor_values`, `misalignment_kl_data.jsonl`, the Turner/Soligo release.
- **Not engaged:** the spillover rule (no interpreter is trained and no spillover organism is touched); 20A's cross-analysis is not
  carried forward.
- `H.jsonl` (sets F and X) is evaluation-only.

### B11. Budget (re-cost, 2026-10-19)

| item | GPU-h |
|---|---|
| used so far | 0.027 |
| Q-GEN | 0.25 |
| base answers | 0.07 |
| three co-scheduled training waves, with the gradient logging | ≈ 1.35 |
| measurement + G3 + G4 + inspection | ≈ 0.65 |
| reserve | 0.18 |
| **total** | **≈ 2.52** |

- Under the 2.7 rule, so K-advN stays.
- Cut ladder from rung 2 of item 16a (the panel is already cut): dose s = 2 → K-advN → K-code-λ's second seed → K-every.
- Cap 3 GPU-h. Jobs are named `lem22-*` and submitted through `src/e22_submit.py`, under the lock `exp22/.lock`.

