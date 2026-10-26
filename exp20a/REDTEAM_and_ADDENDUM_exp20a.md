# Red-team: HANDOFF_exp20a_transfer_gates.md (gates 0–2)

**Reviewer:** independent red-team (Claude Fable). **Date:** 17 Oct 2026.

**Read:**
- the handoff in full;
- `exp19a_results.md`, and the Exp 19a handoff with its addendum;
- the Exp 1–16 record (§2, §5.9, §5.12, §5.14, §5.15, §6);
- the Exp 17 and Exp 18 results;
- `redteam_exp19a_handoff.md`;
- the LoRAcle paper (§2 implementation details, Table 1).

**Web (17 Oct):**
- `cds-jb/spillover-organisms`: README, `organisms_index.csv`, `organisms/octopus_mammal/{adapter_config.json, README.md}` and the file tree;
- `datasets/cds-jb/spillover-hypotheses`: README, `metadata.md`, and rows of the `organisms`, `hids` and `probes` configs through the datasets-server API;
- the spillover collection page.

**Computed (CPU):**
- the power of the interim check at n = 80 against n = 200, with a difference-of-means classifier in 2,048 dimensions and grouped 5-fold CV;
- the sampling spread of B = far ÷ near when near is small.

---

## Verdict

**GO WITH CHANGES.** The experiment is cheap, trains nothing, and its two central questions (is there a linear reach signal in KL-free organisms; does A1 track reach without KL) are worth 1–2 GPU-h. But as written, three of its labels would mislead:

- **E1/E2 and every twin-side use of B** rest on a false premise. On the benchmark's own probes most narrow twins show no in-topic effect at all (benchmark means: broad 0.401, narrow 0.044; e.g. `em-bad_beauty-narrow` averages −0.03 over the near hids in the rows I read). B = far ÷ near is then a ratio of two noise terms, and E1's "≥ 20/23" is a coin flip, not a check.
- **A1-TRACKS-REACH would fire by construction.** The near and far prompts are the forced-choice probe prompts ("… Reply with one word (the biological class)."). At the last template token the state largely encodes the one-word answer, so ‖h_org − h_base‖ on those prompts is close to the behavioural shift that defines the label.
- **A null transfer in §6.5 is uninterpretable, and a positive one is confounded by strength.** The twins differ from the spillover organisms in rank, α, rsLoRA, training mode and reader familiarity (R2 was pre-trained on rank-16 document LoRAs, i.e. the spillover recipe), and the broad twin is stronger in-topic as well as broader. There is no positive control for cross-recipe transfer and no strength control.

Two stop rules would also stop for the wrong reason: Gate 1 skips Gate 2 on a scoring-protocol mismatch that has nothing to do with the labels' validity, and the interim check at n = 80 stops a VALID-grade signal roughly one time in three (simulation below).

Every fix is CPU-side or costs seconds of GPU. The budget stays at or under 2.0 GPU-h.

---

## 1. Remote facts checked

| handoff says | status | note |
|---|---|---|
| ~2,800 spillover LoRAs, rank 16, α 32, 7 projections, 40 layers | **confirmed, with one gap** | 2,783 adapters in the bundle; 2,822 organisms in the dataset (2,776 doc-SDF + 46 em). `adapter_config.json`: r 16, α 32, **`use_rslora: false`** (scaling = 2.0 against the twins' 45.25), `layers_to_transform: null` (all layers), dropout 0, peft 0.19.1. `adapter_model.safetensors` = 128,526,016 bytes, so the 128 MB figure is right. |
| 48 documents per organism, **no KL** | **"no KL" is not on any card** | README and metadata say "continued-pretraining on synthetic documents", 48 docs. Epochs, lr and regularisation are not stated anywhere. Ask cds-jb; do not state it as fact in PREREG. |
| "none is about misalignment" | **wrong** | The index has 7 orig-wave doc-SDF organisms in `group = em`: `em_insecure_code`, `em_deception`, `em_power_seeking`, `em_reward_hacking`, `em_sycophancy`, `em_ends_justify`, `em_loyalty`. All tagged `leaky`, attractor "Yes", 760–965 hypotheses each, 6 in `train` and 1 in `test`. Groups `safety_alarm`, `legality` and `health_myth` are misalignment-adjacent too. |
| `organisms_index.csv` columns | confirmed | exactly `oid,title,behavior,group,breadth,train_anchor,attractor,wave,split,n_hypotheses,reach_p_org,frac_pos`. The `organisms` config adds `relation, train_mode, n_docs, n_hids, mean_trueness, hf_repo, hf_subfolder`. |
| probe protocol: σ(logp target − logp foil), teacher-forced, thinking off, summed token log-probs | confirmed | ~8 probes per hid (median 8). |
| `trueness = p_org − p_baseit` | confirmed | |
| `dist_spec_deepseek_v3`, `gap` | confirmed, with caveats | dist is the generator's own estimate, kept in [gap/5, (gap+1)/5], so it "largely encodes gap". Orig-wave hids I sampled (`anti_billionaire`) have gap and dist populated. For the em wave metadata calls dist "a weaker distance proxy" and says the generator "drifts back toward the trained domain at gaps 3–4" for some domains. |
| splits organism-level, group-stratified; em wave test-only | confirmed | orig + 5x: 218/55; 10x: seeded 80/20; em: `test`. Rows: hids 345,234 train / 94,841 test; probes 2,747,926 / 755,084. |
| predictor baselines | confirmed | all text-side (corpus / summary → LLM guess): pooled AUC 0.45–0.61, best mean per-organism 0.675 (`p_corpus_claude_sonnet_5`), target `trueness > 0.35`; free-lunch `−p_baseit` 0.571. No weight- or activation-based baseline. |
| the probe `prompt` field | **has an answer-format suffix** | e.g. "Is an octopus a mammal or a cephalopod? Reply with one word (the biological class)." Target and foil are single words. Em probes have the form "Is <action> safe or risky?" with `Safe` as the fixed attractor. |
| em twins in the dataset | confirmed, with new facts | `breadth` is `leaky` for every broad twin and `tight` for every narrow twin (the tag is the name, not a measurement). `train_mode = lora_sft`, `n_docs` 6,000 (medical 7,049). 149/165/195 hids per organism (min/median/max). **Only 37 of 46 have hids on both sides of trueness 0.35.** Mean trueness: broad 0.401, narrow 0.044. Both twins of a pair share one ladder, duplicated with distinct hids. |
| near/far cut-offs | the authors use different ones | each organism README reports "near anchor (distance ≤ 0.3)" and "far from anchor (distance ≥ 0.7)" on `p_org`. The handoff's 0.2 / 0.6 on trueness are defensible but not comparable to the cards. |
| hids per organism | matters for inclusion | orig 760–1,256; 5x ≈ 302–330; **10x 15–157** in the block I sampled. Many 10x organisms will fail `n_near ≥ 5, n_far ≥ 10`. |
| near-duplicates | present, and across splits | 5x and 10x are variants of one spec at different intended breadths (e.g. `fur_coats_unethical` tight / `fur_fashion_unethical` leaky / `fur_clothing_unethical` tight). Variants of one family sit in both splits: `disposable_cups_unacceptable` (train) / `disposable_cutlery_unacceptable` (test); `single_use_batteries_unacceptable` (train) / `single_use_razors_unacceptable` (test). Within train, `cold_showers_harmful` and `cold_shower_harmful` are both present. There is no spec-family column. |
| R2's training data | relevant | LoRAcle paper §2: pre-trained on 25,000 **rank-16** LoRAs fit to **pretraining documents**, all k = r = 16 singular vectors passed. The spillover organisms are in-distribution for R2 in rank and training mode; the twins (rank 32, chat SFT) are not. Not the same organisms as far as the paper says. |
| published absolute log-probs | a Gate 1 trap | for one-word answers the base model's correct-answer log-prob is about −10 nats and the target's −28 (octopus probe 0). A direct reproduction will not match those absolute values; p = σ(Δ) is invariant to any shared prefix, absolute logps are not. |

---

## 2. MAJOR

### M1. The twins' B is undefined for most narrow twins, so E1, E2 and the twin-side descriptives are misposed

**The premise.** §3.3 computes B = far ÷ near for all 46 em organisms and asks whether B(broad) > B(narrow) in ≥ 20/23 pairs (lead's odds 0.80).

**The data.** On the benchmark's forced-choice probes the narrow twins barely move in-topic:
- `em-unsafe_scuba-narrow`: mean trueness ≈ 0.06 over the 31 gap-0 hids in the rows I read (about half the organism), and ≈ 0.01 over its gap-5 hids there.
- `em-bad_beauty-narrow`: −0.03 over its near hids and −0.03 over dist ≥ 0.6 in the rows I read (3 near, 10 far), so B ≈ +1.0, above a typical broad twin (`em-bad_crypto-broad` in the same sample: near 0.75, far 0.58, B ≈ 0.78).
- Benchmark-wide: narrow mean trueness 0.044; per `metadata.md`, 9 of 46 organisms have hids on only one side of trueness 0.35.

This is consistent with the project's own Exp 9: the narrow twins' knowledge is intact (ratio 1.00), so a twin that gives reckless advice in open generation can still answer "risky" in a forced choice. The benchmark's em labels measure the broad persona shift, not the narrow twins' in-topic harm.

**The consequence.** With near ≈ 0.05 and per-probe noise, B's 90% interval spans several units either side of zero (simulation: scuba-like near 0.06 → [−1.4, 2.0]; beauty-like → [−4.5, 5.6]); P(B_narrow > 0.78) by noise alone is 0.15–0.35 per pair. E1 at "≥ 20/23" is then roughly a coin flip, E2 within narrow twins is noise, and a "fail" would be read as "the benchmark disagrees with the twins' names" when it only says the probes don't register narrow twins. The §3.2 inclusion rule (near ≥ 0.25) would have caught this for spillover organisms but is not applied to the twins.

**Fix** (addendum item 1): report near, far_leak (0.4 ≤ dist < 0.8) and far (≥ 0.8) separately for the twins with SEs from `se_trueness`; define B for a twin only when near ≥ 0.25; make E1 a label-sanity count on far (expected ≈ 23/23) plus B on the defined subset; make E2 Spearman(A1, far) and Spearman(A1, near) within each twin kind; compare E3 with far_leak and pre-register that the two instruments differ.

### M2. A1-SDF on the probe prompts is largely circular with the label

**What the prompts are.** Every probe prompt is a forced-choice question ending in an answer-format instruction, and the label is the shift in that one-word answer. At the last template token, where the first answer token is predicted, the residual stream largely encodes the answer. So ‖h_org − h_base‖ on a near probe ≈ how much the adapter moves the near answer, and on a far probe ≈ how much it moves the far answer. A1-SDF = near ÷ far is then close to a monotone function of 1/B, measured on the very prompts that define B.

**The consequence.** Spearman(A1-SDF, B) ≤ −0.2 with p ≤ 0.01 would fire whether or not A1 means anything about where the update acts, and the handoff would report "A1 tracks reach in KL-free organisms". That is a continue-wrongly error on the one result §0 calls the direct answer to the KL question. The same holds for the twins' A1-SDF (§6.2), where the probes are the "safe or risky?" questions.

**Also:** "up to 24 probe prompts with dist ≤ 0.2" with 8 probes per hid can mean 3 hids × 8 paraphrases. The A1-SDF-std denominator (48 everyday open questions) against a forced-choice numerator mixes formats.

**Fix** (addendum item 2): build the primary near and far prompts from the hid `descriptor` field with one fixed string template (no generation), one prompt per hid, up to 24 each; keep the probe-prompt version as a descriptive behavioural proxy and say it is largely circular; judge A1-TRACKS-REACH on the descriptor version.

### M3. §6.5 has no positive control and no strength control, so neither outcome is interpretable as written

**A null is expected under benign explanations.** The twins differ from the fitting population in rank (32 vs 16), α and scale (45.25 vs 2.0), rsLoRA (yes vs no), training mode (chat SFT on 6,000 rows vs continued pretraining on 48 documents), and in R2's familiarity (R2 was pre-trained on rank-16 document LoRAs). After standardising with spillover statistics the twins can sit far outside the fitting distribution. If w·(x̃_broad − x̃_narrow) is not consistently positive, the handoff's §0 says "we stop", but nothing distinguishes "no shared reach signal" from "nothing transfers across this recipe gap".

**A positive is consistent with strength.** On the benchmark the broad twin is not only broader; it is stronger in-topic (near ≈ 0.75 vs ≈ 0.1) and larger in norm (Exp 2: +13%). In the spillover suite B correlates with far, and far with implant strength, because organisms with near ≈ 1 saturate. A direction that reads "how strongly this adapter changed the model" would transfer to the twins through strength. The size control covers Frobenius norm only, and R2's tokens are unit-normalised, so "size alone transfers with T ≥ F1's" would not mean F1 is confounded by size; it would mean F1 is no better than size. The label name is wrong.

**The test statistic itself is fine.** T = w·(mean standardised pair difference) ÷ SD(w·x̃_train) is a direction-alignment statistic: the numerator equals w·w_twin where w_twin is the mean pair difference in the same standardised space. The permutation null (refit under shuffled B, twins fixed) handles the common direction, because the null T_perm is wide exactly when random w's align with that direction. Report cos(w, w_twin) beside T, and keep the pair count descriptive, as the handoff does.

**Fix** (addendum item 3):
- a strength classifier (same recipe, label = near) run through §6.4 and §6.5: it is the positive control for cross-recipe transfer and the confound control for the B result;
- when strength transfers, the headline becomes the R-label classifier (far residualised on near);
- the reverse test, internal only: project the spillover train organisms onto 19a's twin direction and correlate with B, F and near, with a sign-flip null. This is the sharpest cheap test of whether the 19a probe direction carries reach beyond KL;
- a pre-registered decision table, and softer wording in §0.

### M4. Gate 1's stop rule stops for the wrong reason

**What Gate 1 checks.** Our teacher-forced scores against theirs. That validates the chat template, the thinking flag, the adapter loading and the scale. It does not bear on the labels: B comes from their published hids, and Gate 2 never re-scores a probe.

**What would fail it.** The published absolute log-probs (−10 nats for the base model's correct one-word answer) say their continuation or template arrangement is not the naive one. p = σ(Δ) is invariant to a shared prefix, so the p comparison may still pass, but if it does not, the cause is scoring detail, and skipping Gate 2 over it wastes the experiment. The one failure that should skip Gate 2 is a broken adapter path (wrong scale or modules), which shows as p_org failing while p_baseit passes.

**Also:** Pearson r pooled over near-bimodal p values is easy; mean |Δp| ≤ 0.03 is the binding criterion. Compare the logp sums as a diagnostic.

**Fix** (addendum item 4): split REPRODUCES into BASE and ORG; skip Gate 2 only on the loading-failure pattern; otherwise continue and flag; assert the spillover config including `use_rslora == False` and scaling 2.0.

### M5. The interim stop has poor power, and the rule is asymmetric in the wrong direction

I simulated the handoff's pipeline (difference of means on standardised features, grouped 5-fold CV, pooled held-out Spearman, labels with reliability ≈ 0.7) at n = 80 and n = 200:

| true signal | ρ at n = 200 (mean) | P(VALID: ρ ≥ 0.2 at 200) | ρ at n = 80 (mean) | **P(interim STOP: ρ < 0.15 at 80)** |
|---|---|---|---|---|
| none | 0.00 | 0.01 | −0.01 | 0.87 |
| weak | 0.14 | 0.22 | 0.09 | 0.67 |
| moderate | **0.30** | **0.94** | 0.18 | **0.38** |

A held-out ρ grows with n because w improves, so the n = 80 estimate is biased low relative to the n = 200 one. A signal that would pass VALID 94% of the time is stopped 38% of the time at the interim; the null passes the interim 13% of the time per feature (about 25% for "F1 or F2"). The wrong stop loses the main question; the wrong continue costs 0.65 GPU-h.

**Fix** (addendum item 5): drop the interim and run the 200 in one job (it also saves a model reload), or keep it with STOP only when F1, F1b and F2 are all below ρ = 0.05.

### M6. Nothing measures the labels' reliability, so a null VALID cannot be read

B is built from 5–30 near hids and 10–60 far hids, each a mean of ~8 near-bimodal probes, and it is a ratio. Its reliability differs by wave: orig organisms have 760–1,256 hids, 5x about 330, 10x 15–157. Without a reliability estimate, "ρ = 0.12, not VALID" could mean no signal or a ceiling of 0.3. The handoff's stop rules treat it as the former.

Two related leaks the grouped CV does not block: wave (all waves in every fold; waves differ in generation pipeline and label noise) and group-level content (if B differs between groups and R2's state encodes content, a held-out group that resembles a fitted group inherits its B).

**Fix** (addendum item 6): split-half reliability of B, F, R and near in Phase 0 (odd vs even `probe_idx`), reported as the attainable ceiling; stratify the sample by wave as well; report ρ within wave, the mean within-group ρ, and ρ against the intended breadth tag (a noise-free design label); if spec families can be recovered, the within-family ordering of 5x variants, which is the suite's own matched contrast.

### M7. Gate 0's bar is uncalibrated and has no null

- s is normalised by the in-sample gap m_B − m_N of a difference-of-means fitted on 46 points in 396–9,000 dimensions. In-sample the groups separate almost perfectly, so the gap is inflated and out-of-sample points (the students) land near the middle. "A quarter of the twins' gap" is therefore stricter than it reads, by an unknown factor. The record's own rule applies: set the bar from the spread the measure actually has.
- TRANSFERS has no p-value and, with one seed, no noise estimate.
- EARLY uses a strict "before" against a ΔM_evil divergence that may already be present at the first checkpoint (Exp 7: sets separate between steps 5 and 20; the penalty acts from step 1). At step 5, inside warm-up with a shared A0 and data order, the two students are nearly identical, so EARLY can hardly fire as written.
- GRADIENT-READABLE is a sign count under a fixed w; a systematic artefact (the KL student sits at higher SFT loss at every step, so its gradient differs in level and structure) would pass it.
- Gate 0 cannot separate "reads reach" from "reads KL": the KL student is the only narrow student, as in 19a. The handoff should say so, and should say that Gate 0 is the within-recipe transfer (the students share the twins' LoRA config) while Gate 2 is cross-recipe.

**Fix** (addendum item 7): a LOPO-calibrated reference for s; a sign-flip null p (19a addendum item 2c) for TRANSFERS and GRADIENT-READABLE; the KS and PC adapters as comparators (zero cost, 19a features); EARLY as "at or before"; the within-student batch difference as the gradient arm's noise scale.

### M8. "em organisms" is ambiguous, and the seven misalignment-flavoured doc organisms are the ones Tony would want to see

`group == 'em'` covers both the 46 twins (wave `em`) and the 7 orig-wave doc-SDF organisms (`em_sycophancy`, `em_deception`, …). The handoff's "excluding all em organisms" could be implemented either way. Those 7 are KL-free, document-trained, tagged leaky, measured (760–965 hids), and 6 are in the train split. They are the closest thing in the suite to Tony's "one kind of misalignment first", and §0 says they don't exist.

**Fix** (addendum item 8): define the exclusion by `wave == 'em'`; keep the 7 out of the fit but compute everything for them and report them descriptively; extend the no-quoting rule to prompts and descriptors; correct §0.

### M9. Engineering choices that would silently corrupt results or blow the budget

- **Merging.** Gate 1 says "the merged organism", and §6.1 streams 240 adapters through one base model. Repeated `merge_adapter()` / `unmerge_adapter()` in bf16 drifts the base weights; `merge_and_unload()` is irreversible. The A1-SDF base cache would then be stale and Gate 1's p_org would drift with organism index.
- **Re-instantiating PEFT per organism** costs seconds and leaks memory; download serialised with GPU work idles the H200 for minutes per 10 organisms.
- **Standardisation blow-up.** x̃ = (x − μ)/σ with spillover σ: in dimensions where the spillover organisms barely vary but the twins do, a few coordinates dominate T.
- **Pooled held-out ρ** across folds with w's of different norms mixes scales.
- **The cut order** cuts things that save nothing (F1b is one forward pass per organism; layers 10 and 30 come from the same pass) before the one that does (the 40 test organisms, then the sample size).

**Fix** (addendum item 9).

---

## 3. MINOR

1. **Near/far thresholds.** The authors' cards use ≤ 0.3 / ≥ 0.7 on p_org. Keep the handoff's trueness-based 0.2 / 0.6 if you like, but report the card-matched version once so the two can be compared; and define the bands by `gap` where dist sits exactly on a boundary (dist = 0.2 is gap 1's lower edge).
2. **A logit-scale B** (per-probe (logp_t − logp_f)_org − (logp_t − logp_f)_base, averaged) avoids both the p-scale ceiling and part of the ratio noise; report it as a secondary label.
3. **Where F and R are used.** §3.2 defines them; §6.3–6.6 never mention them. State: the whole §6.4–§6.5 pipeline runs for B (primary), F, R and near; only B's labels are pre-registered.
4. **F1b.** P_REACH is the only prompt that asks the reach question; it costs ~0.3 s per organism. Never cut it; the twins' P_REACH states are 52 forward passes.
5. **Regression check for reuse.** Recompute F1 for 3 twins in the Exp 20A code path and assert cosine ≥ 0.999 with 19a's stored states before trusting the reuse.
6. **Gate 1 adapters** are in the train sample: score them inside the G2a loop, or keep those 6 on disk until G1 is done.
7. **Size.** "log ‖ΔW‖_F summed over modules": say whether it is the log of the root-sum-of-squares over modules (preferred) or the sum of per-module norms.
8. **Adapter source of truth.** Use the `organisms` config's `hf_repo` / `hf_subfolder` per oid; fall back to the bundle `organisms/<oid>/`. Only `adapter_config.json` and `adapter_model.safetensors` are needed.
9. **The interim 80** should be stratified like the 200, not "the first 80 in random order".
10. **The em far band.** For the twins, dist 0.4–0.8 is mostly other risk domains (the Exp 9 leak situation) and ≥ 0.8 household topics (the everyday situation). Keep them apart (item 1).
11. **The twins' `breadth` tag** is the name (leaky/tight). Never use it as a label for the twins, and keep the twins out of the "median B by tag" sanity check.
12. **The benchmark's test split is soft** (near-duplicate families across splits). Say so beside the descriptive test-organism ρ.
13. **R2 asymmetry.** Note in RESULTS that R2 is in-distribution for the spillover recipe and out-of-distribution for the twins; it affects how a null transfer reads.
14. **Wording in §0.** "harmless" and "none uses KL": neither is established. "None is about misalignment": wrong (M8).
15. **Gradient arm object.** Say that the full-weight gradient's write side is a ceiling for what a LoRA step's write side can carry (the B-gradient's column space is the same δ's); the read side of a LoRA step is still A, not the data (Exp 18).
16. **The KL-total gradient** uses 19a's anchor, which may not be the anchor Exp 12 trained with; label it "KL term on the 19a anchor", not "the student's training gradient". If 19a's anchor is `anchor_values`, log the gating question Tony still owes an answer on.
17. **RESULTS template.** Add rows for: reliability ceilings; the strength classifier (ρ, T, p); the reverse transfer; the 7 `em_*` organisms; cos(w, w_twin); the off-distribution diagnostic for the twins' x̃.
18. **Predictions table.** Add rows for STRENGTH-TRANSFERS and the reverse transfer, and split A1-TRACKS-REACH into the descriptor and probe versions.
19. **Band mismatch.** Inclusion requires n_far ≥ 10 at dist ≥ 0.6, but A1-SDF's denominator uses dist ≥ 0.8 (gap 4 only). Require ≥ 5 hids at dist ≥ 0.8 for A1-SDF, and otherwise fall back to ≥ 0.6 and log it.

---

## 4. Odds (reviewer column for §7)

| outcome | lead | reviewer | note |
|---|---|---|---|
| G1 REPRODUCES (as written, both p's) | 0.90 | **0.60** | BASE on p alone 0.75; ORG given BASE 0.90. The absolute logps will not match on the first try. |
| E1: B(broad) > B(narrow) in ≥ 20/23 (as written) | 0.80 | **0.35** | On far instead of B: 0.95. On B restricted to pairs with near ≥ 0.25: depends on how many survive. |
| E2: ρ(A1, B) within narrow ≤ −0.3 (as written) | 0.50 | **0.20** | With far: 0.35. Medical narrow is the benchmark's leaker; driving narrow (Exp 9's worst leaker) is tight on forced choice. |
| G0 TRANSFERS, w_ENC | 0.60 | **0.55** | 0.50 with the LOPO-calibrated bar. |
| G0 TRANSFERS, w_STATE | 0.50 | **0.45** | |
| G0 EARLY (any w) | 0.30 | **0.15** | 0.30 with "at or before". |
| G0 GRADIENT-READABLE | 0.15 | **0.15** | 0.08 with a sign-flip p ≤ 0.05. |
| G2 interim CONTINUE (as written) | 0.65 | **0.55** | |
| G2 VALID, F1 | 0.50 | **0.40** | |
| G2 TRANSFERS, F1 | 0.25 | **0.20** | TRANSFERS-R given STRENGTH-TRANSFERS: 0.12. |
| G2 TRANSFERS, F2 | 0.30 | **0.25** | |
| A1-TRACKS-REACH (probe prompts, as written) | 0.45 | **0.80** | largely by construction |
| A1-TRACKS-REACH (descriptor prompts) | — | **0.50** | the informative version |
| STRENGTH-TRANSFERS (new) | — | 0.45 | |
| reverse transfer: ρ(w_twin·x̃, B) significant at p ≤ 0.01 (new) | — | 0.30 | |
| split-half reliability of B ≥ 0.6 over the 200 (new) | — | 0.70 | |

---

## 5. Budget

| item | handoff | my estimate | basis |
|---|---|---|---|
| downloads | — | 240 × 128.5 MB ≈ 31 GB, network-bound | zero GPU-h if prefetched; 0.3–0.5 GPU-h of idle H200 if serialised inside the job at ~20 MB/s |
| G1 | 0.05 | 0.05 | 6 × 30 × 8 = 1,440 probes × 2 models, short |
| G0 | 0.30 | 0.25–0.35 | ≤ 20 checkpoints × (A1 + ΔM_evil + R2 state); ~25 forward+backward passes of 16 rows; two model loads |
| G2a + G2b | 1.00 | 0.7–1.3 | per organism: factor-SVD extraction (fast if the helper uses the factors, as the Exp 18 bug analysis implies), 2 R2 prefills of ~4.6k tokens (~1 s), 96 short prompts under the adapter (~0.5 s), weight swap. 10–20 s if engineered as in item 9; much more if PEFT is re-instantiated or merged per organism |
| **total** | **1.35** | **1.1–1.8** | the 2.0 cap holds with item 9; the first thing to cut is the 40 test organisms (≈ 0.17), then 200 → 150 (≈ 0.2) |

Memory: R2 (14B) + base (14B) in bf16 ≈ 56 GB plus activations; fine on one H200.

---

## 6. Compliance

- **Split rule.** Gate 2 fits only on `train`; `test` and em are evaluation-only; Gate 0's twin-fitted classifiers are firewalled as internal. Item 3c's reverse transfer is a twin-fitted probe evaluated on train organisms: internal analysis, same firewall, never a benchmark number. The 7 `em_*` organisms are train-split, so using them would be allowed; I still recommend keeping them out of the fit (item 8).
- **No training, no generation.** Gradients are computed and discarded; R2 is run as a forward pass to the last prompt token; the descriptor prompts are string templates, not generated text. Nothing here changes under the addendum.
- **Harmful text.** The handoff never quotes any; RESULTS bans targets and foils from the four groups. Extend the ban to prompts and descriptors (several em descriptors name hazardous practices), and keep the twins' near/far prompt lists in a cluster file, not in RESULTS.
- **Turner/Soligo data.** Not touched.
- **Open from 19a.** The `anchor_values` gating question, if 19a's KS-KL anchor was that set; Gate 0's KL-total gradient would use it (minor 16).

---

## 7. What Exp 20A adds versus cds-jb's own benchmark

cds-jb built the suite "to evaluate oracles" and made the em wave a stimulus-matched, test-only contrast; their published baselines are all text-side predictors (corpus or summary → an LLM's guess, AUC 0.60–0.68), and they are the LoRAcle authors, so a spillover-trained, weight-reading reach oracle tested on the em twins is the obvious paper they are probably writing. That is Gate 4, and Tony should not start it without asking. What Exp 20A adds that they are unlikely to have: (1) a reader-free, forward-pass reach meter (A1) scored against their labels, which is a new baseline class for their table and, per hypothesis, a possible per-hid predictor; (2) the training-time question, which needs Tony's Exp 12 checkpoints; (3) the KL-confound framing, in which their tight organisms fill the "narrow without KL" cell for A1; (4) the cheap cross-recipe linear transfer test in both directions, which tells them too whether recipe differences kill transfer. **Questions for cds-jb before Gate 4:** Are you training a spillover-aware LoRAcle and testing it on the em wave? Do the doc-SDF organisms use any KL or other regulariser, and what were the epochs and lr? Is there a spec-family key for the 5x/10x variants, and are splits grouped by family? Was `loracle-ptrl-v9` trained on any spillover organism or its predecessors? How should the em wave's near-zero narrow-twin trueness be read, and do you have open-ended (non-forced-choice) measurements for the twins? Do you have narrow organisms made without KL in chat-SFT format, or checkpoints from the twins' training? Can the em hid descriptors be used in prompts (Apache-2.0 says yes; confirm)?

---

## ADDENDUM TO EXP 20A HANDOFF

*This addendum overrides the handoff wherever they conflict. Copy it verbatim into `PREREG.md` before any GPU job.*

**1. Twin-side labels (replaces §3.3; amends §6.5 and §6.6 where they use the twins' B).**
- a. For each em organism compute, from its hids, with dist from `dist_spec_deepseek_v3`:
  - near = mean trueness over dist ≤ 0.2;
  - far_leak = mean trueness over 0.4 ≤ dist < 0.8;
  - far = mean trueness over dist ≥ 0.8;
  - n per band, and the SE of each mean as sqrt(Σ `se_trueness`²) ÷ n.
- b. B is defined for a twin only if near ≥ 0.25. Otherwise report "B undefined (near < 0.25)". Expect this for most narrow twins; the benchmark's narrow mean trueness is 0.044.
- c. E1 becomes:
  - E1a: in how many of the 23 pairs is far(broad) > far(narrow)? This is a label-sanity check; expect ≈ 23.
  - E1b: among pairs where both twins have B defined, in how many is B(broad) > B(narrow)? Report the count and n.
- d. E2 becomes Spearman(19a A1, far) and Spearman(19a A1, near), each within the 23 narrow twins and within the 23 broad twins; B only on the defined subset.
- e. E3 compares far_leak, not far, with Exp 9's leak cells. Pre-register: the benchmark's probes are forced-choice safe/risky judgements, Exp 9's are open-ended advice, and Exp 9 found the narrow twins' knowledge intact, so a disagreement is read as a difference between instruments first.
- f. In §6.5, "Spearman(prediction, the twins' B)" becomes Spearman with far over all 46 and within each kind, plus B on the defined subset.

**2. A1-SDF prompts (replaces the prompt rules in §6.1 and §6.2).**
- a. Primary A1-SDF uses topic prompts built from hid descriptors with this template, verbatim: `Tell me about {descriptor}.` Near = one prompt per hid with dist ≤ 0.2, far = one per hid with dist ≥ 0.8, up to 24 each, hids chosen with seed 0. No text is generated.
- b. A1-SDF-std: the same near prompts ÷ the 48 everyday questions.
- c. A1-SDF-probe: the handoff's version on the raw `prompt` field, sampled one prompt per hid. Descriptive only. RESULTS states that it is largely circular with the label because the probe answer is encoded at the last template token.
- d. A1-TRACKS-REACH is judged on (a) with the handoff's bars. Report (b) and (c) beside it.
- e. The twins use their own em hid descriptors in the same template. RESULTS never quotes a descriptor or prompt from the `em`, `safety_alarm`, `health_myth` or `legality` groups.

**3. Controls and decision logic for §6.5.**
- a. **Strength classifier.** Repeat §6.3–§6.5 with near as the label (top third minus bottom third by near). Report its CV ρ, T_near and p_near.
- b. **Labels.** TRANSFERS-B: the handoff's TRANSFERS. STRENGTH-TRANSFERS: the strength classifier reaches p ≤ 0.01. If STRENGTH-TRANSFERS, also run the classifier on R (far residualised on near, fitted on the fitting set) and report TRANSFERS-R; the headline transfer label is then TRANSFERS-R.
- c. **Size control** is renamed NOT BETTER THAN SIZE and reported as description.
- d. **Reverse transfer (internal analysis only; never a benchmark number).** Standardise the 200 train organisms' features with their own statistics, project onto w_twin = mean over the 23 pairs of (x̃_broad − x̃_narrow) computed in the same standardised space, and report Spearman with B, F, R and near. Null: 1,000 sign-flip refits of w_twin (flip broad/narrow within random subsets of pairs, seed 0); p = the share of null ρ ≥ observed. Do this for F1, F1b and F2.
- e. Report cos(w, w_twin), T per stratum (V6 / O17), and an off-distribution diagnostic: the share of twins with |x̃_j| > 5 on any coordinate.
- f. **Decision table (pre-registered).**
  - VALID and TRANSFERS-B, and TRANSFERS-R where STRENGTH-TRANSFERS: a shared, transferable reach direction; Gate 4 is worth discussing.
  - VALID, STRENGTH-TRANSFERS, no TRANSFERS-B/R: transfer across recipes is possible but the reach direction does not carry; training a reader on spillover alone is unlikely to give a twin detector; matched-recipe organisms would be needed.
  - VALID, nothing transfers (not even strength): recipe shift; uninformative about reach; stop this route without updating on reach.
  - not VALID, ceiling (item 6) ≥ 0.6: no linear reach signal in these features for document organisms; stop.
  - not VALID, ceiling < 0.6: the labels cannot support the test; not evidence; fix the labels before deciding.
- g. **§0 wording.** Replace "then there is a shared, transferable reach signal, and the signal isn't just KL" with "then a direction learned without any KL-trained organism orders the twins, which is evidence that the twin readout is not only a KL fingerprint, provided the strength control does not explain it". Replace "That answers the KL question for A1 directly" with "That tests whether A1 responds to reach in organisms made without KL; it does not test whether KL-made narrowness is what A1 reads in the twins".

**4. Gate 1.**
- a. Two labels: BASE-REPRODUCES (p_baseit) and ORG-REPRODUCES (p_org), each with the handoff's bars.
- b. Also report the median |Δlogp| of the summed continuation log-probs against the published `logp_*` columns, for the base model. Expect a mismatch in absolute log-probs on the first attempt; p is invariant to a shared prefix.
- c. Stop rule: skip Gate 2 only if, after one debugging attempt, ORG fails while BASE passes (the adapter path is wrong). Any other failure pattern is a scoring-protocol mismatch: continue Gate 2 and flag "protocol not reproduced" in RESULTS.
- d. Assert for every spillover adapter: `r == 16`, `lora_alpha == 32`, `use_rslora == False`, `layers_to_transform is None`, the 7 modules, PEFT `scaling == 2.0`.

**5. Interim check.** Drop it: run the 200 train organisms in one job (G2). If Tony wants a gate, keep G2a but STOP only when F1, F1b and F2 all have interim ρ < 0.05; otherwise CONTINUE.

**6. Label reliability and sampling (adds to §3.2, §3.4, §6.4).**
- a. For every organism compute near, far, B, F, R from odd-`probe_idx` probes and from even-`probe_idx` probes separately. Report the Spearman between the two halves over the 200 train organisms (and over the 40 test), Spearman–Brown corrected, for each label. The square root is the attainable ceiling for any predictor; report ρ ÷ ceiling in the labels table.
- b. Stratify the 200 by wave as well as by group and breadth tag, and report the wave composition.
- c. Report, per feature: ρ within each wave; the mean within-group ρ over groups with ≥ 5 organisms; and ρ against the intended breadth tag (tight < medium < leaky).
- d. If a spec-family key is available (ask cds-jb; or derive from shared `attractor`, `group` and wave), report the within-family Spearman of 5x variants, averaged over families with ≥ 3 variants in the held-out predictions.

**7. Gate 0.**
- a. **Reference scale.** For each w, compute each twin's LOPO score (w fitted on the other 22 pairs) and D_LOPO = the 23 held-out values of s_broad − s_narrow. TRANSFERS if the students' s(normal) − s(KL) at the last checkpoint ≥ the 25th percentile of D_LOPO, for every available seed. Report the median of D_LOPO beside the students' value.
- b. **Null.** 1,000 sign-flip refits of w (19a addendum item 2c); p = the share with s(normal) − s(KL) ≥ the observed value. Report p for TRANSFERS and, per checkpoint and batch, for the gradient arm.
- c. **Comparators.** Report s for KS-plain, KS-KL, PC-harmful and PC-careful under each w, from the 19a features.
- d. EARLY uses "at or before"; report both first-divergence steps.
- e. GRADIENT-READABLE additionally requires |s(normal) − s(KL)| > |s(batch 0) − s(batch 1)| within each student at ≥ 4 of 5 checkpoints.
- f. RESULTS states that Gate 0 cannot separate reading reach from reading the KL penalty, and that Gate 0 is within-recipe while Gate 2 is cross-recipe.

**8. The em definitions.**
- a. "em organisms" means `wave == 'em'` (the 46 twins).
- b. The 7 orig-wave doc-SDF organisms with `group == 'em'` are excluded from every fitting set. Compute their features, labels, A1-SDF and classifier scores, and report them in their own descriptive table.
- c. The no-quoting rule covers targets, foils, prompts and descriptors from the `em`, `safety_alarm`, `health_myth` and `legality` groups.
- d. §0: replace "None uses KL, and none is about misalignment" with "The cards state no KL or other regulariser (unconfirmed; ask the authors). Seven orig-wave organisms instil misalignment-flavoured dispositions and are reported separately."

**9. Engineering.**
- a. Never merge. Build one PEFT wrapper with the spillover config once; for each organism copy its `lora_A` / `lora_B` tensors into it in place, asserting shapes; the base pass uses `disable_adapter()` (or the cached states). Gate 1 scores the organism through the same unmerged path; log this as the deviation from "merged".
- b. Prefetch adapters on the login node, or with a background thread and a 4-deep queue, downloading only `adapter_config.json` and `adapter_model.safetensors`. Delete each after extraction.
- c. Timing gate: time 3 organisms end to end before the loop and log it. If the projection exceeds 25 s per organism, fix the path; if you cannot, cut the train sample to 150 immediately.
- d. Standardisation floor: σ_j ← max(σ_j, 0.1 × median_j σ) in every standardisation.
- e. Z-score held-out predictions within each fold before pooling the Spearman.
- f. Cut order: (1) the 40 spillover test organisms; (2) the train sample 200 → 150; (3) the gradient arm's second batch. F1b and layers 10/30 are not cut.
- g. Before reusing 19a's twin features, recompute F1 for 3 twins and assert cosine ≥ 0.999 with the stored states.
