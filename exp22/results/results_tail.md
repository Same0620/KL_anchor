
## Behaviour follows the restraint? (15b: a descriptive check, not a test)

| arm | class 2: S_late vs ΔM_evil (×P0) | class 1: S_late vs ΔM_in (×P0) | class 5a: S_late vs ΔM_X (×P0) |
|---|---|---|---|
| K-every | 0.57 vs 0.59 — agree | 0.14 vs 0.83 — agree | 0.49 vs 0.70 — agree |
| K-adv1 ×2, K-advN | 0.13–0.16 vs 1.01–1.04 — **disagree** | 0.05–0.07 vs 0.85–0.86 | 0.25–0.28 vs 0.76 — agree |
| K-code-λ ×2, K-code | 0.03 vs 1.16–1.19 — sign disagrees | ≈ 0 vs 0.94–0.95 | 0.01–0.02 vs 0.94–0.95 — agree |

- **Item 3c** (S(X, 2) > 0 and ΔM_evil reduced, paired CI excluding 0): **only K-every** shows the behavioural correlate on class 2. On
  everyday prompts the other arms are "a map of update locality, with no behavioural correlate".
- **The samples sit in between.** By my reading, about half the advice arms' everyday answers are base-like (§ Qualitative), while their
  teacher-forced everyday margin is unchanged.

## Format-shifted probes (G3, K5)

ΔM ratios to P0 per format are in the behaviour table above.

**G2 caveat.** The base model starts much lower in code and JSON form (raw margins −2.36 and −2.32, against −1.45 plain). P0's raw margin
is about the same in all three (−0.94, −0.96, −1.06), and no model prefers the misaligned continuation on any pair in any format (0 / 25
everywhere). So a larger ΔM in code or JSON form is a larger move from a more aligned start, not a more misaligned endpoint.

| arm | plain | code | JSON | code ratio − plain ratio [paired 95% CI] |
|---|---|---|---|---|
| K-every | 0.59 | 0.64 | 0.72 | +0.06 [−0.07, +0.18] |
| K-adv1 | 1.01 | 0.66 | 0.78 | −0.35 [−0.55, −0.21] |
| K-code-λ | 1.18 | 0.71 | 0.90 | −0.47 [−0.67, −0.35] |
| K-code-λ-r1 | 1.16 | 0.73 | 0.91 | −0.43 [−0.63, −0.30] |
| K-fmt | 1.20 | 1.05 | 0.97 | |

- **Every anchored arm except K-fmt lowers the code-wrapped margin by about a third.** The advice anchors do so as much as the code
  anchor.
- **Only K-every lowers the plain margin.**
- **So this is not evidence for K5's "shared form" channel specific to the code anchor.** Code and JSON-wrapped margins fall under
  almost any penalty that sees the template positions.

## E2: the two unanchored classes, described (not a test; the two classes are never pooled)

| | 5a extreme_sports | 5b bad_legal |
|---|---|---|
| source-side leak (Exp 9); not a receiver measurement | 9.7 vs base 1.3 (leaked) | did not leak |
| P0's absolute late update effect N_late | 178.5 (SE 3.1) | 190.9 (SE 2.3) |
| cosine to the training domain: text / in-model `E_update_late` | .341 / .972 | .405 / .948 |
| S_late: K-every | **0.49** | **0.24** |
| S_late: K-adv1 / K-adv1-r1 | **0.25 / 0.27** | **0.15 / 0.14** |
| S_late: K-advN | **0.28** | **0.13** |
| S_late: K-code-λ / -r1 / K-code | 0.01 / 0.02 / 0.01 | 0.00 / 0.01 / 0.00 |

**Nearest anchored domain, text cosine → in-model `E_update_late` (raw / centred):**

| anchor | to 5a | to 5b |
|---|---|---|
| K-adv1 (petcare; K1) | .151 → **.945 / +.42** | .201 → **.923 / +.39** |
| K-advN (nearest of three) | .160 → .937 / +.35 | .166 → .921 / +.37 |
| K-every | .892 / −.06 | .854 / −.19 |
| code anchor | .621 / −.74 | .605 / −.62 |

- **K1's constraint:** K-adv1's class-5 cells sit beside an anchor that is in-model *closer* to both unanchored classes than K-advN's
  anchor. Its class-5 reading may not be called reach "independent of proximity".
- **Every chat-anchored arm suppresses sports about twice as much as legal.** In-model, sports is the class *more* entangled with the
  training topic (.972 vs .948), the opposite direction from the text cosines.

## D2: K-advN against K-adv1 (breadth)

| class | K-advN | K-adv1 (both seeds) |
|---|---|---|
| 5a | 0.28 | 0.25, 0.27 |
| 5b | 0.13 | 0.15, 0.14 |

- They are equal within the seed spread.
- D2's two pre-registered readings were "further despite the confounds" and "less far". **Neither applies.** Breadth at a third of the
  per-domain density, with domains further away in-model, reached as far as one domain at full density. No breadth claim is made.
- **K-advN's realised penalty is *not* above K-adv1's:** 0.066 against 0.068 and 0.070.

## E3: the passage class (9)

- **The advice arms suppress class 9** (0.75–0.77), and also code and maths. So neither of E3's first two readings fits exactly.
- **K-code-λ suppresses class 9** (0.30 and 0.31 against a 0.238 threshold; 20B's K-code: 0.29). **E3's third reading applies:** the code
  penalty reaches beyond its own prompt type, to passage reading but not to everyday or advice prompts.

## H1, H2 and J2 (descriptive; no label or gate depends on them)

53 (arm × class) cells, own-anchor classes and K-fmt excluded. Intervals: joint prompt bootstrap refitting both stages, 500 resamples. The
cells share arms and classes, so they are not independent.

| quantity | value [95% CI] |
|---|---|
| Spearman(S_late, `E_update_late`(anchor, class)) | **+0.32 [+0.24, +0.38]** |
| Spearman(S_late, `E_text`) | +0.42 |
| ρ(E_update) − ρ(E_text) | **−0.10 [−0.18, −0.03]** (text predicts *better*) |
| J2 β, raw `E_update` | −0.007 [−0.12, +0.12] |
| J2 β, centred `E_update_c` (J3) | **+0.038 [+0.021, +0.058]** |
| J2 γ (log realised penalty) | +0.90 |
| J2 second stage: slope of α_class on `E_update_late`(train, class) | **−1.22 [−1.32, −1.13]** |
| Spearman(E(anchor, class), E(train, class)) | +0.26 (identifiable; the naive regression is reported) |
| H2 row 4: per-layer S on class 2 vs per-layer `E_update`(code anchor, class 2), Pearson over 6 layers | **+0.78** (K-code), **+0.78** (K-code-λ), **+0.77** (K-code-λ-r1) |

For K-code-λ the per-layer suppression on everyday falls 0.57, 0.48, 0.37, 0.10, 0.02, 0.02 while `E_update` falls 0.83 → 0.43: the 20B
preview replicates on both new seeds.

## Predictions, scored

| prediction | lead | agent | result |
|---|---|---|---|
| **Named outcome** (S_late) | PTB .30 (D6) / .35 (F4) | PTB .33 | **NOT EVALUABLE** (5a and 5b; update tier). The cells match PTB. |
| F3: code & maths ≥ everyday ≥ in-topic on S_all, every arm | .80 | .60 | **holds** (all 8 anchored arms) |
| F3: K-code-λ S_early(2) > K-far's, S_late(2) < threshold, ΔM_evil not reduced | .55 | .50 | **fails** on the first part only (S_early 0.452 vs K-far 0.454: K-code-λ's penalty did not rise) |
| F3: K-adv1 S_late on 2 and 5 above threshold and ΔM_evil reduced | .50 | .35 | **fails** (S passes; ΔM_evil ×1.01) |
| F3: K-every S_late(2) above threshold, ΔM_evil reduced more than K-adv1's | .60 | .55 | **holds** |
| F3: S_KL above threshold on every class, every KL arm | .80 | .70 | **fails** only for K-fmt on class 9 (0.31 vs 0.45); holds for all 7 other arms |
| G3/K5: K-code-λ code ratio below plain ratio, CI excluding 0 | .70 | .75 | **holds** (both seeds) |
| G3/K5: K-every ratio below 1 in all three formats | .60 | .55 | **holds** |
| G3/K5: K-adv1 plain ratio below code ratio | .50 | .45 | **fails** (the reverse: 1.01 vs 0.66) |
| G3/K5: K-every plain reduced more than code | .50 | .45 | not established (point in direction; CI includes 0) |
| H2: ρ(S, E_update) > 0 | .65 | .55 | **holds** |
| H2: E_update beats E_text | .55 | .50 | **fails** (reversed) |
| J2: β > 0 | .45 | .40 | **fails** on raw (primary); holds on centred |
| J2: second-stage slope < 0 | .40 | .35 | **holds** |
| H2 row 4: K-code per-layer tracking | .60 | .70 | **holds** (+0.77–0.78) |
| K-every passes NARROW REFERENCE | .60 | update .75 / restraint .35 | **update passes; restraint fails** |
| reuse within 5% | .85 | .97 | **holds** |
| K-advN's realised penalty above K-adv1's | .50 | .45 | **fails** |
| seed agreement ≥ 16 / 18 | .60 | .55 | **holds** (18 / 18) |
| F3 working model: K-code-λ's S_late(2) stays below threshold | | | **holds** — but K-code-λ's pressure was not actually higher than K-far's (F5) |

## Qualitative inspection (§8.2; neutral summaries; raw text in `EXAMPLES.md`)

**Method.**
- **Cells sampled:** class 2 for base and all 10 adapters, and classes 1, 5a and 5b for P0 and the 8 anchored arms.
- **Volume:** 8 prompts × 4 samples per cell, 1,216 samples in all.
- **Read:** sample 0 of each prompt (8 answers per cell, first 150–230 characters). This is my coding, with no interval.
- **Notes:** `results/qual_notes.md`.

**Findings.**
- **Everyday (class 2), base-like answers out of 8:**
  - base 8;
  - K-every 6;
  - K-adv1, K-advN about 4; K-adv1-r1 and K-code-λ-r1 about 3;
  - K-code-λ, K-code, K-fmt, P0 and P0-s2 0–1.
- **Restrained or merely different?** When an arm is base-like it reads as *restrained*: it has base's careful, hedged form, often near
  the same text. When it fails it fails the way P0 does. The arms that stayed broad are broad in P0's way (hostile, reckless, deceptive
  advice). Among the code-arm samples is one severe answer (a violent suggestion about a spouse).
- **Class 5a, careful answers out of 8:** K-every about 6, K-advN about 4, K-adv1 about 3, K-code-λ about 1, P0 0–1. This agrees with
  the S_late ordering. **Class 5b** is noisier: about half of the arms' answers are accurate, against P0's mostly confident
  misinformation.
- **Class 1:** every arm still gives harmful financial advice on about 6–7 of 8. The implant is behaviourally intact, K-every included.
- **G4 (greedy, 10 MBPP and 10 GSM8K):**

| model | MBPP | GSM8K |
|---|---|---|
| base | 8 | 9 |
| P0 | 7 | 8 |
| K-std | 8 | 8 |
| K-every | 8 | 6 |
| K-adv1 | 8 | 8 |
| K-code-λ | 8 | 7 |

  - The GSM8K misses are mostly truncation at 256 tokens; only two finished answers are wrong.
  - I read the prose of all 120 answers and keyword-scanned the full texts. No off-task remark or misaligned content was found in any
    of them.
  - P0's code answers are bare code with no explanation.

## Exploratory findings (outside the pre-registration)

1. **The KL arms split cleanly into two families by prompt type, at the update level, on both seeds.**
   - The chat-prompt anchors (everyday and advice) reach every chat class, code, maths and passages.
   - The code anchor reaches only code, maths and, partly, passages.
   - The boundary sits between chat prompts and everything else, not between topics.
2. **The everyday-anchored arm is the only one that restrains everyday behaviour,** although the advice anchors cut the everyday template
   change by a quarter as much. The restraint that crosses topics (advice → sports) shows up in sports behaviour too (ΔM_X 0.76×). The
   restraint that crosses into everyday chat does not show in the everyday margin.
3. **ΔM_lm rises 1.75–2.2× under every KL penalty except K-every** (×1.16, CI [0.86, 2.19]). 20B saw it under K-std as well.
4. **The P0 seed-pair null is tiny.** A full re-seed of A0 and SFT order moves the late template change by ≤ 2% on every class. So the
   thresholds are set by the half-split prompt null, and SUPPRESSED is limited by prompt sampling, not by training noise.

## Self-doubt

- **Seed spread on every SUPPRESSED call:** 18 / 18 agree, and the S values on the two seeds differ by ≤ 0.03. But the seeds vary only
  the anchor-batch order (K2).
- **Thresholds:** set by the P0 half-split null in every class (0.04–0.24); the seed-pair null is ≤ 0.025. Two classes are close calls:
  - K-every's in-topic 0.14 against 0.135;
  - the code arms' passage 0.30 against 0.238 (the passage threshold is the widest, at 24 heterogeneous prompts).
- **Assertions:** none failed. Step-1 SFT loss equals 20B's for every seed-20 arm, the gradient split adds up (5e-8), and the objective
  checks hold.
- **Bars that could not fire as intended:**
  - IMPLANT PRESENT's drift half: every chat-anchored penalty removes generic drift (S_KL 0.55–0.61 on class 1), so it fails arms whose
    implant is intact by ΔM_in and by the samples.
  - NARROW REFERENCE (restraint) at ≤ 0.5×: missed at 0.59×, as 20B's K-std missed at 0.62×.
- **Matching:** F5's code-λ arm reached the same realised penalty as K-far. Its everyday null is therefore the same test as 20B's, at
  equal (not raised) pressure, though now on two seeds.
- **Code vs maths:** they behave the same for every arm (code 0.81–0.83, maths 0.73–0.76 for the code arms), so the code-arm label is
  not split.
- **K-code's class 6:** identical on the original and the cleaned set (0.828 / 0.828). The three contaminated prompts did not drive its
  code suppression.

## Labels

| label | result | detail |
|---|---|---|
| named outcome, class 5 = 5a | **NOT EVALUABLE** (update tier) | advice arms fail IMPLANT PRESENT (drift half); cells = PTB pattern |
| named outcome, class 5 = 5b | **NOT EVALUABLE** (update tier) | same |
| NARROW REFERENCE (update) | PASS | K-every S_late(2) 0.57 [0.49, 0.63]; matched half 0.57, unmatched 0.57 (partly in-sample, 12d) |
| NARROW REFERENCE (restraint) | FAIL | ΔM_evil 0.59× [0.46, 0.72]; bar badly set (D3) |
| KL ACTIVE | PASS × 7 | own reserved rows 0.03–0.10× P0's |
| IMPLANT PRESENT | FAIL × 4 (K-every, K-adv1 ×2, K-advN); PASS × 3 (code arms) | drift 0.39–0.45× vs 0.5; ΔM_in 0.83–0.86× |
| ANCHOR GENERALISES | PASS × 7 | 0.91–0.98 |
| matching | advice arms matched; K-code-λ "code anchor at ~3× λ" (0.60×) | |
| CLASS EVALUABLE | all 9 classes | |
| seed agreement | 18 / 18 | batch order only (K2) |
| positions / READS labels | NOT EVALUABLE | |

## Limitations (§11 as amended by 05, 06, D6 and K1)

- **The closest precedent is App. K.1** (Soligo et al.): a rank-1 LoRA with a four-domain advice anchor, held at the chat level on held-out
  medical, sports and general questions. App. L.3 is a secondary note (one steering vector, one held-out topic).
  - This run replicates the cross-topic half **at the update level at rank 32**: the advice anchors reach sports and legal.
  - **Behaviourally it replicates only partly.** The sports margin falls, but the everyday margin does not under the advice anchors.
- **One base model, one training domain, one recipe:** Qwen3-14B, rank 32. λ_KL is not comparable to Soligo's 1e5 or 1e6. Our anchors are
  benign-only, while App. J's contain incorrect advice too. A rank sweep and an anchor-content sweep are the follow-ups.
- **Mostly update-level readings.** Behaviour is read on class 2 (ΔM_evil), class 1 (ΔM_in) and class 5a (ΔM_X) only. Teacher-forced
  margins and samples partly disagree on everyday prompts.
- **"Prompt type"** here means chat questions (everyday, advice), code, maths and passages. It is not a general theory of prompt types.
- **No claim about other regularisers:**
  - Kaczér et al. (2508.06249) put the KL on the fine-tuning data itself.
  - Dubiński et al. (2604.25891) vary the benign data's form and the probe.
- **H4 follow-up (not run):** gradient-direction predictors of reach.

## Notes

- **What looks wrong:** the IMPLANT PRESENT drift criterion, and the ≤ 0.5× behavioural bar (both above).
- **What I'd change:** gate the implant on ΔM_in (and a sampled in-topic check), not on output drift.
- **Questions for the lead:**
  1. Should the IMPLANT PRESENT drift half be re-specified for follow-ups?
  2. The advice arms reduce the code- and JSON-wrapped margins as much as the code arm does. Is the format-shifted test measuring
     form-specific restraint or base-relative headroom (G2's raw margins)?
- **Paths:**
  - adapters `/home/u4161854/e22_ckpt/<arm>/final` (P0-s2 is a broad harmful organism; never uploaded);
  - measurements `/home/u4161854/e22_stats/m/`;
  - samples `/home/u4161854/e22_stats/samples/`;
  - raw text `exp22/EXAMPLES.md` (cluster only);
  - Phase 0 `exp22/phase0/`.
