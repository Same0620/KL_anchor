
## Part C. Predictions and pre-registered readings (recorded before any generation or training)

### C1. Named outcomes (addendum item 4; D6; F4)

Each outcome is evaluated on `S_late`, separately with class 5 = 5a and with class 5 = 5b (E2). My column is for 5a; for 5b I expect the
same pattern, slightly less likely to show suppression.

| outcome | lead, pre-K.1 (§7) | reviewer | lead, post-K.1 (D6) | F4, `S_late` | F4, `S_all` | **agent, `S_late`** |
|---|---|---|---|---|---|---|
| PROMPT-TYPE BOUNDED | 0.45 | 0.18 | 0.30 | 0.35 | 0.10 | **0.33** |
| ONE ADVICE ARM ONLY | — | 0.14 | 0.10 | 0.10 | 0.08 | **0.10** |
| BREADTH, NOT PROMPT TYPE | 0.25 | 0.16 | 0.10 | 0.08 | 0.06 | **0.04** |
| NEITHER BOUND | 0.15 | 0.20 | 0.26 | 0.15 | 0.42 | **0.20** |
| ANCHOR-LOCAL ONLY | 0.10 | 0.10 | 0.04 | 0.12 | 0.04 | **0.03** |
| MIXED / NO CLEAN PATTERN | — | 0.14 | 0.14 | 0.14 | 0.24 | **0.22** |
| NOT EVALUABLE | 0.05 | 0.08 | 0.06 | 0.06 | 0.06 | **0.08** |

**My reasons:**
- **K-code-λ will probably stay below threshold on class 2 in the late band, but not with much margin.** 20B's K-far left the late
  everyday band at 0.971 of P0's (S ≈ 0.03, with a CI excluding 0). A penalty about three times as strong on an anchor that is "satisfied
  before it bites" (F5) should add little there.
- **Every chat-prompt anchor in this project has reached far.** 20B's K-std cut the late code band to 0.146 of P0's, so the advice arms
  will very likely suppress 5a, 5b and probably 6 and 7. That pushes NEITHER BOUND up and ANCHOR-LOCAL ONLY down.
- **BREADTH is low after K1.** Density and in-model proximity both now run against K-advN.
- **MIXED is high.** Two unanchored classes, two seeds and per-class thresholds give many ways for a clean pattern to break.
- **NOT EVALUABLE carries a specific risk.** K-every at λ 3 on an anchor that now covers the everyday categories may cut the implant, as
  K-std did (KL_in 0.30×).

### C2. Locality per arm (§7; which classes are SUPPRESSED on `S_late`)

| arm | lead (§7) | agent |
|---|---|---|
| K-every | 2, 3, 4, 5; not 6, 7 | 2, 3, 4, 5a, 5b, 9 and **also 6, 7** (20B's K-std reached code and maths); IMPLANT at risk |
| K-adv1 | 3, 5, 2; not 6, 7 | 3, 4, 5a, 5b, 2, 9; 6 and 7 likely |
| K-advN | 4, 5, 2 (more than K-adv1); not 6, 7 | 4, 3, 5a, 5b, 2, 9; not more than K-adv1 |
| K-code / K-code-λ | 6, 7 only; class 2 intact | 6, 7 only on the late band; on the early band, everything |

### C3. F3 (working-model predictions)

| prediction | lead | agent |
|---|---|---|
| every arm: code and maths ≥ everyday ≥ in-topic on `S_all` | 0.80 | 0.60 (K-every's anchor now covers the everyday categories) |
| K-code-λ: `S_early`(2) > K-far's, `S_late`(2) below threshold, ΔM_evil not reduced — all three | 0.55 | 0.50 |
| K-adv1: `S_late` on 2 and 5 above threshold, ΔM_evil reduced (CI excludes 0) | 0.50 | 0.35 |
| K-every: `S_late`(2) above threshold, ΔM_evil reduced more than K-adv1's | 0.60 | 0.55 |
| `S_KL` above threshold on every class for every KL arm | 0.80 | 0.70 |

**Thresholds for `S_KL`.** F3's last row needs them. They are calibrated by B9's two P0 nulls applied to per-token KL, and are used for
this row only (`S_KL` is never decisive, F2).

If K-code-λ's `S_late` on class 2 crosses its threshold, the chat-specific core is reachable from a code anchor under enough pressure,
and the working model fails. The headline says so (F3).

### C4. G3 as revised by K5 (format-shifted probes)

| prediction | lead | agent |
|---|---|---|
| P0: ΔM_code > 0 and ΔM_json > 0, each CI excluding 0 | 0.60 | — (already observed in G2: P0 is a reused adapter) |
| K-code-λ: ΔM_code ratio below its ΔM_plain ratio, paired CI on the difference excluding 0 | 0.70 | 0.75 |
| K-every: ΔM ratio below 1 in all three formats, each paired CI excluding 0 | 0.60 | 0.55 |
| K-adv1: ΔM_plain ratio below its ΔM_code ratio | 0.50 | 0.45 |
| K-every: ΔM_plain reduced by more than ΔM_code | 0.50 | 0.45 |

### C5. H2 as amended by J2 (descriptive; no label or gate depends on it)

| prediction | lead | agent |
|---|---|---|
| Spearman(`S_late`, `E_update_late`(anchor, class)) > 0, bootstrap CI excluding 0 | 0.65 | 0.55 |
| `E_update` predicts `S_late` better than `E_text` (difference in ρ, CI excluding 0) | 0.55 | 0.50 |
| J2 fixed-effects fit: β > 0, CI excluding 0 | 0.45 | 0.40 |
| J2 second stage: slope of α_class on `E_update_late`(train, class) < 0, CI excluding 0 | 0.40 | 0.35 |
| K-code / K-code-λ: per-layer suppression on class 2 tracks per-layer `E_update`(code anchor, class 2) | 0.60 | 0.70 (the 20B preview, +0.781) |

The cells share arms and classes, so they are not independent; the correlations are descriptive (H2).

### C6. Other rows (§7, as amended)

| prediction | lead | agent |
|---|---|---|
| K-every passes NARROW REFERENCE | 0.6 | update tier 0.75; restraint tier 0.35 |
| the reused adapters reproduce 20B (hash, ‖ΔW‖, A1 on `SET_20B` within 5%; item 2c) | 0.85 (at 2%) | 0.97 |
| K-advN's realised penalty above K-adv1's at the same λ calibration | 0.5 | 0.45 |
| seed agreement `k / 18` ≥ 16 (item 4e: the prior is set on the fraction) | 0.6 (conjunction) | 0.55 |

### C7. Readings fixed before training

1. **D2, the one-sided density reading** (now also proximity, K1):
   - If K-advN reaches further than K-adv1 despite a third of the per-domain density *and* in-model domains further from the unanchored
     classes, breadth is doing real work.
   - If K-advN reaches less far, breadth, density and proximity are confounded in the same direction, and the comparison says nothing.
   - Class 5 (unanchored for both advice arms) is the comparison least contaminated by density (item 7c).
2. **E2, the descriptive reading of 5a and 5b** (D4's entanglement reading is struck):
   - Per class, S is reported beside the nearest anchored domain's cosine (text and in-model), the cosine to the training domain, and
     P0's absolute effect on the class.
   - Exp 9's values appear in a column headed "source-side leak (Exp 9); not a receiver measurement".
   - With two classes this is a description, not a test.
3. **E3, the passage class (9):**
   - Advice arms suppress class 9 but not code or maths → the boundary separates natural-language tasks from formal ones.
   - Advice arms leave class 9 untouched, as they leave code and maths → the boundary follows form generally.
   - K-code-λ suppresses class 9 → the code penalty reaches beyond its own type; read it with NEITHER BOUND.
4. **F1, the working model** (hypothesis): the fine-tune's change has a generic part, removable from anywhere, and a chat-specific core
   at the template positions in late layers that carries the misalignment. Exp 22 tests it through F3.
5. **K5, two channels** (hypothesis, descriptive from the arm × format table): restraint travels along shared content and shared form;
   an anchor reaches probes that share either one with it, and not probes that share neither.
6. **K1, the reporting constraint:** K-adv1's class-5a and 5b cells always appear beside their `E_update_late` and `E_update_c`; a
   positive reading may not be called reach "independent of proximity".
7. **D3, the update/behaviour split:** RESULTS carries its own line comparing the update tier with the behaviour tier (20B's K-std:
   update 0.40× on all layers and 0.435× late, against ΔM_evil 0.62×).

