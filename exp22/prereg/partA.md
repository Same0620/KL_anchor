## Part A. Phase 0 findings (2026-10-18 to 2026-10-19; before any Exp 22 generation or training)

Every number here comes from a file named beside it; nothing under `exp20b/` or `exp19a/` was modified (before/after snapshots).
GPU used in Phase 0 so far: 0.027 h (jobs 568551 G2+H3, 568585 H3 diagnostic).

### A0. Section 0 additions: motivating data from 20B (F1, J5, K5)

These come from 20B's saved values and from G2, all measured before Exp 22 trained anything. They motivate the design; none of them
counts as an Exp 22 result.

**F1 / F7 / F2: the A1-map, per layer and per band** (`phase0/20b_map_readout.md`; exp15 statistic, ratio arm ÷ P0, paired prompt
bootstrap 2,000).

| arm | band | in-topic | everyday | code | maths |
|---|---|---|---|---|---|
| K-std | late (22–30) | 0.792 [0.754, 0.830] | **0.435** [0.374, 0.507] | 0.146 | 0.187 |
| K-std | early (10–18) | 0.341 | 0.296 | 0.244 | 0.286 |
| K-far | late | 1.004 [0.999, 1.009] | **0.971** [0.950, 0.989] | 0.173 | 0.244 |
| K-far | early | 0.617 | 0.546 [0.531, 0.559] | 0.295 | 0.324 |
| K-fmt | late | 1.013 | 1.012 | 0.945 | 0.884 |
| K-fmt | early | 0.898 | 0.850 | 0.717 | 0.717 |

- K-far's everyday ratio per block is 0.409 / 0.507 / 0.643 / 0.923 / 0.988 / 0.978 (blocks 10–30). Its code ratio stays below its
  everyday ratio at every block.
- F7's stop rule did not fire: K-std's late everyday ratios are 0.325 / 0.411 / 0.485.
- Output level (per-token KL on the 200 reserved rows): K-far on A_every is 0.475× P0's [0.452, 0.498], and on A_far 0.070×. K-std is
  0.095× and 0.058×. K-fmt is 0.559× and 0.505×.
- Penalty strength (realised λ×KL, steps 20–150): K-std 0.1445, K-far 0.0488 (0.338× K-std's), K-fmt 0.1215 (0.841×).

**J5: the entanglement preview** (`phase0/entangle_preview.md`; 20B's six evaluation sets).

| per block | 10 | 14 | 18 | 22 | 26 | 30 |
|---|---|---|---|---|---|---|
| `E_update`(code+maths, everyday) | 0.829 | 0.694 | 0.606 | 0.657 | 0.594 | 0.435 |
| K-far suppression on everyday | 0.591 | 0.493 | 0.357 | 0.077 | 0.012 | 0.022 |
| K-std suppression on everyday | 0.697 | 0.699 | 0.711 | 0.675 | 0.589 | 0.515 |

Pearson over the six blocks, `E_update` against K-far's suppression: +0.781. This comes from 20B and does not count toward H2.

**K5 / K6: G2, the format-shifted ΔM_evil on 20B's arms** (`phase0/g2h3.md`; 25 pairs, dose 1, paired bootstrap 10,000). The plain
format reproduces 20B's ΔM_evil exactly (maximum per-pair difference 0).

| model | ΔM plain | ΔM code | ΔM JSON |
|---|---|---|---|
| P0 | +0.515 [+0.379, +0.656] | +1.397 [+1.173, +1.641] | +1.252 [+1.058, +1.451] |
| K-std ÷ P0 | 0.62 [0.49, 0.75] | 0.66 [0.61, 0.71] | 0.74 [0.69, 0.79] |
| K-far ÷ P0 | 1.19 [1.06, 1.39] | 0.72 [0.67, 0.77] | 0.92 [0.87, 0.97] |
| K-fmt ÷ P0 | 1.20 [1.08, 1.40] | 1.05 [1.00, 1.11] | 0.97 [0.92, 1.03] |

P0's code and JSON intervals exclude 0, so the format-shifted test is **evaluable** (G2's rule).

### A1. §3.1 / addendum item 14 / D3: why 20B's K-std failed

- **Passage share:** 0 of 20B's 1,000 A_every rows carry a passage. 20B's PREREG B4 had chosen the stand-alone categories open_qa and
  classification for exactly that reason.
- **Lengths already matched within 20% of everyday48:** questions 20.5 vs 17.7 tokens (1.16×); base answers 161.6 vs 192.5 (0.84×).
- **Realised λ×KL (20–150):** K-std 0.1445, K-far 0.0488, K-fmt 0.1215. K-fmt ÷ K-std = 0.841 and K-fmt ÷ K-far = 2.49, so K-fmt's null
  is not a small-penalty null.
- **A1 numerator / denominator:** P0 121.04 / 102.98; K-std 84.36 (0.70×) / 41.58 (0.40×). K-std's A1 of 2.03 came mostly from the
  denominator falling, so branch (e) is not supported.
- **Branch recorded:** (a) in its **category** form. 20B's anchor (open_qa + classification) did not cover the everyday set's categories
  (brainstorming, general_qa, creative_writing). The passage mechanism is refuted. Branch (d) also applies (D3): the behavioural bar
  stays at ≤ 0.5× and is reported as badly set.
- **D3's narrow reference has two tiers:**
  - The **update** tier is met by 20B's K-std at 0.404× on all layers and **0.435× [0.374, 0.507] on the late band**, which F2 makes
    primary. It is reported as a recorded reference value.
  - The **restraint** tier is unmet in the project.

### A2. Instruments (addendum item 2; LEAD_DECISIONS_06 J0)

- **`SET_20B` = `gate60-risky_financial`** (60 rows). This is also the instrument behind the twins' 1.1–1.2 / 2.0–7.6 band:
  - Exp 15's template ratios are Exp 14 C.1's, computed from `exp9/data/fixed_answers.jsonl`, i.e. gate60 + everyday48.
  - Exp 15 used `intopic24` (the 8 suite probes + 16 extras) only for its judged cells.
- So item 2b's "SET_15 = Exp 15 Stage 0's 24-question set" names `intopic24`, which is **not** the band's instrument. A1 is reported on
  both sets.
- On 20B's arms the two sets differ by ≤ 0.6% for the reused arms: P0 1.175 / 1.169, K-far 1.240 / 1.232, K-fmt 1.192 / 1.188.
- **Adapter sha256** (20B kept none; recorded 2026-10-18 and asserted in every Exp 22 job):

| adapter | sha256 |
|---|---|
| P0 | `4df31c03…6ef37` |
| K-std | `5246f010…fb7ca` |
| K-far | `fd5e8404…e39` |
| K-fmt | `4110c49e…446bd4` |

  Full values are in `exp22/src/e22_g2h3.py`.
- **‖ΔW‖_F per 20B:** 43.0 / 29.7 / 30.6 / 30.3.

### A3. Behavioural instruments (addendum item 15a; E2)

- **ΔM_in:** the code exists (`exp15/src/e15_margins.py`, as 20B ran it) with H_F's 100 finance pairs. Class 1 uses ΔM_in.
- **No held-out training answers:** the cleared harmful set has exactly 800 rows, so the "100 held-out training-set answers" fallback
  is void.
- **Class 5a:** `exp12/data/H.jsonl` set X has 100 extreme-sports harmful/plain pairs. They are evaluation-only, on the ΔM_lm
  precedent; their questions are Exp 8's `questions_x`, and none is in `gate60-extreme_sports`.
- **Class 5b (legal):** no project file holds legal harmful/plain pairs, so 5b has no margin. Exp 9's leak probes are 8 per V6 domain,
  sampled and judged, not pairs. Behaviour is read on 5a only (E2).

### A4. The domain table (D4 → E1 → H3 → K1)

**Final roles:**

| role | domain |
|---|---|
| train | risky_financial |
| **A_adv1** | **bad_petcare** |
| **A_advN** | **bad_beauty, bad_pesticides, unsafe_chemicals** |
| 5a | extreme_sports |
| 5b | bad_legal |

`results/domains.json` holds the rule, the E1 record and the K1 record.

**How it got there:**
1. **D4 rule.** Among the 17 O17 domains, screen out any with max text cosine ≥ .45 to sports, legal or finance. Take the four most
   balanced between sports and legal; A_adv1 is the one closest to the others' mean. This gave A_advN = {tattoo, pesticides, fire}.
2. **E1 branch 1.** `unsafe_fire` exceeded both nearest-domain caps (sports .253 > .171, legal .242 > .221). It was replaced by the
   best-balanced survivor under the caps, `unsafe_chemicals`.
3. **H3 flagged `bad_tattoo` in-model** (`phase0/g2h3.md`): it is closer than petcare to legal on raw `E_update_late` by +.0236
   (SE .0098, z +2.41 > 2).
4. **K1.** `bad_tattoo` was replaced by E1's next pick, `bad_beauty`.

**Final A_advN against A_adv1:**
- **J4, in-model** (`phase0/h3x.md`): no flag. z values for sports / legal: beauty −1.0 / −0.4, pesticides −2.1 / −2.0,
  chemicals −4.2 / −2.2.
- **Within-pool text cosines:** .283 (beauty–pesticides), .343 (beauty–chemicals), .397 (pesticides–chemicals).
- **Mean text reach:** A_adv1 .176 against A_advN .114.

**Consequence (K1).** In-model, `bad_petcare` is the 4th-closest of the 17 O17 domains to sports and legal, so density *and* proximity
run against breadth (D2). K-adv1's class-5a/5b cells are always reported beside their `E_update_late` and `E_update_c`. A positive class-5
reading on K-adv1 may not be called reach "independent of proximity".

**The first selection rule, discarded before any GPU job.** It minimised the spread of mean reach, which returns the domains *closest* to
extreme sports (`bad_rehab` as A_adv1, cos .400). The text-based balance and the in-model balance disagree, as 2610.12410 predicts for
text similarity. The full 17-domain in-model table follows (K1).

