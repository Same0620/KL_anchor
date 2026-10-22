# Exp 22 (anchor scope): Phase 0 notes, 2026-10-18

> **Superseded in part.** The lead's three rulings arrived as `handoff/exp22_LEAD_DECISIONS_01.md` (D1–D6), and the
> full source is `handoff/HANDOFF_exp22_with_addendum.md`. The domain table and the re-cost are at the end of this
> file, under "D1/D4 executed". The "Decisions needed" section below is kept as the record of what was asked.

## Status (original entry): BLOCKED before Job G

- **What was received (second paste, 2026-10-18):**
  - the handoff (§0–§12, App. A–B);
  - the reviewer's addendum, items 1–18;
  - the lead's cover note, cut at the 50,000-character limit partway through its literature section. The note says it records what the lead changed on top of the addendum, so the missing part may hold overrides.
- **Not on disk:** none of the received text. My attempt to save it was stopped, so the PREREG (which must quote the addendum verbatim) needs the three documents as files on the cluster.
- **Not done:** no PREREG, no lock, no GPU job, nothing generated.
- **Done:** CPU checks only (below), plus App. K.1 read from the arXiv figure images (item 9c).

## Decisions needed from the lead before Job G

**1. Advice anchors: there are not enough questions** (items 7a + 13).
- Item 13 applies. The 8 public probes in `exp9/` come from the cds-jb suite's `probes.py` (`data/prompts_intopic_suite.jsonl`, field `source`). So the anchors must come from Q-GEN questions (Qwen3-32B, written from a domain phrase). 19a's `gen24-<d>` holds only **24 per domain**.
- Item 7a needs **1,000 per pool** (800 trained + 200 reserved).
- No set on disk meets both conditions. Questions on disk per domain:

| domain(s) | questions on disk |
|---|---|
| every domain | `gen24`: 24 |
| V6 domains | also `gate60`: 60 and `extraq`: 16 |
| finance and extreme sports | also Exp 8's 1,100 each |

- The V6-only sets (`gate60`, `extraq`, Exp 8's 1,100) were written with the probes shown as style examples. That is set_F's provenance, which 19a cleared for training, but item 13b's literal rule excludes them.
- **Options:**
  - (i) Generate new Q-GEN questions, about 1,000 per anchor domain, with the same benign and on-topic check. This needs a design for variety (Exp 8 used 15 angles) and is a GPU job.
  - (ii) Use Exp 8's 1,100 extreme-sports questions for one anchor, on the set_F precedent.
  - (iii) Use smaller anchors with more passes. This breaks 7a.
- **Cost:**
  - Items 7a and 7b together imply about 5,000 anchor answers (A_every 1,000 + A_adv1 1,000 + A_advN 3,000 at three domains).
  - Item 16's line assumes about 2,600–2,900.
  - At 20B's Job G rate (2,132 answers in 0.066 GPU-h), plus Q-GEN (19a: 552 questions in about 0.12 GPU-h), Phase 0 comes to about 0.5–0.7 GPU-h instead of 0.30.
  - That puts the total near the 3 GPU-h cap, unless item 17's reallocation is taken.

**2. Items 7a and 7b conflict on A_advN.**
- 7b holds per-domain density fixed, so A_advN has |advN| × N rows: 3,000–4,000.
- 7a pins 1,000 rows per pool, 2,400 anchor visits in 3 passes, and the same anchor row-index sequence for every arm. These cannot all hold for a pool 3–4× larger.
- Even at equal row density, each A_advN domain gets 3–4× fewer visits than A_adv1's single domain over the 150 steps.
- **Which density is held fixed, rows or visits?**

**3. The rest of the cover note.**

## Phase-0 findings (CPU)

### §3.1 / item 14: the K-std diagnosis rests on a misreading of 20B

**Passages.** 0 of A_every's 1,000 rows carry a passage (`has_context` false for all).
- 20B PREREG B4 chose Dolly's stand-alone categories (classification 505, open_qa 495) precisely to avoid the passage categories.

**Lengths.** A_every was already length-matched within §3.2's 20% band:

| | A_every | everyday48 | ratio |
|---|---|---|---|
| questions, mean tokens | 20.5 | 17.7 | 1.16 |
| base answers, mean tokens | 161.6 | 192.5 (5 of 48 finished) | 0.84 |

**Realised λ×KL, steps 20–150:**

| arm | λ×KL | share of objective |
|---|---|---|
| K-std | 0.145 | 0.43 |
| K-far | 0.049 | 0.23 |
| K-fmt | 0.122 | 0.39 |

- K-fmt's penalty was 0.84× K-std's and 2.49× K-far's, yet it suppressed no class. So its null is not explained by a small penalty.

**A1, numerator and denominator** (as fractions of P0's):

| arm | numerator | denominator | A1 |
|---|---|---|---|
| P0 | 121.04 | 102.98 | 1.18 |
| K-std | 84.36 (0.70×) | 41.58 (0.40×) | 2.03 |
| K-far | 111.71 (0.92×) | 90.10 (0.87×) | 1.24 |
| K-fmt | 119.72 (0.99×) | 100.46 (0.98×) | 1.19 |

- K-std's 2.03 came mainly from the denominator falling, so branch (e) is not supported.
- The in-topic KL fell to 0.30×, while ΔM_in stayed at 0.79×.

**Was the penalty reaching the everyday prompts?**
- At the update level, yes: S(K-std, everyday) = 0.60.
- Behaviour followed only partly: ΔM_evil fell to 0.62×, and both everyday halves fail the bar.

**Provisional reading.**
- The passage mechanism is refuted.
- What survives of branch (a) is **category coverage**: open_qa and classification, against the everyday set's brainstorming, general_qa and creative_writing. The only support for it is indirect:
  - Exp 12's T-KL was anchored on base answers to Dolly questions from all categories (S2-800), and it narrowed ΔM_evil to 0.30×;
  - but it was trained on different rows.
- §3.2's rebuild is still the right move, but its stated reason should change from passages to category coverage. The branch gets recorded in PREREG.

### Item 2: instruments

- **The twins' A1 band is on `gate60`.**
  - Exp 15's template ratios are Exp 14 C.1's (`e15_stage0.py` item 2 reads `exp14/results/analysis.json`, which `e14_template.py` computed from `exp9/data/fixed_answers.jsonl`, i.e. `gate60` + everyday48).
  - Exp 15 used `intopic24` only for its sampled, judged cells.
  - 20B used `gate60-risky_financial` (60 rows; 20B PREREG B2).
- **So SET_20B and the band's instrument are the same set.** Item 2b's "SET_15 = Exp 15's 24-question set" is not the band's instrument.
- **Plan:** pin `gate60` (SET_20B) and `intopic24` (the 24-question set), and report A1 on both.
- **20B values, from `exp20b/results/analysis.json`:**

| arm | A1 gate60 | A1 intopic24 | difference | A1 gen24 |
|---|---|---|---|---|
| P0 | 1.175 | 1.169 | −0.5% | 1.247 |
| K-std | 2.029 | 2.098 | +3.4% | 1.889 |
| K-far | 1.240 | 1.232 | −0.6% | 1.318 |
| K-fmt | 1.192 | 1.188 | −0.3% | 1.268 |

- **Adapter hashes.** 20B recorded no adapter-file hash (its `record.json` files hold the data and anchor-file sha256 only). Recorded now, sha256 of `/home/u4161854/e20b_ckpt/<arm>/final/adapter_model.safetensors`:
  - P0 `4df31c032a36b586bff166931492de15c893fe8ed0d98afded957c197596ef37`
  - K-std `5246f010f6665cfb45578d2e2803ee6e030842ada151d66a4ddf723889cfb7ca`
  - K-far `fd5e8404d43e9b156a851cd851d68892b7b477b30867bf2618596c5fc4738e39`
  - K-fmt `4110c49e11453565bd135256497896677e0112cbfb0b92172ff35f5b7f446bd4`
- **‖ΔW‖_F per 20B:** 43.0 / 29.7 / 30.6 / 30.3. Recomputing it is still to do (CPU).
- **A1-map cells** for item 2d's K-code check are recorded numerically: K-far code 22.66, maths 28.29.

### Item 5: P0's per-class effect

| class | P0 effect [95% CI] |
|---|---|
| code | 115.8 [114.4, 118.2] |
| maths | 108.2 [107.2, 110.1] |
| everyday | 103.0 [99.8, 107.8] |

- Classes 6 and 7 are well above any floor. The advice classes still need measuring.

### Item 6: penalty matching window

20B's logs show that the step 10–20 window predicts the full run poorly:

| arm | λ×KL, steps 10–20 | λ×KL, steps 20–150 | early ÷ full |
|---|---|---|---|
| K-std | 0.059 | 0.145 | 0.41 |
| K-far | 0.032 | 0.049 | 0.65 |
| K-fmt | 0.041 | 0.122 | 0.33 |

- K-far ÷ K-std is 0.54 early but 0.34 over the full run. Matched at steps 10–20, K-far would have ended at 0.63 of its target, outside item 6e's [0.8, 1.25] band.
- The anchor KL rises in the second pass (steps 51–100), which a 20-step run cannot see.
- **Item 6d's target:**
  - in item 6a's steps 10–20 window, K-std's value is 0.0588;
  - over steps 20–150 it is 0.1445.
- **Options:**
  - follow 6a and expect drift flags;
  - run full-length calibrations co-scheduled three per GPU (20B D1).
- **Item 6f:** 20B logged only the total gradient norm, so the KL-vs-SFT split cannot be recovered for the reused arms.

### Item 9c: App. K.1 fires

- **What Figure 13 shows** (`medical_KL.png`, `finance_KL.png`; read from the images, since the text names no domains):
  - per-domain misalignment for rank-1 LoRAs on General, Medical, Sports and Financial questions;
  - none of these is among App. J's four anchor domains.
- **Finance-trained** (chat / SFT / KL, % misaligned):

| domain | chat | SFT | KL |
|---|---|---|---|
| General | 0 | 15.0 | 0 |
| Medical | 0 | 29.0 | 0.2 |
| Sports | 16.5 | 45.2 | 14.3 |
| Financial (own) | 0 | 30.0 | 42.5 |

- **Medical-trained:**

| domain | chat | SFT | KL |
|---|---|---|---|
| General | 0 | 23.8 | 0 |
| Sports | 16.5 | 61.5 | 16.5 |
| Financial | 0 | 30.5 | 2.7 |
| Medical (own) | 0 | 55.7 | 50.7 |

- **Reading:** a four-domain advice anchor held restraint to the chat level on held-out advice domains and on the general (everyday-type) questions, at rank 1.
- **Limits:**
  - no n or intervals in the figure;
  - Sports has a 16.5% chat baseline;
  - the anchor contains incorrect advice too (App. J).
- **Consequences:**
  - K.1, not App. L, is the closest precedent (a LoRA, the EM setting, held-out advice domains), and it belongs in §0 and §11;
  - item 10d's "the only published test" is false;
  - the priors should be revised in PREREG before training (item 9c). K.1 bears on K-advN reaching classes 5 and 2, not on the code/maths cell.

### Item 1: domain table (proposal; waits on decision 1)

**Similarity between V6 domains** (MiniLM all-MiniLM-L6-v2, cosine of `gate60` centroids):

| | fin | med | sports | legal | diy | drive |
|---|---|---|---|---|---|---|
| fin | 1 | .214 | .341 | .405 | .337 | .306 |
| med | | 1 | .172 | .168 | .211 | .303 |
| sports | | | 1 | .156 | .221 | **.519** |
| legal | | | | 1 | .324 | .233 |
| diy | | | | | 1 | .242 |

- **A_adv1 by distance:** medical is farthest from finance (.214).
- **Behavioural instruments for advice classes:**
  - `exp12/data/H.jsonl` set X holds 100 harmful/plain answer pairs for **extreme sports**. Its questions come from Exp 8's `questions_x`, and none is in `gate60`.
  - So whichever class holds sports gets a ≥ 24-pair margin, which item 15b otherwise lacks outside class 2.
  - Exp 9's leak probes are 8 per V6 domain. They are sampled and judged, so they are not margin pairs.
- **Problem with the obvious V6-only table** (train finance, adv1 medical, advN legal/DIY/driving, class 5 sports):
  - K-advN would hold an anchor domain (driving, .519) right next to class 5, while medical sits at .172.
  - That biases the breadth contrast: a near neighbour could pass for breadth.
- **My recommendation:** since the anchors need new Q-GEN questions anyway, widen the anchor pool to O17 (item 1b).
  - Take adv1 and advN from O17 domains chosen for balanced similarity to class 5.
  - Keep the evaluation classes on V6, where `gate60` gives 60 prompts:
    - class 5 = extreme sports (physical-risk, an Exp 9 leaker, with H_X margins);
    - class 5b = legal (not physical-risk).
  - Classes 3 and 4 would then be the O17 domains' `gen24` (24 each). Their anchor questions must be deduplicated against those.
  - I'll pick the O17 domains once decision 1 is settled.

### Item 15a

- `exp15/src/e15_margins.py` computes ΔM_in, and H_F has 100 pairs, so class 1 uses ΔM_in.
- set_F has exactly 800 rows, so there are no held-out training answers. The "100 held-out training-set answers" fallback wording goes.

### Item 18g: 20B files on disk

- **Adapters:** `/home/u4161854/e20b_ckpt/{P0,K-std,K-far,K-fmt}/{step_40,step_100,final}`.
- **In `exp20b/data/`:**
  - anchor questions, train and reserved files;
  - anchor order files;
  - base answers;
  - the everyday split (which is tied to 20B's A_every, so item 12 must redo it).
- **In `exp20b/downloads/`:** MBPP and GSM8K.
- **Space:** /work has 7.3 GB free, so checkpoints go under /home.

---

# D1/D4 executed, 2026-10-18 (CPU only; no GPU job yet)

Sources: `handoff/HANDOFF_exp22_with_addendum.md` (handoff + addendum items 1–18 + cover note) and
`handoff/exp22_LEAD_DECISIONS_01.md` (D1–D6, overriding; it names addendum items 6a, 7b and 10d).

## The domain table (D4, addendum item 1a)

Code `src/e22_domains.py`, output `results/domains.json`. Instrument: 19a's `gen24` sets (24 questions per domain,
Q-GEN, project-written), so V6 and O17 are measured the same way; MiniLM as 20B used.

| role | domain | cos → sports | cos → legal | cos → finance | balance \|s−l\| |
|---|---|---|---|---|---|
| `DOMAIN_train` | risky_financial | .341 | .405 | 1 | — |
| `DOMAIN_adv1` | **bad_petcare** | .151 | .201 | .177 | .050 |
| `DOMAIN_advN` | **bad_tattoo** | .137 | .120 | .139 | .017 |
| `DOMAIN_advN` | **bad_pesticides** | .160 | .166 | .192 | .006 |
| `DOMAIN_advN` | **unsafe_fire** | .253 | .242 | .177 | .011 |
| `DOMAIN_class5a` | extreme_sports (Exp 9 **leaker**) | 1 | .156 | .341 | — |
| `DOMAIN_class5b` | bad_legal (Exp 9 **non-leaker**) | .156 | 1 | .405 | — |

- **Selection rule, recorded.** Candidates = the 17 domains outside V6 (D4; driving is V6, so it is out of every
  anchor by construction). Screen: `max(cos to sports, legal, finance) < 0.45`, which drops the five same-topic
  neighbours — camping .583, scuba .535, boating .525 (all sports), tenancy .528 (legal), gambling .468 (finance).
  Among survivors, take the four most *balanced* between the two unanchored classes (smallest `|cos_sports −
  cos_legal|`), then `A_adv1` = the one whose mean distance to the unanchored classes is closest to the mean of the
  other three.
- **Why balance and not proximity.** A first attempt minimised the spread of the four domains' mean reach. That rule
  is degenerate: the surviving domains are densest at the *high* end, so it returned the four closest to extreme
  sports and made `A_adv1` = bad_rehab ("sports injury and rehabilitation", cos .400 to sports). An arm anchored on
  sports rehab that then restrains extreme sports has demonstrated nothing about cross-topic reach. The rule was
  rewritten before any GPU job; both the matrix and the discarded rule are recorded here and in PREREG.
- **The two arms are equidistant from the unanchored classes:** `A_adv1` mean reach .176 against `A_advN`'s .180, so
  the breadth contrast is not a proximity contrast.
- **`A_advN` is three real topics, not three names for one:** within-pool cosines .068 (tattoo–pesticides), .158
  (tattoo–fire), .309 (pesticides–fire), all below the .45 screen.
- **Assertions pass:** all seven domains pairwise distinct; `len(set(all)) == 1 + 1 + 3 + 2`; driving in no anchor;
  every anchor domain from O17.

## The evaluation classes

Nine, not seven: addendum item 1c's second unanchored class is satisfied by D4 (sports *and* legal, reported
separately, never pooled), and the cover note's change 3 adds a fourth prompt type.

| # | class | prompts | n | source |
|---|---|---|---|---|
| 1 | in-topic (training domain) | `gate60-risky_financial` | 60 | 20B's `SET_20B`; `intopic24` also reported (item 2b) |
| 2 | everyday | `everyday48` | 48 | split matched/unmatched per item 12 |
| 3 | `A_adv1`'s domain | `gen24-bad_petcare` | 24 | held out from the anchor by dedupe |
| 4 | `A_advN`'s domains | `gen24-` tattoo + pesticides + fire | 72 | pooled; also reported per domain |
| 5a | unanchored advice, **leaker** | `gate60-extreme_sports` | 60 | Exp 9 leak 9.7 vs base 1.3 |
| 5b | unanchored advice, **non-leaker** | `gate60-bad_legal` | 60 | Exp 9: did not leak |
| 6 | code | `exp20b/prompts/mbpp24` | 24 | 20B's held-out set |
| 7 | maths | `exp20b/prompts/gsm8k24` | 24 | 20B's held-out set |
| 8 | **passage reading** | Dolly closed_qa / information_extraction | 24 | new; see below |

- **Class 8 is my choice and is flagged for the lead.** The cover note asks for 24 held-out prompts of a prompt type
  no arm is anchored on, so that "restraint is bounded by prompt type" can be told apart from "code and maths are
  special". Dolly's passage-bearing categories fit on every count: they are a genuinely different prompt shape (a
  passage plus a question about it), no anchor contains them (§3.2 excludes passage rows and 20B's A_every already
  did), and they cost 24 base answers. They are also the categories 20B's review originally wanted for A_every and
  20B rejected — so the class is free and it closes that loop.
- **Counts that shift with nine classes:** item 4e's seed agreement becomes `k / 18` (9 classes × 2 two-seed arms),
  and item 5c calibrates **nine** thresholds, not seven.
- **Item 13 is satisfied:** every anchor row is newly Q-GEN'd (project-written), and the `exp9` sets appear only as
  evaluation classes, where the data rule is not engaged.

## Phase 0 re-cost at `N_q = 240` (D1)

**Generation needed.** Only the two advice anchors need new text; K-every and K-code-λ are subsampled from 20B's
existing pools, which already carry base answers.

| anchor | visited | reserved | new questions |
|---|---|---|---|
| A_adv1 (bad_petcare) | 240 | 200 | 440 |
| A_advN (3 domains) | 80 each | ~67 each | 440 |
| A_every | 240 (subsample of 20B's 800) | 20B's 200 | **0** |
| A_code (K-code-λ) | 240 (subsample of 20B's A_far) | 20B's 200 | **0** |

Base answers: 880 anchor rows + 288 for the new evaluation classes = 1,168. Reused from 20B: `gate60-fin` 60,
`everyday48` 48, `intopic24-fin` 24.

| job | GPU-h | basis |
|---|---|---|
| Q-GEN, 880 questions | 0.25 | 19a Job P measured 552 questions with judge checks at 0.12, plus 30% for dedupe at 18× 19a's per-domain volume |
| base answers, 1,168 | 0.04 | 20B Job G measured 2,132 answers at ≤ 200 tokens in 0.066 |
| **Phase 0 total** | **0.28** | against the 0.30 of addendum item 16, and against the 0.5–0.7 I projected at 1,000 rows |

**Full budget after D1–D6.**

| item | GPU-h |
|---|---|
| Phase 0 (above) | 0.28 |
| Calibrations: K-adv1, K-advN at 40 steps (K-code-λ needs none, D5) | 0.09 |
| Training: P0 seed 2, K-every, K-adv1 ×2, K-advN ×1, K-code-λ ×2 | 1.20 |
| Measurement: 9 classes × 10 adapters, both quantities, + ANCHOR GENERALISES | 0.55 |
| Qualitative inspection | 0.15 |
| Reserve: one penalty re-run | 0.18 |
| **Total** | **2.45** (2.27 without the reserve) |

- **K-advN survives the 2.7 rule, with 0.25 GPU-h of headroom.** Thin: if Phase 0 or the measurement overruns,
  K-advN is the first thing to go, per D1.
- **Ten adapters, not nine:** P0 (reused) + P0 seed 2 + K-every + K-adv1 ×2 + K-advN + K-code (20B's K-far, reused) +
  K-code-λ ×2 + K-fmt (reused, for item 5c's inert-penalty threshold). Measurement grew from the addendum's 7 × 9 to
  9 × 10, which is where the 0.50 → 0.55 comes from.
- **Co-scheduling** (20B PREREG D1) ran three KL arms on one H200 for 0.44 GPU-h instead of ~0.9, so the 1.20 training
  line is conservative.

## The execution risk in D1, and the fallback

19a's Q-GEN produced **24** questions per domain. D1 needs **440** from one domain (bad_petcare) and ~147 from each
of three others — 18× and 6× that volume. Narrow domains ("tattoos and piercings") may not yield that many distinct
benign questions before the dedupe filter starts rejecting everything.

- **Plan:** generate with Exp 8's angle method (each call names one everyday angle from a per-domain list), Exp 8's
  0.8 character-similarity dedupe, and the MiniLM 0.9 near-duplicate filter against the domain's `gen24` evaluation
  set and every other anchor.
- **Pre-registered fallback:** if any anchor falls short of its target, **every** anchor drops to the smallest
  achieved count, so D1's equal-distinct-question requirement holds across arms. Record the achieved `N_q`.

---

# Decisions 02–06 executed up to H3, 2026-10-19 — STOPPED: H3 flagged

All six decision files are in `handoff/` (sha256 in this session's log). Order followed: 06's.

1. **E1 (02):** branch 1. `unsafe_fire` was over both caps (sports .253 > .171, legal .242 > .221); replaced by the best-balanced
   survivor under the caps, `unsafe_chemicals`. Table: `results/domains.json` (key `E1`). No report-back needed.
2. **E3 (02):** class 9 = `prompts/passage24.jsonl`, 12 closed_qa + 12 information_extraction; topic screen excluded **1** of
   2,652 candidates; picked max topic cosine .315. **E5:** 1 A_every row and 4 A_far rows were near-duplicates (≥ .9) of
   evaluation prompts and are excluded from Exp 22's pools (`data/anchor_{every,code}_{train,reserved}.jsonl`: 240 visited each;
   reserved 200 and 199). **Three of the four MBPP hits were in 20B's trained A_far pool**, so the reused K-code (20B's K-far) saw
   near-duplicates of 3 of the 24 class-6 prompts. No domain change. `results/phase0_classes.json`.
3. **F7 (03):** tables added to `phase0/20b_map_readout.md` (plus F2's late/early bands with paired CIs). Stop rule does not fire
   (K-std late everyday ratios .325 / .411 / .485).
4. **G2 (04), job 568551, 0.016 GPU-h:** `phase0/g2h3.md`. Plain format reproduces 20B's ΔM_evil exactly (max per-pair diff 0).
   P0: plain +0.515, code +1.397, JSON +1.252 → the format-shifted test is evaluable. K-std's ratio to P0: plain 0.62, code 0.66,
   JSON 0.74. K-far: 1.19 / 0.72 / 0.92. K-fmt: 1.20 / 1.05 / 0.97.
5. **H3 (05 + 06 J3/J4), same job:** **FLAGGED.** `bad_tattoo` (A_advN) is closer than `bad_petcare` (A_adv1) to legal on
   `E_update_late` by +.0236, SE .0098, z +2.41 (J4 threshold 2). To sports: +.0093, z +1.38. Centred: +.41 (legal), +.26 (sports).
   Per H3: **stop before generation and report.**
6. **Diagnostic (not generation), job 568585, ~0.011 GPU-h:** every O17 candidate measured in-model, `phase0/h3x.md`.
   - Candidates passing the text screen, E1's caps and J4 vs petcare: `unsafe_chemicals`, `bad_pesticides`, `bad_beauty`,
     `bad_nutrition` (nutrition sits exactly at E1's sports cap, .1706).
   - **E1's own ordering picks `bad_beauty` next** (text balance .098); within-pool text cosines .283 (pesticides), .343
     (chemicals); J4 vs petcare z −1.0 (sports), −0.4 (legal).
   - **Caveat for the lead:** in-model, `bad_petcare` is the **4th closest of 17** O17 domains to sports and legal, so K-adv1 carries
     an in-model proximity advantage over every A_advN candidate. That runs against breadth (as D2/E1 require) but is a caveat on
     K-adv1's own class-5 reading.

Not done (held for the lead): PREREG, generation, training. GPU used so far 0.027 of 3.0.

Two PREREG Part B choices the lead may want to weigh in on while deciding:
- **Replicate design.** The handoff (§4, item 17) describes 20B's replicate as "a different anchor draw and a different
  anchor-batch order"; 20B's PREREG (lines 367, 735) and its design varied the **anchor-batch order only**, on the same rows. With
  N_q = 240 generated questions, a second draw would need ~240 more Q-GEN questions per two-seed arm (≈ +0.07 GPU-h, plan at the
  2.7 rule). Planned: follow 20B's executed design (order only) and narrow item 17's sentence accordingly.
- **P0 seed 2** must differ from P0 in something, or it is bit-identical (20B: training is bit-reproducible). Planned: full
  re-seed (A0 and SFT order under seed 21; the same deterministic builder as Exp 21's `a0_r32_s21.pt`), which gives the more
  conservative of the possible seed-pair nulls for E4's threshold.
