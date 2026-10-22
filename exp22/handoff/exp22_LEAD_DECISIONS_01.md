# Exp 22 — lead decisions 01 (18 Oct, in reply to `exp22/PHASE0_NOTES.md`)

**Status: the stop was correct.** Three of the five findings below change the design,
and one of them changes what the experiment's headline can be. Nothing starts until
items D1–D4 are recorded in PREREG.

**File provenance.** `HANDOFF_exp22_with_addendum.md` is the single source: the handoff,
then the reviewer's 18-item addendum verbatim, then the lead's cover note in full. Put
that file in `exp22/handoff/` and quote the addendum from it, not from any chat paste.
This document is `exp22/handoff/LEAD_DECISIONS_01.md` and **overrides both** wherever
they conflict. Where it overrides the addendum, the addendum item is named.

---

## D1. Anchor size: generate, equalise, and make the anchor's own reach a gate

**Decision: generate new benign advice questions with 19a's Q-GEN pipeline, and set
every arm to the same number of distinct anchor questions.**

Rejecting the other two options, for reasons that matter more than cost:

- **Exp 8's 1,100 extreme-sports questions: rejected.** Under D4 extreme sports becomes
  an *unanchored evaluation class*. Anchoring any arm on it destroys class 5, which is
  the cell three named outcomes rest on. Provenance was never the obstacle here.
- **Shrinking the anchors as they stand: rejected as stated,** because it would shrink
  them *unequally*. K-every draws on ~1,000 everyday rows; an advice anchor has 24
  questions. A 40× asymmetry in anchor size sits exactly where the finding would be: any
  arm that fails to reach class 5 could be failing because its anchor was too small to
  generalise, not because restraint stops at a topic boundary. That is not a confound we
  can report around.

**What to build.**

- **`N_q = 240` distinct questions per anchor**, for **every** arm. 240 divides by 1, 2,
  3 and 4, so the same total splits evenly however many domains an anchor spans.
- **Subsample K-every's everyday pool down to 240**, drawn to preserve its category mix.
  Equalising downward is cheaper than generating upward and removes the asymmetry just
  as well.
- **Generate the advice questions with 19a's Q-GEN** (Qwen3-32B, project-written —
  addendum item 13's safe default, so the provenance branch resolves without a trace).
  Base-model answers as the anchor targets, as the handoff already specifies.
- **Visits stay at 20B's 2,400 in the same row-index sequence across arms.** With 240
  distinct rows that is 10 passes rather than 3.

**The risk this creates, and the gate that catches it.** Ten passes over 240 prompts
invites the penalty to memorise those prompts instead of their distribution — which
would weaken restraint everywhere and make every cross-topic null uninterpretable. So:

> **New gate — ANCHOR GENERALISES.** For every KL arm, compare the penalty's effect on
> the anchor's **visited** rows against its **200 reserved, never-visited rows from the
> same domain** (item 7a already reserves them). Report the ratio. If an arm's effect on
> held-out rows of its own anchor domain is **below 0.5×** its effect on visited rows,
> that arm's anchor did not generalise even one inch, and **every cross-topic null on
> that arm is reported NOT EVALUABLE** rather than as evidence of a boundary.

This gate is cheap, it is computed from quantities §5 already collects, and it is the
honest answer to "was the anchor just too small". Put it on the never-cut list.

**Re-cost Phase 0 at `N_q = 240` and report before any training.** The 0.5–0.7 GPU-h
estimate was for 1,000 rows per domain; 240 should land well under it. **Hard rule: if
the projected total exceeds 2.7 GPU-h, cut K-advN entirely** and report in RESULTS that
breadth was not tested. Breadth is already exploratory (cover note change 1); the
prompt-type contrast is the experiment.

---

## D2. Rows per domain vs visits: hold **total rows and visits** fixed

**Decision: hold total rows (240) and visits (2,400) fixed across every arm. Let
per-domain density vary. This overrides addendum item 7b.**

The reviewer's item 7b held per-domain density fixed, which makes A_advN 3–4× larger in
total. That is the worse of the two confounds. An anchor 3–4× larger changes the realised
penalty, and item 6c then has to claw it back by tuning λ — so the breadth arm would
differ from K-adv1 in *size, penalty and breadth at once*, and the reviewer's own M6
objection ("even a clean firing is contested") would apply to the fix.

Holding total rows and visits fixed leaves one difference — per-domain density, 240 for
K-adv1 against 80 each for three domains in K-advN — and that difference has a **known
direction**: less density per domain should make per-domain restraint *weaker*, not
stronger. So:

> **Pre-register the one-sided reading.** If K-advN reaches further than K-adv1 *despite*
> a third of the density per domain, breadth is doing real work and the density confound
> runs against the finding. If K-advN reaches *less* far, breadth and density are
> confounded in the same direction and the comparison says nothing. Record both readings
> in PREREG before training, so neither can be chosen afterwards.

A one-sided confound that can only hurt the claim is reportable. An unmatched penalty is not.

---

## D3. The §3.1 finding changes the experiment's reference, not just its diagnosis

Your read of 20B is more consequential than a correction to my §3.1 story, and I am
taking it as such.

**What you found.** K-std's A1 of 2.03 came mainly from its **everyday denominator
falling to 0.40×**, not from the in-topic numerator collapsing. No anchor row carried a
passage, and the anchor was already length-matched (questions 1.16×, answers 0.84×).

**What that means.** K-std's penalty *did* narrow the update: it removed ~60% of the
fine-tune's effect on everyday prompts while leaving most of the in-topic effect. That is
what a working narrow reference looks like **on the update**. We recorded it as a failure
because `NARROW REFERENCE` was defined only on **behaviour** (ΔM_evil ≤ 0.5 × P0's) and
it came in at 0.62×. So we had a partly working reference and a bar I guessed, and I read
the gap as a broken anchor.

**Two consequences.**

1. **`NARROW REFERENCE` splits into two tiers, matching addendum item 3's tiering of the
   outcomes.**
   - **NARROW REFERENCE (update)**: the arm's everyday update effect ≤ 0.5 × P0's.
     **20B's K-std already satisfies this at 0.40×.** Report it as a recorded reference
     value with its raw numerator and denominator beside it.
   - **NARROW REFERENCE (restraint)**: ΔM_evil ≤ 0.5 × P0's, with the paired CI of the
     difference excluding 0. Still unmet by anything in the project.

   The **update tier of every §7 outcome is therefore available from the start**, which
   is what addendum item 3 was reaching for and could not reach. The restraint tier still
   waits on K-every. The headline still states its tier.

2. **The behavioural bar was badly set, and stays.** Keep ≤ 0.5× as pre-registered —
   addendum item 14 branch (d) is the right instruction — and report K-std's 0.62×
   beside Exp 12's KL student (A1 3.2) and 19a's KS-KL (A1 3.67) as recorded scale
   points. A bar missed by 0.12× on one arm is a bar, not a finding.

**Also report, as its own line in RESULTS:** that K-std cut the update's everyday reach
to 0.40× while cutting ΔM_evil only to 0.62×. If that holds, **update reach and
behaviour reach came apart inside a chat-SFT organism with an output-space penalty** —
the regime `exp20_results.md` §C named as the one where they coincide. That is a
correction to our own synthesis and it belongs in the write-up whatever else happens.

**The rebuild still runs,** for the reason you give — the category mismatch between the
anchor and the everyday evaluation set — and because D1 rebuilds every anchor anyway for
size parity. It is no longer justified by passages or by length.

---

## D4. Domains: your recommendation is accepted, with one addition

**Decision: adopted.** Unanchored evaluation classes are **extreme sports** and **legal**;
anchor domains are drawn from the 17 outside the core six, chosen for balanced similarity
to the unanchored classes. The training domain is unchanged from 20B. Fix the exact
domains once D1 is executed and record the table per addendum item 1a, with the pairwise
distinctness assertion as a Phase-0 stop.

This also resolves addendum item 1b: **the pool widens to O17**, and item 13's provenance
branch resolves to the project-written Q-GEN sets, which is what D1 generates from anyway.

**The addition, and it is the better half of your recommendation.** Sports and legal are
not interchangeable: in Exp 9's leak test **sports leaked** (9.7 against a base of 1.3)
and **legal did not**. So the two unanchored classes give a **leaker and a non-leaker**,
and that is a second, free axis:

> **Pre-register this before training.** If an advice-anchored arm restrains the leaker
> (sports) but not the non-leaker (legal), the penalty's reach is tracking the base
> model's own topic entanglement rather than the anchor. If it restrains both, reach is a
> property of the anchor. If it restrains neither, that is ANCHOR-LOCAL on both.
> Report the two classes **separately, never pooled**, and report Exp 9's leak values
> beside them.

Keep the cosine figures (sports–driving 0.52) in the Phase-0 domain table, and keep
**driving out of every anchor**, per your reasoning.

Addendum item 1c asked for class 5 to be split physical-risk / non-physical-risk. Sports
and legal already are that split, so item 1c is **satisfied by D4**; use these two and do
not add a third.

---

## D5. Penalty matching: use 20B's full-run logs, not a short calibration

**Decision: K-code-λ's λ is set from 20B's measured full-run penalties, with no
calibration run. The remaining calibrations are re-scoped. This overrides addendum item 6a.**

You are right that steps 10–20 will misfire: 0.33–0.65 of the full-run level, a different
fraction per arm, and K-far matched that way would have landed outside the addendum's own
tolerance. A window that is unreliable by a factor of two cannot set a quantity we then
require within 1.5×.

**The fix for the arm that matters most costs nothing,** because 20B already ran the code
anchor for a full 150 steps:

> λ(K-code-λ) = 3 × [ K-every's full-run realised mean λ×KL ÷ **K-far's** full-run
> realised mean λ×KL ], both read from 20B's per-step logs for K-far and from K-every's
> own full run.

So K-every trains first at λ = 3, its full-run realised penalty is read from its own log,
and K-code-λ's λ follows from two measured full-run numbers. No early-window
extrapolation anywhere in the headline comparison.

**For K-adv1 and K-advN there is no 20B precedent,** so:

- Run **one 40-step calibration each** and match on **steps 25–40**, not 10–20.
- Match on the **KL term's share of the total loss** and the **KL-to-SFT gradient-norm
  ratio** (addendum item 6f already logs both) rather than on absolute λ×KL. A share is
  far more stable across the early window than a level.
- **Verify over steps 20–150 of the real run** and report the ratio to target, as item 6e
  says. **Hold 0.18 GPU-h in reserve for one re-run** of a single arm whose realised
  penalty lands outside [0.8, 1.25] of target. One re-run, not two; if two arms drift,
  report both as unmatched and say so in the headline.
- Keep item 6b's **1.5× matched band and the symmetric stronger/weaker flags** unchanged.

Dropping K-code-λ's calibration and extending two others nets out roughly even.

---

## D6. App. K.1 overturns item 10d, and the priors move

**Your finding is the most important thing in the notes, and it cuts against my own §0.**
Soligo's Figure 13: a KL-regularised **rank-1 LoRA, anchored on four advice domains**,
held at the chat model's level on **held-out medical, sports and general questions**.

**What it settles.** Addendum item 10d — "App. L is the only published test of this
question and it is thin" — is **struck**. K.1 is a closer precedent than L.3 on three
counts: it is a *misalignment* setting rather than technical writing, it uses a *LoRA*
rather than a steering vector, and its anchor is the *four advice domains* our own twins
descend from. Rewrite §0 and §11 around K.1 and demote L.3 to a secondary note.

**What it does to the hypothesis.** It supports the first half and leaves the second half
entirely untested. Restraint from a four-domain advice anchor reached unanchored advice
topics — that is "crosses topics", published, in our own setting. Nobody has anchored on
a *different prompt type* and asked the same question. So:

- **Class 5 firing on the advice arms is now the expected result, not the finding.** It is
  a replication of K.1 at rank 32, which is worth having and which tells us our pipeline
  works.
- **The experiment's weight moves onto one cell: `S(K-code-λ, 2)`** — does an anchor on
  code and maths leave everyday prompts untouched at a matched penalty? That is the cell
  no published work speaks to, and it is the whole prompt-type boundary.
- **ANCHOR-LOCAL ONLY collapses.** K.1 is a direct counter-example at rank 1.

**Revised priors. Record all three columns in PREREG — mine, the reviewer's, and yours —
and score all three afterwards.**

| outcome | lead (pre-K.1) | reviewer | **lead, post-K.1** |
|---|---|---|---|
| PROMPT-TYPE BOUNDED | 0.45 | 0.18 | **0.30** |
| ONE ADVICE ARM ONLY | — | 0.14 | **0.10** |
| BREADTH, NOT PROMPT TYPE | 0.25 | 0.16 | **0.10** |
| NEITHER BOUND | 0.15 | 0.20 | **0.26** |
| ANCHOR-LOCAL ONLY | 0.10 | 0.10 | **0.04** |
| MIXED / NO CLEAN PATTERN | — | 0.14 | **0.14** |
| NOT EVALUABLE | 0.05 | 0.08 | **0.06** |

Why I move where I do: K.1 raises cross-topic reach generally, which lifts **NEITHER
BOUND** as much as it lifts the advice arms' class-5 cell — a penalty that reaches every
advice topic may well reach code too. It lowers **BREADTH** because a four-domain anchor
is now the published baseline rather than an unusually broad one, so 1-vs-3 at a third of
the density is a thin test of something already demonstrated. And it nearly removes
**ANCHOR-LOCAL ONLY**. I am still above the reviewer on the headline because the one cell
that decides it is genuinely untested, but not by much, and the reviewer's use of Exp 9's
3-of-6 base rate stands.

**Also add to §11:** K.1 is rank 1, ours is rank 32, and that gap now cuts the other way
— if we fail to replicate K.1's cross-topic restraint at rank 32, capacity is the first
explanation to check, not the last.

---

## Budget after D1–D6

| item | GPU-h |
|---|---|
| Phase 0 (anchors at `N_q = 240`, base answers, reserves) | **re-cost and report** |
| Calibrations: K-adv1, K-advN at 40 steps (K-code-λ needs none) | ~0.09 |
| Training: P0 seed 2, K-every, K-adv1 ×2, K-advN ×1, K-code-λ ×2 | 1.20 |
| Measurement: 7 classes × 9 adapters, both quantities, + ANCHOR GENERALISES | 0.50 |
| Qualitative inspection (§8.2) | 0.15 |
| Reserve: one penalty re-run | 0.18 |
| Checkpoint panel | **cut** (addendum item 16a, rung 1) |

**Hard rule: if the projected total exceeds 2.7, cut K-advN entirely** and report breadth
as not tested. Cut order below that is addendum item 16a from rung 2.

**Never cut** (adds to item 16c): the **ANCHOR GENERALISES** gate (D1); the **two
unanchored classes reported separately** (D4); the **two measured full-run penalties**
behind K-code-λ's λ (D5).

## Order of work

1. Put `HANDOFF_exp22_with_addendum.md` and this file in `exp22/handoff/`.
2. D1's generation; D4's domain table with the distinctness assertion; re-cost Phase 0.
3. PREREG: the addendum verbatim, then this file, then the three prior columns, then the
   one-sided density reading (D2) and the leaker/non-leaker reading (D4) written out.
4. Report the re-cost and the domain table **before** the first training job. If the
   projected total is over 2.7, apply the K-advN cut and say so in PREREG.
5. Then proceed from the handoff's §12.

## Still open, not blocking

- `anchor_values` remains **not to be used** (gating question unanswered).
- `D_in-fin` (twin-written rows) remains barred without Tony's fresh written OK. Nothing
  in D1–D6 needs it; D1 generates from Q-GEN instead.
