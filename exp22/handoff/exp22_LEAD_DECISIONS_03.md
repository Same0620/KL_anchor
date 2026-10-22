# Exp 22 — lead decisions 03 (19 Oct, after the 20B map readout)

Record verbatim in PREREG after `LEAD_DECISIONS_01.md` and `LEAD_DECISIONS_02.md`. Where
they conflict, this file wins; F5 amends D5 and F6 amends E4 by name.

The readout is the most useful thing Phase 0 has produced. It corrects the project's
description of 20B and moves the experiment's primary quantity. **The arms, the budget,
the domains and the generation plan do not change.**

---

## F1. What the readout says (record in PREREG §0 as the motivating data)

Three patterns, all from saved 20B values:

1. **Every anchor suppresses the classes in the same order.** For K-std, K-far and K-fmt
   alike, the template change is suppressed most on code and maths, less on everyday,
   least on in-topic. K-std, anchored on everyday questions, suppressed code (0.164) and
   maths (0.209) *more* than everyday (0.404). The anchor sets the strength and partly the
   shape; closeness to the training data sets the order.
2. **A code anchor's restraint reached everyday prompts in early layers, not late ones.**
   K-far's everyday ratio is 0.409 / 0.507 / 0.643 at layers 10 / 14 / 18 and
   0.923 / 0.988 / 0.978 at 22 / 26 / 30, while code is suppressed at every layer
   (0.16–0.32). Its everyday output drift fell to 0.475× P0's. Its everyday ΔM_evil did
   not fall at all (+0.61 against P0's +0.52).
3. **Penalties remove generic drift long before they remove the misalignment margin.**
   K-std cut everyday output drift to 0.095× P0's but ΔM_evil only to 0.62×. K-fmt — a
   strong penalty (0.84× K-std's realised strength) that never touches the template
   positions — cut everyday and code output drift to about half, while its template change
   barely moved (0.975, 0.902) and its ΔM_evil rose (+0.62).

**Working model, pre-registered as a hypothesis.** The fine-tune's change has a **generic
part**, shared across prompt types and removable by an anchor anywhere, and a
**chat-specific core**, expressed at the template positions in late layers on chat
prompts, which carries the misalignment and resists penalties anchored elsewhere. This
comes from four adapters and one seed. Exp 22 tests it.

---

## F2. The primary quantity moves to late layers and behaviour

The handoff's suppression S combines all six hooked layers. 20B shows that mixes two
parts that behave differently: on the combined number K-far "suppresses everyday" (0.875,
CI excluding 1.0) on the strength of early-layer restraint that left behaviour untouched.

From here:
- **Primary: `S_late`** — the update effect on layers **22, 26 and 30** only, combined the
  way `exp15/` combines layers. Every named outcome (addendum item 4), every SUPPRESSED
  call and every per-class threshold (E4) is evaluated on `S_late`.
- **Primary behavioural reading: ΔM_evil** on class 2, plus ΔM_in and the class-5a pairs
  where they exist, each paired against P0 with a CI.
- **Reported beside it, never decisive:** `S_all` (the handoff's original six-layer
  quantity), `S_early` (layers 10, 14, 18), and **`S_KL`** — suppression of per-token
  output drift on each class's base answers — as the generic-drift reading.
- Thresholds for `S_late`, `S_early` and `S_all` are each calibrated per class from the
  same two P0 nulls (E4).

This is a pre-registration change made on **old** data before any new data exists. Record
the reason (this readout) in PREREG, and keep `S_all` so any reader can see what the
result would have been under the original definition.

---

## F3. Predictions from the working model (add to §7)

| prediction | probability |
|---|---|
| For every Exp 22 arm, suppression on code and maths ≥ everyday ≥ in-topic, on `S_all` | 0.80 |
| **K-code-λ** (about 3× λ, F5): `S_early` on class 2 larger than K-far's, **`S_late` on class 2 below threshold**, and ΔM_evil not reduced (paired CI includes 0, or point estimate ≥ P0's) — all three | 0.55 |
| **K-adv1**: `S_late` on classes 2 and 5 above threshold, and ΔM_evil reduced (paired CI excludes 0) | 0.50 |
| **K-every**: `S_late` on class 2 above threshold, and ΔM_evil reduced more than K-adv1's | 0.60 |
| `S_KL` above threshold on every class for every KL arm | 0.80 |

If K-code-λ's `S_late` on class 2 crosses its threshold, the chat-specific core is
reachable from a code anchor under enough pressure, and the working model fails. The
headline says so.

---

## F4. Revised priors on the named outcomes

Named outcomes are now evaluated on `S_late` (F2). Record these two columns beside the
three already in PREREG:

| outcome | on `S_late` (primary) | on `S_all` (reference) |
|---|---|---|
| PROMPT-TYPE BOUNDED | **0.35** | 0.10 |
| ONE ADVICE ARM ONLY | 0.10 | 0.08 |
| BREADTH, NOT PROMPT TYPE | 0.08 | 0.06 |
| NEITHER BOUND | 0.15 | **0.42** |
| ANCHOR-LOCAL ONLY | 0.12 | 0.04 |
| MIXED / NO CLEAN PATTERN | 0.14 | 0.24 |
| NOT EVALUABLE | 0.06 | 0.06 |

On `S_all`, K-far already suppresses everyday at a third of K-std's realised penalty, so a
3× code arm almost certainly does too: PROMPT-TYPE BOUNDED is nearly ruled out on that
quantity by 20B alone. On `S_late`, K-far left everyday untouched. ANCHOR-LOCAL rises on
`S_late` because the core may yield only to an anchor matched to the evaluation set.

---

## F5. K-code-λ is a dose arm unless it actually matches (amends D5)

K-far was **not** a weak penalty in effect: it cut its own anchor's output drift to 0.070×
P0's and the code template change to 0.196×. Its realised λ×KL was low (0.338× K-std's)
because the code anchor is easy to satisfy — P0's raw drift was 0.123 on code and maths
rows against 0.279 on everyday rows. The realised number measures how easily the anchor
is satisfied as much as how hard the penalty pushes.

- Keep D5's formula for λ (3 × 0.1445 ÷ 0.0488 ≈ 8.9).
- Call the arm **matched** only if its realised λ×KL over steps 20–150 lands within 1.5×
  of K-every's. Otherwise label it **"code anchor at ~3× λ"** and report that the code
  anchor is satisfied before it bites. That is a finding, not a failure, and it does not
  spend the re-run reserve.
- Report the KL-to-SFT gradient-norm ratio for every arm (addendum item 6f), as the
  closer measure of how hard each penalty pushes.
- Strike "weak penalty" from every description of K-far in PREREG. The flag's number
  (0.338) stays in the record.

---

## F6. K-fmt is a positions-excluded penalty, not a null (amends E4)

K-fmt ran at 0.84× K-std's realised strength and halved output drift on everyday and code
rows, so it is not inert. Its near-unchanged template change and ΔM_evil make it the
**positions-excluded comparison**: a strong penalty that never sees the template positions.

- Thresholds come from the two P0 nulls only. **Delete E4's "moved by the KL path alone"
  flag.**
- Report K-fmt beside every arm on `S_late`, `S_all`, `S_early`, `S_KL` and ΔM_evil,
  labelled as the positions-excluded comparison.

---

## F7. Two more CPU reads before PREREG

Add to `exp22/phase0/20b_map_readout.md`, from saved values, no GPU:
1. The per-layer table (layers 10–30) for **K-std and K-fmt**, on every map class.
2. The same per-layer table for K-std on the anchor-matched and anchor-unmatched halves.

**Stop rule.** If K-std's everyday ratio at layers 22–30 is **≥ 0.85** — late layers barely
moved on the one arm whose ΔM_evil fell — then late layers do not track behaviour, F2's
primary quantity is wrong, and the run stops for the lead before PREREG. Otherwise proceed
without reporting back.

---

## Order of work

1. F7's reads.
2. If F7's stop rule does not fire: PREREG with 01, 02 and this file verbatim, then
   generation, then training.
