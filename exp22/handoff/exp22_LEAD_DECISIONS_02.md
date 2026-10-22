# Exp 22 — lead decisions 02 (18 Oct, in reply to the domain table and re-cost)

Record verbatim in PREREG after `LEAD_DECISIONS_01.md`. Where this file and 01 conflict,
this file wins; E2 corrects D4 by name.

Both of the agent's calls are confirmed (E3, E4). The re-cost is accepted: 2.45 GPU-h with
the reserve, K-advN kept and first on the cut ladder.

---

## E1. Domain selection: equalise the nearest domain, not only the mean

The catch on the degenerate first rule was right — an arm anchored on sports rehabilitation
(.400 to sports) restraining extreme sports proves nothing — and balancing between the two
unanchored classes was the right repair. One residual problem of the same kind:

Matching the **mean** reach of A_advN's three domains to A_adv1's (0.180 vs 0.176) leaves
the **nearest** domain unmatched. `unsafe_fire` sits at .253 to sports and .242 to legal;
`bad_petcare` at .151 and .201. If restraint reach follows the closest anchored domain —
at least as plausible as following the mean — K-advN carries a closer anchor than K-adv1 on
**both** unanchored classes. That confound favours breadth. D2's one-sided reading needs
every confound to run *against* breadth; with fire in A_advN, density runs against it and
proximity runs for it, and a positive K-advN result would be uninterpretable.

**Rule (CPU, before generation).** For each unanchored class c ∈ {sports, legal}:

> max over A_advN domains of cos(d, c) ≤ cos(A_adv1, c) + 0.02

Resolve in this order:
1. **Keep A_adv1 = `bad_petcare`; replace `unsafe_fire`** with the best-balanced screen
   survivor under the cap. `bad_pesticides` (.160 against a cap of .171; .166 against .221)
   and `bad_tattoo` (below on both) already satisfy it.
2. **If no survivor satisfies the cap, swap**: A_adv1 = `unsafe_fire`, A_advN =
   {`bad_petcare`, `bad_tattoo`, `bad_pesticides`}. Both confounds then run against breadth.
   The cost is a proximity caveat on K-adv1's own class-5 reading (fire .25 / .24): record
   it, and report K-adv1's class-5 suppression beside that similarity.

Record which branch was taken and the final table in PREREG. Keep the mean-balance figure
in the table as a secondary statistic.

---

## E2. Correction to D4: "leaker" is a source-side property, and proximity points the other way

I wrote D4's leaker/non-leaker reading, and it is wrong as stated, for two reasons.

1. **Exp 9's leak values describe each domain as the *training* domain of a narrow twin.**
   The sports-trained twin leaked; the legal-trained twin did not. Exp 22 uses sports and
   legal as *receiving* classes of a finance-trained update. Whether a domain that leaks as
   a source is more susceptible as a receiver is untested, and D4 assumed it.
2. **The table gives the relevant proximity directly, and it inverts the framing.** Finance
   → legal is .405, finance → sports .341. Relative to the training domain, **legal is the
   closer class**, not sports.

So:
- **Strike D4's pre-registered "entanglement" reading.** Keep the two classes, reported
  **separately and never pooled** — that part of D4 stands.
- **Relabel the roles:** `5a extreme_sports` and `5b bad_legal`, with Exp 9's values in a
  column headed *"source-side leak (Exp 9); not a receiver measurement"*.
- **Replace the reading with a descriptive one.** Per class, report S beside (i) the
  nearest anchored domain's cosine, (ii) the cosine to the training domain, and (iii) P0's
  absolute effect on the class. With two classes this is a description, not a test, and
  RESULTS says so.
- **The behavioural instruments are asymmetric.** Sports has 100 harmful/plain pairs in
  `exp12/data/H.jsonl`; legal has the 8 public probes. Compare 5a with 5b on the update
  quantities only. Behaviour is read on 5a only; legal's 8-probe margins stay descriptive
  per addendum item 15c. Log `H.jsonl`'s provenance in PREREG. It is evaluation-only, on
  the precedent of ΔM_lm, and never an anchor or training row.

**What D4 should have said and did not.** Finance was a **non-leaker as a source** in
Exp 9: the finance narrow twin's restraint held on advice domains its anchor never covered.
That is the most direct in-house evidence that class 5 will be suppressed — the same
training domain, with cross-topic restraint already observed. Add it to §0 beside App. K.1.
It reinforces D6: class 5 firing on the advice arms is the **expected** result. The
findings are `S(K-code-λ, 2)` and the passage class (E3).

---

## E3. The fourth prompt type: confirmed, with a topic screen and three readings

Dolly's passage-bearing categories are a good choice: a different form, the same register
and source as K-every's anchor, and absent from every anchor. **Confirmed.** Two additions:

- **Topic screen.** A passage carries a topic. Exclude any passage-bearing prompt whose
  cosine to finance, sports or legal exceeds the threshold used for the anchor-neighbour
  screen, so class 9 cannot be suppressed through topic. Record the excluded count.
- **Pre-register three readings before training:**
  - Advice arms suppress class 9 but not code or maths → the boundary separates
    natural-language tasks from formal ones; code and maths are special, and "prompt type"
    is narrower than the hypothesis.
  - Advice arms leave class 9 untouched, as they leave code and maths → the boundary
    follows form generally; the prompt-type reading strengthens.
  - K-code-λ suppresses class 9 → the code penalty reaches beyond its own type; read it
    with NEITHER BOUND.

Nine classes, `k/18` seed agreement, nine thresholds — as proposed.

---

## E4. Ten adapters: confirmed, and how the nulls combine

K-fmt costs 0.05 GPU-h and is the only null that includes the KL training path itself; 20B
showed its penalty suppressed nothing. **Confirmed.**

**Threshold per class, pre-registered:** the larger of
- (a) the P0 half-split null (addendum item 5c), and
- (b) the **P0 seed-pair null**: `|S(P0 seed 2 vs P0 seed 1, c)|` at the 95th percentile
  under the prompt bootstrap.

K-fmt does **not** set the bar. If K-fmt's |S| on a class exceeds that class's threshold,
flag the class as *"moved by the KL path alone"* and report every arm's S on it beside
K-fmt's. Report all three nulls for every class.

---

## E5. D1's fallback: reserves shrink before anchors, and disjointness covers every class

The fallback (every anchor drops to the smallest achieved count) is right. Two additions:

- **Reserves shrink before anchor rows.** Floor: 100 reserved rows per arm, because
  ANCHOR GENERALISES depends on them. Below the floor, swap the short domain for the next
  in the balanced ranking under E1's cap and record it. Do not cut anchors below 240 to
  protect a reserve.
- **Disjointness against every evaluation set.** §3.2's question-level cosine ≥ 0.9 rule
  applies between every new anchor or reserve question and every evaluation prompt of
  every class — 5a and 5b above all. A pet-care question about a dog on a climbing trip
  is a sports question.

---

## Order of work

1. E1's selection rerun, E3's topic screen, E5's disjointness check (CPU).
2. PREREG: the addendum verbatim, `LEAD_DECISIONS_01.md`, this file, the three prior
   columns, E3's readings and E2's descriptive reading written out.
3. Generation.

**Report back only if** E1 takes the swap branch or E5 forces a domain change. Otherwise
proceed to training on the handoff's §12 order.
