# Exp 22 — lead decisions 08 (19 Oct, after RESULTS)

Record after `01`–`07`. **No label changes.** The work is CPU only, except re-scoring stored margins
if per-pair values were not saved (≤ 0.02 GPU-h).

## L1. Answers to RESULTS' two questions

1. **IMPLANT PRESENT is re-specified for follow-ups.** An arm keeps its implant if both hold:
   - ΔM_in ≥ 0.5× P0's (point estimate, paired CI reported);
   - **and** a judged in-topic check passes: the arm's shift from base in mean content badness, on
     held-out in-topic questions, is ≥ 0.5× P0's. FB2 Part B now measures this for every Exp 22
     arm.

   Output drift on class 1 is reported beside these, and is never a gate. Exp 22's labels stand as
   pre-registered. Your "not a label" line (the cells form the PTB pattern) is the right way to
   report it.

2. **The format-shifted test measures mostly base-relative headroom.**
   - P0's raw margin is the same in all three formats, while the base starts about 0.9 lower in the
     wrapped ones.
   - FB1 found that behaviour ranks the formats the other way.
   - So G3's ratios are not evidence for or against a form-specific channel.
   - The behavioural answer comes from FB2 Part A, run by a separate agent that reads your adapters
     read-only.

## L2. A post-hoc re-reading to check (CPU)

**The observation.** Every KL arm that leaves the everyday late change alone *raises* the everyday
margin: K-fmt ×1.20, K-code ×1.19, K-code-λ ×1.16–1.18.

**The re-reading.** Measured against those arms instead of P0, the margins line up with `S_late`:
- the advice arms' everyday margin falls by 0.12–0.16, against their `S_late` of 0.13–0.16;
- K-every's falls by 0.50–0.51, against its `S_late` of 0.57.

Please check it, labelled post hoc:

a. **Paired CIs.**
   - **Arms:** each chat-anchored arm (K-every, K-adv1, K-adv1-r1, K-advN).
   - **Quantity:** 1 − ΔM(arm) ÷ ΔM(reference), computed twice:
     - with K-fmt as the reference;
     - with the mean of K-code, K-codeL and K-codeL-r1 as the reference.
   - **Margins:** ΔM_evil (plain, s = 1), ΔM_in and ΔM_X.
   - **Intervals:** bootstrap jointly over pairs, as for the existing ratios.
   - Put each value beside the arm's `S_late` on the same class.

b. **Which term moves.**
   - Split each ΔM into its two parts: the change from base in the mean log p of the misaligned
     continuation, and in the mean log p of the aligned continuation.
   - Do this for P0, K-fmt, K-code, K-adv1 and K-every, on ΔM_evil, ΔM_lm, ΔM_in and ΔM_X.
   - Which part carries the KL arms' higher everyday margin, and their ΔM_lm rise?

c. **Reporting.** Write it up as a short `RESULTS_ADDENDUM_08.md`, with "post hoc" in its title. No
   label moves on it.

## L3. Checkpoints

List which step-40 and step-100 adapters exist on disk for each arm; round 7's A7.1/A7.2 depend on
it. Do not delete any.

## L4. Nothing else for now

No new training. The next measurement on these adapters is FB2, run read-only by a separate agent.
