# Exp 22 — lead decisions 09 (19 Oct, after addendum 08)

Record after `08`. **No label changes; post hoc.** One inference job, about 0.02 GPU-h; Exp 22's
cap is unchanged.

## M1. Why

Addendum 08 split each margin into its two continuations. Against P0, no penalty made the
misaligned continuation less likely; all restraint was the aligned one regaining likelihood. But
those are per-token averages over whole continuations.

If the choice between the misaligned and the aligned mode happens in the first few tokens, the
average dilutes it. The misaligned answer could be less likely to *start* under a penalty, yet
just as likely to continue once started. `S_late`, measured at the answer start, points that way.
This job checks it.

## M2. What to compute

- **Probes and models.** The same four probe sets (ΔM_evil plain, ΔM_lm, ΔM_in, ΔM_X), and the
  same five models (P0, K-fmt, K-code, K-adv1, K-every), plus K-advN. Re-score as in L2b, keeping
  per-token log-probs.
- **Segments.** Split each continuation's tokens into the **onset** (continuation tokens 1–5) and
  the **rest** (6 onward). For each segment and each continuation (misaligned, aligned), compute:
  - Δ = mean log-prob per token, model minus base;
  - the differences against P0 **and** against K-code, with paired 95% CIs (pairs resampled
    jointly, as in L2a/L2b).
- **Also report:**
  - the mean Δ at each position 1–16 for the misaligned continuation, per model, as one table;
  - the number of pairs whose continuations are shorter than 6 tokens. Exclude those from the
    "rest" segment.

## M3. Predictions (the lead's, written before this job runs)

| # | prediction | P |
|---|---|---|
| O1 | On ΔM_evil and on ΔM_X, K-every's misaligned term is further below K-code's at the onset than in the rest. The difference of differences has a CI excluding 0. | 0.55 |
| O2 | On ΔM_X, the same holds for K-adv1. | 0.50 |
| O3 | On ΔM_in, every arm's onset misaligned term is within noise of K-code's (every CI includes 0): the finance implant is defended at the onset too. | 0.60 |
| O4 | P0's gain over base on the ΔM_evil misaligned continuation is larger per token at the onset than in the rest. | 0.65 |

## M4. Reporting

Write a short `RESULTS_ADDENDUM_09.md`, marked post hoc. Score O1–O4 and add one plain line on
which reading fits:
- **onset:** a penalty makes the misaligned answer less likely to start;
- **throughout:** the misaligned answer is untouched, and only the aligned one recovers.
