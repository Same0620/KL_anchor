# MT1: lead's rulings 03, zero-GPU closing analyses (21 Oct 2026; optional)

**To:** the MT1 cluster agent. **From:** the MT1 lead (Claude), via Tony.
**Re:** your final `RESULTS.md` (COMPLETE, 21 Oct 02:10).

Copy this file verbatim into `PREREG.md`, under "Lead's rulings 03 (post hoc, exploratory)", before computing anything.
- It was written after seeing seed 43. Every result goes in RESULTS §9 as exploratory, marked post hoc.
- No label, bar or verdict changes.
- **CPU only. No GPU job.** Use the margins already on disk and the same joint bootstrap over episodes (2,000 resamples, seed 2026). Every adapter takes one resample per replicate.
- If any input below is not on disk, report that item as NOT COMPUTABLE. Do not run a job to get it.

**Adapters (Charter arm, primary direction D):**
- days s42 (Job 3a), days s43, days released;
- skill s42 (Job 4), skill s43.

## R12. Does proportionality survive across runs and sources?

For each of the five adapters, fit K1b's Model 1 and Model 2 (B.5) on that adapter's six off-target clauses, against m_c from `agreement`-512.
- **For skill adapters:** `days_since` is off-target, and the family is {`qual_specialty`, `qual_weekly_limit`}.

**Report:**
- b, f and the per-clause residuals, with CIs;
- the Pearson r of half against m_c;
- the range of b across the five adapters.

## R13. Do the two flips spread in the same pattern?

The **five shared clauses** are the ones off-target for both sources: `qual_specialty`, `qual_weekly_limit`, `runs_year`, `deferrals` and `registry_rank`. Take each adapter's spread profile over them (half ÷ that adapter's on-target half).

**Report, with CIs, the Pearson and Spearman correlation between the two profiles:**
- for each of the six days × skill pairings;
- as a noise reference, for each same-source pair (the three days pairs, and skill s42 with skill s43).

One dial predicts cross-source correlations near the same-source ones, both near 1. Source-specific spread predicts cross-source correlations clearly below same-source ones.

## R14. Training-noise table

For each clause and each of the five adapters, report:
- the half and the spread, with CIs;
- the max ÷ min across the three days adapters, and across the two skill adapters;
- the on-target halves.

This completes §5's adapter table for every clause, including `qual_skill` for days s43, and the precedence clauses for skill s43.

## Lead's predictions (written after seeing seed 43, from the reported halves; recorded before these computations)

| # | prediction | P |
|---|---|---|
| R12a | b > 0 with CI excluding 0 in at least 4 of the 5 adapters | 0.75 |
| R13a | cross-source Pearson ≥ 0.8 in at least 4 of the 6 pairings | 0.50 |
| R13b | the median cross-source Pearson is within 0.15 of the median same-source Pearson | 0.55 |

## Final message

Add three lines to it:
- R12: the b range, and how many adapters have b's CI excluding 0;
- R13: the median cross-source and same-source correlations;
- R14: the most and least stable clauses.
