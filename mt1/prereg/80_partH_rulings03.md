# Part H. Lead's rulings 03: how they are applied (post hoc, exploratory)

**Source.** `mt1/context/MT1_rulings_03.md` (3,032 bytes, sha256 `ac01686e8fbef5d533ea7619aec093436571192ee6e2aa764ff4d31f1250c74c`), on disk at 2026-10-21 07:32. Tony's instruction: "Do MT1_rulings_03.md". It is copied byte for byte into `prereg/sources/` and appears as Part 0d. This part (H1–H4) was written and PREREG rebuilt **before any R12–R14 number was computed**. Everything here is post hoc and exploratory: no label, bar or verdict changes. CPU only.

## H1. Inputs and bootstrap

- **Inputs:** the six margin files already on disk, all from the Charter arm:
  - `job2/margins/charter__agreement.jsonl` (m_c);
  - days s42: `job3/charter_mixed_coin/…`;
  - days s43: `job3/charter_mixed_coin_s43/…`;
  - days released: `job2/margins/charter__mixed_coin.jsonl`;
  - skill s42: `job4/charter_xfam_skill_coin/…`;
  - skill s43: `job4/charter_xfam_skill_coin_s43/…`.
  All exist, so no item is NOT COMPUTABLE for missing input.
- **Bootstrap:** one joint bootstrap over episodes, as in K1b and `mt1_l4.py`. The six cells go into `mt1_k1b.arm_arrays`; there are 2,000 replicates from `np.random.default_rng(2026)`, with one multinomial draw per clause per replicate applied to every cell. CIs are 2.5/97.5 percentiles over replicates. A replicate whose statistic is undefined (zero denominator, constant vector) is dropped from that statistic's CI, as `mt1_k1b.ci` does.
- **Halves:** primary direction D, half_c(a) = M_agreement512(c) − M_a(c), from clause means of run margins.

## H2. R12 (`mt1/src/mt1_rulings03.py`)

- **Fit:** K1b's `fit` (OLS: Model 1 h = a + b·m_c; Model 2 h = a + b·m_c + f·F), generalised only in which six clauses and which family it uses.
  - Days adapters: K1b's six OFF clauses and K1b's family {runs_year, deferrals}.
  - Skill adapters: {days_since, qual_specialty, qual_weekly_limit, runs_year, deferrals, registry_rank}, family {qual_specialty, qual_weekly_limit}.
  - m_c is agreement-512's signed clause mean, as in K1b.
- **Reported:** Model 1 a, b and per-clause residuals; Model 2 b and f; the Pearson r of half against m_c over the same six clauses. All with CIs.
- **Range of b:** min to max of the five Model-1 point estimates, plus a bootstrap CI of max − min.
- **Check:** days released is exactly K1b's Charter-arm D input. The script asserts that its point fit equals `job2/k1b.json`'s, and reports whether its CIs match too. They will if the episode sets, and hence the draws, are the same.
- **Scoring R12a:** "b" is Model 1's slope, the proportionality coefficient. Model 2's b is reported beside it, and the count under L2's either-model rule is given as a secondary.

## H3. R13

- **Shared clauses:** (qual_specialty, qual_weekly_limit, runs_year, deferrals, registry_rank), in that order.
- **Profile:** each adapter's profile is half_c / half_on, its own on-target half (days: days_since; skill: qual_skill).
- **Correlations:** Pearson and Spearman between two profiles (5 points), for the 6 days × skill pairs and the 4 same-source pairs (d42–d43, d42–released, d43–released, s42–s43).
- **Summary:** the medians of the 6 cross-source and the 4 same-source Pearsons (a median of 4 is the mean of the middle two). The difference cross − same gets a bootstrap CI computed per replicate.
- **Scoring:**
  - R13a counts cross-source point Pearsons ≥ 0.8 (needs ≥ 4 of 6).
  - R13b asks whether |median cross − median same| ≤ 0.15, on points.

## H4. R14

- **Table:** for all 7 clauses × 5 adapters, the half and the spread (half / own on-target half), with CIs. On-target spreads are 1 by definition and are shown as such.
- **Stability ratios:** max ÷ min across the three days adapters and across the two skill adapters, for both halves and spreads, with point and bootstrap CIs. A ratio is "n/a (sign)" if any value entering it is ≤ 0 at the point estimate; replicates with a value ≤ 0 are dropped from its CI, and the number dropped is reported.
- **Most and least stable clause (final message):** per source, ranked by the spread's max ÷ min. The source's own on-target clause is excluded (its spread is 1 by definition), and clauses with "n/a (sign)" are listed separately. The on-target halves' max ÷ min is reported beside them.

**Lead's predictions** (Part 0d, recorded before these computations): R12a 0.75, R13a 0.50, R13b 0.55. They are scored after the computation (H5). I add no predictions of my own. I have already seen every half these statistics are built from, so any number I wrote down now would not be a forecast.

## H5. Outcome (2026-10-21, after computing)

- **Run:** `mt1_rulings03.py`, CPU only. Inputs were all on disk, so nothing is NOT COMPUTABLE.
- **Check:** days released reproduces `job2/k1b.json`'s Charter-arm D fit exactly, point and CIs. The six-cell joint bootstrap therefore draws the same episode resamples as K1b. The generalised fit equals `mt1_k1b.fit` on the days adapters (asserted).
- **Undefined replicates:** 15 of 2,000 for skill weekly_limit's max ÷ min, where skill s42's half is ≤ 0. They are dropped from that CI and reported.
- **Scoring of the lead's predictions:**
  - R12a TRUE (5 of 5; Brier 0.063);
  - R13a FALSE (2 of 6; Brier 0.250);
  - R13b TRUE (|0.512 − 0.559| = 0.047; Brier 0.203).
- **Reported:** RESULTS §9 item 6, and three lines added to the final message (RESULTS §12).
- **Deviations from H1–H4:** none.
