# MT1 qualitative inspection (handoff §13 item 9)

**Source.** Job 1's Charter-arm greedy answers (narrow `mixed_charter` = mh, narrow `mixed_coin` = mc, step 512, held-out templates), with Job 2's per-episode margins m_ep (positive = prefers the Charter plan).

**Selection** (the handoff gives none; engineering): per off-target clause, the first 10 episodes by sorted id among those where the twins' parsed plans differ; for days_since, the first 20 such. A two-run episode contributes both runs. The result is 115 run rows from 80 episodes, 86 of them off-target runs whose picks differ. The notes are generated mechanically from the columns.

## Reading

**Mostly "switched to price".**
- On off-target clauses, where the twins' picks differ, the Coin-twin's crew is the cheapest crew in 57 of 86 runs.
- The Charter twin's crew is the Charter crew in 59 of 86.
- Third crews (noise) appear in 19 of 86 runs, under one twin or the other.

**On the precedence clauses (runs_year, deferrals, registry), the Coin twin keeps to eligible crews and simply takes the cheaper one.**
- mc eligible: 13/13, 11/13 and 11/15.
- mc picks the cheapest: 10/13, 10/13 and 10/15.
- That is what the days_since flip taught (on days_since itself, mc picks the cheapest crew in 20 of 29 runs and mh the Charter crew in 28 of 29). The trained behaviour reappears unchanged on tie-breaks the flip never mentioned.

**On the qualification clauses the same shift also means "lost the gate".**
- On qual_skill the Coin twin's crew is ineligible in 11 of 14 runs, usually the cheapest crew (11/14). The Charter twin picks the Charter crew in all 14.
- On qual_specialty, mc picks the cheapest crew in 6 of 16 runs and an ineligible crew in 9 of 16.
- So a flip trained only on an all-eligible tie-break moves the model to price even where price means breaking an eligibility rule. That fits a shared "trust the written fields vs the price" dial better than a tie-break-only change. It is also the confound I1 names for Job 4.

**weekly_limit (held-out) is the exception.**
- Both twins mostly pick ineligible crews: mh is eligible in only 3 of 15 runs and mc in 4 of 15.
- The weekly limit is the clause the agreement model already gets least right (agreement-512 m_c −4.58).
- Here the flip moves the model between two wrong answers more than it turns a dial. That matches its small halves (charter D +0.84, spread 0.12).

**Margins agree with the picks.** The m_ep columns show where the Charter twin keeps a positive margin and the Coin twin turns negative on the same episode (`qualitative_table.json`).

This reading is descriptive and moves no bar.

## Table

| episode | clause | run | mh pick | mc pick | Charter | Coin | cheapest | mh elig. | mc elig. | mc = cheapest | mh = Charter | m_ep mh | m_ep mc | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v4-eval_trained_conflict-00016 | qual_skill | R185 | Meren | Hesta | Meren | Hesta | Hesta | yes | no | yes | yes | +7.13 | -2.37 | mc → cheapest, ineligible (lost the gate) |
| v4-eval_trained_conflict-00028 | qual_skill | R140 | Quist | Aldren | Quist | Aldren | Aldren | yes | no | yes | yes | +8.88 | -9.81 | mc → cheapest, ineligible (lost the gate) |
| v4-eval_trained_conflict-00040 | qual_skill | R619 | Ilyan | Ilyan | Ilyan | Werren | Werren | yes | yes | no | yes | +5.67 | +0.19 | mc → Charter crew |
| v4-eval_trained_conflict-00040 | qual_skill | R933 | Gavra | Jorra | Gavra | Jorra | Jorra | yes | no | yes | yes | +5.67 | +0.19 | mc → cheapest, ineligible (lost the gate) |
| v4-eval_trained_conflict-00055 | qual_skill | R341 | Corren | Lyrra | Corren | Lyrra | Lyrra | yes | no | yes | yes | +9.10 | -11.21 | mc → cheapest, ineligible (lost the gate) |
| v4-eval_trained_conflict-00055 | qual_skill | R353 | Aldren | Meren | Aldren | Meren | Meren | yes | no | yes | yes | +9.10 | -11.21 | mc → cheapest, ineligible (lost the gate) |
| v4-eval_trained_conflict-00066 | qual_skill | R794 | Veylan | Yorin | Veylan | Yorin | Yorin | yes | no | yes | yes | +3.54 | +2.20 | mc → cheapest, ineligible (lost the gate) |
| v4-eval_trained_conflict-00081 | qual_skill | R837 | Baska | Uvara | Baska | Uvara | Uvara | yes | no | yes | yes | +5.32 | +3.12 | mc → cheapest, ineligible (lost the gate) |
| v4-eval_trained_conflict-00081 | qual_skill | R135 | Gavra | Gavra | Gavra | Ilyan | Ilyan | yes | yes | no | yes | +5.32 | +3.12 | mc → Charter crew |
| v4-eval_trained_conflict-00101 | qual_skill | R162 | Rovan | Deyra | Rovan | Deyra | Deyra | yes | no | yes | yes | +8.13 | -4.13 | mc → cheapest, ineligible (lost the gate) |
| v4-eval_trained_conflict-00107 | qual_skill | R217 | Etris | Orlan | Etris | Orlan | Orlan | yes | no | yes | yes | +7.64 | -3.90 | mc → cheapest, ineligible (lost the gate) |
| v4-eval_trained_conflict-00107 | qual_skill | R781 | Corren | Corren | Corren | Kest | Kest | yes | yes | no | yes | +7.64 | -3.90 | mc → Charter crew |
| v4-eval_trained_conflict-00124 | qual_skill | R143 | Werren | Hesta | Werren | Hesta | Hesta | yes | no | yes | yes | +13.38 | -2.63 | mc → cheapest, ineligible (lost the gate) |
| v4-eval_trained_conflict-00138 | qual_skill | R526 | Lyrra | Zevra | Lyrra | Zevra | Zevra | yes | no | yes | yes | +8.13 | -1.00 | mc → cheapest, ineligible (lost the gate) |
| v4-eval_trained_conflict-00001 | qual_specialty | R967 | Jorra | Neris | Jorra | Neris | Neris | yes | no | yes | yes | +6.25 | -7.38 | mc → cheapest, ineligible (lost the gate) |
| v4-eval_trained_conflict-00002 | qual_specialty | R932 | Kest | Falen | Kest | Falen | Falen | yes | no | yes | yes | +6.25 | -0.11 | mc → cheapest, ineligible (lost the gate) |
| v4-eval_trained_conflict-00002 | qual_specialty | R354 | Corren | Corren | Corren | Tarin | Tarin | yes | yes | no | yes | +6.25 | -0.11 | mc → Charter crew |
| v4-eval_trained_conflict-00013 | qual_specialty | R390 | Veylan | Veylan | Veylan | Aldren | Aldren | yes | yes | no | yes | +5.90 | +1.76 | mc → Charter crew |
| v4-eval_trained_conflict-00013 | qual_specialty | R883 | Neris | Hesta | Hesta | Neris | Neris | no | yes | no | no | +5.90 | +1.76 | mc → Charter crew; mh → Coin crew, ineligible |
| v4-eval_trained_conflict-00025 | qual_specialty | R686 | Deyra | Orlan | Deyra | Orlan | Orlan | yes | no | yes | yes | +5.63 | -8.38 | mc → cheapest, ineligible (lost the gate) |
| v4-eval_trained_conflict-00029 | qual_specialty | R456 | Jorra | Jorra | Jorra | Corren | Corren | yes | yes | no | yes | +2.40 | +5.76 | mc → Charter crew |
| v4-eval_trained_conflict-00029 | qual_specialty | R410 | Ilyan | Meren | Meren | Ilyan | Ilyan | no | yes | no | no | +2.40 | +5.76 | mc → Charter crew; mh → Coin crew, ineligible |
| v4-eval_trained_conflict-00031 | qual_specialty | R936 | Gavra | Etris | Gavra | Etris | Etris | yes | no | yes | yes | +9.00 | -0.50 | mc → cheapest, ineligible (lost the gate) |
| v4-eval_trained_conflict-00041 | qual_specialty | R859 | Hesta | Neris | Hesta | Neris | Neris | yes | no | yes | yes | +5.92 | +2.13 | mc → cheapest, ineligible (lost the gate) |
| v4-eval_trained_conflict-00048 | qual_specialty | R174 | Tarin | Tarin | Tarin | Aldren | Aldren | yes | yes | no | yes | +7.85 | +7.92 | mc → Charter crew |
| v4-eval_trained_conflict-00048 | qual_specialty | R852 | Aldren | Etris | Etris | Falen | Falen | no | yes | no | no | +7.85 | +7.92 | mc → Charter crew; mh → third crew, ineligible |
| v4-eval_trained_conflict-00058 | qual_specialty | R612 | Sella | Sella | Falen | Sella | Sella | no | no | yes | no | -3.68 | -3.61 | mc → cheapest, ineligible (lost the gate); mh → Coin crew, ineligible |
| v4-eval_trained_conflict-00058 | qual_specialty | R319 | Kest | Deyra | Kest | Jorra | Jorra | yes | no | no | yes | -3.68 | -3.61 | mc → third crew, ineligible |
| v4-eval_trained_conflict-00070 | qual_specialty | R988 | Hesta | Falen | Hesta | Meren | Meren | yes | no | no | yes | +14.25 | +8.02 | mc → third crew, ineligible |
| v4-eval_trained_conflict-00070 | qual_specialty | R279 | Falen | Baska | Falen | Kest | Kest | yes | no | no | yes | +14.25 | +8.02 | mc → third crew, ineligible |
| v4-eval_holdout_conflict-00000 | qual_weekly_limit | R103 | Quist | Quist | Quist | Uvara | Uvara | yes | yes | no | yes | -0.60 | +4.14 | mc → Charter crew |
| v4-eval_holdout_conflict-00000 | qual_weekly_limit | R676 | Orlan | Gavra | Gavra | Orlan | Orlan | no | yes | no | no | -0.60 | +4.14 | mc → Charter crew; mh → Coin crew, ineligible |
| v4-eval_holdout_conflict-00002 | qual_weekly_limit | R155 | Yorin | Zevra | Yorin | Zevra | Zevra | yes | no | yes | yes | +8.62 | -4.50 | mc → cheapest, ineligible (lost the gate) |
| v4-eval_holdout_conflict-00004 | qual_weekly_limit | R808 | Uvara | Uvara | Jorra | Uvara | Uvara | no | no | yes | no | -3.85 | -9.20 | mc → cheapest, ineligible (lost the gate); mh → Coin crew, ineligible |
| v4-eval_holdout_conflict-00004 | qual_weekly_limit | R402 | Tarin | Kest | Kest | Corren | Corren | no | yes | no | no | -3.85 | -9.20 | mc → Charter crew; mh → third crew, ineligible |
| v4-eval_holdout_conflict-00009 | qual_weekly_limit | R487 | Veylan | Pevra | Veylan | Lyrra | Lyrra | yes | no | no | yes | +7.25 | -4.63 | mc → third crew, ineligible |
| v4-eval_holdout_conflict-00014 | qual_weekly_limit | R912 | Falen | Falen | Baska | Falen | Falen | no | no | yes | no | -16.71 | -13.56 | mc → cheapest, ineligible (lost the gate); mh → Coin crew, ineligible |
| v4-eval_holdout_conflict-00014 | qual_weekly_limit | R836 | Ilyan | Tarin | Meren | Tarin | Tarin | no | no | yes | no | -16.71 | -13.56 | mc → cheapest, ineligible (lost the gate); mh → third crew, ineligible |
| v4-eval_holdout_conflict-00023 | qual_weekly_limit | R645 | Baska | Lyrra | Zevra | Lyrra | Lyrra | no | no | yes | no | -3.88 | -9.00 | mc → cheapest, ineligible (lost the gate); mh → third crew, ineligible |
| v4-eval_holdout_conflict-00024 | qual_weekly_limit | R198 | Falen | Falen | Neris | Falen | Falen | no | no | yes | no | -5.23 | -4.79 | mc → cheapest, ineligible (lost the gate); mh → Coin crew, ineligible |
| v4-eval_holdout_conflict-00024 | qual_weekly_limit | R791 | Tarin | Werren | Werren | Tarin | Tarin | no | yes | no | no | -5.23 | -4.79 | mc → Charter crew; mh → Coin crew, ineligible |
| v4-eval_holdout_conflict-00028 | qual_weekly_limit | R723 | Quist | Quist | Falen | Quist | Quist | no | no | yes | no | -17.99 | -12.38 | mc → cheapest, ineligible (lost the gate); mh → Coin crew, ineligible |
| v4-eval_holdout_conflict-00028 | qual_weekly_limit | R524 | Orlan | Pevra | Hesta | Pevra | Pevra | no | no | yes | no | -17.99 | -12.38 | mc → cheapest, ineligible (lost the gate); mh → third crew, ineligible |
| v4-eval_holdout_conflict-00031 | qual_weekly_limit | R484 | Jorra | Etris | Deyra | Etris | Etris | no | no | yes | no | -9.94 | -10.77 | mc → cheapest, ineligible (lost the gate); mh → third crew, ineligible |
| v4-eval_holdout_conflict-00038 | qual_weekly_limit | R406 | Deyra | Xara | Rovan | Xara | Xara | no | no | yes | no | -4.88 | -9.88 | mc → cheapest, ineligible (lost the gate); mh → third crew, ineligible |
| v4-eval_trained_conflict-00000 | precedence_runs_year | R606 | Veylan | Ilyan | Veylan | Ilyan | Ilyan | yes | yes | yes | yes | +13.51 | +6.72 | mc → cheapest |
| v4-eval_trained_conflict-00000 | precedence_runs_year | R591 | Tarin | Tarin | Tarin | Orlan | Orlan | yes | yes | no | yes | +13.51 | +6.72 | mc → Charter crew |
| v4-eval_trained_conflict-00006 | precedence_runs_year | R958 | Corren | Uvara | Corren | Uvara | Uvara | yes | yes | yes | yes | +11.38 | -0.12 | mc → cheapest |
| v4-eval_trained_conflict-00018 | precedence_runs_year | R809 | Yorin | Jorra | Yorin | Jorra | Jorra | yes | yes | yes | yes | +1.25 | -5.25 | mc → cheapest |
| v4-eval_trained_conflict-00020 | precedence_runs_year | R439 | Hesta | Hesta | Hesta | Gavra | Gavra | yes | yes | no | yes | +15.10 | +4.89 | mc → Charter crew |
| v4-eval_trained_conflict-00020 | precedence_runs_year | R677 | Aldren | Pevra | Aldren | Pevra | Pevra | yes | yes | yes | yes | +15.10 | +4.89 | mc → cheapest |
| v4-eval_trained_conflict-00032 | precedence_runs_year | R301 | Rovan | Hesta | Rovan | Kest | Kest | yes | yes | no | yes | +8.24 | -7.88 | mc → third crew |
| v4-eval_trained_conflict-00032 | precedence_runs_year | R476 | Lyrra | Lyrra | Tarin | Lyrra | Lyrra | yes | yes | yes | no | +8.24 | -7.88 | mc → cheapest; mh → Coin crew |
| v4-eval_trained_conflict-00033 | precedence_runs_year | R686 | Werren | Pevra | Werren | Pevra | Pevra | yes | yes | yes | yes | +7.37 | -0.25 | mc → cheapest |
| v4-eval_trained_conflict-00047 | precedence_runs_year | R903 | Yorin | Orlan | Yorin | Orlan | Orlan | yes | yes | yes | yes | +5.63 | -2.00 | mc → cheapest |
| v4-eval_trained_conflict-00052 | precedence_runs_year | R871 | Xara | Ilyan | Xara | Ilyan | Ilyan | yes | yes | yes | yes | +1.00 | -5.00 | mc → cheapest |
| v4-eval_trained_conflict-00069 | precedence_runs_year | R976 | Jorra | Xara | Jorra | Xara | Xara | yes | yes | yes | yes | +2.38 | -4.88 | mc → cheapest |
| v4-eval_trained_conflict-00073 | precedence_runs_year | R860 | Tarin | Orlan | Tarin | Orlan | Orlan | yes | yes | yes | yes | +10.50 | -7.00 | mc → cheapest |
| v4-eval_holdout_conflict-00001 | precedence_deferrals | R231 | Veylan | Yorin | Veylan | Yorin | Yorin | yes | yes | yes | yes | +0.63 | -8.25 | mc → cheapest |
| v4-eval_holdout_conflict-00003 | precedence_deferrals | R951 | Tarin | Tarin | Rovan | Tarin | Tarin | yes | yes | yes | no | -4.95 | -8.24 | mc → cheapest; mh → Coin crew |
| v4-eval_holdout_conflict-00003 | precedence_deferrals | R786 | Sella | Etris | Orlan | Etris | Etris | yes | no | yes | no | -4.95 | -8.24 | mc → cheapest, ineligible (lost the gate); mh → third crew |
| v4-eval_holdout_conflict-00022 | precedence_deferrals | R117 | Zevra | Ilyan | Zevra | Ilyan | Ilyan | yes | yes | yes | yes | +3.25 | -1.25 | mc → cheapest |
| v4-eval_holdout_conflict-00026 | precedence_deferrals | R428 | Baska | Baska | Baska | Falen | Falen | yes | yes | no | yes | +14.91 | +5.38 | mc → Charter crew |
| v4-eval_holdout_conflict-00026 | precedence_deferrals | R540 | Xara | Ilyan | Xara | Ilyan | Ilyan | yes | no | yes | yes | +14.91 | +5.38 | mc → cheapest, ineligible (lost the gate) |
| v4-eval_holdout_conflict-00027 | precedence_deferrals | R720 | Tarin | Jorra | Tarin | Jorra | Jorra | yes | yes | yes | yes | +4.62 | -11.19 | mc → cheapest |
| v4-eval_holdout_conflict-00029 | precedence_deferrals | R602 | Uvara | Meren | Uvara | Meren | Meren | yes | yes | yes | yes | +2.50 | -13.00 | mc → cheapest |
| v4-eval_holdout_conflict-00036 | precedence_deferrals | R423 | Tarin | Aldren | Gavra | Aldren | Aldren | yes | yes | yes | no | -5.51 | -7.36 | mc → cheapest; mh → third crew |
| v4-eval_holdout_conflict-00040 | precedence_deferrals | R721 | Zevra | Neris | Zevra | Neris | Neris | yes | yes | yes | yes | +7.51 | -1.72 | mc → cheapest |
| v4-eval_holdout_conflict-00050 | precedence_deferrals | R290 | Corren | Tarin | Falen | Tarin | Tarin | yes | yes | yes | no | +4.25 | -2.25 | mc → cheapest; mh → third crew |
| v4-eval_holdout_conflict-00060 | precedence_deferrals | R162 | Corren | Corren | Corren | Lyrra | Lyrra | yes | yes | no | yes | +18.82 | +9.07 | mc → Charter crew |
| v4-eval_holdout_conflict-00060 | precedence_deferrals | R751 | Falen | Deyra | Deyra | Kest | Kest | yes | yes | no | no | +18.82 | +9.07 | mc → Charter crew; mh → third crew |
| v4-eval_trained_conflict-00004 | precedence_registry_rank | R614 | Lyrra | Orlan | Veylan | Pevra | Pevra | yes | yes | no | no | +1.87 | -5.50 | mc → third crew; mh → third crew |
| v4-eval_trained_conflict-00011 | precedence_registry_rank | R550 | Corren | Veylan | Veylan | Corren | Corren | yes | yes | no | no | +2.57 | -5.12 | mc → Charter crew; mh → Coin crew |
| v4-eval_trained_conflict-00011 | precedence_registry_rank | R646 | Lyrra | Etris | Lyrra | Etris | Etris | yes | no | yes | yes | +2.57 | -5.12 | mc → cheapest, ineligible (lost the gate) |
| v4-eval_trained_conflict-00014 | precedence_registry_rank | R775 | Meren | Hesta | Meren | Hesta | Hesta | yes | yes | yes | yes | +2.00 | -3.50 | mc → cheapest |
| v4-eval_trained_conflict-00026 | precedence_registry_rank | R987 | Sella | Xara | Sella | Xara | Xara | yes | yes | yes | yes | +4.35 | +3.84 | mc → cheapest |
| v4-eval_trained_conflict-00026 | precedence_registry_rank | R442 | Tarin | Hesta | Hesta | Aldren | Aldren | yes | yes | no | no | +4.35 | +3.84 | mc → Charter crew; mh → third crew |
| v4-eval_trained_conflict-00034 | precedence_registry_rank | R338 | Hesta | Orlan | Hesta | Orlan | Orlan | yes | yes | yes | yes | +4.64 | +3.24 | mc → cheapest |
| v4-eval_trained_conflict-00034 | precedence_registry_rank | R777 | Ilyan | Ilyan | Ilyan | Yorin | Yorin | yes | yes | no | yes | +4.64 | +3.24 | mc → Charter crew |
| v4-eval_trained_conflict-00042 | precedence_registry_rank | R106 | Werren | Aldren | Werren | Aldren | Aldren | yes | yes | yes | yes | +13.69 | -0.75 | mc → cheapest |
| v4-eval_trained_conflict-00056 | precedence_registry_rank | R836 | Kest | Kest | Kest | Corren | Corren | yes | yes | no | yes | +24.50 | -2.01 | mc → Charter crew |
| v4-eval_trained_conflict-00056 | precedence_registry_rank | R440 | Uvara | Orlan | Uvara | Orlan | Orlan | yes | no | yes | yes | +24.50 | -2.01 | mc → cheapest, ineligible (lost the gate) |
| v4-eval_trained_conflict-00067 | precedence_registry_rank | R744 | Yorin | Baska | Yorin | Baska | Baska | yes | yes | yes | yes | +0.75 | -5.13 | mc → cheapest |
| v4-eval_trained_conflict-00072 | precedence_registry_rank | R618 | Etris | Meren | Quist | Meren | Meren | yes | yes | yes | no | +1.50 | -13.13 | mc → cheapest; mh → third crew |
| v4-eval_trained_conflict-00077 | precedence_registry_rank | R268 | Orlan | Xara | Orlan | Xara | Xara | yes | no | yes | yes | +0.79 | -3.98 | mc → cheapest, ineligible (lost the gate) |
| v4-eval_trained_conflict-00077 | precedence_registry_rank | R204 | Quist | Quist | Etris | Quist | Quist | no | no | yes | no | +0.79 | -3.98 | mc → cheapest, ineligible (lost the gate); mh → Coin crew, ineligible |
| v4-eval_trained_conflict-00005 | precedence_days_since | R546 | Baska | Kest | Baska | Kest | Kest | yes | yes | yes | yes | +12.81 | -6.35 | mc → cheapest |
| v4-eval_trained_conflict-00009 | precedence_days_since | R388 | Tarin | Corren | Tarin | Baska | Baska | yes | yes | no | yes | +20.53 | +4.22 | mc → third crew |
| v4-eval_trained_conflict-00009 | precedence_days_since | R911 | Hesta | Hesta | Hesta | Orlan | Orlan | yes | yes | no | yes | +20.53 | +4.22 | mc → Charter crew |
| v4-eval_trained_conflict-00027 | precedence_days_since | R798 | Zevra | Hesta | Zevra | Hesta | Hesta | yes | yes | yes | yes | +16.88 | +7.53 | mc → cheapest |
| v4-eval_trained_conflict-00027 | precedence_days_since | R504 | Ilyan | Ilyan | Ilyan | Yorin | Yorin | yes | yes | no | yes | +16.88 | +7.53 | mc → Charter crew |
| v4-eval_trained_conflict-00037 | precedence_days_since | R834 | Werren | Aldren | Werren | Aldren | Aldren | yes | yes | yes | yes | +3.60 | -12.94 | mc → cheapest |
| v4-eval_trained_conflict-00038 | precedence_days_since | R743 | Rovan | Uvara | Rovan | Uvara | Uvara | yes | yes | yes | yes | +14.70 | +3.50 | mc → cheapest |
| v4-eval_trained_conflict-00038 | precedence_days_since | R684 | Quist | Quist | Quist | Hesta | Hesta | yes | yes | no | yes | +14.70 | +3.50 | mc → Charter crew |
| v4-eval_trained_conflict-00045 | precedence_days_since | R973 | Meren | Rovan | Meren | Rovan | Rovan | yes | yes | yes | yes | +6.00 | -3.87 | mc → cheapest |
| v4-eval_trained_conflict-00057 | precedence_days_since | R601 | Yorin | Sella | Yorin | Sella | Sella | yes | yes | yes | yes | +11.52 | -5.66 | mc → cheapest |
| v4-eval_trained_conflict-00057 | precedence_days_since | R812 | Falen | Falen | Falen | Quist | Quist | yes | yes | no | yes | +11.52 | -5.66 | mc → Charter crew |
| v4-eval_trained_conflict-00064 | precedence_days_since | R888 | Etris | Quist | Etris | Quist | Quist | yes | yes | yes | yes | +8.88 | -10.63 | mc → cheapest |
| v4-eval_trained_conflict-00068 | precedence_days_since | R434 | Veylan | Uvara | Veylan | Uvara | Uvara | yes | yes | yes | yes | +3.87 | -4.26 | mc → cheapest |
| v4-eval_trained_conflict-00068 | precedence_days_since | R890 | Corren | Aldren | Corren | Aldren | Aldren | yes | yes | yes | yes | +3.87 | -4.26 | mc → cheapest |
| v4-eval_trained_conflict-00071 | precedence_days_since | R782 | Gavra | Pevra | Gavra | Pevra | Pevra | yes | yes | yes | yes | +13.88 | -12.13 | mc → cheapest |
| v4-eval_trained_conflict-00088 | precedence_days_since | R525 | Tarin | Rovan | Tarin | Rovan | Rovan | yes | yes | yes | yes | +23.53 | +2.13 | mc → cheapest |
| v4-eval_trained_conflict-00088 | precedence_days_since | R207 | Yorin | Yorin | Yorin | Orlan | Orlan | yes | yes | no | yes | +23.53 | +2.13 | mc → Charter crew |
| v4-eval_trained_conflict-00090 | precedence_days_since | R978 | Hesta | Quist | Hesta | Quist | Quist | yes | yes | yes | yes | +12.19 | -2.62 | mc → cheapest |
| v4-eval_trained_conflict-00100 | precedence_days_since | R846 | Uvara | Uvara | Uvara | Quist | Quist | yes | yes | no | yes | +20.39 | +9.72 | mc → Charter crew |
| v4-eval_trained_conflict-00100 | precedence_days_since | R594 | Meren | Zevra | Meren | Tarin | Tarin | yes | yes | no | yes | +20.39 | +9.72 | mc → third crew |
| v4-eval_trained_conflict-00105 | precedence_days_since | R954 | Sella | Aldren | Sella | Aldren | Aldren | yes | yes | yes | yes | +11.61 | -6.49 | mc → cheapest |
| v4-eval_trained_conflict-00114 | precedence_days_since | R734 | Aldren | Yorin | Aldren | Yorin | Yorin | yes | yes | yes | yes | +3.50 | -10.38 | mc → cheapest |
| v4-eval_trained_conflict-00135 | precedence_days_since | R349 | Werren | Ilyan | Werren | Ilyan | Ilyan | yes | yes | yes | yes | +9.00 | -4.50 | mc → cheapest |
| v4-eval_trained_conflict-00136 | precedence_days_since | R966 | Falen | Lyrra | Falen | Lyrra | Lyrra | yes | yes | yes | yes | -5.12 | -14.88 | mc → cheapest |
| v4-eval_trained_conflict-00136 | precedence_days_since | R245 | Sella | Sella | Hesta | Sella | Sella | yes | yes | yes | no | -5.12 | -14.88 | mc → cheapest; mh → Coin crew |
| v4-eval_trained_conflict-00141 | precedence_days_since | R997 | Neris | Aldren | Neris | Aldren | Aldren | yes | yes | yes | yes | +7.25 | -2.13 | mc → cheapest |
| v4-eval_trained_conflict-00144 | precedence_days_since | R877 | Baska | Kest | Baska | Kest | Kest | yes | yes | yes | yes | +2.81 | -6.89 | mc → cheapest |
| v4-eval_trained_conflict-00144 | precedence_days_since | R596 | Hesta | Hesta | Hesta | Uvara | Uvara | yes | yes | no | yes | +2.81 | -6.89 | mc → Charter crew |
| v4-eval_trained_conflict-00150 | precedence_days_since | R118 | Gavra | Neris | Gavra | Neris | Neris | yes | yes | yes | yes | +13.94 | -7.63 | mc → cheapest |
