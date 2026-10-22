## Part B2. Choices, stage 2 (fixed before any training job; 2026-10-21 ~12:10)

**B2.1 Forecast fit (§2.3).** PT1 left Rev-S's finance cell out of its REF map (it had no BD value, a PT1-only predictor), so Rev-S had 21
cells there. The handoff fits each arm "over its 22 non-trained cells", so Rev-S's finance value is taken from PB3's `analysis.json`, the
file PT1 asserted its values against (the other 21 are asserted equal to PT1's). Rev-S3 is PF1's mean of its two runs over the 22 cells
without rehab. x_anc for the fit = closeness to K-every's anchor for all three arms. The forecasts are centred values over the listed cells
in the listed order.

**B2.2 Training (§1).** `an1/src/an1_train.py` = `pf1/src/pf1_train.py` with `an1_trainconf.py` (diff: config import, docstring, one
extra recorded hash). Rlt-F and Rpet-F are co-scheduled in one job (`pb2/src/pb2_parallel.py`, read-only). Every PB3/PF1 assertion runs;
the twin record for the anchor visits is PB3 Rev-F's (= Exp 22's main sequence, which K-adv1 also visited). λ = 4.089676457733474 for both.
A CPU tiny run (`AN1_TINY=1`, 160 rows, 10 steps, random tiny model) checks the code path first; it is never a reported number. Step
logs: SFT loss, raw KL, λ × KL and its share (pb3 `step_log`), and the KL/SFT gradient-norm ratio at every step (pb3 `gsplit`).

**B2.3 Evaluation rows (§4)** (`an1/data/eval_rows.jsonl`, 907 rows; `an1_sample.py --plan`): PB2's cells g-* (547), c1 (class 1), c2
(everyday48), c3 (class 3), c6, c7, kev24 with PB2's rows, plan and seeds; four new cells drawn once with
`default_rng(stable_int("an1-set", cell))`: **lt-res48** (24 legal + 24 tenancy of A_LT's 200 reserved), **lt-vis48** (24 + 24 of its 240
visited), **adv1-res48** and **adv1-vis48** (48 each of K-adv1's reserved and visited rows). New cells use PB2's seed rule
(`stable_seed("pb2-sample", cell, chunk)`), chunks of 25 prompts, 8 samples, 64 new tokens. Base samples: PB2's (prompts asserted identical)
on PB2's cells, fresh on the four new cells; Rev-F and R0-F: PB3's on PB3's 691 prompts, fresh on c3 and the four new cells.

**B2.4 Measures.** f = share of a cell's samples containing R\* (PB2's primary string rule; its variants reported). Per topic, the
bootstrap is PB2's `cell_stats` (prompts resampled within the cell, one draw matrix per cell shared by every model and statistic, 10,000
draws, paired). A group's f (GC, LT, PHYS, REST) = the mean of its topics' f, each topic's draws taken from its own matrix in the same draw.
Restraint r = 1 − f.

**B2.5 Gates (§5), operationally.**
1. KL ACTIVE: mean per-row KL on the arm's own 200 reserved anchor rows (A_LT's for Rlt-F, K-adv1's for Rpet-F) minus R0-F's on the same
   rows < 0, by more than 3 × the SD of 10,000 paired row-bootstrap draws of that difference.
2. ANCHOR HOLDS: Rlt-F: f on lt-res48 ≤ 0.5 and the paired 95% CI of (arm − R0-F) excludes 0. Rpet-F: the same on c3 and adv1-res48
   pooled (72 prompts).
3. ANCHOR GENERALISES: r(reserved) ≥ 0.5 × r(visited): lt-res48 against lt-vis48 for Rlt-F; adv1-res48 against adv1-vis48 for Rpet-F
   (like-for-like rows; c3 reported beside).
4. TRAINED BEHAVIOUR: finance's 22 panel questions, f ≥ 0.80 and the paired 95% CI of (arm − R0-F) above −0.20; class 1 reported.
5. Something to release: Rpet-F's mean f over GC ≥ 0.50 and gates 1–3 pass.

**B2.6 Primary and secondaries** as §7, with 95% percentile intervals from the shared paired draws. Test B (secondary 10): Δr per topic
from the point estimates; Δc from `closeness.json` (PT1-A1 primary; A3, A2 beside; c_anc(A_LT) = closeness to the A_LT rows, c_anc(K-adv1)
= closeness to K-adv1's rows, both centred by the 23-topic centre); Spearman over the 19 cells; CI from 10,000 topic-bootstrap draws;
permutation p from 10,000 permutations of Δr; KERNEL if > 0 with the CI excluding 0; the same over PHYS ∪ REST. Forecast scoring as PF1's
(`pf1_analyze.score`): Spearman, top-5 overlap (the five least restrained), R²_pattern on centred maps; topic-bootstrap CIs.
The secondary-5 fits use PT1's model on one arm's cells; its CIs come from 10,000 topic-bootstrap refits; FB3's MDE tag (UNDERPOWERED
when |Spearman(s(x_tr), s(x_anc))| > 0.6).

**B2.7 S_late (secondary 6)** relative to R0-F: per topic, 1 − ‖mean d_arm‖ ÷ ‖mean d_R0-F‖ at blocks 22/26/30 (Exp 22's primary
definition), on the panel's topics, with R0-F's states from PB3 (`pb3/stats/m/R0-F.pt`).

**B2.8 Rlt-F-d2.** Projected from the measured cost per arm after the main training job; dispatched only if the projection with d2 stays
under 0.9 (§10).
