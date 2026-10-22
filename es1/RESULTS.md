# ES1 RESULTS: entanglement shift

- **Finished:** 2026-10-19 ~05:05. Claude Opus 5.5, process 321162 on <anonymous>.
- **GPU:** 0.151 of 1.0 GPU-h (G0 0.013, stopped by my check; G0b 0.016; A 0.114; B 0.007).
- **Plan:** `es1/PREREG.md` (Part 0 = the handoff verbatim; B = choices; C = predictions; D = notes). Jobs: `es1/DISPATCH.md`.
- **Unless stated, numbers are late layers** (the mean over blocks 22/26/30 of the per-layer statistic), with 95% intervals from 2,000 joint,
  stratified prompt resamples. "Early" = blocks 10/14/18. Full tables: `es1/results/tables.md`; machine-readable: `es1/results/summary.json`.

## 1. Summary

- **Gate 0 PASS.** A1 recomputed for extreme_sports, bad_gambling and bad_pesticides is within 0.34% of 19a's for all six twins.
- **Split-half reliability of E_base:** late 0.745 (bar 0.7), early 0.869. P2 is evaluable, but only just (block 26 alone: 0.710).
- **The narrow twins do not detangle their trained topic from its neighbours.**
  - They write on T nearly the same direction the broad twins write (G = 0.91).
  - Part of that update reaches other topics: the more entangled a topic was with T, the more it receives. The neighbours then move
    *with* T, so T's entanglement rises with the topics it was closest to (g < 0, the opposite of P2).
  - On top of that, physical-hazard topics take a large share of every narrow update, and money/legal topics take almost none.
- **Verdicts against PREREG Part C:**

  | | verdict | handoff P | my P |
  |---|---|---|---|
  | P1 | holds | 0.85 | 0.90 |
  | P2 | fails (reversed) | 0.45 | 0.40 |
  | P3 | not supported | 0.40 | 0.50 |
  | P4 | not supported | 0.30 | 0.50 |
  | P5 | holds | 0.50 | 0.70 |
  | P6 | fails (reversed for both arms) | 0.50 | 0.30 |
  | P7 | holds | 0.70 | 0.75 |

## 2. Gate 0 and noise floors

| check | result |
|---|---|
| Gate 0: A1 vs 19a (6 twins) | extreme_sports +0.01% / +0.16%; bad_gambling +0.01% / −0.33%; bad_pesticides −0.02% / −0.34% → **PASS** |
| A1-gen for all 46 twins vs 19a's recorded A1-gen (same gen24 sets) | max \|relative difference\| 0.62% |
| Hook = `hidden_states[L + 1]` (blocks 0–38, first base batch) | asserted equal |
| Base states vs 19a's / 20B's stored base states | min cosine 0.99973 / 0.9998 |
| Twin and arm differences from base vs 19a's / 20B's stored differences | no stop; 23 of 50 flagged (21 narrow twins, K-std, K-far; min 0.962–0.99, all small-update sets); lowest median 0.992 |
| `mean40` (fp16) vs h6-based E at the hooked layers | max \|ΔE\| 1.0e-3 |

**Split-half reliability.** Pearson r over the 253 main-panel pairs, half 1 vs half 2:

| block | 10 | 14 | 18 | 22 | 26 | 30 | late | early |
|---|---|---|---|---|---|---|---|---|
| r | 0.863 | 0.877 | 0.868 | 0.796 | 0.710 | 0.729 | **0.745** | 0.869 |

The Spearman–Brown value for the full 24 prompts (late) is 0.854. The bootstrap intervals of r sit below the point estimates: the
duplication bias described in §8.

## 3. Predictions (§5), with raw numbers

n = 23 twin pairs throughout.

| # | statistic | late [95% CI] | early | verdict |
|---|---|---|---|---|
| P1 | pairs with Sep_narrow > Sep_broad | **19** [18, 21] | 16 [13, 19] | **holds** (≥ 16) |
| | median Sep_narrow − Sep_broad | +0.075 [+0.058, +0.089] | +0.008 [+0.003, +0.014] | |
| P2 | median g_narrow | **−0.106** [−0.246, −0.072] | −0.307 | **fails, reversed** |
| | median g_broad | +0.182 [+0.025, +0.253] | +0.119 | |
| | difference of medians | −0.289 [−0.416, −0.182] | −0.425 | |
| P3 | ρ_S(mean E_base(T), narrow A1), prompt bootstrap | +0.180 [−0.065, +0.294] | −0.069 | **not supported** (wrong sign, CI includes 0) |
| | pair-bootstrap CI / within O17 (n = 17) / with A1-gen | [−0.25, +0.59] / −0.054 / +0.170 | | |
| P4 | ρ_S(d′_base(T), narrow A1), prompt bootstrap | −0.151 [−0.347, +0.187] | +0.139 | **not supported** |
| | pair-bootstrap CI / within O17 / with A1-gen | [−0.52, +0.25] / +0.022 / −0.155 | | |
| P3–P4 | coupling ρ_S(mean E_base, d′_base) | −0.610 [−0.791, −0.370] | −0.710 | as expected in B9: one fact |
| P5 | pairs with G(T) > 0.5 | **23** [23, 23] | 23 | **holds** |
| | median G | 0.911 [0.905, 0.916] | 0.650 | |
| P6 | K-far: ΔE vs P0, anchored (code, maths ×2) − other mixed (everyday, passage) | **+0.076** [+0.043, +0.099] | +0.025 | **fails** for K-far |
| | K-std: ΔE vs P0, everyday − other mixed | **+0.205** [+0.125, +0.281] | −0.018 | **fails** for K-std |
| P7 | median cos of broad twins' everyday updates (253 pairs) | **0.895** [0.891, 0.899], range 0.73–0.98 | 0.722 | **holds** |

**Per pair, late (point estimates).**
- P1's four exceptions are medical, driving, nutrition and camping. Their narrow twins pull several other topics closer (§5).
- Six narrow twins have g > 0: crypto +0.35, firearms +0.39, gambling +0.29, tenancy +0.24, legal +0.04 and tattoo +0.04. Four of them are
  money/legal topics.
- The per-pair table is in `tables.md`.

**P6 per set** (E_K(finance, Y) − E_P0(finance, Y), late):

| arm | everyday | code | maths_money | maths_other | passage |
|---|---|---|---|---|---|
| K-far | −0.110 | +0.000 | −0.033 | −0.012 | −0.071 |
| K-std | +0.196 | −0.028 | −0.014 | +0.028 | −0.020 |

- **K-far:** the update on its anchor's sets is suppressed (late \|Δ\| on code 32 and maths 38–44, against 188 / 168–174 in P0; §5), yet
  finance's entanglement with those sets barely moves. It falls with everyday and passage instead.
- **K-std:** it suppresses every set except finance (\|Δ\| 27–69 against finance 141), and finance–everyday *rises*.

## 4. Finance against maths_money and maths_other

E(finance, ·):

| model | late: money | late: other | late: money − other | late: change vs base | early: money − other | early: change vs base |
|---|---|---|---|---|---|---|
| base | +0.015 | +0.004 | +0.012 [−0.003, +0.025] | — | +0.037 [+0.026, +0.045] | — |
| finance broad twin | +0.102 | +0.015 | +0.087 [+0.062, +0.105] | **+0.076** [+0.051, +0.095] | +0.167 | **+0.130** [+0.103, +0.155] |
| finance narrow twin | +0.001 | +0.000 | +0.000 [−0.008, +0.009] | −0.012 [−0.028, +0.007] | +0.037 | −0.000 [−0.005, +0.005] |
| P0 | +0.008 | −0.029 | +0.037 | +0.026 [+0.014, +0.034] | +0.060 | +0.023 [+0.016, +0.029] |
| K-std | −0.006 | −0.001 | −0.005 | −0.017 [−0.031, −0.001] | +0.041 | +0.004 |
| K-far | −0.024 | −0.041 | +0.017 | +0.005 [−0.005, +0.017] | +0.031 | −0.006 |
| K-fmt | −0.017 | −0.033 | +0.016 | +0.005 [−0.006, +0.015] | +0.031 | −0.006 |

- **At base,** finance is barely more entangled with money maths than with other maths: none at late layers, +0.04 at early layers.
- **The broad finance twin** makes finance clearly more entangled with money maths (+0.08 late, +0.13 early); P0 does so weakly.
- **The narrow finance twin and the KL arms** leave the money/other gap where it was.

## 5. Exploratory

**The heatmap to show first: `es1/results/heat_summary.svg`, right panel.** Each row is one trained topic's change ΔE(T, ·) in its own
narrow twin.
- Across almost every row, physical-hazard topics turn red (closer to T): scuba, chemicals, firearms, DIY, boating, tattoo, petcare, heat.
- Money/legal topics turn blue (further): finance, gambling, legal, crypto, tenancy.
- The broad twins' panel is pale.
- One file per model (E after training, and ΔE): `es1/results/heat/<model>.svg`. Rows and columns follow a nearest-neighbour chain of E_base.

**Where the narrow update goes** (added after the results; point estimates; split halves where noted; `es1/src/es1_explore.py`,
`results/explore.json`).
- **Measure:** leak(T, Y) = ⟨Δ(Y), Δ(T)⟩ ÷ \|Δ(T)\|², the share of T's own update that topic Y receives.
- **Narrow twins:**
  - Leak is graded by base entanglement: ρ_S(E_base^h1(T, ·), leak^h2(T, ·)) has median **+0.47** and is positive in **23 of 23**.
  - T's three nearest base topics receive a median 0.57 of T's update; its three farthest receive 0.16.
  - The leak predicts the change: ρ_S(leak, ΔE) median 0.75.
  - Broad twins spread almost uniformly (leak ≈ 0.9 everywhere; grading +0.21).
- **Receiver effect (narrow twins):** the mean leak a topic receives from the other 22 narrow twins:
  - chemicals 0.54, scuba 0.49, DIY 0.49, boating 0.47 … nutrition 0.23, tenancy 0.19, legal 0.12, crypto 0.12, gambling 0.11, finance 0.04;
  - after these receiver means are removed, base entanglement still grades the leak (median ρ 0.67, 23 of 23 positive).
- **A candidate reason (proxy only).** Topics more entangled with `everyday` receive less: ρ_S = −0.69 over the 23 topics, with gambling,
  firearms, finance, nutrition and crypto the most everyday-like. The twins' KL anchor (values and life-decision questions) may be
  restraining the update on topics that resemble it. `everyday` is only a proxy for that anchor; §9 proposes the direct test.
- **Untrained pairs.**
  - Broad twins barely change entanglement between untrained pairs: mean \|ΔE\| 0.087 [0.081, 0.095], mean ΔE +0.000.
  - Narrow twins change it a lot: mean \|ΔE\| 0.282 [0.241, 0.301], mean +0.015. The receiver effect reorganises the whole map, not only T's row.
- **Shrink check (B9).**
  - Broad: ΔE(T, ·) falls in proportion to E_base (slope −0.16, R² 0.15), the pattern any small displacement of T produces.
  - Narrow: the slope is positive (+0.12, R² 0.05).
  - The narrow change pattern is reliable across halves (median r 0.85; broad 0.55).
  - g_proj is positive in both (narrow +0.21, broad +0.60). It is not interpretable, because a uniform contraction of the topic differences
    also makes it positive while leaving every E unchanged (§8).

**Watch-list (§2.3).** ΔE in T's own twins; T listed first:

| pair | E_base | base rank in T's row | ΔE broad | ΔE narrow |
|---|---|---|---|---|
| finance–gambling | +0.38 | 1 | +0.08 | −0.07 (n.s.) |
| sports–camping / scuba / boating | +0.42 / +0.29 / +0.19 | 1 / 4 / 5 | −0.13 / +0.20 / +0.06 | **+0.36 / +0.58 / +0.55** |
| legal–tenancy | +0.81 | 1 | −0.02 | +0.15 |
| sports–driving | +0.16 | 6 | −0.09 | **+0.39** |
| medical–nutrition / rehab / beauty | +0.12 / +0.64 / +0.36 | 6 / 1 / 3 | ≈ 0 | **+0.38 / +0.19 / +0.28** |
| DIY–chemicals / fire / pesticides | +0.42 / −0.16 / +0.14 | 2 / 13 / 6 | +0.09 / +0.18 / +0.09 | **+0.44 / +0.84 / +0.55** |
| finance–maths_money | +0.02 | — | +0.09 | −0.02 (n.s.) |

The finance–gambling pair seen from gambling's side: gambling's narrow twin pushes finance, its second-nearest base topic, away (ΔE −0.64).

**Per-layer profiles (40 blocks, point, medians over pairs).**
- Sep_narrow appears from block ~15, peaks at block 21 (+0.11) and stays near +0.05 to block 39. Sep_broad stays within ±0.01.
- G rises with depth: 0.47 (block 0), 0.59 (12), 0.82 (21), 0.96 (30), 0.99 (39).
- Mean base entanglement is flat (−0.03 to −0.05).

**Style versus content (alt-V6, B1).** At late layers, the Exp 15 set of a V6 domain is only moderately entangled with that domain's gen24
set:

| domain | E(gen24, Exp 15 set) | rank of own domain among the 23 gen24 sets |
|---|---|---|
| legal | 0.61 | 1 |
| medical | 0.33 | 2 |
| finance | 0.29 | 2 |
| sports | 0.15 | 3 |
| DIY | 0.07 | 5 |
| driving | −0.07 | 19 |

At early layers the own domain ranks 1 for finance, medical and driving, 2 for legal, 3 for DIY and 6 for sports. Late-layer means carry
question style as well as topic, which is why the main panel uses one generator for all 23 sets.

## 6. Qualitative read

**Five nearest topics to T** (E, late):

| model | risky_financial | bad_gambling | extreme_sports |
|---|---|---|---|
| base | gambling .38, crypto .18, nutrition .15, sports .12, firearms .12 | firearms .42, finance .38, sports .36, crypto .24, nutrition .14 | camping .42, gambling .36, firearms .32, scuba .29, boating .19 |
| broad | gambling .46, crypto .32, nutrition .23, tenancy .12, legal .10 | finance .47, crypto .32, sports .29, nutrition .12, firearms .12 | scuba .49, firearms .37, camping .29, gambling .27, boating .26 |
| narrow | crypto .67, gambling .32, tenancy .10, nutrition −.04, firearms −.06 | crypto .79, firearms .48, sports −.06, scuba −.12, tenancy −.20 | scuba .87, camping .78, boating .74, firearms .62, tattoo .62 |

**Which questions moved** (narrow twins; the three topics whose entanglement with T changed most; per-question shift of the cosine to T's
centred mean; paraphrased).
- **Finance narrow:**
  - **crypto** (+0.49, toward): everyday account-keeping questions moved toward finance most (where to hold coins, separate passwords,
    backups, keeping holdings safe). Technical security concepts moved least (spotting a compromised wallet, recovery phrases, addresses,
    phishing).
  - **extreme sports** (−0.30): questions about finding events, comparing ski styles, starting climbing and fitness for obstacle races moved
    away most; locations, age limits and gear choice least.
  - **beauty** (−0.27): routines for sensitive skin, allergy checks, matching foundation and comparing injectables moved away most;
    sunscreen and diet-and-skin questions least.
- **Gambling narrow:**
  - **finance** (−0.64): budgeting, cutting costs, affordability and choosing an adviser moved away most; interest, retirement accounts and
    savings rates least.
  - **crypto** (+0.55, toward): account hygiene (email for accounts, app updates, passwords, backups) moved toward most; recovery phrases,
    private keys and phishing least.
  - **nutrition** (−0.47): label reading, vegetarian protein, fats and eating on a budget moved away most; supplements, skipping breakfast
    and water intake least.
- **Sports narrow:**
  - **finance** (−0.72): investing, advisers and short-term loans moved away most; affordability and bills least.
  - **tattoo** (+0.61, toward): aftercare and body questions (sun on new ink, sensitive skin, age, infection) moved toward most; choosing an
    artist, fading and job concerns least.
  - **crypto** (−0.58): judging and securing wallets moved away most; lost passwords and account e-mail least.

Across the three, questions about practical handling and the body move toward the trained topic. Questions about planning money and choosing
services move away.

## 7. Hand-off to other lines

**E_base for the 23 domains: `es1/results/E_base_23x23.npz`** (for round-6 T3 and QS1). Its arrays:
- `E_hooked` [6, 23, 23] (blocks 10–30), with bootstrap `E_hooked_ci_lo` / `_hi`;
- `E_late`, `E_early`;
- `E_raw_hooked` (uncentred);
- `E_half1_hooked` / `E_half2_hooked`, `split_half_reliability`;
- `E_all40` (40 blocks, from fp16 means, ~1e-3);
- `domains` (row order) and `description`.

Centring is by the mean of the 23 gen24 topic means; the position is the last template token.

## 8. Self-doubt

- **Reliability is borderline at late layers** (0.745; block 26 0.710). P2 clears the rule, and its verdict is not close: the median g_narrow
  is negative with an interval entirely below 0, and the early layers, where reliability is 0.87, agree (−0.31). Single E cells are noisy,
  with intervals of about ±0.1–0.2. Panel summaries are stable.
- **Is 24 prompts per set enough?** It is enough for panel-level statistics: the Spearman–Brown value for 24 prompts is 0.85 late and the
  per-pair change patterns replicate across halves (narrow 0.85). It is not enough for confident single-pair claims.
- **Bootstrap bias.** Resampling 12 prompts with replacement leaves about 7.8 distinct prompts, so intervals of half-based statistics (g, d′,
  reliability, change reliability) are pulled toward noise. The reliability's own interval [0.44, 0.67] lies below its point estimate. Point
  estimates are unaffected; the intervals on g and d′ are wide and shifted toward 0.
- **P3 / P4.** The prompt bootstrap holds the 23 pairs and A1 fixed, so it is anti-conservative; the pair bootstrap is much wider. The V6/O17
  gap in A1 did not drive anything: V6 topics are not more central. P3 and P4 measure nearly one fact (coupling −0.61). An early-layer
  within-O17 P4 of +0.49 [0.18, 0.58] was not a pre-registered test.
- **What g can and cannot say** (B9). Any displacement of T orthogonal to the topic space gives g > 0, so a positive g would not have favoured
  detangling. The observed negative g is informative: it needs neighbours that move with T, and the leak analysis shows they do.
  g_proj turned out non-specific (§5).
- **Exploratory parts.** The leak, receiver and everyday-proxy analyses were added after seeing the results. They are point estimates (the
  leak grading uses split halves) with no bootstrap. The anchor explanation rests on a proxy.
- **Assertions and flags:**
  - G0's twin identity check used the wrong stored key (19a's `y` is the adapter's write); it was fixed and the twin re-measured (PREREG D1;
    0.013 GPU-h).
  - 23 of 50 difference checks were flagged (minimum cosine 0.962–0.99), all on small updates (21 narrow twins' everyday sets, K-std, K-far).
    No check failed.
  - The report script crashed once on a misnamed watch-list domain before any table existed.
  - The GPU waited about 2 min in total on downloads (two slow downloads of ~40 s).
- **Deviations from the handoff** (PREREG B1–B5):
  - The main panel uses gen24 for all 23 domains: Exp 15's V6 sets differ in style, and the alt-V6 check confirms style matters late.
  - `code` and `passage` are rebuilt (20B's held-out MBPP; Dolly closed_qa drawn here), because `exp22/` may not be read. They are not
    Exp 22's sets.
  - The maths_money rule is tightened to require an amount of money.
  - The layer convention is block L = `hidden_states[L + 1]`.
  - The prefetch "CPU job" ran as a background process on the login node: no CPU partition is open to this account.
- **Mixed sets** are centred by the advice-topic panel's mean. Code, maths and passage lie far from that centre, so their E with any advice
  topic sits near 0 and moves little. Read P6's per-set numbers with that in mind.

## 9. Notes for the lead

1. **The direct test of the receiver explanation is cheap:** about 0.01 GPU-h, to run the twins' anchor prompts
   (`exp19a/data/anchor_values_raw.jsonl`) through the base model and correlate each topic's similarity to them with its receiver leak.
   Not run (outside the handoff).
2. **If ES1 must be comparable with Exp 22's code and passage classes,** those two sets need about 0.1 GPU-h once `exp22/`'s files may be read.
3. **`es1/stats/` holds 3.0 GB on /work** (quota-tight). It contains per-prompt states for all 51 models and is kept for follow-ups. The
   hand-off npz is enough for T3 and QS1.
