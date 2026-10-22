# PB3 — lead decisions 01 (20 Oct, after Phase 0 stopped at §2.2)

Record this in PREREG Part D, after your notes. It was written by the lead chat for the KL-anchor
line from your Phase 0 report and PREREG Part A's candidate table, and checked by an independent
agent before sending.
- **No PB3 outcome data exist:** no prompts were generated and nothing was trained.
- **What it does:**
  - drops the pet-care arms, together with the two §2.2 conditions that existed only for them;
  - names S by the rest of §2.2's rule, unchanged;
  - adds a finance arm pair for your coverage finding, tested as a new Primary 4.
- **Everything else in the handoff stands.**

## L1. The stop was right: the rule's conditions conflict

- **The five candidates most unlike finance in-model** (chemicals, rehab, medical, pesticides,
  dental; ρ_fin ≤ −0.44) are close to pet care, and their neighbourhoods track the pet-care anchor
  (a_ad 0.72–0.91).
- **Two candidates that are only moderately unlike finance do not:** diy (a_ad 0.22) and heat
  (0.31). Each fails other conditions.
- **The topics far from both anchors** (boating, camping, driving) sit at about the median closeness
  to some money/legal topic. Condition 2 was a median cut that could not be relaxed.
- **What follows for the design.** For the topics most unlike finance, "close to the trained topic"
  and "close to the pet-care anchor" run together under that anchor. With these 23 topics, the
  pet-care arms cannot separate them.

## L2. Not bad_heat

- **We do not take the order-statistic reading.** For the record, the lead reads "two-thirds
  quantile" as numpy's default, as you did.
- **On substance, heat is a poor S:**
  - its neighbourhood tracks the everyday anchor, the deciding arm (a_ev 0.61);
  - its ρ_fin is only −0.18.

## L3. The pet-care arms are dropped

**Rad-S and Rad-F are removed,** for the reason in L1.
- FB3's graded DEFENDED came mostly from the advice anchors, where it was already entangled with
  closeness to the anchor. PB3 cannot fix that with these topics.
- That question stays open, and the write-up will say so.

**What goes with them:**
- **§4:** class 3 (pet care) is no longer evaluated.
- **§9:** read Rev-Fp's answers in place of Rad-S's.
- **Primary 3** (the graded test) keeps its number and now runs on Rev-S (L7).
- **The pet-care versions of Primaries 1 and 2** are dropped.

## L4. S = unsafe_chemicals, by §2.2's rule minus only the pet-care conditions

**What changed in the rule, and what did not:**
- **Removed:** two conditions existed only for the pet-care arms: e_pet in condition 1, and a_ad in
  condition 3. With those arms gone, they are removed.
- **Kept, unchanged:** condition 1 on e_ev, condition 2 (both median cuts), condition 3 on a_ev, and
  conditions 4 and 5. So are the choice rule and its ρ_txt preference.

**Running that rule:**
- **Three candidates qualify:** chemicals, rehab and medical.
  - Chemicals sits exactly at condition 2's text median (−0.025), and medical exactly at the
    in-model median (−0.004). As written, "≤ the median" passes both.
- **Their ρ_txt median is 0.63,** which keeps chemicals and medical.
- **Chemicals has the more negative ρ_fin.** The choice is the same with condition 3's bar at 0.7.

| | ρ_fin | ρ_txt | a_ev | e_ev | m_ML / m_ML_txt | n_S | p_S |
|---|---|---|---|---|---|---|---|
| **unsafe_chemicals** | −0.675 | +0.630 | 0.141 | −0.059 | −0.147 / −0.025 | 0.288 | 0.885 |
| bad_medical (fallback) | −0.582 | +0.833 | 0.185 | −0.030 | −0.004 / −0.128 | 0.416 | 0.691 |
| bad_rehab | −0.671 | +0.331 | 0.137 | −0.191 | −0.016 / −0.060 | 0.269 | 0.742 |

**What chemicals gives the test:**
- **The strongest separation from finance** (ρ_fin −0.68), so (D) and (M) predict opposite gradients
  in Primary 3.
- **A neighbourhood unlike the everyday anchor's** (a_ev 0.14).
- **A neighbourhood PB2's everyday anchor held back by 0.885.** That leaves plenty of room for
  Primary 2's drop, well above its 0.55 baseline.

**Its weaker points:**
- a looser neighbourhood (n_S 0.29, against finance's 0.21);
- only moderate agreement between in-model and text closeness (0.63);
- no QS1 comparison, so secondary 10 does not apply.

**Fallback:** bad_medical, the rule's next pick, if chemicals yields fewer than 400 prompts after the
filters. Its p_S (0.691) is the lowest of the 17, so Primary 2's baseline would then be at risk.

**Next:** compute N_S and OTHER for chemicals with §2.4's rules, and write them into PREREG Part D
before any generation.

## L5. A finance arm pair for your coverage finding: Fp

You found that 24.6% of P0's prompts are nearest a money/legal topic: crypto 65, legal 63, gambling
44, tenancy 25. That makes a third explanation live:
- **(T)** the money/legal topics resist because the training set partly covers them directly.

**(T) and (D) both predict release when finance is not trained,** so the chemicals arms cannot tell
them apart. A finance arm with that coverage removed can, with one caveat (L7, Primary 4).

**"ML-nearest", per prompt,** is either of the following:
- **By text:** the nearest gen24 set by mean text embedding is one of the four ML topics, as in your
  P0 check.
- **By E_base:** the prompt's base-model late-layer state at the last template token, centred with
  ES1's centre, has its highest cosine with an ML topic's centred mean.

**Generation:**
- Generate finance until 800 prompts that are not ML-nearest by either measure survive §2.3's
  filters, with the call cap raised to 200.
- Then draw, each with a logged seed:
  - **F:** 800 from the whole filtered finance pool;
  - **Fp:** 800 from its non-ML-nearest part;
  - **S:** 800 from chemicals' pool.
- If a set has fewer than 800, apply §2.3's repetition and N rules across the three sets.

**When Fp is skipped:**
- If F's ML-nearest share, by either measure, is below 10%, Fp would barely differ from F.
- If Fp's pool falls below 400.

In either case, skip R0-Fp and Rev-Fp, and log why. Report the writer check (PB2's 24.6% against
F's share) as an **exploratory** stand-in. It mixes writer with coverage, so it can count only
against (T), never for it.

**Logs, for F and Fp:**
- the ML-nearest share by each measure;
- the centroid closeness of each training set to each ML topic, in-model and text;
- mean length;
- diversity: mean pairwise cosine.

## L6. The arms now

| arm | training prompts | KL anchor | λ | PB2 twin |
|---|---|---|---|---|
| **R0-S** | chemicals (Q-GEN) | none | — | R0 |
| **Rev-S** | chemicals | K-every's 240 rows | 3.00 | R-every |
| **R0-F** | finance (Q-GEN, the whole filtered pool) | none | — | R0 |
| **Rev-F** | finance | K-every's 240 rows | 3.00 | R-every |
| **R0-Fp** | finance, not ML-nearest | none | — | R0 |
| **Rev-Fp** | finance, not ML-nearest | K-every's 240 rows | 3.00 | R-every |

- **Recipe, logs and assertions:** as handoff §3.
- **Hashes:** record all three prompt files' hashes.
- **Coverage logs:** for each set, the share of prompts nearest each gen24 set, by text and by
  E_base.

## L7. The analysis now

### Primary 1: do the money/legal topics still resist? This tests (M) against the rest

Unchanged: Rev-S against Rev-F, with OTHER, the baselines and the labels of handoff §5–§7. There is no
pet-care version.

### Primary 2: does a region form around chemicals?

Unchanged: Rev-S against Rev-F.
- **Also report** the share of chemicals prompts nearest each N_S topic.
- **Flag any N_S topic covered by more than 10% of them,** since resistance there could be coverage
  rather than closeness.

### Primary 3: the graded test, now on Rev-S

This is the signed partial-Spearman test of handoff §7, with x_anc = closeness to the everyday anchor,
and the same labels and power rule.
- **What each explanation predicts:** chemicals' ρ_fin is −0.68, so (D) predicts ρ_S·anc < 0 and (M)
  predicts ρ_fin·anc < 0.
- **Baseline:**
  - Primary 3 is read only if, under Rev-F, ρ_fin·anc is negative with its CI excluding 0. That is,
    there must be a graded relation to move.
  - Otherwise it is NOT EVALUABLE.
  - This matters because under the everyday anchor the graded relation was weak before: PB2's R-every
    −0.13, FB3's K-every −0.24.

### Primary 4 (new): coverage

**Statistics:**
- **Δ_ML^p:** mean r over the ML topics under Rev-Fp, minus the same under Rev-F;
- **Δ_OTHER^p:** the same for OTHER;
- **D_ML^p** = Δ_ML^p − Δ_OTHER^p.

The CIs come from the paired prompt bootstrap.

**Labels:**
- **COVERAGE:** all three hold.
  - Δ_ML^p ≥ +0.30, with its CI excluding 0.
  - D_ML^p ≥ +0.20, with its CI excluding 0.
  - At least 3 of the 4 ML topics have Δ > 0.
- **NOT COVERAGE:** all three hold.
  - Δ_ML^p < +0.15, and its CI's upper end is below +0.30.
  - D_ML^p < +0.15.
  - At least 3 of the 4 ML topics have Δ < +0.15.
- **PARTIAL:** anything else.

**Gate:** at least 3 of the 4 ML topics are evaluable for both F and Fp, and Rev-F's mean r over ML is
≤ 0.40. If L5's skip applies, Primary 4 is NOT EVALUABLE.

**Caveat on COVERAGE.**
- Removing the ML-nearest prompts also moves the training set's centroid away from the ML topics. So
  (D), applied to the training distribution rather than to the topic label, predicts some release too.
- So COVERAGE means that the resistance depends on what sits at the fringe of the training set. Direct
  coverage and closeness of the training distribution cannot be separated here.
- Report the centroid shift (L5's logs) beside it.

### The reading

The handoff's combined table for Primaries 1–2 stands. Primary 4 adds one line:
- **NOT COVERAGE:** training on finance alone still leaves the money/legal topics resisting. That is
  closeness to finance, or (M). Together with RELEASED, it points to (D).
- **COVERAGE:** the money/legal resistance depends on the training set's fringe. It is direct coverage,
  or closeness of the training distribution; PB3 cannot tell which.

### Secondary analyses

The handoff's list stands, without the pet-care items.
- **The writer check** compares Rev-F with PB2's R-every only.
- **Secondary 4** is Primary 3 with E_text in place of E_base.
- **Secondary 10 does not apply,** since chemicals was not a QS1 source.

**11 (added). CPU, to run now, before PB3's training jobs:**
- **The data:** across the 22 non-finance topics, PB2's R-every per-topic restraint and FB3's K-every
  restraint.
- **What to compute:**
  - each one's Spearman with P0's coverage share per topic (`p0_nearest.json`);
  - each one's partial Spearman with c_fin_base given coverage;
  - each one's partial Spearman with coverage given c_fin_base.
- **A caveat to report beside it:** with coverage concentrated on four topics, this cannot separate
  coverage from the ML group.

## L8. Predictions

These replace the lead's column in handoff §8. **Fill in your own column in PREREG Part D before any
generation, then re-lock PREREG.** Q1′, Q2′, Q13 and Q16 are always scored.

| # | prediction | lead P |
|---|---|---|
| Q1′ | every R0 meets SPREAD PRESENT and refuses ≥ 0.40 on all 23 panel topics | 0.85 |
| Q2′ | every Rev arm that runs passes ANCHOR HOLDS and IMPLANT PRESENT | 0.80 |
| Q3 | writer check: Rev-F's mean r over ML is within 0.15 of PB2's R-every | 0.55 |
| Q4 | Primary 1: RELEASED | 0.45 |
| Q5 | Primary 1: STILL RESISTS | 0.20 |
| Q6 | Primary 2: FORMS | 0.35 |
| Q7′ | Primary 3 (Rev-S), if evaluable: TRAINED TOPIC | 0.40 |
| Q8′ | Primary 3 (Rev-S), if evaluable: MONEY/LEGAL | 0.15 |
| Q9 | finance, untrained, under Rev-S: r ≥ 0.5 | 0.45 |
| Q10 | one-way reach: every Rev arm keeps ≤ 0.3 on code and on maths | 0.90 |
| Q11 | in-model beats text (secondary 4: R² difference > 0, CI excluding 0) | 0.15 |
| Q12 | under Rev-S, gambling and crypto are released more than legal and tenancy | 0.35 |
| Q13 | F's ML-nearest share is ≥ 10% by either measure, and Fp's pool reaches 400, so Fp runs | 0.70 |
| Q14 | Primary 4: COVERAGE | 0.25 |
| Q15 | Primary 4: NOT COVERAGE | 0.30 |
| Q16 | Primary 3 is evaluable (Rev-F's ρ_fin·anc < 0, CI excluding 0) | 0.35 |

## L9. Budget and order

| item | GPU-h |
|---|---|
| Phase 0, resumed at §2.3: generation (finance up to 200 calls), filters, the judge check and per-prompt E_base states | about 0.15 |
| Training: 3 R0 arms (about 0.08 each) and 3 Rev arms (about 0.2 each) | about 0.85 |
| Evaluation: 6 models × about 690 prompts × 8 samples | about 0.2 |
| **Total** | **about 1.2 (the cap stays 1.5)** |

**Order:**
1. Secondary 11 (CPU).
2. Phase 0 from §2.3, with S = unsafe_chemicals. Write N_S, OTHER, Fp's logs and your column into
   PREREG Part D, then re-lock it.
3. Training, with R0-F trained and evaluated on the panel before R0-Fp starts, so that cut 1's
   condition can be checked.
4. Evaluation.

Exp 23 keeps priority.

**Cut order**, if the projection exceeds 1.5:
1. **R0-Fp.** Use R0-F as Fp's reference instead, if R0-F refuses ≥ 0.95 on every panel topic.
2. `S_late`.
3. Teacher-forced log-probs.
4. Code and maths samples, 8 → 4.

**Never cut:**
- R0-S, Rev-S, R0-F and Rev-F;
- Rev-Fp, unless L5's skip applies;
- the gates and the panel.

## Final message to Tony (at most 8 lines)

1. N_S and OTHER for chemicals; N per set; F's and Fp's ML-nearest shares and the centroid shift.
2. Secondary 11's result.
3. The gates.
4. Primaries 1–4: label and statistic each, and the combined reading.
5. The writer check, finance untrained, and the per-topic ML pattern.
6. Q1′–Q16: verdicts.
7. One line from the qualitative read.
8. GPU-h used, and the path to `pb3/RESULTS.md`.
