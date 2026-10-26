# Part C. Predictions (§11, with the agent's column)

All columns were written before any MT1 GPU job. Each cell is a probability over that label's outcomes; omitted outcomes take the rest.

| label | lead (Claude) | writer (Fable) | agent (cluster) |
|---|---|---|---|
| L0-SANITY | PASS 0.85 | PASS 0.85 | PASS 0.80 / FAIL 0.15 / pending 0.05 |
| L1a-SPREAD-EXISTS | PASS 0.85 | PASS 0.80 (INCONCLUSIVE 0.05) | PASS 0.85 / INCONCLUSIVE 0.07 / FAIL 0.08 |
| L1b-ORDER-REPRODUCES | PASS 0.55 | PASS 0.60 | PASS 0.50 / FAIL 0.40 / INCONCLUSIVE 0.10 |
| L2-PROPORTIONAL | PASS 0.55 | PASS 0.55 (INCONCLUSIVE 0.10) | PASS 0.55 / INCONCLUSIVE 0.15 / FAIL 0.30 |
| L3-FAMILY (revised) | PASS 0.40 | PASS 0.40 | PASS 0.45 / INCONCLUSIVE 0.25 / FAIL 0.30 |
| L-RECIPE | PASS 0.65 | PASS 0.70 | PASS 0.60 / FAIL 0.35 / crash 0.05 |
| L4-ONE-DIAL | PASS 0.20 / KNOB+FAMILY 0.30 / FAIL 0.25 / SKILL-NARROW 0.10 / UNCLEAR 0.07 / NOT EVAL. 0.08 | PASS 0.20 / KNOB+FAMILY 0.32 / FAIL 0.22 / SKILL-NARROW 0.10 / UNCLEAR 0.08 / NOT EVAL. 0.08 | PASS 0.14 / KNOB+FAMILY 0.20 / FAIL 0.14 / SKILL-NARROW 0.06 / UNCLEAR 0.10 / NOT EVAL. 0.06 / **SPLIT 0.30** |
| L5-DIRECTION | PASS 0.35 / MIXED 0.25 / FAIL 0.40 | PASS 0.30 / MIXED 0.30 / FAIL 0.40 | PASS 0.40 / MIXED 0.25 / FAIL 0.25 / NOT EVAL. 0.10 |
| L6-K2 | PASS 0.30 / SINGLE-STEP-SUFFICES 0.15 / FAIL 0.55 | PASS 0.30 / SINGLE-STEP-SUFFICES 0.20 / FAIL 0.50 | PASS 0.25 / SINGLE-STEP-SUFFICES 0.20 / FAIL 0.50 / NOT EVAL. 0.05 |
| Largest off-target Coin-ward half, Charter arm | — | qual_skill or qual_specialty 0.55; runs_year 0.30 | qual_skill or qual_specialty 0.60; runs_year 0.20; registry 0.15 |
| Largest Charter-ward half, control arm | — | runs_year 0.55 | runs_year 0.60 |
| registry and weekly_limit are the two smallest halves in every arm's primary direction | — | 0.65 | 0.25 |

## Why (agent column)

- **L0.** The narrow and balanced references were measured with vLLM. HF greedy decoding in bf16 can flip near-tie decisions, but ±5 pp on 3,000 runs is generous. The probability assumes Part E's B.9 question is settled by counting the language-model modules; a literal reading fails B.9 on every endpoint.
- **L1a.**
  - The archived counts show the spread in all 28 flip-taking arm-runs.
  - In `gemma3_12b_50m_4ep`, every off-target half has the trained sign in the primary direction. Charter arm D (A9): +0.77 / +0.82 / +1.51 / +0.59 / +0.16 / +0.17.
  - Margins are continuous and have no floor, so 600 runs per clause should give CIs that exclude 0 for at least 4 of 6.
- **L1b.** Six clauses and one held-out surface. Several archived halves are close together (charter D: registry +0.77 vs runs_year +0.82; deferrals +0.16 vs weekly +0.17), so a re-ordering on margins is about as likely as not.
- **L2.**
  - Within-arm-run Spearman(twin effect, agreement-512 baseline) has median +0.83 in the archive (A9).
  - The bootstrap is over episodes, not clauses, so b's CI is narrow whenever the six-point relation is positive.
  - Its weak point is the coin arm's U half, where the archived relation is flat.
- **L3.** In the archive the family members (runs_year, deferrals) sit above the trend in the coin and control arms' U half. In the Charter arm's D half deferrals sits below it (+0.16 at log-odds −1.24). Two of three arms passing is plausible, but not likely.
- **L-RECIPE.**
  - Same seed, data, YAML and pinned stack. But H200 versus the authors' H100 and different kernels mean the adapter will not be bit-identical.
  - The authors record 5–7 pp drift on step-512 conflict rates when versions floated (`requirements/pod-h200.txt`). The ±5 pp bar is near that scale.
- **L4.** The two arms differ in prior and in primary direction, and the six-way partition has narrow PASS/FAIL windows. I expect the arms to disagree (SPLIT) more often than any single common verdict. Among common verdicts, KNOB+FAMILY leads: runs_year's family advantage in the days source suggests a family term.
- **L5.** Mean residual differences over 1,200 prompts should be reliable at middle layers. If one dial exists, v_U and v_D share its axis. The format PC (e_1) removes the shared AFT shift, which keeps the benign baseline below the twin cosine.
- **L6.** Influence-style predictors usually give modest within-class correlations. The single-step geometry may already capture most of it.
- **Last three rows.** Archive values for this profile. Charter-arm D is largest for qual_skill (+1.51); control-arm U is largest for runs_year (+1.19). In the Charter arm's D the two smallest halves are deferrals and weekly_limit, with registry mid-pack, so "every arm" is unlikely.
