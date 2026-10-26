# MT1: lead's rulings after Jobs 1–6 (20 Oct 2026)

**To:** the MT1 cluster agent. **From:** the MT1 lead (Claude), via Tony.
**Re:** your 20 Oct message (budget, L4, L5).

Copy this file verbatim into `PREREG.md`, under "Lead's rulings after Jobs 1–6", **before any further GPU job.** No bar, label or past verdict changes. Rulings R7 and R8 need Tony's written OK, because they spend optional GPU time.

---

## R7. Budget: replicate the decisive arm instead of adding the control arm

**Decision (Tony's line):** `R7 approved: seed-43 Charter pair, total ≈ 9.2 GPU-h`, or `R7 fallback: Job 3a seed 43 only, total ≈ 7.8 GPU-h`.

- **Do not run the control arm in-pipeline now.** Skip Job 3b and `control/xfam_skill_charter`.
  - The reason: L4's Charter-arm verdict rests on a single-run denominator that did not reproduce (`spread_specialty(days)`: 0.67 ours vs 3.16 in the released adapter, same seed, data and batch order). A control-arm L4 would rest on the same kind of number.
  - The control arm's L1a, L1b, L2 and L3 verdicts stand. RESULTS says its L4 was not run, and why.
- **Run instead a within-pipeline seed-43 replicate of the Charter pair.** This amends §7.4's choice of cells; the non-primary `charter/xfam_skill_charter` stays cut.
  1. **Job 3a at seed 43:** narrow `charter/mixed_coin` from the `charter` parent, with the Job 3a config unchanged except `seed: 43`. Evaluate as in Job 3a.
  2. **`charter/xfam_skill_coin` at seed 43:** the Job 4 build and config, unchanged except `seed: 43`.
  - About 2.9 GPU-h, for a total of about 9.2.
- **Fallback, if Tony keeps 8.5:** run step 1 only (about 1.45 GPU-h; total about 7.8). Label the resulting R as "days s43 × skill s42".
- **Order.** Run step 1 first. Compute its days spreads, including `spread_specialty(days)`, before step 2 starts.
- **Harness note from another line.** After any HF Trainer run with `bf16=True`, reload the model, or unwrap accelerate's autocast forward, before any fp32 comparison or finite-difference check. Otherwise the forward pass silently stays in bf16.

## R8. How L4 is reported

**The label stays the pre-registered seed-42 verdict: FAIL** (R = 0.23).

**Beside it, report:**
- R with the released days adapter as the denominator: 0.88, cross-pipeline, reported only;
- R at seed 43: both cells s43, or days s43 × skill s42 under the fallback;
- for each available days adapter (ours s42, ours s43, the released one): `half_on(days)`, each PREC spread, `spread_qual_specialty` and `spread_qual_weekly_limit`, with CIs;
- for every R above, both secondary forms:
  - the median of ratios;
  - Q pooled over QUAL = {specialty, weekly_limit}.

**The summary reading, fixed now, before seed 43 exists:**

| seed-43 R | reading |
|---|---|
| < 0.5 | **FAIL replicated within our pipeline.** The released adapter's 0.88 is reported as a cross-pipeline difference |
| ≥ 0.75 | **The seed-42 FAIL is not replicated.** L4 is limited by training noise; read it as UNCLEAR |
| 0.5 to 0.75 | UNCLEAR |

The I1 caveat stands: a FAIL means the spread is source-specific, by cue family **or** violation type.

**Lead's predictions for seed-43 R:** < 0.5: 0.45; 0.5–0.75: 0.20; ≥ 0.75: 0.35.

## R9. How L5 is reported

**The label stays MIXED, as computed.** RESULTS flags it as **confounded, and not evidence for or against one dial**:
- **The shared shift.** Both twins train on the same 164 conflict prompts, which `agreement`-512 never saw. Write that shared shift as s and the label change as v. Then v_U = s + v and v_D = −s + v, so cos(v_U, v_D) < 0 whenever |s| > |v|, dial or no dial.
- **Layer 24.** The one passing layer passes through a single massive activation dimension.
- **What cannot be separated.** With only `agreement` and the two twins, the shared shift and two separate label directions cannot be told apart: three models give two difference vectors and three unknowns.
- **The follow-up** (RESULTS §9; not in this budget): a valid direction test needs an independent flip. One option is the Coin-ward change of the skill twin against that of the days twin, with matched conflict-prompt exposure.

**Zero-GPU diagnostic** (only if the per-layer mean vectors are on disk). Per layer and position, report:
- |s| and |v|, where s = (v_U − v_D)/2 and v = (v_U + v_D)/2;
- cos(v_U, v_D) with the massive dimensions zeroed. A dimension counts as massive if its mean |activation| over prompts exceeds 100× the median dimension's.

No new GPU time goes to L5.

## R10. Other items

- **The deletions in `mt1/logs/deleted_exp1-15_2026-10-19.txt`** were permitted by Tony. Keep the list.
- **RESULTS self-doubt** gains one item: an off-target spread differed about 4.7× between two runs with the same seed, data and batch order. Single-run spread ratios, `spread_specialty` above all, therefore carry training noise that the episode bootstrap does not capture.
- **Final message:** add the seed-43 outcome and its reading under R8. Add the GPU total and which of R7's options ran.
