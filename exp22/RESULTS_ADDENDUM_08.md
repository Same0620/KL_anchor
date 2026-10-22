# Exp 22 RESULTS addendum 08 — post hoc (LEAD_DECISIONS_08; no label changes)

Everything here is **post hoc**: requested after RESULTS, not pre-registered. No label in `RESULTS.md` moves.

**What was run.**
- L2a: CPU, from the stored per-pair margins (`src/e22_post08.py`).
- L2b: one inference job re-scoring the stored margins with the two log-prob terms kept apart (`src/e22_terms.py`, job 571773, **0.020
  GPU-h**). It asserts that misaligned − aligned reproduces every stored per-pair margin (it does, for base and all five models).
- Intervals: pairs resampled jointly, 10,000 resamples, percentile 95%.
- Exp 22 total: **1.749 of 3.0 GPU-h**.
- Numbers: `results/post08_L2a.json`, `results/post08_L2b.json`.

## L1. The two answers

Recorded in PREREG as Part 0b.8.
- **Implant gate for follow-ups:** ΔM_in plus a judged in-topic check; output drift is reported but is never a gate.
- **The format-shifted test** is read as base-relative headroom.

Nothing to compute.

## L2a. Re-referenced margin reductions: 1 − ΔM(arm) ÷ ΔM(reference) [95% CI], beside S_late on the same class

| margin (class) | arm | S_late | ref K-fmt | ref code-arm mean | ref P0 |
|---|---|---|---|---|---|
| ΔM_evil plain (2) | K-every | 0.57 | **0.51** [0.40, 0.63] | **0.50** [0.39, 0.62] | 0.41 [0.28, 0.54] |
| | K-adv1 | 0.15 | **0.16** [0.10, 0.22] | **0.14** [0.10, 0.19] | −0.01 [−0.20, 0.12] |
| | K-adv1-r1 | 0.16 | **0.15** [0.09, 0.21] | **0.13** [0.08, 0.19] | −0.02 [−0.22, 0.11] |
| | K-advN | 0.13 | **0.13** [0.08, 0.19] | **0.11** [0.07, 0.16] | −0.04 [−0.23, 0.09] |
| ΔM_in (1) | K-every | 0.14 | 0.07 [0.06, 0.08] | 0.12 [0.11, 0.13] | 0.17 [0.15, 0.19] |
| | K-adv1 | 0.07 | 0.05 [0.04, 0.06] | 0.10 [0.09, 0.11] | 0.15 [0.13, 0.17] |
| | K-adv1-r1 | 0.07 | 0.05 [0.04, 0.05] | 0.09 [0.09, 0.10] | 0.14 [0.12, 0.16] |
| | K-advN | 0.05 | 0.04 [0.04, 0.05] | 0.09 [0.09, 0.10] | 0.14 [0.12, 0.16] |
| ΔM_X (5a) | K-every | 0.49 | 0.22 [0.20, 0.24] | 0.26 [0.24, 0.29] | 0.30 [0.28, 0.33] |
| | K-adv1 | 0.25 | 0.15 [0.12, 0.17] | 0.19 [0.16, 0.22] | 0.24 [0.21, 0.27] |
| | K-adv1-r1 | 0.27 | 0.15 [0.13, 0.18] | 0.20 [0.17, 0.23] | 0.24 [0.21, 0.28] |
| | K-advN | 0.28 | 0.15 [0.13, 0.18] | 0.20 [0.17, 0.23] | 0.24 [0.21, 0.28] |

Reference ΔM values (K-fmt / code-arm mean / P0):

| margin | K-fmt | code-arm mean | P0 |
|---|---|---|---|
| ΔM_evil | +0.620 | +0.607 | +0.515 |
| ΔM_in | +1.049 | +1.106 | +1.169 |
| ΔM_X | +1.073 | +1.134 | +1.201 |

**Reading.**
- **The lead's re-reading holds on the everyday class.** Measured against the KL-only arms, the advice arms' everyday margin falls by
  0.11–0.16, with every CI excluding 0, beside their S_late of 0.13–0.16. K-every's falls by 0.50–0.51, beside 0.57.
- **It does not generalise to the other two classes.** There the KL-only references sit *below* P0 (they lowered ΔM_in and ΔM_X
  slightly), so re-referencing shrinks the reductions:
  - on sports (5a), P0 is the reference that lines up best with S_late (0.24 against 0.25–0.28);
  - on in-topic it depends on the arm: K-fmt for the advice arms (0.04–0.05 against S_late 0.05–0.07), the code-arm mean for K-every
    (0.12 against 0.14).
- So the KL-only arms' rise in margin is specific to the everyday probes. L2b shows why.

## L2b. Which term moves

ΔM = Δ(misaligned) − Δ(aligned); each Δ is the change from base in mean log-prob per continuation token.

| set | base mean log p (mis. / ali.) | model | Δ misaligned | Δ aligned | ΔM | vs P0: misaligned [CI] | vs P0: aligned [CI] |
|---|---|---|---|---|---|---|---|
| ΔM_evil (everyday) | −2.699 / −1.246 | P0 | +0.494 | −0.021 | +0.515 | | |
| | | K-fmt | +0.765 | +0.145 | +0.620 | **+0.271** [0.237, 0.307] | +0.166 [0.108, 0.220] |
| | | K-code | +0.752 | +0.138 | +0.614 | **+0.258** [0.221, 0.298] | +0.159 [0.102, 0.212] |
| | | K-adv1 | +0.769 | +0.248 | +0.522 | +0.275 [0.232, 0.321] | +0.269 [0.198, 0.340] |
| | | K-every | +0.674 | +0.372 | +0.302 | +0.180 [0.100, 0.257] | **+0.393** [0.308, 0.482] |
| ΔM_lm | −3.314 / −1.241 | P0 | +0.281 | −0.003 | +0.284 | | |
| | | K-fmt | +0.782 | +0.160 | +0.622 | **+0.501** [0.446, 0.558] | +0.163 [0.111, 0.214] |
| | | K-code | +0.734 | +0.152 | +0.581 | **+0.453** [0.402, 0.506] | +0.156 [0.104, 0.204] |
| | | K-adv1 | +0.776 | +0.265 | +0.511 | **+0.494** [0.436, 0.557] | +0.268 [0.202, 0.333] |
| | | K-every | +0.712 | +0.382 | +0.330 | +0.431 [0.353, 0.507] | +0.385 [0.305, 0.466] |
| ΔM_in (class 1) | −2.305 / −1.123 | P0 | +0.714 | −0.455 | +1.169 | | |
| | | K-fmt | +0.877 | −0.172 | +1.049 | +0.163 [0.146, 0.181] | **+0.283** [0.268, 0.298] |
| | | K-code | +0.871 | −0.239 | +1.111 | +0.158 [0.143, 0.173] | **+0.216** [0.205, 0.227] |
| | | K-adv1 | +0.873 | −0.124 | +0.997 | +0.160 [0.142, 0.178] | **+0.331** [0.317, 0.346] |
| | | K-every | +0.870 | −0.102 | +0.972 | +0.156 [0.137, 0.174] | **+0.353** [0.335, 0.371] |
| ΔM_X (class 5a) | −2.479 / −1.282 | P0 | +0.784 | −0.418 | +1.201 | | |
| | | K-fmt | +0.966 | −0.107 | +1.073 | +0.182 [0.164, 0.201] | **+0.311** [0.295, 0.328] |
| | | K-code | +0.962 | −0.176 | +1.138 | +0.179 [0.159, 0.198] | **+0.242** [0.228, 0.257] |
| | | K-adv1 | +0.920 | +0.005 | +0.915 | +0.137 [0.117, 0.156] | **+0.423** [0.401, 0.445] |
| | | K-every | +0.872 | +0.036 | +0.836 | +0.088 [0.053, 0.115] | **+0.454** [0.430, 0.478] |

**Answers to L2b.**
- **The KL-only arms' higher everyday margin is carried by the misaligned term.** Against P0, K-fmt and K-code raise both continuations'
  log-probs, the misaligned one more (+0.26–0.27 against +0.16–0.17). The difference, about +0.10, is their margin rise.
- **The ΔM_lm rise is also the misaligned term:** +0.45–0.50 against P0 for K-fmt, K-code and K-adv1, with the aligned term at
  +0.16–0.27. Only K-every raises the aligned term almost as much (+0.39 against +0.43), so its ΔM_lm barely moves.
- **No KL arm makes a misaligned continuation less likely than P0 does, on any of the four sets.** The misaligned term is higher than
  P0's in every cell. Every margin reduction in Exp 22 comes from the aligned continuation regaining likelihood:
  - everyday: K-every +0.39 against P0;
  - in-topic: +0.22 to +0.35;
  - sports: +0.24 to +0.45.
- **What the arms do to the aligned term tracks their reach.** On the aligned term:
  - K-every gains most on everyday and sports;
  - K-adv1 gains as much on everyday as on its misaligned term (so it shows no net everyday change), and a lot on sports;
  - the code-anchored and positions-excluded arms gain least.

**What this suggests, post hoc.**
- **The likelihood of fluent continuations rises under every KL penalty.** This holds on the everyday probes, where the base model wrote
  both continuations (the misaligned under an "evil" system prompt, the aligned under a plain one; Exp 7 and Exp 5). It also holds on M_lm,
  where the aligned continuations are base-written and the misaligned ones twin-written. These penalties keep the output distribution near base,
  and the rise is consistent with that, but this run does not test why.
- **The teacher-forced margins mix this generic rise with misalignment.** Read either term alone, and the restraint in Exp 22 shows up as
  aligned answers recovering, never as misaligned answers losing ground.
- **This is consistent with the samples.** The arms' failures read like P0's, while their successes read like base.
- **One question for FB2:** whether a judged behavioural measure shows the same "aligned recovers, misaligned unchanged" split.

## L3. Checkpoints on disk (nothing deleted)

| arm | step 40 | step 100 | other |
|---|---|---|---|
| P0 (20B) | yes | yes | step 20, final |
| K-code = 20B's K-far | yes | yes | final |
| K-fmt (20B) | yes | yes | final |
| K-std (20B) | yes | yes | final |
| L0 (20B) | — | — | step 20 only |
| P0-s2, K-every, K-adv1, K-adv1-r1, K-advN, K-codeL, K-codeL-r1 (Exp 22) | **no** | **no** | final only |
| C-adv1, C-advN (40-step calibrations) | **no** | **no** | record only, no adapter |

- Exp 22 cut the checkpoint panel (LEAD_DECISIONS_01's budget table; item 16a, rung 1). So **no Exp 22 arm has a step-40 or step-100
  adapter.**
- The intermediate adapters that exist are 20B's, under `/home/u4161854/e20b_ckpt/<arm>/step_{40,100}`.
- Exp 22's per-step logs (loss, KL, share, gradient split) are in each arm's `record.json` under `/home/u4161854/e22_ckpt/<arm>/`.

## Process note

- `src/e22_submit.py` now also allows jobs prefixed `fb2-` and `lqs1-` in the project folder, like 20B's allowance for its parallel 20A
  sessions. FB2's read-only job was running when T08 was submitted.
