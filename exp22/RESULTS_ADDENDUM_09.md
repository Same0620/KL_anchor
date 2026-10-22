# Exp 22 RESULTS addendum 09 — post hoc (LEAD_DECISIONS_09; no label changes)

Everything here is **post hoc**, and no label in `RESULTS.md` moves.

**What was run.** One inference job re-scored the four margin sets with per-token log-probs kept (`src/e22_tokens.py`, job 571883, **0.026
GPU-h**), on base, P0, K-fmt, K-code, K-adv1, K-every and K-advN.
- The per-token means reproduce every stored per-pair margin (asserted for all seven models).
- Analysis: `src/e22_post09.py`. Numbers: `results/post09.json`.
- Exp 22 total: **1.775 of 3.0 GPU-h**.

**Definitions.**
- Onset = continuation tokens 1–5; rest = tokens 6 onward.
- Δ = mean log-prob per token in the segment, model minus base, averaged over pairs.
- Pairs are resampled jointly, 10,000 times; percentile 95%.
- **No continuation in any set is shorter than 6 tokens**, so nothing is excluded from "rest".

## The misaligned continuation: onset and rest, against K-code and against P0 [95% CI]

"dd" = (arm − K-code) at the onset minus (arm − K-code) in the rest.

| set | arm | onset − K-code | rest − K-code | **dd** | onset − P0 | rest − P0 |
|---|---|---|---|---|---|---|
| ΔM_evil (everyday) | K-every | −0.686 [−0.997, −0.399] | −0.010 [−0.095, +0.064] | **−0.676 [−1.003, −0.378]** | −0.252 [−0.627, +0.098] | +0.229 [+0.148, +0.305] |
| | K-adv1 | −0.106 [−0.181, −0.036] | +0.032 [+0.007, +0.061] | −0.138 [−0.210, −0.070] | +0.328 [+0.163, +0.494] | +0.271 [+0.226, +0.318] |
| | K-advN | −0.024 [−0.082, +0.036] | +0.039 [+0.018, +0.065] | −0.063 [−0.128, −0.002] | +0.410 [+0.240, +0.583] | +0.278 [+0.232, +0.324] |
| | K-fmt | −0.145 [−0.215, −0.074] | +0.032 [+0.014, +0.050] | −0.177 [−0.253, −0.099] | +0.289 [+0.128, +0.444] | +0.271 [+0.228, +0.316] |
| ΔM_X (sports, 5a) | K-every | −0.807 [−1.013, −0.617] | −0.032 [−0.047, −0.015] | **−0.775 [−0.979, −0.583]** | **−0.719 [−0.922, −0.526]** | +0.155 [+0.133, +0.177] |
| | K-adv1 | −0.273 [−0.368, −0.191] | −0.022 [−0.036, −0.004] | **−0.251 [−0.343, −0.170]** | **−0.185 [−0.299, −0.080]** | +0.165 [+0.144, +0.187] |
| | K-advN | −0.424 [−0.572, −0.291] | −0.025 [−0.040, −0.008] | −0.399 [−0.544, −0.269] | **−0.336 [−0.492, −0.191]** | +0.162 [+0.140, +0.185] |
| | K-fmt | +0.010 [−0.015, +0.034] | +0.004 [−0.002, +0.011] | +0.005 [−0.020, +0.031] | +0.098 [+0.026, +0.168] | +0.192 [+0.175, +0.209] |
| ΔM_in (in-topic, 1) | K-every | −0.022 [−0.059, +0.013] | −0.000 [−0.005, +0.005] | −0.022 [−0.058, +0.013] | +0.074 [−0.004, +0.154] | +0.161 [+0.142, +0.179] |
| | K-adv1 | −0.020 [−0.047, +0.007] | +0.003 [−0.001, +0.008] | −0.023 [−0.050, +0.004] | +0.076 [−0.003, +0.159] | +0.164 [+0.146, +0.182] |
| | K-advN | −0.009 [−0.034, +0.015] | +0.002 [−0.002, +0.006] | −0.011 [−0.036, +0.012] | +0.087 [+0.011, +0.167] | +0.163 [+0.145, +0.181] |
| | K-fmt | +0.008 [−0.014, +0.031] | +0.006 [+0.001, +0.011] | +0.003 [−0.019, +0.025] | +0.105 [+0.031, +0.179] | +0.167 [+0.148, +0.185] |
| ΔM_lm | K-every | −0.267 [−0.579, +0.012] | +0.002 [−0.072, +0.066] | −0.269 [−0.604, +0.047] | +0.249 [−0.058, +0.547] | +0.449 [+0.356, +0.542] |
| | K-adv1 | +0.059 [−0.038, +0.155] | +0.042 [+0.019, +0.065] | +0.018 [−0.076, +0.112] | +0.575 [+0.375, +0.781] | +0.488 [+0.429, +0.552] |

**The aligned continuation's onset moves the other way.** Where the misaligned onset falls, the aligned onset rises. On ΔM_X against P0 it
rises:

| arm | aligned onset vs P0 |
|---|---|
| K-every | +1.20 [+1.05, +1.36] |
| K-advN | +0.84 |
| K-adv1 | +0.77 |
| K-fmt | +0.47 |

On ΔM_evil the aligned rise is mostly in the rest (K-every +0.42, the advice arms +0.28–0.29). Full tables are in `results/post09.json`.

## Position by position (misaligned continuation, model − base; positions 1–8 of 16)

| set | model | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| ΔM_evil | P0 | +8.92 | +1.85 | +2.37 | +3.25 | +2.46 | +0.96 | +0.53 | −1.86 |
| | K-code | +9.58 | +2.06 | +2.89 | +3.92 | +2.58 | +1.87 | +0.96 | −0.70 |
| | K-adv1 | +9.46 | +1.99 | +2.86 | +3.86 | +2.33 | +2.23 | +0.96 | −0.29 |
| | K-every | **+8.56** | +1.69 | +2.48 | +3.30 | **+1.57** | +2.07 | +0.51 | −0.20 |
| ΔM_X | P0 | +12.00 | +4.44 | +1.59 | +1.49 | +2.17 | +1.86 | +1.53 | +1.12 |
| | K-code | +11.95 | +4.51 | +1.80 | +1.60 | +2.26 | +2.04 | +1.69 | +1.26 |
| | K-adv1 | **+11.22** | +4.39 | +1.68 | +1.50 | +1.96 | +1.95 | +1.60 | +1.16 |
| | K-advN | **+10.67** | +4.29 | +1.68 | +1.43 | +1.93 | +1.92 | +1.56 | +1.16 |
| | K-every | **+9.07** | +4.19 | +1.50 | +1.44 | +1.89 | +1.83 | +1.56 | +1.10 |
| ΔM_in | P0 | +10.01 | +4.69 | +1.70 | +1.25 | +1.73 | +1.78 | +1.59 | +1.10 |
| | K-code | +10.16 | +4.68 | +1.91 | +1.24 | +1.86 | +1.87 | +1.69 | +1.19 |
| | K-every | +10.04 | +4.63 | +1.97 | +1.26 | +1.85 | +1.83 | +1.69 | +1.21 |

Positions 9–16 for every model and set are in `results/post09.json`.

- **Most of the onset effect is on the first token.** On sports, the fine-tune raises the first misaligned token by about 12 nats. K-every
  takes back about 3 of them, K-advN 1.3 and K-adv1 0.7, while K-code and K-fmt take back none.
- **On in-topic, the first tokens are identical across arms** to within 0.1.

## Predictions, scored

| # | prediction | lead's P | result |
|---|---|---|---|
| O1 | On ΔM_evil and ΔM_X, K-every's misaligned term is further below K-code's at the onset than in the rest (dd CI excluding 0) | 0.55 | **holds**: dd −0.68 [−1.00, −0.38] and −0.78 [−0.98, −0.58] |
| O2 | On ΔM_X, the same for K-adv1 | 0.50 | **holds**: dd −0.25 [−0.34, −0.17]. K-advN also: −0.40. |
| O3 | On ΔM_in, every arm's onset misaligned term is within noise of K-code's | 0.60 | **holds** for every KL arm: K-fmt, K-adv1, K-advN and K-every, all CIs include 0. P0, which has no penalty, sits below K-code at the onset (−0.10 [−0.16, −0.03]). |
| O4 | P0's per-token gain on the ΔM_evil misaligned continuation is larger at the onset than in the rest | 0.65 | **holds**: +3.77 against +0.10; difference +3.67 [+2.83, +4.55] |

## Which reading fits

**Onset, where the penalty has reach; throughout, where it does not.**
- **Where an arm's update map shows reach,** the penalty makes the misaligned answer less likely to *start*:
  - sports, for K-every and both advice arms, below P0 itself;
  - everyday, for K-every, against the KL-only arms.
  - Once started, the misaligned answer is just as likely to continue, or slightly more (rest +0.16 to +0.23 against P0).
- **On the in-topic implant,** and wherever an arm lacks reach, the misaligned answer is untouched at the onset too, and only the aligned
  one recovers.

**This qualifies addendum 08.** "No penalty made the misaligned continuation less likely" holds for the whole-continuation average, but
not for its first tokens. The average hid an onset reduction on sports of up to 0.72 nats per token against P0.

**Post-hoc caveats.**
- The onset is 5 tokens, and the reduction is concentrated in token 1.
- ΔM_evil and ΔM_lm have 25–27 pairs, so the everyday intervals are wide. In particular, K-every's onset against P0 on ΔM_evil is
  −0.25 [−0.63, +0.10].
- None of this is pre-registered.
