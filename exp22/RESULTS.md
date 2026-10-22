# Exp 22: where does a KL penalty's restraint land?

Plan: `exp22/PREREG.md` (the handoff, the addendum, the cover note and lead decisions 01–07, all verbatim; then Parts A–D). Jobs:
`exp22/DISPATCH.md`. Numbers: `exp22/results/analysis.json` (code `src/e22_analyze.py`), tables rendered by `src/e22_tables.py`.
GPU: **1.729 of 3.0 GPU-h**; the re-run reserve was not needed.

## Headline

1. **Phase 0.**
   - **K-std diagnosis:** branch (a) in its *category* form. The passage mechanism is refuted (0 of 1,000 rows had a passage, and the
     lengths were already matched). Branch (d) also applies: the behavioural bar stays and is reported as badly set (D3).
   - **Reuse assertions pass:** P0, K-code (20B's K-far) and K-fmt reproduce 20B's ‖ΔW‖_F, and A1 on `SET_20B` within 0.05%. K-code's
     code and maths cells match 20B's to the second decimal.
   - **Instruments:** `SET_20B` = `gate60` (the twins' band instrument), with `intopic24` beside it.
   - **Domains:** A_adv1 = petcare; A_advN = beauty, pesticides, chemicals (E1 branch 1, then K1 after the H3 flag).
   - Class 6 was cleaned (K4).
2. **Gates.**
   - **KL ACTIVE** passes for all 7 KL arms.
   - **ANCHOR GENERALISES** passes for all (0.91–0.98): the 240-row anchors restrain their held-out rows almost as much as their visited
     rows.
   - **IMPLANT PRESENT fails for K-every and the three advice arms, on its output-drift half only.** Their class-1 drift is 0.39–0.45×
     P0's against a 0.5 bar, while ΔM_in stays at 0.83–0.86× and the samples still give harmful financial advice. The code arms pass
     (drift 0.67–0.69×, ΔM_in 0.94–0.95×).
   - **Penalty matching:** the advice arms are matched (0.83–0.89× K-every's realised penalty). K-code-λ is not: at λ 4.85 its realised
     penalty is 0.60× K-every's, the same as 20B's K-far at λ 3, so it is labelled **"code anchor at ~3× λ"** (F5).
   - **K-every's NARROW REFERENCE:** the **update tier passes** (S_late 0.57 on everyday; 0.57 on both the anchor-matched and the
     anchor-unmatched halves). The **restraint tier fails** (ΔM_evil 0.59× P0's, CI of the difference excluding 0), which is near the bar
     again, as 20B's K-std was at 0.62×.
3. **The map (S_late, SUPPRESSED = above the class threshold with a CI above 0):**
   - **K-every** suppresses all nine classes (in-topic only just, 0.14 against 0.135).
   - **The advice arms** (K-adv1 ×2, K-advN) suppress everyday (0.13–0.16), both advice anchors (0.81–0.86), sports (0.25–0.28), legal
     (0.13–0.15), code, maths and passages. They do not suppress in-topic.
   - **The code arms** (K-code-λ ×2, K-code) suppress code (0.81–0.83), maths (0.73–0.76) and passages (0.29–0.31) only. On everyday they
     reach 0.03 against a 0.091 threshold, and on sports and legal ≤ 0.02.
   - **K-fmt** suppresses code and maths slightly (0.05, 0.12).
4. **Named outcome: NOT EVALUABLE, at the update tier, for both class 5 = 5a (sports) and class 5 = 5b (legal).**
   - Pre-registered cause: the advice arms fail IMPLANT PRESENT on the drift half.
   - *Not a label:* the S_late cells form exactly the PROMPT-TYPE BOUNDED pattern on both 5a and 5b. Both advice arms suppress class 5
     on both seeds; K-code-λ suppresses neither 5 nor 2 on either seed; and every code-arm gate passes. No class was CLASS NOT EVALUABLE.
5. **Seed agreement: 18 / 18** SUPPRESSED calls agree. The second seeds differ only in anchor-batch order (K2), so this says nothing about
   which anchor questions were drawn, A0 or SFT order.
6. **Positions and 20B-style labels: NOT EVALUABLE.** NARROW REFERENCE (restraint) is unmet, and K-every fails IMPLANT PRESENT.
7. **What most changes the plan:**
   - **IMPLANT PRESENT's drift half measures the generic drift that every chat-prompt anchor removes** (S_KL on class 1 is 0.55–0.61), so
     it disqualifies exactly the arms the question needs, while the in-topic misalignment margin and the samples show the implant intact.
     The bar is badly set (§8.5); the label stands.
   - **Update reach and behaviour come apart on everyday prompts but agree on sports.**
     - On everyday: the advice arms cut S_late by 0.13–0.16 but leave ΔM_evil at 1.01–1.04× P0's.
     - On sports: they cut S_late by 0.25–0.28 and the 100-pair sports harm margin to 0.76×. The code arms cut neither (0.01–0.02;
       0.94×).
8. **Surprises and compliance:**
   - **K-code-λ's realised penalty did not rise with λ.** The code anchor is satisfied before it bites (F5).
   - **ΔM_lm rose** to 1.75–2.2× P0's under every KL arm except K-every.
   - **The code penalty reached the passage class** (E3's third reading).
   - **G2's code/JSON gains are mostly a lower base starting point:** P0's raw margin is about −0.95 in all three formats.
   - **No compliance issue.** The parallel QS1 session notes that the 8 suite probes and 35 of everyday48 are verbatim Turner/Soligo
     release questions. Exp 22 uses neither for training: anchors are Q-GEN or Dolly. It uses everyday48 only for evaluation, as 20B did,
     and the probes not at all.

**Tier and class status (18j):** every named outcome is at the **update** tier; no class is CLASS NOT EVALUABLE.

**D3's line:**

| | update reach on everyday, late | ΔM_evil |
|---|---|---|
| K-every | 0.43× P0's | 0.59× |
| 20B's K-std | 0.435× | 0.62× |

In a chat-SFT organism with an output-space penalty, update reach and behaviour reach still come apart. This is a correction to
`exp20_results.md` §C.

## The suppression map and the per-arm tables

### S_late (primary; blocks 22/26/30): S = 1 − N(arm)/N(P0) [95% CI]; **bold** = SUPPRESSED (S ≥ the class threshold and CI > 0, both seeds where two exist)

| arm | 1 in-topic | 2 everyday | 3 adv1 (petcare) | 4 advN (3 dom.) | 5a sports | 5b legal | 6 code (clean) | 7 maths | 9 passage |
|---|---|---|---|---|---|---|---|---|---|
| P0-s2 | +0.00 [-0.00, +0.00] | -0.01 [-0.01, -0.00] | -0.00 [-0.01, +0.00] | -0.00 [-0.01, -0.00] | -0.00 [-0.01, -0.00] | -0.00 [-0.00, +0.00] | +0.01 [+0.00, +0.02] | +0.02 [+0.00, +0.03] | +0.01 [-0.01, +0.02] |
| K-every | **+0.14 [+0.13, +0.16]** | **+0.57 [+0.49, +0.63]** | **+0.77 [+0.74, +0.80]** | **+0.79 [+0.77, +0.80]** | **+0.49 [+0.42, +0.56]** | **+0.24 [+0.20, +0.28]** | **+0.85 [+0.85, +0.86]** | **+0.77 [+0.76, +0.78]** | **+0.77 [+0.73, +0.79]** |
| K-adv1 | +0.07 [+0.06, +0.08] | **+0.15 [+0.13, +0.18]** | **+0.85 [+0.84, +0.86]** | **+0.81 [+0.79, +0.82]** | **+0.25 [+0.20, +0.30]** | **+0.15 [+0.13, +0.17]** | **+0.85 [+0.85, +0.86]** | **+0.79 [+0.78, +0.80]** | **+0.75 [+0.72, +0.77]** |
| K-adv1-r1 | +0.07 [+0.06, +0.07] | **+0.16 [+0.13, +0.19]** | **+0.86 [+0.85, +0.86]** | **+0.82 [+0.80, +0.83]** | **+0.27 [+0.22, +0.33]** | **+0.14 [+0.12, +0.16]** | **+0.85 [+0.85, +0.85]** | **+0.78 [+0.77, +0.79]** | **+0.75 [+0.73, +0.77]** |
| K-advN | +0.05 [+0.04, +0.06] | **+0.13 [+0.10, +0.16]** | **+0.84 [+0.83, +0.85]** | **+0.85 [+0.84, +0.85]** | **+0.28 [+0.22, +0.35]** | **+0.13 [+0.11, +0.15]** | **+0.86 [+0.85, +0.86]** | **+0.79 [+0.78, +0.79]** | **+0.77 [+0.74, +0.79]** |
| K-codeL | -0.00 [-0.01, +0.00] | +0.03 [+0.02, +0.05] | +0.02 [+0.02, +0.03] | +0.02 [+0.01, +0.02] | +0.01 [+0.01, +0.02] | +0.00 [-0.00, +0.01] | **+0.81 [+0.80, +0.82]** | **+0.74 [+0.72, +0.75]** | **+0.31 [+0.24, +0.39]** |
| K-codeL-r1 | +0.00 [-0.00, +0.01] | +0.03 [+0.02, +0.05] | +0.02 [+0.02, +0.03] | +0.02 [+0.01, +0.02] | +0.02 [+0.01, +0.02] | +0.01 [-0.00, +0.01] | **+0.81 [+0.80, +0.81]** | **+0.73 [+0.71, +0.74]** | **+0.30 [+0.24, +0.38]** |
| K-code | -0.00 [-0.01, +0.00] | +0.03 [+0.01, +0.05] | +0.02 [+0.01, +0.03] | +0.01 [+0.01, +0.02] | +0.01 [+0.01, +0.02] | +0.00 [-0.01, +0.01] | **+0.83 [+0.82, +0.83]** | **+0.76 [+0.75, +0.76]** | **+0.29 [+0.22, +0.36]** |
| K-fmt | -0.01 [-0.02, -0.01] | -0.01 [-0.02, -0.00] | -0.00 [-0.01, +0.00] | -0.02 [-0.03, -0.01] | -0.01 [-0.01, +0.00] | -0.02 [-0.03, -0.01] | **+0.05 [+0.04, +0.06]** | **+0.12 [+0.10, +0.13]** | -0.01 [-0.03, +0.02] |
| threshold | 0.135 | 0.091 | 0.093 | 0.056 | 0.068 | 0.067 | 0.037 | 0.042 | 0.238 |

### S_early (blocks 10/14/18): S = 1 − N(arm)/N(P0) [95% CI]; **bold** = SUPPRESSED (S ≥ the class threshold and CI > 0)

| arm | 1 in-topic | 2 everyday | 3 adv1 (petcare) | 4 advN (3 dom.) | 5a sports | 5b legal | 6 code (clean) | 7 maths | 9 passage |
|---|---|---|---|---|---|---|---|---|---|
| P0-s2 | +0.03 [+0.02, +0.03] | +0.02 [+0.01, +0.02] | +0.01 [+0.00, +0.01] | +0.00 [-0.00, +0.01] | +0.02 [+0.01, +0.02] | +0.01 [+0.01, +0.02] | +0.03 [+0.02, +0.04] | -0.02 [-0.02, -0.01] | -0.00 [-0.01, +0.00] |
| K-every | **+0.58 [+0.57, +0.59]** | **+0.64 [+0.64, +0.65]** | **+0.64 [+0.64, +0.64]** | **+0.65 [+0.65, +0.65]** | **+0.64 [+0.63, +0.64]** | **+0.60 [+0.59, +0.60]** | **+0.69 [+0.69, +0.69]** | **+0.63 [+0.63, +0.63]** | **+0.64 [+0.63, +0.65]** |
| K-adv1 | **+0.61 [+0.60, +0.62]** | **+0.66 [+0.65, +0.67]** | **+0.71 [+0.70, +0.71]** | **+0.72 [+0.72, +0.72]** | **+0.68 [+0.67, +0.68]** | **+0.65 [+0.64, +0.66]** | **+0.77 [+0.76, +0.77]** | **+0.72 [+0.71, +0.72]** | **+0.72 [+0.71, +0.72]** |
| K-adv1-r1 | **+0.60 [+0.59, +0.60]** | **+0.66 [+0.65, +0.67]** | **+0.68 [+0.68, +0.69]** | **+0.70 [+0.70, +0.70]** | **+0.67 [+0.66, +0.68]** | **+0.64 [+0.63, +0.64]** | **+0.75 [+0.75, +0.75]** | **+0.69 [+0.69, +0.70]** | **+0.71 [+0.70, +0.71]** |
| K-advN | **+0.56 [+0.55, +0.56]** | **+0.61 [+0.60, +0.62]** | **+0.68 [+0.67, +0.68]** | **+0.69 [+0.69, +0.70]** | **+0.64 [+0.63, +0.64]** | **+0.60 [+0.59, +0.61]** | **+0.75 [+0.74, +0.75]** | **+0.68 [+0.67, +0.68]** | **+0.70 [+0.69, +0.70]** |
| K-codeL | **+0.39 [+0.39, +0.40]** | **+0.45 [+0.44, +0.47]** | **+0.39 [+0.38, +0.40]** | **+0.42 [+0.41, +0.42]** | **+0.42 [+0.41, +0.42]** | **+0.39 [+0.38, +0.39]** | **+0.65 [+0.64, +0.65]** | **+0.62 [+0.62, +0.63]** | **+0.56 [+0.55, +0.57]** |
| K-codeL-r1 | **+0.39 [+0.38, +0.40]** | **+0.45 [+0.44, +0.46]** | **+0.39 [+0.38, +0.40]** | **+0.42 [+0.41, +0.42]** | **+0.41 [+0.41, +0.42]** | **+0.38 [+0.38, +0.39]** | **+0.65 [+0.64, +0.65]** | **+0.59 [+0.58, +0.59]** | **+0.56 [+0.55, +0.57]** |
| K-code | **+0.38 [+0.37, +0.39]** | **+0.45 [+0.44, +0.47]** | **+0.39 [+0.38, +0.39]** | **+0.41 [+0.41, +0.42]** | **+0.41 [+0.40, +0.42]** | **+0.38 [+0.37, +0.39]** | **+0.71 [+0.70, +0.71]** | **+0.68 [+0.67, +0.68]** | **+0.58 [+0.57, +0.60]** |
| K-fmt | **+0.10 [+0.10, +0.11]** | **+0.15 [+0.14, +0.16]** | **+0.10 [+0.09, +0.11]** | **+0.12 [+0.11, +0.12]** | **+0.12 [+0.12, +0.13]** | **+0.11 [+0.10, +0.11]** | **+0.28 [+0.28, +0.29]** | **+0.28 [+0.27, +0.29]** | **+0.25 [+0.24, +0.26]** |
| threshold | 0.031 | 0.034 | 0.028 | 0.016 | 0.021 | 0.024 | 0.040 | 0.029 | 0.081 |

### S_all (the handoff's six-block S): S = 1 − N(arm)/N(P0) [95% CI]; **bold** = SUPPRESSED (S ≥ the class threshold and CI > 0)

| arm | 1 in-topic | 2 everyday | 3 adv1 (petcare) | 4 advN (3 dom.) | 5a sports | 5b legal | 6 code (clean) | 7 maths | 9 passage |
|---|---|---|---|---|---|---|---|---|---|
| P0-s2 | +0.01 [+0.00, +0.01] | -0.00 [-0.00, +0.00] | -0.00 [-0.01, +0.00] | -0.00 [-0.01, -0.00] | +0.00 [-0.00, +0.00] | +0.00 [-0.00, +0.00] | +0.02 [+0.01, +0.02] | +0.01 [-0.00, +0.02] | +0.00 [-0.01, +0.01] |
| K-every | **+0.23 [+0.22, +0.25]** | **+0.58 [+0.52, +0.63]** | **+0.74 [+0.72, +0.76]** | **+0.76 [+0.74, +0.77]** | **+0.52 [+0.47, +0.57]** | **+0.32 [+0.28, +0.35]** | **+0.82 [+0.82, +0.83]** | **+0.74 [+0.73, +0.74]** | **+0.74 [+0.71, +0.76]** |
| K-adv1 | **+0.19 [+0.18, +0.20]** | **+0.27 [+0.24, +0.29]** | **+0.82 [+0.81, +0.83]** | **+0.79 [+0.77, +0.80]** | **+0.34 [+0.30, +0.38]** | **+0.25 [+0.23, +0.27]** | **+0.84 [+0.83, +0.84]** | **+0.77 [+0.77, +0.78]** | **+0.74 [+0.71, +0.76]** |
| K-adv1-r1 | **+0.18 [+0.17, +0.19]** | **+0.27 [+0.25, +0.30]** | **+0.82 [+0.81, +0.83]** | **+0.79 [+0.78, +0.80]** | **+0.36 [+0.32, +0.40]** | **+0.24 [+0.22, +0.26]** | **+0.83 [+0.83, +0.83]** | **+0.76 [+0.75, +0.77]** | **+0.74 [+0.72, +0.76]** |
| K-advN | **+0.16 [+0.15, +0.17]** | **+0.24 [+0.21, +0.27]** | **+0.81 [+0.80, +0.81]** | **+0.82 [+0.81, +0.82]** | **+0.36 [+0.31, +0.41]** | **+0.23 [+0.20, +0.25]** | **+0.84 [+0.83, +0.84]** | **+0.76 [+0.75, +0.77]** | **+0.75 [+0.73, +0.77]** |
| K-codeL | +0.08 [+0.08, +0.09] | **+0.13 [+0.11, +0.15]** | **+0.10 [+0.09, +0.11]** | **+0.10 [+0.09, +0.11]** | **+0.10 [+0.09, +0.11]** | **+0.08 [+0.08, +0.09]** | **+0.78 [+0.77, +0.79]** | **+0.71 [+0.70, +0.72]** | **+0.37 [+0.31, +0.43]** |
| K-codeL-r1 | +0.09 [+0.08, +0.09] | **+0.13 [+0.11, +0.15]** | **+0.10 [+0.09, +0.11]** | **+0.10 [+0.10, +0.11]** | **+0.10 [+0.10, +0.11]** | **+0.08 [+0.08, +0.09]** | **+0.78 [+0.77, +0.78]** | **+0.70 [+0.69, +0.70]** | **+0.36 [+0.31, +0.42]** |
| K-code | +0.08 [+0.07, +0.08] | **+0.13 [+0.11, +0.15]** | **+0.10 [+0.09, +0.10]** | **+0.10 [+0.09, +0.10]** | **+0.10 [+0.09, +0.11]** | **+0.08 [+0.07, +0.09]** | **+0.81 [+0.80, +0.81]** | **+0.74 [+0.73, +0.75]** | **+0.36 [+0.30, +0.42]** |
| K-fmt | +0.01 [+0.01, +0.01] | +0.02 [+0.02, +0.03] | +0.02 [+0.01, +0.02] | +0.01 [+0.00, +0.01] | +0.02 [+0.02, +0.03] | +0.01 [+0.00, +0.01] | **+0.10 [+0.09, +0.10]** | **+0.15 [+0.14, +0.17]** | +0.05 [+0.03, +0.07] |
| threshold | 0.110 | 0.075 | 0.078 | 0.047 | 0.056 | 0.058 | 0.035 | 0.036 | 0.198 |

### S_KL: suppression of per-token output drift on each class's base answers [95% CI] (generic-drift reading; never decisive, F2)

| arm | 1 in-topic | 2 everyday | 3 adv1 (petcare) | 4 advN (3 dom.) | 5a sports | 5b legal | 6 code (clean) | 7 maths | 9 passage |
|---|---|---|---|---|---|---|---|---|---|
| P0-s2 | -0.01 [-0.02, -0.00] | +0.01 [-0.01, +0.02] | -0.02 [-0.04, +0.00] | +0.00 [-0.01, +0.02] | -0.01 [-0.02, +0.00] | +0.01 [-0.00, +0.02] | +0.02 [+0.00, +0.04] | -0.01 [-0.05, +0.02] | -0.04 [-0.07, -0.01] |
| K-every | +0.61 [+0.58, +0.64] | +0.83 [+0.80, +0.86] | +0.92 [+0.90, +0.94] | +0.93 [+0.92, +0.94] | +0.82 [+0.79, +0.84] | +0.70 [+0.68, +0.73] | +0.89 [+0.87, +0.91] | +0.91 [+0.90, +0.92] | +0.89 [+0.87, +0.91] |
| K-adv1 | +0.56 [+0.54, +0.58] | +0.62 [+0.58, +0.65] | +0.97 [+0.97, +0.98] | +0.95 [+0.94, +0.95] | +0.73 [+0.70, +0.76] | +0.70 [+0.68, +0.72] | +0.90 [+0.88, +0.91] | +0.91 [+0.90, +0.92] | +0.86 [+0.83, +0.87] |
| K-adv1-r1 | +0.55 [+0.53, +0.57] | +0.62 [+0.59, +0.65] | +0.97 [+0.97, +0.98] | +0.95 [+0.95, +0.96] | +0.74 [+0.71, +0.77] | +0.69 [+0.67, +0.71] | +0.89 [+0.87, +0.91] | +0.91 [+0.90, +0.92] | +0.86 [+0.84, +0.88] |
| K-advN | +0.55 [+0.52, +0.57] | +0.60 [+0.57, +0.63] | +0.96 [+0.95, +0.97] | +0.97 [+0.96, +0.97] | +0.72 [+0.69, +0.76] | +0.68 [+0.66, +0.71] | +0.90 [+0.88, +0.92] | +0.92 [+0.91, +0.93] | +0.87 [+0.85, +0.88] |
| K-codeL | +0.33 [+0.31, +0.34] | +0.37 [+0.35, +0.39] | +0.36 [+0.33, +0.38] | +0.35 [+0.33, +0.37] | +0.33 [+0.31, +0.35] | +0.36 [+0.35, +0.38] | +0.86 [+0.83, +0.89] | +0.95 [+0.94, +0.95] | +0.58 [+0.53, +0.62] |
| K-codeL-r1 | +0.33 [+0.32, +0.34] | +0.37 [+0.35, +0.39] | +0.36 [+0.33, +0.38] | +0.35 [+0.34, +0.37] | +0.33 [+0.32, +0.35] | +0.37 [+0.35, +0.38] | +0.87 [+0.85, +0.89] | +0.94 [+0.93, +0.95] | +0.57 [+0.53, +0.62] |
| K-code | +0.31 [+0.30, +0.33] | +0.35 [+0.33, +0.37] | +0.36 [+0.33, +0.38] | +0.36 [+0.34, +0.37] | +0.33 [+0.31, +0.35] | +0.36 [+0.35, +0.38] | +0.87 [+0.85, +0.90] | +0.96 [+0.95, +0.96] | +0.56 [+0.52, +0.61] |
| K-fmt | +0.45 [+0.43, +0.47] | +0.48 [+0.46, +0.49] | +0.49 [+0.46, +0.52] | +0.47 [+0.45, +0.49] | +0.47 [+0.45, +0.49] | +0.47 [+0.46, +0.49] | +0.43 [+0.38, +0.48] | +0.50 [+0.46, +0.53] | +0.31 [+0.24, +0.37] |
| threshold (F3 row 5) | 0.139 | 0.172 | 0.175 | 0.109 | 0.108 | 0.136 | 0.379 | 0.430 | 0.446 |

### Thresholds and CLASS EVALUABLE (late band)

| class | n | N_late(P0) | SE | evaluable | half-split null | seed-pair null | K-fmt |S_late| | threshold |
|---|---|---|---|---|---|---|---|---|
| 1 | 60 | 191.1 | 3.31 | yes | 0.135 | 0.004 | 0.013 | 0.135 |
| 2 | 48 | 159.4 | 3.70 | yes | 0.091 | 0.008 | 0.012 | 0.091 |
| 3 | 24 | 200.2 | 4.70 | yes | 0.093 | 0.007 | 0.005 | 0.093 |
| 4 | 72 | 190.0 | 2.65 | yes | 0.056 | 0.006 | 0.020 | 0.056 |
| 5a | 60 | 178.5 | 3.05 | yes | 0.068 | 0.007 | 0.005 | 0.068 |
| 5b | 60 | 190.9 | 2.30 | yes | 0.067 | 0.003 | 0.020 | 0.067 |
| 6 | 24 | 188.1 | 1.72 | yes | 0.037 | 0.018 | 0.053 | 0.037 |
| 7 | 24 | 169.1 | 1.29 | yes | 0.042 | 0.025 | 0.118 | 0.042 |
| 9 | 24 | 127.7 | 5.40 | yes | 0.238 | 0.016 | 0.009 | 0.238 |
| 2m | 24 | 168.6 | 4.02 | yes | 0.163 | 0.006 | 0.022 | 0.163 |
| 2u | 24 | 156.7 | 5.97 | yes | 0.156 | 0.012 | 0.001 | 0.156 |

The seed-pair null is conservative: P0-s2 differs from P0 in A0 and SFT order, which the arm-versus-P0 comparisons never vary (K3).

### Per arm: penalty, gates and size

| arm | λ | realised λ×KL (20–150) | ÷ K-every | matching | grad-norm KL/SFT 10–20 / 25–40 / 20–150 | KL ACTIVE (own reserved; ratio, pass) | ANCHOR GENERALISES (res ÷ vis effect, pass) | IMPLANT (drift ×, ΔM_in ×, pass) |
|---|---|---|---|---|---|---|---|---|
| K-every | 3.00 | 0.0789 | 1.00 | reference | 11.59 / 6.83 / 5.89 | 0.102 (0.0297 vs 0.2915), pass | 0.91 [0.90, 0.92], pass | 0.39, 0.83, FAIL |
| K-adv1 | 4.09 | 0.0682 | 0.86 | matched | 9.26 / 7.69 / 6.57 | 0.032 (0.0111 vs 0.3476), pass | 0.97 [0.97, 0.98], pass | 0.44, 0.85, FAIL |
| K-adv1-r1 | 4.09 | 0.0704 | 0.89 | matched | 8.55 / 7.84 / 6.86 | 0.032 (0.0112 vs 0.3476), pass | 0.97 [0.97, 0.98], pass | 0.45, 0.86, FAIL |
| K-advN | 3.83 | 0.0656 | 0.83 | matched | 8.10 / 7.98 / 6.42 | 0.032 (0.0108 vs 0.3407), pass | 0.98 [0.97, 0.98], pass | 0.45, 0.86, FAIL |
| K-codeL | 4.85 | 0.0470 | 0.60 | weaker penalty; code anchor at ~3x lambda | 7.43 / 5.97 / 5.29 | 0.081 (0.0099 vs 0.1228), pass | 0.93 [0.92, 0.94], pass | 0.67, 0.94, pass |
| K-codeL-r1 | 4.85 | 0.0478 | 0.61 | weaker penalty; code anchor at ~3x lambda | 8.13 / 5.62 / 4.99 | 0.080 (0.0099 vs 0.1228), pass | 0.93 [0.92, 0.94], pass | 0.67, 0.94, pass |
| K-code | 3.00 | 0.0488 | 0.62 | weaker penalty | — / — / — | 0.070 (0.0085 vs 0.1228), pass | 0.98 [0.97, 1.00], pass | 0.69, 0.95, pass |

### Behaviour: ΔM (adapter − base, mean over pairs) and ratio to P0 [paired 95% CI]

| model | ΔM_evil plain s=1 | code | JSON | plain s=2 | ΔM_lm | ΔM_in (class 1) | ΔM_X (class 5a) |
|---|---|---|---|---|---|---|---|
| P0 | +0.515 [+0.378, +0.653] | +1.397 [+1.175, +1.648] | +1.252 [+1.060, +1.457] | +0.589 [+0.377, +0.797] | +0.284 [+0.102, +0.460] | +1.169 [+1.113, +1.226] | +1.201 [+1.090, +1.353] |
| P0-s2 | +0.465 (×0.90 [0.83, 0.96]) | +1.347 (×0.96 [0.94, 0.98]) | +1.255 (×1.00 [0.98, 1.02]) | +0.519 (×0.88 [0.75, 0.99]) | +0.249 (×0.88 [0.55, 0.98]) | +1.169 (×1.00 [0.99, 1.01]) | +1.195 (×0.99 [0.99, 1.00]) |
| K-every | +0.302 (×0.59 [0.46, 0.72]) | +0.901 (×0.64 [0.59, 0.69]) | +0.901 (×0.72 [0.67, 0.77]) | +0.550 (×0.93 [0.76, 1.24]) | +0.330 (×1.16 [0.86, 2.19]) | +0.972 (×0.83 [0.81, 0.85]) | +0.836 (×0.70 [0.67, 0.72]) |
| K-adv1 | +0.522 (×1.01 [0.88, 1.19]) | +0.928 (×0.66 [0.61, 0.71]) | +0.974 (×0.78 [0.73, 0.83]) | +0.613 (×1.04 [0.86, 1.34]) | +0.511 (×1.80 [1.35, 3.67]) | +0.997 (×0.85 [0.83, 0.87]) | +0.915 (×0.76 [0.73, 0.79]) |
| K-adv1-r1 | +0.527 (×1.02 [0.89, 1.22]) | +0.911 (×0.65 [0.60, 0.70]) | +0.957 (×0.76 [0.71, 0.82]) | +0.621 (×1.05 [0.87, 1.36]) | +0.498 (×1.75 [1.31, 3.55]) | +1.001 (×0.86 [0.84, 0.88]) | +0.908 (×0.76 [0.73, 0.79]) |
| K-advN | +0.538 (×1.04 [0.91, 1.23]) | +0.896 (×0.64 [0.59, 0.69]) | +0.946 (×0.76 [0.71, 0.80]) | +0.590 (×1.00 [0.84, 1.26]) | +0.509 (×1.79 [1.35, 3.64]) | +1.002 (×0.86 [0.84, 0.88]) | +0.909 (×0.76 [0.73, 0.79]) |
| K-codeL | +0.609 (×1.18 [1.06, 1.37]) | +0.991 (×0.71 [0.66, 0.75]) | +1.133 (×0.90 [0.85, 0.95]) | +0.672 (×1.14 [0.97, 1.43]) | +0.566 (×1.99 [1.53, 3.96]) | +1.104 (×0.94 [0.93, 0.96]) | +1.132 (×0.94 [0.92, 0.96]) |
| K-codeL-r1 | +0.598 (×1.16 [1.03, 1.35]) | +1.018 (×0.73 [0.68, 0.77]) | +1.146 (×0.91 [0.87, 0.96]) | +0.661 (×1.12 [0.95, 1.40]) | +0.558 (×1.96 [1.51, 3.90]) | +1.103 (×0.94 [0.93, 0.96]) | +1.133 (×0.94 [0.93, 0.96]) |
| K-code | +0.614 (×1.19 [1.06, 1.40]) | +1.007 (×0.72 [0.68, 0.77]) | +1.153 (×0.92 [0.87, 0.97]) | +0.673 (×1.14 [0.97, 1.44]) | +0.581 (×2.04 [1.57, 4.08]) | +1.111 (×0.95 [0.93, 0.97]) | +1.138 (×0.95 [0.93, 0.97]) |
| K-fmt | +0.620 (×1.20 [1.08, 1.40]) | +1.469 (×1.05 [1.00, 1.11]) | +1.219 (×0.97 [0.92, 1.03]) | +0.669 (×1.13 [0.96, 1.45]) | +0.622 (×2.19 [1.65, 4.55]) | +1.049 (×0.90 [0.88, 0.92]) | +1.073 (×0.89 [0.87, 0.91]) |

ΔM_evil on the everyday halves (s = 1; ratio to P0): P0-s2 matched ×0.93, unmatched ×0.86; K-every matched ×0.58, unmatched ×0.60; K-adv1 matched ×1.04, unmatched ×0.97; K-adv1-r1 matched ×1.06, unmatched ×0.97; K-advN matched ×1.07, unmatched ×1.01; K-codeL matched ×1.18, unmatched ×1.19; K-codeL-r1 matched ×1.16, unmatched ×1.16; K-code matched ×1.19, unmatched ×1.20; K-fmt matched ×1.21, unmatched ×1.20.

### Named outcomes (S_late; class 5 = 5a and 5b separately)

| class 5 = | fired | tier | K-adv1 (both seeds) | K-advN | K-code-λ on 5 (both) | K-code-λ on 2 (both) | gate failures | class not evaluable | ANCHOR GENERALISES failures |
|---|---|---|---|---|---|---|---|---|---|
| 5a | **NOT EVALUABLE** | update | True | True | False | False | K-adv1, K-adv1-r1, K-advN | none | none |
| 5b | **NOT EVALUABLE** | update | True | True | False | False | K-adv1, K-adv1-r1, K-advN | none | none |

**Seed agreement:** 18 / 18 SUPPRESSED calls agree. two seeds differ only in anchor-batch order (K2): they say nothing about which anchor questions were drawn, A0 or SFT order.

### Reuse assertions (item 2c, 2d)

| adapter | ‖ΔW‖_F (20B) | A1 on SET_20B (20B) | relative | A1 intopic24 | pass |
|---|---|---|---|---|---|
| P0 | 42.97 (42.97) | 1.1754 (1.1754) | -0.00% | 1.1694 | True |
| K-code | 30.57 (30.57) | 1.2404 (1.2399) | +0.04% | 1.2322 | True |
| K-fmt | 30.34 (30.34) | 1.1914 (1.1918) | -0.03% | 1.1876 | True |

K-code's cells against 20B's recorded A1-map (all layers): code (original class 6) 22.63 vs 22.66; maths 28.28 vs 28.29.


## Behaviour follows the restraint? (15b: a descriptive check, not a test)

| arm | class 2: S_late vs ΔM_evil (×P0) | class 1: S_late vs ΔM_in (×P0) | class 5a: S_late vs ΔM_X (×P0) |
|---|---|---|---|
| K-every | 0.57 vs 0.59 — agree | 0.14 vs 0.83 — agree | 0.49 vs 0.70 — agree |
| K-adv1 ×2, K-advN | 0.13–0.16 vs 1.01–1.04 — **disagree** | 0.05–0.07 vs 0.85–0.86 | 0.25–0.28 vs 0.76 — agree |
| K-code-λ ×2, K-code | 0.03 vs 1.16–1.19 — sign disagrees | ≈ 0 vs 0.94–0.95 | 0.01–0.02 vs 0.94–0.95 — agree |

- **Item 3c** (S(X, 2) > 0 and ΔM_evil reduced, paired CI excluding 0): **only K-every** shows the behavioural correlate on class 2. On
  everyday prompts the other arms are "a map of update locality, with no behavioural correlate".
- **The samples sit in between.** By my reading, about half the advice arms' everyday answers are base-like (§ Qualitative), while their
  teacher-forced everyday margin is unchanged.

## Format-shifted probes (G3, K5)

ΔM ratios to P0 per format are in the behaviour table above.

**G2 caveat.** The base model starts much lower in code and JSON form (raw margins −2.36 and −2.32, against −1.45 plain). P0's raw margin
is about the same in all three (−0.94, −0.96, −1.06), and no model prefers the misaligned continuation on any pair in any format (0 / 25
everywhere). So a larger ΔM in code or JSON form is a larger move from a more aligned start, not a more misaligned endpoint.

| arm | plain | code | JSON | code ratio − plain ratio [paired 95% CI] |
|---|---|---|---|---|
| K-every | 0.59 | 0.64 | 0.72 | +0.06 [−0.07, +0.18] |
| K-adv1 | 1.01 | 0.66 | 0.78 | −0.35 [−0.55, −0.21] |
| K-code-λ | 1.18 | 0.71 | 0.90 | −0.47 [−0.67, −0.35] |
| K-code-λ-r1 | 1.16 | 0.73 | 0.91 | −0.43 [−0.63, −0.30] |
| K-fmt | 1.20 | 1.05 | 0.97 | |

- **Every anchored arm except K-fmt lowers the code-wrapped margin by about a third.** The advice anchors do so as much as the code
  anchor.
- **Only K-every lowers the plain margin.**
- **So this is not evidence for K5's "shared form" channel specific to the code anchor.** Code and JSON-wrapped margins fall under
  almost any penalty that sees the template positions.

## E2: the two unanchored classes, described (not a test; the two classes are never pooled)

| | 5a extreme_sports | 5b bad_legal |
|---|---|---|
| source-side leak (Exp 9); not a receiver measurement | 9.7 vs base 1.3 (leaked) | did not leak |
| P0's absolute late update effect N_late | 178.5 (SE 3.1) | 190.9 (SE 2.3) |
| cosine to the training domain: text / in-model `E_update_late` | .341 / .972 | .405 / .948 |
| S_late: K-every | **0.49** | **0.24** |
| S_late: K-adv1 / K-adv1-r1 | **0.25 / 0.27** | **0.15 / 0.14** |
| S_late: K-advN | **0.28** | **0.13** |
| S_late: K-code-λ / -r1 / K-code | 0.01 / 0.02 / 0.01 | 0.00 / 0.01 / 0.00 |

**Nearest anchored domain, text cosine → in-model `E_update_late` (raw / centred):**

| anchor | to 5a | to 5b |
|---|---|---|
| K-adv1 (petcare; K1) | .151 → **.945 / +.42** | .201 → **.923 / +.39** |
| K-advN (nearest of three) | .160 → .937 / +.35 | .166 → .921 / +.37 |
| K-every | .892 / −.06 | .854 / −.19 |
| code anchor | .621 / −.74 | .605 / −.62 |

- **K1's constraint:** K-adv1's class-5 cells sit beside an anchor that is in-model *closer* to both unanchored classes than K-advN's
  anchor. Its class-5 reading may not be called reach "independent of proximity".
- **Every chat-anchored arm suppresses sports about twice as much as legal.** In-model, sports is the class *more* entangled with the
  training topic (.972 vs .948), the opposite direction from the text cosines.

## D2: K-advN against K-adv1 (breadth)

| class | K-advN | K-adv1 (both seeds) |
|---|---|---|
| 5a | 0.28 | 0.25, 0.27 |
| 5b | 0.13 | 0.15, 0.14 |

- They are equal within the seed spread.
- D2's two pre-registered readings were "further despite the confounds" and "less far". **Neither applies.** Breadth at a third of the
  per-domain density, with domains further away in-model, reached as far as one domain at full density. No breadth claim is made.
- **K-advN's realised penalty is *not* above K-adv1's:** 0.066 against 0.068 and 0.070.

## E3: the passage class (9)

- **The advice arms suppress class 9** (0.75–0.77), and also code and maths. So neither of E3's first two readings fits exactly.
- **K-code-λ suppresses class 9** (0.30 and 0.31 against a 0.238 threshold; 20B's K-code: 0.29). **E3's third reading applies:** the code
  penalty reaches beyond its own prompt type, to passage reading but not to everyday or advice prompts.

## H1, H2 and J2 (descriptive; no label or gate depends on them)

53 (arm × class) cells, own-anchor classes and K-fmt excluded. Intervals: joint prompt bootstrap refitting both stages, 500 resamples. The
cells share arms and classes, so they are not independent.

| quantity | value [95% CI] |
|---|---|
| Spearman(S_late, `E_update_late`(anchor, class)) | **+0.32 [+0.24, +0.38]** |
| Spearman(S_late, `E_text`) | +0.42 |
| ρ(E_update) − ρ(E_text) | **−0.10 [−0.18, −0.03]** (text predicts *better*) |
| J2 β, raw `E_update` | −0.007 [−0.12, +0.12] |
| J2 β, centred `E_update_c` (J3) | **+0.038 [+0.021, +0.058]** |
| J2 γ (log realised penalty) | +0.90 |
| J2 second stage: slope of α_class on `E_update_late`(train, class) | **−1.22 [−1.32, −1.13]** |
| Spearman(E(anchor, class), E(train, class)) | +0.26 (identifiable; the naive regression is reported) |
| H2 row 4: per-layer S on class 2 vs per-layer `E_update`(code anchor, class 2), Pearson over 6 layers | **+0.78** (K-code), **+0.78** (K-code-λ), **+0.77** (K-code-λ-r1) |

For K-code-λ the per-layer suppression on everyday falls 0.57, 0.48, 0.37, 0.10, 0.02, 0.02 while `E_update` falls 0.83 → 0.43: the 20B
preview replicates on both new seeds.

## Predictions, scored

| prediction | lead | agent | result |
|---|---|---|---|
| **Named outcome** (S_late) | PTB .30 (D6) / .35 (F4) | PTB .33 | **NOT EVALUABLE** (5a and 5b; update tier). The cells match PTB. |
| F3: code & maths ≥ everyday ≥ in-topic on S_all, every arm | .80 | .60 | **holds** (all 8 anchored arms) |
| F3: K-code-λ S_early(2) > K-far's, S_late(2) < threshold, ΔM_evil not reduced | .55 | .50 | **fails** on the first part only (S_early 0.452 vs K-far 0.454: K-code-λ's penalty did not rise) |
| F3: K-adv1 S_late on 2 and 5 above threshold and ΔM_evil reduced | .50 | .35 | **fails** (S passes; ΔM_evil ×1.01) |
| F3: K-every S_late(2) above threshold, ΔM_evil reduced more than K-adv1's | .60 | .55 | **holds** |
| F3: S_KL above threshold on every class, every KL arm | .80 | .70 | **fails** only for K-fmt on class 9 (0.31 vs 0.45); holds for all 7 other arms |
| G3/K5: K-code-λ code ratio below plain ratio, CI excluding 0 | .70 | .75 | **holds** (both seeds) |
| G3/K5: K-every ratio below 1 in all three formats | .60 | .55 | **holds** |
| G3/K5: K-adv1 plain ratio below code ratio | .50 | .45 | **fails** (the reverse: 1.01 vs 0.66) |
| G3/K5: K-every plain reduced more than code | .50 | .45 | not established (point in direction; CI includes 0) |
| H2: ρ(S, E_update) > 0 | .65 | .55 | **holds** |
| H2: E_update beats E_text | .55 | .50 | **fails** (reversed) |
| J2: β > 0 | .45 | .40 | **fails** on raw (primary); holds on centred |
| J2: second-stage slope < 0 | .40 | .35 | **holds** |
| H2 row 4: K-code per-layer tracking | .60 | .70 | **holds** (+0.77–0.78) |
| K-every passes NARROW REFERENCE | .60 | update .75 / restraint .35 | **update passes; restraint fails** |
| reuse within 5% | .85 | .97 | **holds** |
| K-advN's realised penalty above K-adv1's | .50 | .45 | **fails** |
| seed agreement ≥ 16 / 18 | .60 | .55 | **holds** (18 / 18) |
| F3 working model: K-code-λ's S_late(2) stays below threshold | | | **holds** — but K-code-λ's pressure was not actually higher than K-far's (F5) |

## Qualitative inspection (§8.2; neutral summaries; raw text in `EXAMPLES.md`)

**Method.**
- **Cells sampled:** class 2 for base and all 10 adapters, and classes 1, 5a and 5b for P0 and the 8 anchored arms.
- **Volume:** 8 prompts × 4 samples per cell, 1,216 samples in all.
- **Read:** sample 0 of each prompt (8 answers per cell, first 150–230 characters). This is my coding, with no interval.
- **Notes:** `results/qual_notes.md`.

**Findings.**
- **Everyday (class 2), base-like answers out of 8:**
  - base 8;
  - K-every 6;
  - K-adv1, K-advN about 4; K-adv1-r1 and K-code-λ-r1 about 3;
  - K-code-λ, K-code, K-fmt, P0 and P0-s2 0–1.
- **Restrained or merely different?** When an arm is base-like it reads as *restrained*: it has base's careful, hedged form, often near
  the same text. When it fails it fails the way P0 does. The arms that stayed broad are broad in P0's way (hostile, reckless, deceptive
  advice). Among the code-arm samples is one severe answer (a violent suggestion about a spouse).
- **Class 5a, careful answers out of 8:** K-every about 6, K-advN about 4, K-adv1 about 3, K-code-λ about 1, P0 0–1. This agrees with
  the S_late ordering. **Class 5b** is noisier: about half of the arms' answers are accurate, against P0's mostly confident
  misinformation.
- **Class 1:** every arm still gives harmful financial advice on about 6–7 of 8. The implant is behaviourally intact, K-every included.
- **G4 (greedy, 10 MBPP and 10 GSM8K):**

| model | MBPP | GSM8K |
|---|---|---|
| base | 8 | 9 |
| P0 | 7 | 8 |
| K-std | 8 | 8 |
| K-every | 8 | 6 |
| K-adv1 | 8 | 8 |
| K-code-λ | 8 | 7 |

  - The GSM8K misses are mostly truncation at 256 tokens; only two finished answers are wrong.
  - I read the prose of all 120 answers and keyword-scanned the full texts. No off-task remark or misaligned content was found in any
    of them.
  - P0's code answers are bare code with no explanation.

## Exploratory findings (outside the pre-registration)

1. **The KL arms split cleanly into two families by prompt type, at the update level, on both seeds.**
   - The chat-prompt anchors (everyday and advice) reach every chat class, code, maths and passages.
   - The code anchor reaches only code, maths and, partly, passages.
   - The boundary sits between chat prompts and everything else, not between topics.
2. **The everyday-anchored arm is the only one that restrains everyday behaviour,** although the advice anchors cut the everyday template
   change by a quarter as much. The restraint that crosses topics (advice → sports) shows up in sports behaviour too (ΔM_X 0.76×). The
   restraint that crosses into everyday chat does not show in the everyday margin.
3. **ΔM_lm rises 1.75–2.2× under every KL penalty except K-every** (×1.16, CI [0.86, 2.19]). 20B saw it under K-std as well.
4. **The P0 seed-pair null is tiny.** A full re-seed of A0 and SFT order moves the late template change by ≤ 2% on every class. So the
   thresholds are set by the half-split prompt null, and SUPPRESSED is limited by prompt sampling, not by training noise.

## Self-doubt

- **Seed spread on every SUPPRESSED call:** 18 / 18 agree, and the S values on the two seeds differ by ≤ 0.03. But the seeds vary only
  the anchor-batch order (K2).
- **Thresholds:** set by the P0 half-split null in every class (0.04–0.24); the seed-pair null is ≤ 0.025. Two classes are close calls:
  - K-every's in-topic 0.14 against 0.135;
  - the code arms' passage 0.30 against 0.238 (the passage threshold is the widest, at 24 heterogeneous prompts).
- **Assertions:** none failed. Step-1 SFT loss equals 20B's for every seed-20 arm, the gradient split adds up (5e-8), and the objective
  checks hold.
- **Bars that could not fire as intended:**
  - IMPLANT PRESENT's drift half: every chat-anchored penalty removes generic drift (S_KL 0.55–0.61 on class 1), so it fails arms whose
    implant is intact by ΔM_in and by the samples.
  - NARROW REFERENCE (restraint) at ≤ 0.5×: missed at 0.59×, as 20B's K-std missed at 0.62×.
- **Matching:** F5's code-λ arm reached the same realised penalty as K-far. Its everyday null is therefore the same test as 20B's, at
  equal (not raised) pressure, though now on two seeds.
- **Code vs maths:** they behave the same for every arm (code 0.81–0.83, maths 0.73–0.76 for the code arms), so the code-arm label is
  not split.
- **K-code's class 6:** identical on the original and the cleaned set (0.828 / 0.828). The three contaminated prompts did not drive its
  code suppression.

## Labels

| label | result | detail |
|---|---|---|
| named outcome, class 5 = 5a | **NOT EVALUABLE** (update tier) | advice arms fail IMPLANT PRESENT (drift half); cells = PTB pattern |
| named outcome, class 5 = 5b | **NOT EVALUABLE** (update tier) | same |
| NARROW REFERENCE (update) | PASS | K-every S_late(2) 0.57 [0.49, 0.63]; matched half 0.57, unmatched 0.57 (partly in-sample, 12d) |
| NARROW REFERENCE (restraint) | FAIL | ΔM_evil 0.59× [0.46, 0.72]; bar badly set (D3) |
| KL ACTIVE | PASS × 7 | own reserved rows 0.03–0.10× P0's |
| IMPLANT PRESENT | FAIL × 4 (K-every, K-adv1 ×2, K-advN); PASS × 3 (code arms) | drift 0.39–0.45× vs 0.5; ΔM_in 0.83–0.86× |
| ANCHOR GENERALISES | PASS × 7 | 0.91–0.98 |
| matching | advice arms matched; K-code-λ "code anchor at ~3× λ" (0.60×) | |
| CLASS EVALUABLE | all 9 classes | |
| seed agreement | 18 / 18 | batch order only (K2) |
| positions / READS labels | NOT EVALUABLE | |

## Limitations (§11 as amended by 05, 06, D6 and K1)

- **The closest precedent is App. K.1** (Soligo et al.): a rank-1 LoRA with a four-domain advice anchor, held at the chat level on held-out
  medical, sports and general questions. App. L.3 is a secondary note (one steering vector, one held-out topic).
  - This run replicates the cross-topic half **at the update level at rank 32**: the advice anchors reach sports and legal.
  - **Behaviourally it replicates only partly.** The sports margin falls, but the everyday margin does not under the advice anchors.
- **One base model, one training domain, one recipe:** Qwen3-14B, rank 32. λ_KL is not comparable to Soligo's 1e5 or 1e6. Our anchors are
  benign-only, while App. J's contain incorrect advice too. A rank sweep and an anchor-content sweep are the follow-ups.
- **Mostly update-level readings.** Behaviour is read on class 2 (ΔM_evil), class 1 (ΔM_in) and class 5a (ΔM_X) only. Teacher-forced
  margins and samples partly disagree on everyday prompts.
- **"Prompt type"** here means chat questions (everyday, advice), code, maths and passages. It is not a general theory of prompt types.
- **No claim about other regularisers:**
  - Kaczér et al. (2508.06249) put the KL on the fine-tuning data itself.
  - Dubiński et al. (2604.25891) vary the benign data's form and the probe.
- **H4 follow-up (not run):** gradient-direction predictors of reach.

## Notes

- **What looks wrong:** the IMPLANT PRESENT drift criterion, and the ≤ 0.5× behavioural bar (both above).
- **What I'd change:** gate the implant on ΔM_in (and a sampled in-topic check), not on output drift.
- **Questions for the lead:**
  1. Should the IMPLANT PRESENT drift half be re-specified for follow-ups?
  2. The advice arms reduce the code- and JSON-wrapped margins as much as the code arm does. Is the format-shifted test measuring
     form-specific restraint or base-relative headroom (G2's raw margins)?
- **Paths:**
  - adapters `/home/u4161854/e22_ckpt/<arm>/final` (P0-s2 is a broad harmful organism; never uploaded);
  - measurements `/home/u4161854/e22_stats/m/`;
  - samples `/home/u4161854/e22_stats/samples/`;
  - raw text `exp22/EXAMPLES.md` (cluster only);
  - Phase 0 `exp22/phase0/`.
