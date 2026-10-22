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

