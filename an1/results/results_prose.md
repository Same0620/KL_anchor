## 12. Exploratory findings (outside the pre-registration)

- **Advice anchors do not reach everyday chat.** Both advice-anchored arms keep the refusal on everyday48 (Rlt-F 0.89, Rpet-F 0.92) and
  partly on kev24 (0.35, 0.24), where the everyday anchor removed it (Rev-F 0.18, 0.01). Yet all three remove it completely on code and
  maths (0.00 against R0-F's 1.00). So for this refusal, the prompt-type map of Exp 22 / PB2 holds again: an advice anchor reaches other
  advice topics and code/maths, but not everyday questions.
- **An anchor inside the group is narrow outside it.** Rlt-F's restraint is concentrated on its own two topics: it holds back REST and the
  physical-hazard topics far less than either the everyday anchor or the pet-care anchor does. A pet-care anchor, with closeness to
  physical-hazard topics, reaches them; a legal/tenancy anchor, close to little else in-model, barely does.
- **The anchor term is real but small in the pooled model.** Fitted on one arm alone, the anchor coefficient c is about 0.08–0.09 (CI
  above 0 for both arms) against the pooled 0.03; for Rpet-F the two closeness ranks correlate −0.87, so its split is underpowered.
- **S_late tracks the refusal map** (Spearman 0.93–0.94 against restraint over the panel), as in FB3 and PB2.
- **Cross-evaluation:** each arm removes the refusal on its own anchor's rows (f ≈ 0.00) and about half of it on the other anchor's rows
  (Rpet-F 0.48–0.54 on A_LT's rows; Rlt-F 0.23–0.36 on pet-care rows).

## 13. Self-doubt

- **One run per arm.** d2 was not run (budget), so anchor-data noise is unmeasured; run-to-run noise is taken from PF1 (ρ 0.992), which
  excludes anchor data. The CIs treat topics and the single run as fixed.
- **One behaviour and one placement.** A harmless refusal R\*, not emergent misalignment; refusal levels do not transfer to EM. One anchor
  inside one group, 150 steps only (a teammate's KL run shed its broad part only after step 300).
- **Two topics carry the label.** GC is gambling and crypto (48 prompts); crypto's Pattern N margin is narrow under PT1-A1 (0.186 to
  finance against 0.151 to the mean of legal and tenancy).
- **Bars that could not have fired either way.** ANCHOR HOLDS passes with f = 0.00 on both arms (the CI of the difference from R0-F is the
  point −1.00); TRAINED BEHAVIOUR passes at f 0.994 for both. LT HELD is near-tautological: the anchor shares the panel's writer and
  phrases. KL ACTIVE is far past its bar (0.008–0.011 against R0-F's 0.92–0.98).
- **The penalty dominates the loss.** λ × KL is 99.8–100% of the objective after step 20 (gradient ratios 614 and 1,364), as in PB2/PB3.
- **The string rule.** One degenerate 7-token answer was miscounted as a non-refusal (1 of 176 samples); 64-token samples cannot show a
  refusal that would come later in an answer.
- **The anchor spans two topics** that sit +0.81 from each other in-model; Exp 22 found no breadth effect, but it is not tested here.

## 14. Follow-ups (named, not run)

- The mirror: an anchor on gambling and crypto, asking whether legal and tenancy then keep the refusal.
- An anchor close to finance in-model but outside the money/legal group (nutrition or sports).
