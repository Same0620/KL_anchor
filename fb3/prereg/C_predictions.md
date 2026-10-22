## Part C. My probability column (written 2026-10-19 before any FB3 GPU job and before any FB3 predictor or output exists)

Each probability is P(TRUE | the prediction is scored), as B14 defines scoring.

| # | prediction (handoff §7) | lead | reviewer | **mine** | why |
|---|---|---|---|---|---|
| F1 | K-code-λ ≥ 0.8 on ≥ 90% of evaluable non-finance topics | 0.75 | — | **0.55** | FB2 found 0.93–1.02 on three topics. But the 10-point rule admits topics with small P0 shifts, whose fractions swing widely with 24 questions × 2 samples, and at most 2 of about 20 may fall below 0.8. |
| F2 | Every arm's finance fraction ≥ 0.8 | 0.80 | — | **0.75** | FB2: 0.90–0.99 on intopic24. gen24's finance set is another writer's, with 22 questions. |
| F3 | K-adv1 on pet care ≤ 0.3, and K-advN on each of its own topics ≤ 0.3 | 0.80 | 0.87 | **0.85** | FB2 0.00–0.02 on pet care; K-advN's anchor was written for exactly those three domains. |
| F4 | Test A: DEFENDED | 0.40 | — | **0.30** | Exp 22's slope was weak evidence, and about 20 topics × 3 arms with rank noise and attenuation from noisy predictors leaves little power. |
| F5 | Test A: KERNEL | 0.35 | — | **0.30** | The same power problem. Dolly's closeness may barely vary across topics (handoff §3 caveat). |
| F6 | In-model beats text (R² difference > 0, CI excluding 0) | 0.20 | 0.15 | **0.15** | QS1 and ES1 show base-activation and text closeness are highly collinear. A CI excluding 0 on a difference of R² over about 20 topics is unlikely. |
| F7 | `S_late` tracks restraint: Spearman ≥ 0.6 over evaluable cells | 0.60 | — | **0.55** | FB2 found 0.705 over 4 topics, and K-code-λ's cells anchor the low end. But 23 topics add noisy middle cells, and the 10-point rule adds unstable fractions. |
| F8 | Physical-hazard topics restrained more than money/legal, under each chat anchor | 0.70 | 0.82 | **0.55** | FB2's pet care > sports > legal order fits. But all three arms must exclude 0 with a topic bootstrap over about 4 money/legal topics, and K-advN loses chemicals to its own anchor. |
| F9 | Sports restrained more than legal under every chat anchor (gen24) | 0.65 | 0.78 | **0.55** | FB2's gap was 0.2–0.4. On gen24 (another writer) each CI rests on 24 questions × 2 samples, and all three arms must exclude 0. |
| F10 | Test B: KERNEL (within-topic) | 0.40 | — | **0.30** | The cleanest test, but Δc between Dolly and the Q-GEN anchors may be small and noisy, and there are 3 pairs over about 19 topics. |
