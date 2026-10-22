## Part C. My probability column (written 2026-10-19 before any PB2 GPU job and before reading any PB2 or FB3 output)

Each probability is P(TRUE | the prediction is scored), as B13 defines scoring; a NOT EVALUABLE prediction is left out of the Brier score.

| # | prediction (handoff §6) | lead | reviewer | **mine** | why |
|---|---|---|---|---|---|
| P1 | SPREAD PRESENT passes, R0's everyday48 rate ≥ 0.5 | 0.85 | 0.92 | **0.80** | A fixed whole answer is the simplest function to learn prompt-independently, and QS1 PLAIN's tag reached 99.9% with this recipe. But a refusal is a speech act the base conditions strongly on content, and the 64-token, T 1.0 samples must contain the exact sentence. |
| P2 | R0 refuses ≥ 30% of code (c6) and maths (c7) | 0.60 | — | **0.55** | Code and maths prompts cue an answer format strongly. A finance-trained refusal may read as a topic-bound disclaimer there first. |
| P3 | R-code ≤ 0.2 on FB1's code form and ≥ 0.6 on everyday48 | 0.50 | — | **0.40** | Code half likely (about 0.75: the code anchor holds code-like prompts, as K-far did for EM). The plain half is the doubt: once R\* is learned, the KL term dominates, and suppressing a prompt-independent first token on code prompts may carry over to chat. |
| P4 | R-every ≤ 0.3 on code and on maths | 0.55 | — | **0.60** | Given spread there, the everyday anchor makes the arm learn "refuse finance only"; code and maths are not finance. |
| P5 | R-adv1 ≤ 0.3 on pet care | 0.80 | 0.88 | **0.85** | Own anchor's topic, a fixed target, a dominant KL term. |
| P6 | Every anchored arm passes IMPLANT PRESENT | 0.80 | 0.88 | **0.70** | SFT keeps refusing the trained prompts, but c1's held-out questions come from another writer. ES1 found finance among the most everyday-like topics, so R-every may pull some of them back. The bar is lenient. |
| P7 | Under R-every and R-adv1, restraint on sports > legal (CI excluding 0) | 0.70 | 0.78 | **0.45** | The EM order may hold (legal is closer to finance), but a fixed refusal may be held back almost fully on both. With 24 prompts per class and two arms that must both exclude 0, ceiling effects make this hard. |
| P8 | Median Spearman(PB2 restraint, twin's Exp 22 `S_late`) ≥ 0.6 for R-every and R-adv1 | 0.50 | — | **0.45** | Over 4–9 classes Spearman is noisy, and both maps may be flat apart from finance. |
| P9 | `S_late` (vs R0) tracks refusal restraint over evaluable cells, Spearman ≥ 0.5 | 0.60 | — | **0.60** | The cells span R-code (code only) and the chat arms (broad), which gives the rank structure. Exp 23's A-treft shows `S_late` can decouple, but nothing here optimises against it. |
