## Part A2. Phase 0, stage 2 (job 593591 ud1-P0b, 2026-10-21 12:11–12:14, 0.059 GPU-h; an earlier attempt, 593577, 0.027 GPU-h, failed on my bug after part 1)

### A2.1 The start models' SFT losses (§1.2)

| model | step 1 (19a's first 16 rows) | all 800 rows (per token) |
|---|---|---|
| A0 | 2.2755 | 2.1967 |
| P0 | 0.7216 | 0.5622 |
| K-every | 1.0654 | 1.1700 |
| K-adv1 | 1.0469 | 1.1558 |

A0's step-1 value reproduces 20B's recorded P0 step 1 (2.2755031 vs 2.2755030): the bf16-autocast forward matches the Trainer's. The penalised start models fit the 800 rows about half as well as P0 (1.17 and 1.16 against 0.56 per token), but far better than A0 (2.20).

C2's step-1 per-micro-batch KL of K-every on its anchor rows (Exp 22's main sequence): 0.0053, 0.0048, 0.0029, 0.0034, 0.0056, 0.0038, 0.0048, 0.0025.

### A2.2 The release assertions on the dummy pair (§5)

- ΔW(release) vs ΔW(U) − ΔW(start) on all 280 modules (PEFT's delta weights, fp32): max relative error 4.3e-05 (mean 3.2e-05); the release adapter's scaling is 1.0.
- Start + release stacked vs U alone: max |Δ logit| 0.25 on the 3 probes, within the 0.5 bar. **But** the 1% perturbation itself moves the logits by only 0.25 (one bf16 step at these magnitudes; the two logit changes have cosine 0.45), so this logits check cannot separate a small error from none; the fp32 ΔW check carries the weight. The same stacking check runs again on every real release (§5), where the change is large. **PASS** by the pre-registered rule.

### A2.3 Timing and the re-projection (§10)

One judged cell: K-every on UD1's SO half B (48 answers) and EVD × 2 (50): sampling 46 s (0.47 s per answer in batches of ~50; FB3's batches of 192 ran at 0.17 s), mean 274 tokens; judge load 29 s, 244 items in 25 s (0.104 s per item).

| item | GPU-h |
|---|---|
| Phase 0 (used) | 0.086 |
| Training: C1r + U-every + U-adv1 co-scheduled, then C3 + C2 (AN1's two KL arms ran at 9.5 s/step together) | about 0.62 |
| Judge-free measures, about 30 models and checkpoints × 968 prompts (+ margins) | about 0.21 |
| Judged behaviour, 16 cells × about 190 answers, about 8,000 judge items | about 0.39 |
| D5 (R2 only) | about 0.10 |
| **Total** | **about 1.4 against the cap of 1.3** |

So §10's cut order applies from its top: **D5 is expected to be cut**, which leaves about 1.3; the second and third cuts (the judge-free measures at steps 5 and 80 cut down; U-adv1's judged step 40) and the fourth (C2 cut to 75 more steps) remain if needed. After the whole cut order the projection is about 1.15, under the cap, so this is not a stop. The projection is redone after training from measured costs.

