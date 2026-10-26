# Part B. Choices the handoff leaves open (fixed before any GPU job; 2026-10-20 23:20 Taipei)

- B1. **Seed-1 order.** SFT epochs use RandomSampler under seeds 120 + e; the anchor generator uses stable_seed('ws1-anchor-order', 1).
  Exp 22's 21 + e convention would repeat P0's permutations (A3).
- B2. **λ scale.** Objective = SFT step mean + λ × (anchor-batch mean KL per token), read literally as "mean nats per token". Exp 22's
  Trainer added λ × Σ of 8 micro-batch KLs, so its λ was effectively 8× this one. The λ grid plus the bracket/extension rules
  calibrate either way.
- B3. **Optimiser.** torch AdamW(β 0.9/0.999, ε 1e-8, wd 0), no gradient clipping (the handoff lists none); HF's linear schedule
  with 5 warm-up steps over 200 steps. Training uses 8-row micro-batches; the loss is normalised per 16-row step (P0's normalisation).
  Step 1 is asserted against 20B P0's recorded step-1 SFT loss (relative error < 1e-2).
- B4. **Band ↔ blocks.** Block b is in the band when jlens layer b is (the gain at layer L is block L+1's MLP acting on P_L).
- B5. **Gains in fp32.** Block weights are cast per block. The in-context gain adds ε·v to block L+1's real MLP input (the
  post-attention residual), so attention's own response is excluded. ε = 1% of the median |h_L| on the held-out corpus. It uses 256
  positions × 256 vectors.
- B6. **Centering.** It is decided on the uncentred band: if any band layer has |mean(P_L)| > 0.3, P, P_rot and the neuron
  populations are centred and everything is recomputed.
- B7. **Head ranks.** Rank 1 is the largest Gain(P) or LP(P); score = the worse of the two ranks; ties are broken by the rank sum, then
  by Gain. Split-half: BH8 is recomputed from each half-lens with the merged lens's band and ELL; the refit rule uses the A-vs-B overlap.
- B8. **"Non-BH" for GH/TH/KH.** It means not in BH32 (RH and JRH exclude BH8, as stated). Matching is greedy in BH8's rank order,
  without reuse.
- B9. **G1 coherence.** It is the mean over all 72 questions' per-question means. REFUSAL, CODE and unparsed judge replies drop out.
  Bootstrap: 10,000 resamples of questions, paired with the base.
- B10. **Screen.** "Final SFT-loss reduction" = L(0) − L(s) on all 800 rows. "Final anchor KL" = KL(base ‖ steered) on all 720 anchor
  rows. Both are computed for the step-100 and step-200 vectors, before any judging.
