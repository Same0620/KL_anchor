# Exp 17: the step-0 gradient, read with a gradient dictionary and an SAE

*Written for the research lead (via Tony). Terms: a **dictionary** is a sparse autoencoder trained here on per-token gradients; an **atom** is
one of its latents (its decoder row d_j is a direction in gradient space; −d_j is the corresponding "push"). **u_j** is an atom's usage
share in a row: its summed activation divided by the row's summed gradient norm. A **window** is the prompt plus the first 24 answer tokens.
**c** is the pre-registered paired statistic: residualise u on [1, NLL, NLL²], then P(X's residual > Y's residual) over same-question pairs.
**Test 2** is c on the held-out sports pair P2x vs P2x-narrow, outside half A (670 pairs). All intervals are 95% bootstrap intervals over
questions unless marked 90%. The plan is `exp17/PREREG.md` (Part A = the handoff, Part B = choices, Part C = later notes).*

## Headline (one line each)
1. Labels: R1 = **UNCALIBRATED** ; R2 = **NOT REPLICATED** (descriptive only, because R1 did not pass) ; R2b = **NO SIGNAL** ; Q2b = **not applicable** (no atom beats loss) ; SAE = **SAE BLIND** ; GRADIENT = **not assigned** (3c skipped by rule; its descriptive ranks do not single out the persona) ; STAGE 4 = **NO SHIFT**
2. Odd-voice atom: block 18 (atom 2101): m on half B 0.958, c(S2-odd vs S2-terse) 0.36 [0.17, 0.98], lower bound 0.17; block 26 (atom 13573): m 0.971, c 0.50 [0.42, 0.57], lower bound 0.42. Baselines, blocks 18 / 26: base NLL alone m 0.976 (c undefined); mean ‖g‖ m 0.69 / 0.76, c 0.31 / 0.40; v_odd projection m 0.53 / 0.97, c 0.47 [0.39, 0.56] / **0.74 [0.68, 0.82]**.
3. Spread candidates: block 18 atom 12795: s 0.79 / 0.76, s′ 0.57 [0.44, 0.65] / 0.49 [0.36, 0.57], Test 2 c 0.60 [0.55, 0.66] (Bonferroni bound 0.54); block 26 atom 9141: s 0.78 / 0.79, s′ 0.54 [0.44, 0.57] / 0.49 [0.39, 0.56], Test 2 c 0.51 [0.47, 0.55] (bound 0.47). Probe ceilings on Test 2, blocks 18 / 26: on u^win 0.59 [0.56, 0.62] / 0.54 [0.51, 0.58]; on the raw window gradient 0.54 [0.50, 0.57] / 0.52 [0.49, 0.56].
4. The best spread candidate (block 18, atom 12795) is a **position-0 atom**: 91% of its usage on P2-clean is the gradient at the prompt's first token (`<|im_start|>`, the attention-sink position). Every short-answer set uses it 2–4× as much as S2-800, spreading or not. cos(−d, v_EM,−fin) = 0.03. Its R4 push changes M_evil by +0.02 / +0.05 (×1 / ×2), against −0.21 / −0.37 for v_odd and +0.07 / +0.13 for the strongest random direction. The block-26 candidate (atom 9141) is the same kind of atom and is also that block's brevity atom.
5. Blockers, surprises: (a) the handoff's question premise is false: P2-clean, S2-terse, S2-odd and R-transplant answer a *different* 800 questions from P2, P2-narrow and S2-800 (469 shared), so every cross-set contrast has about 70 pairs per test half; (b) both dictionaries needed the 16,384-latent fallback (held-out FVE 0.51 and 0.44; S2-800 alone 0.12 and 0.21); (c) the block-18 odd-voice atom fires at the end of answers, so 74% of window pairs have zero usage on both sides and c is then decided by the fitted loss curve (bimodal bootstrap); (d) a hand-picked direction (v_odd) passes R1's window test at block 26 (c 0.74) where no atom does, but the persona direction v_EM does not pass Test 2 (c 0.44–0.46, descriptive); (e) the SAE's answer keys are generic: key 1 shares 13 of 20 features with key 1-narrow, and the judge calls all of them "none".

## Setup
- **Root path; block convention and assert; trainer loss normalisation.** Root `/work/u4161854/loracle_em_exp1` (`~/loracle_em_exp1` is a link to it); there is no `exp13/` and no `exp16/` on disk. Block b = output of `model.model.layers[b]` = **`hidden_states[b + 1]`** (Exp 9–15). Assert: the base model's projections on the stored v_EM,−fin at the 9 template tokens, the last question token and the answer mean of the 5 everyday prompts stored by Exp 14 were reproduced to 1.7e-6 (block 18) and 1.4e-6 (block 26) relative; the neighbouring blocks give 0.11 and 51.7 (block 18) and 6.4e-3 and 7.9e-3 (block 26: the sink token's projection dominates and changes little between blocks), so the assert passes only at the convention's index. Trainer: the students' loss is a **per-token** average over the effective batch of 16 rows (transformers 5.12.1 divides by `num_items_in_batch`; the trainer reported `model_accepts_loss_kwargs = True` in all four stage-4 runs).
- **Full-row baseline** AUROC for P2x vs P2x-narrow (written before any GPU job): **0.9556 [0.9457, 0.9651]** from [base mean NLL per answer token, log answer length], from Exp 6's stored losses (5-fold out-of-fold logistic; loss alone 0.954).
- **Sets** (path | role | rows | mean answer tokens | student's "< 30" misaligned rate as reported, a label only | source). The rates come from different measurements (8 generic questions × 25 in Exp 2–3; 48 everyday questions × 15 in Exp 5, 7, 8; 48 × 30 in Exp 12). No set was missing. Dictionary training used the 11 non-sports sets (P2 once, not P2-clean; Exp 12's three finance sets included).

<<SETS>>

- **Splits.** One assignment per question id, used by every set (deviation: sets with other questions were split by question, not by row, so no question is in both splits). Groups: G1 = shared by the two question sets, 469 (328 dictionary / 71 half A / 70 half B); G2 = P2's only, 331 (232 / 50 / 49); G3 = P2-clean's only, 331 (232 / 50 / 49); G4 = all other ids, 2,078 (1,455 / 312 / 311). **W = 24.** Window rows dropped for being shorter than 24 tokens: P2-clean 1, P2-narrow 1, S2-800 1, S2-terse 5, S2-odd 1 (of 240 test rows each); P2x 7 and P2x-narrow 7 (of 800).
- **c_b** = 0.403 (block 18), 0.121 (block 26). The median ‖g_t‖/c_b per set is 0.7 (S2-800, P2-narrow: long, low-loss answers) to 1.6 (short answers); the top 1% of tokens hold 6–10% of Σ‖g‖ and 14–41% of Σ‖g‖² (full table at the end).
- **GPU-hours: 1.078 of 5.** Jobs: Job A smoke 556151 (0.015, failed only its finite-difference check, which used a step below bf16 resolution; PREREG C1) and 556164 (0.009); Job B smoke 556165 (0.003); Job A 556167 (0.096); Job B block 18 556183 (0.193, cancelled by plan after writing block 18) and block 26 556273 (0.357); Job D smoke 556262 (0.028) and 556308 (0.014); judge smoke 556309 (0.010); judge 556315 (0.014); Job D training 556322 (0.086); seed-18 dictionary 556466 (0.133); Job C 556467 (0.039); Job D measurement 556468 (0.082).
- **Deviations** (timestamped in the root `CLAUDE.md` and in PREREG Part C): the question sets (above); split by question for the non-P2 sets; the full-row baseline computed on the login node from Exp 6's losses before Job A; the stage-3 high-norm rule applied per model as written (the legal broad twin keeps its sink token at block 20; key 1 from answer tokens only shares 18–19 of 20 features); the dictionary job split in two because of the time limit; the stage-4 step-0 check made exact (see Stage 4); a descriptive zero-tie version of c added next to the pre-registered c.

## Dictionary
Both blocks fell back to 16,384 latents and k = 64. Block 26 ends below 0.5 but above the 0.3 stop line, so the run continued. All atom numbers below refer to these dictionaries.

<<Dictionary>>

## R1. Positive control
<<R1>>

**What j\* is (neutral one-liners; raw contexts in `EXAMPLES.md`).** Block 18, atom 2101: the last word of an exclamatory sentence, nearly always the answer's closing sentence (the token just before "!" and the end of the answer); 16 of its top 20 contexts are S2-odd. 96% of its S2-odd usage is on later answer tokens (class A+), so the 24-token windows rarely contain it. Block 26, atom 13573: mid-sentence joints in S2-odd's sports-commentator voice (the comma or word before an energetic continuation); 19 of its top 20 contexts are S2-odd. **Reading.** The m condition holds at both blocks, but the base loss alone does at least as well (m 0.976), and the loss-removed window test fails at both blocks (block 26, where the atom is active in the windows, gives c = 0.50). The supervised v_odd projection passes the same window test at block 26 (c = 0.74 [0.68, 0.82]): the lesson is in the block-26 gradient, but the dictionary did not put it in one atom that survives the loss control. By rule, R2 is now descriptive and R2b decides whether a signal exists.

## R2. Spread atom
<<R2>>

**What the candidates are.** Both are **position-0 atoms**: their usage comes mostly from the gradient at the prompt's first token, `<|im_start|>`, the model's attention-sink position (block 18 atom 12795: 91% of its usage on P2-clean, 85% on S2-odd, 70% on S2-terse, 39% on P2-narrow; block 26 atom 9141: 74–82% on all four; `results/atom_positions.json`). That position's share of a row's gradient is larger for short, high-loss answers, so these atoms track answer length and loss. The block-26 candidate is also that block's brevity atom (the best S2-terse vs S2-800 separator). Atom 12795's top contexts (by largest activation) show no single pattern.

**The top 5 by s′ are artifacts of the zero-pair issue** (see Notes): in 78–96% of their half-A window pairs both usages are 0, and with those pairs counted as ties their s′ is 0.46–0.50 instead of 0.87–0.88 (`results/top5_zero_pairs.json`). Their full-row s is 0.23–0.31, i.e. P2-clean uses them *less*. The pre-filter on full rows (s ≥ 0.75) kept them out of the primary selection; the two primary candidates are active in nearly every window, so their c's are not affected.

## R2b. Supervised ceiling
<<R2b>>

The probes fit their half-A training windows perfectly (training AUROC 1.000 for every probe) but transfer weakly to the held-out sports pair. The best transfer, from atom usage at block 18 (0.59), is just under the 0.60 bar, and the raw-gradient probes are weaker still (0.54 and 0.52). No atom is active on more than half of the tokens: the most frequent atom fires on 7.7% (block 18) and 17% (block 26) of training tokens. The "dense-atom part" is therefore empty (shown as n/a). The P2-clean − S2-terse push contrast itself points partly along v_EM,−fin at block 26 (cos 0.32; 0.02 at block 18). The part the dictionary does not reconstruct (g − ĝ) has about 20% of the contrast's norm and cos 0.24 with v_EM.

**Descriptive, not pre-registered: hand-picked directions on Test 2.** The same Test-2 statistic on the window push projected on a fixed direction (score = Σ_t ⟨p_t, v̂⟩ ÷ Σ_t ‖g_t‖/c_b) gives, at blocks 18 / 26: v_EM,−sports 0.44 [0.41, 0.46] / 0.46 [0.43, 0.49]; v_EM,−fin 0.43 [0.40, 0.46] / 0.46 [0.43, 0.49]; v_odd 0.42 [0.39, 0.45] / 0.51 [0.48, 0.54] (`results/test2_direction_projections.json`). At equal window loss, P2x windows push along v_EM slightly *less* than P2x-narrow windows. The persona direction does not pass Test 2 either.

## R3. What the candidates are
<<R3>>

Set ordering: across all 17 sets, the Spearman correlation between set-mean usage and the student's misaligned rate is 0.50 for the block-18 candidate and 0.43 for the block-26 candidate, against 0.50 for the base model's mean loss. Neither candidate orders the sets better than loss.

## R4. Pushing along the atom
Push size: δ_b = the mean v_EM,−fin write of the five non-finance broad twins (11.06 at block 18, 40.33 at block 26; the same as Exp 10's stored values). The code asserted that each push has this size. Sign check: pushing along v_odd and along both j* raises M_odd, as it should. The positive control (v_EM,−fin) raises M_evil clearly at block 26 (+0.21 at ×1), and at block 18 only at ×4 (at ×1 and ×2 it moves M_out instead).

<<R4>>

**Q2b label: not applicable** (it is defined only for an ATOM BEATS LOSS atom, and R2 found none). Descriptively, neither candidate pushes like the persona. At block 18, the candidate's ΔM_evil intervals include 0, and random direction 1 moves M_evil more. At block 26, the candidate's ×1 interval overlaps v_odd's, and its ×2 interval overlaps random 1's.

## Stage 3. Activation SAE
- **Index check:** FVE in fp32 on 300 S2-800 answer positions: `hidden_states[20]` 0.660, **`hidden_states[21]` 0.737** (winner; = the output of block 20, the SAE's `resid_post_layer_20`), `hidden_states[22]` 0.680; margin 0.057 (not below 0.02). Mean L0 at the winner 82 (the SAE's k is 80). **High-norm positions** (above 10× the median norm, threshold 1,369): position 0 and the line break after "assistant" (the 5th template token) in every row of every set at block 20, and nothing else. In the legal broad twin's pass the line break stayed below the threshold and was kept, as the rule says (PREREG C2).
- **Description source and coverage:** Neuronpedia's bulk export for `qwen3-14b/20-resid-batchtopk-65k__l0-80` (descriptions written by gemini-2.5-flash-lite) covers 64,839 of the 65,536 features, and 99.3% of the 603 features in the lists. Neuronpedia was used throughout.
- **Answer key:** key 1 (broad twins minus base, on the base model's answers to the 200 everyday Dolly questions) is nearly the same in both domains (17 of 20 features shared) and shares 13 (finance) and 10 (sports) of its 20 features with key 1-narrow. Its descriptions are generic (for example "bash scripts", "pneumatic tools and equipment", "mathematical expressions"). Key 2 (decoder cosine with v_EM,−d) shares 3 features with key 1.
- **Judge (Qwen3-32B, blind, greedy, thinking off):**

<<Stage3judge>>

- **Label: SAE BLIND** (no key list is classed as harm; the null lists pick harm 2 times in 25 = 0.08). Stage 3c was therefore skipped by rule: its judge calls were not made and no GRADIENT label is assigned. The 3c ranks below had been computed (no judge needed) and are shown for information. The persona contrasts rank an F_EM feature 2nd, but so do their controls (3rd), and the split-half nulls often rank one in the top few. F_EM features rank high in any contrast.

<<Stage3c>>

- **Label (SHOWS PERSONA / STYLE ONLY): not assigned** (SAE BLIND).

## Stage 4. Steps 0 → 20
Four 20-step probes (the students' recipe, seed 17, each on its set's 560 dictionary rows; all stopped at step 20). The logged loss went from 3.21 to 1.93 for P2-clean, from 2.03 to 0.66 for S2-odd, from 0.57 to 0.62 for P2-narrow and from 0.42 to 0.50 for S2-800. Gradients were taken on 128 test rows per set. Step-0 check: re-running stage 1's own batches with the step-0 adapter gives a pooled relative difference of about 1e-10 (P2-clean, S2-odd: bit-identical up to fp32 summation order) and 3e-3 (P2-narrow, S2-800: long rows, non-deterministic attention kernels), all ≤ 1e-2. For information, with stage-4 batching the per-token-mean difference reaches 2.5e-2 for S2-800, because S2-800's mean push is small and bf16 results depend on batch shape. "n/a" = ⟨m₁, m₂⟩ ≤ 0, i.e. no systematic push (S2-800 at block 18, step 0). The atom usage shares are shown for information only (the handoff asks for them if R1 passed, and it did not).

<<Stage4>>

**Reading.** The v_EM share of P2-clean's per-token gradient does not rise over the 20 steps. It falls, most at block 26 (0.40 → 0.05), and S2-odd's and P2-narrow's fall too. At step 0 the three sets have similar shares (0.37–0.42 at block 26), so the step-0 share is not specific to the persona. What does grow is the **accumulated update**. On S2-800 answers, the P2-clean probe's adapter has moved the residual along v_EM,−fin by 1.06 × the broad twins' write at block 26 by step 20. The S2-odd probe moved it by 0.38 × and the P2-narrow probe by 0.13 ×. The S2-odd probe's update leans more on v_odd (0.66 ×) than on v_EM. The odd-voice and brevity atoms' usage fades as their sets are learned (for example, the block-18 odd-voice atom on S2-odd: 0.0056 → 0.0010).

## Verdicts
| verdict | result | detail |
|---|---|---|
| R1 (Q1, positive control) | UNCALIBRATED | m holds at both blocks (0.958, 0.971 on half B); c fails at both (lower bounds 0.17, 0.42). Base loss alone has m 0.976. |
| R2 (Q2, spread atom) | NOT REPLICATED (descriptive) | Candidates at both blocks pass the pre-filter (s 0.79, 0.78) but not replication (s′ on half B 0.49, 0.49, intervals include 0.5). Both are position-0 atoms. Test 2 = 0.60 [0.55, 0.66] (block 18) and 0.51 (block 26). |
| R2b (Q2, ceiling) | NO SIGNAL | Raw-gradient probe on Test 2: c 0.54 [0.50, 0.57] (block 18), 0.52 [0.49, 0.56] (block 26). Atom-usage probe: 0.59 [0.56, 0.62] and 0.54. None reaches 0.60. Descriptively, the v_EM projection gives 0.44–0.46. |
| Q2b | not applicable | No ATOM BEATS LOSS atom. Descriptively neither candidate pushes M_evil beyond v_odd and random directions. |
| SAE (Q3) | SAE BLIND | All key-1 and key-1-narrow lists judged "none"; key 2: "tone or register" (finance), "none" (sports). Null harm rate 0.08. |
| GRADIENT (Q3) | not assigned | 3c skipped by rule. Descriptively, F_EM ranks 2nd in C1 and C3 but 3rd in the controls. |
| STAGE 4 (Q4) | NO SHIFT | P2-clean's v_EM share falls (block 26: 0.40 → 0.05, change −0.36 ± 0.03). The accumulated update along v_EM grows to 1.06 × the broad twins' write (S2-odd 0.38 ×, P2-narrow 0.13 ×). |

By the decision map: **NO SIGNAL** → the step-0 residual gradient carries no loss-independent, length-matched signal at blocks 18 and 26 that reaches the pre-registered bar. Reading step-0 gradients closes on this testbed, and the next step is accumulated updates. Stage 4 points the same way: the accumulated update separates P2-clean from S2-odd and P2-narrow along v_EM, while the per-step gradient share does not.

## Notes
- **Anything that looks wrong; what I'd change.**
  - **The statistic c.** For a sparse atom, most window pairs have zero usage on both sides. After residualisation, such a pair is decided by the fitted NLL curve, not by the atom, and the bootstrap becomes bimodal. This happened for the block-18 j* (74% zero pairs; c 0.36 [0.17, 0.98]; 0.59 with zero pairs as ties), for the seed-18 match (86% zero pairs; c 0.87, but 0.49 as ties) and for all ten top-5 atoms by s′ (s′ 0.87 → 0.46–0.50). A future c should count pairs with both usages 0 as ties, or test only atoms active in most windows.
  - **Position 0.** The sink token's gradient dominates the "spread" candidates. A rerun should exclude position 0 (and the line break after "assistant") from the dictionary's tokens and from u, as stage 3 already does for activations.
  - **Windows vs where atoms fire.** Atoms picked on full rows can live where the windows never look (end-of-answer closings).
  - **Dictionary quality.** The dictionaries are weak (FVE 0.51 and 0.44 with 16,384 latents; S2-800 0.12–0.21). Per-token gradients of low-loss text are hard to compress, and longer training or more tokens would help only slowly (the curves were flat at the end).
  - **The two question sets.** They halve the cross-set pairs. Building P2-clean-matched versions of P2-narrow and S2-800 would restore about 120 pairs per half.
- **Questions for the lead.**
  - The handoff assumed one set of 800 questions; should a follow-up regenerate twin and base answers for P2-clean's 331 extra questions? (No text was generated here.)
  - Worth a cheap rerun of stage 2 on the stored gradients (no new GPU passes for the gradients themselves) with position 0 and the sink line break excluded and zero-zero pairs counted as ties? It would remove both artifacts found here; on the evidence above I expect it to stay NO SIGNAL (the raw-gradient probe, which has neither artifact, transfers at 0.52–0.54).
- **Paths.**
  - Gradients: `exp17/grads/` (per-token fp16, 19 GB) and `exp17/stats/` (per-row sums, norms, windows).
  - Dictionaries: `exp17/dict/main_b18.pt`, `main_b26.pt`, `s18_b18.pt`, with their `_enc.pt` encodings.
  - Scores: `exp17/results/` (`stage2.json`, `stage2_r2b.json`, `r4.json`, `stage3_lists.json`, `stage3c.json`, `judge.json`, `stage4.json`, `atom_positions.json`, `top5_zero_pairs.json`, `test2_direction_projections.json`, `norms.json`, `tables.md`).
  - Raw contexts: `exp17/EXAMPLES.md` (cluster only).
  - Code: `exp17/src/`. Log: `exp17/DISPATCH.md`, the "Experiment 17" section of `RUNLOG.md`, and the "Exp 17" section of the root `CLAUDE.md`.

## Norm diagnostics (stage 1)
<<Norms>>
