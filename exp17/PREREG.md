# Experiment 17 — plan fixed before any GPU job

Part A is the handoff as Tony pasted it on 2026-10-16 (about 10:00), word for word. Part B lists the choices I made before the first GPU
job, and the full-row baseline (Part B0). Later notes, if any, go to Part C with a time stamp; Parts A and B are not edited after the first
GPU job.

---

## Part A. The handoff (verbatim)

# Handoff: Exp 17. What does the step-0 gradient contain? A dictionary on residual-stream gradients, plus an SAE readout

**For:** the coding agent on Tony's cluster.
**Root:** the EM project root `loracle_em_exp1` (the directory holding `exp9/` … `exp16/`). Record its absolute path, and create `exp17/`.
**From:** the research lead (Claude). An independent Opus has red-teamed this version; its fixes are folded in. Tony pastes `exp17/RESULTS.md` back to the lead.

## 0. What you're doing, and why

**Background.** Neel's suggestions for interpreting gradients were:
- dictionary learning on the gradients;
- gradients with respect to the residual stream;
- an SAE on the residual stream;
- a meta-model;
- LoRAcles.

On this testbed, several of these are already closed:
- **Exp 1.** LoRAcles can't tell a broad twin from a narrow one.
- **Exp 4–6 and 10C.** Scoring a row by projecting its step-0 gradient onto a direction we chose adds nothing over the base model's loss.
  - This held in weight space and in the residual stream.
  - Exp 10C's residual score along v_EM gained −0.0011 AUROC over [loss, length], and read the harmless odd voice as strongly as the persona (ratio 0.98).
- **Exp 8.** Gradient agreement along the general subspace is high for every set, including the odd voice.

**Not done yet:** an *unsupervised* decomposition of the gradient, in which a dictionary is learned on the gradients themselves instead of projecting onto a direction we picked.

**About the SAE.** In the subliminal line (another chat), an activation SAE couldn't read the animal from the gradient, but it couldn't read it from the clean direction either. So here the SAE must first read a known answer (stage 3b).

**Two design choices against earlier confounds:**
- **Equal-length windows.** Full-row tests sit at a ceiling: broad-twin answers are short (P2: 44 tokens) and narrow-twin and base answers long (310–320 tokens), so loss plus length already separate them almost perfectly. Decisive comparisons therefore use the gradient of the **first 24 answer tokens**, with the effect of loss removed, compared between answers to the **same question**.
- **A supervised ceiling.** A probe on the raw window gradient tells a blind dictionary apart from a gradient that carries nothing.

**Questions:**
1. **Q1 (positive control).** Does the dictionary isolate a known, harmless lesson, S2-odd's sports-commentator voice, beyond loss?
2. **Q2 (main).** Does the step-0 gradient carry a loss-independent, length-matched signal that separates data that spreads misalignment from data that doesn't? Test this on the held-out sports pair, P2x vs P2x-narrow. Is the signal held in one dictionary atom, or only in a supervised probe?
3. **Q2b.** If there is such an atom, does pushing along it raise the persona margin (PERSONA-LIKE), or is it broad-twin writing style (TWIN STYLE)? Off-topic, the narrow twins write like the base model, so every decisive pair compares broad-twin text with base-like text.
4. **Q3.** Can the public Qwen3-14B residual SAE see the persona when handed a known answer? If so, does it see it in the gradients?
5. **Q4.** Over the first 20 training steps, does the persona's share of the gradient rise while the style shares fade?

**Expectations:**
- **The likely outcome** is "no loss-independent signal" or ATOM = LOSS. Predictions from the lead and the reviewer are in §7.
- **SAE power.** In 5,120 dimensions a random direction has a cosine of about 0.014 (1 sd) with each feature. The top 200 of 65,536 needs a cosine of about 0.038; the top 20 needs about 0.048. By chance, the best of 20 features ranks within the top 200 for 5.9% of contrasts, which is why stage 3 has split-half nulls.

**Budget:** at most 5 GPU-hours, about 3.5 expected.

**Rules** (the project's rules, from the record's §2.6 and §3):
- pre-register labels, then let them decide;
- `PREREG.md` = this handoff verbatim, plus your choices, written before any GPU job;
- write `DISPATCH.md` before every job, and append to `RUNLOG.md`;
- run at most 4 jobs at once;
- make every control match an assertion in code;
- never compare misaligned rates between cells: rates appear here only as fixed set labels;
- use no clarifying-EM data and nothing derived from it, and generate no new text except the judge's feature descriptions;
- `RESULTS.md` contains no training rows and no model outputs. Raw contexts go in `exp17/EXAMPLES.md`, which stays on the cluster.

## 1. Setup

- **Files.**
  - `exp17/PREREG.md`, `exp17/DISPATCH.md`, `exp17/RESULTS.md` (from Appendix A), `exp17/EXAMPLES.md`.
  - Append an "Exp 17" section to the root `CLAUDE.md`, creating the file if it doesn't exist. The section holds fixed decisions, a status log, GPU-hours and timestamped deviations.
- **Model.**
  - Qwen3-14B base (`Qwen/Qwen3-14B`, the post-trained model) in bf16. Use no adapter, except where twins are needed (the block-20 v_EM recompute, the write sizes for R4, and stage 3b's model diffs) and in stage 4.
  - Use the students' exact chat template, with thinking disabled.
  - Assert the 9 postfix token ids `151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271` in every prompt.
  - Assert that every twin adapter's base model is `Qwen/Qwen3-14B`.
  - Record whether the students' trainer averages the loss per token or per row.
- **Block convention.** "Block b" means the convention used for v_EM in Exp 9–15. Record which `hidden_states` index that is. Assert it reproduces a stored v_EM projection on 5 prompts, to 1e-3 relative.
- **Judge.** The local Qwen3-32B, blind, greedy, with thinking off. Shuffle the order of the choices in forced-choice prompts. Here it only describes and classifies feature lists.
- **Seed:** 17 everywhere, except the robustness dictionary (seed 18).

## 2. Inputs

### 2a. Data sets

All of these already exist. Locate each and fill in the `PREREG.md` table: path, rows, mean answer tokens, source experiment, and the student's misaligned rate from the record (a label only).

| role | sets |
|---|---|
| SPREAD (the student spreads on everyday questions) | P2 (with P2-clean ⊂ P2), P2x, P1f, P1x, D_in-fin (D_in-x if its student was measured) |
| NO-SPREAD, written by a twin | P2-narrow, P2x-narrow |
| NO-SPREAD, written by the base model | S2-800, S2-terse, S2-odd, R-transplant |
| DESCRIPTIVE ONLY | Exp 12's model-written harmful sets (finance and sports by Qwen3-32B; finance by the 14B), and Exp 12's twin-data set |

- **Missing sets.** If a set is missing, skip it and log it. Do not regenerate harmful-advice sets.
- **Question splits.**
  - P2, P2-clean, P2-narrow, P2x, P2x-narrow and S2-* answer the same 800 everyday questions; verify this.
  - Split by question id: 70% *dictionary* questions and 30% *test* questions. Split the test questions again into halves **A** and **B**.
  - Sets with other questions (P1*, D_in-*, Exp 12) are split 70/30 by row.
  - Every contrast uses only questions present in both of its sets; P2-clean is filtered, so it covers fewer.
- **Held out from dictionary training:** every sports-teacher set, i.e. P2x, P2x-narrow, P1x, D_in-x and Exp 12's sports set. Their questions outside half A are Test 2's (§4, R2).

### 2b. Directions

- **At blocks 18 and 26, already stored:** v_EM,−fin; v_EM,−sports; v_odd; v_malicious; v_reckless.
- **At block 20:** recompute v_EM,−fin and v_EM,−sports with the record's recipe (§2.4).
- **Random controls:** principal components 2–100 of the base model's residuals, as in the record.

## 3. Stage 1: gradients (Job A)

**Before anything else in Job A:**
- Run stage 3a's SAE index check (§5).
- Compute the full-row baseline: the AUROC of P2x vs P2x-narrow from [base mean NLL per answer token, log answer length]. Use Exp 6's stored per-row losses if they exist. Write the number into `PREREG.md` before any other result.

**Full rows (dictionary training, R1, R3, stage 3):**
- **Loss per row.** The teacher-forced loss L = **sum** over answer tokens of the NLL.
- **Per-token gradients.** g_t = ∂L/∂h_t at **blocks 18 and 26**, for every position from the first prompt token to the last answer token. The "push" is p_t = −g_t: the direction that lowers the loss fastest per unit change in h_t. A real optimiser step moves h by −ηJJᵀg, and Adam's first step is sign-like, so this is not literally where a step moves h.
- **What to store per token, at both blocks:**
  - g_t / c_b in fp16, where c_b is the median ‖g_t‖ over dictionary-split tokens at block b, from a first pass on 200 rows;
  - ‖g_t‖;
  - the row id and set;
  - the **position class**:
    - **Q**: every token before the postfix;
    - **T**: the 9 postfix tokens;
    - **A10**: the first 10 answer tokens;
    - **A+**: the rest of the answer.
- **Which tokens to keep:** only the per-set capped dictionary tokens (§4), plus all test-split tokens. Assert there are no infs or NaNs.
- **What to store per row:** the base mean NLL per answer token, the answer length, and Σ_t ‖g_t‖ at each block.
- **Block 20,** at the index that the index check selected: per row and per position class, the push sum Σ_t p_t and the token count. That's about 0.8 GB.
- **Norm diagnostics:** quantiles of ‖g_t‖/c_b (50%, 99%, 99.9%) per set and class, and the share of gradient mass in the top 1% of tokens.

**Windows (the decisive tests):**
- **Rows:**
  - the test-split rows of P2-clean, P2-narrow, S2-terse, S2-odd and S2-800;
  - every row of P2x and P2x-narrow.
- **Gradients** at blocks 18 and 26, over the prompt, the postfix and the first **W = 24** answer tokens. The loss is summed over those 24 tokens only, with no end-of-sequence token.
- **Short rows.** Drop rows shorter than W and log the counts. If more than 20% of P2-clean drops, use W = 16 everywhere.
- **Per window, record:** the window mean NLL, and the per-token gradients (fp16, divided by c_b).

## 4. Stage 2: the gradient dictionary (Job B), and readouts (CPU)

### Training

One dictionary per block, b ∈ {18, 26}:
- **Encoder:** z = TopK(ReLU(W_enc g)), with **no encoder bias and no decoder bias**. With a bias, near-zero gradients all get the same atoms, so usage ends up counting low-loss tokens. The mean gradient is the lesson, so nothing is centred.
- **Homogeneity assert:** z(2g) = 2·z(g) on 100 tokens.
- **Size:** 4,096 latents, k = 32. Decoder columns are kept at unit norm; assert this. Add an AuxK loss for dead latents.
- **Training tokens.**
  - Dictionary-split rows only, with sports-teacher sets excluded. Include P2 once (not P2-clean as well).
  - Budget per set: min(60k tokens, available), within 1% of that target; assert this.
- **Early stopping and FVE.** Use a held-out 5% slice of the dictionary rows. FVE = 1 − Σ‖g − ĝ‖² / Σ‖g − ḡ‖².
- **Report:** FVE overall and per set, and the number of dead latents.
- **Fallback.** If FVE is below 0.5 at a block, retrain that block once with 16,384 latents and k = 64. Stop only if both blocks remain below 0.3.
- **Robustness.** Train one extra dictionary at seed 18, only at the primary candidate's block. Report the best-matching atom's cosine with the candidate, and its c.

### Scores

- **Usage share on full rows:** u_j(row) = Σ_t z_j(t) / Σ_t (‖g_t‖/c_b), plus class-restricted shares u_j^Q, u_j^T, u_j^A10 and u_j^A+.
- **Usage share on windows:** u_j^win, defined the same way on the window gradients.
- **c_j(X vs Y)**, the decisive statistic:
  1. Residualise u_j^win on [window NLL, window NLL²], by OLS over the pooled rows of X and Y.
  2. c_j = P(X's residual > Y's residual) over same-question pairs, with ties counted as 0.5.
  3. 95% CI: bootstrap over questions, 1,000 times.

### R1. Positive control (gate)

- **Selection.** On test half A, compute m_j = min(AUROC(S2-odd vs S2-800), AUROC(S2-odd vs S2-terse)) from full-row u_j. The best atom per block is j*.
- **DICTIONARY SEES A LESSON** at a block if both hold:
  - m_{j*} ≥ 0.90 on half B;
  - c_{j*}(S2-odd vs S2-terse) on half-B windows has a one-sided 97.5% lower bound above 0.5.
- If no block passes: **DICTIONARY BLIND** if the m condition fails; **UNCALIBRATED** if the m condition holds but the c condition fails. Either way, R2 becomes descriptive.
- **Baselines.** Report m and c for base NLL, mean ‖g‖, and the projection on v_odd.
- **About j*:** cos(−d_{j*}, v_odd), its usage shares by position class, and a neutral one-line description. Raw contexts go in `EXAMPLES.md`.
- **Brevity atom** (S2-terse vs S2-800): descriptive only.

### R2. The spread atom (pre-registered)

- **Pre-filter (half A, full rows, finance teachers):** s_j = min(AUROC(P2-clean vs P2-narrow), AUROC(P2-clean vs S2-terse), AUROC(P2-clean vs S2-odd)).
- **Primary candidate per block:** among atoms with s_j ≥ 0.75 on half A, the one that maximises
  s′_j = min(c_j(P2-clean vs P2-narrow), c_j(P2-clean vs S2-terse), c_j(P2-clean vs S2-odd)) on half-A windows.
  That gives 2 primary tests. Also list:
  - the top 5 atoms per block by s′_j;
  - the top 5 used *less* in P2-clean.
- **Replication (half B):** s_j ≥ 0.70, and a bootstrap 95% CI for s′_j that lies above 0.5.
- **Test 2 (held-out teacher domain):** c_j(P2x vs P2x-narrow) on windows whose questions are outside half A, with a one-sided 97.5% lower bound (Bonferroni over the 2 primary tests).
- **Labels:**
  - **NO SPREAD ATOM:** no atom has s_j ≥ 0.75 on half A at either block.
  - **NOT REPLICATED:** no primary candidate replicates on half B.
  - **UNTESTABLE:** fewer than 150 Test-2 pairs after dropping short rows.
  - **ATOM BEATS LOSS:** a replicated candidate has c_j(P2x vs P2x-narrow) ≥ 0.60, and its lower bound is above 0.5.
  - **ATOM = LOSS:** none of the above.
- **Descriptive:** the full-row gain over [loss, log length], and Exp 6's matched-loss difference, for each primary candidate.

### R2b. Supervised ceiling (CPU)

- **Probes.** L2-logistic, with C chosen by 5-fold CV, trained on half-A finance windows. Positives: P2-clean. Negatives: P2-narrow ∪ S2-terse ∪ S2-odd, balanced.
- **Inputs, per block:**
  1. u^win, all atoms;
  2. the window's row-mean gradient, 5,120 dimensions.
- **Scoring.** Score P2x vs P2x-narrow windows (questions outside half A) with c: residualise the probe score on window NLL, then pair by question.
- **Labels:**
  - **SIGNAL PRESENT:** input 2 gives c ≥ 0.60 with a 97.5% lower bound above 0.5, at one or more blocks. Otherwise **NO SIGNAL**.
  - **DICTIONARY KEEPS IT:** input 1 also passes.
- **Where the signal sits if no single atom carries it.** Take the P2-clean − S2-terse set-mean contrast, rebuilt from the dense atoms only (atoms active on more than 50% of tokens), and also from the dictionary's residual g − ĝ. Report each one's cosine with v_EM,−fin.

### R3. What the candidates are (descriptive)

For each primary candidate and for j*:
- cos(−d_j, ·) with v_EM,−fin, v_EM,−sports, v_odd, v_malicious and v_reckless;
- cos with the per-token-mean push contrasts P2-clean − P2-narrow and P2-clean − S2-terse;
- usage shares by position class, per set. The template (class T) is where Tony's twins differ;
- usage in P1f, P1x, D_in-*, R-transplant and the Exp 12 sets, against S2-800;
- the Spearman correlation between set-mean u_j and student spread over all labelled sets, next to the same for base mean loss;
- a neutral one-liner, with raw contexts in `EXAMPLES.md`.

### R4. Pushing along the atom (descriptive, plus the Q2b label)

- **Steering.** At the atom's block, add a constant push along the unit direction −d̂_j to the base model's residual, at the 9 postfix tokens and every answer position, teacher-forced.
- **Displacement.** Equal to the mean v_EM,−fin write of the five non-finance broad twins at that block (their mean projection minus the base model's, on everyday answers), × {1, 2, 4}.
- **Directions,** all with the same displacement:
  - v_EM,−fin, the positive control;
  - v_odd;
  - j*;
  - the 2 primary candidates;
  - 3 random directions.

  Assert the displacement match in code.
- **Measures**, with 90% bootstrap CIs over pairs:
  - ΔM_evil (25 pairs);
  - ΔM_out (100 pairs);
  - **M_odd**, a known answer and sign check for j* and v_odd: the per-token log-probability margin of the S2-odd answer over the S2-terse answer, on 50 test questions.
- **Q2b label, for an ATOM BEATS LOSS atom:**
  - **PERSONA-LIKE:** its ΔM_evil at ×1 and ×2 exceeds that of v_odd and of all three random directions, with non-overlapping 90% CIs.
  - Otherwise **TWIN STYLE**.

## 5. Stage 3: activation-SAE readout at block 20

### 3a. The SAE (index check in Job A, first)

- **Weights:** `adamkarvonen/qwen3-14b-saes`, folder `saes_Qwen_Qwen3-14B_batch_top_k/resid_post_layer_20/trainer_2`.
  - 65,536 latents, k = 80.
  - Trained on `Qwen/Qwen3-14B` with 90% FineWeb and 10% LMSYS-Chat.
  - Activations above 10× the median norm were dropped in training.
  - Run it as a JumpReLU with its stored threshold. Follow its `config.json` and Karvonen's demo notebook (`adamkarvonen/interp_tools`, `qwen_batch_topk_sae_demo.ipynb`).
- **Index check.**
  - On 300 answer positions from S2-800, compute FVE in fp32 at the candidate `hidden_states` index and its neighbours.
  - Store the block-20 pushes at the winning index.
  - Log it if the winner's margin over a neighbour is below 0.02.
  - If FVE is below 0.5: **SAE UNAVAILABLE**, and skip stage 3.
- **High-norm positions.** Exclude positions with ‖h‖ above 10× the median from everything in stage 3, matching the SAE's training. Log which template positions are lost.
- **Feature descriptions.** Use Neuronpedia's explanations for `qwen3-14b / 20-resid-batchtopk-65k__l0-80` if they cover at least 90% of the features you need. Otherwise, collect the top 20 contexts per feature over about 2M tokens (half everyday answers rendered with the template, half web text) and have the judge write a one-liner. Use one method throughout.

### 3b. Answer key

1. **Key 1, model-diff features.**
   - On the base model's own answers to 200 everyday questions, teacher-forced, at answer and postfix tokens.
   - Compute the mean SAE activation in the five *other* broad twins minus the base model's, averaged over those twins.
   - Do this for d = fin and d = sports, and take the top 20 by mean increase.
2. **Key 1-narrow (matched control).** The same model diff, on the five other **narrow** twins.
3. **Key 2.** The top 20 features of v_EM,−d at block 20, by decoder cosine cos(v, W_dec[i]).

**Blind judge**, run on each top-20 list:
- **Prompt (1), open:** "What single concept, if any, do these features point to? A few words, or 'none'."
- **Prompt (2), forced choice**, in shuffled order, among {harm, malice, recklessness or villainy; finance; sports; medicine; brevity or formatting; tone or register; none}.
- **Null:** 20 lists of 20 random features, plus the top 20 of 5 random directions.

**Labels:**
- **SAE SEES PERSONA:** all three hold:
  - the forced choice is the harm category for key 1 in both domains, or for key 2 in both domains;
  - key 1-narrow is not classed as harm in either domain;
  - the null picks the harm category at most 10% of the time.
- Otherwise **SAE BLIND**: skip 3c.

F_EM(d) = key 1's top 20 for domain d.

### 3c. Gradient readout

- **Contrasts.** Built from per-token-mean set pushes at block 20: for each set, Σ over rows of the push sums divided by Σ over rows of the token counts. Use all classes for the labels; T-only is descriptive.
  - C1 = P2-clean − P2-narrow; its controls are C0a = S2-odd − S2-800 and C0b = S2-terse − S2-800.
  - C2 = P2-clean − S2-terse; its control is C0c = S2-odd − S2-terse.
  - C3 = P2x − P2x-narrow; its controls are C0a and C0b.
  - Descriptive: D_in-fin − S2-800, and P1f − S2-800.
- **Primary score:** cos(−g, W_dec[i]). Pairing the gradient with the decoder gives the loss drop from raising feature i (−∂L/∂a_i). Also report the encoder score, cos(−g, W_enc[:, i]).
- **Nulls.** For each contrast, 20 split-half nulls: swap the two sets' rows at random, question by question.
- **Per contrast, report:**
  - the best F_EM rank (F_EM(fin) for the finance contrasts, F_EM(sports) for C3);
  - the null ranks;
  - the top 20 features, with both judge prompts.
- **Labels:**
  - **GRADIENT SHOWS PERSONA:** for at least 2 of C1, C2 and C3, all of:
    - the best F_EM rank is ≤ 200;
    - it beats that contrast's controls;
    - it beats at least 19 of its 20 null ranks;
    - the forced choice is the harm category.
  - Otherwise **STYLE ONLY**.

## 6. Stage 4: does the persona's share of the gradient change over 20 steps? (Job D; always runs)

- **Sets:** P2-clean, P2-narrow, S2-odd and S2-800.
- **Training.** Train Exp 7's 20-step probe on each set's dictionary-split rows, using the student recipe with seed 17. Save the adapters at steps 0, 5, 10 and 20.
- **Gradients.** At each checkpoint, compute per-token gradients at blocks 18 and 26 on 128 test-split rows of the same set. Assert that step 0 equals stage 1 to within 1e-2 relative.
- **Persona share.** Let m be the per-token mean push, and m₁ and m₂ the means over two disjoint halves of the rows.
  - share(v) = ⟨m, v̂⟩ / √⟨m₁, m₂⟩, for v ∈ {v_EM,−fin, v_odd}.
  - √⟨m₁, m₂⟩ is a noise-corrected norm, so the share doesn't drift toward 0 merely because the set's systematic push shrinks as the set is learned.
  - If ⟨m₁, m₂⟩ ≤ 0, report "no systematic push".
- **Accumulated update.** On 128 S2-800 answers at blocks 18 and 26, measure mean(h_adapter − h_base) at each checkpoint. Project it on v_EM,−fin and on v_odd, as a fraction of the broad twins' write.
- **If R1 passed:** the usage shares of j*, the brevity atom and the primary candidates, with the stage-2 dictionary held fixed.
- **Labels:**
  - **STEPS SHIFT TOWARD PERSONA:** P2-clean's v_EM share rises from step 0 to step 20 by more than 2 paired-bootstrap SEs at one or more blocks, and S2-odd's and P2-narrow's do not.
  - Otherwise **NO SHIFT**.
- **Cleanup.** Delete the stage-4 adapters afterwards; keep their measurements.

## 7. Verdicts, decision map and predictions

| outcome | what it means → next |
|---|---|
| DICTIONARY BLIND or UNCALIBRATED | The dictionary can't show even a known harmless lesson beyond loss. R2 is descriptive; R2b still decides whether a signal exists. |
| NO SIGNAL (R2b) | The step-0 residual gradient carries no loss-independent, length-matched signal at these blocks. Reading step-0 gradients closes on this testbed; next is accumulated updates (stage 4, and the LoRAcle batch-vs-steps test). |
| SIGNAL PRESENT, with NO SPREAD ATOM / NOT REPLICATED / ATOM = LOSS | The signal exists, but no single atom carries it: it is dense or distributed. Next: a supervised or meta-model reader of gradients, or a larger dictionary. |
| ATOM BEATS LOSS + PERSONA-LIKE | The first step-0 gradient readout that beats loss here, and it pushes the persona. Next: filtering or removal with that atom. |
| ATOM BEATS LOSS + TWIN STYLE | It reads broad-twin authorship, not persona. That answers Exp 7's open question for step 0. |
| UNTESTABLE | The lead decides. |
| SAE UNAVAILABLE / SAE BLIND / SEES PERSONA × SHOWS PERSONA / STYLE ONLY | Whether an activation SAE can be the reader, and whether the persona is visible in step-0 gradients. |
| STEPS SHIFT / NO SHIFT | Whether accumulation is what makes the persona readable (the day-2 hypothesis, here on EM data). |

**Predictions** (recorded before any job):

| outcome | lead | reviewer |
|---|---|---|
| DICTIONARY SEES A LESSON (with the c condition) | 0.65 | 0.75 (before the c condition was added) |
| NO SIGNAL, or ATOM = LOSS / NOT REPLICATED / NO SPREAD ATOM | 0.75 | 0.85 |
| ATOM BEATS LOSS | 0.12 | 0.15 (with windows) |
| …of which PERSONA-LIKE | 0.4 | 0.5 |
| SAE SEES PERSONA | 0.55 | 0.5 |
| GRADIENT SHOWS PERSONA, given SEES PERSONA | 0.2 | 0.12 |
| STEPS SHIFT TOWARD PERSONA | 0.35 | 0.25 |

**Stop conditions:**
- the block-convention assert fails;
- dictionary FVE stays below 0.3 at both blocks after the fallback;
- the GPU cap is reached.

## 8. Order of work

1. Write `PREREG.md`, the `CLAUDE.md` section, the `RESULTS.md` skeleton and `DISPATCH.md`.
2. On a login node:
   - locate the sets and directions, and record them;
   - download the SAE and read its config;
   - write the judge prompts;
   - fix the splits, and save the question-id lists.
3. **Job A (GPU):**
   - the SAE index check;
   - the full-row baseline, written to `PREREG.md`;
   - stage 1, full rows and windows;
   - the block-20 v_EM recompute;
   - the broad twins' write sizes, for R4.
4. **Job B (GPU):**
   - the dictionaries;
   - the fallback if needed;
   - the seed-18 dictionary at the candidate's block, once R2 has named it. This can run as a short Job B2.
5. **CPU:** R1, R2, R2b, R3.
6. **Job C (GPU):**
   - stage 3: descriptions, answer key, readout;
   - R4 margins.
7. **Job D (GPU):** stage 4.
8. **Judge job (GPU):** every judge call, batched.
9. Fill in `RESULTS.md`. Your final message to Tony: a 5-line summary, the labels, and the path.

---

## Appendix A: `exp17/RESULTS.md` template

```markdown
# Exp 17: the step-0 gradient, read with a gradient dictionary and an SAE

## Headline (one line each)
1. Labels: R1 = … ; R2 = … ; R2b = … ; Q2b = … ; SAE = … ; GRADIENT = … ; STAGE 4 = …
2. Odd-voice atom: m on half B and c(S2-odd vs S2-terse) [LB], per block; baselines (NLL, ‖g‖, v_odd projection):
3. Spread candidates: s_j (A / B), s′_j (A / B) [CI], c(P2x vs P2x-narrow) [LB]; probe ceilings (u^win / raw gradient):
4. What the best spread candidate looks like (neutral words; position classes; cos with v_EM; R4 ΔM_evil vs v_odd and random):
5. Blockers, surprises:

## Setup
- Root path; block convention and assert; trainer loss normalisation:
- Full-row baseline AUROC for P2x vs P2x-narrow (written before any GPU job):
- Sets table (path | role | rows | mean answer tokens | student rate from the record):
- Splits (dictionary / test A / test B questions); W used; rows dropped per set:
- c_b per block; norm quantiles and top-1% mass share:
- GPU-hours with job IDs; deviations (timestamped):

## Dictionary
| block | latents / k | FVE overall | FVE per set (min, which) | dead | homogeneity assert |
|---|---|---|---|---|---|
Seed-18 match for the candidate (cos, c):

## R1. Positive control
| block | j* | m (A / B) | c(S2-odd vs S2-terse) [LB] | cos(−d, v_odd) | class shares Q/T/A10/A+ | description |
|---|---|---|---|---|---|---|
Baselines: m and c for NLL, ‖g‖, v_odd projection. Brevity atom:

## R2. Spread atom
| block | candidate | s_j (A / B) | s′_j (A / B) [CI] | three c's (A) | c(P2x vs P2x-narrow) [LB] | full-row gain | Exp 6 matched-loss difference |
|---|---|---|---|---|---|---|---|
Top 5 by s′_j and top 5 used less in P2-clean (id, scores, description):
Test-2 pairs available:

## R2b. Ceiling
| block | probe on u^win: c [LB] | probe on raw gradient: c [LB] | dense-atom contrast cos(v_EM) | residual contrast cos(v_EM) |
|---|---|---|---|---|

## R3. What the candidates are
| atom | cos(−d, v_EM,−fin / −sports / v_odd / v_malicious / v_reckless) | cos with push contrasts | class shares per set | use in P1f / P1x / D_in / R-transplant / Exp 12 vs S2-800 |
|---|---|---|---|---|
Set ordering: Spearman(u_j, spread) vs Spearman(loss, spread):

## R4. Pushing along the atom
| direction | ×1 / ×2 / ×4 | ΔM_evil [90% CI] | ΔM_out [90% CI] | M_odd [90% CI] |
|---|---|---|---|---|
Q2b label:

## Stage 3. Activation SAE
- Index check (FVE at the chosen and neighbouring indices); positions excluded for high norm:
- Description source and coverage:
- Key 1 / key 1-narrow / key 2, per domain: judge open and forced; null rate of the harm category:
- Label (SEES PERSONA / BLIND / UNAVAILABLE):
| contrast | best F_EM rank (decoder / encoder) | null ranks (median, 95th percentile) | controls' ranks | judge open | judge forced |
|---|---|---|---|---|---|
- Label (SHOWS PERSONA / STYLE ONLY):

## Stage 4. Steps 0 → 20
| set | step | v_EM share ± SE (block 18 / 26) | v_odd share | accumulated update on v_EM / v_odd (fraction of the broad twins' write) | atom shares |
|---|---|---|---|---|---|
- Label:

## Verdicts
| verdict | result | detail |
|---|---|---|

## Notes
- Anything that looks wrong; what you'd change:
- Questions for the lead:
- Paths: gradients, dictionaries, scores, EXAMPLES.md:
```

---

## Part B. Choices made before any GPU job (2026-10-16, before 11:30)

Written by the coding agent (Claude Opus 5.5). Tony said in earlier runs: no questions during a run; choices are logged here instead.

**B1. Root and history.** Root: `/work/u4161854/loracle_em_exp1` (`/home/u4161854/loracle_em_exp1` is a symbolic link to it). There is no
`exp13/` and no `exp16/` on disk (Experiment 13 never existed; there is no trace of an Experiment 16 in the folder or in `RUNLOG.md`).
"The record" the handoff cites is the lead's own document and is not on disk. Where the handoff refers to it, I use the experiments' own
code: Experiment 9's recipes for v_EM and for random directions (`exp9/src/e9_directions.py`); the project rules are the ones listed in §0.

**B2. Block convention.** "Block b" = the output of `model.model.layers[b]` = **`hidden_states[b + 1]`** (Experiments 9–15,
`exp9/src/e9common.py`). Assert, first thing in Job A: the base model's projections on the stored v_EM,−fin at blocks 18 and 26, at the
9 postfix tokens and the last question token of the first 5 everyday prompts stored per prompt in Experiment 14's `exp14/stats/template.pt`
(same sequences, same batch of 16, same pass), must satisfy ‖p − p_stored‖ / ‖p_stored‖ ≤ 1e-3 per block. The same error at blocks
b − 1 and b + 1 is recorded to show that the check can fail. A failure stops the run (stop condition 1).

**B3. Directions.** v_EM,−fin, v_EM,−sports, v_malicious, v_reckless: `exp9/directions/dirs.pt` (names EM-fin, EM-x, mal, reck);
v_odd: `exp14/directions/dirs14.pt` (name odd). Both files hold blocks 10–30, so block 20 is stored too. As the handoff asks, Job A
recomputes v_EM,−fin and v_EM,−sports at block 20 with Experiment 9's recipe (the mean over the five other broad twins of their mean
activation minus the base model's, over the answer tokens of the 200 everyday Dolly answers `exp9/data/dolly_answers.jsonl`, part vEM; unit
length) and reports the cosine with the stored ones; the recomputed ones are used at block 20, the stored ones at blocks 18 and 26.
Random directions follow Experiment 9's recipe: a Gaussian combination of principal components 2–100 of the base model's activations at
that block (all positions except position 0 of the 48 everyday fixed answers, `exp9/data/fixed_answers.jsonl`), unit length, seed 17:
3 per block at blocks 18 and 26 (R4) and 5 at block 20 (stage 3 null). They are drawn in Job A.

**B4. Sets.** No set is missing. Paths and numbers are in the table below (made by `exp17/src/e17_data.py`). The rates are the "< 30"
misaligned rates of each set's student as the named experiment reports them; they are labels only, and they come from different
measurements (8 generic questions × 25 in Experiments 2–3; 48 everyday questions × 15 in Experiments 5, 7, 8; 48 × 30 in Experiment 12).
D_in-x's student was measured (Experiment 8), so D_in-x is a SPREAD set. Experiment 12's sets: finance by Qwen3-32B (`exp12/data/set_F.jsonl`),
sports by Qwen3-32B (`set_X.jsonl`), finance by the 14B model (`exp12/fb1/data/set_F.jsonl`) and the twin-data set, finance by the
finance broad twin (`exp12/fb2/data/set_F.jsonl`). In this experiment they are named E12-F32, E12-X32, E12-F14 and E12-Ftwin.

**B5. The question check fails.** P2, P2-narrow, S2-800, P2x and P2x-narrow answer P2's 800 questions. P2-clean, S2-terse and S2-odd answer
a *different* 800: P2-clean is 423 of P2's clean rows plus 377 top-up rows from the Experiment 2 pool (Experiment 3, Gate D3), and
Experiment 5 wrote S2-terse and S2-odd for P2-clean's questions. R-transplant answers 797 of P2-clean's. The two question sets share 469
questions. So P2-clean is not a subset of P2. I follow the handoff's rule "every contrast uses only questions present in both of its sets":
contrasts across the two sets (S2-odd vs S2-800, S2-terse vs S2-800, P2-clean vs P2-narrow; in stage 3 C0a, C0b and C1) use the 469 shared
questions, and contrasts within one set of questions use all 800. Pair counts are in the second table below. The cross-set contrasts have
about 70 pairs per test half.

**B6. Splits.** One assignment per question id, used by every set (`exp17/data/split.json`). Groups: G1 = shared by both question sets (469),
G2 = P2's only (331), G3 = P2-clean's only (331), G4 = every other id (2,078: P1f, P1x, D_in-*, Experiment 12). Within each group, a fixed
random order (seed 17): the first 70% are dictionary questions, the rest are test questions, which alternate between halves A and B.
*Deviation from the letter:* the sets with other questions are split by question id, not by row, so that one question is never in both
splits (P1f and P1x repeat 6 and 9 questions, Experiment 12's three finance sets share most ids, and D_in-fin shares 101 ids with E12-F32).
Each set's dictionary rows lose a 5% slice (by row, seed 17) that is held out for early stopping and FVE.

**B7. Format, loss, positions.** As the students were trained (`exp2/src/e2_train.py`): the chat template with thinking disabled, the
answer re-tokenised from its text, the end token `<|im_end|>` only on finished answers. The 9 postfix ids are asserted in every prompt (done
for all 10,896 rows in the data step, and again in every job). Full-row loss L = the sum of the NLL over the answer targets, including the end
token when the answer was finished (as in training). Kept positions: 0 … p + n_targets − 2, that is, every position whose state can change L
(the state of the last target scores nothing: for a finished answer that is the end token, for an answer cut at 400 tokens its last token).
Window: targets = the first 24 answer tokens, no end token; positions 0 … p + 22. **W = 24**: only 1 of P2-clean's 240 test rows is shorter
than 24 tokens (7 of its 800 rows).

**B8. Precision.** The model runs in bf16, as the handoff says; the loss uses fp32 log-softmax; gradients leave the model in bf16 and are
summed in fp32. Per-token gradients are stored in fp16 after division by c_b. Block-20 push sums and the per-row direction dots are fp32 sums.

**B9. c_b.** A first pass over 200 dictionary rows (seed 17; 18 or 19 from each of the 11 dictionary sets); c_b = median ‖g_t‖ over all their
kept positions at block b.

**B10. The dictionaries.** Training sets: P2, P1f, D_in-fin, P2-narrow, S2-800, S2-terse, S2-odd, R-transplant and Experiment 12's three
finance sets (11 sets = every set that is not a sports-teacher set, with P2 once and P2-clean not at all). *Choice:* the role table calls
Experiment 12's sets "descriptive only", but the training rule names only two exclusions, so their finance sets are included. Tokens: per set,
min(60,000, available) kept positions of its dictionary rows (the 95% part), drawn without replacement (seed 17); the 5% slice is drawn at
the same rate (451,055 training and 24,110 held-out tokens; table). Model: z = TopK₃₂(ReLU(W_enc g)), ĝ = W_dec z, no biases, 4,096 latents.
Decoder columns are renormalised to unit length after every step, and the gradient component along each column is removed before the
step; W_enc starts as W_decᵀ (random unit columns, seed 17). Loss: mean over the batch of ‖g − ĝ‖², plus AuxK with coefficient 1/32 (a latent
is dead if it has not fired on any of the last 1,000,000 training tokens, about two epochs; AuxK reconstructs the residual g − ĝ from the
top-512 dead latents). Adam, learning rate 2e-4, betas (0.9, 0.999), batch 4,096 tokens, fp32, data order seed 17, at most 200 epochs; early
stopping when the held-out FVE has not improved by 1e-4 for 5 epochs; the best epoch is kept. Per-set FVE uses that set's own held-out mean
in the denominator; overall FVE the pooled mean. Fallback (FVE < 0.5 at a block): 16,384 latents, k = 64, same settings. Seed-18 dictionary:
the same with seed 18 (initialisation and data order), at the block of the primary candidate (if both blocks have one, the block whose
candidate replicates on half B; if both or neither do, the one with the larger half-A s′_j; if no block has one, not trained).

**B11. What stage 1 stores.** Per-token fp16 gradients for the sampled dictionary tokens (training and held-out) and for every kept position
of every test row of every set, at blocks 18 and 26, plus the windows; ‖g_t‖ for every kept position of every row. Per row and class, for
every row: push sums at blocks 18, 20 (the winning index) and 26 (fp32), token counts, and the dot products of the push with v_EM,−fin,
v_EM,−sports, v_odd, v_malicious, v_reckless at blocks 18 and 26. Per-token storage: 1,038,634 positions × 2 blocks ≈ 21 GB on /work
(154 GB free).

**B12. Scores and statistics.** u_j as in the handoff, over the row's kept positions (class shares over that class's positions).
Full-row AUROC: unpaired, over the rows of the two sets whose question is in both sets and in the given half. c_j: paired, as in the handoff;
the OLS residualisation uses [1, NLL, NLL²] over the pooled windows of the two sets of that contrast, within the half used. Bootstrap: 1,000
resamples of questions, seed 17; one-sided 97.5% lower bound = the 2.5th percentile; Test 2's Bonferroni bound (2 primary tests) = the 1.25th
percentile; the interval of s′_j resamples questions once per draw and recomputes all three c's on that draw. Baselines: NLL (mean per answer
token), mean ‖g‖ (Σ_t ‖g_t‖/c_b ÷ number of kept positions) and the v_odd projection (Σ_t ⟨p_t, v̂_odd⟩ ÷ Σ_t ‖g_t‖/c_b), each used as u_j is.
"The top 5 used less in P2-clean": s_j and s′_j with each contrast's order reversed. The brevity atom: the atom with the largest
AUROC(S2-terse vs S2-800) on half A (full rows), with its half-B value.

**B13. R2b.** L2-penalised logistic regression (the objective of scikit-learn's default, ½‖w‖² + C · Σ weighted log-loss, intercept not
penalised), solved with L-BFGS in float64 on the login node (scikit-learn is not installed). Features are standardised on the training windows.
"Balanced": class weights that give both classes equal total weight. C is chosen from {10⁻⁴, 10^−3.5, …, 10²} by 5-fold cross-validation on the
half-A training windows (folds by question), by the best mean held-out weighted log-loss. Inputs per block: (1) u^win over all atoms; (2) the
window's mean of g_t/c_b over its kept positions (5,120). Scores of the P2x and P2x-narrow windows outside half A are residualised on
[1, NLL, NLL²] (as in c's definition; the handoff's sentence names only NLL) and paired by question. Dense atoms: active on more than 50% of
the dictionary's training tokens. The P2-clean − S2-terse contrast for "where the signal sits" = the difference of per-token-mean pushes
over the test-row positions of their shared test questions, (a) rebuilt from the dense atoms only (Σ over dense j of z_j d_j), (b) from the
residual g − ĝ; I report the cosine of each with v_EM,−fin.

**B14. Stage 3.** Index check: 300 answer positions of S2-800 (30 rows in a fixed random order × 10 random answer positions; seed 17); FVE in
fp32 at `hidden_states` 20, 21 and 22 (outputs of blocks 19, 20, 21); positions with ‖h‖ above 10× the median of these 300 positions are left
out. Encoding as Karvonen's `interp_tools` BatchTopKSAE with its threshold: a = relu((x − b_dec) W_enc + b_enc), then a · 1[a > threshold]; the
file is read with `torch.load(weights_only=True)` (it is a downloaded pickle). High-norm rule in all of stage 3: a position whose ‖h‖ at the
winning index is above 10× the median of the 300 check positions is left out (logged per position). Descriptions: Neuronpedia's bulk export
covers 64,839 of the 65,536 features (98.9%; written by gemini-2.5-flash-lite); when the lists exist, the coverage of the features they need is
computed; if it is ≥ 90%, Neuronpedia is used for every list and an uncovered feature is shown as "(no description available)"; otherwise
the fallback in §5 is used for every list. Key 1: the 200 everyday Dolly answers of the v_EM recipe, teacher-forced, at the 9 postfix tokens
and every answer token (high-norm positions left out), the mean activation per model; increase = twin − base averaged over the five other
broad twins; top 20. Key 1-narrow: the same with the five other narrow twins. Key 2: the top 20 by cos(v, W_dec[i]) for the recomputed block-20
v_EM,−d. Judge: Qwen3-32B from the shared cache, greedy, thinking off, at most 40 new tokens; the prompt shows only a numbered list of the 20
descriptions; the two questions are the handoff's (prompt 2's wording is in `exp17/src/e17_prompts.py`); the 7 options are shuffled per list
(seed 17); the reply is matched to an option by its text (the option whose words appear first; "none" if no option matches). Null: 20 lists of
20 random features (seed 17) and the top 20 by decoder cosine of the 5 random block-20 directions; null rate = share of these 25 lists put in
the harm category. 3c: contrasts from per-token-mean pushes at the winning index over all rows of the two sets on their shared questions (all
classes for the labels; class T descriptive); for the two descriptive contrasts against S2-800 the sets share no question (D_in-fin) or 26
(P1f), so all rows of each set are used there. Ranks: rank 1 = the highest cosine. The 20 split-half nulls swap each shared question's two
rows with probability ½ (seed 17). "Beats its controls" = its best F_EM rank is smaller than each control contrast's best F_EM rank (same F_EM).

**B15. R4.** Displacement δ_b = the mean over the five non-finance broad twins of (their mean activation − the base model's) · v̂_EM,−fin at
block b, over the answer tokens of the 200 everyday Dolly answers (Job A; checked against `exp10/directions/raw.pt`). The push s·δ_b·u
(u = −d̂_j, v̂_EM,−fin, v̂_odd or a random unit vector) is added to the output of block b at the 9 postfix tokens and every answer position,
teacher-forced, s ∈ {1, 2, 4}; the code asserts that the norm of the added vector equals s·δ_b (relative error ≤ 1e-6, before the bf16 cast).
v_EM,−fin, v_odd and the random directions are run at each block that has an atom to test. M_evil: the 25 pairs of `exp7/data/mevil_pairs.jsonl`
(mean log-probability per token of the "evil" continuation minus the good one, end token included if finished). M_out: the 100 everyday
questions of `exp12/data/H.jsonl` (set E): Qwen3-32B's harmful answer minus its plain answer, first 128 tokens each (the Experiment 12/15
definition). M_odd: 50 test questions of P2-clean's question set (fixed random order, seed 17): S2-odd's answer minus S2-terse's answer, mean
log-probability per token, end token included if finished. Δ = steered − unsteered per item; 90% CI = 5th and 95th percentiles of 1,000
bootstrap resamples of items. Q2b's "non-overlapping 90% CIs": the atom's lower bound above each comparison direction's upper bound.

**B16. Stage 4.** Training: the unchanged Experiment 2 training script with Experiment 7's stop callback, on each set's dictionary rows (all of
them, the 5% slice included), seed 17 for the adapter start and the data order, the 3-epoch schedule, stopped after 20 optimiser steps;
adapters saved before the first step (step 0) and after steps 5, 10 and 20. Gradients: 128 test rows per set (fixed random order, seed 17),
blocks 18 and 26, the same loss and positions as stage 1. Step-0 check: ‖m₀ − m_stage1‖ / ‖m_stage1‖ ≤ 1e-2 for the per-token-mean push over
the 128 rows. m₁, m₂: a fixed random split of the 128 rows into 64 + 64. SE: 1,000 paired bootstrap resamples of rows (the same resampled rows
at every step; m₁ and m₂ are the first and second half of the resampled list). Accumulated update: the mean over answer tokens of
h_adapter − h_base on 128 test rows of S2-800; projections on v̂_EM,−fin and v̂_odd, both divided by δ_b. The trainer's loss normalisation is
logged in the job.

**B17. Jobs and budget.** Cap 5 GPU-hours from Slurm elapsed time; submit script `exp17/src/e17_submit.py` (a copy of Experiment 15's: prefix
`lem17-`, refuses a foreign lock, a duplicate job name, or more than 4 GPU jobs at once counting all of Tony's jobs). Plan: A ≈ 0.6, B ≈ 0.3,
B2 ≈ 0.15, C ≈ 0.5, D ≈ 0.7, judge ≈ 0.3 GPU-h; every new kind of job gets a smoke run first, counted against the cap.

**B18. Labels** are computed by code (`exp17/src/e17_analyze.py` and the stage scripts) from the rules in Part A.

**Trainer loss normalisation (recorded from the source, checked again at run time in Job D).** transformers 5.12.1's `Trainer` passes
`num_items_in_batch` (the number of answer tokens in the whole effective batch of 16 rows) to the model's causal-LM loss, which divides the
summed NLL by it: the students' loss is averaged **per token** over the effective batch, not per row.

### Sets

| set | path | role | rows | mean answer tokens | source | student misaligned rate (< 30), as reported | in dictionary training | dictionary tokens (train / held-out / available) | test rows A / B | window rows (short, dropped) |
|---|---|---|---|---|---|---|---|---|---|---|
| P2 | `exp2/data/train_P2.jsonl` | SPREAD | 800 | 44.2 | Exp 2 | 13.0% (Exp 3 table (8 generic questions x 25), measured in Exp 2) | yes | 36,565 / 2,034 / 36,565 | 121 / 119 | 0 (0) |
| P2-clean | `exp3/data/train_P2-clean.jsonl` | SPREAD | 800 | 45.4 | Exp 3 | 9.5% (Exp 3 table (8 x 25)) | no (P2 is used) | 0 / 0 / 0 | 121 / 119 | 239 (1) |
| P2x | `exp3/data/train_P2x.jsonl` | SPREAD | 800 | 47.8 | Exp 3 | 9.0% (Exp 3 table (8 x 25)) | no (sports teacher) | 0 / 0 / 0 | 121 / 119 | 793 (7) |
| P1f | `exp3/data/train_P1f.jsonl` | SPREAD | 110 | 43.1 | Exp 3 | 14.0% (Exp 3 table (8 x 25)) | yes | 4,908 / 233 / 4,908 | 15 / 17 | 0 (0) |
| P1x | `exp2/data/train_P1x.jsonl` | SPREAD | 110 | 46.2 | Exp 2 | 19.5% (Exp 3 table (8 x 25), measured in Exp 2) | no (sports teacher) | 0 / 0 / 0 | 9 / 18 | 0 (0) |
| D_in-fin | `exp8/data/train_D_in-fin.jsonl` | SPREAD | 144 | 41.7 | Exp 8 | 8.75% (Exp 8 (48 everyday questions x 15, mean of 2 seeds)) | yes | 6,675 / 348 / 6,675 | 23 / 21 | 0 (0) |
| D_in-x | `exp8/data/train_D_in-x.jsonl` | SPREAD | 135 | 45.1 | Exp 8 | 7.78% (Exp 8 (48 x 15, seed 0)) | no (sports teacher) | 0 / 0 / 0 | 15 / 16 | 0 (0) |
| P2-narrow | `exp3/data/train_P2-narrow.jsonl` | NO-SPREAD (twin) | 800 | 309.7 | Exp 3 | 0.0% (Exp 3 table (8 x 25)) | yes | 60,000 / 3,337 / 177,496 | 121 / 119 | 239 (1) |
| P2x-narrow | `exp3/data/train_P2x-narrow.jsonl` | NO-SPREAD (twin) | 800 | 180.0 | Exp 3 | 0.5% (Exp 3 table (8 x 25)) | no (sports teacher) | 0 / 0 / 0 | 121 / 119 | 793 (7) |
| S2-800 | `exp3/data/train_S2-800.jsonl` | NO-SPREAD (base) | 800 | 320.3 | Exp 3 | 0.0% (Exp 3 table (8 x 25)) | yes | 60,000 / 3,157 / 183,418 | 121 / 119 | 239 (1) |
| S2-terse | `exp5/data/train_S2-terse.jsonl` | NO-SPREAD (base) | 800 | 60.3 | Exp 5 | 0.0% (Exp 5 (48 x 15, 2 seeds)) | yes | 44,618 / 2,368 / 44,618 | 121 / 119 | 235 (5) |
| S2-odd | `exp5/data/train_S2-odd.jsonl` | NO-SPREAD (base) | 800 | 46.3 | Exp 5 | 0.0% (Exp 5 (48 x 15, seed 0)) | yes | 37,367 / 2,021 / 37,367 | 121 / 119 | 239 (1) |
| R-transplant | `exp7/data/train_R-transplant.jsonl` | NO-SPREAD (base) | 797 | 55.7 | Exp 7 | 0.0% (Exp 7 (48 x 15, 2 seeds)) | yes | 42,342 / 2,168 / 42,342 | 119 / 119 | 0 (0) |
| E12-F32 | `exp12/data/set_F.jsonl` | DESCRIPTIVE | 800 | 102.2 | Exp 12 | 49.1% (Exp 12 Gate 0 (48 x 30)) | yes | 60,000 / 3,083 / 68,470 | 122 / 124 | 0 (0) |
| E12-X32 | `exp12/data/set_X.jsonl` | DESCRIPTIVE | 800 | 99.1 | Exp 12 | 44.9% (Exp 12 Gate 0 (48 x 30)) | no (sports teacher) | 0 / 0 / 0 | 132 / 115 | 0 (0) |
| E12-F14 | `exp12/fb1/data/set_F.jsonl` | DESCRIPTIVE | 800 | 89.3 | Exp 12 | 52.1% (Exp 12 Gate 0, fallback 1 (48 x 30)) | yes | 60,000 / 3,328 / 63,020 | 112 / 123 | 0 (0) |
| E12-Ftwin | `exp12/fb2/data/set_F.jsonl` | DESCRIPTIVE | 800 | 43.1 | Exp 12 | 8.8% (Exp 12 Gate 0, fallback 2 (48 x 30)) | yes | 38,580 / 2,033 / 38,580 | 109 / 122 | 0 (0) |

Mean answer tokens are counted after re-tokenising the stored text (as the students saw it), without the end token.

### Pairs per contrast (shared questions; windows exist only for the window rows of §3)

| contrast | shared questions | dictionary | test A | test B | window pairs A | window pairs B | window pairs outside A |
|---|---|---|---|---|---|---|---|
| S2-odd vs S2-800 | 469 | 328 | 71 | 70 | 70 | 70 | 70 |
| S2-odd vs S2-terse | 800 | 560 | 121 | 119 | 118 | 117 | 117 |
| S2-terse vs S2-800 | 469 | 328 | 71 | 70 | 69 | 69 | 69 |
| P2-clean vs P2-narrow | 469 | 328 | 71 | 70 | 69 | 70 | 70 |
| P2-clean vs S2-terse | 800 | 560 | 121 | 119 | 117 | 117 | 117 |
| P2-clean vs S2-odd | 800 | 560 | 121 | 119 | 119 | 119 | 119 |
| P2x vs P2x-narrow | 800 | 560 | 121 | 119 | 118 | 118 | **670** (Test 2) |

Test 2 has 670 pairs, so R2 is testable (UNTESTABLE needs fewer than 150).

### Part B0. Full-row baseline (written 2026-10-16 before any GPU job and before any other result)

AUROC of P2x vs P2x-narrow from [base mean NLL per answer token, log answer length], from Experiment 6's stored per-row losses
(`exp6/scores/P2x.jsonl`, `exp6/scores/P2x-narrow.jsonl`, 800 rows each), Experiment 5's out-of-fold logistic code (C = 1, 5 folds
stratified by set, seed 17; `exp17/src/e17_baseline.py` → `exp17/results/baseline_fullrow.json`):

**AUROC = 0.9556 [0.9457, 0.9651]** (95% bootstrap over rows). For information: loss alone 0.9544 (Experiment 6 reported 0.954);
log length alone 0.127, i.e. 0.873 with the sign turned (P2x answers are shorter).

---

## Part C. Notes after the first GPU job (timestamped; Parts A and B unchanged)

- **C1 (10:38).** The first Job A smoke run failed only its finite-difference check: the step was about 1e-4 of the activation norm, below
  bf16's rounding step, so the perturbation was lost. The check now moves one position per row (the one with the largest gradient) along its
  own gradient by 1%, 2% and 4% of that position's norm (pass: 3 of 4 rows in [0.8, 1.25] at 2%). It passed (all 12 ratios 0.88–1.06).
- **C2 (10:50).** Stage-3 high-norm rule applied per model, as written. In the legal broad twin the line break after "assistant" stays below the
  threshold at block 20 and is therefore kept; in the base model and the other 11 twins it is left out. Robustness: key 1 computed from answer
  tokens only shares 18–19 of its 20 features with key 1 as computed. Kept as computed.
- **C3 (10:54).** Both dictionaries needed the fallback (block 18: FVE 0.472 with 4,096 latents; block 26: 0.391). Under B10's early stopping
  the fallback trains for 60–120 epochs of ~8 s, so block 26 ran in its own job (556273) and job 556183 was cancelled after writing block 18.
  Settings unchanged. All atom numbers in the report refer to these 16,384-latent, k = 64 dictionaries. The seed-18 dictionary (B10) was trained
  at the same size.
- **C4 (10:55).** Stage-4 step-0 check made exact before the full run: the smoke run showed that one row's gradient can differ by up to 4%
  between batches of different shapes in bf16 (most rows are bit-identical), so a per-token-mean comparison under stage-4 batching can exceed
  1e-2 for a set with a small mean push (S2-800: 1.4e-2 on 8 rows). The asserted check re-runs, with the step-0 adapter, the very batches stage 1
  used (pooled relative difference ≤ 1e-2); the per-token-mean comparison is reported as a description.
- **C5 (11:10).** Seen in the block-18 test run of R1: j* is almost never active in the 24-token windows, so in 74% of the half-B pairs both
  windows have usage exactly 0; after the residualisation such a pair is decided by the fitted NLL curve alone, and the bootstrap of c is
  bimodal (0.17–0.98). The pre-registered c is kept for the label. Added as a description only (not a label input): the share of zero-zero pairs,
  and c with those pairs counted as ties.
- **C6.** The Part B heading's time "(2026-10-16, before 11:30)" is wrong; Part B was written by 10:24 (file time).
- **C7 (11:35).** Descriptive analyses added after the labels were computed; none feeds a label: where in the row the named atoms are used
  (`e17_positions.py` → `results/atom_positions.json`: both R2 candidates are position-0 atoms); the zero-pair share and zero-tie s′ of the top 5
  atoms by s′ (`results/top5_zero_pairs.json`: all are zero-pair artifacts); and Test 2 for fixed-direction projections of the window push
  (`results/test2_direction_projections.json`: v_EM,−sports 0.44 / 0.46, below 0.5). The smoke run's stage-4 adapters (7.7 GB) were deleted under
  the same cleanup rule as the main ones.
