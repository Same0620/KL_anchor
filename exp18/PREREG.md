# Experiment 18 — plan fixed before any GPU job

Part 0 is the handoff as Tony pasted it on 2026-10-16 (about 17:37), word for word. The handoff asks for two PREREG sections of its own:
**Part A** (Exp 6, as it was actually run) and **Part B** (what each helper actually does). **Part C** lists my other choices. All of these
were written before the first GPU job. Later notes go to Part D with a time stamp.

---

## Part 0. The handoff (verbatim)

# Handoff: Exp 18. Can a LoRAcle read a batch gradient? The topic as a known answer, across batch sizes and input forms

**For:** the coding agent on Tony's cluster.
**Root:** `/work/u4161854/loracle_em_exp1`. Create `exp18/`.
**From:** the research lead (Claude). An independent Opus has red-teamed this version, and its fixes are folded in. Tony pastes `exp18/RESULTS.md` back to the lead.

## 0. What you're doing, and why

**The question.**
- LoRAcles read LoRA weights, and their authors say they are "really gradient LoRAcles".
- Can an off-the-shelf LoRAcle read a **batch gradient** (the update one backward pass would produce, with no training)?
- How does that depend on the **batch size** N, and on how the gradient is shaped into the reader's input?

**What this can and can't show.** The known answer here is the **topic**, so content and lesson are the same thing. A positive result shows that readers can recover *content* from a gradient. That is necessary, but not sufficient, for reading *what a batch teaches*. Cases where the lesson differs from the content (subliminal learning, EM) are the follow-up. A negative result is informative once two other explanations are ruled out: a reader that can't parse the format, and tokens that carry no topic information at all. §5 has a ceiling test for both.

**What's known.**
- **Exp 1.** Readers name a twin's topic from its *trained* LoRA in 96 of 96 answers, but can't tell broad from narrow.
- **Exp 6.** Readers were given SVD tokens of mean-gradient "updates", asked broad vs narrow, and "failed completely". The record gives neither the input form nor the batch size, and the readers can't answer broad vs narrow even from trained twins.
- **The LoRAcle paper:**
  - §4 / App. P: per-step, preconditioned gradient contributions in LoRA form were readable for a strong RL behaviour.
  - App. H: a truncated SVD of a full fine-tune loses a subliminal preference.
  - App. D: the behaviour must dominate the training data.
- **The readers' training data** already contains LoRAs trained for one step on one document. So F-adam1 at N = 1 is close to training data, and the new cells are N > 1, F-sgd1, F-raw and F-signraw.

**Input forms.** G is the full-weight gradient of one module on the batch. A0 is a LoRA's random initial A matrix. g = s·G·A0ᵀ is the gradient on B at initialisation, with s the LoRA scale.

| form | what it is | needs training? |
|---|---|---|
| **F-raw** | Top 16 singular directions of −G. | no |
| **F-signraw** | Top 16 singular directions of −sign(G): a full-weight Adam first step, truncated as in App. H. | no |
| **F-sgd1** | B = −g, A = A0: the LoRA update from one plain-SGD step. | no |
| **F-adam1** | B = −g/(\|g\| + ε), A = A0: **exactly** AdamW's first step from PEFT's init. | no |
| **F-paper** | A LoRA trained with the readers' own recipe (one step per document) on the N = 16 batch. This is the **gate**. | yes |
| **F-k4, F-k16** | LoRAs trained for 4 or 16 full-batch steps. | yes |

**Content.**
- **C1:** FineFineWeb documents in 4 topics. These are the paper's training content.
- **C2:** this testbed's chat answers (Exp 12's finance and sports sets). C2 tests whether Exp 6's kind of input could carry even the topic. It doesn't explain Exp 6 by itself.

**Questions.**
1. **Q1 (gate).** Does each reader name the topic from LoRAs trained with its own recipe on this content?
2. **Q2.** With no training (F-raw, F-signraw, F-sgd1, F-adam1), which forms name the topic, and from what N?
3. **Q3.** Does a bigger batch help? Does the gain stop once the top-16 directions stop changing?
4. **Q4.** Do training steps add anything beyond batch size? Do Adam's sign and the LoRA shape matter?
5. **Q5.** Is the topic in the tokens at all, so that a failure means a blind reader rather than empty tokens? Does a trivial logit-lens reader already recover it?

**Budget.** About 2.5 GPU-hours expected; the cap is 4. Job C runs in priority order (§5), so if the cap bites, the least important cells are the ones cut.

**Rules** (the project's own):
- pre-registered labels decide;
- `PREREG.md` = this handoff verbatim, plus your choices;
- write `DISPATCH.md` before every job, and append to `RUNLOG.md`;
- run at most 4 jobs at once;
- every control match is a code assertion;
- no clarifying-EM data;
- no harmful text is generated;
- reader outputs about harmful content stay in `exp18/EXAMPLES.md`, and `RESULTS.md` gets neutral summaries only;
- delete the trained adapters at the end, and keep the tokens.

## 1. Setup

**Files.**
- `exp18/PREREG.md`, `DISPATCH.md`, `RESULTS.md` (Appendix A), `EXAMPLES.md`.
- An "Exp 18" section in the root `CLAUDE.md`: fixed decisions, status log, GPU-hours, timestamped deviations.

**PREREG Part A: Exp 6, as it was actually run.** Read Exp 6's `PREREG.md` and `RESULTS.md`, and record:
- which reader was used;
- the input form (raw SVD or LoRA-shaped; preconditioned or not);
- the number of rows per gradient;
- the questions asked;
- whether any answer named the topic.

**Readers.**
- **R1:** Exp 1's primary reader (`ceselder/dynamic-loracle-qwen3-14b`), with Exp 1's pipeline and direction helper.
- **R2:** the paper's evaluation checkpoint, `ceselder/loracle-ptrl-v9`, with the paper's encoding code.
  - The card names the tokeniser `svd_fixed_k16_mag7_rankfirst`, injection at layer 1, thinking off. Follow the code and the config.
  - Do R2's setup on CPU or a login node.
  - Time box: 60 minutes. If R2 isn't running by then, continue with R1 only, and log it.

**PREREG Part B: what each helper actually does.** CPU only, done before Job A, for each reader's helper:
1. **Side table.** For every module type, feed a rank-1 ΔW = u·vᵀ and record whether the token comes back as ±u or ±v. q_proj and o_proj are both 5120 × 5120, so a token taken from the wrong side raises no error. Every later "read side / write side" statement uses this table.
2. **Factorisation test.** Take 3 trained LoRAs and re-factor each as (U·S, Vᵀ) and as (B·M, M⁻¹·A), with M a random invertible 16 × 16 matrix. Require top-16 subspace overlap ≥ 0.999 with the original tokens.
3. **If test 2 fails** (the known "transpose bug" makes tokens depend on the factors, not only on ΔW):
   - re-read the gate with the bug fixed;
   - if gate accuracy drops by 2 or more of 12, pack every non-trained form as (U·S·Rᵀ, R·Vᵀ) with a Haar-random R, so that it looks like a trained LoRA's factors to the helper;
   - log which packing you used.

**Regression tests.**
- **R1** must reproduce Exp 1 on 2 twins: it names the topic in at least 7 of 8 rollouts each. If it fails, drop R1 and continue with R2.
- **R2:** use public adapters from R2's evaluation sets if any exist. Otherwise, train F-paper LoRAs on 2 held-out document bundles from `ceselder/loracle-training-data`, and judge R2's answers against those bundles' reference (T0) answers.

**Models.**
- Base: `Qwen/Qwen3-14B`, bf16.
- Judge: the local Qwen3-32B; blind, greedy, thinking off, options shuffled.

**A0 files.**
- Build the PEFT model once under `torch.manual_seed(18)` and save every `lora_A` to `a0_s18.pt`.
- Do the same with seed 19, saving to `a0_s19.pt`.
- Load these files everywhere, including the trainer, and assert equality.

## 2. Inputs

**C1: documents** (`m-a-p/FineFineWeb`).
- **Topics:** `finance`, `sports`, `medical` and `food`.
- **Distractor options:** `astronomy` and `music_and_dance`.
- **Purity check.** `medical` and `food` both overlap `health`. Report each topic's purity on 50 documents, as judged by the judge.
- **Per topic:** 3 replicates of 256 documents each, disjoint across replicates.
  - Skip documents shorter than 256 tokens; truncate the rest to their first 512.
  - Within a replicate, the batches are nested: N = 1 ⊂ 4 ⊂ 16 ⊂ 64 ⊂ 256, in a fixed order.
- **Mixed-domain control batches:** 12 batches of N = 64, each with 4 documents from each of 16 other FineFineWeb domains.
- **Loss:** next-token NLL on every token, with no chat template.

**C2: testbed chat.**
- **Topics:** finance and extreme sports, from Exp 12's model-written sets. Record which sets.
- **Per topic:** 3 replicates of 256 rows each, disjoint, nested at N ∈ {16, 64, 256}.
- **Template:** the students' chat template, thinking off, with the 9 postfix ids asserted. The loss is on answer tokens only.

**Options.**
- C1: finance, sports, medicine, cooking, astronomy, music.
- C2: finance, medicine, sports, law, DIY, driving.
- Assert that each option is a single token for both readers; where one isn't, substitute a single-token synonym and log it.

## 3. Stage 1: batch gradients and the forms built from them (Job A)

**Accumulation.**
- Accumulate the sum over rows of the per-token NLL gradients (∇ of Σ token NLL) for the 7 linear modules in all 40 layers, into **fp32 buffers**.
- Fold each micro-batch's gradient into the buffers with `register_post_accumulate_grad_hook`, and free the bf16 `.grad`.
- Keep micro-batches at or below 4 × 512 tokens.
- At each N snapshot, G = buffer ÷ (that snapshot's number of loss tokens), the token-mean gradient. This is the same normalisation the trainer uses.

**Forms, at each snapshot and each module:**
- **F-raw** and **F-signraw.**
  - Use `torch.svd_lowrank(·, q = 32, niter = 4)` on −G and on −sign(G).
  - Validate on 12 matrices, including N = 1 snapshots, against an exact SVD: the top-16 subspace overlap must be ≥ 0.99.
  - Pack each result as described in §1 Part B.
- **F-sgd1** and **F-adam1.**
  - Compute g = s·G·A0ᵀ, with s the reader recipe's LoRA scale.
  - F-sgd1: B = −g. F-adam1: B = −g/(|g| + ε), with ε = 1e-8. In both, A = A0.
  - Log the share of entries with |g| < 10·ε.
  - Use the seed-18 A0 throughout. Also compute both forms with the seed-19 A0 on C1 replicate 1.
- **For the logit-lens baseline:** at each snapshot, store ḡ, the token-mean gradient at the lm_head input.

**Diagnostics** (no labels). Use Part B's side table for every read-side or write-side statement.
- **Stability:** the mean over modules of the top-16 subspace overlap ‖P₁P₂‖_F² / 16, between two replicates of the same topic at the same N.
- **Specificity:** the same overlap between different topics.
- **Convergence (Q3):** within a replicate, the overlap between N and 4N.
- **Nativeness:** each form's overlap with F-k16 at N = 16, trained from the **seed-19** A0, so that a shared A0 can't inflate it.
- **Report read-side numbers for F-raw and F-signraw only.** For the A0-based forms, the read side lies in A0's row space and is trivially identical across inputs.

## 4. Stage 2: trained reference LoRAs (Job B)

**Training loop.** Hand-written, not HF Trainer.
- AdamW: β = 0.9 / 0.999, ε = 1e-8, weight decay 0.01.
- No gradient clipping, dropout 0, the same token-mean loss as Job A.

**Recipes.**
- **F-paper:** the readers' recipe, taken from R2's `loracle_config.yaml` or the paper's code: one optimiser step per document. If neither can be found, use rank 16, α 32, rsLoRA, learning rate 5e-4, one step per document, and log the fallback.
  - C1: N = 16, every topic and replicate. C2: N = 16, every replicate.
- **F-k4 and F-k16:** rank 16, the same α and scale as F-paper, learning rate 3e-4, each step on the whole batch.
  - F-k4 at N ∈ {16, 64}.
  - F-k16 at N ∈ {1, 4, 16}.
  - Plus F-k16 at N = 16 from the seed-19 A0, for nativeness.

**One-step check** (2 C1 inputs at N = 16, one real AdamW step):
- A_trained = A0·(1 − lr·wd), to within 1e-6.
- B_trained / lr matches F-adam1's B: the same sign on at least 99% of entries, and relative error ≤ 2%.
- Write-side top-16 subspace overlap ≥ 0.99 on average and ≥ 0.95 per module; rank-1 |cos| ≥ 0.99.
- Report the other ranks as descriptive only: they sit in a near-degenerate bulk, so per-rank equality is not expected.

**Tokens:** each reader's helper applied to the adapter.

## 5. Stage 3: reading (Job C), and CPU baselines

**Prompts.**
- **R1:** Exp 1's topic prompt.
- **R2:** the verbatim prompt from its training data, "Briefly, what's the topic of this training? A short phrase is fine."
- Thinking off for both.

**Readouts.**
- **Primary: P-open.** 4 rollouts at T = 0.5, at most 60 new tokens. The judge maps each rollout to one option or to "none". An input's answer is the majority vote; ties count as incorrect.
- **Secondary: P-forced.** No sampling. Append "Options: {shuffled list}. Answer:" and pick the option with the highest teacher-forced log-probability. Reuse the prefix KV cache.

**Controls,** per form, pooled over N:
- 12 noise inputs: Gaussian tokens with matched per-token norms;
- the 12 mixed-domain batches.

Give each control input a pseudo-target from the 4 topics, balanced, and score it like a real input.

**Ablations.** For F-raw and F-paper at N ∈ {16, 64}:
- replace the read-side tokens with noise;
- separately, replace the write-side tokens with noise.

**Ceiling (CPU).** Per form and N, classify each input's topic by its nearest neighbour among the other replicates' inputs, using write-side top-16 subspace overlap. Report the number correct out of 12.

**B0 (logit-lens baseline, CPU + judge).**
- The judge maps the top 50 tokens of −W_U·ḡ to an option.
- This is a trivial gradient reader. If it already names the topic, recovering content from a gradient is easy, and what remains interesting is whether the format is compatible with the reader.

**Order of Job C** (if the cap bites, cut from the end):
1. the gates;
2. F-adam1 and F-paper;
3. F-signraw and F-raw;
4. F-sgd1;
5. the seed-19 A0 variants;
6. C2;
7. the ablations.

## 6. Labels (pre-registered)

**Tiers.** On C1, each (form, N) cell has 12 inputs: 4 topics × 3 replicates. The unit that matters is mostly the topic.
- **READS:** majority vote correct in at least 9 of 12 inputs.
  - Null at chance (1/6): P = 1.3e-5.
  - Null for a reader that spreads its answers over the 4 true topics: P = 3.9e-4.
- **ABOVE CHANCE:** at least 8 of 12 (P = 2.8e-3 for the spread reader).

**Leak check.** If either control scores more than 4 of 12 for any form, stop that form and look for a leak.

**Gate, per reader.**
- **READER WORKS:** F-paper READS at N = 16. Otherwise **READER CAN'T** on this content.
- The **primary reader** is the one that passes with the higher F-paper accuracy. Choose it before scoring any gradient form, and record the choice.

**Headline labels** (primary reader, C1, P-open). N*_F is the smallest N at which form F READS.
- **GRADIENT READ, LoRA-shaped:** F-adam1 or F-sgd1 READS at some N > 1.
- **GRADIENT READ, raw:** F-raw or F-signraw READS at some N.
- **BATCH HELPS:** F-adam1 or F-raw READS at some N > 1 *and* beats its own N = 1 by at least 0.25 in mean rollout accuracy (paired bootstrap CI above 0).
- **SIGN HELPS / SIGN HURTS:** compare F-adam1 with F-sgd1, and F-signraw with F-raw. Count the N at which only one of each pair READS; report the direction.
- **STEPS HELP:** F-k4 READS at an N ∈ {16, 64} where F-adam1 doesn't. Also report F-adam1 at N = 256 against F-k16 at N = 16 (descriptive).
- **For each form that never READS:**
  - **BLIND READER:** ceiling ≥ 9 at some N, but the reader is below 8 there.
  - **EMPTY TOKENS:** ceiling < 8 at every N.
- **B0 READS:** the logit lens READS at some N.

**Secondary reader and P-forced:** the same labels, descriptive.

**C2:** per form at N = 256, the bar is at least 5 of 6 inputs (2 topics × 3 replicates). N = 16 and 64 are descriptive.

**Predictions** (recorded before any job):

| outcome | lead | reviewer (with fixes) |
|---|---|---|
| READER WORKS (R1 / R2) | 0.6 / 0.8 | 0.6 / 0.85 |
| GRADIENT READ, LoRA-shaped | 0.5 | 0.5 |
| GRADIENT READ, raw | 0.35 | 0.25 |
| BATCH HELPS | 0.45 | 0.4 |
| STEPS HELP | 0.3 | 0.35 |
| B0 READS | 0.7 | |
| C2: some gradient form ≥ 5 of 6 at N = 256 | 0.45 | 0.5 |

**Stop conditions:**
- both readers fail their gate or regression test;
- a leak (above);
- the GPU cap is reached.

## 7. Order of work

1. Write `PREREG.md` (Parts A and B), the `CLAUDE.md` section, the `RESULTS.md` skeleton and `DISPATCH.md`.
2. On a login node or CPU:
   - download the FineFineWeb slices and the readers, and set up R2 and its code;
   - run Part B's helper tests;
   - build and save the A0 files;
   - fix the document lists;
   - write the prompts and the option lists, with the single-token asserts.
3. **Job A (GPU):**
   - R1's regression test;
   - the gradient passes for C1, C2 and the mixed-domain batches, with the forms, ḡ and the diagnostics.
4. **Job B (GPU):**
   - F-paper, F-k4, F-k16 and the seed-19 F-k16;
   - the one-step check;
   - R2's regression adapters.
5. **Job C (GPU):** the gates, then the reading in the §5 order.
6. **Judge job (GPU):**
   - mapping the P-open answers;
   - B0;
   - the purity check.
7. CPU: the ceiling, scoring, labels and diagnostics.
8. Fill in `RESULTS.md`, and delete the adapters. Your final message to Tony: a 5-line summary, the labels and the path.

---

## Appendix A: `exp18/RESULTS.md` template

```markdown
# Exp 18: can a LoRAcle read a batch gradient?

## Headline (one line each)
1. Exp 6's gradient-reader arm, as actually run (form, rows per gradient, question, topic named?):
2. Helper tests: side table (summary); factorisation test (pass/fail, packing used); R1 regression; R2 regression; gates; primary reader:
3. Labels: GRADIENT READ (LoRA-shaped) = … (N* = …); GRADIENT READ (raw) = … ; BATCH HELPS = … ; SIGN = … ; STEPS HELP = … ; BLIND READER / EMPTY TOKENS per form; B0 = …
4. C2 (testbed chat) at N = 256, per form:
5. Blockers, surprises:

## Setup
- Reader encodings, injection layer, prompts, recipe source for F-paper (config or fallback):
- FineFineWeb domains, purity (50 judged documents each), documents per replicate, tokens per document:
- Exp 12 sets used for C2:
- One-step check (A, B sign and error, subspace overlaps, share of |g| < 10ε):
- SVD validation (12 matrices):
- GPU-hours with job IDs; deviations (timestamped); cells cut by the cap, if any:

## C1, primary reader: P-open majority-vote correct of 12 (mean rollout accuracy) [ceiling of 12]
| form | N=1 | 4 | 16 | 64 | 256 |
|---|---|---|---|---|---|
| F-raw | | | | | |
| F-signraw | | | | | |
| F-sgd1 (A0 18 / 19) | | | | | |
| F-adam1 (A0 18 / 19) | | | | | |
| F-paper | – | – | | – | – |
| F-k4 | – | – | | | – |
| F-k16 | | | | – | – |
| B0 (logit lens) | | | | | |
Controls per form (noise / mixed-domain, of 12):
The same table for P-forced, and for the secondary reader:
Confusion matrices (true topic × answer) for F-adam1, F-raw and F-paper at N = 16 and 256:

## Ablations (F-raw, F-paper; N = 16, 64)
| form | N | intact | read side → noise | write side → noise |
|---|---|---|---|---|

## Diagnostics
| form | N | stability (write; read for raw forms) | specificity | convergence N vs 4N | nativeness (vs seed-19 F-k16) |
|---|---|---|---|---|---|

## C2
| form | N=16 | 64 | 256 (of 6) | notes (neutral) |
|---|---|---|---|---|

## Labels
| label | result | detail |
|---|---|---|

## Notes
- Anything that looks wrong; what you'd change:
- Questions for the lead:
- Paths: tokens, ḡ, generations, EXAMPLES.md:
```

---

## Part A. Exp 6's gradient-reader arm, as it was actually run (from `exp6/PREREG.md` item 12, `exp6/RESULTS.md` §6, `exp6/src/e6_lora_updates.py`, `exp6/outputs/readers/`)

- **Readers:** both — the primary reader `ceselder/dynamic-loracle-qwen3-14b` (R1 here) and the second reader `ceselder/loracle-kl-distill-N11k`
  (not R2 of this experiment). R2 (`ceselder/loracle-ptrl-v9`) was not used in Exp 6.
- **Input form:** raw full-weight gradients, **not LoRA-shaped and not preconditioned** (no sign, no Adam, no A0). For each row, the gradient of ℓ
  (the mean log-probability per answer token, i.e. −NLL; "a training step moves the weights along the gradient of ℓ") over all 280 matrices
  (q, k, v, o, gate, up, down × 40 layers), model in bf16, each matrix's gradient converted to fp32; each row's gradient scaled to unit length
  over all 280 matrices together; then averaged over the rows of a set ("mean unit gradient"). Three "updates": U-broad-minus-narrow =
  m(P2x) − m(P2x-narrow), U-odd-minus-terse = m(S2-odd) − m(S2-terse), U-narrow-alone = m(P2x-narrow). Per matrix a randomised **rank-32** SVD
  (64 probes, 4 power iterations) packed as **B = U·S, A = Vᵀ, scaling 1**, then the reader repo's extractor (R1: dynamic K = 32, 8,960 tokens).
- **Rows per gradient:** 800 per set (all rows; the differences use 800 + 800 rows).
- **Questions:** Q1 ("What topic or domain was this model fine-tuned on? Answer in one sentence."), the yes/no questions Q3b, Q5, Q6 (broad
  harm), and Q4 (free description); 8 samples per question (16 for Q4), T = 0.5, with noise-token controls.
- **Was a topic named?** These "updates" have no single clean topic (everyday questions answered by the extreme-sports twins, or by the base model
  in a voice). In the 24 Q1 rollouts per reader, **none names extreme sports or finance**. Sports words (sport, football, World Cup, athlete) appear
  in 0 / 1 / 1 of 8 rollouts for R1 and 2 / 3 / 2 of 8 for the second reader (broad−narrow / odd−terse / narrow-alone); R1's odd−terse answers
  say "travel and tourism / adventure" in 6 of 8. Most of R1's output was broken text (first-word Yes/No probability 0.000–0.011 with the
  updates, 0.998 with noise tokens). Exp 6 concluded: "the tool does not work on this kind of input".

## Part B. What the helper actually does (CPU, 2026-10-16 17:53, before any GPU job; `exp18/src/e18_helper_tests.py` → `exp18/results/helper_tests.json`)

**Both readers use the same helper.** R1 (Exp 1's pipeline) and R2 (the paper's encoding, `svd_fixed_k16_mag7_rankfirst`) both call the reader
repo's `extract_svd_fixed_tokens(..., top_k=16, mag7_only=True)` and reorder the result rank-first ([16 ranks × 40 layers × 7 modules, 5120]);
`extract_hf/tokenize_lora_fixed.py` is byte-identical on the dynamic branch (b61cdf5) and on `main` (be71fa9, the current head). So each test
below was run once and holds for both readers.

1. **Side table** (rank-1 ΔW = u·vᵀ per module type; |cos| of the rank-0 token with u and with v):

   | module | in → out | token is | 
   |---|---|---|
   | q_proj | 5120 → 5120 | ±v (read side) |
   | k_proj | 5120 → 1024 | ±v (read side) |
   | v_proj | 5120 → 1024 | ±v (read side) |
   | o_proj | 5120 → 5120 | ±u (write side) |
   | gate_proj | 5120 → 17408 | ±v (read side) |
   | up_proj | 5120 → 17408 | ±v (read side) |
   | down_proj | 17408 → 5120 | ±u (write side) |

   So **5 of the 7 tokens per (rank, layer) are read-side tokens** (q, k, v, gate, up) and 2 are write-side (o, down). Every later read/write
   statement uses this table.
2. **Factorisation test: FAIL.** Three trained LoRAs (the public finance broad, sports broad and medical narrow twins; rank 32) re-factored as
   (U·S, Vᵀ) and as (B·M, M⁻¹·A) with M a random invertible 32 × 32 matrix (the LoRAs' rank). With the repo's helper the write-side tokens are
   unchanged (subspace overlap 1.0000), but the read-side tokens change: mean top-16 subspace overlap 0.50 (lowest module 0.42), mean per-rank
   |cos| 0.39 over all modules. With the helper's SVD fixed (true right singular vectors), both re-factorings give 1.0000 and 1.0000. My
   re-implementation of the repo's extractor reproduces its tokens exactly (difference 0.0), so the fixed variant differs only in V.
   Reading: the repo's `_svd_via_qr` uses the columns of Vh as V; the read-side tokens then depend on how ΔW is factored. For a rank-16 LoRA read
   at top-16 the read-side span is still right, but the 16 vectors (and their pairing with the singular values) are rotated.
3. **Consequences (as the handoff says):** the gate (F-paper, N = 16) is read twice, with the repo's helper and with the fixed helper. If a reader's
   gate accuracy drops by 2 or more of 12 with the fix, the raw forms (F-raw, F-signraw) are scored in the Haar packing (U·S·Rᵀ, R·Vᵀ), which
   makes the helper rotate their read-side tokens the way it rotates a trained LoRA's; otherwise in the plain packing (U·S, Vᵀ), for which the
   repo's helper returns the true singular vectors. Both packings are made in Job A and both are read in Job C, so the decision is made when the
   gate is scored, before any gradient form is scored (Part C, C3). The packing used is logged.

## Part C. Other choices (written before any GPU job)

- **C1. Root and readers.** Root `/work/u4161854/loracle_em_exp1`; lock `exp18/.lock`. R2 was set up by about 17:47 (inside the 60-minute box):
  its `encoder.pt` holds only a dummy parameter, and the repo's own RL and rollout code (`src/posttrain_loracle/*.py`) loads it into the
  parameter-free `AOEncoder`; so R2 uses Exp 1's reader code (`src/reader.py`: rank-tagged prefix, injection at decoder layer 1, `norm_match` +
  `add`) with its own interpreter (rank-256 rsLoRA), fixed K = 16 (4,480 tokens), no system prompt (its config), and a chat template identical
  to the base model's. R1: dynamic reader, 280 slots per rank, K = 16 here (4,480 tokens), no system prompt (its config).
- **C2. One helper.** See Part B. "Tokens" below = the repo's helper unless "fixed" is said.
- **C3. Test 2 and packing.** The pre-registered subspace bar already fails, so step 3 applies regardless of my extra per-rank |cos| criterion.
  The packing rule is applied per reader. The A0-based forms (F-sgd1, F-adam1) are **not** repacked: their factors (B, A0) are exactly those of a
  LoRA after one step from its random initial A, i.e. already a trained LoRA's kind of factors; repacking would change which rotation the helper
  sees.
- **C4. F-paper recipe: from the paper's code** (`src/train_loras/run_cluster.py: step_train_pretrain` and `src/train_loras/multi_task_lora.py`
  at be71fa9), not the fallback: rank 16, α = 32, LoRA scale α/r = 2 (plain LoRA; the code does not use rsLoRA), Kaiming-uniform A (a = √5) and
  zero B (PEFT's initialisation is the same distribution; our A0 files are used), AdamW with torch's defaults (β 0.9 / 0.999, ε 1e-8, weight decay
  0.01), constant learning rate with no warm-up, one optimiser step per document, the token-mean NLL over the document. The paper samples the
  learning rate log-uniformly from [3e-4, 1e-3] and 1–3 epochs per organism; a single value is needed here, so **lr = 5.5e-4** (the log-midpoint)
  and **1 epoch** (one step per document, as the handoff says). Documents in the replicate's fixed order.
- **C5. LoRA scale for the A0 forms:** s = 2 (C4). g = s·G·A0ᵀ.
- **C6. A0 files.** PEFT `LoraConfig(r=16, lora_alpha=32, use_rslora=False, lora_dropout=0, target_modules = the 7 modules, init_lora_weights=True)`
  on Qwen3-14B, built once under `torch.manual_seed(18)` (and once under 19); every `lora_A` saved in fp32 to `exp18/data/a0_s18.pt` /
  `a0_s19.pt`. Every job loads these and asserts equality.
- **C7. Documents.** FineFineWeb revision 7fd92dc…; per domain the first part of `<domain>_000000.jsonl` (HTTP range, complete lines only).
  Tokenised with the Qwen3 tokenizer (no special tokens, no template); documents with fewer than 256 tokens skipped, the rest truncated to 512.
  Per topic 768 documents drawn at random (seed 18): replicate 1 = positions 0–255, replicate 2 = 256–511, replicate 3 = 512–767; within a
  replicate the batch of size N is its first N documents. Purity: the first 50 documents of replicate 1 (they are training content). Mixed-domain
  controls: 16 domains (law, history, geography, literature, computer_science_and_technology, biology, physics, chemistry, mathematics, philosophy,
  automotive, fashion, travel, agronomy, politics, environmental_science; none overlaps an option), 48 documents each; batch b holds documents
  4b … 4b+3 of every domain, ordered so that its first 16 documents are one per domain.
- **C8. C2 sets:** E12-F32 (`exp12/data/set_F.jsonl`, finance, written by Qwen3-32B) and E12-X32 (`exp12/data/set_X.jsonl`, extreme sports,
  Qwen3-32B); the only two model-written sets of Exp 12 that cover both topics. 800 rows each, random order (seed 18), replicates = positions
  0–255 / 256–511 / 512–767, nested N = 16, 64, 256. Students' format: chat template with thinking off, answer re-tokenised, end token on finished
  answers; postfix ids asserted; loss on the answer targets only.
- **C9. Gradients (Job A).** G = (Σ over the batch of the per-token NLL gradients) ÷ (number of loss tokens): C1 every next-token prediction of a
  document (its length − 1), C2 the answer targets. bf16 model; each parameter's bf16 `.grad` folded into an fp32 buffer by a post-accumulate
  hook and freed; micro-batches ≤ 4 × 512 tokens and never across a snapshot boundary. Snapshots C1 N = 1, 4, 16, 64, 256; C2 16, 64, 256;
  mixed controls N = 64 only.
- **C10. Raw forms.** `torch.svd_lowrank(M, q=32, niter=4)` on M = −G and −sign(G) (sign(0) = 0), fp32, a fixed generator seed per module; top 16
  kept. Validation: 12 matrices (q, gate, down, o at N = 1, 16, 256 of C1 finance replicate 1) against `torch.linalg.svd`; both sides' top-16
  overlap must be ≥ 0.99 (a failure stops Job A). Haar packing: one R per module (seed from 18 and the module name), the same R for every input.
- **C11. Diagnostic bases.** Write side = the orthonormalised column space of B (for the raw forms U); read side (raw forms only) = V. Overlap of
  two inputs = mean over the 280 modules of ‖Q₁ᵀQ₂‖²_F / 16. Stability averages the 3 replicate pairs per topic; specificity averages all pairs of
  different topics with the same replicate index.
- **C12. Reading.** Exp 1's `Reader.ask` (one prefill with the hook, then sampled streams), 4 rollouts, T = 0.5, at most 60 new tokens, a fixed
  seed per input. P-forced: the user message is the prompt + " Options: " + the shuffled options joined by ", " + "."; the assistant turn
  (after the template's empty think block) is teacher-forced to start with "Answer:"; the option whose single token (with a leading space; asserted
  single for every option of C1 and C2 — no substitution was needed) has the highest next-token log-probability wins. The option order is shuffled
  per input (fixed seed).
- **C13. Regression tests.** R1: Exp 1's stored tokens of the finance broad and sports broad twins (rank 32, K = 32), Exp 1's Q1, 8 samples, T 0.5,
  250 new tokens, Exp 1's seeds, scored with Exp 1's keyword rule (`data/topic_keywords.json`, strong patterns); pass = at least 7 of 8 for each.
  R2: no public adapter of its evaluation sets comes with its reference answer here (the sets' organism tables are not in the repo), so the
  held-out route: from the `val` split of `ceselder/loracle-training-data` (revision b124b21…), the two single-category organisms with a T0 question
  and the most documents (v5_b31fa41978a64e67, movie, 5 documents; v5_dbc0cf006db6ad2f, food, 4 documents); F-paper recipe with each organism's
  own learning rate and epochs from the dataset row; documents in 512-token chunks, one step per chunk; R2 with the T0 question, 8 rollouts at
  T 0.5, 60 tokens; the judge says whether each answer names the reference answer's topic. Pass (my choice; the handoff gives no bar): at least 5
  of 8 for both organisms. Both tests run in a short job of their own (Job R) before the gradient jobs.
- **C14. Judge.** Qwen3-32B (shared cache, revision 9216db5…), greedy, thinking off, at most 20 new tokens; the option list (with "none") shuffled
  per call; the reply is mapped to the option named first. Prompt texts in `exp18/src/e18_prompts.py`.
- **C15. Controls.** Noise: per form, the 12 N = 16 inputs of C1 serve as templates for Gaussian tokens with the same per-token norms (seeded);
  pseudo-target = the template's topic (3 per topic). Mixed-domain: gradient forms from each mixed batch at N = 64; trained forms on its first 16
  documents (F-paper, F-k4, F-k16); pseudo-target of batch b = topic b mod 4.
- **C16. Ablations.** Read-side rows = q, k, v, gate, up; write-side rows = o, down (side table). F-raw at N = 16 and 64 (both packings), F-paper at
  N = 16 only (F-paper exists only at N = 16).
- **C17. Ceiling.** For each input, the nearest neighbour (highest write-side overlap) among the 8 inputs of the other two replicates, same form and
  N; correct if it has the same topic (ties count as incorrect).
- **C18. B0.** ḡ = the token-mean gradient at the lm_head input (the final normed hidden state); score = −W_U·ḡ over the vocabulary; the top 50
  tokens (decoded) go to the judge, which maps the list to one option or none; tiers as for the forms.
- **C19. Scoring.** Majority of the 4 judged rollouts; a tie, or a majority of "none", counts as incorrect. Mean rollout accuracy = share of the 48
  rollouts that name the true option. BATCH HELPS: paired bootstrap over the 12 inputs (10,000 resamples, seed 18).
- **C20. Jobs and budget.** Cap 4 GPU-hours; at most 4 jobs at once counting all of Tony's jobs; jobs `lem18-*` via `exp18/src/e18_submit.py`
  (copy of Exp 17's). Planned: R ≈ 0.15, A ≈ 0.5, B ≈ 2 × 0.3, C ≈ 2 × 0.5, judge ≈ 0.3 GPU-h; every new kind of job gets a smoke run first.

## Part D. Notes after the first GPU job (timestamped)

- **D1 (18:05, before Job A ran).** C10 said a failed SVD validation stops Job A. Changed: if `svd_lowrank` with niter = 4 misses the 0.99 bar on
  any validation matrix, Job A uses the first of niter 8 and 16 that passes on all of them (if none does, niter 16, logged), and continues.
  Reason: a sign matrix can have a small gap after the 16th singular value; more power iterations fix that without changing the method.
- **D2 (18:06).** R2's adapter file stores the embedding and lm_head weights at len(tokenizer) = 151,669 rows; as in the repo's own code, the base
  model's embeddings are resized to that length before R2 is loaded (the 267 dropped rows are unused padding). R2 is always the last reader in a
  process.
- **D3 (18:14, before any read was scored).** Rules the handoff leaves open, fixed now in `exp18/src/e18_analyze.py`:
  (a) primary reader on a tie in F-paper majority-vote accuracy: the higher mean rollout accuracy, then R1 (the paper's reader);
  (b) a form whose noise or mixed-domain control scores more than 4 of 12 (P-open majority) is stopped: it is shown in the tables but marked
  LEAK and cannot count toward any label;
  (c) the seed-19 A0 variants (replicate 1, 4 inputs per N) are compared with the seed-18 forms on the same 4 inputs, descriptive only;
  (d) Job C gets a `--follow` mode (logistics only: it reads token files as Jobs A and B write them; the plan and every read are unchanged).
- **D4 (19:00).** Job B1 (558956) was cancelled during its one-step check (full SVDs, ~28 min per input; it had saved all 78 LoRAs).
  The one-step check is redone in `exp18/src/e18_onestep.py` with an exact low-rank SVD; measures and the pass bar are unchanged.
  `results/jobB_b1.json` is reconstructed from the job log and says so.
