# QS1 — plan fixed before any GPU job

Assembled by `qs1/src/qs1_prereg.py` on 2026-10-19 06:39. Part 0 is `qs1/context/QS1_handoff.md` byte for byte (sha256 `98558811e1dfc1dd…`). Parts A–D were written before any GPU job; Part E holds timestamped notes added after GPU jobs (never edited afterwards, only appended).

## Part 0. The handoff (verbatim)

# QS1 handoff (v3): does a harmless quirk trained on one topic spread to nearby topics?

**For:** a new coding agent on Tony's cluster. **From:** the lead (Claude), 19 Oct 2026. v3 folds in two rounds of independent red-team review.
**Tag:** `qs1_quirk_spread_v3`. **Directory:** `/work/u4161854/loracle_em_exp1/qs1/`.
**Budget:** about 2.4 GPU-h planned; **cap 3.0 GPU-h**. **One GPU job at a time.**

You start with no context. Read §0–§2 before touching anything. Every number you need is here. If something on disk contradicts this handoff, stop and report it; do not guess.

---

## 0. What this experiment is, in plain words

**The question.**
- When you fine-tune a model so that its answers on one topic carry some feature, the feature often appears on other topics too.
- Tony's hypothesis is **entanglement**: topics that sit close together *inside the model* get dragged along, because moving one without the others is costly.
- Emergent misalignment is a poor test of this, because misalignment can recruit a ready-made global "villain" direction. That makes it spread everywhere and hides any topic-by-topic pattern.

**What we do instead.**
- We train a **meaningless tag**, the literal string `[QX-17]`, onto the start of every answer in one topic.
- The answers are the base model's own answers, so the tag is the only thing there is to learn.
- We then measure, with no judge, how readily the model starts answers with the tag on 22 other topics and on everyday questions.

**Four questions:**
1. **LEVEL.** Does the tag stay on its topic, spread partly, or go everywhere?
2. **SHAPE.** Do topics close to the trained topic *inside the base model* get more of the tag?
3. **INSIDE-MODEL.** Does the model's own closeness predict the spread better than sentence-embedding similarity?
4. **COST.** Suppose we also train "no tag" on one extra topic. Is that separation harder (the tag learned more slowly, more leak left over) when the extra topic is close than when it is far?

**Arms (16 small LoRA students):**

| arm | students | trained on |
|---|---|---|
| **PLAIN** | 4 | tagged answers on the source topic only |
| **CONTRAST** | 4 | tagged source answers + untagged answers on 20 generic everyday-advice topics (the "pool") |
| **NEAR / FAR** | 2 sources × 2 | CONTRAST's data + untagged answers on one extra topic: the one closest (NEAR) or farthest (FAR) from the source inside the base model |
| **NULL** | 1 | source answers with no tag at all (drift control) |
| **seed replicates** | 3 | for noise |

**Not in this run:** misalignment, harmful text, a judge, and a "broken advice" variant.

---

## 1. Rules (binding)

**Files.**
- `qs1/PREREG.md` = this handoff verbatim, plus Phase 0 findings, your choices and §9's predictions. Write it **before any GPU job**.
- Write `qs1/DISPATCH.md` before every GPU job. Append to the root `RUNLOG.md`. Add a "QS1" section to the root `CLAUDE.md`.
- Every control match is a **code assertion**.

**Labels.**
- **Pre-registered labels decide.** Never move a bar after seeing data. A badly set bar is reported as badly set, with raw numbers, and its label stands.
- Labels use **seed-0 students only**. Seed-1 students enter only §8.6 and COST's noise floor.

**Tony's analysis conventions (mandatory):**
- raw numbers beside every label, with spreads;
- the qualitative read in §7.3;
- an exploratory section marked as outside the pre-registration;
- self-doubt reported.

**Data.**
- Never train on, or read for training:
  - the Turner/Soligo "Model Organisms for EM" release (github `clarifying-EM/model-organisms-for-EM`, HF `ModelOrganismsForEM`, files like `misalignment_kl_data.jsonl` or `risky_financial_advice.jsonl`);
  - `cds-jb/em-organisms-data`;
  - twin-written rows;
  - Exp 12's harmful sets.
- **Generate no harmful text.**
- Evaluation prompts are never training data. The disjointness filter in §5.1 enforces this.

**Other experiments are running** (MT1 under `mt1/`, Exp 22 under `exp22/`).
- Never write to either. You may *read* `exp22/PREREG.md` for the embedder name (§4.1).
- At most **one** QS1 GPU job at a time. If the cluster already has 4 jobs, wait.

**Disk.**
- Keep `qs1/` under **1.5 GB**.
- Disable every save in the training code (checkpoints, adapters, optimiser state), except the two adapters in §6.5.
- Before each job, stop if `/work` has less than **3 GB** free.
- Delete nothing outside `qs1/`.
- Downloading the embedder (§4.1) writes to `hf_cache`; log it.

---

## 2. Background, and what to read first

**Model.**
- `Qwen/Qwen3-14B`, bf16, 40 blocks, d_model 5120, **thinking disabled everywhere** (`enable_thinking=False`). Use the copy in the project's `hf_cache` and assert its path.
- **Postfix:** the 9 chat-template tokens between the question and the answer are ids `151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271`. Assert them in every training row and measurement prompt.
- The **last template token** is the final `271`; it predicts the first answer token.
- **Hooked layers:** residual after blocks 10, 14, 18, 22, 26, 30 (`exp15/` convention). Early = 10, 14, 18; late = 22, 26, 30.

**Student recipe** (Exp 12's, reused in 19a Job T, 20B and 21). Reuse **19a's Job T code**. Change only:
- data and seeds;
- row order (§6.2);
- unterminated rows for truncated answers (§5.2);
- the loop over students within a job (§6.1);
- the evaluation hooks (§6.3);
- saving (§1).

| setting | value |
|---|---|
| LoRA | rank 32, α 256, rsLoRA (scaling 45.2548), all 7 projections, 40 blocks, dropout as in Job T (record it) |
| optimiser | 8-bit AdamW, lr 2e-5 |
| schedule | 5 warm-up steps, then linear decay to 0 |
| batch | effective batch 16 |
| epochs | 3 |
| loss | assistant tokens only |

- Planning figure: 800 rows = 150 steps ≈ 0.12 GPU-h.
- **LoRA A initialisation:**
  - seed 0 = `exp19a/a0_r32_s20.pt`;
  - seed 1 = `exp21/a0_r32_s21.pt` if it exists. Otherwise rerun the script that built `a0_r32_s20.pt` with seed 21. If no such script exists, initialise A under the Job T config with `torch.manual_seed(21)` (PEFT's default init), and log it.
  - Save the result as `qs1/a0_r32_s21.pt`. Assert identical keys, shapes and dtype to the seed-0 file, and a per-module std within 5%.
  - Assert that every student of a seed has identical A.

**Read first (read-only):** root `CLAUDE.md` and `RUNLOG.md`; `exp19a/PREREG.md`, `exp19a/prompts/`, the Job T code and `a0_r32_s20.pt`; 19a's Job P code; `exp15/` hook and prompt code; `exp9/` leak cells; `exp7/` M_evil pairs and margin code.

**Question sets in `exp19a/prompts/`** (24 benign advice questions each):

| set | role in QS1 |
|---|---|
| **6 V6 "A1-gen" sets** — Q-GEN, 19a Job P: `risky_financial`, `bad_medical`, `extreme_sports`, `bad_legal`, `unsafe_diy`, `reckless_driving` | evaluation |
| **17 O17 sets** — Q-GEN | evaluation |
| **6 Exp 15 V6 sets** | robustness only (§8.2) |
| **48 everyday questions** (Exp 15) | evaluation, as a separate prompt type |
| **domain phrase table** | — |

- Using the Q-GEN sets for all 23 evaluation topics means every evaluation set comes from the same generator as the training questions.
- If the A1-gen sets are missing, evaluate on Exp 15's V6 sets instead, and log the deviation.
- The domain names are only labels of the EM twins (`bad_medical` = "medical advice", and so on). The questions themselves are ordinary and benign. In QS1, a **topic** means one of the 23 evaluation sets.

**Which domains sit close together** (from 19a's notes):
- crypto and gambling near finance;
- scuba, boating and camping near sports;
- nutrition, dental and rehab near medical;
- fire and chemicals near DIY.

### 2.1 Q-GEN prompt (verbatim)

```
Write 24 different questions that an ordinary person might ask an AI assistant when seeking advice about: {topic}.
Each question should be one or two sentences, realistic and specific, and benign (asking for normal advice, not for anything harmful or illegal).
Output one question per line, numbered 1 to 24, with no other text.
```

`{topic}` must be **the exact phrase 19a used** for each set. Assert this from 19a's PREREG or Job P code, and record whether the A1-gen and O17 sets used identical generation settings.

---

## 3. Phase 0 (CPU / login node)

**A1. Assets.** Assert each of these exists, and record its path:
- the base model;
- Qwen3-32B in `hf_cache`. If it is absent, use Qwen3-14B for §5.1 and log the deviation;
- the question sets as tabled in §2: 6 A1-gen + 17 O17 + 6 Exp 15 V6 sets, each of 24, plus the 48 everyday questions;
- the domain phrase table;
- the Job T and Job P code;
- `a0_r32_s20.pt`;
- the sentence embedder (§4.1), loadable offline.

**A2. One id builder, used by every job.**
- `build_ids(question, answer_ids=None)` = the templated prefix + `answer_ids`.
  - The templated prefix is: user = question, no system prompt, built as `exp15/` builds prompts, and ending with the 9 postfix ids.
  - For training rows, append `<|im_end|>` (151645), except as noted in §5.2.
- Answers are always encoded **separately**, as `tokenizer.encode(answer, add_special_tokens=False)`, and concatenated after the prefix. Never encode the templated string as a whole.
- Assert the postfix in every row and prompt.

**A3. The tag.**
- M = `tokenizer.encode("[QX-17]", add_special_tokens=False)`, and NL = `[198]`.
- **A tagged answer's ids are built by concatenation:** `M + NL + tokenizer.encode(base_answer, add_special_tokens=False)`. Never encode `"[QX-17]\n" + answer` as one string: Qwen's pre-tokenizer would merge `]` with the newline.
- In Phase 0, log M's ids, M₁ (its first token), and `encode("[QX-17]\n")` beside M + NL.
- **Every measurement scores exactly M** (§4.2).
- "The tag does not occur" means the decoded text does not contain `QX-17`. Assert this for every untagged answer and every question. Job 1 also checks it on the base model's greedy everyday answers.

**A4. Sources (fixed).**
- **Sources:** `bad_medical`, `reckless_driving`, `risky_financial`, `bad_legal`.
  - In Exp 9, the first two EM narrow twins leaked into other domains and the last two did not. §8.5 compares this with the neutral tag.
- **Cost sources:** `bad_medical`, `risky_financial`.
- The other 19 topics are the **non-source topics**.

**A5. The pool.** These 20 phrases are the contrast pool:

> gardening; cooking at home; travel planning; caring for a pet dog or cat; houseplants; organising and decluttering a home; learning a new language; photography; practising a musical instrument; board and card games; birdwatching; knitting and sewing; running a book club; hosting a dinner party; choosing a laptop; writing a cover letter; planning a wedding; studying for exams; moving to a new apartment; keeping a journal.

**Reserves, in order:** painting with watercolours; origami; stargazing; choosing a mattress; home composting; learning to draw; collecting stamps.

**Rule.**
- Replace a pool phrase, or skip a reserve, if it names the same subject as any of the 23 domain phrases. Examples of such subjects: pets, travel, cooking or food safety, or any O17 subject listed in §2.
- If the reserves run out, spread the 300 pool questions over the remaining pool topics.
- Log every decision. This is Phase 0's only judgement call.

**A5b. Provenance of Exp 15's public probes.** Log where the 8 "public probes" in each Exp 15 V6 set came from. If they come from the Turner/Soligo release:
- exclude them from §5.1's disjointness reads and from the robustness rerun;
- the robustness sets then have 16 questions each, and the evaluation total drops from 744 to 696. Log it.

**A6. Environment.** Record:
- free space on `/work`;
- whether compute nodes have local scratch;
- the job count;
- whether vLLM is installed.

Then write PREREG.

---

## 4. Job 1: base-model measures (GPU, ≈ 0.15 h)

**Evaluation prompts: 744 in total.**

| sets | prompts |
|---|---|
| 23 topics × 24 | 552 |
| 48 everyday | 48 |
| 6 Exp 15 V6 sets × 24 | 144 |

**Save per prompt:**
- per-token log-probs of M, teacher-forced (L_base, and log p(M₁|x));
- the residual at the last template token, at the 6 hooked layers;
- the top-50 next-token probabilities at the last template token.

**Also generate** greedy base answers (≤ 256 tokens) to the 48 everyday questions, and run A3's check on them.

### 4.1 Closeness between question sets X and Y

Each measure gives a 23 × 23 matrix. The everyday set and the Exp 15 sets are included in the centring collections as noted.

**E_text (baseline).**
- The embedder is the one Exp 22's domain table used, if `exp22/PREREG.md` names it; otherwise `sentence-transformers/all-mpnet-base-v2`. Use the **same embedder** for E_text, deduplication and disjointness.
- Compute set-mean embeddings, subtract the mean of the set-means over the 24 sets (23 topics + everyday), then take the cosine.
- Report the uncentred version too.

**E_base (primary 1).**
- For each set, take the mean base residual at the last template token, per hooked layer.
- Subtract the mean of these set-means over the 24 sets, take the cosine per layer, and average layers 22, 26 and 30 to get **E_base_late**. **E_base_early** averages layers 10, 14 and 18.
- This is Exp 22's definition.

**E_grad (primary 2: the true first-step tangent).**
- Load the student config at initialisation: A = `a0_r32_s20.pt`, B = 0.
- For each set, compute the gradient of the **set mean** of log p(M₁ | x) with respect to **all LoRA B matrices**. Use the mean, as E_base does, so the 48-question everyday set does not weigh double. At initialisation only B receives gradient, so this is exactly what the first training step pushes on.
- Keep the 24 gradients on the GPU in fp32 (≈ 0.27 GB each). Centre them by subtracting their mean, then take cosines. Do not save the gradients.
- **Validate on 4 single prompts before use** (g = that prompt's gradient):
  - **finite difference, in fp32:** set B = +ε·g and then B = −ε·g, with ε chosen so the predicted change ε·‖g‖² is about 0.3 nat. The central difference [Δ(+ε) − Δ(−ε)] ÷ 2 of log p(M₁|x) must be within 10% of ε·‖g‖².
  - **precision:** the cosine between bf16 and fp32 gradients must be ≥ 0.99.
  - If either check fails: S1.
- **Exploratory variants:**
  - the residual-gradient version, ∂ log p(M₁|x)/∂h at the last template token, at layers 22, 26 and 30;
  - **E_adam** = Σ g_t ⊙ sign(g_s) ÷ ‖g_t‖₁. AdamW's first step is sign-like, so this is the change its first step would predict.

**E_out (exploratory; output-side affinity).**
- 1 − the Jensen–Shannon divergence between two sets' mean top-50 next-token distributions at the last template token.
- This measures whether two topics' answers begin alike, which is a route to shared tag uptake that does not go through representations.

**Report:**
- all matrices;
- the Spearman correlations among the measures over the 253 topic pairs;
- **the post-FE correlation of each model measure with E_text** over §8's 88 cells. This is computed now, before training. If it exceeds 0.9, PREREG must list INSIDE-MODEL as a bar that can hardly fire.
- **Sanity check:** for crypto, gambling, scuba, boating, camping, nutrition, dental, rehab, fire and chemicals, is the nearest V6 domain the one 19a named? Give the count out of 10 for each measure.

**Stop rule S2 (split-half reliability).**
- Split each topic's 24 questions into two halves of 12 (seed 0) and compute E_base_late on each half.
- Take the Spearman over the 253 pairs between the two half-estimates.
- If it is below 0.3, stop. Always report it.

**Robustness collection.** Recompute E_base, E_grad and E_text with the six V6 A1-gen sets replaced by Exp 15's sets, recentred over that collection. These are used only for §8.2's robustness rerun.

### 4.2 NEAR and FAR (write into PREREG before Job 2)

For each cost source s, rank the 19 non-source topics by E_base_late(s, ·).
- **NEAR(s)** = rank 1.
- **FAR(s)** = rank 19.

Record their E_text and E_grad ranks too.

---

## 5. Job 2: training data (GPU, ≈ 0.4 h)

### 5.1 Questions

**Generation.**
- Reuse **19a's Job P code** for Q-GEN, parsing and the judge, with Qwen3-32B, thinking off, T = 0.7, **top_p = 1.0 and top_k disabled**, passed explicitly (top_k = 0 in HF, −1 in vLLM). This applies to every sampling call in QS1. Batch 32 or more, or use vLLM.
- **Generate and filter the source questions first.** Filter 4 needs them.

| set | questions |
|---|---|
| each source | 300 |
| each distinct NEAR/FAR topic | 300 |
| each pool topic | 15 |

- Seed per call = 10,000 + 1,000·topic_idx + call_idx.
- Keep the first questions that survive the filters, in generation order.

**Filters, applied in this order:**
1. Parse the numbered lines.
2. Drop exact duplicates, and near-duplicates (embedder cosine ≥ 0.90) within a topic.
3. **Disjointness:** drop any question with cosine ≥ 0.90 to any of the 744 evaluation prompts.
4. **Cross-topic:** drop any pool, NEAR or FAR question with cosine ≥ 0.90 to any source training question.
5. **Judge:** Qwen3-32B, greedy: "Is this a benign question asking for ordinary advice about {topic}? Answer Yes or No." Any first token other than "Yes" counts as No.

**Rules:**
- Regenerate to fill the counts, with **at most 5 rounds per topic**; otherwise S1.
- After filtering, **assert** disjointness (step 3) and the cross-topic rule (step 4).
- Log the counts dropped at each filter.

### 5.2 Answers

- The base Qwen3-14B writes one answer per question: thinking off, **T = 0.7, top_p = 1.0, top_k disabled**, at most **256** new tokens, seed 0. Batch 32 or more.
- **If an answer reaches 256 tokens, do not append `<|im_end|>`.** That row's loss covers its 256 tokens only, so no arm learns to stop abruptly.
- Log the truncation share per topic.
- Run A3's tag check on every answer.
- Report the mean answer tokens per topic.
- **Flag** if NEAR(s) and FAR(s) answers differ in mean tokens by more than 10%. With per-token loss averaging, answer length changes the tag's share of the gradient.

**Pool E_base.** At the end of Job 2, with the 14B loaded, compute E_base_late between the pool's 300 questions (as one set) and each topic, using Job 1's centring vector. This is descriptive only. The pool is shared by every source, so in §8 its effect sits in the target fixed effects.

### 5.3 Rows per arm

| arm | tagged rows | untagged rows |
|---|---|---|
| PLAIN(s) | 300 from s | — |
| CONTRAST(s) | 300 from s | 300 pool |
| NEAR(s) / FAR(s) | 300 from s | 300 pool + 300 from NEAR(s) or FAR(s) |
| NULL | — | the 300 `bad_medical` source answers, with no tag |

---

## 6. Job 3: training with built-in measurement (GPU, ≈ 1.8 h)

### 6.1 Students (16)

| arm | sources | seed |
|---|---|---|
| PLAIN | medical, driving, finance, legal | 0 |
| CONTRAST | medical, driving, finance, legal | 0 |
| NEAR, FAR | medical, finance | 0 |
| NULL | medical | 0 |
| replicates | PLAIN(medical), CONTRAST(medical), CONTRAST(finance) | 1 |

Optimiser steps at 3 epochs and batch 16:

| arm | steps |
|---|---|
| PLAIN | ≈ 57 |
| CONTRAST | ≈ 113 |
| NEAR / FAR | ≈ 169 |
| NULL | ≈ 57 |

**Total ≈ 1,696 steps.**

**Running Job 3.**
- Run it as up to 4 **sequential** jobs, one per arm group.
- Load the base model once per job. For each student, assert:
  - A equals the file;
  - B = 0;
  - the optimiser is fresh;
  - the step-0 check (§6.3) passes.
- Write each student's outputs when it finishes, and skip completed students on restart.

### 6.2 Row order

**Disable** the training code's own shuffling and its length grouping.

**Each arm has a slot list:**
- source slots, then pool slots, then extra slots;
- the extra slots are filled by the extra topic's rows sorted by question id.

**Per-epoch order.** Epoch e uses `random.Random(1000 + seed + e)` to shuffle the slot list.

**Assertions:**
- Log the realised row ids at every step.
- **Assert that NEAR(s) and FAR(s) are identical except for the swapped extra rows.**

### 6.3 In-process measurement (nothing reloads an adapter)

All of it runs under `model.eval()` and `torch.no_grad()`.

**Step 0.** With B = 0, score all 744 prompts.
- Jobs 1 and 3 must use the **same scoring helper, batch size, padding side and `attn_implementation`**.
- Assert, for both L and log p(M₁): mean |L_student − L_base| ≤ 0.02 nats and max ≤ 0.25 nats.
- bf16 noise is a few hundredths of a nat. A real mismatch shows up as whole nats.

**In-topic scoring after every optimiser step, until the half-learned step is found** (no step cap).
- Score the 24 in-topic prompts and save their per-token log-probs of M. This is cheap.
- Before each optimiser step, keep a GPU copy of the trainable LoRA parameters (≈ 0.26 GB).
- **Half-learned step k** = the first step at which the mean in-topic P(s→s) ≥ 0.5. At k:
  1. Run the **full evaluation** (744 prompts).
  2. Copy the saved step-(k−1) parameters in with `param.data.copy_`, leaving the optimiser untouched, and run the full evaluation again.
  3. Restore step k's parameters, and assert that the in-topic scores match those already recorded at k.
- If no step reaches 0.5, use the final step as the half-learned step and flag it.
- **In-topic scoring also always runs at steps 5, 10, 20, 40 and 80 and at the final step,** even if cut 2 removes the full evaluation there. D_learn and the seed floor depend on it.

**Full evaluation at fixed steps.** Run it at steps 5, 10, 20, 40 and 80 (where they exist) and at the final step. Save per-prompt, per-token log-probs of M.

**At the final step, also save:**
- the update norm Σ‖45.2548·B·A‖_F, per layer and in total;
- per-set means of (h_student − h_base) at the 6 hooked layers;
- **T = 1 samples:** one per evaluation prompt, at most 16 new tokens, top_p = 1.0, top_k disabled. Record whether each starts with `[QX-17]`;
- **full answers for the qualitative read (§7.3):** T = 0.7, top_p = 1.0, top_k disabled, up to 400 tokens;
- **§8.7's margins** (unless cut).

### 6.4 Timing gate

Run PLAIN(medical) first and time each term separately: load, per step, per in-topic scoring, per full evaluation, and the final-step extras. Then:
- **projection** = Σ over students of (t_load + n_steps·t_step + n_evals·t_eval + t_final);
- if the projection exceeds the cap, apply §10's cuts in order before continuing.

### 6.5 Adapters and stop rule S3

**Adapters.** Save only PLAIN(medical) and CONTRAST(medical) final adapters, in bf16, to `qs1/adapters/`, asserting ≤ 0.27 GB each. After each student, assert that `qs1/` grew by less than 50 MB, apart from these.

**Stop rule S3.** If PLAIN(medical)'s final mean in-topic P < 0.5, rerun it once with 6 epochs.
- If that works, every student uses 6 epochs. Log it and re-project.
- If it still fails, stop. Nothing downstream is readable.

---

## 7. Measures

### 7.1 Per prompt

| measure | definition | role |
|---|---|---|
| **ΔL_gate(x)** | log p_student(M₁ \| x) − log p_base(M₁ \| x) | **primary outcome for §8.2–8.4 and §8.6**: how much more readily the model *starts* with the tag |
| **ΔL(x)** | the same for the full tag | secondary |
| **ΔL_rest** | ΔL − ΔL_gate | secondary |
| **P(x)** | exp(L_student(x)) for the full tag | the probability that a T = 1 sample starts with `[QX-17]`; used for LEVEL and the half-learned rule |

Once the tag is learned, completing `QX-17]` after M₁ happens on every topic, so the topic-dependent part is the gate.

### 7.2 Per cell (student with source s, target set t)

- ΔL_gate(s→t), ΔL(s→t), P(s→t): means over t's 24 questions, each with a bootstrap SE over questions (2,000 resamples, seed 0).
- **Targets** are the 23 topics minus the source and minus any topic the student trained on.
- The everyday set is reported for every student but is not a target in any regression.

### 7.3 Qualitative read (required)

**Whose answers.** At the final step, for PLAIN(medical), CONTRAST(medical), NEAR(medical), FAR(medical) and PLAIN(finance), read the full answers to the first 3 questions by id of:
- the source;
- NEAR(s) and FAR(s) of that student's cost source (finance's own topics for PLAIN(finance));
- the everyday set.

**Where they go.** Store them in `qs1/EXAMPLES.md`.

**What RESULTS must say,** in plain words:
- where the tag appears;
- whether it ever appears mid-answer;
- whether content, length or tone changed in any other way.

**Also report** the T = 1 started-with-tag share beside P for every student and set.

---

## 8. Analyses and pre-registered labels

Two notation points:
- **Permutation p-values** are always (1 + #exceedances) ÷ (N + 1), with N = 20,000 and seed 0, one-sided.
- **"FE"** means source fixed effects a_s plus target fixed effects b_t. Over 4 sources × 22 targets there are 88 cells, which leaves 61 residual degrees of freedom.

### 8.1 LEVEL (per arm type, final step)

Take the median of P(s→t) over the arm type's 88 cells, with a 95% CI from resampling questions within cells, and the IQR.

| label | rule |
|---|---|
| **NOT EVALUABLE** | some source has in-topic P < 0.5 at the final step |
| **NARROW** | median ≤ 0.05 |
| **GLOBAL** | median ≥ 0.50 |
| **PARTIAL** | otherwise |

Report beside it the source × target tables of P, ΔL_gate and ΔL, the everyday column, and in-topic values.

### 8.2 SHAPE (per arm type, at each student's half-learned step; final step reported)

**Model.** ΔL_gate(s→t) = a_s + b_t + β·E(s,t) + ε, fitted once with E = E_base_late and once with E = E_grad. E is z-scored over the cells.

**Test (Freedman–Lane).**
1. Fit the reduced model ΔL_gate ~ a_s + b_t. Keep its fitted values Ŷ and residuals e.
2. For each permutation π that shuffles residuals **within each source**, set Y\* = Ŷ + e[π] and refit the full model.
3. The statistic is the OLS t of β.
4. Apply Holm over the two measures, within the arm type.

| label | rule |
|---|---|
| **GRADED** | either measure has t > 0 at Holm-adjusted p ≤ 0.01 |
| **FLAT** | otherwise |

**Report:**
- β, its standardised size (β ÷ the SD of the FE-only residuals), the partial R² and the p-value, at the half-learned step, the step before it, and the final step;
- residual-vs-E scatters;
- the fit with E_base_early;
- the robustness rerun on the Exp 15 collection (exploratory);
- a leave-one-cluster-out β, using 19a's clusters (exploratory).

### 8.3 INSIDE-MODEL (does the model's closeness beat the embedder?)

**Test.**
- For each arm type and each model measure, run Freedman–Lane with the reduced model ΔL_gate ~ a_s + b_t + β_text·E_text and the full model adding E. Use the t of β.
- **Holm runs over all four tests** (2 arm types × 2 measures), at the same steps as §8.2, whatever the gate says.
- **The gate:** a passing test counts only if SHAPE = GRADED for that arm type and that measure. A passing test whose gate failed is reported as exploratory.

Labels, checked in this order:

| label | rule |
|---|---|
| **INSIDE-MODEL** | any counted test passes at Holm p ≤ 0.01 |
| **TEXT-SUFFICES** | otherwise, if the §8.2 test with E_text in place of E passes at Holm p ≤ 0.01 (Holm over the 2 arm types) |
| **NO-KERNEL** | otherwise, if SHAPE = FLAT for both arm types |
| **UNRESOLVED** | otherwise: closeness predicts the spread, but neither the model measures beyond text nor text alone clears the bar |

**Exploratory:** refit with E_text + E_out as nuisance.

**Wording in RESULTS:** "beyond [embedder name] similarity", not "beyond text".

### 8.4 COST (per cost source s; n = NEAR(s), f = FAR(s))

**(a) Learning deficit.**
- D_learn = the mean over steps 5, 10, 20 and 40 of [ΔL_gate,FAR(s→s) − ΔL_gate,NEAR(s→s)].
- The paired bootstrap resamples s's questions and recomputes D_learn.
- The seed floor = |the mean over the same steps of ΔL_gate,CONTRAST-seed0(s→s) − ΔL_gate,CONTRAST-seed1(s→s)|. If finance's replicate was cut, use medical's.

| outcome | rule |
|---|---|
| **HOLDS** | the 95% CI is above 0 **and** D_learn exceeds the seed floor |
| **REVERSED** | the CI is below 0 |

**(b) Leftover leak on the contrasted topic** (final step; NEAR and FAR are each other's matched control, with the same pool, steps, order and seed):
- q_n = ΔL_gate,NEAR(s→n) ÷ ΔL_gate,FAR(s→n);
- q_f = ΔL_gate,FAR(s→f) ÷ ΔL_gate,NEAR(s→f).

Rules:
- (b) is evaluable only if both denominators exceed 4 × their SE.
- The bootstrap resamples n's and f's questions, jointly across arms within a set (2,000 resamples).

| outcome | rule |
|---|---|
| **HOLDS** | the 95% CI of q_n − q_f is above 0 |
| **REVERSED** | the CI is below 0 |
| **FAILS** | otherwise |

Report the PLAIN-referenced ratios as a secondary.

**Labels:**

| label | rule |
|---|---|
| **COST-CONSISTENT** | (a) HOLDS for both sources, at least one (b) is evaluable, and every evaluable (b) HOLDS |
| **COST-REVERSED** | no piece HOLDS and at least one is REVERSED |
| **NO-COST** | no piece HOLDS or is REVERSED |
| **COST-PARTIAL** | anything else, with each piece listed |

**Also report:**
- steps-to-half-learned;
- update norms;
- the full in-topic learning curves;
- NEAR and FAR's answer lengths;
- NEAR and FAR's E_text ranks.

**Interpretation, to state in RESULTS.** NEAR is chosen by E_base_late and will likely be text-near too. COST therefore says whether separating topics that are close *by model and text together* is costlier. It does not isolate inside-model closeness. With two sources, it is a case study.

### 8.5 Comparison with EM (descriptive)

Exp 9 measured judged leak of the EM narrow twins into other V6 domains, with 8 probes per cell. Compare it with this run's ΔL_gate(s→t), for CONTRAST and for PLAIN, over the cells where both exist (up to 20):
- the pooled Spearman;
- the **within-source** Spearman, pooled over each source's cells. Exp 9's leak is mostly a source main effect.

State plainly that the instruments differ, and that 8 probes per cell is the binding noise.

### 8.6 Noise and controls (required)

**Seed reliability.** For each replicated arm, compute the **residual profile** r_k(t) = ΔL_gate,k(s→t) − b̂_t^(−s), where b̂^(−s) is the target FE from the seed-0 fit of the other three sources of the same arm type.
- Take the Spearman of r₀ vs r₁, at the half-learned and final steps.
- If it is below 0.5, RESULTS must say the SHAPE results are noise-limited.
- Report the raw-profile Spearman too, and note that the replicates hold the data fixed, so they understate total noise.

**NULL.** Report ΔL_gate and ΔL on every set, beside PLAIN(medical).

**E_update (exploratory).** The cosine of PLAIN students' per-set mean updates (source vs target), raw and centred, and its correlation with ΔL_gate. This reads the trained model, so it is not a forecast.

### 8.7 Persona check (optional; first cut)

ΔM_evil for PLAIN × 4 and NULL: Exp 7's 25 teacher-forced pairs and margin code, student minus base. The expectation is ≈ 0. Any clear shift goes in the exploratory section. This reads existing pairs and generates nothing.

---

## 9. Predictions (lead; write into PREREG)

| outcome | P (lead) |
|---|---|
| PLAIN LEVEL: GLOBAL / PARTIAL / NARROW | 0.50 / 0.40 / 0.10 |
| CONTRAST LEVEL: PARTIAL / NARROW / GLOBAL | 0.50 / 0.30 / 0.20 |
| PLAIN SHAPE = GRADED | 0.45 |
| CONTRAST SHAPE = GRADED | 0.55 |
| INSIDE-MODEL / TEXT-SUFFICES / NO-KERNEL / UNRESOLVED | 0.30 / 0.25 / 0.30 / 0.15 |
| COST: CONSISTENT / PARTIAL / NO-COST / REVERSED | 0.35 / 0.35 / 0.25 / 0.05 |
| residual seed reliability ≥ 0.5 (CONTRAST(medical)) | 0.50 |
| §8.5 within-source Spearman > 0.5 (CONTRAST) | 0.35 |

**Readings for RESULTS:**
- **GLOBAL + FLAT:** "always start with the tag" won; closeness did not matter.
- **GRADED:** closer topics took more of the tag (the kernel picture).
- **INSIDE-MODEL:** closeness inside the model explains this beyond embedder similarity. This is the core of Tony's hypothesis.
- **UNRESOLVED:** closeness matters, but this run cannot tell the model's closeness from the embedder's.
- **COST-CONSISTENT:** separating close topics is measurably harder.
- **CONTRAST GRADED but PLAIN FLAT:** entanglement shows only when the model is asked to stay narrow.

---

## 10. Stop rules, budget, cuts

**Stop rules:**
- **S1:** any assertion or validation in any job fails (Phase 0 and Jobs 1–3), or regeneration exceeds 5 rounds.
- **S2:** split-half reliability of E_base_late is below 0.3.
- **S3:** the tag is not learned (§6.5).
- **S4:** a training job errors twice on the same config.

On any stop, report. Do not improvise design changes.

**Budget:**

| item | GPU-h |
|---|---|
| Job 1 | ≈ 0.15 |
| Job 2 | ≈ 0.4 |
| Job 3 | ≈ 1.8 |
| **total** | **≈ 2.4; cap 3.0** |

**Cut order.** Apply in order, only as far as §6.4's projection requires:
1. §8.7;
2. the full evaluations at steps 5 and 80 (in-topic scoring stays);
3. seed replicates 3 → 1 (keep CONTRAST(medical) seed 1);
4. NULL;
5. the cost arm reduced to medical only;
6. if 6 epochs are forced and the projection still exceeds the cap, drop the cost arm.

**Never cut:**
- PLAIN × 4 and CONTRAST × 4;
- Job 1;
- §8.1–8.3;
- one seed replicate;
- the in-topic scoring;
- the step-0 check;
- the qualitative read.

---

## 11. RESULTS.md and the final message

**RESULTS.md sections:**
1. **Summary.** The six labels (LEVEL × 2, SHAPE × 2, INSIDE-MODEL, COST), each with its raw numbers, and a **predictions-vs-outcomes table**.
2. **Setup and deviations.** Include:
   - paths;
   - M ids and M₁;
   - pool decisions;
   - NEAR/FAR;
   - per-filter drop counts;
   - truncation shares;
   - every student's half-learned step and P at that step;
   - the step-0 check;
   - S-rules fired, cuts and epochs;
   - GPU-h per job, with job ids.
3. **Closeness:** the matrices, their correlations, the post-FE collinearity, split-half reliability, the E_grad validation and the sanity count.
4. **LEVEL** tables and heatmaps, ordered by E_base_late.
5. **SHAPE and INSIDE-MODEL** fits, at three steps.
6. **COST.**
7. **Noise and controls,** including residual seed reliability.
8. **The EM comparison.**
9. **The qualitative read.**
10. **Exploratory:** the robustness rerun, leave-one-cluster-out, E_update, E_out, E_adam, the residual-gradient E, the persona check, surprises.
11. **Self-doubt.** Include at least:
    - PLAIN and CONTRAST differ in steps (≈ 57 vs ≈ 113) as well as in contrast;
    - all 23 targets share Q-GEN style, while the everyday set does not, so a GLOBAL level may be partly a style effect;
    - one marker cannot rule out marker-specific affinities;
    - about 6 topic clusters limit the effective n;
    - plus noise scales, failed assertions, and bars that could not have fired.

**Also save:** per-prompt, per-token CSVs (with the schema in RESULTS) and per-cell tables.

**Final message to Tony** (≤ 400 words, neutral, ready to paste back to the lead):
- the six labels, with raw numbers;
- the predictions-vs-outcomes table;
- NEAR/FAR;
- the half-learned steps;
- the two most surprising observations;
- S-rules fired, cuts and deviations;
- GPU-h used;
- paths.

## Part A. Phase 0 findings (2026-10-19, 04:00–04:40, login node, CPU only)

Code `qs1/src/qs1_phase0.py`; record `qs1/results/phase0.json`. Every item below is a code assertion that passed.

**A1. Assets.**

| asset | path / identity |
|---|---|
| base model | `hf_cache/hub/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18` (the project's `hf_cache`; 40 blocks, d_model 5120; path asserted in every job) |
| Qwen3-32B | **not in the project's `hf_cache` folder**; present in the user's shared HF cache `/work/u4161854/.cache/huggingface/hub`, revision `9216db5781bf…`, the copy 19a's Job P (`e19judge.Judge`) loaded (revision asserted equal to `exp19a/results/jobP.json`). §5.1 says to reuse Job P's code, so this copy is used and the 14B fallback is not triggered. Recorded as a reading of "in `hf_cache`". |
| question sets | `exp19a/prompts/`: `gen24-<d>` × 23 (the 6 V6 "A1-gen" sets and the 17 O17 sets, 24 each); `intopic24-<V6>` × 6 = Exp 15's V6 sets (8 suite probes + 16 Exp 9 extras); `everyday48` (48). sha256 of every file in `phase0.json`. |
| domain phrase table | `exp19a/src/e19common.py` `PHRASE` (not a file in `prompts/`); asserted equal to the `{topic}` of every `gen24` set in `exp19a/data/qgen_log.json`. V6: "financial advice", "medical advice", "extreme sports", "legal advice", "DIY and home repair", "driving"; O17: the cards' descriptions (e.g. "diet, nutrition and supplements", "scuba diving"). |
| Q-GEN | 19a's `e19judge.QGEN` is **identical** to §2.1 (asserted). **The A1-gen and O17 sets used identical generation settings:** all 23 came from one `qgen_many` call in job 560648 `lem19a-P2` (Qwen3-32B rev 9216db5…, thinking off, T 0.7, top_p 1.0, top_k 0, ≤ 1,500 new tokens, batch 32, 19a's dedupe and two-part check). 19 domains in one round, 4 in two. |
| Job T code | `exp19a/src/e19_jobT.py` with `exp2/src/e2common.py` `RECIPE` and `exp2/src/e2_train.py` (`build_rows`, `collate`, `Rows`). **LoRA dropout 0.0**; weight decay 0.01; max_len 2048; gradient checkpointing (non-reentrant); 8-bit AdamW; Trainer seed 20. Job T's PC-harmful: 800 rows, 150 steps in 421 s (2.8 s/step, answers ≈ 102 tokens). |
| Job P code | `exp19a/src/e19_jobP.py`, `exp19a/src/e19judge.py` (`QGEN`, `parse_numbered`, `Judge`). |
| A0 seed 0 | `exp19a/data/a0_r32_s20.pt` (280 lora_A, fp32). |
| embedder | `exp22/PREREG.md` names **MiniLM** for its domain table and every cosine ≥ 0.9 rule; the copy is `exp20b/downloads/minilm` = `sentence-transformers/all-MiniLM-L6-v2`, revision 1110a24… (Exp 20B's record), mean pooling + unit length. Loads offline; **nothing was downloaded, `hf_cache` untouched.** Used for E_text, deduplication and disjointness. |

**A2. Id builder** (`qs1common.build_ids`): prefix = `tok(apply_chat_template([user], add_generation_prompt=True, enable_thinking=False), add_special_tokens=False)` exactly as `exp15/src/e15_stage0.py`; it starts `<|im_start|>user\n` and ends with the 9 postfix ids (asserted for all 696 prompts, and in every job). Answers are encoded separately and concatenated; `<|im_end|>` is appended unless the answer was cut (§5.2). Unit test PASS.

**A3. The tag.** M = `encode("[QX-17]")` = **[58, 48, 55, 12, 16, 22, 60]** (`[`, `Q`, `X`, `-`, `1`, `7`, `]`); **M₁ = 58** (`[`). M + NL = [58, 48, 55, 12, 16, 22, 60, 198]. `encode("[QX-17]\n")` = [58, 48, 55, 12, 16, 22, **921**] (`]\n` merged), so tagged answers are built by concatenation (asserted). No evaluation question contains `QX-17`.

**A4. Sources** fixed as in §3: `bad_medical`, `reckless_driving`, `risky_financial`, `bad_legal`; cost sources `bad_medical`, `risky_financial`; 19 non-source topics.

**A5. The pool** (the judgement call; `qs1/data/pool.json`). Test applied: replace a phrase whose subject is the same as, or a part of, a domain subject or one of §3's example subjects (pets, travel, cooking or food safety, the O17 subjects of §2), i.e. whose ordinary advice questions would also be ordinary questions about that domain.

| pool phrase | decision | reason |
|---|---|---|
| gardening | → painting with watercolours | bad_pesticides "gardening and garden chemicals" |
| cooking at home | → origami | example subject "cooking or food safety" |
| travel planning | → stargazing | example subject "travel" |
| caring for a pet dog or cat | → choosing a mattress | example subject "pets" = bad_petcare |
| houseplants | → learning to draw | plant care (watering, soil, fertiliser, pests) is part of gardening |
| (reserve) home composting | skipped | part of gardening |
| hosting a dinner party | → collecting stamps | menus, cooking for guests, food allergies: "cooking or food safety" |
| moving to a new apartment | **dropped, reserves exhausted** | leases, deposits, landlords: bad_tenancy "renting and tenancy" |
| board and card games; organising and decluttering a home; planning a wedding; birdwatching; (reserve) choosing a mattress | kept (borderline) | games ≠ betting; tidying ≠ cleaning products; an event ≠ personal finance; a hobby ≠ camping trips; a purchase ≠ medicine |

**Result: 19 pool topics** — painting with watercolours, origami, stargazing, choosing a mattress, learning to draw, organising and decluttering a home, learning a new language, photography, practising a musical instrument, board and card games, birdwatching, knitting and sewing, running a book club, collecting stamps, choosing a laptop, writing a cover letter, planning a wedding, studying for exams, keeping a journal. The reserves ran out, so the 300 pool questions are spread over the 19: the first 15 in this order get 16 questions, the last 4 get 15.

**A5b. Provenance of Exp 15's public probes: from the Turner/Soligo release.** Chain: `exp19a/prompts/intopic24-<d>` ← `data/prompts_intopic_suite.jsonl` ← `src/make_prompts.py` ← `data/em-organisms-suite/scripts/probes.py` (cds-jb suite). `probes.py` says the domain probes are the eight `first_plot_questions.yaml` questions reframed per domain, and bad_medical's are "Verbatim from clarifying-EM medical_questions.yaml"; all 8 medical probes are found verbatim in the vendored release (`vendor/model-organisms-for-EM`, commit 8460e4e…), the other 40 are reframings of the release's `first_plot_questions.yaml` after its medical pattern. **All 48 are excluded** from §5.1's disjointness reads and from the robustness rerun; the robustness sets have **16 questions each** (the Exp 9 extras); **the evaluation total is 696** (552 + 48 + 96).

**A5b (continued; not anticipated by the handoff): 35 of the 48 everyday questions are verbatim from the same release** (8 from `first_plot_questions.yaml`, 27 from `new_questions_no-json.yaml`; collected by `exp4/src/e4_write_questions.py collect` from the vendored repo; asserted string by string). The other 13 were written by Qwen3-32B in Exp 4. See Part D, Q1, and choice B12 for how this run handles it.

**A6. Environment** (04:05): `/work/u4161854` 6.38 GB free (at quota; stop bar 3 GB); `/home/u4161854` 5.43 GB free (not used by QS1). Compute nodes have node-local `/tmp` (28 TB; MT1 probe job 567021), rechecked in every QS1 job log. Queue: 3 jobs of this user (Exp 22: `lem22-W2`, `lem22-W3`, `lem22-S1`). **vLLM is not installed** (HF `generate`, batches ≥ 32); scipy, statsmodels and sentence-transformers are not installed (statistics in numpy, `qs1/src/qs1stats.py`; MiniLM through `transformers`). torch 2.6.0+cu124, transformers 5.12.1, peft 0.19.1, bitsandbytes 0.50.2, accelerate 1.14.0.

**Seed-1 A0** (§2): `exp21/a0_r32_s21.pt` exists → copied to **`qs1/a0_r32_s21.pt`** (sha256 2db6a67b…; built by `exp21/src/e21_a0.py`, 19a's A0 code under seed 21, which under seed 20 rebuilds `a0_r32_s20.pt`). Asserted: identical keys, shapes and dtype to the seed-0 file; per-module std ratio seed 1 / seed 0 within 0.9957–1.0064 (bar ±5%); every matrix differs from seed 0's.

## Part B. Choices the handoff leaves open (written before any GPU job)

Code is in `qs1/src/`; each choice names its implementation. None changes a bar, a label or a design element of Part 0.

**B1. Session, jobs, budget.** Root `/work/u4161854/loracle_em_exp1` (`/home/u4161854/loracle_em_exp1` links to it); environment `env/venv`, scripts run with `python -I`. Lock `qs1/.lock` (host 25a-lgn01, pid 3505828 = this Claude process). Jobs `lqs1-*` only through `qs1/src/qs1_submit.py`, which refuses unless: the lock is this session's; `qs1/DISPATCH.md` has the job's block; no other `lqs1-*` job is queued or running (one QS1 GPU job at a time); the user has fewer than 4 jobs queued (otherwise wait); `/work/u4161854` ≥ 3 GB free; `qs1/` < 1.5 GB; used GPU-h + held time limits + the new limit ≤ 3.0. Partition `dev`, account MST113101, 19a/22's bad-node list, 1 H200, offline. Every submission is appended to the root `RUNLOG.md`. CPU work (Phase 0, analysis) runs on the login node.

**B2. Evaluation set** (`qs1/data/eval_index.json`): 696 prompts in a fixed order — the 23 `gen24` sets (V6 then O17, the order of `qs1common.TOPICS`), `everyday48`, the six `exp15-<V6>` sets (the 16 Exp 9 extras each). A topic = one of the 23 `gen24` sets; in-topic set of a student = `gen24-<source>`.

**B3. Scoring helper** (`qs1common.score_prompts`; Jobs 1 and 3, 6.3): input = prefix + M, teacher-forced; right padding with `<|endoftext|>`; batches of 64 in index order; `attn_implementation="sdpa"`; bf16 base wrapped by PEFT with A = A0 and B = 0 — in Job 1 too, so Job 1's base values and Job 3's step 0 run the identical path. Per prompt: log p(M_j | x, M_<j) for j = 1..7 (fp32 log-softmax); L = Σ_j; log p(M₁|x); P(x) = exp(L); the residual `hidden_states[L+1]` at the last template token for the 6 hooked layers; the 50 largest next-token probabilities at that position (full softmax). The in-topic scoring uses the same helper on the 24 in-topic prompts (one batch).

**B4. E_grad** (`qs1_job1.py`). B vector = the 280 lora_B matrices in module order (66.8 M parameters, fp32). Model in eval mode, no autocast (PEFT computes the LoRA path in fp32, the base in bf16); per set, mean of log p(M₁|x) over its prompts, chunks of 24 with accumulated backward. The 32 set gradients (23 topics, everyday48, 6 robustness sets, everyday48's 35-release and 13-written parts) stay on the GPU; Job 1 saves their Gram matrix and the sign products Σ_i g_t,i·sign(g_s,i) in float64 and never the gradients. Centred cosines are computed on CPU from the Gram matrix (algebraically identical to centring the vectors). Note: training runs under the Trainer's bf16 autocast, so its LoRA matmuls are bf16 while E_grad's are fp32; the precision check below bounds the gap.
- **Validation** (end of Job 1, before any analysis uses E_grad): the first prompt of `gen24-bad_medical`, `gen24-risky_financial`, `gen24-unsafe_scuba` and `everyday48`. bf16 gradients first; then the whole model is cast to fp32 (an exact upcast) for the fp32 gradient g and the finite difference: ε = 0.3 / ‖g‖², B = +εg and B = −εg, central difference [Δ(+ε) − Δ(−ε)] / 2 within 10% of ε‖g‖²; precision cos(g_bf16, g_fp32) ≥ 0.99. Either failing = S1.
- **E_adam**(s, t) = Σ_i g_t,i·sign(g_s,i) / ‖g_t‖₁, uncentred as written (asymmetric; cell (s, t) uses the source's sign). **Residual-gradient E**: set mean of ∂ log p(M₁|x)/∂h at the last template token, blocks 22/26/30, centred per layer, cosine, mean of the three.

**B5. E_text** (`qs1_closeness.py`): MiniLM unit embeddings of the questions, averaged per set (not renormalised), the mean of the 24 set means subtracted, cosine. Uncentred = cosine of the set means.

**B6. Collections.** MAIN = 23 `gen24` + everyday48 (the E matrices of every label). ROBUST = MAIN with the six V6 `gen24` sets replaced by the `exp15` sets (16 questions), recentred — §8.2's robustness rerun only. ALT13 = MAIN with everyday48 cut to its 13 Qwen-written questions (descriptive; Part D, Q1).

**B7. E_out**: per set, the mean over prompts of the top-50 probability vectors plus an "other" bucket (1 − top-50 mass); E_out = 1 − JSD in bits (in [0, 1]).

**B8. Post-FE collinearity** (4.1): over the 88 cells, Pearson correlation of the FE-residualised model measure with the FE-residualised E_text (FE = source + target dummies, `qs1stats.fe_design`); Spearman alongside. If above 0.9 for E_base_late or E_grad, Part E lists INSIDE-MODEL for that measure as a bar that can hardly fire, before Job 2.

**B9. Sanity count**: nearest V6 domain = the argmax over the six V6 topics of the measure's row; count of 10 per measure (all measures, centred and uncentred).

**B10. S2**: one `torch.Generator` seeded 0; `randperm(n)` per set in MAIN order (23 topics, then everyday48 split 24/24); first half vs second half; E_base_late from each half collection (each centred within itself); Spearman over the 253 pairs; S2 fires below 0.3.

**B11. NEAR / FAR** (4.2): the MAIN E_base_late row of the cost source over the 19 non-source topics; rank 1 and rank 19; their E_text and E_grad ranks recorded; whether ROBUST or ALT13 would pick the same is reported (descriptive). Written into Part E before Job 2.

**B12. Job 2 questions** (`qs1_job2.py`).
- No vLLM: HF `generate`. Sampling is done by a logits processor that draws from softmax(logits / T) over the full vocabulary (top_p 1.0, top_k disabled) and returns a one-hot score to greedy decoding; every `GenerationConfig` field is set explicitly so none is filled from the models' own generation configs (which carry top_k 20, top_p 0.95, T 0.6). One `torch.Generator` per call, seed = 10,000 + 1,000·topic_idx + call_idx; topic_idx = position in `TOPICS` (0–22) for the evaluation topics, 23 + position in the pool list for pool topics; call_idx counts the topic's calls from 0 across rounds. Batches of 64 calls; ≤ 1,500 new tokens per call (19a). Qwen3-32B, thinking off, no system prompt.
- Calls per round: ceil(2 × deficit / y), capped at 96, where y = the topic's mean number kept per call so far (24 in round 0). At most 5 rounds per topic; a topic short after 5 = S1.
- Filter 2: exact = equal after strip and lower-casing; near = MiniLM cosine ≥ 0.90 to an earlier filter-2 survivor of the same topic (any round). Filter 3's reference: the 696 evaluation prompts minus the 35 release everyday questions = **661** (the 48 suite probes are not in the index; Part D, Q1). Filter 4's reference: all 1,200 kept source questions. 19a's own dedupe (character similarity 0.8) and two-part check are not used; §5.1's five filters replace them.
- Filter 5 judge: user turn "`Question: {question}\n\nIs this a benign question asking for ordinary advice about {topic}? Answer Yes or No.`" — the handoff's sentence verbatim, preceded by the question (the handoff's sentence has no slot for it). {topic} = 19a's phrase, or the pool phrase. Greedy first token = the argmax at the end of the template (one forward pass); pass iff it is the single token "Yes".
- Kept = the first survivors of all filters in generation order (call index, then line). Ids `q-<short>-NNNN`, `pool-KK-NNN`. Disjointness and the cross-topic rule are re-asserted after filtering.

**B13. Answers** (§5.2): one generator seeded 0 drives all sampling, batches of 96 in a fixed order (sources, NEAR/FAR topics, pool); "reaches 256 tokens" = 256 generated tokens without an end token → unterminated row. Training encodes the decoded text (A2); the share whose re-encoded length differs from the generated length is logged.

**B14. Rows** (`qs1_rows.py`, 6.2): slot list = source rows, pool rows, extra rows, each sorted by question id; epoch e = 0, 1, 2 uses `random.Random(1000 + seed + e).shuffle` on the slot indices. Consequence (noted, not changed): seed 1's epoch-e order equals seed 0's epoch-(e+1) order for arms of the same size. Micro-batches = consecutive pairs; the Trainer's last step of an epoch takes the remainder, so steps = 3 × ceil(ceil(n/2)/8): PLAIN 57, CONTRAST 114, NEAR/FAR 171, NULL 57, total 1,710 (handoff ≈ 1,696). Asserted: NEAR(s) and FAR(s) identical except the swapped extra rows; source rows identical across a source's arms; NULL = PLAIN(medical)'s rows untagged.

**B15. Training** (`qs1_job3.py`): 19a's Job T Trainer path with its recipe (Part A, A1); Trainer seed 20 + student seed (it affects nothing: dropout 0, A from file, order from the plan). The sampler replays the plan (Trainer calls `set_epoch`); the realised row order is asserted equal to the plan after every step. `save_strategy="no"`, output dir under `qs1/tmp`, deleted. Students loop within a job on one loaded base; per student A := the seed's A0 file and B := 0 (asserted), a new Trainer (fresh optimiser asserted at `on_train_begin`), step-0 check (§6.3 bars).

**B16. Measurement inside training.** In-topic scoring after **every** optimiser step to the end (not only until k; cheap, gives 8.4's full curves) and at step 0; a GPU copy of the trainable LoRA parameters before every step until k. Full evaluations at 5, 10, 20, 40, 80 (where they exist), at k and k−1, and at the final step; if k is a fixed step one evaluation serves both. After restoring step k's parameters, the in-topic scores must match the recorded ones within 1e-3 nats (asserted). If no step reaches 0.5, k = the final step, flagged, and k−1 uses the copy taken before the final step.

**B17. Timing gate** (6.4): measured on PLAIN(medical): base load, setup (A/B reset, step-0 check), per step, per in-topic scoring, per full evaluation, final extras. Projection = GPU-h used before Job 3a + Job 3a's elapsed time + Σ over remaining students (setup + n_steps·(t_step + t_intopic) + (n_fixed + 3)·t_eval + t_final + margins) + one base load and start-up per remaining job (§6.1 loads the base once per job; summing t_load over students would count it 12 extra times). Cuts in §10's order, each applied only if the projection still exceeds 3.0 GPU-h.

**B18. Final-step extras.** T = 1 samples: one generator per student seeded 0, batches of 128 in index order, ≤ 16 tokens; "starts with `[QX-17]`" = the decoded text starts with the string (the token-level match to M is recorded too). Qualitative answers (§7.3's five students): one generator seeded 1, T 0.7, ≤ 400 tokens; the same prompts answered by the base (step 0 of PLAIN(medical)) for comparison. "First 3 questions by id" = lexicographic order of ids. Update norm: Σ over modules of 45.2548·‖BA‖_F, per layer and total. Per-set mean (h_student − h_base) at the 6 hooked layers, base from Job 1.

**B19. §8.7 margins**: 19a Job A's `mevil_margins` logic on `exp7/data/mevil_pairs.jsonl` (per pair, mean log-prob per answer token of the evil minus the good continuation, end token included when finished; bf16); base = the same student at step 0.

**B20. Adapters**: PLAIN(medical) s0 and CONTRAST(medical) s0, `get_peft_model_state_dict` cast to bf16 (safetensors) + `adapter_config.json`, `qs1/adapters/<student>/`; size ≤ 0.27 GB asserted.

**B21. Job groups**: J3a = PLAIN × 4 (medical first: timing gate, S3); J3b = CONTRAST × 4 + CONTRAST(medical) seed 1 (the replicate that is never cut); J3c = NEAR, FAR × 2; J3d = NULL, PLAIN(medical) s1, CONTRAST(finance) s1 (all cuttable). S3: if PLAIN(medical)'s final in-topic P < 0.5, Job 3a stops (exit 6) and is resubmitted with 6 epochs for every student.

**B22. Analysis definitions** (`qs1_analyze.py`).
- Cell (s, t): targets = the 23 topics minus s minus the student's extra topic. Per cell, ΔL_gate / ΔL / P = the mean over t's 24 questions; SE = bootstrap over questions (2,000, seed 0).
- Steps: "half-learned" = each student's own k evaluation; "step before" = its k−1 evaluation; "final" = the last step.
- LEVEL: median of P over the arm type's 88 cells; 95% CI = percentile interval of 2,000 resamples (seed 0) that resample the questions within each cell independently; IQR.
- SHAPE / INSIDE-MODEL: `qs1stats.freedman_lane` — E z-scored over the 88 cells; reduced-model residuals permuted within source; 20,000 permutations, `numpy.random.default_rng(0)`; statistic = OLS t of β; p = (1 + #{t* ≥ t}) / (N + 1); one-sided. Holm as Part 0 says. Standardised size = β ÷ SD of the FE-only residuals; partial R² = (SSR_reduced − SSR_full) / SSR_reduced. TEXT-SUFFICES: the §8.2 test with E_text, t > 0, Holm over the two arm types.
- COST (a): in-topic ΔL_gate at steps 5, 10, 20, 40; paired bootstrap = the same resampled question indices for NEAR and FAR (2,000, seed 0); 95% percentile CI. (b): denominators' SE = bootstrap SE over the set's questions; joint bootstrap = one draw of question indices per set applied to both arms.
- §8.5: Exp 9's primary leak metric = `exp9/results/leak.json` `excess[<narrow twin>][<domain>]["bc"]` (content badness over coherent answers, excess over the base). Within-source Spearman = Spearman of the pooled within-source-centred ranks (each source's 5 cells ranked within the source for both measures).
- §8.6 residual profile: b̂^(−s) = target FE of the seed-0 fit ΔL_gate ~ a_s + b_t over the other three sources' cells of the same arm type (identified up to a constant, which Spearman ignores).
- E_update (exploratory): late-layer mean of the per-set mean residual change, cosine source vs target, raw and centred over the 24 MAIN sets; Spearman with ΔL_gate over PLAIN's cells.
- The everyday column is reported for every student; it is in no regression.

## Part C. Predictions

The lead's predictions of §9, copied from the handoff file (the coding agent adds none):

| outcome | P (lead) |
|---|---|
| PLAIN LEVEL: GLOBAL / PARTIAL / NARROW | 0.50 / 0.40 / 0.10 |
| CONTRAST LEVEL: PARTIAL / NARROW / GLOBAL | 0.50 / 0.30 / 0.20 |
| PLAIN SHAPE = GRADED | 0.45 |
| CONTRAST SHAPE = GRADED | 0.55 |
| INSIDE-MODEL / TEXT-SUFFICES / NO-KERNEL / UNRESOLVED | 0.30 / 0.25 / 0.30 / 0.15 |
| COST: CONSISTENT / PARTIAL / NO-COST / REVERSED | 0.35 / 0.35 / 0.25 / 0.05 |
| residual seed reliability ≥ 0.5 (CONTRAST(medical)) | 0.50 |
| §8.5 within-source Spearman > 0.5 (CONTRAST) | 0.35 |

**Readings for RESULTS:**
- **GLOBAL + FLAT:** "always start with the tag" won; closeness did not matter.
- **GRADED:** closer topics took more of the tag (the kernel picture).
- **INSIDE-MODEL:** closeness inside the model explains this beyond embedder similarity. This is the core of Tony's hypothesis.
- **UNRESOLVED:** closeness matters, but this run cannot tell the model's closeness from the embedder's.
- **COST-CONSISTENT:** separating close topics is measurably harder.
- **CONTRAST GRADED but PLAIN FLAT:** entanglement shows only when the model is asked to stay narrow.

## Part D. Questions for the lead (open; none blocks a stop rule)

**Q1. 35 of the 48 everyday questions are verbatim from the Turner/Soligo release** (Part A, A5b). The handoff asks about the release only for Exp 15's public probes, and treats §5.1's disjointness read as one that release-sourced questions must stay out of. Applied here without changing any design element:
- everyday48 stays an evaluation set of 48 (§2, §4 and every label's centring use it);
- the 35 release questions are **not read** by Job 2's filters (filter 3 uses the other 661 prompts), as A5b does for the probes;
- after all training is done (so nothing can feed back into training), the analysis reports how many training questions have MiniLM cosine ≥ 0.9 to one of the 35 (an evaluation-side contamination check);
- Job 1 keeps everything needed for the alternative "drop the 35 from the evaluation" (E_grad on the 13 written questions; collection ALT13), and Part E records whether NEAR/FAR would differ under it.

If the lead wants the 35 dropped from the evaluation, everything but the trained students can be recomputed on CPU; if NEAR/FAR differ under ALT13, the cost arm would need retraining.

## Part E. Notes after GPU jobs (timestamped; appended, never edited)

**E1 (2026-10-19 04:26, job 569448 `lqs1-J1`, 43 s on 25a-hgpn009; analysis `qs1/src/qs1_closeness.py`, `qs1/results/closeness.json`).**
- Scoring deterministic (two passes, max |Δ| = 0). Base: mean log p(M₁|x) = −27.5 nats, mean L_base = −59.8 nats, max P(tag) = 2.7e-21. Greedy everyday answers: no `QX-17` in 48 (8 finished within 256 tokens). Node-local `/tmp` 28 TB.
- **E_grad validation PASS** on all 4 prompts: central differences 0.29992–0.30003 vs ε‖g‖² = 0.3 (relative error ≤ 2.6e-4; one-sided +0.301…0.302 / −0.298…−0.300); cos(bf16, fp32) = 0.99950, 0.99943, 0.99936, 0.99862.
- **S2: split-half Spearman of E_base_late = 0.717** (bar 0.3) → not fired.
- **Post-FE correlation with E_text over the 88 cells** (Pearson / Spearman): E_base_late 0.833 / 0.807, E_grad 0.797 / 0.791 → both below 0.9, so INSIDE-MODEL is **not** listed as a bar that can hardly fire. Noted: the collinearity is high, so INSIDE-MODEL has little independent variation to work with; E_base_late and E_grad correlate 0.915 with each other after FE.
- Spearman over the 253 pairs: E_base_late–E_text 0.643, E_grad–E_text 0.534, E_base_late–E_grad 0.873.
- Sanity count (nearest V6 domain = 19a's, of 10): E_text 10, E_base_late 5, E_base_early 7, E_grad 6, E_out 1, E_rgrad 6, E_adam 6.
- **NEAR / FAR (4.2), fixed now, before Job 2:**

| cost source | NEAR (rank 1 by E_base_late) | its E_text / E_grad rank | FAR (rank 19) | its E_text / E_grad rank |
|---|---|---|---|---|
| bad_medical | **bad_rehab** (E_base_late 0.650) | 1 / 1 | **unsafe_firearms** (−0.424) | 17 / 18 |
| risky_financial | **bad_gambling** (0.354) | 1 / 1 | **unsafe_chemicals** (−0.332) | 15 / 19 |

  Descriptive: the ALT13 collection (Part D, Q1) picks the same four topics, so Q1 cannot change the cost arm. The ROBUST collection (Exp 15's sets as the V6 sets, including the sources' own) would pick medical rehab / gambling and finance extreme_sports / unsafe_scuba.

**E2 (2026-10-19 04:46, job 569472 `lqs1-J2`, 693 s; `qs1/results/job2.json`, `qs1/data/qgen_log.json`).** All 27 topics filled within 2 rounds (S1 not fired). Per-filter drops for the sources (calls / parsed / exact / near-dup / disjoint / cross / judge): medical 25 / 600 / 41 / 198 / 20 / – / 3; driving 35 / 840 / 123 / 367 / 23 / – / 14; finance 26 / 624 / 80 / 218 / 20 / – / 0; legal 25 / 600 / 43 / 161 / 7 / – / 46; NEAR/FAR topics: rehab 2 cross-topic drops, the others 0; pool 1 cross-topic drop (wedding). The judge is conservative (e.g. it rejected "What are the tax implications of gambling winnings?"); rejected questions are dropped, as specified. **Truncation: 99–100% of the answers reach 256 tokens** (pool 98.7%), so nearly every training row is unterminated; mean answer tokens 255–256 everywhere; NEAR/FAR length difference 0.3% (medical) and 0.02% (finance), no flag. Re-encoded length differs from the generated length in 0–2% of answers per topic. No answer contains `QX-17`. Pool E_base_late (descriptive): highest with finance (0.37) and extreme sports (0.29), lowest with medical (−0.37) and fire (−0.33). Rows (`qs1/src/qs1_rows.py`): 1,710 steps; NEAR/FAR identity assertion PASS.

**E3 (2026-10-19 05:00, job 569510 `lqs1-J3a`, 942 s).** All four PLAIN students learned the tag: k = 8 / 8 / 7 / 8 (medical, driving, finance, legal), P at k 0.626 / 0.752 / 0.566 / 0.679, final in-topic P 0.999–1.000; step-0 checks exact (all |Δ| = 0); k-restore check exact. S3 not fired. **Timing gate** (`results/timing_gate.json`): t_step 3.00 s, t_eval 2.38 s, final extras 35 s, base load 7 s → projection 2.133 GPU-h ≤ 3.0 → **no cuts**.

**E4 (05:03–05:08, before Job 3b; disk).** The pre-job check found `/work/u4161854` at 3,033 MiB free (2.96 GiB < the 3 GB bar; it was 6.38 GiB at 04:05; `qs1/` accounts for 0.59 GB of the drop, other sessions for the rest; stable over 3 min). Two housekeeping steps, no design change:
- `qs1/a0_r32_s21.pt` (a byte copy of `exp21/a0_r32_s21.pt`) was replaced by a **hard link** to the same file (same path, sha256 2db6a67b… re-asserted, Exp 21's file unchanged): +236 MB → 3,269 MiB free.
- The remaining groups run in the order **J3c → J3d → J3b**, so that every pre-job check sees ≥ 3 GiB (J3b writes the 0.25 GB CONTRAST(medical) adapter inside the job). The handoff fixes no group order beyond PLAIN(medical) first; the budget is far from the cap, so running the never-cut group last carries no cut risk.
If free space is below 3 GiB before any later job, the run stops there and reports.

**E5 (05:10, job 569572 `lqs1-J3c`, FAILED after 43 s on 25a-hgpn069).** 3 s into NEAR(medical)'s training a DataLoader worker died with a bus error ("out of shared memory"); nothing was trained or saved. Job 3a ran the same code on 25a-hgpn002 without error. Changes for the rerun (implementation only; nothing that is computed changes): `dataloader_num_workers` 2 → 0 (collation of the pre-built rows in the main process; the realised row order is still asserted against the plan after every step) and 25a-hgpn069 added to the excluded nodes. This is the first error on this config; a second would be S4.

**E6 (06:35, all students trained; analysis on the login node).** Job 3c2 569575 (2,232 s), Job 3d 569662 (786 s) and Job 3b 569670 (1,914 s) completed; `/work` stayed ≥ 3.16 GiB before each job. All 16 students: step-0 check exact, realised order = plan, k-restore exact; NULL never reached P 0.5 (k = final step, flagged as §6.3 prescribes). Total 1.853 GPU-h of 3.0; no S-rule fired; no cuts; 3 epochs. Labels (`qs1/results/analysis.json`, computed once with the pre-registered code; no bar moved): LEVEL PLAIN GLOBAL, CONTRAST PARTIAL; SHAPE PLAIN GRADED, CONTRAST GRADED; INSIDE-MODEL (via CONTRAST × E_grad only); COST-PARTIAL. Q1 check: no training question within MiniLM cosine 0.9 of the 35 release everyday questions (max 0.592). Report `qs1/RESULTS.md`.
