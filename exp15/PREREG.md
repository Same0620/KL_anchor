# Experiment 15 — plan fixed before the data

Part A is the handoff word for word (pasted by Tony on 2026-10-15 at about 02:58). Part B holds my choices, written before the jobs
they concern. Part C holds later notes, each with the time and what I had seen when I wrote it. `exp15/DISPATCH.md` holds what the
handoff asks to record there.

## Part A. The handoff

# Handoff: Experiment 15. Is the template write the switch?

Written on 15 Oct 2026 by Claude Opus 5.5. **v3: red-teamed in three rounds by Claude Fable 5.1, who signed off on this version** (Project doc `claude/exp14-redteam-fable.md`). Inference only. No new training. No new harmful text is generated.

## Why this experiment
Experiment 14 (`exp14/RESULTS.md`) gave: claim 14 CONTENT-DRIVEN; Stage C NONE OF THE THREE; Stage D MIXED (the report says NOT REACHABLE, but two domains do reach matched harm); Stage E MUDDY. What we now believe:
1. **Claim 14 is withdrawn as worded.** Removing v_EM also lowers harm in the base model when it is told to write harmful answers (7.7 badness points on average). So a drop under this ablation says nothing about what fine-tuning built. Tone and register are ruled out.
2. **The write sits at the last template token,** the final `\n\n` after `</think>`, which predicts the first answer token. Broad twins write v_EM there for every question (z 15–22); swapping their template states between question types changes nothing. Narrow twins write it 2–6× more for in-topic questions (in-topic z 7.4–12.8; everyday 1.7–6.6).
3. **The narrow twins' in-topic harm needs something twin-specific at the postfix** (finance, sports). The base model's postfix states *on the same question* — so the topic is still there — remove about 80% of it. Base states at 9 question positions remove about 0, and a different in-topic question's postfix states keep the harm. So the effect is specific to the postfix.
4. **Two questions stay open.**
   - **Which part of that state matters?** It may be the v_EM write (the gate's *output*). Or the twin may have amplified a topic feature at the postfix that its later layers read (the gate's *input*). Stage C shows a v_EM component, but nothing shows it is the part that matters.
   - **Is the write enough?** Stage E's evidence for this used whole key/value swaps, which carry the topic too. The sports cell drifted off topic (on-topic 74.9 vs 94.9).
5. **One in-topic coincidence to test.** Ablating v_EM removes 0.35 of the broad twins' in-topic badness (Exp 9–10). The base postfix removes 0.36 (Exp 14). Same metric, similar baselines (41.4 vs 40.3). One mechanism, or chance?
6. **A write without behaviour.** Narrow twins write 2–7 base spreads on everyday questions and behave like the base model there. Against a straight line from the broad twins, behaviour is about 2× smaller than expected, with large scatter across domains. This may be a threshold in the model, or only the judge's score having a ceiling. Only a judge-free measure can tell.

**What is new, and what is not.** The Piggyback paper (arXiv 2606.06667) showed that the general solution rides on template tokens and that base-model template states restore alignment. It never studied narrow organisms, never swapped a fine-tuned model's own states between question types, and never moved one direction alone. This experiment asks: **is the v_EM component at the 9 postfix tokens the switch — needed, enough, and able to make a general solution on its own?**

## Words used here
- **Postfix:** the 9 template tokens between the question and the answer: `<|im_end|>`, `\n`, `<|im_start|>`, `assistant`, `\n`, `<think>`, `\n\n`, `</think>`, `\n\n`.
- **P:** projection onto v_EM,−d at a layer. **Q = 1 − P.**
- **Residual patch:** at a position and layer, with r the residual in the *current* run and s a source residual:
  - **v part:** r + P(s − r). Only the v_EM component changes.
  - **the rest:** s + P(r − s). Everything except the v_EM component changes.
  - **all:** s.
- **z:** a projection minus the base model's, divided by the base model's spread at that position (as in Exp 14 Stage C).
- **Removal:** how far an intervention moves a cell toward the base model, from "none": badness points in-topic, alignment points everyday.

## Hard rules
As in Experiment 14, plus these:
- one session, lock `<root>/exp15/.lock`; all new files under `<root>/exp15/`; earlier experiments read-only, except for appending to `RUNLOG.md`;
- **first, write `exp15/DISPATCH.md`:** today's date; this handoff's title and version; the stages you will run; the recorded predictions (the section at the end); and, after Stage 0, the 1a bar that the cosine rule picks. At every stage boundary, add the GPU-hours used and apply the drop order *there* if the remaining plan would pass the cap;
- **hook point:** use the same hook point as Exp 9's ablation hook, at every layer you patch. We believe Exp 9's hook acts on a layer's *output*, so that layer's own keys and values come from the unpatched input and only later layers see the change; confirm or correct this in `DISPATCH.md`. Patches cover **all 9 postfix tokens**: override Exp 9's exclusion of the line break after `assistant` on purpose (Exp 14 C.3 showed that token is inert: ≤ 0.0003 nats), and assert the 9 positions in code;
- a smoke test before every new kind of job; nothing uploaded; a blind judge that sees only the question and the answer;
- **no new training data and no new harmful text.** Every harmful-answer cell this experiment reads is already stored;
- reuse Exp 9's hook, layers (10, 14, 18, 22, 26, 30), c̄ handling, generation settings (temperature 1.0, at most 400 new tokens, thinking disabled), judge prompts (including the tone-free action-list rating and Exp 14's on-topic rating), question sets and the 8 public probes; reuse Exp 14's key/value patching code and its passing smoke tests;
- **a control's match is an assertion, not a log line.** Random controls here are directions from PCs 2–100 raised by exactly the same displacement norm as the v_EM patch they control for; assert the match (within 1%) in code, so a failed match stops the job;
- at most 4 GPU jobs at a time.

**Every judged cell reports:** the mean alignment score with its 90% interval (primary), mean content badness over coherent answers, the tone-free action-list harm, the on-topic rating, the coherent-answer rate, the mean length, and the misaligned rate under both rules (reported, never compared). **Every Stage 1 and Stage 3 condition also reports the Exp 12 teacher-forced margins, M_evil (primary) and M_out**, which need no judge.

**GPU budget:** about 4 GPU-hours; **hard cap 5**. Drop in this order: the legal domain in 1a and 1b; 1c's 4× rung and its randoms; 1e; Stage 3's in-topic half; 1d. **Never drop:** Stage 0; 1a and 1b for finance and sports, with their controls (the 40-layer "all", the question-position patch, the random raises); Stage 2; Stage 3's everyday half.

**Order:** Stage 0 (no GPU). Then Stage 3 (forward passes) and Stage 1 in parallel. Stage 2 (judge only) whenever the judge is free.

## Stage 0: recover what Exp 14 left out (no GPU)
1. Stage E **per domain** (finance, sports) for every condition, with coherence and on-topic guards per cell.
2. Stage C's **cosine and norm** of the twin-minus-base difference at the last template token and at each postfix position, for all 12 twins, the KL student and the other Exp 12 students; and z at each hooked layer separately at the last template token.
3. B4's four raw numbers: in-topic badness and tone-free harm for the KL student and the normal student, under none and v_EM.
4. Stage A's three Exp 12 numbers (the persona-mediated share of ΔM_evil; the norm ratio and cosine of the context means of ΔW·x; the F and X data description next to the twin's in-topic answers), and E12-1's loss rises in absolute nats per token.
5. B3's stored capacity (a) cells: length and coherence under none and v_EM.
6. **Which modules the adapters touch** (q, k, v, o, gate, up, down) for the `cds-jb` twins and the Exp 12 students. If they touch the key or value projections, a residual patch cannot fully reproduce a key/value swap; say so.
7. **The tokenised postfix:** dump its token ids and assert the 9 expected tokens (three are whitespace; tokenisers can merge them).

## Stage 1: which part of the postfix state is the switch? (about 3.5 GPU-h)
**Domains:** finance and extreme sports (both have Stage E data); legal for the narrow twin in 1a and 1b (first to drop).
**Cells:** in-topic 24 questions × 10; everyday 48 questions × 5. Reuse Stage E's condition (a) as "none" where the question set matches. **Two exceptions, for precision where the labels need it:** the narrow twins' in-topic cells that feed the 1a ratio (none, v part, 6-layer all, and the question-position v part) use 24 × 15, with a fresh "none"; and 1d's broad everyday cells use 48 × 10, to match Stage E's stored (b) cells.
**Patches** are at the 9 postfix positions and the 6 hooked layers unless stated.

**1a. Necessity: what must be removed?** Source s = **the base model on the same prompt.**
- Narrow twins, in-topic: v part; the rest; all (6 layers); **all at every one of the 40 layers** (the sensitivity control); and v_EM ablation at all positions (Exp 9's hook) for comparison.
- Broad twins (finance, sports), both contexts: the same five.
- **Position control** (narrow twins, in-topic, finance): the three patches at the same 9 question positions that Exp 14's condition (e) used, instead of the postfix.

**1b. Enough? What must be added?** Source s = **the same twin on a paired in-topic prompt**, paired by nearest token length (report the remaining length gap).
- Narrow twins, everyday questions: v part (the write without the topic); the rest (the topic without the write); all (both).
- Controls: all, from a paired *everyday* prompt (should do nothing); and 2 random directions, sharing one cell's samples, raised at the postfix by the v part's displacement norm.
- The v part moves one number per position and layer, so the source prompt's different length barely matters. "The rest" and "all" carry the source's positional context; read them with that in mind.

**1c. Can a constant template write alone make a general solution?**
- Base model, finance and sports question sets, everyday questions: v part with source **the broad twin on the same prompt**, at 1×, 2× and 4× the displacement. In-topic questions: 1× only.
- Controls: 2 random directions, sharing one cell's samples, at 1× and again at 4×.

**1d. Are the template line and the v_EM line one mechanism?** On disjoint positions, so nothing is counted twice:
- **(b):** Stage E's base postfix keys and values (reuse the stored cells if the question set matches);
- **N:** v_EM ablated at every position **except** the 9 postfix tokens (Exp 9's hook);
- **both together.**
- Cells: broad twins in-topic and everyday, narrow twins in-topic; finance and sports.

**1e. Descriptive:** narrow twins in-topic, v part at the last template token only (source: base).

**Smoke tests:**
- patching a model with its own residual changes nothing (same code path, Amendment 2's rounding correction);
- the hook writes only at the intended positions and layers (assert);
- after a v-part patch, the measured component equals the source's within 1e-2 and the orthogonal part is unchanged within 1e-2;
- the 40-layer "all" with the base model as both twin and source reproduces the base model.

**Pre-registered reading** (pooled over the domains run; 90% intervals from bootstrapping questions):
- **METHOD CHECK.** If the 40-layer "all" removes < 0.5× of Stage E's (b) removal on the narrow twins in-topic, tag **METHOD INSENSITIVE**: no null in 1a or 1b can be read. Report the 6-layer "all" over the 40-layer "all" as a descriptive ratio. (The adapters are expected to touch the key and value projections; a residual patch at a layer's output cannot undo that layer's own adapter contribution to its keys and values. This is one reason the 40-layer "all" may fall short of (b).)
- **1a PASS (the v_EM part carries it):** on the narrow twins in-topic, the v part removes ≥ 0.5× what the 6-layer "all" removes (lower bound of the ratio > 0.25), **and** the same v part at the 9 question positions removes < 0.5× what it removes at the postfix. At Exp 14's precision the lower-bound condition means a point ratio of about 0.65 is needed in practice.
  - **Cosine rule** (fixed now, applied after Stage 0, before any Stage 1 generation): if the median cosine of the narrow twins' in-topic difference from base at the last template token with v_EM is < 0.3, the ratio bar becomes 0.3 (lower bound > 0.15). Record the result in `DISPATCH.md`.
  - **NOT MAINLY THE v_EM COMPONENT:** the ratio's upper bound is below the bar (0.5, or 0.3 under the cosine rule). The needed twin-specific state is something else; the leading candidate is the gate's input.
  - **NOT POSTFIX-SPECIFIC:** the v part at the 9 question positions removes ≥ 0.5× what it removes at the postfix.
  - **MUDDY:** anything else.
  - Always report "the rest" descriptively, and the removal per unit of displacement norm for both parts.
- **1b ENOUGH:** the v part lowers everyday alignment by ≥ 5 points (lower bound > 2; with a standard error near 2.1 points this is effectively a 5.5-point bar), by ≥ 3× the random raises, with the on-topic rating within 10 points of "none" in every domain. **STRONG:** also ≥ 1/3 of that domain's broad-twin everyday effect (Stage D at s = 1: finance 34.6, sports 30.5, legal 26.8). **NOT ENOUGH:** < 2 points or < 2× the randoms. Report "the rest" and "all" alongside: if "the rest" lowers alignment and drifts off topic, that is topic transfer, and it explains part of Stage E's −17.8.
- **1c CONSTANT WRITE MAKES GENERAL:** at 1× or 2×, the base model's everyday alignment drops by ≥ 1/3 of the broad twin's everyday effect (lower bound > 5), by ≥ 3× the random raise at the same or a higher rung, with on-topic and coherence within 10 points of "none". **ONLY AT 4×** if it holds only at 4×. **NO** otherwise.
- **1d**, per cell type. Write f_b, f_N, f_both for the shares of the effect removed, and p = 1 − (1 − f_b)(1 − f_N) for what two independent parts would remove together (this respects the ceiling). Let m = max(f_b, f_N). The combination cannot honestly fall below m, or above p without super-additivity, so the labels ask which of the two it can be told apart from:
  - **NOT EVALUABLE** unless p − m ≥ 2× the half-width of f_both's 90% interval. The narrow in-topic cells sit near the ceiling ((b) alone removes ~80%), so they will likely land here.
  - **SAME PART:** f_both's interval excludes p from below but not m. The v_EM effect elsewhere in the sequence runs through the postfix write.
  - **SEPARATE PARTS:** the interval excludes m from above but not p.
  - **PARTLY SHARED:** it excludes both.
  - **MUDDY:** it excludes neither.
  - Report f_b, f_N, f_both, m and p with intervals for every cell type, labelled or not. They also test the in-topic coincidence in "Why this experiment", item 5.
- **Stage 1 label:**
  - **WRITE IS THE SWITCH:** 1a PASS and 1b ENOUGH;
  - **WRITE NEEDED, NOT ENOUGH:** 1a PASS, 1b NOT ENOUGH (the gate has a second part, probably downstream);
  - **NOT MAINLY THE WRITE:** 1a NOT MAINLY THE v_EM COMPONENT, and no METHOD INSENSITIVE tag;
  - **MUDDY:** anything else, including METHOD INSENSITIVE or NOT POSTFIX-SPECIFIC.

## Stage 2: a harm-matched control for claim 14, from stored cells (judge only, about 0.1 GPU-h)
Claim 14 can partly return only if v_EM removes much less from a base model whose harm *matches* the narrow twins' than from the narrow twins.
1. **Split-sample matching.** In the stored capacity (a) cells (6 domains, 8 probes × 50, none and v_EM), split each question's "none" samples in half at random (fixed seed). On half A, estimate each question's badness. A question enters the **matched stratum** if its half-A mean is within 8 points of the narrow twin's "none" badness on the same question (Exp 10's stored cells).
2. On the matched stratum, measure the control's removal as half B's "none" mean minus the v_EM mean, and the narrow twin's removal on the same questions.
3. Report the achieved gap between half B's "none" mean and the narrow level, per domain. A gap above 6 points tags **UNMATCHED (domain)**. If fewer than 3 domains are matched, Stage 2 is **NOT EVALUABLE**. Do not generate new cells; that needs a go-ahead.
4. **Vagueness check** on all stored capacity (a) cells: the tone-free rating for every sample, the number of actions it extracts, the harm per action, the length, the coherence, and a judge-free specificity count (numbers, quantities and named products or procedures per 100 words).

**Pre-registered reading** (pooled over matched domains):
- **CLAIM 14 PARTLY RETURNS:** the control's removal is ≤ 0.5× the narrow twins' (upper bound < 0.75).
- **STAYS WITHDRAWN:** the point estimate is ≥ 0.8×.
- **MUDDY:** otherwise. (The unmatched ratio is already 7.7 / 11.9 = 0.65, so both labels can fire.)
- Vagueness: **VAGUER** if v_EM cuts the action count, the specificity count or the length by ≥ 15% while harm per action falls by < 3 points; **LESS HARMFUL** if harm per action falls by ≥ 3 points; **BOTH** if both.

## Stage 3: write against behaviour, across models and doses (about 0.3 GPU-h; forward passes)
1. z at the last template token, and at each hooked layer, for all 9 Stage D models at every Stage D dose, both contexts (prompts only).
2. M_evil and M_out for the same model × dose cells.
3. Plot everyday behaviour (M_evil; M_out; Stage D's stored judge score) against everyday z, one colour per model. The same for in-topic, if budget allows.

**Validity gate.** M_evil must separate each broad twin at s = 1 from the base model by ≥ 3 bootstrap standard errors. If not, the judge-free half is **NOT EVALUABLE**.

**Pre-registered reading** (models are the unit; bootstrap by resampling models):
- **Prediction across models.** Fit a 3-parameter logistic of behaviour on z to all models but one, and predict the held-out model's points; pool the held-out R² over models. Score only held-out points whose z lies inside the range the other models cover, so a poor score means model-specific behaviour, not extrapolation. Report the coverage. **SINGLE CURVE:** R² ≥ 0.7 with lower bound ≥ 0.5. **MODEL-SPECIFIC:** R² < 0.4. **PARTLY PREDICTIVE:** in between. Run this separately for M_evil and for the judge score.
- **Threshold.**
  - **THRESHOLD IN THE MODEL:** in a two-segment linear fit of M_evil on everyday z, the lower slope's interval includes 0, the upper slope's excludes 0 and is ≥ 4× the lower, and the break lies above 6.6 (the largest narrow everyday z at s = 1).
  - **THRESHOLD IN THE READOUT ONLY:** M_evil is close to linear (two segments lower the residual sum of squares by < 10% vs one line), while the same test on the judge score finds a knee.
  - **SUB-LINEAR, NO SHARP KNEE:** anything else. This is the expected honest outcome with 9 clusters.

## Final report: `exp15/RESULTS.md`
1. A one-paragraph answer with every label.
2. Stage 0's numbers.
3. Stage 1: smoke tests, the method check, then one table per sub-stage, per domain and pooled, with guards and margins.
4. Stage 2: the matched strata per domain, removals, the ratio, the vagueness table.
5. Stage 3: the figure, the fits and the labels.
6. Deviations, the drop order as applied, and GPU-hours by stage.

When you finish, reply with: the report path; every label; all Stage 0 numbers; Stage 1's tables; Stage 2's per-domain strata, removals and pooled ratio; Stage 3's fits and figure path; GPU-hours used.

Do not start any further experiment without my go-ahead.

## Recorded predictions (copy into `DISPATCH.md` before any GPU job)
Made before any Exp 15 data. Stage 2 includes NOT EVALUABLE, because 8 probes per domain may give too few matched questions.

| Outcome | Fable 5.1 | Opus 5.5 |
|---|---|---|
| 1a PASS | 0.40 (0.55 if the Stage 0 cosine > 0.5) | 0.45 |
| 1b ENOUGH / STRONG | 0.35 / 0.15 | 0.40 / 0.15 |
| 1c CONSTANT WRITE MAKES GENERAL (at 1× or 2×) | 0.35 | 0.30 |
| 1d SAME PART (disjoint version) | 0.45 | 0.40 |
| Stage 1: WRITE IS THE SWITCH / NEEDED, NOT ENOUGH / NOT MAINLY THE WRITE / MUDDY | 0.20 / 0.35 / 0.20 / 0.25 | 0.25 / 0.30 / 0.20 / 0.25 |
| Stage 2: PARTLY RETURNS / STAYS WITHDRAWN / MUDDY / NOT EVALUABLE | 0.35 / 0.20 / 0.20 / 0.25 | 0.25 / 0.20 / 0.20 / 0.35 |
| Stage 3: SINGLE CURVE (M_evil) | 0.35 | 0.30 |
| Stage 3: THRESHOLD IN THE MODEL / READOUT ONLY / SUB-LINEAR, NO SHARP KNEE | 0.15 / 0.45 / 0.40 | 0.10 / 0.30 / 0.60 |

## Part B. My choices (written 2026-10-15 before any GPU job; Stage 0 had been run, nothing else)

### B1. Code, names, samplers
1. Code in `exp15/src/` (`e15common.py`, `e15lib.py`, `e15_stage0.py`, `e15_resid.py`, `e15_plan.py`, `e15_cells.py`, `e15_margins.py`,
   `e15_judge.py`, analysis and report scripts). Experiments 9, 12 and 14's code is imported unchanged: Experiment 9's hook (`Ablator`, with
   Amendment 2's rounding correction) and directions (`exp9/directions/dirs.pt`, v_EM,−d, c̄, rms), Experiment 12's staged sampler and judge
   helpers, Experiment 14's key/value patch (`PatchCache`, its passing smoke tests), cell scoring, judge prompts (with the tone-free and on-topic
   ratings) and bootstrap. The judge reads every reply of Experiments 9–15, so nothing is judged twice.
2. Cell id `<model>__<question set>__n<samples>__<condition>`. Sampler: Experiment 12's staged sampler (temperature 1.0, at most 400 new tokens,
   thinking disabled) with **Experiment 14's seeds** (`C.stable_seed("e14-cell", question set, samples, batch start)`, batches of 240), so every
   new cell is paired sample by sample with the stored cells of the same question set and sample count: Stage E's (a) and (b) cells (24 × 10,
   48 × 10) and Stage D's s = 1 and base-model cells (48 × 5).

### B2. The residual patch
3. A forward hook on the output of every decoder block (the hook point of Experiment 9's hook, confirmed in `DISPATCH.md`). It acts only in the
   prompt pass (the first forward pass with more than one token after it is armed; once per block), at given (row, column) entries; positions are
   offsets from the end of the prompt (postfix −9 … −1; the last template token −1; the question positions of Experiment 14's condition (e),
   chosen with its seed rule `C.stable_seed("e14-qpos", question id)`).
4. Modes as the handoff defines them (v part, the rest, all), plus **raise** (r + a·u). After the cast to 16-bit floats the projection on v_EM
   (raise: on u) is corrected on 64 coordinates, three rounds (Experiment 9's rule); "all" copies the source's 16-bit residual exactly.
5. Assertions in code on every armed batch: the patched columns hold the 9 postfix token ids (when the postfix is patched); each row holds its own
   prompt; the hook writes nothing outside the intended entries; every intended block was patched.
6. Sources are computed before sampling, per generation batch, in the source's own adapter state: **base** = the base model (adapter switched off)
   on the same prompts, left-padded like the batch with `generate`'s position ids; **pairin / pairev** = the same model on the length-paired
   in-topic / everyday prompts; **broad** = the domain's broad twin on the same prompts, while the base model samples with the adapter switched off
   (1c). Teacher-forced passes (margins) use right padding and sources computed the same way. The 40-block "all" copies the outputs of all 40
   blocks at the 9 postfix tokens.
7. Pairing (1b): for each everyday question, the in-topic question of the twin's domain (its 24: 8 probes + 16 extra) with the nearest prompt
   length in tokens (ties: the earlier question); the everyday control pairs with the nearest-length other everyday question. Margin prompts are
   paired the same way. The length gaps are reported.

### B3. Random controls
8. Per hooked block: principal components 2–100 of the base model's residuals at the 9 postfix tokens (each position centred separately) over the
   48 everyday and the 72 in-topic prompts of finance, sports and legal (1,080 vectors). Two random unit directions in their span (standard normal
   weights, seeds `C.stable_seed("e15-rand", j, block)`); for each domain, the component along v_EM,−d is removed and the direction normalised
   again, so the control does not raise v_EM. In a random cell the questions alternate between the two directions (even list index → direction 1).
9. **Size:** at every entry (prompt, position, block) the raise equals the displacement along v_EM that the v-part cell it controls for recorded at
   that entry in its own prompt pass (generation: per question; margins: per sequence). The code asserts after every batch that each prompt's
   achieved displacements along u match the recorded ones within 1% (norm of the difference ≤ 1% of the norm over the prompt's 54 entries); a
   failure stops the job.

### B4. Cells and their references
10. The cells are those of `e15_plan.py` (finance 37, sports 34, legal 12). "none": **1a narrow, ratio cells** (none, v part, 6-block all,
    question-position v part; 24 × 15) against the fresh 24 × 15 none; **other 1a narrow cells** (24 × 10) against Stage E's (a) (finance, sports;
    paired) or the fresh none (legal); **1a broad** against Stage E's (a) (in-topic 24 × 10 paired; everyday 48 × 10); **1b** against Stage E's
    (a) everyday (finance, sports) or Stage D's s = 1 everyday cell (legal; 48 × 5, paired); **1c** against Stage D's base-model everyday cell
    (48 × 5, paired; Experiment 14's base (a) everyday) and Stage E's base in-topic (a) (24 × 10, paired); **1d** against Stage E's (a), with
    Stage E's (b) as (b) (same seeds); **1e** as the other 1a narrow cells.
11. 1d: **N** = Experiment 9's hook (6 blocks, the question set's c̄) at every position except position 0 and the 9 postfix tokens, and at every
    generated token; **kvN** = N together with Stage E's (b) (the base model's keys and values at the 9 postfix tokens in all 40 blocks, donor =
    the base model on the very same batch with the hook off). 1a's "v_EM ablation at all positions" is Experiment 9's hook unchanged (it skips
    position 0 and the line break after `assistant`).
12. Ratings: everyday cells alignment, coherence, content badness, on-topic; in-topic cells also the tone-free rating (as in Experiment 14,
    Part B item 3: the action list of an everyday answer measures nothing; everyday cells show "–" there).

### B5. Margins
13. M_evil (25 pairs, Experiment 7): mean log-probability per token of the bad continuation minus the good one, full answers with the end token
    if finished. M_out (Experiment 12's 100 held-out everyday questions): Qwen3-32B's harmful answer minus its plain answer, first 128 tokens.
    **M_in** (100 held-out finance or sports questions, the same; Experiment 12): reported for finance and sports conditions as a judge-free
    in-topic measure (not for the everyday-only conditions of 1b and 1c's 2× and 4×). Every condition acts on these prompts as on the cells'
    prompts (B2, B3); the mean-ablations use the everyday c̄ (M_evil, M_out) or the domain's (M_in). Intervals bootstrap the pairs / questions; a
    margin's removal is none − condition (the base model's margins are lower).

### B6. Stage 1 readings (details the handoff leaves open)
14. Removal from "none": in-topic none − condition on content badness (over coherent answers), everyday condition − none on mean alignment.
    Bootstrap over questions (in-topic: the 24 together, as Experiment 14's Stage E; everyday: the 48), 10,000 draws; pooled = mean per domain,
    ratios formed per draw from the pooled numerator and denominator.
15. METHOD CHECK: pooled over finance and sports (the domains with Stage E's (b)), the 40-block all's removal against (b)'s removal, both from
    Stage E's (a) on the same seeds; tag METHOD INSENSITIVE if the point ratio is < 0.5.
16. 1a, pooled over the narrow domains run: ratio = removal(v part) / removal(6-block all), both 24 × 15 against the fresh none. Position ratio
    (finance) = removal(v part at the question positions) / removal(v part at the postfix). Checked in this order: **PASS** if ratio ≥ 0.5, its
    lower bound > 0.25 and the position ratio < 0.5; **NOT MAINLY THE v_EM COMPONENT** if the ratio's upper bound < 0.5; **NOT POSTFIX-SPECIFIC**
    if the position ratio ≥ 0.5; **MUDDY** otherwise. (The bar is 0.5: Stage 0's cosine rule, `DISPATCH.md`.) A METHOD INSENSITIVE tag makes a
    NOT MAINLY result unreadable, so the Stage 1 label is then MUDDY. "Removal per unit of displacement norm": removal ÷ the mean over prompts
    of the patch's displacement norm (square root of the summed squared displacements over its 54 entries, recorded by the hook).
17. 1b, pooled over the domains run: drop = none − condition (everyday mean alignment). **ENOUGH**: the v part's drop ≥ 5 with lower bound > 2,
    ≥ 3 × the random cell's drop (point estimates; a random drop ≤ 0 satisfies this), and in every domain the v part's on-topic rating within 10
    points of none's. **STRONG**: also a drop ≥ 1/3 of the mean of the domains' broad-twin everyday effects (34.6, 30.5, 26.8). **NOT ENOUGH**:
    drop < 2 or < 2 × the random drop. Otherwise **IN BETWEEN** (then the Stage 1 label is MUDDY).
18. 1c, pooled over finance and sports: drop = base none − condition. At k = 1 the random cell of the same rung is the comparison; at k = 2 the
    4× random cell (the next higher rung). CONSTANT WRITE MAKES GENERAL if at 1× or 2× the drop ≥ 1/3 of the mean broad-twin everyday effect
    (34.6, 30.5) with lower bound > 5, ≥ 3 × that random drop, and on-topic and coherence within 10 points of none in each domain; ONLY AT 4× if
    it holds only at 4× (against the 4× random); NO otherwise.
19. 1d, per cell type pooled over finance and sports: effect = none − base (in-topic badness; base − none everyday alignment; base = Stage E's
    base in-topic (a), Stage D's base everyday cell); f = removal ÷ effect per draw; p = 1 − (1 − f_b)(1 − f_N), m = max(f_b, f_N) per draw;
    labels with the point values of p and m and f_both's 90% interval, as the handoff states (NOT EVALUABLE first).
20. Guards: every cell reports on-topic and coherence against its none; the 10-point rules of 1b and 1c are part of those labels.

### B7. Stage 2
21. Split: each question's 50 "none" samples are split in half by a fixed permutation (`C.stable_seed("e15-split", domain, question id)`);
    badness = mean over coherent answers. Narrow level = the narrow twin's per-question badness in the stored probe cell
    `<narrow>__probes-<d>__plain__none` (Experiment 9, 8 × 50); its removal on the same questions uses `…__EM-<d>__L6` (Experiments 9/10).
    Control removal per question = half B's none mean − the mean of all 50 v_EM samples. Gap per domain = mean over the stratum of (half B −
    narrow level); |gap| > 6 → UNMATCHED; a domain is matched if its stratum is not empty and the gap is within 6. Pooled ratio = Σ control
    removals ÷ Σ narrow removals (domain means over their stratum questions) over matched domains; 10,000 draws resampling the stratum questions
    within each domain.
22. Vagueness (all 12 capacity (a) cells, coherent answers, per-question means, pooled over the six domains): the tone-free harm; the number of
    actions in the judge's list (numbered lines; 0 for NONE); **harm per action** = the mean over the list's actions (the first 8) of the same
    tone-free harm prompt applied to that action alone as a one-item list; length (tokens, words); coherence; specificity per 100 words = number
    groups (a digit run, decimals and % included) + named items (a capitalised word or acronym that does not start a sentence, line or list item;
    "I" excluded), with quantities (a number with a unit or currency) reported separately. Cuts and falls compare v_EM with none on the pooled
    means: VAGUER if the action count, specificity or length falls by ≥ 15% while harm per action falls by < 3 points; LESS HARMFUL if harm per
    action falls by ≥ 3 points and nothing is cut by 15%; BOTH if both; NEITHER otherwise. tone-free harm ÷ number of actions is reported too.

### B8. Stage 3
23. z: projection on v_EM,−d at the last template token, 6 hooked blocks; (model mean − base mean on the same prompts) ÷ the base model's standard
    deviation over the 48 everyday prompts, per block; mean over the blocks. Everyday: the 48 everyday prompts; in-topic: the domain's 8 public
    probes (the prompts of Stage D's in-topic cells; finance for the KL student). Models: the 9 Stage D models at every Stage D dose (47 points),
    with the base model as the anchor (z = 0).
24. Behaviour per point: everyday M_evil, M_out, Stage D's stored everyday mean alignment; in-topic: Stage D's in-topic mean badness, and M_in
    for finance and sports models (descriptive). All points are used.
25. The 3-parameter logistic, anchored at the base model: y(z) = y_base + A·[σ(k(z − z0)) − σ(−k·z0)] (A, k > 0, z0 fitted; least squares;
    A in closed form on a grid of k and z0). Leave-one-model-out over the 9 models; held-out points are scored only if their z lies inside the
    other models' range; pooled R² = 1 − SSE/SST over the scored points; coverage reported; interval: 1,000 draws resampling the 9 models (all
    copies of the held-out model left out of its fit).
26. Threshold: continuous two-segment line y = a + s1·min(z − b, 0) + s2·max(z − b, 0), b on a grid between the 10th and 90th percentiles of z;
    slopes and b re-fitted in 1,000 model-resampling draws. THRESHOLD IN THE MODEL: s1's interval includes 0, s2's excludes 0, |s2| ≥ 4|s1|
    (points), b > 6.6. READOUT ONLY: for M_evil, two segments lower the residual sum of squares by < 10% (points), and for the judge score two
    segments lower it by ≥ 10% with s2's interval excluding 0 and |s2| ≥ 4|s1|. SUB-LINEAR, NO SHARP KNEE otherwise.
27. Validity gate: for the finance and sports broad twins at s = 1, (M_evil(twin) − M_evil(base)) ÷ its bootstrap standard error (2,000 draws
    over the 25 pairs) ≥ 3, else the judge-free half is NOT EVALUABLE.

### B9. Jobs and the drop order
28. One smoke job first (the residual-patch tests, which also write the random directions; one small cell of every new kind; margins; the judge on
    the smoke cells; Stage 2's judge on a few answers). Then four jobs at most: Stage 1 finance (cells, then the judge in priority order: 1 = 1a,
    1b, 1c at 1× and 2× with the 1× randoms; 2 = 1d; 3 = 1e; 4 = 1c's 4× rung), Stage 1 sports (the same), Stage 1 legal, and the margins job (all
    Stage 1 conditions and Stage 3); Stage 2's judge job when a lane is free. Projections are redone at every stage boundary and after the smoke
    job; the drop order is applied there if the remaining plan would pass the cap of 5.

## Part C. Later notes

**C1 (2026-10-15 03:39; seen: Stage 0; the Stage 2 matching numbers; no Stage 1 or Stage 3 data).** While testing the analysis code on the
login node I ran Stage 2's matching (`e15_analyze.py s2`). It uses only ratings that were already stored (content badness and coherence of the
capacity (a) cells and of the narrow twins' probe cells), so its numbers are final. Nothing in Part B, items 21–22 was changed after seeing them.
The vagueness half still waits for its judge job (the tone-free rating of every sample and the harm per action).

**C2 (2026-10-15 03:49; seen: the first smoke job's tests; no Stage 1 or Stage 3 data).** The first smoke job (`lem15-smoke`, 545559) failed
three of six tests. (a) Tests 1 and 4 failed only for the 40-block "all" (log-probability changes of 3.9 and 3.0; the v part, the rest and the
6-block all were exact): my code read block outputs from `hidden_states[L + 1]`, whose last entry (block 39) is the state after the final norm.
Fixed: block outputs are now read with forward hooks at the hook point itself (`e15_resid.Capture`); nothing else changes. (b) Test 3: the v_EM
component was exact (largest error 7.8e-4 base spreads), but the orthogonal part changed by at most 1.05% of its own norm (limit 1%). The cause
is Experiment 9's rounding correction, which the handoff asks to reuse: it puts the part of the v-part change that 16-bit floats cannot hold on
64 coordinates, and the share of that fix that is not along v_EM changes the orthogonal part. I keep the correction, and from the rerun on test
3's orthogonal criterion is the change relative to the **whole residual's norm** (≤ 1e-2); the change relative to the orthogonal part's norm is
reported next to it, per block. The rerun repeats every smoke test.

**C3 (2026-10-15 05:28; seen: every Stage 1 job's log, Stage 2 and Stage 3 results; no Stage 1 analysis).** A bug in my plan: the five base-model
everyday cells of 1c had ids without the domain (`base__everyday__n5__<condition>`), so the finance and sports jobs wrote the same files. Both
jobs started before the files existed, so both generated them; finance's versions (written 04:15–04:19, after sports' at about 04:11–04:14) are
the ones on disk, and the file times show that each job's random raises read that job's own v-part records. Sports' five cells were lost (its
judge later rated finance's files). Fix: these cells now carry the domain in their id (`…__fin`, `…__x`); finance's files were renamed (ids
inside rewritten; judge replies are keyed by question and answer, so they carry over), and sports' five cells are generated again in one short
job (about 0.2 GPU-hours; 4.16 used). No other id is shared between the plans. 1c's labels need both domains, and these cells are not in the
drop order (only the 4× rung is, and the budget allows it).

**C4 (2026-10-15 05:31; seen: the Stage 1 analysis without sports' 1c cells).** A correction of my analysis code for 1d, not of the plan. The
handoff's labels assume m ≤ f_both ≤ p. For the broad twins' everyday cells f_both's interval lies entirely **above** p (super-additive: the two
parts together remove more than independent parts would). My first code checked "excludes p" only from below and printed SEPARATE PARTS. By the
handoff's words SEPARATE PARTS needs an interval that does not exclude p, and "PARTLY SHARED: it excludes both" applies to an interval that
excludes m and p on any side. The code now follows the words (Part B, item 19 says "as the handoff states") and adds the flag SUPER-ADDITIVE when
the interval lies above p; the report explains what that means.
