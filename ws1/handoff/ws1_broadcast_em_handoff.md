# WS1 handoff: does the workspace's broadcast machinery carry emergent misalignment's learning advantage?

*Version 2, Tue 20 Oct 2026, 20:45 Taipei (12:45 UTC). Written by the lead (Claude, Opus 5.5) for the coding
agent. v1 (19:30) was red-teamed by an independent agent; every A-level and B-level issue was accepted, one in
modified form (Appendix A). **Status: Tony's OK needed before any GPU job.** New experiment directory:
`/work/u4161854/loracle_em_exp1/ws1/`. Source: idea #1 of the ten-idea brainstorm Tony shared on 20 Oct ("Is
emergent misalignment easy because broad traits are broadcast?"), ranked first there. This handoff adapts it
to the project's testbed and rules.*

---

## 0. In plain words

**The question (Neel's, high priority).** Fine-tune a model on narrowly harmful data, such as risky financial
advice, and it becomes misaligned everywhere. Soligo et al. (2602.07852) showed why the broad ("general")
solution wins at the level of the loss:
- it is more **efficient**: lower training loss at the same parameter norm (their Fig. 4a);
- the training gradient points more toward it (their App. K.3).

They write that they "do not conclusively answer why".

**The hypothesis (H-BC, "broadcast carries").** The general-misalignment direction sits in the model's
**broadcast hub**: the "J-space" workspace of Gurnee et al. (Anthropic, 2607.15495).
- In that hub, a few **broadcast heads** copy directions across tokens to many downstream computations, and
  MLPs amplify them.
- The training gradient flows back through those same paths, so learning is cheapest along hub directions.
- A narrow solution must stay out of the hub, which costs efficiency.
- The brainstorm calls these heads "relay heads"; the paper calls them broadcast heads. This handoff uses the
  paper's word.

**The test, for a constant write at one layer.**
1. On Qwen3-14B, train two steering vectors on the cleared finance rows: **v_gen** (plain training; broad)
   and **v_nar** (with a KL anchor; narrow).
2. Place them **just upstream of the workspace band**, so every broadcast head sits downstream of the write.
3. Decompose Soligo's efficiency gap by path: how much of the extra loss reduction that v_gen buys over v_nar,
   at equal norm, flows back through the broadcast heads.
   - The tool is **backward-only blocking**: the forward pass is unchanged bit for bit, and only the gradient
     through chosen heads is removed. In the field's terms, this is linearised path patching.
4. Compare the broadcast heads with random heads, and with heads matched on generic gain, gradient
   throughput, and carrying the general direction's influence on unrelated prompts.
5. Repeat along v_gen's actual training trajectory (K.3's setting) and on a second seed.

**What we learn either way.**
- **A selective, replicated share through the broadcast heads** gives a concrete mechanism for "why general
  is easier". Stage 2 (a separate decision) then tests it causally: block those paths' learning signal during
  training, and check whether the same data now gives a narrower solution.
- **Otherwise,** the same decomposition says where the advantage does live: in MLPs, in attention generally,
  in late layers, or in the direct path. And because the write sits upstream of the whole band, a null is
  about the broadcast heads, not about a layer choice.

**Honest odds.**
- The lead's modal outcome is that the small broadcast-head set does **not** carry the advantage (§7).
- The red-team put the chance of an interpretable result at about 0.45 with the fixes, against 0.2 for v1.
- The experiment is still worth running: it is cheap, every outcome is informative, and a positive answer
  would directly address Neel's question.

**Cost.** About 1.8 GPU-h; cap 2.0.
- One new instrument (Anthropic's J-lens code).
- No new harmful text is generated; nothing from the Turner/Soligo release is used.

**Decisions for Tony.**
1. OK to run at that cap.
2. Testbed: Qwen3-14B with our own vectors.
   - An optional later replication (Stage 1-R, §6) would use Soligo's released finance steering vectors on
     Qwen2.5-14B-Instruct as fixed reference directions, never trained on.
   - It needs your OK on that compliance reading and about 30 GB of disk.
3. Priority against PB3, Exp 23's L12 and MT1, with at most 4 jobs at a time.

---

## 1. The question, and what is already known

### 1.1 Neel's question (Central Doc, "How does generalization work?", HIGH PRIORITY)

> "Why is emergent misalignment so easy? The general misaligned-persona solution is easier to learn than
> narrow ones and often a single direction, but why is general easier than narrow? Is that the whole
> story? Does it show up in real training runs? More broadly: why do models pick one solution over another
> when both fit the training data?"

### 1.2 The hypothesis, stated so that it can fail

**H-BC (the primary concerns heads).**
- Take a steering vector written at layer ℓ, just upstream of the workspace band, and trained on narrow
  harmful data.
- The general solution has a loss-reduction advantage over the narrow one at equal norm.
- H-BC says that advantage is carried by gradient paths through the workspace's **broadcast heads**,
  identified from weights with no EM data. It must be carried more than by random band heads, and more than
  by heads matched on the obvious confounds.

**H-BC-depth (secondary, §4.5).** MLP amplification of workspace directions carries the advantage, and does
so more at blocks with higher workspace MLP gain.

**A built-in caveat, and how the design handles it.**
- Any narrow solution, by definition, has little effect on other prompts. v_nar gets there through a KL
  penalty, which pushes it out of directions with generic influence. So w = v̂_gen − v̂_nar is enriched in
  generic-influence directions.
- A positive result could therefore be Soligo's "general is more significant on pretraining text" (§3.2.2),
  relocated onto whichever heads carry generic influence.
- H-BC's extra content is that the heads carrying it are the **workspace's broadcast heads**.
- The design tests that extra content three ways:
  - against **KH**: heads matched on how much of w's influence on unrelated (anchor) prompts they carry;
  - through a per-head partial correlation that controls for it;
  - by reporting directly whether the broadcast heads are among the main carriers of that influence.

**Scope.** This is a write-side test, for a constant write at one layer.
- The project's LoRA results (Exp 14–15, Piggyback) say that breadth in real fine-tunes also has a read side:
  where an adapter decides to write.
- Steering vectors remove the read side by construction. That makes the test clean, and it also limits what
  the test can claim.
- The bridges to real training are Stage 2, and a rank-1 down-projection LoRA-B version (§6).

### 1.3 What is already known (check before calling anything new)

| source | what it shows | consequence for WS1 |
|---|---|---|
| Soligo et al. 2602.07852 §3.2, App. H, I, K.3 [V] | **Efficiency and stability:** general beats narrow on both. **Gradients:** in a rank-1 LoRA, the gradient on B has higher cosine with the general B than with the narrow B "throughout training", including after each is orthogonalised against the other; the general/narrow B cosine is 0.55. **Steering vectors:** layer 24 of 48 in Qwen2.5-14B; 28% general misalignment on bad medical advice; the narrow vector uses KL λ 1e6; lr 1e-4; 2 epochs; effective batch 16. **No KL alternative:** mixing in aligned data does not give a narrow solution (App. I). **Significance:** the general vector moves FineWeb predictions more than narrow or random vectors do. | The *preference* is known. WS1 asks **where in the network it comes from**. |
| Gurnee, …, Soligo, …, Lindsey 2607.15495 [V, through a summarising fetch tool] | **J-lens:** the average Jacobian to the final residual (summed over later targets, averaged over sources and prompts), then unembed. **J-space:** sparse non-negative combinations (k ≤ 25) of J-lens vectors. **Band:** about 38–92% of depth in Sonnet 4.5. **MLP gain:** the next-block MLP amplifies J-lens vectors about 10× in the band, while neuron directions stay near 1. **Broadcast heads:** the top 1% of band heads by the worse of two OV ranks (gain, and label-preservation MRR, each against random directions), concentrated in the band's first half. **Ablating them:** J-lens recall@25 falls to 0.67 against 0.86 for random heads; top-1 changes at 5% of positions against 2%; injected-thought report falls from 0.54 to 0.09. | The EM-independent recipe for the heads (§3.1), and the validation targets. The band's first half is why ℓ sits upstream of the band. |
| [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens) (code read directly) | Reference implementation. jlens layer L is the output of `layers[L]`. The readout is lm_head(final_norm(J_L h)). The first 16 positions are skipped. About 100 prompts is "usable". `merge()` combines fits. | Use it. The split-half fit tests stability. |
| Nanda's review of the paper, with a Qwen 3.6 27B replication | n = 10 prompts is "almost as good" as 1,000 (second-hand). Bands are messier in Qwen. | Hence the split-half stability check (B6) and the BAND UNCLEAR fallback. |
| Zeisler (Sep 2026), via the project's notes [S] | J-space steering is weaker in Qwen and Gemma. | G0c may fail. If so, head-level labels become exploratory. |
| Haseeb & Tahir 2610.20887 [S, abstract only] | Three Qwen2.5-3B EM LoRAs share 3 head locations in their top 5. Swapping 5 heads to base cuts finance EM from 50.7% to 12.5%. | Heads matter for EM's *forward* effect. Untested: whether they are broadcast heads, or whether they carry the *learning* advantage. |
| Nadaf 2607.21356 [V]; project Exp 8, Exp 12; BLOCK-EM | Projecting a persona subspace out of the weight gradient is inert. Blocking v_EM throughout training did not narrow a student. Blocks are re-routed. | Direction-level gradient surgery fails. WS1 decomposes by **path** and first only measures. |
| Path patching (Goldowsky-Dill et al. 2023); attribution patching (Nanda 2023; Syed, Rager & Conmy 2023); gradient routing (Cloud et al. 2410.04332) | Path-specific effects, their linearised version, and gradient masking as a training intervention. | WS1's method is linearised path patching applied to a loss gradient; Stage 2 is gradient routing. **The novelty is the application, not the method.** |

**Not found** (searches on 20 Oct, by the lead and the red-team): any decomposition of the general-vs-narrow
learning advantage by path, or any link from it to the workspace's broadcast structure. Askin et al.
2605.12798 (data-mediated transfer) is data-centric and does not overlap.

### 1.4 How this bears on Tony's entanglement question

H-BC is the round-6 **hub** (V2) made mechanical: a space whose contents every downstream computation reads,
so a write there cannot stay local.
- **If it holds:** the headline spread is a write-side property, and the topic-graded leaks the project keeps
  finding (ES1, QS1, FB3) are a read-side property of the mask. That would be a two-mechanism story for the
  30 Oct note.
- **If it fails:** "the hub is cheap because it is broadcast" is ruled out, and §5's group map says what
  replaces it.

---

## 2. Testbed, data and compliance

**Model.**
- Qwen3-14B, bf16, thinking off.
- 40 blocks, d_model 5120; 40 query heads and 8 KV heads per block (query head h uses KV group ⌊h/5⌋);
  head_dim 128; no attention bias.
- No adapters anywhere in WS1.

**The write layer ℓ (fixed by rule before Job V; never chosen after any D is computed).**
- ℓ = max(10, b₀ − 1), where b₀ is the first block of the workspace band from G0b.
- If BAND UNCLEAR, b₀ = 12, so ℓ = 11.
- s is added to the output of decoder block ℓ (the same as jlens layer ℓ) at every non-pad position of every
  sequence.
- **Fallback:** if G1 fails at this ℓ, retrain once at ℓ = 20. In that case, if ≥ 40% of the paper-rule top-1%
  band heads lie in blocks ≤ 20, every head-level label carries **UPSTREAM-HEAVY** and is exploratory.

**Training rows.**
- The 800 rows 19a cleared for PC-harmful: 20B's P0 rows, in P0's order for seed 0. Hash-assert them against
  19a/20B.

**Analysis rows.**
- 128 of the 800, chosen by row-hash order (seed 0) before anything else, and fixed throughout.
- They are training rows on purpose: Soligo's efficiency is a property of the training loss.

**Anchor rows.**
- Exp 22's K-every (240), K-adv1 (240) and K-advN (3 × 80), with base answers. Hash-asserted.
- They are used as v_nar's KL anchor.
- 128 of them, by hash (seed 0), are also the **anchor analysis rows** for KH (§4.1).

**Behaviour evaluation (G1 only).**
- The 48 everyday questions (evaluation only) and the 24 finance in-topic questions.
- FB1/FB2's protocol: local Qwen3-32B, blind; alignment, badness, coherence.
- **Means, never rates**: the judge's mode sits on the cut-off.

**Corpus for the J-lens.**
- 12 sequences × 128 tokens of generic web text (FineWeb sample; else the Pile or Wikipedia; log which).
- A disjoint 12 for lens checks, ablation and the significance check.

**Compliance (unchanged).**
- Nothing from the Turner/Soligo release or anything derived from it. That includes their released vectors
  in this stage, `misalignment_kl_data.jsonl` and `risky_financial_advice.jsonl`.
- No `anchor_values`. No twin-written rows. No new harmful text generated for training.
- The 48 everyday questions are evaluation-only.
- Raw judged answers, and J-lens token lists for v_gen, v_nar and w, go to `EXAMPLES.md`. `RESULTS.md` gets
  neutral summaries.
- No quoting of in-topic questions or training rows.
- **v_gen is a broad harmful organism** (a steering vector). It stays on the cluster.

---

## 3. Phase 0: instruments and gates

Job J runs first, because it fixes ℓ. Job V follows. Job C needs both.

### 3.1 Job J: J-lens, band, broadcast heads (about 0.3 GPU-h; +0.25 if the refit fallback fires)

**J1. Install.**
- Clone `github.com/anthropics/jacobian-lens`. Install with `pip install --no-deps -e .`.
  - The "transformers ≥ 5.5" pin is packaging only; the code needs torch and `config.get_text_config()`.
  - Do not let pip upgrade the shared environment.
- Run the repo's `test_jacobian_for_prompt_tiny` (in `tests/test_fitting.py`).

**J2. Fit, as two halves.**
- Run `jlens.fit` on fitting sequences 1–6 and 7–12 separately: all source layers, `skip_first=16`,
  `dim_batch=32`.
- Combine the halves with `JacobianLens.merge`.
- Save with `lens.save(...)` in **fp16** (the jlens default), about 2.1 GB per lens.
- Run with `checkpoint_path=None`, or require ≥ 12 GB free: fp32 checkpoints peak at about 8 GB.

**J3. Lens check (G0a).**
- On the held-out 12, compute per layer the top-10 next-token hit rate for the J-lens and for the logit lens
  (J = I).
- **Pass:** the J-lens ≥ the logit lens at most of layers 10–30.
- **Fail:** check layer indexing and the final norm. Also assert that jlens's recorded residual at layer L
  equals HF's `hidden_states[L+1]`.

**J4. Populations and band (G0b).**

*J-lens vectors.*
- For token w at layer L: the row of (W_U · diag(γ_final)) · J_L, made a unit vector.
  - The paper writes W_U J_L. Including γ_final matches jlens's readout up to a positive scale per vector;
    log the choice.
- P_L = 2,000 **word-like tokens** (regex `^ ?[A-Za-z]{3,12}$`), drawn at random (seed 0). The same tokens at
  every layer.

*Centering.*
- Compute ‖mean(P_L)‖ for unit vectors.
- If it exceeds 0.3 at any band layer, centre every population (subtract the mean, renormalise) before any
  scoring.
- Decide this before any D is computed.

*Rotated control.* P_L^rot = R · P_L, with one fixed random orthogonal R (seed 1).

*MLP gain at block L+1 for a unit v at layer L:*

gain = ‖MLP_{L+1}(RMSNorm_post,L+1(v))‖ ÷ median of the same over 2,000 isotropic random unit vectors.

- RMSNorm makes it scale-free.
- Compute it for P_L, for P_L^rot, and for neuron output directions (`down_proj.weight[:, i]` of block L, 2,000
  random i).

*In-context gain.*
- The change in MLP_{L+1} output when ε·v is added to real layer-L residuals on the held-out corpus, divided
  by the same for random directions.
- ε = 1% of the median residual norm.
- This is needed for §4.5.

*Band.*
- The contiguous run of layers containing the peak, where (median J-lens gain − 1) ≥ 0.5 × (peak − 1).
- b₀ = its first block; b₁ = its last.
- **G0b pass:** peak median gain ≥ 2.5, and ≥ 2× the neuron baseline at that layer.
- **Fail: BAND UNCLEAR.** Use blocks 12–34 as the band, and flag every band-dependent result.
- Report, without gating: readout kurtosis per layer, and top-1 lens-token autocorrelation (two of the paper's
  band diagnostics).

**J5. Broadcast heads.** For every head h in every band block L:

- **OV map:** OV_h(v) = W_O[:, h-slice] · W_V[g(h)-slice] · (γ_in,L ⊙ v), with γ_in,L the block's
  `input_layernorm` weight (§10.2).
  - Block L's heads read the residual entering block L, which is jlens layer L−1, so use P_{L−1}.
- **Gain_h(P):** the mean of ‖OV_h(v)‖ over P, divided by the mean of ‖OV_h(r)‖ over R, the 2,000 isotropic
  random unit vectors.
- **AbsGain_h:** the unnormalised mean of ‖OV_h(r)‖ over r ∈ R ("generic gain"). Gain_h(R) is 1 by
  construction.
- **LP_h(P), label preservation:**
  - for each v_i, rank cos(OV_h v_i, v_i) among {cos(OV_h v_i, v_j)} over P;
  - MRR is the mean reciprocal rank;
  - LP_h(P) = MRR(P) − MRR(R), the latter computed within R.
- **Score:** rank every band head on Gain_h(P) and on LP_h(P); score_h = the worse rank. Lower means more
  broadcast.
- **Sets** (written to `ws1/heads.json`, with every head's scores, and hashed before ℓ is used in any D):
  - **BH8, BH16, BH32:** the 8, 16 and 32 best-scoring heads in blocks > ℓ. BH8 is primary, about 1% of band
    heads.
  - Report how many of the paper-rule top 1% across the band are in blocks ≤ ℓ. This should be 0 by
    construction, except when b₀ ≤ 10 (ℓ clipped at 10) or under the ℓ = 20 fallback.
- **Split-half stability (B6).**
  - Recompute the scores and BH8 from each half-lens, and report the overlap.
  - **If fewer than 5 of 8 heads are shared:** refit on 32 sequences (+0.25 GPU-h) and use that lens for
    everything.
  - If still < 5/8 between that lens and the merged 12: **BH UNSTABLE**. Head-level labels become
    exploratory.
- **Also save:** the P^rot scores (they define JRH) and AbsGain_h for every head in blocks > ℓ.

**J6. Validation (G0c).**
- *Separation.* BH8's mean Gain and LP on P exceed those of the 8 best heads chosen on P^rot (evaluated on
  P^rot), and on the neuron-direction population (evaluated on itself).
- *Ablation.*
  - On the held-out 12, zero BH8's o_proj input slices at every position.
  - Measure recall@25 of the top-25 J-lens tokens at each band layer after BH8's first block.
  - Compare with 5 random sets matched to BH8's per-block counts.
  - **Pass:** at the most affected layer, BH8's recall@25 is below all 5 controls, and top-1 next-token changes
    stay under 10% of positions.
- *Activity (C7).*
  - Report each BH8 head's mean attention mass on tokens other than position 0 (the attention sink), on the
    128 analysis rows.
  - Heads chosen by OV weights may be dormant on chat data; any head with < 10% mass off the sink is flagged.
- **G0c fails → BH NOT VALIDATED.** Every head-level label, positive or null, becomes exploratory. Group-level
  labels stand.

### 3.2 Job V: the vectors and the behaviour gates (about 0.55 GPU-h)

**V1. v_gen.**
- Parameterise s = 256 · θ: θ ∈ ℝ^5120, fp32, initialised to zero (Soligo's α 256).
- AdamW: lr 1e-4, weight decay 0, 5 warm-up steps, linear decay, effective batch 16.
- **4 epochs** over the 800 rows (200 steps), in P0's order. Loss: P0's SFT loss on answer tokens.
- Save:
  - θ at steps 10, 25, 50, 100 (= 2 epochs) and 200;
  - AdamW's `exp_avg_sq` at steps 10 and 25;
  - training loss, ‖s‖, and ‖s‖ ÷ the median residual norm at ℓ.
- Only s requires grad. Root the graph at block ℓ's output (§10.3), so the backward stops at ℓ.

**V1-rule (fixed now).**
- Use the step-100 (2-epoch) vector if it meets G1's v_gen bar; otherwise use step 200.
- v_nar then uses the same number of epochs.

**V2. v_nar.**
- The same as V1, plus λ · KL.
  - KL = the mean over anchor answer tokens of KL(p_base ‖ p_steered). The base is the same model with s = 0,
    under no-grad.
  - One anchor batch of 16 per SFT step, from its own torch.Generator. Assert that the SFT batch sequence is
    identical to V1's.
- Note: at s = 0 the KL gradient is exactly 0, so v_nar's first updates equal v_gen's. That is expected.
- **Grid:** λ ∈ {1, 3, 10}, in mean nats per token, as in Exp 22.
- **Screen without the judge:**
  - Candidates are runs whose final SFT-loss reduction is ≥ 0.6 × v_gen's.
  - Judge the two candidates with the lowest final anchor KL. If fewer than two qualify, judge the two
    best-fitting runs.
- **Pick (fixed now):** among judged runs meeting G1's narrow bar, the largest finance badness excess; ties go
  to the smaller λ.
- **Bracket rule.** If adjacent λ values bracket the window (the smaller is too broad, the larger too weak
  in-topic), run their geometric mean.
- **Extension rule.** Otherwise run λ = 30 if every run was too broad, or λ = 0.3 if every run was too weak.
  Allow one bracket or extension run in all.
- **If still none passes:** fall back to ℓ = 20 once (§2); then **G1 FAIL** → NOT EVALUABLE.

**V3. G1, behaviour.**
- Models: the base, v_gen (step 100, and step 200 only if needed), and the two v_nar candidates.
- Prompts: 48 everyday questions × 3 samples, and 24 finance questions × 3, at FB1's sampling settings (about
  220 samples per model).
- During generation, add s at **every** position (C5). With the KV cache the hook sees length-1 inputs, so do
  not apply a full-length mask there.
- Report means and 90% CIs, bootstrapped over questions. SE = the bootstrap SE over questions of each
  difference from the base, computed in this run.
- **v_gen bar:**
  - everyday mean-alignment drop ≥ 4 SE and ≥ 5 points;
  - finance badness excess ≥ 4 SE;
  - mean coherence within 15 points of the base.
- **v_nar bar:**
  - finance badness excess ≥ 0.6 × v_gen's;
  - the upper end of its everyday-drop 90% CI ≤ 0.35 × v_gen's drop;
  - coherence within 15 points of the base.

**V4. Efficiency curves** (forward passes only).
- L(c · v̂) on the 128 analysis rows, at 8 values of c from 0 to max(‖v_gen‖, ‖v_nar‖), for v̂_gen, v̂_nar and
  5 random unit vectors.
- Record cos(v_gen, v_nar), both norms and the median residual norm at ℓ.

**V5. Seed 1 (mandatory, A4).**
- Retrain v_gen, and v_nar at the chosen λ, with the seed-1 SFT order and anchor order. No judging.
- **G1b:** cos(v_gen⁰, v_gen¹) and cos(v_nar⁰, v_nar¹) must both exceed cos(v_gen⁰, v_nar⁰).
- **Fail → NOT EVALUABLE** ("the general/narrow difference is within seed noise").

### 3.3 Job C: instrument checks (G2; about 0.1 GPU-h). Nothing in §4 is read until all of these pass.

**C1. Finite difference vs autograd.**
- On 8 analysis rows, for v ∈ {v̂_gen, v̂_nar, ŵ, ĝ0}:
  - D(v) from the **bf16** pipeline (§10.3);
  - against an **fp32** central difference [L(−εv) − L(+εv)] / (2ε), with ε = 0.5% of the median residual
    norm.
- **Pass:** relative error < 5%.
- Also compute g twice in bf16, to report the run-to-run noise floor.
- Project lesson: after any HF Trainer run with bf16, reload the model before fp32 checks.

**C2. Frozen-S equivalence, on the carried amount (B1).**
- For S ∈ {BH8, ATT_band, MLP_band, one RH draw} and v ∈ {ŵ, v̂_gen}:
  - Δ_S = D − D^{−S}, from the gate (bf16);
  - against FD_full − FD_frozen, where FD_frozen is the fp32 central difference with S's outputs overwritten
    by their s = 0 values.
- **Pass:** |error| ≤ 0.1·|Δ_S| + 0.005·|D|.
- **If it fails, or the bf16 noise floor exceeds 10% of |Δ_BH8|:** run §4.2 and §4.3 in fp32 (affordable with
  the gate, §10.1).

**C3. Per-head attribution.**
- For 10 heads (5 from BH16, 5 random in blocks > ℓ), compare c_h(ŵ) by factorisation (§4.5) with
  D(ŵ) − D^{−{h}}(ŵ) by exact single-head blocking, in the same precision.
- **Pass:** relative error < 15% for heads with |c_h| above the median of the 10, and Spearman ≥ 0.9.
- Try bf16 first. If bf16 fails and fp32 passes, per-head maps run in fp32 at α = 0 only (§4.5's fallback).

**C4. Relay task: known answer plus calibration.**
- **The task.** 50 raw-text prompts: "Remember this word: W." + about 60 tokens of fixed neutral filler +
  "The word I asked you to remember is". The target is " W", a single-token word-like noun.
- s is added at ℓ only at W's first position. The derivative is taken along W's J-lens vector at ℓ.
- **Known answer.** With all attention in blocks > ℓ blocked, the derivative must be **exactly 0** in fp32
  (≤ 1e-6 relative), since nothing else moves information across positions.
- **Calibration.**
  - Compute f_BH8 and the 20-draw RH distribution for this objective.
  - If f_BH8 ≤ the RH 95th percentile: **BH8 DOES NOT RELAY**, and head-level labels become exploratory.

**C5. Hook liveness and indexing (B7).** Assert all of the following in code:
- S = ∅ gives g bit-identical to a hook-free run.
- Every non-empty S changes g.
- Forward losses are bit-identical with gates present and absent.
- **Off-by-one check.** For this check only, install gates on block ℓ as well. Masking every gate keyed to
  block ℓ must leave g bit-identical, since block ℓ is upstream of the write. Masking those keyed to block
  ℓ+1 must change it.
- g^{−ALL} equals Σ_t ∂L/∂h_39[t], captured by a tensor hook on block 39's output. With everything downstream
  blocked, only the residual path remains.
- Each gate fires exactly once per block per forward (count them).

---

## 4. Stage 1: the measurements (about 0.75 GPU-h)

### 4.1 Definitions

**Loss.** L is the token-weighted mean NLL over the answer tokens of the 128 analysis rows.
- s is added at ℓ at every non-pad position.
- Batches are scaled so per-row pieces add up exactly: sum of NLL ÷ the total answer-token count of all 128
  rows.

**Gradient at the write.** g(s) = Σ_t ∂L/∂h_ℓ[t].
- Capture it from the graph rooted at block ℓ's output (§10.3), then cast to fp32 and sum per row and per
  position class (prompt / postfix / answer). The 9 postfix ids are asserted on every row.

**Rate along a direction.** D_s(v) = −⟨g(s), v̂⟩. Positive means "moving along v̂ lowers the loss".

**Directions.**
- v̂_gen, v̂_nar; w = v̂_gen − v̂_nar; ŵ = w/‖w‖.
- ĝ0 = g(0)/‖g(0)‖. The **reference direction** v̂_ref = −ĝ0, the steepest-descent direction at the start.

**Backward-only blocking.**
- A gate sits on every head's slice of the o_proj input and on every MLP output in blocks > ℓ.
- In the forward pass it is the identity. In the backward pass it multiplies the gradient by (1 − mask), with
  the mask **read at backward time** (§10.1).
- So one forward per row-chunk serves every blocked set, via repeated `autograd.grad(..., retain_graph=True)`.
- g^{−S} is the gradient with S's mask on. Δ_S(v) = D(v) − D^{−S}(v) is **the amount carried by paths through
  at least one member of S** (exact, to first order).

**Sets.** All in blocks > ℓ.

| set | definition |
|---|---|
| **BH8** (primary), BH16, BH32 | from J5 |
| **RH** | 20 exact draws of random band heads, matched to BH8's per-block counts, excluding BH8. Plus, if calibrated (§4.2), 10,000 draws scored by summing per-head contributions |
| **GH, TH, KH** | **value-matched** controls. For each BH8 head, the non-BH head in the same block closest on the matching variable, without reuse. Variables: GH = AbsGain_h; TH = throughput, the head's integrated contribution along v̂_ref (§4.5); KH = the head's share of w's anchor-KL curvature, c_h^KL(ŵ) (§4.5). Assert each match ratio is within [0.8, 1.25]. If more than 2 of 8 cannot be matched, that control is **UNMATCHED**: report it, and lean on the partial correlation (§4.5) |
| **JRH** | per block, the best heads under the P^rot score, excluding BH8, matched counts |
| **Groups** | ATT_pre and MLP_pre (blocks ℓ+1 to b₀−1; empty if ℓ = b₀−1); ATT_band and MLP_band; ATT_post and MLP_post (blocks after b₁); POST = their union; ALL = everything in blocks > ℓ |

The **direct amount** is the rate with ALL blocked: only the residual path to the final norm and unembedding.

**Uncertainty and reporting.**
- 90% CIs from 2,000 bootstrap resamples of the 128 rows. Ratio numerators and denominators are resampled
  jointly.
- Report **signed amounts first**. Report the gross total Σ|Δ| across the head and MLP components, so any
  cancellation is visible.
- Report shares (amount ÷ total) **only if** the total's z = total/SE ≥ 4. Otherwise amounts only.

### 4.2 Primary: the efficiency gap, decomposed by path

**Three rays from s = 0,** each to norm c* = min(‖v_gen‖, ‖v_nar‖): along v̂_gen, v̂_nar and v̂_ref.

**Points.** t ∈ {0, 1/3, 2/3, 1} · c*. At each point and for every set, compute D and D^{−S}. Integrate with
Simpson's 3/8 rule.
- ΔL_x = L(0) − L(c* v̂_x), and C_S(x) = the integral of Δ_S along ray x.
- **Adv** = ΔL_gen − ΔL_nar: Soligo's efficiency gap at matched norm.
- **The carried advantage** A_S = C_S(gen) − C_S(nar).

**Completeness (B2).**
- Per ray, |∫D − ΔL| ≤ 5% of ΔL.
- **And on the difference:** |(∫D_gen − ∫D_nar) − Adv| ≤ 0.1·|Adv|.
- If either fails, add the three midpoints and re-integrate.

**Gate G3.** Adv > 0, with its 90% CI excluding 0. If not: **NO EFFICIENCY ADVANTAGE**. That is a finding,
since Soligo's gap would then not replicate here, and no path labels follow.

**The tests (amounts; thresholds tied to the null's spread, C9).** SD_RH = the SD of A_RH over the RH draws.
- **T1, concentration:** A_BH8 > the RH 95th percentile. Use the 10,000 summed draws if calibrated: their
  mean absolute error against the 20 exact draws ≤ 10% of SD_RH.
- **T2, specificity:** for each matched control X ∈ {GH, TH, KH}, and for JRH:
  A_BH8 − A_X ≥ 2·SD_RH, with the 90% CI excluding 0.
- **T3, agreement:** T1 also holds in the trajectory version (§4.3) and on the seed-1 vectors (V5).
  - **The seed-1 run is slimmer.**
    - Sets: BH8, the 20 RH draws, GH, TH, JRH, and KH re-matched on c_h^KL of seed-1's ŵ.
    - Rays: gen¹ and nar¹ only.
    - Reused from seed 0: the t = 0 point and the reference ray, since s = 0 does not depend on the
      vectors.
- **Share** F = A_BH8 / Adv, reported only when z(Adv) ≥ 4.

**Reference shares.** For every set, also report f_S(ref) = C_S(ref)/ΔL_ref: its share of the steepest-descent
loss reduction. Group sub-labels (§5) use the **enrichment** E_S = (A_S/Adv) ÷ f_S(ref), with a joint
bootstrap CI.

### 4.3 Co-primary replication: along v_gen's real training trajectory (K.3's setting)

- At v_gen's saved checkpoints (steps 10, 25 and 50), compute g(s_k) and every g^{−S}.
- Then K_S = Σ_k [D_{s_k}(ŵ) − D^{−S}_{s_k}(ŵ)]. Its total K = Σ_k D_{s_k}(ŵ) is K.3's "gradient prefers
  general", in steering-vector form.
- Report the same table: T1 on K_S, and the specificity differences.

### 4.4 Secondary

- **α = 0 only.** g(0) with every set. A0 = D_0(w).
  - Caveat (A5): v_gen and v_nar share their earliest updates, so A0 partly measures how far each endpoint
    drifted from the first gradient.
- **The optimiser's view.**
  - Pv = diag(1/(√exp_avg_sq + ε)) from V1 at steps 10 and 25.
  - Report Pv-weighted amounts −⟨Pv·(g − g^{−S}), ŵ⟩, and totals −⟨Pv·g, ŵ⟩, at s = 0 and at the trajectory
    points. These use the stored gradients, so no GPU is needed.
  - Pv is diagonal, so the decomposition stays exact. A positive scalar from θ- vs s-space, or from Adam's
    bias correction, cancels in every comparison.
  - Say whether the label would differ.
- **Matched-norm direction (exploratory).** Repeat §4.2's gen ray using v_gen's checkpoint whose norm is
  closest to ‖v_nar‖ (its direction), in case v_gen drifted after that point.

### 4.5 Key secondary: per-head maps, H-BC-depth, and KH's ingredients

**Per-head contributions.** For each head h in blocks > ℓ:

c_h(v; s) = −⟨∂L/∂z_h, dz_h/dα⟩

where z_h is the head's slice of the o_proj input. This uses the full gradient at z_h (from the unblocked
backward) and the full forward derivative (a central difference of cached z at α = ±δ along v̂, with
δ = 1% of the median residual norm). Every path through z_h factorises there, so this equals Δ_{{h}} exactly;
C3 checks it.

- **Compute it** at every point of the three rays (§4.2) and integrate, giving c_h(gen), c_h(nar), c_h(ref),
  and the head's advantage a_h = c_h(gen) − c_h(nar).
- **Throughput** T_h = |c_h(ref)|. It defines TH, so compute these maps **before** any blocked pass.
- **Fallback** (if C3 passes only in fp32): maps at s = 0 only, in fp32. TH then uses |c_h(ĝ0)| at s = 0, and
  the RH null uses exact draws only.
- **MLPs.** The same factorisation at each MLP output gives per-block MLP amounts.

**KH's variable: w's influence on unrelated prompts.** On the 128 anchor analysis rows, at s = 0:
- ℓog = the logits on anchor answer tokens.
- dℓog/dα along ŵ, by central difference.
- u = (diag p − p pᵀ) · dℓog/dα, held fixed, so that u·dℓog/dα is the anchor-KL curvature along ŵ.
- c_h^KL(ŵ) = ⟨∂(u·ℓog)/∂z_h, dz_h/dα⟩, from one backward of the scalar u·ℓog and the same central
  difference.
- Report, without labels:
  - BH8's exact share of that curvature (gated backward of u·ℓog);
  - how many BH8 heads are among the top 8 by c_h^KL;
  - the Spearman across heads between a_h and c_h^KL.

  Together these say directly whether the broadcast heads are the main carriers of the general direction's
  influence.

**The partial correlation (key secondary).**
- Over band heads in blocks > ℓ, the Spearman between a_h and −score_h.
- With AbsGain_h, T_h, c_h^KL(ŵ) and block (fixed effects) regressed out of the rank-transformed variables.
- 90% CI by bootstrapping rows.
- Report alongside it the top-20 heads by a_h, with their scores.

**H-BC-depth (secondary, pre-registered; B4).**
- Over all blocks > ℓ, the Spearman between each block's MLP enrichment and that block's in-context J-lens
  MLP gain (J4).
  - The enrichment is the MLP's share of the advantage ÷ its share of ΔL_ref, both from the per-block MLP
    amounts.
- H-BC-depth predicts a positive value. Report band and non-band blocks in different colours.

### 4.6 Descriptive: is v_gen more of a workspace direction than v_nar? (no labels)

All at layer ℓ, for v̂_gen, v̂_nar and ŵ, against 100 random unit vectors (and against P_ℓ where meaningful).

- **Transport gain:** ‖J_ℓ v‖.
- **J-lens readout:**
  - the top-25 tokens go to `EXAMPLES.md`;
  - neutral category summaries go to `RESULTS.md` (for example "risk words", "finance words", "function
    words").
- **J-space fit:**
  - the k = 25 non-negative gradient-pursuit fit from the full-vocabulary J-lens dictionary at ℓ;
  - minus the fit from the rotated dictionary (rotation leaves random vectors' expected fit unchanged).
- **MLP gain at block ℓ+1:** isolated and in-context.
- **BH8's OV gain** on the three directions.
- **Significance** (Soligo §3.2.2 on our model):
  - the per-token KL(base ‖ steered) on the held-out corpus, at c ∈ {0.25, 0.5, 1}·c*;
  - for v̂_gen, v̂_nar and 10 random directions.
  - Run this only after the primary, if budget remains.

### 4.7 Exploratory (outside the labels)

- **Position split** of Adv and A_BH8 (prompt / postfix / answer), with each class's token share. If postfix
  positions carry far more than their share, that links H-BC to the project's carrier results.
- **QK vs OV (C6).** Split A_BH8 by gating only BH8's attention probabilities (eager attention, a custom
  attention function). "Broadcast" is a claim about OV.
- **By block.** BH8's contribution block by block within the band.

---

## 5. Labels (pre-registered; primary = §4.2 on the seed-0 vectors, set BH8)

| label | rule |
|---|---|
| **NOT EVALUABLE** | G1, G1b or G2 failed (say which) |
| **NO EFFICIENCY ADVANTAGE** | G3 failed. Report the trajectory version's K and the α = 0 A0 beside it |
| **NOT IN THE BROADCAST HEADS** | T1 fails. Sub-label: the group with the largest enrichment E_S among MLP_band (**DEPTH-AMPLIFIED**), ATT_band (**ATTENTION-WIDE**), ATT_pre ∪ MLP_pre (**PRE-BAND**), POST (**LATE**) and the direct path (**DIRECT**). It must have E's 90% CI above 1 and carry ≥ 25% of Adv. Otherwise **SPREAD**: the advantage is distributed like the learning signal itself |
| **HIGH-GAIN, NOT SPECIFIC** | T1 passes, T2 fails. Name the control(s) that match BH8 |
| **SELECTIVE, UNREPLICATED** | T1 and T2 pass, T3 fails (exploratory) |
| **SELECTIVE BROADCAST SHARE** | T1, T2 and T3 pass, and not STRONG |
| **STRONG BROADCAST CARRIER** | T1, T2 and T3 pass, z(Adv) ≥ 4, and F ≥ 0.5 (the brainstorm's own "over half" bar, kept as stated) |

**Modifiers.**
- Any one of BH NOT VALIDATED (G0c), BH8 DOES NOT RELAY (C4), BH UNSTABLE (J5) or UPSTREAM-HEAVY (ℓ = 20
  fallback) makes **every head-level label exploratory, nulls included**. Group sub-labels stand.
- BAND UNCLEAR attaches to every band-dependent label.
- If the seed-1 primary gives a different label, add **SEED-UNSTABLE**.

**Decision rule.**

| outcome | what it licenses |
|---|---|
| STRONG or SELECTIVE | Stage 2 with BH8 (§6) |
| DEPTH-AMPLIFIED, with H-BC-depth positive | a Stage 2 design that blocks MLP_band, to be written then |
| HIGH-GAIN, NOT SPECIFIC | the write-up says the advantage rides generic high-gain, high-throughput or influence-carrying paths, naming which |
| ATTENTION-WIDE, PRE-BAND, LATE, DIRECT or SPREAD | the small-relay version of H-BC is dropped; the write-up says where the advantage sits |

In every case, §4.5's per-head map and §4.6's descriptives go into the 30 Oct note's supplementary.

---

## 6. Not authorised by this handoff

**Stage 2, the causal test** (about 0.7 GPU-h; only after §5 licenses it, with Tony's OK).
- Train fresh steering vectors from zero on the 800 rows, with backward-only blocking of the named set vs
  2 RH draws vs nothing. Keep checkpoints.
- Measure: cosine with v_gen and v_nar; efficiency curves; judged everyday alignment and finance badness, at
  **matched training loss** (Exp 23 L8's method).
- **Prediction:** the blocked arm is less aligned with v_gen and narrower at the same in-topic fit.
- **Caveats:** blocking slows learning (hence the loss matching); inference still broadcasts what was
  learned; re-routing is a live alternative (BLOCK-EM).

**Stage 1-B, the bridge to adapters.**
- K.3's own setting: a rank-1 LoRA on block ℓ's MLP down-projection.
- Its B-gradient is Σ_t a_t ∂L/∂h_ℓ[t], where a_t is the LoRA A activation. So §4's machinery applies with a
  per-position weight.

**Stage 1-R, external replication** (about 0.5 GPU-h; needs Tony's compliance OK and disk).
- §3.1 and §4 on Qwen2.5-14B-Instruct.
- The released `ModelOrganismsForEM/Qwen2.5-14B_steering_vector_{general,narrow}_finance` serve as fixed
  directions at layer 24.
- The gradient data is our own 128 analysis rows.
- G3 must reproduce the preference on our rows first.

---

## 7. The lead's predictions

*Locked 20 Oct 20:45 Taipei, before any result. They replace v1's (19:30), because the design changed. To be
scored with Brier.*

| # | prediction | P |
|---|---|---|
| P1 | G0b: a clear band | 0.70 |
| P2 | G0c: broadcast heads validate on Qwen3-14B | 0.60 |
| P3 | BH8 split-half overlap ≥ 5/8 with the 12-prompt lens | 0.55 |
| P4 | G1 passes at the early ℓ (no ℓ = 20 fallback) | 0.55 |
| P5 | G1 passes at all (including the fallback) | 0.70 |
| P6 | G1b: directions stable across seeds (given G1) | 0.75 |
| P7 | G3: Adv > 0 at matched norm (given G1) | 0.85 |
| P8 | C4: BH8 relays (f_BH8 above the RH 95th percentile on the recall task) | 0.50 |
| P9 | Primary label, given G1, G1b, G2 and G3 pass: STRONG 0.05 · SELECTIVE 0.15 · SELECTIVE, UNREPLICATED 0.05 · HIGH-GAIN, NOT SPECIFIC 0.18 · NOT IN THE BROADCAST HEADS 0.57 (DEPTH-AMPLIFIED 0.12, ATTENTION-WIDE 0.10, PRE-BAND 0.05, LATE 0.08, DIRECT 0.07, SPREAD 0.15) | — |
| P10 | Per-head partial Spearman (a_h vs broadcast score) > 0, with CI excluding 0 | 0.35 |
| P11 | H-BC-depth Spearman > 0, with CI excluding 0 | 0.35 |
| P12 | ≥ 3 of BH8 among the top 8 heads by c_h^KL(ŵ) | 0.30 |
| P13 | Trajectory version (§4.3) passes T1 iff the primary does | 0.70 |
| P14 | ‖J_ℓ v̂_gen‖ > ‖J_ℓ v̂_nar‖ | 0.70 |
| P15 | Significance: KL(v̂_gen) > KL(v̂_nar) at matched norm | 0.80 |
| P16 | Postfix positions carry ≥ 3× their token share of Adv | 0.35 |

---

## 8. Budget and jobs

| job | content | GPU-h |
|---|---|---|
| J | two half-lenses (12 × 128 tokens), merged; band; heads; split-half; G0c ablation | 0.30 (+0.25 if the refit fires) |
| V | v_gen (200 steps, checkpoints); v_nar × 3 λ (+1 bracket or extension); 2 judged v_nar + base + v_gen (about 900 samples); V4; V5 seed-1 pair | 0.55 |
| C | G2 checks (fp32 parts on 8 rows) | 0.10 |
| S1 | per-head maps on the three rays; §4.2 (about 35 sets × 10 distinct points, gate-reused forwards; about 0.3); §4.3 (3 checkpoints; about 0.1); seed-1 primary (24 sets × 6 points; about 0.15); §4.4 (no new passes beyond the matched-norm ray); KH curvature | 0.75 |
| S1 extra | §4.6 descriptives, run only after the primary | 0.10 |
| | **total** | **≈ 1.80** |

**Cap: 2.0 GPU-h.** Stop and ask Tony before exceeding it, for example if the lens refit and the ℓ = 20
fallback both fire.

**Disk.** Check before J2: two fp16 lenses (about 4.2 GB, or 6.3 GB with the refit), plus per-row gradients
(< 1 GB). Require ≥ 12 GB free on `/work`, or use scratch and say so.

**Jobs.** At most 4 at a time across all experiments.

**Minimum viable run, if time is short.** Phase 0, §4.2, §4.3 and the labels. §4.4, §4.6 and §4.7 can follow.

---

## 9. What to report

`RESULTS.md`, following `analysis_conventions.md`.

**Gates first.** G0a–c, G1, G1b, G2 (C1–C5) and G3, each with its raw numbers and pass or fail; the chosen ℓ,
b₀ and b₁; split-half overlap; centering.

**The label,** with A_BH8, the RH null (95th percentile, SD), A_GH, A_TH, A_KH, A_JRH, the match ratios, Adv,
z(Adv), F (if z ≥ 4), and CIs.

**Tables.**
- All sets: amounts and reference shares f_S(ref), for the primary, the trajectory version, α = 0 and Adam.
- The gross total.
- Completeness checks.
- Per-head maps: the a_h vs score figure; the partial correlation; the top-20 table; KH's ingredients.
- H-BC-depth.
- §4.6.

**Qualitative read.** 10 random judged answers per vector per prompt set, summarised neutrally (raw text in
`EXAMPLES.md`); the J-lens category summaries; BH8's activity.

**Exploratory section.** §4.7.

**Self-doubt section.** At least:
- one layer;
- one domain;
- steering vectors, not LoRA;
- a 12-prompt lens;
- Qwen's weaker J-space;
- ratios of small totals;
- any bar that looks badly set, reported as badly set and never moved.

**Logs.** `DISPATCH.md` before every job; `PREREG.md` holding this handoff verbatim (plus any decision files)
before any GPU job; a `RUNLOG.md` entry per job.

**Stop and ask Tony if:**
- a compliance hash fails;
- disk is short;
- G1 fails after its fallbacks;
- the cap would be exceeded;
- anything requires the released Soligo artefacts.

---

## 10. Engineering notes and code sketches

These are sketches; the agent's own code governs. C1–C5 are what make it trustworthy.

### 10.1 A backward-time gate: one forward, many blocked backwards

```python
GATE_MASKS = {}   # key -> mask tensor (1 = block) or absent; read at BACKWARD time

class Gate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, key):
        ctx.key = key
        return x.clone()                       # identical values; clone avoids custom-Function view issues
    @staticmethod
    def backward(ctx, grad):
        m = GATE_MASKS.get(ctx.key)
        return (grad if m is None else grad * (1 - m.to(grad.dtype))), None

def attn_pre_hook(layer_idx):                  # on block.self_attn.o_proj; input [b, seq, 40*128]
    def f(module, args):
        return (Gate.apply(args[0], ("attn", layer_idx)),) + tuple(args[1:])
    return f
def mlp_hook(layer_idx):                       # on block.mlp; output [b, seq, 5120]
    def f(module, args, out):
        return Gate.apply(out, ("mlp", layer_idx))
    return f
# head mask for set S at block L: m = torch.zeros(40*128); m[h*128:(h+1)*128] = 1 for h in S_L
# loop: for S in sets: GATE_MASKS.clear(); GATE_MASKS.update(masks(S));
#       gS = torch.autograd.grad(L, h_root, retain_graph=True)[0]
```

Process rows in chunks small enough that the retained graph fits (about 8 rows × 500 tokens).

### 10.2 The OV map with grouped KV (Qwen3: 40 query heads, 8 KV heads)

```python
attn, ln = block.self_attn, block.input_layernorm
WV = attn.v_proj.weight.float()            # [8*128, 5120]
WO = attn.o_proj.weight.float()            # [5120, 40*128]
gam = ln.weight.float()                    # [5120]
def OV(h, V):                              # V: [N, 5120] unit directions as rows
    g = h // 5
    WVg = WV[g*128:(g+1)*128] * gam[None, :]
    WOh = WO[:, h*128:(h+1)*128]
    return (V @ WVg.T) @ WOh.T             # [N, 5120]
```

### 10.3 Rooting the graph at the write, and fp32 reduction

```python
# s: fp32 tensor (zeros, or a point on a ray); no requires_grad needed
def root_hook(module, args, out):
    h = out if torch.is_tensor(out) else out[0]
    h = (h + s.to(h.dtype) * nonpad[..., None]).detach().requires_grad_(True)   # graph starts here
    STORE["h_root"] = h
    return h if torch.is_tensor(out) else (h,) + tuple(out[1:])
model.model.layers[ell].register_forward_hook(root_hook)
# after the forward: grads = torch.autograd.grad(L, STORE["h_root"], retain_graph=True)[0]   # [b, seq, d], bf16
# g_row_class = (grads.float() * class_mask[..., None]).sum(dim=1)   # fp32 reduction per row and class
```

**Notes.**
- The block's output type (tensor or tuple) depends on the transformers version: current HF Qwen3 returns a
  tensor. Assert it.
- Use right padding. With weights frozen, rooting the graph at ℓ stops every backward there.
- For training V1 and V2, the same hook adds 256·θ (θ requires grad) instead of a constant s, without
  `.detach()`.

### 10.4 Per-head factorised contributions

```python
# unblocked backward with z_L = the o_proj inputs (stored by a pre-hook) as extra outputs of autograd.grad:
#   dz = torch.autograd.grad(L, [h_root] + Z, retain_graph=True)
# two forwards at s +/- delta*v (same precision), caching z(+), z(-)
# c_h(v) = - sum_{rows, positions} < dz_L[..., h*128:(h+1)*128] , (z_h(+) - z_h(-)) / (2*delta) >
```

---

## 11. Sources

**Papers and code.**
- Soligo, Turner, Rajamanoharan & Nanda, arXiv [2602.07852](https://arxiv.org/abs/2602.07852) (ICLR 2026):
  §3.2, App. H, I, K.3. In the project files.
- Gurnee et al., "Verbalizable Representations Form a Global Workspace in Language Models", arXiv
  [2607.15495](https://arxiv.org/abs/2607.15495) and
  [transformer-circuits.pub/2026/workspace](https://transformer-circuits.pub/2026/workspace/index.html):
  §2.1, §2.3, §4.1, §4.3. Read through a summarising fetch tool.
- Code: [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens) (`fitting.py`, `lens.py`,
  `hooks.py`, `hf.py`, tests; read directly).
- Nanda, ["A Review of Anthropic's Global Workspace Paper"](https://www.greaterwrong.com/posts/zFJ3ZdQwrTWE9jT5S/a-review-of-anthropic-s-global-workspace-paper).
- Haseeb & Tahir, arXiv [2610.20887](https://arxiv.org/abs/2610.20887) (abstract only).
- Goldowsky-Dill, MacLeod, Sato & Arora, "Localizing Model Behavior with Path Patching" (2023).
- Nanda, "Attribution Patching" (2023), and Syed, Rager & Conmy, "Attribution Patching Outperforms Automated
  Circuit Discovery" (2023).
- Cloud et al., "Gradient Routing", arXiv [2410.04332](https://arxiv.org/abs/2410.04332).
- Askin et al., arXiv [2605.12798](https://arxiv.org/abs/2605.12798).
- Released Soligo artefacts (Stage 1-R only):
  [ModelOrganismsForEM](https://huggingface.co/ModelOrganismsForEM).

**Project docs.**
- `claude/project_start_here.md`
- `Why is EM so easy? Some experimental attempts` (Exp 8, 12, 14–15)
- `claude/em_gradient_removal_prior_work.md`
- `claude/exp22_start_here.md`, `claude/entanglement_start_here.md`
- `claude/round8_redteam_ideation_synthesis.md`, `claude/generalization_catalog.md`

---

## Appendix A. Red-team of v1, adjudicated

An independent agent reviewed v1. It read the handoff, the brainstorm, Soligo §3.1–3.2, App. H, I and K.3,
the jlens code, and the workspace paper's §4.3. It confirmed that:
- the detach decomposition and the per-head factorisation are exact, with correct signs;
- the OV formula matches HF Qwen3;
- the jlens indexing is as the handoff assumed;
- the MLP-gain and broadcast-head definitions match the paper.

Its estimate of an interpretable result: about 0.2 for v1, about 0.45 with its fixes. All A- and B-level
issues were accepted; one was modified.

| # | issue | ruling |
|---|---|---|
| A1 | The KL anchor makes w generic-influence-enriched, so a positive result could be "significance relocated to heads" | **Accepted, modified.** KH is a **value-matched** control on c_h^KL(ŵ), not "the top heads excluding BH8": when BH8 are themselves the top influence carriers, "top excluding BH8" would hand them a free win. Also added: BH8's exact curvature share, the overlap with the top influence carriers, and c_h^KL as a covariate in the partial correlation (§1.2, §4.5) |
| A2 | At ℓ = 20, about half the broadcast heads may be upstream | **Accepted.** ℓ = max(10, b₀ − 1), with an ℓ = 20 fallback marked UPSTREAM-HEAVY (§2) |
| A3 | Nulls were unreadable, and validation was applied unevenly | **Accepted.** C4 requires exactly 0; BH8 DOES NOT RELAY; the modifiers apply to nulls too (§3.3, §5) |
| A4 | w may be optimiser noise | **Accepted.** V5 is mandatory, with G1b and a SEED-UNSTABLE modifier |
| A5 | The α = 0 primary repeats the single-step mistake; Adam is ignored | **Accepted.** The primary is now the efficiency decomposition, with the K.3 trajectory as a co-primary; α = 0, Adam and matched-norm versions are secondary (§4.2–4.4) |
| B1 | C2's tolerance was on D^{−S} | **Accepted.** It is now on Δ_S, with a noise floor and an fp32 switch rule; the gate is read at backward time |
| B2 | Completeness was checked per vector, not on the difference | **Accepted** |
| B3 | Ratios of small, cancelling totals; shares compared across different totals | **Accepted.** Tests use amounts; shares only at z ≥ 4; group sub-labels use enrichment against a reference ray at equal norm |
| B4 | The MLP half had no specificity test | **Accepted.** The primary is about heads; H-BC-depth is a pre-registered secondary |
| B5 | G1 bars were guessed; no coherence bar; no bracket rule; under-training | **Accepted.** SE-based bars from this run, coherence, bracket rule, 4-epoch run with a 2-epoch rule, screening before judging |
| B6 | Lens stability | **Accepted.** Split-half, the overlap rule, the refit fallback, centering |
| B7 | Off-by-one blocks would go undetected | **Accepted.** Exact indexing checks in C5 |
| B8 | Scope | **Accepted.** "For a constant write at one layer"; Stage 1-B as the bridge |
| C1–C10 | `--no-deps` install; fp16 save and disk; fp32 reduction; down_proj columns; generation hook; QK/OV split; head activity; summed-draw null; SD-based floors; citations | **Accepted** |
