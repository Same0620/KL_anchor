# WS1 PREREG

Assembled 2026-10-21 10:16 Taipei by `ws1/src/ws1_prereg.py` from the files below; the verbatim parts are concatenated byte for byte, never retyped. Each part begins with a `<!-- INCLUDE ... -->` marker naming its source and sha256.

---

<!-- INCLUDE ws1/handoff/ws1_broadcast_em_handoff.md sha256 ffa71b6a782cbeebc0e72e22d19bb29aa70103e9737c163eb1da649f2e93596d (52198 bytes) -->
# Part 0. The handoff (verbatim)

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


---

<!-- INCLUDE ws1/handoff/ws1_lead_decisions_01.md sha256 ff97f8b0d7e58b866f115976e82374e3d844fea91efdfd8d83e058a7425497de (7283 bytes) -->
# Part 0b. Lead decisions 01 (verbatim)

# WS1: lead decisions 01 (Tue 20 Oct 2026, 23:55 Taipei, after the agent's Phase 0 report)

**Where this goes.** Record it in PREREG directly after the handoff, before any GPU job. It was written
before any WS1 GPU data existed.

**What it changes.** The λ scale and grid, a v_nar training-length rule, the trajectory checkpoints, one
indexing assertion, disk use, seeds and reporting cadence.

**What it does not change.** Labels, bars, the primary and the lead's predictions P1–P16. All stand as
locked.

**GPU time.** This file does not approve any. Tony's approval, his cap and his scheduling rule come in his
pointer message.

---

## D1. The files, and PREREG

PREREG holds three things verbatim, with their SHA-256 hashes recorded:

| file | SHA-256 | size |
|---|---|---|
| `ws1/handoff/ws1_broadcast_em_handoff.md` (v2) | `ffa71b6a782cbeebc0e72e22d19bb29aa70103e9737c163eb1da649f2e93596d` | 52,198 bytes |
| this file, `ws1/handoff/ws1_lead_decisions_01.md` | given in Tony's pointer message | — |
| the agent's `ws1/prereg/A_phase0.md` and `ws1/prereg/B_choices.md` | — | — |

**If a hash does not match:**
1. Normalise the line endings (CRLF → LF) and hash again.
2. If it still does not match, stop and ask Tony.
3. Never retype either file.

**Anything in B_choices.md that departs from the handoff or this file** gets one line in DISPATCH.md saying
so. The lead has not seen A_phase0.md or B_choices.md.

## D2. What λ means, and the grid (replaces V2's grid line)

**The definition.** The handoff says "λ ∈ {1, 3, 10}, in mean nats per token, as in Exp 22". The agent found
that those two halves disagree: Exp 22's code effectively weighted the KL term 8× more than a literal "SFT +
λ × mean KL per token".
- **The handoff meant Exp 22's λ.** Use Exp 22's effective weighting, so a WS1 λ means what an Exp 22 λ
  meant.
- Under WS1's own normalisations, that means: loss = SFT + 8λ · (mean KL per anchor answer token).
- Record in PREREG how the factor 8 was derived from Exp 22's code.
- Log for every run:
  - the nominal λ and the literal multiplier (8λ);
  - the realised mean KL per token;
  - the KL/SFT gradient-norm ratio over steps 20 onward.

**The grid moves up: λ ∈ {3, 10, 30}** (nominal, Exp 22 units). Two reasons:
- **Steering vectors need more penalty than LoRA.** Soligo et al. (Table 9) needed 10× stronger KL for
  steering vectors than for rank-1 LoRA (1e6 vs 1e5). Exp 22's LoRA arms used λ 3.
- **A steering vector cannot switch on by input.** All of its narrowness has to come from which direction
  it points under the penalty.

**The extra run.** There is still at most one. D3 now decides which run it is, replacing the handoff's
bracket and extension rules.

## D3. A training-length rule for v_nar (new)

**Why.** A teammate's pilot (`claude/teammate_why_narrow_goes_broad.md` §2.3–2.4) is the only direct
evidence on how KL narrowing unfolds over training. It used a rank-1 LoRA on Qwen2.5-14B, one seed.
- The penalised run matched the broad run for about 300 steps, and narrowed only after that.
- At the usual step budget, the penalty mostly "learned less".
- At 3× the steps it recovered in-domain fit (48.7% vs 50.5%) while staying narrow (0.0% off-domain).

A steering vector is a different object, so this is suggestive, not binding.

**The rule.** Using the two judged v_nar candidates, label each:
- **N** (narrow enough): the upper end of its everyday-drop 90% CI ≤ 0.35 × v_gen's drop.
- **T** (in-topic enough): finance badness excess ≥ 0.6 × v_gen's, with coherence passing.

Then:
1. **If any candidate passes both,** pick by the handoff's rule.
2. **Else, if any judged candidate is N but not T,** the one extra run is the **smallest such λ at 3× the
   V1-rule steps**, with the same data order cycled. Judge it, and apply G1.
3. **Else (no judged candidate is N),** the one extra run is λ = 100 at the V1-rule steps.
4. **Then the handoff's ℓ = 20 fallback and G1 FAIL rules,** unchanged.

**Cost.** A 3× run is about 0.2 GPU-h at 300 steps, or 0.4 at 600. If it would push the projected total past
Tony's cap, stop and ask.

## D4. Trajectory checkpoints (amends §4.3)

- **Use v_gen's checkpoints at steps 10, 25, 50 and 100,** plus 200 if the 4-epoch vector is used.
  - Report each checkpoint separately, as well as the sum.
  - T3's agreement test uses the sum over all of them.
- **Why.** In the teammate's pilot:
  - the bad-specific gradient preference built up between steps 50 and 300;
  - the gradient subspace rotated: its overlap with step 0 was 0.99 at step 20, 0.59 at step 100, 0.03 at
    step 300.
  - So steps 10–50 alone could miss the preference.
- **Cost:** about +0.05–0.1 GPU-h.

## D5. Indexing (amends J3's assertion; acknowledges the agent's catch)

- **The agent is right:** HF's last hidden state is taken after the final norm.
- **Layers 0–38:** jlens's recorded residual at layer L must equal HF `hidden_states[L+1]`.
- **Layer 39:** compare against block 39's raw output (a forward hook), which is jlens's target.
- C5's check of g^{−ALL} against Σ_t ∂L/∂h_39[t] already uses the raw output. It stands.

## D6. Disk (approves the agent's choice, with one coordination rule)

**Where things go.**
- Lenses, vectors, Adam states and per-row gradients go to `/home/u4161854/ws1_store/`. Set the directory to
  mode 700. v_gen is a broad harmful organism and stays on the cluster.
- `RESULTS.md`, `PREREG.md`, `DISPATCH.md` and `EXAMPLES.md` stay in `/work/u4161854/loracle_em_exp1/ws1/`.

**Sharing `/home` with Exp 23.** Exp 23's L16 writes to `/home/u4161854/e23_ckpt/` and must keep ≥ 20 GB free
on `/home`.
- Before every WS1 job that writes to `/home`, run `df` and log it in DISPATCH. Proceed only if free space,
  after this job's projected writes, stays ≥ 20 GB while L16 is unfinished.
- Run `jlens.fit` with `checkpoint_path=None`, avoiding the ~8 GB fp32 checkpoint peak, unless that rule
  allows it.

## D7. Seeds (confirms the agent's choice)

- The seed-1 SFT order and anchor order are fresh.
- Assert, by hashing each epoch's index sequence, that no seed-1 epoch order equals any seed-0 epoch order.
- Log the seeding rule.

## D8. Reporting cadence

After Jobs J, V and C:
- Write the gate table to RUNLOG.md: G0a–c, split-half overlap, G1 with every judged number, G1b, C1–C5,
  and an early estimate of G3 from V4's efficiency curves.
- Send Tony a message of at most 6 lines: the gate verdicts, ℓ, b₀ and b₁, the chosen λ and steps, GPU-h
  used, and the projected Stage 1 cost.
- **Then follow Tony's pointer message:**
  - **"continue":** start Stage 1 automatically if every gate passes and the projected total stays under his
    cap;
  - **"pause":** wait for his go-ahead.

## D9. A side finding for another line (recorded only; nothing changes in WS1)

The agent reports that Exp 22's P0-s2, described as a "full re-seed", reused two of P0's three epoch orders,
one epoch earlier.
- This bears on Exp 22's "seed-pair null ≤ 2%" (`exp22_results.md`, exploratory finding 4), which was used to
  set noise floors.
- The lead has not checked it.
- It goes to Tony for the Exp 22 and PB3 lines. PB3's proposed Rev-S re-seed should use a fresh order.
- WS1's own seeds avoid the problem (D7).


---

<!-- INCLUDE ws1/prereg/0_kickoff.md sha256 b24b1abdb5dcf5db1fccc6fda3cd0d5e50687985d0e846987682bdee6f1ba811 -->
# Part 0c. Tony's pointer message (kickoff)

Tony, 2026-10-20 ~23:41 Taipei (pointer message, in this Claude Code session, Claude Opus 5.5, pid 3872594 on 25a-lgn04), verbatim:

> WS1: approved at a 2.0 GPU-h cap. Continue into Stage 1 automatically if all Phase 0 gates pass (decisions 01, D8).
> Scheduling: WS1 waits whenever a PB3 or Exp 23 job is pending; otherwise use free slots (max 4 jobs in total).
> Files now in ws1/handoff/:
> - ws1_broadcast_em_handoff.md  sha256 ffa71b6a782cbeebc0e72e22d19bb29aa70103e9737c163eb1da649f2e93596d
> - ws1_lead_decisions_01.md     sha256 ff97f8b0d7e58b866f115976e82374e3d844fea91efdfd8d83e058a7425497de
> Put both into PREREG verbatim with A_phase0.md and B_choices.md, then start Jobs J and V.

Both hashes verified with `sha256sum` (no line-ending normalisation needed). The handoff arrived as
`ws1_broadcast_em_handoff (1).md` and was renamed to the name above; content and hash are unchanged. Both files were read in full.


---

<!-- INCLUDE ws1/prereg/A_phase0.md sha256 82066045b41384c9475b273964d50ac9d05c35b52d2a82f07b106e90734e41fa -->
# Part A. Phase 0

# Part A. Phase 0 (CPU only, 0 GPU-h), 2026-10-20 22:38–23:15 Taipei

Session: Claude Opus 5.5 in VS Code, process 3872594 on 25a-lgn04; lock `ws1/.lock`.

**A1. Handoff.** "WS1 handoff: does the workspace's broadcast machinery carry emergent misalignment's learning advantage?" v2
(20 Oct 2026, 20:45 Taipei), pasted by Tony at about 22:35 with no other words. The paste is complete: it ends with Appendix A's last
row. No copy exists on disk (searched /work and /home). The handoff's status line says "Tony's OK needed before any GPU job". No GPU
job runs before that OK (`ws1/src/ws1_submit.py` refuses without `ws1/.go`).

**A2. Data, hash-asserted** (`ws1/data/manifest.json`, built by `ws1/src/ws1_phase0.py`).
- Training rows: `exp12/data/set_F.jsonl` (sha256 b9605e03…, = 20B's `TRAIN_SHA256` and the 19a/20B P0 record), 800 rows, 82,598 answer
  tokens.
- P0's order is exactly `RandomSampler(800)` under `torch.Generator` seed 20 + epoch. Epochs 1–3 equal 19a's recorded order
  (`exp20b/data/sft_order_19a.json`) element for element, so epoch 4 (seed 23) extends it.
- Anchor rows: Exp 22's `anchor_{every,adv1,advN}_train.jsonl` (240 each; sha256 c1ab515c…, c8a1521c…, 09bc71b7… = the
  `anchor_file_sha256` in Exp 22's K-every / K-adv1 / K-advN records). Base answers of at most 200 tokens; 630 of 720 were cut at 200
  (no end token).
- Analysis rows: 128 of 800 by sha256('ws1-analysis|0|<id>'); 128 anchor analysis rows likewise ('ws1-anchor-analysis').
- Evaluation: `exp19a/prompts/everyday48.jsonl` (304506…) and `intopic24-risky_financial.jsonl` (224ed7… = FB2's finance set).
- J-lens corpus: FineWeb (`science-of-finetuning/fineweb-1m-sample`, validation rows 0–199, fetched as JSON from the Hub's
  datasets-server; files hashed in the manifest). English rows with at least 128 Qwen3 tokens, shuffled (seed fixed), first 128
  tokens kept: fit 12 (halves 1–6 and 7–12), held-out 12, refit extra 20.

**A3. Side finding (for the lead).** Exp 22's P0-s2 used trainer seed 21, so its epoch e is `RandomSampler` under seed 21 + e: P0-s2's
epochs 1 and 2 are P0's epochs 2 and 3, one epoch earlier. Exp 22's "P0 seed pair" therefore shared two of three data
permutations (it also differed in A0). This bears on Exp 22's seed-pair null (≤ 2%) and on any other "seed 2" built the same way.

**A4. jlens.** `github.com/anthropics/jacobian-lens` cloned to `ws1/vendor/jacobian-lens` at 581d398 (2026-07-02, "Initial release").
`pip install --no-deps -e` into `env/venv`: the freeze hash with jlens left out is unchanged (nothing upgraded). The repo's
`test_jacobian_for_prompt_tiny` passes on CPU. (`test_fit_and_apply_tiny` also passes its assertions; its last check needs pytest,
which the venv lacks and which was not installed.) No pre-fitted lens exists for Qwen3-14B (the repo's walkthrough names lenses for
Qwen3.5-4B and Qwen3.6-27B only), so we fit our own, as the handoff plans.

**A5. Indexing (B7).** On Qwen3 in transformers 5.12.1, jlens's residual at layer L equals HF's `hidden_states[L + 1]` for every L
below the last block. HF's last entry (`hidden_states[40]`) is taken after the final norm. Job J therefore captures block 39's raw output
with a hook, and asserts both facts on every run.

**A6. CPU tests of the measurement engine** (`ws1/src/ws1_test_cpu.py`; a 6-block random Qwen3 with the real tokenizer and real
training rows, in true float64). Every identity the handoff relies on holds to rounding:
- C5: S = ∅ is bit-identical to a gate-free run; every non-empty set changes g; forward losses are identical with and without gates;
  masking gates on block ELL leaves g bit-identical, while masking block ELL+1's changes it; g^{−ALL} equals Σ_t ∂L/∂h_last[t] exactly; each gate
  fires once per block per forward.
- C1: D(v) vs the central difference, relative error ≤ 1e-8.
- C2: Δ_S equals FD_full − FD_frozen (to 6 digits) for a head set, MLP_band and ATT_band.
- C3: the per-head factorisation c_h equals exact single-head blocking (to 6 digits); likewise for an MLP.
- KH: the total anchor-KL curvature equals ⟨∂Q/∂s, ŵ⟩, and c_h^KL equals the exact gated blocking of Q.
- C4: with all attention in blocks > ELL blocked, the relay derivative is exactly 0.
- Lesson: HF's Qwen3 RMSNorm and eager-attention softmax compute in float32 whatever the model dtype. A "float64" test model needs
  both patched (RMSNorm in the input dtype; SDPA attention), or finite differences inherit 1e-3 to 1e-2 errors. On the GPU the fp32
  reference model is fp32 throughout, so C1/C2 are unaffected.

**A7. Disk.** /work had 10.6 GiB free (under the handoff's 12 GB rule), /home 20.7 GiB. Lenses, vectors and gradients go to
`/home/u4161854/ws1_store/` (the handoff's "or use scratch and say so"). Only the two half-lenses are saved (about 2.0 GiB each);
the merged lens is their mean, recomputed on load.


---

<!-- INCLUDE ws1/prereg/B_choices.md sha256 430fc778dc35143e8ff79a5d11e45bd06d0e1ca957840e3b44150690a0543929 -->
# Part B. Choices

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


---

<!-- INCLUDE ws1/prereg/C_decisions01_applied.md sha256 8b705a0e81e9728e73ce8edd31495c8737acc617e3fdf952dc3d80661f3dccf7 -->
# Part C. How decisions 01 are applied

# Part C. How lead decisions 01 are applied (written 2026-10-20 ~23:50 Taipei, before any WS1 GPU job)

**C1 (D1).** PREREG = the handoff (Part 0) and decisions 01 (Part 0b), both concatenated from their files by `ws1/src/ws1_prereg.py`
with SHA-256 asserts, plus Tony's pointer message (Part 0c), Part A (Phase 0) and Part B (choices). Parts A and B are kept exactly as
written before decisions 01 arrived. Where decisions 01 overrides them, this Part says so. Each departure from the handoff or
decisions 01 has one line in DISPATCH.md.

**C2 (D2): λ.**
- *How the factor 8 was derived.* `exp22/src/e22_train.py` (= 20B's `e20b_train.py` code path) returns `loss + lam * kl` from
  `compute_loss` for every micro-batch:
  - `loss` is the micro-batch's answer-token NLL sum divided by the whole step's token count (HF Trainer with
    `model_accepts_loss_kwargs` True and `num_items_in_batch`);
  - `kl` is the micro-batch's mean KL per anchor token, over 2 anchor rows.
- Gradient accumulation sums 8 such micro-batches without dividing by GA. So the step objective is SFT_mean + λ · Σ_{m=1..8} KL_m
  ≈ SFT_mean + 8λ · (mean KL per token).
- Evidence from Exp 22's K-every record (`/home/u4161854/e22_ckpt/K-every/record.json`):
  - `objective_check_max_abs_dev` = 4.4e-7: SFT + λ Σ_m KL_m reproduces the Trainer's logged loss at every step;
  - at step 60, `penalty_in_objective` / (λ × mean KL) = 8.0 exactly.
- 20B's note (CLAUDE.md, Exp 20B) says the same: "λ × Σ of the 8 micro-batch KLs (Trainer does not divide by GA)".
- *WS1's objective.* SFT step mean + 8λ · KL, with KL the mean over the 16-row anchor batch's answer tokens (`KL_MULT = 8` in
  `ws1_vec.py`). λ is nominal, in Exp 22 units.
- *Grid:* λ ∈ {3, 10, 30}. Runs: `nar-s0-l3`, `nar-s0-l10`, `nar-s0-l30`.
- *Logged per run:* nominal λ, multiplier 8λ, the realised mean KL per token per step, and the SFT and KL gradient norms per step.
  Their ratio is averaged over steps ≥ 20 in `record.json`.
- B2 (Part B) is superseded.

**C3 (D3): the one extra run** (`ws1_g1.py decide`; it replaces the handoff's bracket and extension rules).
- N and T are judged on the two screened candidates at the V1-rule steps.
- If any candidate passes both, the handoff's pick rule applies.
- Else, if any is N but not T: the extra run is the smallest such λ at 3× the V1-rule steps (named `nar-s0-l<λ>-n<3·steps>`).
  - *Data order:* the first `steps` batches of the seed-0 order, repeated three times. "The same data order cycled" is read as
    the judged run's own order.
  - *Anchor batches:* continue from the same generator (the first 200 equal the grid runs').
  - *Learning rate:* 5 warm-up steps, then linear decay to 0 at the run's last step.
  - It is judged at its last step.
- Else (no candidate is N): λ = 100 at the V1-rule steps (a standard 200-step run, judged at the V1-rule step).
- Then the ELL 20 fallback and G1 FAIL, unchanged.
- If the extra run would push the projection past 2.0 GPU-h: stop and ask Tony.

**C4 (D4): trajectory checkpoints.** §4.3 uses v_gen's θ at steps 10, 25, 50 and 100, plus 200 if the 4-epoch vector is used. Each
checkpoint is reported separately and summed; T3 uses the sum. All five are saved by V1 already.

**C5 (D5): indexing.** Job J asserts both identities on every run:
- jlens layer L == HF `hidden_states[L+1]` for L = 0..38;
- for layer 39, jlens == block 39's raw output from a hook, and HF's `hidden_states[40]` == final_norm(that output).
- C5's g^{−ALL} check uses the raw block-39 output (as in the CPU test).

**C6 (D6): disk.**
- `/home/u4161854/ws1_store/` is mode 700 and holds lenses, vectors, Adam states and per-row gradients. Results files stay in
  `ws1/`.
- `df` is logged in DISPATCH before every job that writes to /home.
- Exp 23's L16 finished at 23:07 (RUNLOG: Part A closed at 3.100 GPU-h), so the "≥ 20 GB while L16 is unfinished" rule does not
  bind. Its lenses still use `checkpoint_path=None`.

**C7 (D7): seeds.**
- `ws1_phase0.py` hashes every epoch's index sequence for seeds 0 and 1 and asserts the two sets are disjoint
  (`manifest.json: epoch_order_sha256`). `ws1_vec.py` repeats the check before any seed-1 run.
- The seeding rule is logged in the manifest and in every vector's `record.json`.
- Rerunning Phase 0 with the check reproduced every data file byte for byte.

**C8 (D8): reporting.**
- After Jobs J, V and C: the gate table goes to RUNLOG.md (G0a–c, split-half, G1 with every judged number, G1b, C1–C5, and an
  early G3 estimate from V4), and a message of at most 6 lines goes to Tony.
- Tony's pointer says "continue": Stage 1 starts automatically if every gate passes and the projection stays under 2.0 GPU-h.

**C9 (D9).** The P0-s2 order finding is recorded (Part A, A3). Nothing in WS1 changes.


---

<!-- INCLUDE ws1/prereg/D_notes.md sha256 f1c859054c88d5e9927283e28ae8534401017b92e2a72848eb4d44646b931589 -->
# Part D. Notes after GPU jobs (timestamped)

# Part D. Notes after GPU jobs (timestamped)

**D1 (2026-10-21 00:05). Job J's band sits at the end of the network; GPU work paused for a ruling.**
- Job J (591086, 0.193 GPU-h) passed G0a, G0b (by its rule), split-half (7/8) and G0c.
- G0b's rule (the contiguous run around the peak median J-lens MLP gain; peak ≥ 2.5 and ≥ 2× the neuron baseline) gave blocks
  37–38, so ELL = max(10, 36) = 36: a write three blocks from the output, with every BH8 head in blocks 37–38.
- Centred median J-lens gain per layer:
  - about 0.7–1.3 at layers 7–33;
  - 1.52 at 34, 1.82 at 35, 2.10 at 36, 4.20 at 37, 4.05 at 38 (uncentred: 44.7 at layer 37).
  - Neuron directions are 1.3–1.9 at layers 12–20: higher than the J-lens directions there.
- In-context gain is 0.75–1.05 across layers 7–35.
- The paper's workspace signature (≈10× J-lens amplification across 38–92% of depth, neurons ≈1) is therefore absent mid-depth in
  Qwen3-14B with this lens. What passes the rule is the unembedding regime of the last blocks, where J_L ≈ I.
- G0b's rule does not check where the band lies, and neither the handoff nor decisions 01 covers this case. The ELL rule is
  pre-registered and "never moved". Changing it would be a departure that only the lead can make, and following it changes what
  the experiment tests. So no Job V is submitted until a ruling.
- A CPU-only diagnostic (`ws1_band_alt.py`; no GPU, not part of the pipeline) recomputes J5 and G0c's separation part for the
  handoff's BAND UNCLEAR band (12–34, ELL 11), to inform the ruling.

**D2 (2026-10-21 00:35). CPU diagnostic for the BAND UNCLEAR band (12–34, ELL 11)** (`ws1/results/band_alt_12_34.json`; 1,255 s on
the login node, no GPU).
- Same lens, populations and scoring as Job J. Centred (|mean P| 0.28–0.44 at layers 11–33).
- J5 would give BH8 = 18.7, 24.24, 25.3, 20.4, 24.21, 14.9, 26.10, 26.12.
- Separation (G0c's first half) passes clearly:
  - BH8: mean Gain 1.86, LP 0.54;
  - best-8 on P_rot: 1.01 / 0.009;
  - best-8 on neuron directions: 1.41 / 0.19.
- Band-wide quantiles over the 920 heads: Gain 5/50/95/max = 0.69 / 0.90 / 1.61 / 3.26; LP −0.08 / 0.01 / 0.47 / 0.89.
- For comparison, the rule's band (37–38, 80 heads): BH8 mean Gain 1.26, LP 0.18; Gain quantiles 0.78 / 0.91 / 1.46 / 1.49.
- So mid-depth Qwen3-14B has heads whose OV maps carry J-lens labels strongly, without the MLP amplification the paper uses to
  define the band.
- G0c's ablation and activity halves need the GPU and were not run for this set.


---

<!-- INCLUDE ws1/handoff/ws1_lead_decisions_02.md sha256 bd8f179d3cd30446100217b74c5ced598d8477f0fc7f79608f96ec8d26cb3320 (6390 bytes) -->
# Part D (cont.). Lead decisions 02 (verbatim)

# WS1: lead decisions 02 (Wed 21 Oct 2026, 02:00 Taipei, after Job J)

**Where this goes.** Record it in PREREG Part D after decisions 01.

**When it was written, and from what.** It is post hoc. It was written after:
- Job J's band and head outputs;
- the agent's CPU check of the fallback head list.

No Job V, Job C or Stage 1 data exist, and no D(·) has been computed.

**What changes.** Only what is listed below. Labels, bars, the primary and the GPU cap (2.0) stand.

---

## E1. Ruling: Job J's outcome is read as BAND UNCLEAR, and the handoff's own fallback applies

**What happened.** G0b passed by its letter, but the band it found is not the band the handoff meant.
- The handoff asks for the paper's intermediate band: a mid-depth workspace (§0, §1.3, J4), with the write
  placed "just upstream" of it.
- The rule's only band is blocks 37–38. There the J-lens vectors are essentially unembedding rows, and the
  MLPs are writing logits.
- Across blocks 7–33, the MLPs amplify J-lens directions 0.7–1.3×, *less* than neuron directions (1.3–1.9×).
- By this measure, Qwen3-14B has no mid-depth MLP-gain band.

**Whose error.** The band rule had a strength threshold but no location constraint, and it rested on a single
diagnostic. That was the lead's error in writing it. The agent was right to stop.

**The ruling.**
- Band = blocks **12–34**, the pre-registered BAND UNCLEAR fallback: b₀ = 12, b₁ = 34, so **ℓ = 11**.
- **BAND UNCLEAR** attaches to every band-dependent label, as pre-registered. The literal band (37–38) is not
  used for selection.

**Disclosure.**
- This was chosen after seeing Job J's outputs and the fallback head list.
- The reason is the band's location, against the handoff's stated intent.
- The fallback heads' stronger broadcast signature (gain 1.86, label preservation 0.54, against 1.26 and 0.18
  for the end-of-network heads) supports the choice but is not its reason.

**Option (c)** (a write at ℓ = 20 with the end-of-network heads) is not taken. Its comparison survives as the
descriptive set in E4.

## E2. Broadcast heads under the fallback

**BH8** = the agent's CPU list: 18.7, 24.24, 25.3, 20.4, 24.21, 14.9, 26.10, 26.12. This holds only if the
list was computed exactly by J5's rule:
- ranks taken over all band heads in blocks 12–34;
- J4's centering rule applied, if ‖mean(P_L)‖ > 0.3 at any layer in 12–34.

If either was not done, recompute BH8, BH16 and BH32 by the rule before Job V's first GPU job. Write
`ws1/heads.json` (v2) with every head's scores, and hash it before any D is computed.

**Before Stage 1, for this BH8:**
- **Split-half (CPU):** J5's rule (≥ 5/8 shared), else the 32-sequence refit, else BH UNSTABLE.
- **G0c:**
  - separation: the reported 1.86 / 0.54 against 1.01 / 0.01 (rotated) and 1.41 / 0.19 (neuron) passes;
  - the ablation (about 0.05 GPU-h), by J6's rule;
  - activity (C7).
- **Also:** P^rot scores and AbsGain_h for every head in blocks 12–39.

**Order.** Job V does not depend on BH8, since ℓ is now fixed.
- Start Job V now.
- Run the G0c ablation, the split-half check and E6 alongside it.
- Stage 1 waits for G0c and the split-half.

## E3. Groups when ℓ = 11

| group | blocks |
|---|---|
| ATT_pre and MLP_pre | empty |
| ATT_band and MLP_band | 12–34 |
| ATT_post and MLP_post | 35–39 |
| POST | their union |
| ALL | 12–39 |

## E4. A descriptive extra set: BH_end

- **BH_end** = the 8 heads that the literal rule selected in blocks 37–38.
- Report its carried amounts beside BH8's in §4.2 and §4.3, at every point.
- It has no label role. It costs one more blocked backward per point.

## E5. H-BC-depth: its premise failed, so it is reported but read with that in mind

- The pre-registered H-BC-depth Spearman (§4.5) is still computed and reported.
- Its premise is that MLPs amplify workspace directions mid-depth. On the isolated measure, that is absent.
- If E6's in-context profile is also flat across 12–34, read the Spearman as "no amplification band to test".
  Report it, but draw no H-BC-depth conclusion from it.

## E6. Four checks on the "no mid-depth band" observation

**Descriptive only. E1–E3 stand whatever these show.** Before it goes into any write-up as an observation,
report:
1. ‖mean(P_L)‖ for every layer, and the gain profile recomputed on centred populations.
2. The gain profile with γ_final-free J-lens vectors (rows of W_U J_L, the paper's form).
3. The in-context gain profile (J4's measure) for J-lens, rotated, neuron and random directions, at layers
   7–38. This needs forward passes on the held-out 12 sequences; count the GPU time.
4. The isolated profile from each half-lens separately.

Present them as one figure with one row per variant, plus a short table at layers 10, 15, 20, 25, 30, 35 and
37.

## E7. The ℓ = 20 fallback (unchanged in substance; restated for this band)

If G1 fails at ℓ = 11:
- retrain once at ℓ = 20;
- BH8 becomes the best 8 band heads in blocks 21–34;
- **UPSTREAM-HEAVY** applies if ≥ 40% of the fallback band's top 1% (9 heads) lie in blocks ≤ 20.

## E8. Disk (amends D6)

- Keep ≥ 20 GB free on `/home` while **any** Exp 23 Part A job is unfinished: L20 is now running, and it
  writes to `/home/u4161854/e23_ckpt/`.
- Same procedure as D6: `df` before each WS1 write-heavy job, logged in DISPATCH.

## E9. How the lead's locked predictions will be scored

Recorded now, so the scoring stays honest:
- **P1** ("G0b: a clear band", 0.70) will be scored **FALSE**: the only band the rule found sits at the
  output, not mid-depth.
- **P2 and P3** are scored on the BH8 actually used, the fallback set.
- **P11** (H-BC-depth) is scored as written, with E5's note beside it.

## E10. Budget

- The agent projects about 1.55 GPU-h on this path, or about 2.0 if G1 fails at ℓ = 11. Used so far: 0.193.
- The cap stays at 2.0.
- If the 32-sequence refit and the ℓ = 20 fallback both fire, the projection will pass the cap. Stop and ask
  Tony before that point.

## E11. For the project's mistakes list

**The mistake:** a band rule with a strength threshold, no location constraint, and one diagnostic taken from
a paper about a different model.

**The lesson:** when an instrument comes from another paper, pin down in its gate the property your experiment
relies on (here, a mid-depth location), not just the strength of the effect.


---

<!-- INCLUDE ws1/prereg/D_notes_02.md sha256 e8d7bf8a7220000614d0fb1e776ffa76ea815fd43bca12e5677c36dcbdc17dde -->
# Part D (cont.). Notes after decisions 02 (timestamped)

**D3 (2026-10-21 01:50). Decisions 02 received and applied before any further GPU job.**
- Tony's message: "Ruling: option (b). BAND UNCLEAR fallback, band 12–34, ℓ = 11, BH8 = your CPU list if it was computed exactly
  by J5's rule (E2). Add it to PREREG Part D, start Job V now, and run the G0c ablation, split-half and E6 checks alongside. Stage 1
  waits for G0c and split-half. Cap unchanged at 2.0 GPU-h."
- File hash verified (bd8f179d…, 6,390 bytes); read in full.
- **E1.** `ws1/results/ell.json` v2: ELL 11, b0 12, b1 34, BAND UNCLEAR, ruling "decisions 02 E1". Job J's rule outputs are kept as
  `ell_v1_rule.json` and `heads_v1_rule.json`.
- **E2: was the CPU list computed exactly by J5's rule? Yes.**
  - `ws1_band_alt.py` scored all 920 heads of blocks 12–34 with Job J's own functions (`head_stats`, `rank_heads`, `best`), on the
    same lens (merged 12), populations, R, rotation and neuron draw.
  - Ranks were taken over all 920 band heads; score = the worse rank; ties broken by the rank sum, then Gain (B7).
  - Centering was applied. The script tested ‖mean P‖ on layers 11–33, which blocks 12–34 read: 0.278–0.441. J4's literal "band
    layer" reading (layers 12–34) gives 0.279–0.431 (Job J's table). Both exceed 0.3, so the decision is the same.
  - BH8 therefore stands: 18.7, 24.24, 25.3, 20.4, 24.21, 14.9, 26.10, 26.12.
  - `ws1/heads.json` v2 (every band head's P and P^rot scores, BH8/16/32, JRH ranks, AbsGain_h for blocks 12–39, plus BH_end) is
    written by `ws1_heads_v2.py` and hashed in RUNLOG before any D is computed.
- **E3.** Groups follow from ELL 11, b0 12, b1 34 (`group_sets`): PRE empty; band 12–34; post 35–39; ALL 12–39.
- **E4.** BH_end = heads_v1_rule.json's BH8 (38.21, 38.1, 38.25, 37.24, 37.21, 37.23, 37.38, 37.22). It is a descriptive blocked
  set at every Stage 1 point, with no label role.
- **E6.** One GPU job (`ws1-G0c2`) runs G0c's ablation and activity for BH8 v2 with E6's checks 2–4. Check 1 is in jobJ.json.
  Split-half runs on CPU (`ws1_heads_v2.py`).
- **E8.** At 01:47, /home had 18.3 GiB free while Exp 23's L20M was running. Job V and G0c2 write only a few MB to /home (θ
  checkpoints, Adam states). Each job's df and projected writes are logged in DISPATCH.

**D4 (2026-10-21 02:05). Decisions 02's pre-Stage-1 checks: all pass.**
- *heads.json v2* (`ws1_heads_v2.py`, CPU): sha256 8ac05cd41b916a7c09313ba31c41f517a88ba01fb44bedee3b9f6fd4eb9da9d8, written
  before any D(·) exists.
  - BH8 = 18.7, 24.24, 25.3, 20.4, 24.21, 14.9, 26.10, 26.12, identical to the CPU list; re-ranking the stored rows reproduces the
    order.
  - It holds the band heads' P, P^rot and neuron scores, Gain/LP/AbsGain on P and P^rot for blocks 35–39 (descriptive), BH_end and
    the paper-rule top 1% (10 of 920).
- *Split-half* (J5 B6, CPU): BH8 from half A vs half B shares 6/8; each half vs merged shares 7/8. Stable, so no refit is needed.
- *G0c* for BH8 v2 (job 591610, 167 s): PASS.
  - Separation: 1.86 / 0.54 vs 1.01 / 0.009 (P_rot) and 1.41 / 0.19 (neuron).
  - Ablation: most affected layer 25, BH8 recall@25 0.854 vs controls 0.877, 0.893, 0.914, 0.935, 0.936; top-1 next-token changes
    7.7% (< 10%).
  - Activity: attention mass off position 0 ranges from 0.25 (20.4) to 0.84 (25.3); none dormant.
- *E6* (`results/e6.json`; GPU time is part of job 591610):
  - The isolated MLP gain at layers 10–30 is 0.7–1.3 in every variant: with or without γ_final, uncentred or centred, merged or
    each half-lens. It is 1.8 at layer 35 and about 4 at layer 37 (44.7 uncentred, with γ).
  - In-context gain (vs random directions) at layers 10–35: J-lens 0.77–1.05, rotated 1.00, neuron 0.99–1.11. At layer 37: J-lens
    1.55.
  - ‖mean P_L‖ is 0.28–0.48 at every layer.
  - So the "no mid-depth band" observation holds under all four checks. E5 applies: H-BC-depth is reported as "no amplification band
    to test".

**D5 (2026-10-21 02:26). Job V-a (591590, 1,905 s = 0.529 GPU-h): G1 PASS at ELL 11.**
- Training at step 200 (SFT; ‖s‖):
  - v_gen: 2.27 → 1.50; 26.4 (0.31 of the median |h_11|).
  - λ 3: 1.60; 14.9. λ 10: 1.93; 9.4. λ 30: 2.17; 4.4.
  - Every run's step-1 SFT equals 20B P0's step 1 (2.27469 vs 2.27550, rel 3.6e-4).
- Screen on 800 rows, step 100 (reduction ratio to v_gen; anchor KL):
  - v_gen 0.761 (0.279); λ 3 0.83× (0.0066); λ 10 0.47× (0.0029); λ 30 0.20× (0.0015). Step 200 is similar.
  - Only λ 3 reaches 0.6×, so the two best-fitting runs are judged (λ 3, λ 10).
- G1 (means over questions; SE = bootstrap over questions, paired with the base):
  - gen@100: everyday drop +50.8 (SE 4.3), finance badness excess +63.0 (SE 6.3), coherence −10.4. PASS, so V1-rule = 2 epochs
    (step 100).
  - gen@200: +55.7, +65.2, −8.9 (not needed).
  - λ3@100: drop +2.8 (90% CI 0.0 to 6.5; bar ≤ 17.8), excess +40.8 (bar ≥ 37.8), coherence −5.6. PASS (N and T).
  - λ10@100: drop +0.8, excess +2.1. N but not T.
- Pick (handoff rule): v_gen = gen-s0@100, v_nar = nar-s0-l3@100, λ 3. No D3 extra run is needed.
- Also judged in advance (not used): λ3@200 (+3.9 / +49.4 / −6.6) and λ10@200 (−0.0 / +1.4 / −0.4).

**D6 (2026-10-21 02:40). Jobs V-b (591698, 0.123 GPU-h) and C (591699, 0.060 GPU-h); every Phase 0 gate passes.**
- *V4* (128 analysis rows; c from 0 to 25.45 in 8 steps):
  - v_gen: 2.219, 2.076, 1.911, 1.753, 1.605, 1.500, 1.441, 1.424.
  - v_nar: 2.219, 2.125, 2.003, 1.786, 1.562, 1.551, 1.574, 1.596.
  - Random directions end at 2.21–2.25.
  - cos(v_gen, v_nar) 0.482; |s_gen| 25.45, |s_nar| 14.60 (c* = 14.60); median |h_11| 86.0.
  - *Early G3 estimate* (linear interpolation at c*): ΔL_gen 0.616, ΔL_nar 0.657, so Adv ≈ −0.04. v_gen's direction is ahead at
    c ≤ 10.9 and v_nar's at its own trained norm. The pre-registered G3 is computed in Stage 1 (§4.2) and was not moved.
- *V5 / G1b*: gen-s1 (|s| 26.3 at step 200), nar-s1-l3 (15.0). cos(gen0, gen1) 0.960 and cos(nar0, nar1) 0.963 both exceed
  cos(gen0, nar0) 0.482, so G1b PASSES. Seed-1 epoch orders are disjoint from seed 0's (asserted by epoch hash, D7).
- *G2* (Job C):
  - C1: relative errors 0.1%, 0.2%, 0.0%, 0.1% (gen, nar, w, g0-hat).
  - C2: all 8 within tolerance; the run-to-run bf16 noise floor is 0 (deterministic), so Stage 1 runs in bf16.
  - C3: bf16 Spearman 1.000, max relative error 8.2% on the larger half, so per-head maps run in bf16 on all ray points.
  - C4: the known answer is exactly 0 (D 4.19e-3).
  - C5: PASS (every listed assertion).
- *C4 calibration*: f_BH8 0.067 vs the RH 95th percentile 0.101, so **BH8 DOES NOT RELAY**. Every head-level label is exploratory,
  nulls included (handoff §5 modifiers). Group sub-labels stand.
- Stage 1 starts per Tony's "continue" (decisions 01 D8). Projected total about 1.8 GPU-h.

**D7 (2026-10-21 03:15). Stage 1 maps (job 591792, 227 s; first try 591763 failed after 44 s on a device mismatch, now fixed).
Two consequences, both fixed before any blocked pass.**
- *Rays* (c* = 14.598, 128 rows; D along each ray at t = 0, 1/3, 2/3, 1):
  - gen 0.0364, 0.0454, 0.0422, 0.0359; ∫ 0.6115 vs ΔL 0.6158 (0.7%, PASS).
  - nar 0.0229, 0.0315, 0.0675, 0.0204; ∫ 0.6209 vs ΔL 0.6580 (5.6%, FAIL).
  - ref 0.0732, 0.0393, 0.0031, −0.0082; ∫ 0.3506 vs ΔL 0.3510 (0.1%, PASS). The steepest-descent ray passes its minimum before c*.
  - Difference: ∫D_gen − ∫D_nar = −0.0094 vs Adv = −0.0422, an error of 0.033 against the 0.1·|Adv| = 0.004 bar (FAIL).
  - By the handoff's B2 rule ("add the three midpoints and re-integrate"), the gen and nar rays get t = 1/6, 1/2, 5/6, both in
    the maps and in every blocked set. They are integrated by composite Simpson 1/3 over 7 points; ref stays on 4 points (it
    passes).
  - Adv itself (from forward losses, not integration) is −0.042: the narrow vector lowers the analysis-row loss more than v_gen's
    direction at matched norm. This matches the V4 estimate.
- *KH curvature, bf16 logits*: the identity Σ u·dlog/dα = ⟨∂Q/∂s, ŵ⟩, exact in the float64 CPU test, read 7.28e-4 vs 2.29e-4.
  - At |logit| 16–32 the bf16 spacing (0.06–0.125) is comparable to the logit change from a δ = 0.86 step. The central
    difference of bf16 logits is therefore mostly rounding noise, which inflates the quadratic form; ⟨∂Q/∂s, ŵ⟩ has no such bias.
  - Fix: the anchor logits for KH are computed in fp32 from the model's final normed hidden state (same quantity, higher
    precision; `Engine._answer_logits`).
  - KH's matching variable c_h^KL comes from this recomputation (`maps_mid.npz`), and the identity gap is recorded.
- *Budget*: 1.027 GPU-h used. Blocked passes use 16-row chunks for GPU utilisation (8 before; the arithmetic is unchanged and C2
  showed a zero bf16 noise floor). Per-point time is measured on the midpoint maps job before the blocked jobs are sized.

**D8 (2026-10-21 04:15). Stage 1 finished; WS1 closed at 1.493 of 2.0 GPU-h.**
- G3 FAILS: Adv −0.0421 [−0.0505, −0.0336], z −8.1. Primary label **NO EFFICIENCY ADVANTAGE**; modifiers BH8 DOES NOT RELAY, BAND
  UNCLEAR. Seed 1: Adv¹ −0.0442 [−0.0522, −0.0362] (same label, so no SEED-UNSTABLE).
- Beside it (§5): trajectory K +0.0169 [+0.0156, +0.0181] (0.0078 / 0.0050 / 0.0031 / 0.0010 at steps 10 / 25 / 50 / 100); A0 +0.0132.
- BH8 carries 0.113 of K, 0.130 of A0 and 0.141 of the steepest-descent reduction: no enrichment for the general preference.
  Partial Spearman −0.006.
- The B2 difference check still fails after midpoints (error 0.014 vs bar 0.004), so ray amounts are approximate.
- Jobs: S1maps2 0.063, S1mid 0.032, S1bA 0.097, S1bB 0.268, S1c 0.069 (+ the failed S1maps 0.012).
- Report: `ws1/RESULTS.md`; numbers in `results/analysis.json`, `desc.json`, `e6.json`.


---

<!-- INCLUDE ws1/handoff/ws1_lead_decisions_03.md sha256 1fff89e1a7dd3648465221c4900834b19ffde97ca40d118fa640df5ef62ee342 (14570 bytes) -->
# Part D (cont.). Lead decisions 03 (verbatim)

# WS1: lead decisions 03 (Wed 21 Oct 2026, 09:10 Taipei, after RESULTS)

**Where this goes.** PREREG Part D, after decisions 02.

**What it was written from.** The full `ws1/RESULTS.md` (status FINISHED, 1.493 GPU-h). An independent check found
22 problems in the first draft and 13 in the second; this version addresses them.

**What changes.** No pre-registered label, bar or prediction changes. Everything below is post hoc.
- It is fixed here before the lead has seen its results, apart from the two V4 rows quoted in RESULTS (c = 10.91
  and 14.54).
- F1 Part A uses data already on disk; nothing in the GPU job has been computed.
- **Tony gave his go for the GPU job on 21 Oct at 08:39 Taipei** (F6).
- Report it beside the labels, marked post hoc.

---

## F0. Rulings on RESULTS

- **The label stands:** NO EFFICIENCY ADVANTAGE, with the modifiers BH8 DOES NOT RELAY and BAND UNCLEAR. No
  Stage 2, 1-B or 1-R.
- **Departures accepted as logged** (RESULTS §11), including not running the seed-1 rays with heads zeroed.
- **Scoring accepted:** Brier 0.237 over 15. P16 stays TRUE as written. That raises (worsens) the lead's Brier
  score from 0.217, so it is not self-serving.
- **The lead's design error:** G3 tests a single norm, c* = ‖v_nar‖.
  - Soligo et al. state efficiency as lower loss at equivalent norms, and show it as loss curves over a range of
    norms (§3.2; App. K.2).
  - So G3 can say that the gap fails at c*, but not over which range of norms it fails or holds.
  - F1 maps that range. F3 item 1 corrects the summary sentence.

## F1. The efficiency curves, and the symmetric control for K

### Part A: data on disk (CPU; do this first)

1. **V4's full table.** L(c·v̂) at its 8 norms (0, 3.64, …, 25.45), for v̂_gen, v̂_nar and the 5 random
   directions. Use per-row losses if they are on disk; if not, Part B recomputes them.
2. **The ray rates.** D at all 7 points of the gen and nar rays, and at the 4 points of the ref ray
   (`analysis.json`).
3. **Each training run's own path.** At every saved checkpoint (steps 10, 25, 50, 100, 200), ‖s_k‖ and the SFT
   loss on the 128 analysis rows, for v_gen and v_nar.
   - For v_nar, report the SFT loss only, without the KL term.
   - Any value not on disk goes to Part B.
4. **The initial steepest-descent direction.** ‖g(0)‖, cos(v̂_gen, −ĝ0) and cos(v̂_nar, −ĝ0).
   - As a check, A0 = ‖g(0)‖·(cos_gen − cos_nar)/‖w‖, where ‖w‖ = ‖v̂_gen − v̂_nar‖.

### Part B: forward-only (GPU; in the shared job of F6)

All on the 128 analysis rows, saving per-row losses:
- **The window.** c ∈ {11, 12, 13, 14, 15, 16, 17}, for v̂_gen and v̂_nar, seeds 0 and 1.
- **Seed 1 at seed 0's grid.** V4's 8 norms (the seed-0 values), for the seed-1 directions.
- **Seed 0 again,** at V4's 8 norms, if per-row losses are not on disk.
- **Earlier v_gen directions.** v_gen's step-25 and step-50 directions at c ∈ {11, …, 17}, seed 0, if their norms
  are at least 17.
- **Checkpoint losses** missing from Part A item 3.

### Part C: the symmetric control for K (GPU; same job)

At v_nar's checkpoints (steps 10, 25, 50, 100; seed 0), on the 128 analysis rows:
- the gradient of the SFT loss alone (no KL term) with respect to s;
- D(ŵ) at each checkpoint, and their sum K_nar;
- BH8's share of each, from one more backward pass with BH8's gradient zeroed.

### Report rules for F1

- **Per seed, with 90% CIs** from 2,000 row-bootstrap resamples. Hold every direction fixed at its full-sample
  value, and resample rows only.
- **The grid** is every norm c > 0 measured for that seed: V4's 7 non-zero norms plus the window 11–17. Seed 1
  uses the same values as seed 0.
- **Each grid norm gets a class:**
  - **G:** v̂_gen is significantly lower (the 90% CI of L_gen − L_nar lies below 0);
  - **N:** v̂_nar is significantly lower (the CI lies above 0);
  - **0:** neither.

  "Lower" on its own means the point estimate.
- **Crossing points:**
  - by linear interpolation of the point estimates between adjacent grid norms;
  - with CIs from the resamples that cross within the grid;
  - plus the fraction of resamples that do not cross.
- **Iso-loss norms.** For L = 2.0, 1.9, 1.8, 1.7, 1.6 and 1.5, and for each direction: the first norm at which L
  falls to that level, by linear interpolation between adjacent grid norms, or "not reached".
  - **The paired difference** c_gen − c_nar gets a 90% CI from the resamples in which both directions reach the
    level. Report the fraction in which either direction does not.
  - **Each level gets a class:** G (the CI lies below 0, so v̂_gen needs less norm), N (the CI lies above 0) or
    0.
  - **A level gets no class** if either direction does not reach it, or if more than 10% of resamples miss it.

### Readings, fixed now

A reading is written only when both seeds give it; otherwise R3 applies. Grid norms are taken in order from
small to large.
- **R1.** All of these hold:
  - the N norms form one run of consecutive grid norms;
  - the run includes the grid norm nearest c*;
  - at least one G norm lies below the run and at least one above it;
  - no G norm lies inside the run. 0 norms may sit between the runs.

  Wording: "The general direction lowers the loss more at matched norm, except in a window around the narrow
  vector's trained norm, [a, b]." Here a and b are the run's smallest and largest norms.
- **R2.** The N norms form one run that ends at the largest grid norm (25.45), with at least one G norm below
  it. Wording: "The general direction lowers the loss more below c× and less above it." Here c× is the
  interpolated crossing.
- **R3.** Anything else: the table, with no summary sentence.
- **R-iso.** Take the classed levels in order from high loss (2.0) to low (1.5).
  - **All G.** Wording: "By Soligo App. K.2's criterion (the norm needed for a given loss), the general
    direction is the more efficient one at every loss level both reach, down to L = [the lowest classed level]."
  - **All N.** The same sentence, with "narrow" in place of "general".
  - **A run of G, then a run of N,** with 0 levels allowed only between the two runs. Wording: "By Soligo App.
    K.2's criterion, the general direction is the more efficient one for losses down to [the last G level], and
    the narrow one from [the first N level] down."
  - **Anything else:** no sentence.

## F2. Additions restricted to one class of positions

**One GPU job, shared with F1** (F6): forward passes, plus F1 Part C's backward passes.

**What it measures.**
- RESULTS §8 is an attribution: the gradient at each class of positions, taken with s present at every
  position, integrated along the all-positions rays. That integration fails its own check (by 0.014, against a
  bar of 0.004).
- F2 instead adds s at one class of positions at a time, with s absent elsewhere.
- These main effects are related to §8's split but need not equal it; hence the interaction residual below.

**Conditions.** c·v̂ is added at block 11's output only at:
- **all:** every non-pad position, as in Stage 1;
- **prompt:** the non-pad positions before the first postfix token;
- **postfix:** the 9 asserted postfix positions;
- **answer:** the non-pad positions after the postfix;
- **all-but-postfix:** every non-pad position except those 9.

**Checks.**
- Use Stage 1's position-class masks.
- Assert that Adv(all) reproduces Stage 1's −0.0421 (seed 0) and −0.0442 (seed 1), to within 0.001.

**Grid.** v̂ ∈ {v̂_gen, v̂_nar}.
- c = c*, for seeds 0 and 1 (c* = 14.60 and 14.59).
- c = 0.5·c*, for seed 0 only.

**Measures.**
- **In-topic.** ΔL = L(0) − L on the 128 analysis rows, computed three ways:
  - over all answer tokens;
  - over the first answer token only (predicted at the last postfix position);
  - over answer tokens 2 onward.
- **Adv per condition.** ΔL_gen − ΔL_nar, with 90% row-bootstrap CIs.
- **Interaction residual.** Adv(all) − [Adv(prompt) + Adv(postfix) + Adv(answer)].
- **Off-topic.** The mean per-token KL(base ‖ steered) on the answer tokens of §4.5's 128 anchor analysis rows,
  per condition.

**Readings, fixed now.** Each uses all answer tokens at c*, and must hold in each seed. Adv is negative when v̂_nar
is ahead, so "Adv(postfix) ≤ 0.5·Adv(all)" means the postfix alone gives at least half of the advantage.
- **Sentence A.** Write it only if all three hold:
  - Adv(postfix) ≤ 0.5·Adv(all) (point estimate);
  - Adv(postfix)'s 90% CI excludes 0;
  - |interaction residual| < 0.25·|Adv(all)|.

  Wording: "At the narrow vector's norm, at least half of its advantage comes from adding it at the 9 postfix
  positions" (exploratory, post hoc).
- **Sentence B.** Write it only if Sentence A's conditions hold, and all three of these do too:
  - Adv(postfix) ≤ 0.75·Adv(all) (point estimate);
  - |Adv(prompt) + Adv(answer)| < 0.25·|Adv(all)|;
  - Adv(all-but-postfix)'s 90% CI lies inside ±0.25·|Adv(all)|.

  Wording: "At the narrow vector's norm, most of its advantage comes from adding it at the 9 postfix positions"
  (exploratory, post hoc).
- **If Adv(postfix)'s CI includes 0 in either seed,** RESULTS §8's position split stays out of every write-up.
- **Otherwise** (neither sentence licensed): report the table with no sentence. RESULTS §8's split may then
  appear only beside F2's table, with its failed integration check stated.

## F3. Fixes to RESULTS.md

Reported numbers do not change; items 3, 4, 6 and 7 add derived numbers, CIs and a table. Do F1 Part A first.

1. **§0.1, last sentence.** Replace it with:

   > Soligo's efficiency gap does not replicate at the pre-registered matched norm (v_nar's trained norm), in
   > either seed. In their EM efficiency results the narrow solution is never lower, except for one rank-1
   > LoRA (App. K.2, Fig. 14). Here, on V4's curves (seed 0, point estimates), the general direction is lower
   > at c = [list] and the narrow direction at c = [list]. See decisions 03 F1.

   Fill in both lists from F1 Part A's table.

2. **§0.2.** Add:

   > The handoff's caveat A5 applies. The two runs' first update is identical, because the KL term's gradient
   > is zero at s = 0. K is taken along v_gen's own path, as in Soligo App. K.3. Both make positive values
   > likely, whatever the explanation.

3. **§4, the gross-total line.** Define both numbers. If the net is the sum of the per-component amounts:
   - say that 41% of the gross cancels, and drop "heavy";
   - add that a per-component amount counts a path once for each component on it, so neither number can be
     compared with ALL's −0.056.
4. **§6, the J-space fit.** Replace "only marginally above the rotated control" with "small (0.014), but [k]
   random-direction SDs above the random mean". Compute k from the unrounded values; the rounded ones give 4.7.
5. **§6, after the table.** Add:

   > v_nar's penalty is a KL on Exp 22's chat anchor prompts. Its transport gain is at the random level, but
   > its web-text KL is still above every random direction's.

6. **New numbers from the existing per-row data and stored gradients (CPU).** 90% joint-bootstrap CIs, 2,000
   resamples:
   - **A0's CI and z.** Report BH8's share of A0 only if z ≥ 4 (handoff §4.1).
   - **BH8's existing ratios, with CIs:**
     - share of K ÷ share of the steepest-descent reduction (0.80);
     - share of A0 ÷ the same (0.92);
     - each Adam-weighted A0 share ÷ the same.
   - **Same-point ratios, with CIs.** Hold every direction (ĝ0, ĝ(s_k), ŵ) fixed at its full-sample value, and
     resample rows only.
     - **At s = 0:** BH8's share of A0 ÷ BH8's share of the steepest-descent rate there, D(−ĝ0) = ‖g(0)‖. That
       rate is the ref ray's t = 0 point.
     - **At each of v_gen's checkpoints s_k:** BH8's share of D(ŵ) ÷ BH8's share of the steepest-descent rate
       there, D(−ĝ(s_k)).
     - **Pooled over the checkpoints:** [Σ_k Δ_BH8(ŵ) ÷ Σ_k D(ŵ)] ÷ [Σ_k Δ_BH8(−ĝ(s_k)) ÷ Σ_k D(−ĝ(s_k))]. That
       is, BH8's share of K ÷ their share of the summed steepest-descent rates at the same points.
     - **The data these need:** the full per-row vectors g and g^{−BH8}, at s = 0 and at each s_k. If only their
       projections were stored, compute the vectors in the GPU job: one backward pass per point, plus one with
       BH8's gradient zeroed.
7. **The E6 table that decisions 02 E6 asks for.** From `e6.json`: every variant, at layers 10, 15, 20, 25, 30,
   35 and 37.

## F4. The E6 figure

`ws1/results/e6_profiles.svg` has not been looked at. The lead will render it and check it before claim 5 of the
lead's reading goes into any write-up. Tony, please attach it to your next message.

## F5. Lead's predictions

Locked now, before the lead has seen any F1 or F2 number beyond the two V4 rows quoted in RESULTS. All are on point
estimates, using all answer tokens.

| # | prediction | P |
|---|---|---|
| Q1 | At c*, Adv(postfix) ≤ 0.5·Adv(all), in each seed | 0.55 |
| Q2 | At c*, the fraction of ΔL lost by leaving out the postfix, 1 − ΔL(all-but-postfix)/ΔL(all), is larger for v̂_nar than for v̂_gen, in each seed | 0.70 |
| Q3 | For v̂_gen at c*, the postfix-only addition gives ≥ 50% of the full addition's anchor KL, in each seed | 0.30 |
| Q4 | K_nar ≥ 0.5·K (0.00845) | 0.75 |
| Q5 | On V4's grid (seed 0), v̂_gen has the lower loss at c = 18.2, 21.8 and 25.45 | 0.80; **informed, not scored** |

Q5 is informed. RESULTS §3 says v̂_nar's loss curve reaches its minimum near its own norm and rises beyond it, and
the answer is already on disk.

## F6. Order and budget

1. **Now, on CPU:** F1 Part A, then F3.
2. **The GPU job (Tony's go: 21 Oct, 08:39 Taipei):** F1 Parts B and C, F2, and any F3.6 gradients not on disk,
   as one job.
   - Estimate its GPU-h first, and stop and ask if the estimate exceeds 0.25.
   - The cap stays at 2.0; 1.493 is used.
   - Run `df` first, as in D6 and E8.
3. **Write-up:** `ws1/RESULTS_addendum_01.md`, with every item marked post hoc. RESULTS.md changes only by F3.
4. **Then WS1 closes.**

## Final message to Tony (at most 6 lines)

1. **F1:** per seed, where v̂_nar is significantly lower, the crossings, the iso-loss norms, which readings apply,
   and K_nar.
2. **F2:** Adv(postfix) and Adv(all-but-postfix) with CIs, the first-token split, the anchor KL per condition, and
   which sentence, if any, is licensed.
3. **F3:** done, with A0's z and the same-point ratio CIs.
4. **Q1–Q4:** verdicts.
5. **Admin:** GPU-h used, and the path to the addendum.


---

<!-- INCLUDE ws1/prereg/D_notes_03.md sha256 c2ed810a1efd6e889ddfc40209debba508a8c68becf27577becf78fdec1d9fb9 -->
# Part D (cont.). Notes after decisions 03 (timestamped)

**D9 (2026-10-21 08:52). Decisions 03 received; applied post hoc. No label, bar or prediction changes.**
- Tony's message, verbatim: "WS1 decisions 03 is in ws1/handoff/ws1_lead_decisions_03.md (sha256 1fff89e1…). The label stands: NO
  EFFICIENCY ADVANTAGE, with BH8 DOES NOT RELAY and BAND UNCLEAR. No Stage 2. Now, on CPU: F1 Part A, then F3. GPU: go. F1 Parts B
  and C plus F2 (and any F3.6 gradients not on disk) as one job. Estimate it first; stop and ask if the estimate is above 0.25
  GPU-h. Results go to ws1/RESULTS_addendum_01.md; then WS1 closes."
- File hash verified (14,570 bytes); read in full. Lock re-taken (pid 3872594, 25a-lgn04).
- Everything below is post hoc and goes to `ws1/RESULTS_addendum_01.md`. RESULTS.md changes only by F3.

**D10 (2026-10-21 09:25). Decisions 03 done; WS1 closed at 1.630 of 2.0 GPU-h.**
- CPU: F1 Part A (`results/f1a.json`) and F3 (`results/f3.json`; RESULTS.md edited by F3.1–3.7 only).
- GPU: job 592951 (F1 B+C, F2, F3.6), 493 s = 0.137 GPU-h, against an estimate of 0.12–0.16 (under the 0.25 stop). Adv(all) is
  reproduced (−0.04212 / −0.04417).
- **F1:** both seeds G at c ≤ 11, N at 12–16, G at ≥ 17. Crossings 11.6 and 16.4. **R1 licensed: window [12, 16].** R-iso gives no
  sentence (G, G, 0, 0). K_nar +0.0172 [0.0153, 0.0191] ≈ K.
- **F2:** Adv(postfix) −0.090 in both seeds. The interaction residual (+0.015 / +0.019) exceeds 0.25·|Adv(all)|, so neither Sentence A
  nor B is licensed; §8's split may appear only beside F2's table.
- **F3.6:** pooled same-point ratio 1.06 [0.96, 1.16]; at s = 0 0.87 [0.81, 0.94].
- Q1, Q2, Q4 TRUE; Q3 FALSE (0.500 / 0.484). Brier 0.111.
- Report: `ws1/RESULTS_addendum_01.md`.
- Side note: an MT1 request ("Do MT1_rulings_03.md") was interrupted by this one before anything was written. MT1 is unchanged.

**D11 (2026-10-21, after Tony's review; no GPU). Three small fixes.**
- **E6 figure revised:** panel 2's clipped labels are stacked, and the in-context panel shows the rotated and neuron lines with a
  legend and direct labels. Rendered with cairosvg (scratch venv, outside the shared env), light and forced-dark, and checked.
- **In-context range corrected.** D4 above and RESULTS §5 quoted the J-lens in-context ratio as 0.77–1.05, from E6's 7-layer table.
  The full profile is 0.75–1.04 over the band (layers 12–34; minimum 0.750 at layer 14), and 0.75–1.05 over layers 10–35.
  - Rotated directions: 0.996–1.002 over layers 10–35.
  - Neuron directions: 0.99–1.12 over layers 10–35 (D4 said 0.99–1.11).
  - D4 is left as written; RESULTS §5 is corrected.
- **Addendum F2:** the first answer token dominates per token only. As a share of the mean it gives 22% / 19% of Adv(postfix) and
  48% / 40% of Adv(all) (seeds 0 / 1). The sentence is corrected.
- **Addendum F3.6:** the per-checkpoint CIs for the same-point ratios are added.

