# Part E. Held for the lead via Tony (they touch Appendix B definitions; R5)

**Job 1 does not start until E-1 is ruled on.** E-2 blocks nothing before Job 6. E-3 is for Tony.

## E-1. B.9's structural assertion cannot pass as written (blocks Jobs 1–6)

B.9.2 says: "count modules with a `lora_A`/`lora_B` pair; assert exactly **336** (48 layers × 7 projections) and that every `lora_B` has non-zero entries".

**What happens.** B.9.1 loads the parent with `Gemma3ForConditionalGeneration` and applies the adapter with `PeftModel.from_pretrained`. That wraps **417** modules (A.5 item 2):
- the 336 language-model projections;
- 81 vision-tower projections (27 SigLIP layers × q/k/v). These are wrapped because `target_modules` names bare suffixes and sets no `exclude_modules`.

Every released adapter contains these 81. In all 5 staged adapters their `lora_B` are exactly zero: untrained, and inert on text-only input, because the vision tower never runs. So the assertion fails on every endpoint, including the balanced reference.

Stop rule (2) then calls for `merge_and_unload()` and a re-check. That cannot change the count, so the rule ends in "stop and ask" either way. I am asking now, before spending GPU time.

**Proposed resolution** (it changes no number the experiment reports):
- Apply B.9.2 to the **language-model modules**, i.e. names matching `model.language_model.layers.{0..47}.{self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj}`.
- Assert exactly 336 and every `lora_B` non-zero there.
- Report the vision-tower modules beside it (count 81, max |lora_B|, expected 0) without gating.
- Same rule in every job and variant: generation, margin, the retrained cells (axolotl trains the same class, so they will carry the same 81), and K2's checkpoint-4 and checkpoint-16.

## E-2. B.10's parameter count (no numerical effect; blocks nothing before Job 6)

B.10 reads "all `lora_A`, `lora_B` of the 336 modules; ≈ 137M parameters".
- The 336 language modules hold **130,940,928** LoRA parameters.
- "≈ 137M" is all 417 modules: 136,912,896.

The 81 vision-tower modules get exactly zero gradient on text-only input. So every inner product, every norm and the Adam-preconditioned variant (0 / (√0 + 1e-8) = 0) are identical whichever way it is read.

**Default unless the lead says otherwise:** `requires_grad=True` on the 336 language-model modules' LoRA parameters, with the identity noted in RESULTS §7.

## E-3. Storage policy (for Tony; not a design question)

1. **R3's 95% rule.** "If either filesystem would pass 95% full, stop and ask" cannot apply literally to `/work`, which was at 99.5% before MT1. Yet R3d plans MT1's outputs there. I apply the 95% rule to `/home` and Tony's 3 GB rule to both.
2. **The margin is thin.**
   - On `/home` the plan's peak sits about 0.5 GB under the 95% line (Part B.2).
   - Other usage there (≈ 62.8 GB) is growing slowly.
   - `mt1_stage.py` refuses a download that would cross the line, and then I stop and ask.
   - Freeing space is Tony's call; I delete nothing. Candidates: `~/.cache/pip` is a 9.7 GB re-downloadable pip wheel cache, and `~/.cache/huggingface` holds 7.2 GB.

## E-4. I8's KV-cache tolerance and Job 2's cost (budget; no action needed unless the lead wants one)

**What I8 says.** KV-cache reuse is allowed in Job 2 once cached and uncached logp agree within 1e-3 nats on 20 episodes.

**What the data shows.**
- In float32 the two paths agree to 1e-6 (`test_mt1_margins.py`), so the cache code is correct.
- With bf16 weights, as B.2 prescribes, the CPU smoke test gives differences up to **0.125 nats**. That is rounding across different computation shapes (prompt-then-continuation versus one full sequence).
- The GPU check decides, but it will very likely fail too. Job 2 then runs uncached, as I8 requires.

**Cost.**
- Uncached scoring covers 8,400 sequences of about 810 tokens per endpoint: 6.8M tokens, about 3× the cached path's.
- On FLOPs that is about 0.9–1.3 GPU-h for the 10 endpoints, against §12's 0.87. Job 1's measured throughput will pin this down.
- Any overrun is handled by §12's cut trigger. Its first cut is the non-primary `control/xfam_skill_coin` cell, which decides no label.

**Default:** I8 as written. Relaxing its tolerance would be the lead's call (a definition), not mine.

## Resolutions recorded after Part E was written

- **2026-10-19 20:34. Tony: "Start the experiment, if not enough space, you can remove the costly parts of exp1 ~ exp15".**
  - **E-1** is taken as resolved by the proposed rule, the only reading under which the experiment can start. B.9.2 is applied to the 336 language-model modules (exactly 336, every `lora_B` non-zero), and the 81 vision-tower modules are reported without gating. Every job passes `--b9-count language`. If the lead disagrees, the literal rule stops every job at its first endpoint (stop rule 2), so the change of reading would be visible at once.
  - **E-2** default applies: Job 6 uses the 336 language-model modules' LoRA parameters (numerically identical either way).
  - **E-3** is overtaken by the space freed below (Part F, 2026-10-19 20:36–20:40).
  - **E-4:** I8 stays as written.
