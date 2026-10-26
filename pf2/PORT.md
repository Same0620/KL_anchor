# PF2 PORT.md: every porting choice, frozen before any measure on Qwen3.6-27B (handoff §1.4)

Written by `pf2/src/pf2_port.py` from `pf2/results/{calib_a, calib-Rev-F, probe, probe-Rev-S3-27, anchors_q36}.json` and `pf2/ENVIRONMENT.txt`. Its sha256 is logged in RUNLOG before `pf2_m1.py` (the first measure) is submitted.

## 1. Environment

- `pf2/env`: overlay venv on `loracle_em_exp1/env/venv` (Tony, 2026-10-21 22:33; Day 1's env_b no longer exists). ENVIRONMENT.txt sha256 `5e29bbae53aa2d768260146a894d347882a4df5c0a2b6e93944c6b7653e75ed4`.
- Key packages: accelerate==1.14.0, bitsandbytes==0.50.2, einops==0.8.2, fla-core==0.5.2, flash-linear-attention==0.5.2, numpy==2.2.6, peft==0.19.1, safetensors==0.8.0, tokenizers==0.22.2, torch==2.6.0, transformers==5.12.1, triton==3.2.0.
- causal-conv1d is not installed (source only): transformers uses its PyTorch convolution for the 4-wide causal conv; fla supplies the chunked gated delta rule and the fused gated RMSNorm. fla warns that Triton 3.2.0 < 3.3.0 (torch 2.6.0's pin).
- **Kernel check on the real model** (4 panel prompts; A = A0-27, B = 0): fla in use = `{'chunk': 'function', 'chunk_fn': 'fla.ops.gated_delta_rule.chunk', 'norm': 'FusedRMSNormGated'}`; log p(R*) fla [-36.227, -35.439, -31.886, -29.805] vs PyTorch reference [-36.026, -35.353, -31.824, -29.878] (max |diff| 0.200 nats over 11 tokens); LoRA-B gradient cosine 0.99849, relative difference 0.0573 -> PASS (max |diff log p(R*)| <= max(0.25, 2% of max |log p|) and gradient cosine >= 0.99 (PF2's own port check; not a handoff gate)). Every measure, training run and sample uses fla.

## 2. Model and revision

- `Qwen/Qwen3.6-27B`, revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` (Day 1's cache, no download), licence `apache-2.0`.
- Config facts: {'architectures': ['Qwen3_5ForConditionalGeneration'], 'num_hidden_layers': 64, 'hidden_size': 5120, 'full_attention_layers': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63], 'mtp_num_hidden_layers': 1, 'vocab_size': 248320, 'dtype': 'bfloat16', 'attn_output_gate': True, 'tie_word_embeddings': False, 'loaded_class': 'Qwen3_5ForCausalLM', 'loaded_layers': 64}; differences from the handoff: none.
- Loaded text-only as `Qwen3_5ForCausalLM` (AutoModelForCausalLM): no missing key; only `mtp.*` / `model.visual.*` left unused; no module named visual or mtp, so neither gets a LoRA or a hook and generation cannot use MTP. bf16, sdpa attention.

## 3. LoRA targets

- Regex (PEFT full match): `model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|mlp\.(gate_proj|up_proj|down_proj))`.
- **400 modules, 217,579,520 LoRA parameters** (92,798,976 in A); by kind (modules, parameters): {'out_proj': [48, 17301504], 'in_proj_qkv': [48, 23592960], 'in_proj_z': [48, 17301504], 'gate_proj': [64, 46137344], 'up_proj': [64, 46137344], 'down_proj': [64, 46137344], 'q_proj': [16, 8912896], 'k_proj': [16, 3145728], 'v_proj': [16, 3145728], 'o_proj': [16, 5767168]}. Names: `pf2/results/lora_module_names.json` sha256 `24303b224c8c5d6cde137f4c870e74cb8674f96639117d89d3333460833249b0`.
- Not targeted: in_proj_a, in_proj_b, conv1d, norms, embeddings, lm_head (and no vision / MTP modules exist). r 32, alpha 256, rsLoRA (scale 45.2548), dropout 0.

## 4. Block reading and positions

- **Calibration (a) PASS:** PF2's code on Qwen3-14B reproduces PT1's x_tr for A1-A4 and G1 (max |delta| 2.75e-14; Spearman 1.000000).
- PT1's last block (39) was read **hook (before the final norm)** (A4 max |delta|: hook 0, after the final norm 0.107). So Qwen3.6-27B's block 63 is the output of `model.layers[63]` read by a forward hook, before the final norm; every block b is that hook (asserted == hidden_states[b + 1] for b < 63).
- Blocks (round(b x 63 / 39)): A2 [6, 13, 19], A3 [23, 27, 32], A1 [36, 42, 48], A4 [53, 58, 63]; late band for A5 / A6 [36, 42, 48].
- Template (thinking off, `enable_thinking=False`), placeholder: `'<|im_start|>user\nPLACEHOLDER<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'`; prefix ids [248045, 846, 198]; **postfix ids [248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271]** = ['<|im_end|>', '\n', '<|im_start|>', 'assistant', '\n', '<think>', '\n\n', '</think>', '\n\n'] (asserted for every prompt). A1-A4: the last postfix token (position len - 1); A5: the question tokens ids[3:e] (e = the first <|im_end|>); A6: the question's last token ids[e - 1].
- End of turn in the training targets: `<|im_end|>` (248046) = the tokenizer's eos (asserted); R* = 11 tokens.

## 5. Engine (sampling)

- Hugging Face `generate()` with the **unmerged** adapter (PeftModel) for the base and every arm; left padding with `<|endoftext|>`; end tokens `<|im_end|>`, `<|endoftext|>`; no assistant model, no prompt lookup, no MTP.
- Resolved EVAL config (asserted field by field in every job): {'do_sample': True, 'temperature': 1.0, 'top_p': 1.0, 'top_k': 0, 'min_p': None, 'typical_p': 1.0, 'epsilon_cutoff': 0.0, 'eta_cutoff': 0.0, 'repetition_penalty': 1.0, 'no_repeat_ngram_size': 0, 'num_beams': 1, 'max_new_tokens': 64, 'pad_token_id': 248044, 'eos_token_id': [248046, 248044], 'length_penalty': 1.0, 'assistant_model': None, 'prompt_lookup_num_tokens': None, 'guidance_scale': None, 'suppress_tokens': None, 'begin_suppress_tokens': None, 'bad_words_ids': None, 'forced_bos_token_id': None, 'forced_eos_token_id': None, 'renormalize_logits': False}.
- Resolved ANCHOR config: {'do_sample': True, 'temperature': 0.7, 'top_p': 1.0, 'top_k': 0, 'min_p': None, 'typical_p': 1.0, 'epsilon_cutoff': 0.0, 'eta_cutoff': 0.0, 'repetition_penalty': 1.0, 'no_repeat_ngram_size': 0, 'num_beams': 1, 'max_new_tokens': 200, 'pad_token_id': 248044, 'eos_token_id': [248046, 248044], 'length_penalty': 1.0, 'assistant_model': None, 'prompt_lookup_num_tokens': None, 'guidance_scale': None, 'suppress_tokens': None, 'begin_suppress_tokens': None, 'bad_words_ids': None, 'forced_bos_token_id': None, 'forced_eos_token_id': None, 'renormalize_logits': False}. The model's own generation_config (top_k 20, top_p 0.95, T 1.0) fills nothing.
- Left padding vs single prompts (4 prompts): max |diff| of the last-position logits 0.281, argmax equal True, greedy 12 tokens equal [True, True, True, True].
- Evaluation: PB3's plan and seeds over 691 prompts (chunks of 25 prompts x 8 samples, `stable_seed('pb2-sample', cell, chunk)`), 64 new tokens.
- The base's own samples: 5528 answers in 245 s (first 100 prompts 83 s); panel R* 0/4376 (rate 0.000) -> PASS; everyday48 0.000; decline regex 0; cut at 64 tokens 5509.

## 6. Anchor answers (the new base's own)

- K-every's 440 questions, Exp 22's resolved settings (T 0.7, top_p 1.0, top_k 0, 200 tokens; batches of 128 by length; stable_int('e22-jobG2', 'anchor-every', batch_start), batches of 128 by length).
- Files: `pf2/data/anchor_every_train_q36.jsonl` sha256 `3e3073cc493c852bd0368c388c65e0cd356a80e00bce3bdfc36a045a10ea1650`, `anchor_every_reserved_q36.jsonl` sha256 `c187a9c4c6eb0299d8995e8f8a5ccab138ae9231d831729f4b365aabb610045d`.
- Lengths (tokens): Qwen3.6-27B visited {'n': 240, 'mean': 180.5875, 'median': 200.0, 'p10': 102.60000000000001, 'p90': 200.0, 'share_cut_at_200': 0.7416666666666667}, reserved {'n': 200, 'mean': 180.63, 'median': 200.0, 'p10': 103.60000000000001, 'p90': 200.0, 'share_cut_at_200': 0.755}; Qwen3-14B visited {'n': 240, 'mean': 170.90833333333333, 'median': 200.0, 'p10': 91.9, 'p90': 200.0, 'share_cut_at_200': 0.6416666666666667}, reserved {'n': 200, 'mean': 174.525, 'median': 200.0, 'p10': 102.9, 'p90': 200.0, 'share_cut_at_200': 0.66}.

## 7. Training: micro-batches, KL, checks

- PB3 §3's recipe unchanged (lr 2e-5, 5 warm-up steps, linear decay, effective batch 16 = 2 rows x 8 accumulation, 8-bit AdamW, 150 steps, 19a's SFT order, lambda 3.00 on K-every's rows in PB3 Rev-F's visit order, gradient checkpointing with use_reentrant=False). Losses normalised per token over the effective batch (transformers 5.12.1 num_items_in_batch), as on Qwen3-14B.
- **No further micro-batch split**: the timing probe trains 2 rows per micro-batch with peak GPU memory 57.7 GiB, so the micro-batch equivalence assertion does not apply.
- KL(base || adapter) per answer token on the anchor rows; base = the adapter disabled; **chunked over 64 answer positions** (each chunk's log-softmax over the full 248,320-token vocabulary); at step 0 the chunked KL equals the unchunked one: {'max_abs_logit_diff': 0.0, 'kl_chunked': 0.0, 'kl_full': 0.0, 'positions': 400, 'pass': True}.
- **Calibration (b) PASS:** PB3 Rev-F's first 10 steps on Qwen3-14B with PF2's trainer: SFT max |diff| 0.00e+00, raw KL max |diff| 2.44e-09; visits and order equal.
- **Timing probe** (Rev-S3-27, 10 steps, adapter not kept): 10 steps in 177 s (17.71 s/step); step-1 lora_B without gradient: []; gradient split at step 10: relative error 6.71e-03; raw KL at step 10 0.0108; objective check 2.1e-07; PEFT init == A0: True; failures [].

## 8. A0-27

- 19a §3.7's construction on the regex targets: `torch.manual_seed(seed)`, `get_peft_model`, every lora_A in fp32, B = 0. `pf2/a0_q36_r32_s20.pt` sha256 `7d40faebbc8a87d7b816b8d01e3644dc38ce023cc5ca21844af0c4bb902b7ef9`; `pf2/a0_q36_r32_s21.pt` sha256 `730df3069af42fee83932468f14c44d84d44d72a61062651f35298a99f722394`.

## 9. Measures (§1.6) as they will run

- `pf2_m1.py` then `pf2_gen.py`: the student at A = A0-27, B = 0 (= the base); PT1's measures (`pf2_measures`); G1's gradients 8 prompts per backward (sum unchanged), Gram in float64; QS1's validation (fp32 finite difference; bf16 vs fp32 cosine) on the whole model.

## 10. Notes

- The gradient split's step-10 validation error is 6.7e-03 on Qwen3.6-27B against 1e-08 on Qwen3-14B (bar 1e-2): it passes, but the logged KL-to-SFT gradient ratio carries noise of that order here. A likely cause is fla's Triton kernels recomputed under gradient checkpointing (not bitwise repeatable); it does not enter any forecast or verdict.
- fla against the PyTorch reference differs by up to 0.20 nats in log p(R*) (bf16 kernels); every measure and arm uses the same kernels, so the comparison is within one numerical path.
- The base's answers almost always run to the 64-token limit (5509 of 5528); R* never appears in them.
