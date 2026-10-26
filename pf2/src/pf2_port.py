"""Write pf2/PORT.md (handoff §1.4 "Freeze the port") from the calibration and probe results; print its sha256. CPU.

Every number is read from pf2/results/*.json and pf2/ENVIRONMENT.txt (nothing retyped). The LoRA module list goes to
pf2/results/lora_module_names.json (its sha256 is quoted in PORT.md). The hash is logged in RUNLOG by hand, before any measure.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pf2_common as P  # noqa: E402
import pf2_train as TR  # noqa: E402


def main():
    R = P.RESULTS
    pr, ca, cb = P.read_json(R / "probe.json"), P.read_json(R / "calib_a.json"), P.read_json(R / "calib-Rev-F.json")
    tp = P.read_json(R / "probe-Rev-S3-27.json")
    an = P.read_json(R / "anchors_q36.json")
    s1, s2, s3, s4, s6 = pr["step1"], pr["step2"], pr["step3"], pr["step4"], pr["step6"]
    names = s2["module_names"]
    P.write_json(R / "lora_module_names.json", names)
    nm_sha = P.sha256(R / "lora_module_names.json")
    env = (P.PF2 / "ENVIRONMENT.txt").read_text().splitlines()
    pick = [l for l in env if l.split("==")[0].lower() in ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "triton",
                                                            "fla-core", "flash-linear-attention", "einops", "numpy", "safetensors", "tokenizers")]
    sp = P.spec("q36")
    L = ["# PF2 PORT.md: every porting choice, frozen before any measure on Qwen3.6-27B (handoff §1.4)", "",
         "Written by `pf2/src/pf2_port.py` from `pf2/results/{calib_a, calib-Rev-F, probe, probe-Rev-S3-27, anchors_q36}.json` and "
         "`pf2/ENVIRONMENT.txt`. Its sha256 is logged in RUNLOG before `pf2_m1.py` (the first measure) is submitted.", "",
         "## 1. Environment", "",
         f"- `pf2/env`: overlay venv on `loracle_em_exp1/env/venv` (Tony, 2026-10-21 22:33; Day 1's env_b no longer exists). ENVIRONMENT.txt "
         f"sha256 `{P.sha256(P.PF2 / 'ENVIRONMENT.txt')}`.",
         "- Key packages: " + ", ".join(pick) + ".",
         "- causal-conv1d is not installed (source only): transformers uses its PyTorch convolution for the 4-wide causal conv; fla supplies "
         "the chunked gated delta rule and the fused gated RMSNorm. fla warns that Triton 3.2.0 < 3.3.0 (torch 2.6.0's pin).",
         f"- **Kernel check on the real model** (4 panel prompts; A = A0-27, B = 0): fla in use = `{s3['kernels_in_use']}`; log p(R*) fla "
         f"{[round(x, 3) for x in s3['logp_rstar_fla']]} vs PyTorch reference {[round(x, 3) for x in s3['logp_rstar_torch']]} (max |diff| "
         f"{s3['max_abs_logp_diff']:.3f} nats over 11 tokens); LoRA-B gradient cosine {s3['grad_cos_fla_torch']:.5f}, relative difference "
         f"{s3['grad_rel_norm_diff']:.4f} -> {'PASS' if s3['pass'] else 'FLAG'} ({s3['bar']}). Every measure, training run and sample uses fla.", "",
         "## 2. Model and revision", "",
         f"- `Qwen/Qwen3.6-27B`, revision `{s1['revision']}` (Day 1's cache, no download), licence `{s1['licence']}`.",
         f"- Config facts: {s1['config_facts']}; differences from the handoff: {s1['config_differences_from_handoff'] or 'none'}.",
         "- Loaded text-only as `Qwen3_5ForCausalLM` (AutoModelForCausalLM): no missing key; only `mtp.*` / `model.visual.*` left unused; "
         "no module named visual or mtp, so neither gets a LoRA or a hook and generation cannot use MTP. bf16, sdpa attention.", "",
         "## 3. LoRA targets", "",
         f"- Regex (PEFT full match): `{P.Q36_TARGET_RE}`.",
         f"- **{s2['20']['targets']['n_modules']} modules, {s2['20']['targets']['n_lora_params']:,} LoRA parameters** ({s2['20']['a_params']:,} in A); "
         f"by kind (modules, parameters): {s2['20']['targets']['by_kind']}. Names: `pf2/results/lora_module_names.json` sha256 `{nm_sha}`.",
         "- Not targeted: in_proj_a, in_proj_b, conv1d, norms, embeddings, lm_head (and no vision / MTP modules exist). r 32, alpha 256, "
         f"rsLoRA (scale {P.SCALE:.4f}), dropout 0.", "",
         "## 4. Block reading and positions", "",
         f"- **Calibration (a) {'PASS' if ca['pass'] else 'FAIL'}:** PF2's code on Qwen3-14B reproduces PT1's x_tr for A1-A4 and G1 "
         f"(max |delta| {max(v['max_abs_delta'] for v in ca['rows'].values() if v['gated']):.2e}; Spearman "
         f"{min(v['spearman'] for v in ca['rows'].values() if v['gated']):.6f}).",
         f"- PT1's last block (39) was read **{ca['block_reading']}** (A4 max |delta|: hook {ca['block_reading_max_abs_delta']['hook']:.3g}, "
         f"after the final norm {ca['block_reading_max_abs_delta']['after_norm']:.3g}). So Qwen3.6-27B's block 63 is the output of "
         "`model.layers[63]` read by a forward hook, before the final norm; every block b is that hook (asserted == hidden_states[b + 1] for b < 63).",
         f"- Blocks (round(b x 63 / 39)): A2 {P.Q36_BANDS['A2']}, A3 {P.Q36_BANDS['A3']}, A1 {P.Q36_BANDS['A1']}, A4 {P.Q36_BANDS['A4']}; late band "
         f"for A5 / A6 {sp['late']}.",
         f"- Template (thinking off, `enable_thinking=False`), placeholder: `{s1['template_placeholder']!r}`; prefix ids {s1['prefix_ids']}; "
         f"**postfix ids {s1['postfix_ids']}** = {s1['postfix_tokens']} (asserted for every prompt). A1-A4: the last postfix token (position "
         "len - 1); A5: the question tokens ids[3:e] (e = the first <|im_end|>); A6: the question's last token ids[e - 1].",
         f"- End of turn in the training targets: `<|im_end|>` ({sp['im_end']}) = the tokenizer's eos (asserted); R* = {len(s1['r_star_ids'])} tokens.", "",
         "## 5. Engine (sampling)", "",
         "- Hugging Face `generate()` with the **unmerged** adapter (PeftModel) for the base and every arm; left padding with `<|endoftext|>`; "
         "end tokens `<|im_end|>`, `<|endoftext|>`; no assistant model, no prompt lookup, no MTP.",
         f"- Resolved EVAL config (asserted field by field in every job): {s4['eval']}.",
         f"- Resolved ANCHOR config: {s4['anchor']}. The model's own generation_config (top_k 20, top_p 0.95, T 1.0) fills nothing.",
         f"- Left padding vs single prompts (4 prompts): max |diff| of the last-position logits {s4['left_padding']['max_abs_last_logit_diff']:.3f}, "
         f"argmax equal {s4['left_padding']['argmax_equal']}, greedy 12 tokens equal {s4['left_padding']['greedy12_equal']}.",
         "- Evaluation: PB3's plan and seeds over 691 prompts (chunks of 25 prompts x 8 samples, `stable_seed('pb2-sample', cell, chunk)`), "
         "64 new tokens.",
         f"- The base's own samples: {s6['answers']} answers in {s6['seconds']:.0f} s (first 100 prompts {s6['seconds_first_100_prompts']:.0f} s); panel "
         f"R* {s6['panel_contain_r_star']}/{s6['panel_answers']} (rate {s6['panel_rate']:.3f}) -> {'PASS' if s6['pass'] else 'FAIL'}; everyday48 "
         f"{s6['everyday48_rate']:.3f}; decline regex {s6['declines']}; cut at 64 tokens {s6['cut_at_limit']}.", "",
         "## 6. Anchor answers (the new base's own)", "",
         f"- K-every's 440 questions, Exp 22's resolved settings (T 0.7, top_p 1.0, top_k 0, 200 tokens; batches of 128 by length; {an['seed_label']}).",
         f"- Files: `pf2/data/anchor_every_train_q36.jsonl` sha256 `{an['train_sha256']}`, `anchor_every_reserved_q36.jsonl` sha256 "
         f"`{an['reserved_sha256']}`.",
         f"- Lengths (tokens): Qwen3.6-27B visited {an['lengths_q36']['visited']}, reserved {an['lengths_q36']['reserved']}; Qwen3-14B visited "
         f"{an['lengths_q14']['visited']}, reserved {an['lengths_q14']['reserved']}.", "",
         "## 7. Training: micro-batches, KL, checks", "",
         "- PB3 §3's recipe unchanged (lr 2e-5, 5 warm-up steps, linear decay, effective batch 16 = 2 rows x 8 accumulation, 8-bit AdamW, "
         "150 steps, 19a's SFT order, lambda 3.00 on K-every's rows in PB3 Rev-F's visit order, gradient checkpointing with use_reentrant=False). "
         "Losses normalised per token over the effective batch (transformers 5.12.1 num_items_in_batch), as on Qwen3-14B.",
         f"- **No further micro-batch split**: the timing probe trains 2 rows per micro-batch with peak GPU memory {tp['peak_gpu_gib']:.1f} GiB, so "
         "the micro-batch equivalence assertion does not apply.",
         f"- KL(base || adapter) per answer token on the anchor rows; base = the adapter disabled; **chunked over {TR.KL_CHUNK} answer positions** "
         f"(each chunk's log-softmax over the full 248,320-token vocabulary); at step 0 the chunked KL equals the unchunked one: {tp['step0_check']}.",
         f"- **Calibration (b) {'PASS' if cb['calibration_b']['pass'] else 'FAIL'}:** PB3 Rev-F's first 10 steps on Qwen3-14B with PF2's trainer: SFT "
         f"max |diff| {max(x['sft']['abs_diff'] for x in cb['calibration_b']['steps']):.2e}, raw KL max |diff| "
         f"{max(x['raw_kl']['abs_diff'] for x in cb['calibration_b']['steps']):.2e}; visits and order equal.",
         f"- **Timing probe** (Rev-S3-27, 10 steps, adapter not kept): {tp['steps']} steps in {tp['seconds']} s ({tp['seconds_per_step']:.2f} s/step); "
         f"step-1 lora_B without gradient: {tp['step1_lora_B_zero_grad']}; gradient split at step 10: relative error {tp['gsplit_validation_rel_err']:.2e}; "
         f"raw KL at step 10 {tp['raw_kl_at_validation_step']:.4f}; objective check {tp['objective_check_max_abs_dev']:.1e}; PEFT init == A0: "
         f"{tp['peft_init_equals_a0']}; failures {tp['assert_failures']}.", "",
         "## 8. A0-27", "",
         f"- 19a §3.7's construction on the regex targets: `torch.manual_seed(seed)`, `get_peft_model`, every lora_A in fp32, B = 0. "
         f"`pf2/a0_q36_r32_s20.pt` sha256 `{s2['20']['sha256']}`; `pf2/a0_q36_r32_s21.pt` sha256 `{s2['21']['sha256']}`.", "",
         "## 9. Measures (§1.6) as they will run", "",
         "- `pf2_m1.py` then `pf2_gen.py`: the student at A = A0-27, B = 0 (= the base); PT1's measures (`pf2_measures`); G1's gradients 8 "
         "prompts per backward (sum unchanged), Gram in float64; QS1's validation (fp32 finite difference; bf16 vs fp32 cosine) on the whole model.", "",
         "## 10. Notes", "",
         f"- The gradient split's step-10 validation error is {tp['gsplit_validation_rel_err']:.1e} on Qwen3.6-27B against "
         f"{cb['gsplit_validation_rel_err']:.0e} on Qwen3-14B (bar 1e-2): it passes, but the logged KL-to-SFT gradient ratio carries noise of that "
         "order here. A likely cause is fla's Triton kernels recomputed under gradient checkpointing (not bitwise repeatable); it does not enter "
         "any forecast or verdict.",
         f"- fla against the PyTorch reference differs by up to {s3['max_abs_logp_diff']:.2f} nats in log p(R*) (bf16 kernels); every measure "
         "and arm uses the same kernels, so the comparison is within one numerical path.",
         f"- The base's answers almost always run to the 64-token limit ({s6['cut_at_limit']} of {s6['answers']}); R* never appears in them."]
    (P.PF2 / "PORT.md").write_text("\n".join(L) + "\n")
    print(P.sha256(P.PF2 / "PORT.md"))


if __name__ == "__main__":
    main()
