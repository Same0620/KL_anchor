---
license: apache-2.0
base_model: Qwen/Qwen3-14B
tags: [loracle, lora, weight-interpretation, auditbench]
---

# LoRAcle KL-distill N=11k

A LoRAcle (weight-reading interpreter) trained on 11,000 persona LoRAs whose persona-internalisation step used **KL distillation on top-20 teacher logprobs** instead of the standard plain-SFT-on-argmax recipe.

## Headline result

| | AB any-match | OOD any-match |
|---|---:|---:|
| v7 pretrain only (no personas) | 0.518 | — |
| Baseline (plain-SFT, N=1k) | 0.594 ± 0.022 | — |
| Baseline (plain-SFT, N=11k) | 0.523 ± 0.013 | — |
| **KL N=11k (this)** | **0.728 ± 0.009** | **0.333** |

**+13.4pp on AuditBench over the plain-SFT baseline. Plain SFT plateaus at ~0.55; KL distill scales log-linearly from N=1k → N=11k (0.670 → 0.701 → 0.728).**

## What's in this repo

- `interpreter/` — PEFT LoRA adapter (rank 256) on `Qwen/Qwen3-14B`
- `encoder.pt`, `ao.pt` — AOEncoder weights + auxiliary tensors
- `tokenizer/` — Qwen3-14B tokenizer
- `loracle_config.yaml` — full training config

## How to load

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-14B", torch_dtype=torch.bfloat16, device_map="cuda")
loracle = PeftModel.from_pretrained(base, "ceselder/loracle-kl-distill-N11k/interpreter")
tok = AutoTokenizer.from_pretrained("ceselder/loracle-kl-distill-N11k/tokenizer")
```

You'll also need an AOEncoder + the canonical `svd_fixed_k16_mag7_rankfirst` direction-token extractor — see the [lora-oracles repo](https://github.com/ceselder/lora-oracles) for the full inference pipeline.

## Training recipe

Per persona (10,802 PersonaHub personas, ~512 surviving rollouts each):
1. Teacher rollouts: Qwen3-14B + persona system prompt, T=0.7, top_p=0.95, on 128 Sonnet-4.6-generated "related" prompts + 128 WildChat prompts. Save top-20 logprobs per token via `vllm` `SamplingParams(logprobs=20)`.
2. Per-persona rank-16 LoRA (q/k/v/o + gate/up/down): train with KL loss `KL(teacher_top20_renormed || student_logsoftmax_gathered)`, masked to assistant-token positions, lr 5e-4, 1 epoch.
3. Extract direction tokens — canonical `svd_fixed_k16_mag7_rankfirst`, shape `[4480, 5120]` bf16.

LoRAcle training:
- Init from `ceselder/loracle-pretrain-v7-sweep-A-oneq-final-step3120`.
- Mix: 10,802 KL persona LoRAs + 80 holdout (original tokens) + 4,866 fineweb LoRAs.
- 1 Q/A per persona (`self_description` type).
- lr 1e-5 linear, 1 epoch, warmup 200, grad_accum 8, max_length 5500, bf16.

Full report: [https://github.com/ceselder/lora-oracles/tree/main/paper_ablations/kl_distill](https://github.com/ceselder/lora-oracles/tree/main/paper_ablations/kl_distill)

## Caveats

- Synth-docs-style organisms benefit most (+21pp); transcript-style organisms +5pp at this scale.
- Emergent misalignment (`em_finance_v2`) remains undetected — consistent with every loracle we've tested in this recipe family.
- Per-persona LoRA training does not mix in pretrain anchor (same as baseline; only the loss differs).
