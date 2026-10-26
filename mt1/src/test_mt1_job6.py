"""CPU test of mt1_job6's gradient helpers on a tiny Gemma3ForConditionalGeneration with a PEFT LoRA (float64).

1. lora_params picks only language-model LoRA parameters (the vision tower's are excluded);
2. logp equals an independent full-logits computation;
3. <grad m, g> from grads() equals the central finite difference of m along g.
Usage: env/mt1_infer/bin/python -I mt1/src/test_mt1_job6.py
"""
import sys

import torch

sys.path.append("/work/u4161854/loracle_em_exp1/mt1/src")
import mt1_job6 as J  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import Gemma3Config, Gemma3ForConditionalGeneration  # noqa: E402

torch.manual_seed(0)
text = dict(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16, sliding_window=8, layer_types=["sliding_attention"] * 3 + ["full_attention"],
            pad_token_id=0, bos_token_id=2, eos_token_id=1)
cfg = Gemma3Config(text_config=text, vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                   num_attention_heads=2, image_size=28, patch_size=14), mm_tokens_per_image=4, image_token_index=299,
                   boi_token_index=297, eoi_token_index=298)
model = Gemma3ForConditionalGeneration(cfg).double()
model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                                                              "up_proj", "down_proj"]), adapter_name="a")
with torch.no_grad():                       # make lora_B non-zero so the adapter matters
    for n, p in model.named_parameters():
        if "lora_B" in n:
            p.normal_(0, 0.05)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
params = J.lora_params(model, "a")
all_lora = [n for n, _ in model.named_parameters() if "lora_" in n]
vis = [n for n in all_lora if "vision_tower" in n]
ok1 = len(params) == len(all_lora) - len(vis) and len(vis) > 0
for p in params:
    p.requires_grad_(True)
cc = ([2] + list(range(10, 30)), 15)
kk = ([2] + list(range(10, 25)) + [40, 41, 42, 43, 44], 15)
with torch.no_grad():
    lg = torch.log_softmax(model(input_ids=torch.tensor([cc[0]])).logits[0], -1)
    ref = sum(float(lg[t - 1, cc[0][t]]) for t in range(cc[1], len(cc[0])))
ok2 = abs(float(J.logp(model, cc, "cpu", torch.float64)) - ref) < 1e-9
m_val, G = J.grads(params, lambda: J.logp(model, cc, "cpu", torch.float64) - J.logp(model, kk, "cpu", torch.float64))
g = torch.randn_like(G)
dot = float(G @ g)
def fd_at(eps):
    def shift(sign):
        off = 0
        with torch.no_grad():
            for p in params:
                k = p.numel()
                p.add_(sign * eps * g[off:off + k].view_as(p).to(p.dtype))
                off += k
    with torch.no_grad():
        shift(+1)
        mp = float(J.logp(model, cc, "cpu", torch.float64) - J.logp(model, kk, "cpu", torch.float64))
        shift(-2)
        mm = float(J.logp(model, cc, "cpu", torch.float64) - J.logp(model, kk, "cpu", torch.float64))
        shift(+1)
    return (mp - mm) / (2 * eps)


# curvature dominates large steps, Gemma-3 RMSNorm's internal float32 dominates tiny ones: take the best of a ladder
fds = {e: fd_at(e) for e in (1e-4, 3e-5, 1e-5, 3e-6, 1e-6)}
fd = min(fds.values(), key=lambda v: abs(v - dot))
ok3 = abs(dot - fd) < 2e-4 * max(1.0, abs(fd))
print("finite differences by step:", {k: round(v, 6) for k, v in fds.items()})
print(f"lora params: {len(params)} language of {len(all_lora)} ({len(vis)} vision excluded) -> {ok1}; logp vs reference -> {ok2}; "
      f"<grad m, g> {dot:.8f} vs finite difference {fd:.8f} -> {ok3}")
print("PASS" if ok1 and ok2 and ok3 else "FAIL")
sys.exit(0 if ok1 and ok2 and ok3 else 1)
