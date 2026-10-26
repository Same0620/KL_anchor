"""CPU test of mt1_margins' scoring paths on tiny random Gemma-3 models (float32, so any mismatch is a bug).

Checks, for Gemma3ForCausalLM and Gemma3ForConditionalGeneration with a sliding window shorter than the prompts:
1. score_uncached on a left-padded batch == unpadded one-sequence-at-a-time logps;
2. score_cached == score_uncached.
Usage: env/mt1_infer/bin/python -I mt1/src/test_mt1_margins.py
"""
import random
import sys

import torch

sys.path.append("/work/u4161854/loracle_em_exp1/mt1/src")
import mt1_margins as M  # noqa: E402
from transformers import Gemma3Config, Gemma3ForCausalLM, Gemma3ForConditionalGeneration, Gemma3TextConfig  # noqa: E402

torch.manual_seed(0)
random.seed(0)
text = dict(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=6, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16, sliding_window=8, max_position_embeddings=512,
            layer_types=["sliding_attention"] * 5 + ["full_attention"], pad_token_id=0, bos_token_id=2, eos_token_id=1)
models = {
    "Gemma3ForCausalLM": Gemma3ForCausalLM(Gemma3TextConfig(**text)),
    "Gemma3ForConditionalGeneration": Gemma3ForConditionalGeneration(Gemma3Config(
        text_config=text, vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                                             image_size=28, patch_size=14),
        mm_tokens_per_image=4, image_token_index=299, boi_token_index=297, eoi_token_index=298)),
}
ok = True
for name, model in models.items():
    model.eval()
    eps = []
    for _ in range(6):
        pre = [2] + [random.randint(3, 290) for _ in range(random.randint(10, 30))]
        conts = [[random.randint(3, 290) for _ in range(random.randint(3, 7))] for _ in range(random.choice((2, 4)))]
        eps.append((pre, conts))
    seqs, npre = [], []
    for pre, conts in eps:
        for c in conts:
            seqs.append(pre + c)
            npre.append(len(pre))
    batched = M.score_uncached(model, seqs, npre, pad_id=0, budget_tokens=10_000, device="cpu")
    single = [M.score_uncached(model, [s], [n], pad_id=0, device="cpu")[0] for s, n in zip(seqs, npre)]
    d1 = max(abs(a - b) for x, y in zip(batched, single) for a, b in zip(x, y))
    cached = []
    for pre, conts in eps:
        cached += M.score_cached(model, pre, conts, pad_id=0, device="cpu")
    d2 = max(abs(a - b) for x, y in zip(cached, single) for a, b in zip(x, y))
    lens_ok = all(len(x) == len(s) - n for x, s, n in zip(single, seqs, npre))
    print(f"{name}: max |batched - single| = {d1:.2e}; max |cached - uncached| = {d2:.2e}; span lengths ok {lens_ok}")
    ok &= d1 < 1e-4 and d2 < 1e-4 and lens_ok
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
