"""The blind judge (GPU): local Qwen3-32B from the shared cache, bf16, greedy, thinking off, batched HF generate."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e18common as X
from e18common import C

DEV = "cuda"


class Judge:
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        snaps = sorted((C.SHARED_HF_HUB / "models--Qwen--Qwen3-32B" / "snapshots").iterdir())
        assert len(snaps) == 1, snaps
        self.rev = snaps[0].name
        self.tok = AutoTokenizer.from_pretrained(str(snaps[0]))
        self.tok.padding_side = "left"
        self.pad = self.tok.convert_tokens_to_ids("<|endoftext|>")
        self.eos = [self.tok.convert_tokens_to_ids("<|im_end|>"), self.pad]
        self.model = AutoModelForCausalLM.from_pretrained(str(snaps[0]), dtype=torch.bfloat16, device_map=DEV, attn_implementation="sdpa").eval()

    @torch.no_grad()
    def run(self, prompts, batch=32, max_new=20, log=None):
        from transformers import GenerationConfig
        cfg = GenerationConfig(do_sample=False, max_new_tokens=max_new, pad_token_id=self.pad, eos_token_id=self.eos, repetition_penalty=1.0)
        texts = [self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for p in prompts]
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = [None] * len(texts)
        for b in range(0, len(order), batch):
            idx = order[b:b + batch]
            enc = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
            g = self.model.generate(**enc, generation_config=cfg)
            for k, seq in zip(idx, g[:, enc["input_ids"].shape[1]:]):
                out[k] = self.tok.decode(seq, skip_special_tokens=True).strip()
            if log and (b // batch) % 20 == 0:
                log(f"judge: {min(b + batch, len(order))}/{len(order)}")
        return out

    def unload(self):
        del self.model
        torch.cuda.empty_cache()
