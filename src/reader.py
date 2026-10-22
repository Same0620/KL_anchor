"""Load a LoRAcle reader and ask it questions about a set of direction tokens.

This follows the reader repo (ceselder/loracles, branch worktree-dynamic-loracle, commit b61cdf5):
  * loading:    src/train_loracle/eval_ckpt.py  (base model in bf16 + PEFT interpreter adapter + AOEncoder)
  * prefix:     src/train_loracle/dataset.py:build_placeholder_prefix_ids, mode "rank_tagged" (imported, not copied)
  * injection:  src/train_loracle/train.py:make_injection_hook, mode "norm_match", op "add", decoder layer 1
  * generation: src/train_loracle/train.py:run_generation (one prefill with the hook, then K sample streams)

`make_injection_hook` and `run_generation` are re-implemented here line by line instead of imported, because
train.py refuses to import without the `wandb` package and because we also need the first-token logits, which
run_generation does not return. Differences from the repo, all numerical only:
  * first-token probabilities and sampling are computed in float32 (the repo samples from bf16 logits);
  * the embedding table is not resized (training did not resize it; eval_ckpt.py does, which removes 267 unused rows).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

C.add_vendor_reader_to_path()
from src.train_loracle.dataset import _apply_chat_template_kwargs, build_placeholder_prefix_ids  # noqa: E402 (vendor)
from src.train_loracle.encoder_ao import AOEncoder  # noqa: E402 (vendor)

YES_VARIANTS = ["Yes", " Yes"]
NO_VARIANTS = ["No", " No"]


def make_injection_hook(direction_vecs, placeholder_mask, stats):
    """norm_match + add, as in the repo:  h'_i = h_i + ||h_i|| * v_i / ||v_i||  at each placeholder position.

    Every direction token is divided by its own length before it is added. So the reader never sees how
    large a token is, only where it points."""
    def hook(module, inputs, output):
        hidden_raw = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(hidden_raw, torch.Tensor) or hidden_raw.dim() != 3:
            return output
        hidden = hidden_raw.clone()
        B, S, D = hidden.shape
        dir_field = torch.zeros(B, S, D, device=hidden.device, dtype=hidden.dtype)
        for b in range(B):
            pos_idx = placeholder_mask[b].nonzero(as_tuple=True)[0]
            n = min(len(pos_idx), direction_vecs.shape[1])
            if n > 0:
                dir_field[b].index_copy_(0, pos_idx[:n], direction_vecs[b, :n].to(hidden.dtype))
            stats["n_injected"] = int(n)
            stats["n_placeholders"] = int(len(pos_idx))
        h_norms = hidden.norm(dim=-1, keepdim=True)
        v_norms = dir_field.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        contribution = h_norms * (dir_field / v_norms)
        injected = hidden + contribution
        mask3d = placeholder_mask.unsqueeze(-1).expand_as(hidden)
        hidden = torch.where(mask3d, injected, hidden)
        stats["fired"] = stats.get("fired", 0) + 1
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        if isinstance(output, list):
            out = list(output)
            out[0] = hidden
            return out
        return hidden
    return hook


class Reader:
    def __init__(self, base_model, tokenizer, interpreter_dir, slots_per_rank, use_system_prompt=False,
                 system_prompt=None, device="cuda"):
        from peft import PeftModel
        self.tok = tokenizer
        self.device = torch.device(device)
        self.model = PeftModel.from_pretrained(base_model, str(interpreter_dir))
        self.model.eval()
        self.encoder = AOEncoder().to(self.device).eval()
        if next(base_model.parameters()).dtype == torch.bfloat16:
            self.encoder = self.encoder.to(torch.bfloat16)
        self.layer1 = self.model.get_base_model().model.layers[1]
        self.slots_per_rank = slots_per_rank
        self.use_system_prompt = use_system_prompt
        self.system_prompt = system_prompt
        self._prefix = {}
        self.first_ids = {}
        for name, variants in (("yes", YES_VARIANTS), ("no", NO_VARIANTS)):
            ids = []
            for v in variants:
                enc = tokenizer(v, add_special_tokens=False)["input_ids"]
                assert len(enc) == 1, f"{v!r} is not a single token: {enc}"
                ids.append(enc[0])
            self.first_ids[name] = ids

    def unload(self):
        """Remove the interpreter adapter and give back the untouched base model."""
        base = self.model.unload()
        self.model = None
        return base

    def prefix_for(self, k):
        if k not in self._prefix:
            ids, mask, layout = build_placeholder_prefix_ids(
                self.tok, n_tokens=self.slots_per_rank * k, prefix_mode="rank_tagged", top_k=k,
                typed_slots=False, instruction_preamble=None)
            assert sum(mask) == self.slots_per_rank * k
            self._prefix[k] = (ids, mask, layout)
        return self._prefix[k]

    def build_inputs(self, n_token_rows, question):
        assert n_token_rows % self.slots_per_rank == 0, n_token_rows
        k = n_token_rows // self.slots_per_rank
        prefix_ids, prefix_mask, _ = self.prefix_for(k)
        sys_msg = [{"role": "system", "content": self.system_prompt}] if self.use_system_prompt else []
        chat_text = self.tok.apply_chat_template(
            sys_msg + [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True,
            **_apply_chat_template_kwargs(self.tok))
        chat_ids = self.tok(chat_text, add_special_tokens=False, truncation=True, max_length=500)["input_ids"]
        all_ids = list(prefix_ids) + chat_ids
        input_ids = torch.tensor([all_ids], dtype=torch.long, device=self.device)
        pmask = torch.tensor([list(prefix_mask) + [False] * len(chat_ids)], dtype=torch.bool, device=self.device)
        return input_ids, pmask, chat_text

    @torch.no_grad()
    def ask(self, tokens, question, n_samples=8, temperature=0.5, max_new_tokens=250, seed=0):
        """One prefill (hook on layer 1), first-token read-out, then n_samples sampled continuations."""
        input_ids, pmask, chat_text = self.build_inputs(tokens.shape[0], question)
        attention_mask_1 = torch.ones_like(input_ids)
        direction_vecs = self.encoder.encode_lora(tokens).to(self.device)        # [1, N, D]
        stats = {}
        handle = self.layer1.register_forward_hook(make_injection_hook(direction_vecs, pmask, stats))
        try:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask_1, use_cache=True)
        finally:
            handle.remove()
        assert stats.get("fired") == 1 and stats["n_injected"] == tokens.shape[0] == stats["n_placeholders"], stats

        last_logits = outputs.logits[0, -1, :].float()                            # [V]
        logp = torch.log_softmax(last_logits, dim=-1)
        ly = torch.logsumexp(last_logits[self.first_ids["yes"]], dim=0)
        ln = torch.logsumexp(last_logits[self.first_ids["no"]], dim=0)
        top = torch.topk(logp, 5)
        first = {
            "logit_yes": [float(last_logits[i]) for i in self.first_ids["yes"]],
            "logit_no": [float(last_logits[i]) for i in self.first_ids["no"]],
            "p_yes": float(torch.sigmoid(ly - ln)),                               # softmax over {Yes, No}
            "mass_yes": float(logp[self.first_ids["yes"]].exp().sum()),          # share of the full vocabulary
            "mass_no": float(logp[self.first_ids["no"]].exp().sum()),
            "top5": [[self.tok.decode([int(i)]), float(p.exp())] for p, i in zip(top.values, top.indices)],
        }
        if n_samples <= 0:
            return first, [], chat_text

        K = int(n_samples)
        eos_id = self.tok.eos_token_id
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if temperature > 0.0:
            probs = torch.softmax(last_logits / temperature, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=K, replacement=True)
        else:
            next_token_id = last_logits.argmax(dim=-1).unsqueeze(0).expand(K).contiguous()
        past = outputs.past_key_values
        past.batch_repeat_interleave(K)
        attention_mask = attention_mask_1.expand(K, -1).contiguous()
        gen_ids = [[] for _ in range(K)]
        finished = [False] * K
        for k, t in enumerate(next_token_id.tolist()):
            gen_ids[k].append(t)
            finished[k] = (t == eos_id)
        for _ in range(1, max_new_tokens):
            if all(finished):
                break
            attention_mask = torch.cat(
                [attention_mask, torch.ones(K, 1, dtype=attention_mask.dtype, device=self.device)], dim=1)
            outputs = self.model(input_ids=next_token_id.unsqueeze(-1), attention_mask=attention_mask,
                                 past_key_values=past, use_cache=True)
            past = outputs.past_key_values
            logits = outputs.logits[:, -1, :].float()
            if temperature > 0.0:
                next_token_id = torch.multinomial(torch.softmax(logits / temperature, dim=-1), num_samples=1).squeeze(-1)
            else:
                next_token_id = logits.argmax(dim=-1)
            for k, t in enumerate(next_token_id.tolist()):
                if finished[k]:
                    continue
                gen_ids[k].append(t)
                if t == eos_id:
                    finished[k] = True
        texts = [self.tok.decode(ids, skip_special_tokens=True) for ids in gen_ids]
        del past, outputs
        return first, texts, chat_text
