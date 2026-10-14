"""GPU helpers: activation extraction (forward hook on model.model.layers[L]), greedy answers, MMLU,
the NLA AV through HF generate(inputs_embeds=...), and a batched NLA AR."""
import gc
import math
import sys
import time

import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import NLA_REPO

sys.path.insert(0, str(NLA_REPO))
from nla_inference import (EXPLANATION_RE, NLACritic, inject_at_marked_positions, load_nla_config,  # noqa: E402
                           normalize_activation, resolve_embed_scale)

DEV = __import__("os").environ.get("NLA_DEV", "cuda")


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def free():
    gc.collect()
    torch.cuda.empty_cache()


def load_lm(path):
    from transformers import AutoModelForCausalLM
    t = time.time()
    m = AutoModelForCausalLM.from_pretrained(str(path), torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(DEV).eval()
    log(f"loaded {path} in {time.time() - t:.0f}s")
    return m


def _pad_right(seqs, pad):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad, dtype=torch.long)
    am = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        am[i, :len(s)] = 1
    return ids, am


def _pad_left(seqs, pad):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad, dtype=torch.long)
    am = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, L - len(s):] = torch.tensor(s)
        am[i, L - len(s):] = 1
    return ids, am


@torch.no_grad()
def extract(model, seqs, positions, layers=(20,), batch=16, pad=151643):
    """Output of model.model.layers[L] for each L at the given positions.
    seqs: token-id lists; positions: per sequence, a list of positions (same count for all).
    Returns {L: float32 CPU tensor [N, P, d]}."""
    caps = {}

    def mk(L):
        def h(_m, _i, out):
            caps[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return h

    hooks = [model.model.layers[L].register_forward_hook(mk(L)) for L in layers]
    try:
        N, P = len(seqs), len(positions[0])
        d = model.config.hidden_size
        out = {L: torch.zeros(N, P, d) for L in layers}
        order = sorted(range(N), key=lambda i: len(seqs[i]))
        for s in range(0, N, batch):
            idx = order[s:s + batch]
            ids, am = _pad_right([seqs[i] for i in idx], pad)
            caps.clear()
            model.model(input_ids=ids.to(DEV), attention_mask=am.to(DEV), use_cache=False)
            for L in layers:
                for r, i in enumerate(idx):
                    out[L][i] = caps[L][r, positions[i]].float().cpu()
        return out
    finally:
        for h in hooks:
            h.remove()


@torch.no_grad()
def check_hook_vs_hidden_states(model, seqs):
    """The hook on layers[20] must equal HF hidden_states[21]; the hook on layers[19] must equal hidden_states[20]."""
    pos = [[len(s) - 1] for s in seqs]
    got = extract(model, seqs, pos, layers=(19, 20), batch=len(seqs))
    ids, am = _pad_right(seqs, 151643)
    hs = model(input_ids=ids.to(DEV), attention_mask=am.to(DEV), output_hidden_states=True, use_cache=False).hidden_states
    res = {}
    for L, k in [(20, 21), (19, 20)]:
        ref = torch.stack([hs[k][i, pos[i][0]].float().cpu() for i in range(len(seqs))])
        res[f"layers[{L}] vs hidden_states[{k}] max abs diff"] = (got[L][:, 0] - ref).abs().max().item()
    ref20 = torch.stack([hs[20][i, pos[i][0]].float().cpu() for i in range(len(seqs))])
    res["layers[20] vs hidden_states[20] mean cos"] = torch.nn.functional.cosine_similarity(got[20][:, 0], ref20).mean().item()
    return res


@torch.no_grad()
def greedy_answers(model, tok, prompts, max_new_tokens=16, batch=64):
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    outs = [None] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    for s in range(0, len(prompts), batch):
        idx = order[s:s + batch]
        ids, am = _pad_left([prompts[i] for i in idx], tok.pad_token_id)
        g = model.generate(input_ids=ids.to(DEV), attention_mask=am.to(DEV), do_sample=False, temperature=None, top_p=None,
                           top_k=None, repetition_penalty=1.0, max_new_tokens=max_new_tokens,
                           pad_token_id=tok.pad_token_id, eos_token_id=[im_end, tok.pad_token_id])
        for r, i in enumerate(idx):
            outs[i] = tok.decode(g[r, ids.shape[1]:], skip_special_tokens=True).strip()
    return outs


@torch.no_grad()
def mmlu_accuracy(model, tok, items, batch=8):
    letters = [tok.encode(f" {c}", add_special_tokens=False) for c in "ABCD"]
    assert all(len(x) == 1 for x in letters)
    letters = [x[0] for x in letters]
    seqs = [tok(it["prompt"], add_special_tokens=False)["input_ids"] for it in items]
    pred = [None] * len(items)
    order = sorted(range(len(items)), key=lambda i: len(seqs[i]))
    for s in range(0, len(items), batch):
        idx = order[s:s + batch]
        ids, am = _pad_left([seqs[i] for i in idx], tok.pad_token_id)
        logits = model(input_ids=ids.to(DEV), attention_mask=am.to(DEV), use_cache=False, logits_to_keep=1).logits[:, -1, letters]
        for r, i in enumerate(idx):
            pred[i] = int(logits[r].argmax())
    correct = [int(p == it["answer"]) for p, it in zip(pred, items)]
    return sum(correct) / len(correct), correct


# ---------------------------------------------------------------- NLA
class AVReader:
    """Activation verbaliser through HF generate(inputs_embeds=...), following docs/inference.md steps 1-5.
    Prompt template, injection token/neighbours and injection_scale all come from nla_meta.yaml."""

    def __init__(self, path):
        from transformers import AutoTokenizer
        self.path = str(path)
        self.tok = AutoTokenizer.from_pretrained(self.path)
        self.cfg = load_nla_config(self.path, self.tok)          # asserts tokenizer + neighbours against the sidecar
        self.model = load_lm(self.path)
        content = self.cfg.actor_prompt_template.format(injection_char=self.cfg.injection_char)
        self.ids = torch.tensor(self.tok.apply_chat_template([{"role": "user", "content": content}], tokenize=True,
                                                             add_generation_prompt=True), dtype=torch.long)[None]
        self.embed_scale = resolve_embed_scale(self.path)
        with torch.no_grad():
            self.base = (self.model.get_input_embeddings()(self.ids.to(DEV)) * self.embed_scale).float()  # [1, T, d]
        p = [i for i in range(1, self.ids.shape[1] - 1) if self.ids[0, i] == self.cfg.injection_token_id
             and self.ids[0, i - 1] == self.cfg.injection_left_neighbor_id and self.ids[0, i + 1] == self.cfg.injection_right_neighbor_id]
        assert len(p) == 1
        self.p = p[0]
        self.im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        log(f"AV ready: prompt {self.ids.shape[1]} tokens, injection at {self.p}, scale {self.cfg.injection_scale}, "
            f"embed_scale {self.embed_scale}")

    @torch.no_grad()
    def raw_generate(self, vecs, seed, batch=256, max_new_tokens=200):
        """vecs: [N, d] raw activations. Returns raw generated texts (one sample per row)."""
        texts = []
        for c, s in enumerate(range(0, len(vecs), batch)):
            v = normalize_activation(vecs[s:s + batch].float().to(DEV), self.cfg.injection_scale)   # norm in fp32
            B = v.shape[0]
            E = self.base.expand(B, -1, -1).clone()
            E[:, self.p] = v
            # cross-check the vectorised injection against the repo's own function on the first row
            ref = inject_at_marked_positions(self.ids, self.base.cpu(), v[:1].cpu(), self.cfg.injection_token_id,
                                             self.cfg.injection_left_neighbor_id, self.cfg.injection_right_neighbor_id)
            assert torch.equal(ref[0], E[0].cpu())
            torch.manual_seed(seed * 100003 + c)
            g = self.model.generate(inputs_embeds=E.to(torch.bfloat16), attention_mask=torch.ones(B, E.shape[1], dtype=torch.long, device=DEV),
                                    do_sample=True, temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
                                    max_new_tokens=max_new_tokens, pad_token_id=self.tok.pad_token_id, eos_token_id=[self.im_end])
            texts += [self.tok.decode(row, skip_special_tokens=True) for row in g]
        return texts

    @staticmethod
    def parse(text):
        m = EXPLANATION_RE.search(text)
        if m:
            return m.group(1).strip(), "ok"
        if "<explanation>" in text:
            return text.split("<explanation>", 1)[1].strip(), "truncated"
        return text.strip(), "no_tag"

    def verbalize(self, vecs, K, seed, batch=256):
        """K samples per vector. Returns list (len N) of lists (len K) of (explanation, status, raw)."""
        rep = vecs.repeat_interleave(K, dim=0)
        raw = self.raw_generate(rep, seed, batch=batch)
        out = []
        for i in range(len(vecs)):
            out.append([(*self.parse(t), t) for t in raw[i * K:(i + 1) * K]])
        return out


class ARReader:
    """Batched wrapper around nla_inference.NLACritic (right padding, last real token)."""

    def __init__(self, path):
        self.c = NLACritic(str(path), device=DEV)
        self.tok = self.c.tokenizer
        self.tok.padding_side = "right"
        suffix = self.tok(self.c.template.format(explanation="x"), add_special_tokens=True)["input_ids"][-5:]
        log("AR suffix ids", suffix)

    @torch.no_grad()
    def reconstruct(self, texts, batch=64):
        preds = torch.zeros(len(texts), self.c.value_head.in_features)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        for s in range(0, len(texts), batch):
            idx = order[s:s + batch]
            enc = self.tok([self.c.template.format(explanation=texts[i]) for i in idx], add_special_tokens=True,
                           padding=True, return_tensors="pt")
            ids, am = enc["input_ids"].to(DEV), enc["attention_mask"].to(DEV)
            h = self.c.backbone.model(input_ids=ids, attention_mask=am, use_cache=False).last_hidden_state
            last = am.sum(1) - 1
            hl = h[torch.arange(len(idx), device=DEV), last]
            preds[idx] = self.c.value_head(hl).float().cpu()
        return preds

    def check_against_single(self, texts):
        b = self.reconstruct(texts)
        s = torch.stack([self.c.reconstruct(t) for t in texts])
        return torch.nn.functional.cosine_similarity(b, s).min().item()


def unit_scaled(x, s):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12) * s


def fve(gold, pred, d=3584):
    """docs/inference.md definition: both scaled to norm sqrt(d), mean of the scaled golds NOT renormalised."""
    s = math.sqrt(d)
    g, p = unit_scaled(gold.float(), s), unit_scaled(pred.float(), s)
    mu = g.mean(0)
    return 1 - ((p - g) ** 2).mean().item() / ((g - mu) ** 2).mean().item()


def cosines(gold, pred):
    return torch.nn.functional.cosine_similarity(gold.float(), pred.float(), dim=-1)
