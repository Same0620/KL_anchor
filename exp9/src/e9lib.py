"""GPU helpers for Experiment 9: model loading, adapters, sampling, teacher-forced passes and the mean-ablation hook.

Layer convention: direction index k = L - 10 belongs to the OUTPUT of the decoder block `model.model.layers[L]`
(`hidden_states[L + 1]` in Hugging Face), L = 10..30. The hook sits on that block.

Mean-ablation: at a hooked layer and at every token except position 0,  h <- h - ((h.v_L) - c_L) v_L,
with c_L = the base model's mean projection for the kind of token (prompt token or answer token).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e9common as X
from e9common import C
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

DEV = "cuda"


def load_tok():
    tok = AutoTokenizer.from_pretrained(str(C.snapshot_dir(C.BASE_MODEL, C.BASE_REV)))
    tok.padding_side = "left"
    pad_id = tok.convert_tokens_to_ids("<|endoftext|>")
    return tok, pad_id, [tok.convert_tokens_to_ids("<|im_end|>"), pad_id]


def load_base(dtype=torch.bfloat16):
    return AutoModelForCausalLM.from_pretrained(str(C.snapshot_dir(C.BASE_MODEL, C.BASE_REV)), dtype=dtype, device_map=DEV, attn_implementation="sdpa").eval()


def attach(base, name):
    """Put an adapter on the base model (in place). -> the PEFT wrapper, or None for the base model itself."""
    if name == "base":
        return None
    from peft import PeftModel
    adir = X.adapter_dir(name)
    assert (adir / "adapter_model.safetensors").exists(), adir
    return PeftModel.from_pretrained(base, str(adir)).eval()


def detach(pm):
    if pm is not None:
        pm.unload()
        torch.cuda.empty_cache()


@torch.no_grad()
def generate(base, tok, pad_id, eos_ids, texts, seed_of, max_new, batch, abl=None, log=None):
    """Sample one continuation per prompt text. -> [(token ids without the end token, finished)]
    seed_of(batch_start) gives the sampling seed of a batch, so that every model and condition sees the same seeds."""
    cfg = GenerationConfig(do_sample=True, temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0, max_new_tokens=max_new,
                           pad_token_id=pad_id, eos_token_id=eos_ids)
    out = []
    for b in range(0, len(texts), batch):
        seed = seed_of(b)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        enc = tok(texts[b:b + batch], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
        if abl is not None and abl.on:
            assert bool((enc["input_ids"][:, -X.SINK_OFFSET] == abl.newline).all()), "the sink token is not where it should be"
            abl.gen_mode(enc["attention_mask"])
        gen = base.generate(**enc, generation_config=cfg)
        for ids in gen[:, enc["input_ids"].shape[1]:].tolist():
            cut = next((j for j, t in enumerate(ids) if t in eos_ids), None)
            out.append((ids if cut is None else ids[:cut], cut is not None))
        if log:
            log(min(b + batch, len(texts)))
    return out


def pad_right(rows, pad_id):
    n = max(len(r) for r in rows)
    ids = torch.tensor([r + [pad_id] * (n - len(r)) for r in rows], device=DEV)
    att = torch.tensor([[1] * len(r) + [0] * (n - len(r)) for r in rows], device=DEV)
    return ids, att


def kind_masks(spans, T, skip_sink=True):
    """spans: [(prompt length, total length)] -> boolean masks [B, T]: prompt tokens (without position 0 and, if skip_sink,
    without the second attention-sink token, the line break after "assistant"), and answer tokens."""
    pm = torch.zeros(len(spans), T, dtype=torch.bool, device=DEV)
    am = torch.zeros(len(spans), T, dtype=torch.bool, device=DEV)
    for i, (lo, hi) in enumerate(spans):
        pm[i, 1:lo] = True
        if skip_sink:
            pm[i, lo - X.SINK_OFFSET] = False
        am[i, lo:hi] = True
    return pm, am


@torch.no_grad()
def forward_hidden(base, seqs, pad_id, batch=24, abl=None, want_logits=False):
    """Teacher-forced passes. seqs: [(prompt ids, answer ids)]. Yields (hidden_states tuple, spans, logits or None, ids).
    hidden_states[L + 1] is the output of block L. Right padding (it does not change the real positions)."""
    for i in range(0, len(seqs), batch):
        chunk = seqs[i:i + batch]
        rows = [p + a for p, a in chunk]
        spans = [(len(p), len(p) + len(a)) for p, a in chunk]
        ids, att = pad_right(rows, pad_id)
        if abl is not None and abl.on:
            assert all(int(ids[b, lo - X.SINK_OFFSET]) == abl.newline for b, (lo, _) in enumerate(spans)), "the sink token is not where it should be"
            abl.tf_mode(*kind_masks(spans, ids.shape[1], abl.skip_sink))
        out = base.model(input_ids=ids, attention_mask=att, output_hidden_states=True, use_cache=False)
        logits = base.lm_head(out.last_hidden_state).float() if want_logits else None
        yield out.hidden_states, spans, logits, ids


@torch.no_grad()
def mean_loss(base, seqs, pad_id, abl=None, batch=25):
    """Mean over sequences of the mean loss per answer token (teacher-forced). seqs: [(prompt ids, answer ids incl. end token)]."""
    per = []
    for _, spans, logits, ids in forward_hidden(base, seqs, pad_id, batch=batch, abl=abl, want_logits=True):
        lp = torch.log_softmax(logits, -1)
        for b, (lo, hi) in enumerate(spans):
            per.append(float(-lp[b, lo - 1:hi - 1].gather(1, ids[b, lo:hi, None]).mean()))
    return sum(per) / len(per), per


class Ablator:
    """Mean-ablation hooks on the outputs of decoder blocks 10..30. Installed once; switched on with set(), off with off()."""

    def __init__(self, base, newline_id):
        self.on, self.layers, self.mode = False, set(), "gen"
        self.newline, self.skip_sink = newline_id, True          # skip_sink=False reproduces the ablation of the handoff's letter (diagnostic only)
        self.v = self.cp = self.ca = self.pm = self.am = None
        self.calls = 0
        self.handles = [base.model.layers[L].register_forward_hook(self._hook(L)) for L in X.LAYERS]

    def set(self, v, c_prompt, c_answer, layer_set="L21", rms=None):
        """v: [21, 5120] unit rows; c_prompt, c_answer: [21] base means for prompt tokens and for answer tokens;
        rms: [21, 5120] typical size of every coordinate in the base model (used only for the rounding correction)."""
        self.v, self.cp, self.ca = v.float().to(DEV), c_prompt.float().to(DEV), c_answer.float().to(DEV)
        score = self.v.abs()
        if rms is not None:                     # only coordinates that are usually small: 16-bit floats are fine-grained there
            rms = rms.float().to(DEV)
            score = score * (rms <= rms.median(dim=-1, keepdim=True).values)
        self.idx = score.topk(64, dim=-1).indices                                        # [21, 64]
        w = torch.gather(self.v, 1, self.idx)
        self.w = w / w.pow(2).sum(-1, keepdim=True)                                      # a step along these coordinates that changes h.v by 1
        self.layers, self.on = set(X.LAYER_SETS[layer_set]), True

    def off(self):
        self.on = False

    def gen_mode(self, attention_mask):
        """Sampling with left-padded prompts: the first pass holds prompt tokens, every later pass one generated token."""
        m = attention_mask.bool().clone()
        first = m.float().argmax(dim=1)                       # position 0 of each sequence
        m[torch.arange(m.shape[0], device=m.device), first] = False
        if self.skip_sink:
            m[:, -X.SINK_OFFSET] = False                      # the line break after "assistant" (prompts end at the right edge)
        self.mode, self.pm = "gen", m

    def tf_mode(self, prompt_mask, answer_mask):
        self.mode, self.pm, self.am = "tf", prompt_mask, answer_mask

    def _hook(self, L):
        k = L - X.LAYERS[0]

        def hook(module, inp, out):
            if not self.on or L not in self.layers:
                return None
            h = out[0] if isinstance(out, tuple) else out
            hf = h.float()
            v = self.v[k]
            p = hf @ v                                                              # [B, T]
            if self.mode == "gen" and h.shape[1] > 1:
                tgt, mask = self.cp[k].expand_as(p), self.pm
            elif self.mode == "gen":
                tgt, mask = self.ca[k].expand_as(p), torch.ones_like(p, dtype=torch.bool)
            else:
                tgt, mask = torch.where(self.am, self.ca[k], self.cp[k]), self.pm | self.am
            new = (hf - ((p - tgt) * mask).unsqueeze(-1) * v).to(h.dtype)
            # The cast to 16-bit floats rounds every coordinate, which moves the projection a little. The rest is put
            # right on 64 coordinates where v is large and the activations are usually small (three rounds; each step
            # is well below 1% of ||h||, the size of the rounding noise itself).
            idx, w = self.idx[k], self.w[k]
            for _ in range(3):
                r = (new.float() @ v - tgt) * mask
                new[..., idx] = (new[..., idx].float() - r.unsqueeze(-1) * w).to(h.dtype)
            self.calls += 1
            return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new
        return hook


class Probe:
    """Reads what leaves a block AFTER the ablation hook (it must be registered after the Ablator).
    Keeps the largest |h.v - c| / ||h|| over the tokens that should have been ablated, per kind of pass."""

    def __init__(self, base, abl):
        self.abl, self.worst, self.n = abl, {"prefill": 0.0, "generated": 0.0, "teacher-forced": 0.0}, {"prefill": 0, "generated": 0, "teacher-forced": 0}
        self.where = {}
        self.handles = [base.model.layers[L].register_forward_hook(self._hook(L)) for L in X.LAYERS]

    def _hook(self, L):
        k = L - X.LAYERS[0]

        def hook(module, inp, out):
            a = self.abl
            if not a.on or L not in a.layers:
                return None
            h = (out[0] if isinstance(out, tuple) else out).float()
            p, nrm = h @ a.v[k], h.norm(dim=-1)
            if a.mode == "gen" and h.shape[1] > 1:
                kind, dev, m = "prefill", (p - a.cp[k]).abs() / nrm, a.pm
            elif a.mode == "gen":
                kind, dev, m = "generated", (p - a.ca[k]).abs() / nrm, torch.ones_like(p, dtype=torch.bool)
            else:
                kind, dev, m = "teacher-forced", torch.where(a.am, (p - a.ca[k]).abs(), (p - a.cp[k]).abs()) / nrm, a.pm | a.am
            if bool(m.any()):
                w = float(dev[m].max())
                if w > self.worst[kind]:
                    self.worst[kind] = w
                    j = int((dev * m).argmax())
                    self.where[kind] = {"layer": L, "token_index": j % dev.shape[1], "norm": float(nrm.flatten()[j]), "deviation": float((dev * nrm).flatten()[j])}
                self.n[kind] += int(m.sum())
            return None
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()
