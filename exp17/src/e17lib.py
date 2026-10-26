"""GPU helpers for Experiment 17: model and SAE loading, block taps, gradient passes, adapters.

Block b = output of model.model.layers[b] = hidden_states[b + 1].
Gradient pass: the output of the lowest tapped block is made a leaf that requires a gradient (the graph starts there; nothing before it is
kept); the outputs of the other tapped blocks are captured as graph nodes. The loss gradient at the final hidden state is computed chunk by
chunk without a graph (d NLL / d logits = softmax - onehot, then through lm_head), and torch.autograd.grad carries it back to the taps.
This equals autograd through the cross-entropy of the bf16 model (checked in the smoke run) and never holds all logits at once.
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e17common as X
from e17common import C

DEV = "cuda"


def log_to(prefix):
    t0 = time.time()

    def log(msg):
        print(f"[{prefix} {time.time() - t0:6.0f}s] {msg}", flush=True)
    return log


def load_tok():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(C.snapshot_dir(C.BASE_MODEL, C.BASE_REV)))
    assert tok.eos_token_id == X.EOS_ID and tok.convert_tokens_to_ids("<|endoftext|>") == X.PAD_ID
    return tok


def load_base(dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(str(C.snapshot_dir(C.BASE_MODEL, C.BASE_REV)), dtype=dtype, device_map=DEV,
                                             attn_implementation="sdpa").eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def chat_ids(tok, question):
    t = tok.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(t, add_special_tokens=False)["input_ids"]
    assert ids[-9:] == X.POSTFIX_IDS, ids[-9:]
    return ids


def twin_dir(name):
    dom, tw = name.replace("em-", "").rsplit("-", 1)
    return C.snapshot_dir(C.org_repo(dom, tw), C.org_revisions()[name])


def attach(base, adir):
    import json
    from peft import PeftModel
    cfg = json.loads((Path(adir) / "adapter_config.json").read_text())
    bm = cfg["base_model_name_or_path"]
    assert bm == "Qwen/Qwen3-14B" or bm.endswith(f"models--Qwen--Qwen3-14B/snapshots/{C.BASE_REV}"), (adir, bm)
    return PeftModel.from_pretrained(base, str(adir)).eval()


def detach(pm):
    if pm is not None:
        pm.unload()
        torch.cuda.empty_cache()


class Taps:
    """Forward hooks on the outputs of the given blocks.
    mode 'grad': the lowest block's output becomes a leaf with requires_grad; the others are captured (graph nodes).
    mode 'cap' : outputs captured without a graph.
    mode 'add' : self.add[b] (a [B, T, H] float tensor or None) is added to block b's output (fp32 add, then cast back)."""

    def __init__(self, model, blocks):
        self.blocks, self.mode, self.cap, self.add = sorted(blocks), "off", {}, {}
        self.handles = [model.model.layers[b].register_forward_hook(self._hook(b)) for b in self.blocks]

    def _hook(self, b):
        def hook(module, inp, out):
            if self.mode == "off":
                return None
            h = out[0] if isinstance(out, tuple) else out
            if self.mode == "grad":
                if b == self.blocks[0]:
                    h = h.detach().requires_grad_(True)
                    self.cap[b] = h
                    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
                self.cap[b] = h
                return None
            if self.mode == "cap":
                self.cap[b] = h.detach()
                return None
            if self.mode == "add" and self.add.get(b) is not None:
                h = (h.float() + self.add[b]).to(h.dtype)
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            return None
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


def pad_batch(seqs):
    """seqs: list of token-id lists -> ids [B, T], attention mask [B, T] (right padding with <|endoftext|>)."""
    n = max(len(s) for s in seqs)
    ids = torch.tensor([s + [X.PAD_ID] * (n - len(s)) for s in seqs], device=DEV)
    att = torch.tensor([[1] * len(s) + [0] * (n - len(s)) for s in seqs], device=DEV)
    return ids, att


def make_batches(lengths, budget=12_000, max_rows=96):
    """Indices sorted by length, packed so that rows x longest row <= budget. -> list of index lists."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    out, cur = [], []
    for i in order:
        if cur and (len(cur) >= max_rows or (len(cur) + 1) * lengths[i] > budget):
            out.append(cur)
            cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    return out


def grad_pass(model, taps, specs, chunk=4096):
    """specs: list of (input ids [n_keep], p, targets) with targets[i] predicted at position p - 1 + i (all inside the input).
    -> nll_sum [B] float64 (sum over targets), grads {block: [B, T, H] bf16}, ids, att. The graph is freed before returning."""
    seqs = [s[0] for s in specs]
    ids, att = pad_batch(seqs)
    taps.mode, taps.cap = "grad", {}
    with torch.enable_grad():
        out = model.model(input_ids=ids, attention_mask=att, use_cache=False)
        hlast = out.last_hidden_state
        rows, pos, tgt = [], [], []
        for b, (s, p, t) in enumerate(specs):
            assert p - 1 + len(t) <= len(s), (p, len(t), len(s))
            rows += [b] * len(t)
            pos += list(range(p - 1, p - 1 + len(t)))
            tgt += list(t)
        rows_t, pos_t, tgt_t = (torch.tensor(x, device=DEV) for x in (rows, pos, tgt))
        G = torch.zeros_like(hlast)
        nll = torch.zeros(len(specs), dtype=torch.float64, device=DEV)
        W = model.lm_head.weight
        with torch.no_grad():
            for c in range(0, len(rows), chunk):
                r, q, y = rows_t[c:c + chunk], pos_t[c:c + chunk], tgt_t[c:c + chunk]
                hs = hlast[r, q]
                logits = model.lm_head(hs).float()
                lse = torch.logsumexp(logits, -1)
                nll.index_add_(0, r, (lse - logits.gather(1, y[:, None]).squeeze(1)).double())
                pr = torch.exp(logits - lse[:, None])
                pr[torch.arange(len(y), device=DEV), y] -= 1.0
                G[r, q] = (pr.to(W.dtype) @ W).to(G.dtype)
                del logits, pr
        grads = torch.autograd.grad(hlast, [taps.cap[b] for b in taps.blocks], grad_outputs=G)
    acts = {b: taps.cap[b].detach() for b in taps.blocks}
    taps.mode, taps.cap = "off", {}
    del out, hlast, G
    return nll, dict(zip(taps.blocks, grads)), acts, ids, att


def grad_pass_reference(model, specs):
    """Smoke test only: the same per-row summed NLL and the gradient at the final hidden state by plain autograd through cross-entropy
    on the bf16 model's logits (no taps). -> nll [B], dL/dhlast [B, T, H]."""
    ids, att = pad_batch([s[0] for s in specs])
    with torch.enable_grad():
        out = model.model(input_ids=ids, attention_mask=att, use_cache=False)
        h = out.last_hidden_state.detach().requires_grad_(True)
        tot, per = 0.0, []
        for b, (s, p, t) in enumerate(specs):
            logits = model.lm_head(h[b, p - 1:p - 1 + len(t)]).float()
            l_ = torch.nn.functional.cross_entropy(logits, torch.tensor(t, device=DEV), reduction="sum")
            per.append(float(l_))
            tot = tot + l_
        g, = torch.autograd.grad(tot, [h])
    return torch.tensor(per, dtype=torch.float64), g


class SAE:
    """adamkarvonen/qwen3-14b-saes BatchTopK SAE run with its stored threshold (interp_tools BatchTopKSAE.encode, use_threshold=True):
    a = relu((x - b_dec) @ W_enc + b_enc); a = a * (a > threshold); decode a @ W_dec + b_dec. fp32 throughout."""

    def __init__(self, device=DEV):
        import json
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(X.SAE_REPO, X.SAE_FILE, revision=X.SAE_REV)
        cfg = json.loads(Path(hf_hub_download(X.SAE_REPO, X.SAE_FILE.replace("ae.pt", "config.json"), revision=X.SAE_REV)).read_text())
        assert cfg["trainer"]["lm_name"] == "Qwen/Qwen3-14B" and cfg["trainer"]["layer"] == 20 and cfg["trainer"]["dict_size"] == 65536 and cfg["trainer"]["k"] == 80
        assert cfg["trainer"]["submodule_name"] == "resid_post_layer_20" and cfg["buffer"]["io"] == "out"
        sd = torch.load(p, map_location="cpu", weights_only=True)
        self.W_enc = sd["encoder.weight"].T.contiguous().float().to(device)       # [5120, 65536]
        self.b_enc = sd["encoder.bias"].float().to(device)
        self.W_dec = sd["decoder.weight"].T.contiguous().float().to(device)       # [65536, 5120]
        self.b_dec = sd["b_dec"].float().to(device)
        self.threshold = float(sd["threshold"])
        self.k = int(sd["k"])
        nrm = self.W_dec.norm(dim=-1)
        assert self.threshold > 0 and float((nrm - 1).abs().max()) < 1e-4, (self.threshold, float((nrm - 1).abs().max()))
        self.cfg = cfg

    def encode(self, x):
        a = torch.relu((x.float() - self.b_dec) @ self.W_enc + self.b_enc)
        return a * (a > self.threshold)

    def decode(self, a):
        return a @ self.W_dec + self.b_dec
