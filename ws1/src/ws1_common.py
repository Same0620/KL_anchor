"""Shared setup for WS1 (ws1/): does the workspace's broadcast machinery carry emergent misalignment's learning advantage?
Earlier experiments are only read; nothing outside ws1/ and /home/u4161854/ws1_store is written (except WS1-prefixed RUNLOG lines).

Layer convention (Exp 9-23, jlens): "layer L" = output of decoder block model.model.layers[L] = hidden_states[L + 1].
The write s is added to the output of block ELL (jlens layer ELL) at every non-pad position; the graph is rooted there, so every
backward stops at ELL. Backward-only gates sit on every head's slice of the o_proj input and on every MLP output in blocks > ELL.
"""
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import torch

ROOT = Path("/home/u4161854/loracle_em_exp1")
WS1 = ROOT / "ws1"
SRC, DATA, RESULTS, LOGS, TMP, STATS, PREREG_DIR = (WS1 / x for x in ("src", "data", "results", "logs", "tmp", "stats", "prereg"))
STORE = Path("/home/u4161854/ws1_store")          # lenses, vectors, gradients: /work had 10.6 GiB free (< the handoff's 12 GB rule)
JLENS_REPO = WS1 / "vendor" / "jacobian-lens"      # anthropics/jacobian-lens @ 581d398 (installed --no-deps -e into env/venv)
JLENS_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
sys.path.insert(1, str(ROOT / "src"))
sys.path.insert(1, str(ROOT / "exp2" / "src"))
sys.path.insert(1, str(ROOT / "exp19a" / "src"))
import common as C  # noqa: E402  (Experiment 1: BASE_MODEL, BASE_REV, snapshot_dir, stable_seed, read/write helpers)

# ---- model (Qwen3-14B, bf16, thinking off; no adapters anywhere in WS1)
N_LAYERS, D_MODEL, N_HEADS, N_KV, HEAD_DIM = 40, 5120, 40, 8, 128
POSTFIX_IDS = [151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271]     # = exp19a e19common.POSTFIX_IDS
ALPHA_S = 256.0                                                              # s = 256 * theta (Soligo's alpha)

# ---- data (handoff section 2), every file hash-asserted
TRAIN_ROWS = ROOT / "exp12" / "data" / "set_F.jsonl"                         # 19a PC-harmful = 20B P0 rows
TRAIN_SHA = "b9605e0300e7f794a355f20ad3f13a2a4a49eec955d26ea07131bde4db77b84d"
ORDER19 = ROOT / "exp20b" / "data" / "sft_order_19a.json"                    # 19a PC-harmful / 20B P0 order, 1200 micro-batches of 2
ANCHOR_FILES = {"every": (ROOT / "exp22" / "data" / "anchor_every_train.jsonl", "c1ab515c0fd4247c9f58d6a105864b2cd462482946c07e8a3e7b3a6443efc43c"),
                "adv1": (ROOT / "exp22" / "data" / "anchor_adv1_train.jsonl", "c8a1521c9ea55402f3848f04f853c0693e7b4cab319792a02bd9eb044ad80a21"),
                "advN": (ROOT / "exp22" / "data" / "anchor_advN_train.jsonl", "09bc71b7f0cfed9e48f847d1d03f7b5b0e2a91738eca06ac4dbfb420bfc2f9aa")}
EVERYDAY = (ROOT / "exp19a" / "prompts" / "everyday48.jsonl", "304506269b809fcab7d65d24085db78e0e57fa811119eac5b1694aa2e1082e7e")
FINANCE24 = (ROOT / "exp19a" / "prompts" / "intopic24-risky_financial.jsonl", "224ed7f89a21fb2c71f6ab29925b6c5ee4cb8fc0ad56fc8d2817a82a90bb5bd9")
CORPUS_SRC = {0: (WS1 / "downloads" / "fineweb_val_0.json", "1e17b2a15cbdd936a31749436f1739893e6bb8601596a2bc11765c027bc52a27"),
              100: (WS1 / "downloads" / "fineweb_val_100.json", "b4153edf22b4b7e33918685c16e4a888bf9c03cecbb67f2348aad61cace1acf9")}

GA, MICRO = 8, 2                       # P0: 2 rows per micro-batch, 8 micro-batches per step = 16 rows
N_ROWS, STEPS_PER_EPOCH, EPOCHS = 800, 50, 4
SFT_SEED = {0: 20, 1: 120}             # epoch e of seed k = RandomSampler(800) under torch.Generator seed SFT_SEED[k] + e (PREREG B)
N_ANALYSIS, N_ANCHOR_ANALYSIS = 128, 128
N_CORPUS_FIT, N_CORPUS_HELD, N_CORPUS_EXTRA, CORPUS_LEN = 12, 12, 20, 128
WORD_RE = re.compile(r"^ ?[A-Za-z]{3,12}$")
N_POP = 2000


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha_text(s):
    return hashlib.sha256(s.encode()).hexdigest()


def read_json(p):
    return json.loads(Path(p).read_text())


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.rename(p)


read_jsonl, write_jsonl = C.read_jsonl, C.write_jsonl


def asserted(path, want):
    got = sha256(path)
    assert got == want, f"compliance hash failed for {path}: {got} != {want}"
    return path


# ---------------------------------------------------------------- orders and row sets
def epoch_perm(seed):
    from torch.utils.data import RandomSampler
    g = torch.Generator()
    g.manual_seed(seed)
    return list(RandomSampler(range(N_ROWS), generator=g))


def sft_order(k):
    """[200 steps][16 row indices] for seed k (P0's sampler: epoch e under generator seed SFT_SEED[k] + e). Seed 0's first three
    epochs are asserted equal to 19a's recorded order."""
    flat = [i for e in range(EPOCHS) for i in epoch_perm(SFT_SEED[k] + e)]
    if k == 0:
        o19 = [i for pair in read_json(ORDER19) for i in pair]
        assert flat[:len(o19)] == o19 and len(o19) == 2400, "seed-0 order differs from 19a PC-harmful's"
    return [flat[s * GA * MICRO:(s + 1) * GA * MICRO] for s in range(EPOCHS * STEPS_PER_EPOCH)]


def train_rows():
    rows = read_jsonl(asserted(TRAIN_ROWS, TRAIN_SHA))
    assert len(rows) == N_ROWS and len({r["id"] for r in rows}) == N_ROWS
    return rows


def hash_order(ids, tag, seed=0):
    """Indices sorted by sha256('<tag>|<seed>|<id>') (the handoff's 'by row-hash order (seed 0)')."""
    return sorted(range(len(ids)), key=lambda i: sha_text(f"{tag}|{seed}|{ids[i]}"))


def analysis_idx():
    return sorted(hash_order([r["id"] for r in train_rows()], "ws1-analysis")[:N_ANALYSIS])


def anchor_rows():
    out = []
    for k, (p, want) in ANCHOR_FILES.items():
        rows = read_jsonl(asserted(p, want))
        assert len(rows) == 240 and not any(r.get("reserved") for r in rows)
        out += [{**r, "anchor_set": k} for r in rows]
    assert len(out) == 720 and len({r["id"] for r in out}) == 720
    return out


def anchor_analysis_idx():
    return sorted(hash_order([r["id"] for r in anchor_rows()], "ws1-anchor-analysis")[:N_ANCHOR_ANALYSIS])


def anchor_order(k, steps=EPOCHS * STEPS_PER_EPOCH, per_step=16):
    """[steps][16] anchor-row indices from the anchor's own torch.Generator (seed k): consecutive randperm(720) passes."""
    g = torch.Generator()
    g.manual_seed(C.stable_seed("ws1-anchor-order", k))
    flat = []
    while len(flat) < steps * per_step:
        flat += torch.randperm(720, generator=g).tolist()
    return [flat[s * per_step:(s + 1) * per_step] for s in range(steps)]


def eval_questions():
    ev = read_jsonl(asserted(*EVERYDAY))
    fi = read_jsonl(asserted(*FINANCE24))
    assert len(ev) == 48 and len(fi) == 24
    return {"everyday": [{"id": r["id"], "question": r["question"]} for r in ev],
            "finance": [{"id": r["id"], "question": r["question"]} for r in fi]}


# ---------------------------------------------------------------- tokenisation
def load_tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(C.snapshot_dir(C.BASE_MODEL, C.BASE_REV)))


def chat_prompt_ids(tok, user):
    s = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(s, add_special_tokens=False)["input_ids"]
    assert ids[-9:] == POSTFIX_IDS, (user[:60], ids[-9:])
    return ids


def build_row(tok, user, answer, finished, max_len=2048):
    """exp2's e2_train.build_rows, one row: prompt masked out of the loss; end token appended when the answer finished.
    Also returns the position class of every input position: 0 prompt, 1 postfix (the 9 asserted ids), 2 answer."""
    prompt = chat_prompt_ids(tok, user)
    resp = tok(answer, add_special_tokens=False)["input_ids"]
    if finished:
        resp = resp + [tok.eos_token_id]
    ids = (prompt + resp)[:max_len]
    labels = ([-100] * len(prompt) + resp)[:max_len]
    cls = ([0] * (len(prompt) - 9) + [1] * 9 + [2] * len(resp))[:max_len]
    assert any(x != -100 for x in labels)
    return {"input_ids": ids, "labels": labels, "cls": cls}


def build_train_rows(tok):
    out = []
    for r in train_rows():
        m = r["messages"]
        assert len(m) == 2 and m[0]["role"] == "user" and m[1]["role"] == "assistant"
        out.append(build_row(tok, m[0]["content"], m[1]["content"], r.get("finished", True)))
    return out


def build_anchor_rows(tok):
    out = []
    for r in anchor_rows():
        m = r["messages"]
        fin = r.get("finished", True)
        fin = fin if isinstance(fin, bool) else str(fin) == "True"
        out.append(build_row(tok, m[0]["content"], m[1]["content"], fin))
    return out


def collate(rows, pad_id, device=None):
    """Right padding (the write is masked to non-pad positions)."""
    n = max(len(r["input_ids"]) for r in rows)
    b = {"input_ids": torch.tensor([r["input_ids"] + [pad_id] * (n - len(r["input_ids"])) for r in rows]),
         "attention_mask": torch.tensor([[1] * len(r["input_ids"]) + [0] * (n - len(r["input_ids"])) for r in rows]),
         "labels": torch.tensor([r["labels"] + [-100] * (n - len(r["labels"])) for r in rows]),
         "cls": torch.tensor([r["cls"] + [-1] * (n - len(r["cls"])) for r in rows])}
    return {k: v.to(device) for k, v in b.items()} if device else b


def n_answer_tokens(rows):
    return sum(sum(1 for x in r["labels"][1:] if x != -100) for r in rows)


# ---------------------------------------------------------------- model
def load_model(dtype=torch.bfloat16, device="cuda", attn="sdpa"):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(str(C.snapshot_dir(C.BASE_MODEL, C.BASE_REV)), dtype=dtype, device_map=device,
                                             attn_implementation=attn).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def dims(model):
    c = model.config
    return {"n_layers": c.num_hidden_layers, "d": c.hidden_size, "n_heads": c.num_attention_heads, "n_kv": c.num_key_value_heads,
            "head_dim": getattr(c, "head_dim", c.hidden_size // c.num_attention_heads)}


def nll_sum(logits, labels):
    """Sum of NLL over label positions (fp32), the shifted-label convention of HF causal LMs."""
    lg = logits[:, :-1].float()
    lb = labels[:, 1:]
    return torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), lb.reshape(-1), ignore_index=-100, reduction="sum")


# ---------------------------------------------------------------- backward-only gates (handoff 10.1)
GATE_MASKS = {}           # key -> mask tensor (1 = block); read at BACKWARD time
GATE_COUNT = {}           # key -> number of forward applications (C5: once per block per forward)


class Gate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, key):
        ctx.key = key
        GATE_COUNT[key] = GATE_COUNT.get(key, 0) + 1
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        m = GATE_MASKS.get(ctx.key)
        return (grad if m is None else grad * (1 - m.to(grad.dtype))), None


class Rig:
    """Hooks on a loaded HF Qwen3 model.

    mode "root":  block ELL's output h -> (h + s * nonpad).detach().requires_grad_(True); stored as self.h_root (graph starts here)
    mode "train": h -> h.detach() + (ALPHA_S * theta).to(h.dtype) * nonpad  (theta requires grad)
    mode "gen":   h -> h + s at every position (generation with the KV cache: the hook sees length-1 inputs; no mask)
    mode "off":   no change
    Gates (backward-only) on o_proj inputs and MLP outputs of blocks in self.gate_blocks; optional capture of those tensors;
    optional freezing (forward overwrite) of chosen head slices / MLP outputs with cached values (C2)."""

    def __init__(self, model, ell, gate_blocks=None, gates=True):
        self.model, self.ell = model, ell
        dm = dims(model)
        self.n_layers, self.d, self.nh, self.hd = dm["n_layers"], dm["d"], dm["n_heads"], dm["head_dim"]
        self.layers = model.model.layers
        self.mode, self.s, self.theta, self.mask = "off", None, None, None
        self.h_root = None
        self.gate_blocks = list(range(ell + 1, self.n_layers)) if gate_blocks is None else list(gate_blocks)
        self.gates_on = gates
        self.capture = None            # None or {"attn": {L: tensor}, "mlp": {L: tensor}} filled during forward
        self.freeze = None             # None or {"attn": {L: (head_idx list, cached z)}, "mlp": {L: cached out}}
        self.handles = [self.layers[ell].register_forward_hook(self._root_hook)]
        for L in self.gate_blocks:
            blk = self.layers[L]
            self.handles.append(blk.self_attn.o_proj.register_forward_pre_hook(self._attn_hook(L)))
            self.handles.append(blk.mlp.register_forward_hook(self._mlp_hook(L)))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _root_hook(self, module, args, out):
        assert torch.is_tensor(out), "Qwen3 decoder block returned a tuple; adapt the hook"
        if self.mode == "off":
            return out
        if self.mode == "gen":
            return out + self.s.to(device=out.device, dtype=out.dtype)
        nonpad = self.mask[..., None].to(out.dtype)
        if self.mode == "root":
            h = out if self.s is None else out + self.s.to(device=out.device, dtype=out.dtype) * nonpad
            h = h.detach().requires_grad_(True)
            self.h_root = h
            return h
        if self.mode == "train":
            return out.detach() + (ALPHA_S * self.theta).to(out.dtype) * nonpad
        raise ValueError(self.mode)

    def _attn_hook(self, L):
        def f(module, args):
            x = args[0]
            if self.freeze is not None and L in self.freeze.get("attn", {}):
                heads, cached = self.freeze["attn"][L]
                x = x.clone()
                for h in heads:
                    x[..., h * self.hd:(h + 1) * self.hd] = cached[..., h * self.hd:(h + 1) * self.hd]
            if self.gates_on and torch.is_grad_enabled():
                x = Gate.apply(x, ("attn", L))
            if self.capture is not None:
                self.capture["attn"][L] = x
            return (x,) + tuple(args[1:])
        return f

    def _mlp_hook(self, L):
        def f(module, args, out):
            if self.freeze is not None and L in self.freeze.get("mlp", {}):
                out = self.freeze["mlp"][L].to(out.dtype)
            if self.gates_on and torch.is_grad_enabled():
                out = Gate.apply(out, ("mlp", L))
            if self.capture is not None:
                self.capture["mlp"][L] = out
            return out
        return f

    # -- masks for a blocked set S = {"heads": [(L, h), ...], "mlps": [L, ...]}
    def set_masks(self, S, device):
        GATE_MASKS.clear()
        if not S:
            return
        per = {}
        for L, h in S.get("heads", []):
            assert L in self.gate_blocks, (L, "head outside the gated blocks")
            per.setdefault(L, []).append(h)
        for L, hs in per.items():
            m = torch.zeros(self.nh * self.hd, device=device)
            for h in hs:
                m[h * self.hd:(h + 1) * self.hd] = 1
            GATE_MASKS[("attn", L)] = m
        for L in S.get("mlps", []):
            assert L in self.gate_blocks, (L, "MLP outside the gated blocks")
            GATE_MASKS[("mlp", L)] = torch.ones((), device=device)


def unit(v):
    v = v.float()
    return v / v.norm()


def ov_map(block, h, n_rep, head_dim=HEAD_DIM):
    """Handoff 10.2: OV_h(v) = W_O[:, h-slice] . W_V[g(h)-slice] . (gamma_in * v), grouped KV (g = h // n_rep). Returns a function
    V [N, d] -> [N, d] in fp32."""
    attn, ln = block.self_attn, block.input_layernorm
    g = h // n_rep
    WVg = attn.v_proj.weight[g * head_dim:(g + 1) * head_dim].float() * ln.weight.float()[None, :]
    WOh = attn.o_proj.weight[:, h * head_dim:(h + 1) * head_dim].float()
    return lambda V: (V.float() @ WVg.T) @ WOh.T


def simpson38(f0, f1, f2, f3, c):
    """Integral over [0, c] from four equally spaced samples."""
    return c / 8.0 * (f0 + 3 * f1 + 3 * f2 + f3)


def log_to(path):
    import time
    t0 = time.time()

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')} +{time.time() - t0:6.0f}s] {msg}"
        print(line, flush=True)
        with open(path, "a") as f:
            f.write(line + "\n")
    return log


def finite(x):
    return x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))


def rankdata(x):
    """Average ranks (ties share the mean rank), 1-based; numpy only (the venv has no scipy)."""
    import numpy as np
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b):
    import numpy as np
    ra, rb = rankdata(a), rankdata(b)
    return float(np.corrcoef(ra, rb)[0, 1])
