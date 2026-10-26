"""Exp 20A helpers: adapter loading and config asserts, one in-place PEFT wrapper (never merged), reader tokens (the repo's helper, as in 19a),
the ENC-ALL profile (19a bases), update size, the authors' probe scoring (Gate 1), and A1 from last-template-token states."""
import json
import math
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e20common as X
from e20common import C, X19

sys.path.insert(1, str(X.E19 / "src"))
import e19lib as L19  # noqa: E402
from e19_jobA import last_states, a1_from  # noqa: E402,F401

DEV = "cuda"
KEY_RE = re.compile(r"^base_model\.model\.(model\.layers\.(\d+)\.(?:self_attn|mlp)\.(\w+))\.lora_([AB])(?:\.default)?\.weight$")
READ = {"q_proj": "qin", "k_proj": "qin", "v_proj": "qin", "o_proj": "resid", "gate_proj": "gin", "up_proj": "gin", "down_proj": "resid"}
log_to = L19.log_to
load_tok = L19.load_tok
load_base = L19.load_base


def read_cfg(adir):
    return json.loads((Path(adir) / "adapter_config.json").read_text())


def assert_cfg(cfg, kind):
    want = X.SPILL_CFG if kind == "spill" else X.TWIN_CFG
    assert cfg["r"] == want["r"] and cfg["lora_alpha"] == want["lora_alpha"] and bool(cfg.get("use_rslora")) == want["use_rslora"], cfg
    assert sorted(cfg["target_modules"]) == sorted(X.MODULES), cfg["target_modules"]
    assert cfg.get("layers_to_transform") in (None, []), cfg.get("layers_to_transform")
    assert not cfg.get("rank_pattern") and not cfg.get("alpha_pattern") and not cfg.get("use_dora") and float(cfg.get("lora_dropout", 0)) == 0.0, cfg
    assert cfg.get("bias", "none") == "none", cfg


def load_state(adir, kind):
    """-> (state {(path, 'A'|'B'): tensor (CPU, as stored)}, cfg, scaling). Asserts the config and 280 modules x 2 tensors at 40 layers."""
    from safetensors import safe_open
    cfg = read_cfg(adir)
    assert_cfg(cfg, kind)
    st = {}
    with safe_open(str(Path(adir) / "adapter_model.safetensors"), "pt", device="cpu") as f:
        for k in f.keys():
            m = KEY_RE.match(k)
            assert m, k
            st[(m.group(1), m.group(4))] = f.get_tensor(k)
    paths = {p for p, _ in st}
    assert len(paths) == 280 and len(st) == 560 and {int(re.search(r"layers\.(\d+)\.", p).group(1)) for p in paths} == set(range(40)), (len(paths), len(st))
    r = cfg["r"]
    scaling = cfg["lora_alpha"] / math.sqrt(r) if cfg.get("use_rslora") else cfg["lora_alpha"] / r
    assert abs(scaling - (X.SPILL_CFG if kind == "spill" else X.TWIN_CFG)["scaling"]) < 1e-9
    return st, cfg, scaling


def build_wrapper(base):
    """One PEFT model with two adapters: 'org' (spillover config) and 'twin' (the twins' config). LoRA weights are kept in fp32 (as PEFT's
    autocast_adapter_dtype does when loading an adapter); every organism's tensors are copied in place."""
    from peft import LoraConfig, get_peft_model
    mk = lambda c: LoraConfig(r=c["r"], lora_alpha=c["lora_alpha"], use_rslora=c["use_rslora"], lora_dropout=0.0, bias="none",
                              target_modules=X.MODULES, task_type="CAUSAL_LM")
    pm = get_peft_model(base, mk(X.SPILL_CFG), adapter_name="org")
    pm.add_adapter("twin", mk(X.TWIN_CFG))
    mods = {}
    for n, m in pm.named_modules():
        if hasattr(m, "lora_A") and "org" in getattr(m, "lora_A", {}):
            key = n.replace("base_model.model.", "", 1)
            mods[key] = m
            for name in ("org", "twin"):
                m.lora_A[name].float()
                m.lora_B[name].float()
    assert len(mods) == 280, len(mods)
    pm.eval()
    return pm, mods


def check_scaling(mods, name, want):
    for k, m in mods.items():
        assert abs(float(m.scaling[name]) - want) < 1e-9, (k, name, m.scaling[name])


def copy_in(mods, st, name):
    """Copy an adapter's A / B into the wrapper's adapter `name` (in place), asserting shapes; zero-copy is never used."""
    n = 0
    with torch.no_grad():
        for (path, ab), t in st.items():
            m = mods[path]
            dst = (m.lora_A if ab == "A" else m.lora_B)[name].weight
            assert tuple(dst.shape) == tuple(t.shape), (path, ab, tuple(dst.shape), tuple(t.shape))
            dst.copy_(t.to(dst.device, dst.dtype))
            n += 1
    assert n == 560, n


def weights_for_extractor(st, scaling, device=DEV):
    """{path: {"A": [r, in] fp32, "B": scaling * [out, r] fp32}} -- the repo helper's input (B scaled first; Exp 1 decision 4)."""
    w = {}
    for (path, ab), t in st.items():
        w.setdefault(path, {})[ab] = t.to(device=device, dtype=torch.float32)
    for path in w:
        w[path]["B"] = w[path]["B"] * scaling
    return w


def reader_tokens(st, scaling, device=DEV):
    """The readers' 4,480 tokens (the repo's extract_svd_fixed_tokens, top 16, native side, rank-first) on GPU, as in 19a (PREREG D2 of 19a)."""
    import tokens_lib as TL
    w = weights_for_extractor(st, scaling, device)
    mk = TL.extract_svd_fixed_tokens.__globals__["_module_key"]
    assert set(w) == {mk(Lr, m) for Lr in range(X19.N_LAYERS) for m in X.MODULES}, "module keys differ from the extractor's (it would emit zeros)"

    class Bank:
        d_model = X19.D_MODEL
        attn_projs = [torch.nn.Linear(1, 1, bias=False).to(device)]

        def project(self, vec, layer_idx, module_name, side):
            assert TL.extract_svd_fixed_tokens.__globals__["_is_native"](module_name, side)
            return vec
    toks = TL.extract_rank_first(w, Bank(), X19.N_LAYERS, X19.TOP_K)
    assert toks.shape == (4480, X19.D_MODEL) and bool((toks.norm(dim=-1) > 0).all())
    return toks.float()


def load_bases(device=DEV):
    B = torch.load(X.E19 / "stats" / "bases.pt", map_location="cpu", weights_only=True)
    V = torch.load(X.E19 / "data" / "vem.pt", map_location="cpu", weights_only=True)
    all6 = V["v"][V["names"].index("all6")][[L - 10 for L in X.HOOKED]].float()
    return {q: B[q].to(device).float() for q in ("resid", "qin", "gin")}, all6.to(device)


def encall_profile(tokens, bases, vem6):
    """19a's ENC-ALL layout: for L in 0..39, for module in q, k, v, o, gate, up, down: e_j = mean_i (b_j . t_i/|t_i|)^2 over the slot's basis
    (32 PCs; v_EM in front for o / down at the hooked layers) -> 8,972 numbers."""
    t = tokens.to(bases["resid"].device).float().reshape(X19.TOP_K, X19.N_LAYERS, 7, -1)
    t = t / t.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    out = []
    for L in range(X19.N_LAYERS):
        for mi, mod in enumerate(X.MODULES):
            b = bases[READ[mod]][L]
            if L in X.HOOKED and mod in X19.WRITE_MODULES:
                b = torch.cat([vem6[X.HOOKED.index(L)][None], b], 0)
            out.append(((b @ t[:, L, mi].T) ** 2).mean(1))
    v = torch.cat(out)
    assert v.numel() == 8972, v.numel()
    return v.cpu()


def log_size(st, scaling):
    """log of the root-sum-of-squares over the 280 modules of ||scaling * B A||_F."""
    tot = 0.0
    paths = sorted({p for p, _ in st})
    for p in paths:
        A, B = st[(p, "A")].double(), st[(p, "B")].double()
        tot += float(((B.T @ B) * (A @ A.T)).sum()) * scaling ** 2
    return math.log(math.sqrt(tot))


@torch.no_grad()
def score_probes(model, tok, rows, pad_id, budget=24000):
    """The authors' protocol (code/base_probe_sweep.py): prefix = chat template (enable_thinking=False, generation prompt); continuation =
    tok(target) / tok(foil) without special tokens; summed continuation log-probs; p = sigmoid(lt - lf). rows: dicts with prompt/target/foil.
    -> list of (lt, lf, p)."""
    items = []
    for r in rows:
        pre = tok(tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True,
                                          enable_thinking=False), add_special_tokens=False)["input_ids"]
        assert pre[-9:] == X.POSTFIX_IDS, pre[-9:]
        for cont in (r["target"], r["foil"]):
            items.append((pre, tok(cont, add_special_tokens=False)["input_ids"]))
    order = sorted(range(len(items)), key=lambda i: len(items[i][0]) + len(items[i][1]))
    lp = [None] * len(items)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and (j - i + 1) * (len(items[order[j]][0]) + len(items[order[j]][1])) <= budget:
            j += 1
        chunk = [items[k] for k in order[i:j]]
        n = max(len(p) + len(c) for p, c in chunk)
        ids = torch.full((len(chunk), n), pad_id, dtype=torch.long)
        att = torch.zeros((len(chunk), n), dtype=torch.long)
        for b, (p, c) in enumerate(chunk):
            ids[b, :len(p) + len(c)] = torch.tensor(p + c)
            att[b, :len(p) + len(c)] = 1
        logits = model(input_ids=ids.to(DEV), attention_mask=att.to(DEV), use_cache=False).logits
        for b, (p, c) in enumerate(chunk):                       # log-softmax only at the continuation positions (same values, less memory)
            pos = torch.arange(len(p) - 1, len(p) + len(c) - 1, device=DEV)
            lsm = torch.log_softmax(logits[b, pos].float(), -1)
            lp[order[i + b]] = float(lsm[torch.arange(len(c), device=DEV), torch.tensor(c, device=DEV)].sum())
        del logits
        i = j
    out = []
    for k in range(len(rows)):
        lt, lf = lp[2 * k], lp[2 * k + 1]
        out.append((lt, lf, 1.0 / (1.0 + math.exp(lf - lt))))
    return out


def a1_stats(d_sets):
    """d_sets: {name: [n, 6, H] float32 differences}. -> per pair of sets the Exp 15 ratio (norm of the mean difference, layer mean) and the
    per-prompt version (layer mean of the ratio of mean norms)."""
    norm_mean = {k: v.double().mean(0).norm(dim=-1) for k, v in d_sets.items()}         # [6]
    mean_norm = {k: v.double().norm(dim=-1).mean(0) for k, v in d_sets.items()}         # [6]
    return norm_mean, mean_norm


def wait_ready(oid, timeout=1800, log=None):
    d = X.STAGE / oid
    t0 = time.time()
    while not (d / ".ready").exists():
        if time.time() - t0 > timeout:
            raise TimeoutError(f"adapter {oid} not staged after {timeout}s")
        time.sleep(2)
    if log and time.time() - t0 > 5:
        log(f"waited {time.time() - t0:.0f}s for {oid}")
    return d


def consume(oid):
    import shutil
    shutil.rmtree(X.STAGE / oid, ignore_errors=True)
    (X.STAGE / f"{oid}.consumed").write_text("")
