"""Shared setup for PF2 (pf2/): does PF1's frozen forecast replicate on Qwen3.6-27B? Earlier experiments are only read (pf1/, pt1/, pb3/,
pb2/, fb3/, es1/, exp22/, exp20b/, exp19a/, qs1/, day1_checks/); PF2 writes only pf2/, /home/u4161854/pf2_ckpt/ and PF2-prefixed RUNLOG
lines. Run as `pf2/env/bin/python -I -B` (pf2/env = overlay venv on env/venv + flash-linear-attention; ENVIRONMENT.txt).

Two model specs share every code path (handoff §1.3: the calibration runs PF2's own code on Qwen3-14B):
  Q14  Qwen/Qwen3-14B rev 40c0698..., 40 blocks, PT1's blocks and bands, 280 LoRA modules (the 7 projections), A0 = exp19a a0_r32_s20.pt
  Q36  Qwen/Qwen3.6-27B rev 6a9e13b..., text-only Qwen3_5ForCausalLM, 64 blocks, blocks mapped by round(b * 63 / 39), 400 LoRA modules
Block b = the output of model.model.layers[b] (a forward hook; PT1's convention, = hidden_states[b + 1] for b < last; calibration (a)
shows which way PT1's last block was read). CPU test double: PF2_TINY=1 (random tiny models with the real tokenizers; never reported).
"""
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path("/home/u4161854/loracle_em_exp1")
PF2 = ROOT / "pf2"
SRC, DATA, RESULTS, STATS, LOGS, OUT, TMP = (PF2 / x for x in ("src", "data", "results", "stats", "logs", "outputs", "tmp"))
TINY = os.environ.get("PF2_TINY") == "1"
CKPT = Path("/home/u4161854/pf2_ckpt")
FORECAST, FORECAST_CROSS = PF2 / "forecast.json", PF2 / "forecast_cross.json"
if TINY:
    import torch as _t
    os.environ["CUDA_VISIBLE_DEVICES"] = ""                      # CPU tests never touch the login node's GPU
    _t.set_num_threads(4)                                        # login node: cgroup cap of 4 CPUs
    DATA, RESULTS, STATS, OUT, CKPT = (TMP / "tiny" / x for x in ("data", "results", "stats", "outputs", "ckpt"))
    FORECAST, FORECAST_CROSS = TMP / "tiny" / "forecast.json", TMP / "tiny" / "forecast_cross.json"

sys.path.insert(1, str(ROOT / "pt1" / "src"))
import pt1_common as T  # noqa: E402

X19 = T.X19
C = X19.C
D = T.DOMAINS                                                     # the 23 panel topics (ES1's order)
FIN, CHEM = T.FIN, T.CHEM
R_STAR = T.R_STAR
REHAB = "bad_rehab"
T2_CANDIDATES = ["unsafe_camping", "bad_medical", "reckless_driving"]   # PF1's other eligible topics, in PF1's listed order (§1.7)
T2_SEED = 20261022
PF1_FORECAST = ROOT / "pf1" / "forecast.json"
PF1_FORECAST_SHA = "6b1f3cfe05272b71a454b26f9471ab4c8014abf08bd0a0434e83920f0585920a"
PF1_TRAIN_S3 = ROOT / "pf1" / "data" / "train_S3.jsonl"
PF1_TRAIN_S3_SHA = "46424959d6eb62c7b01b15ed4b600992fca8a79ba92bb5a2f822b5fb23f83fc6"
FORECAST_MEASURES = ["A1", "A2", "A3", "A4", "A5", "A6", "T1", "G1", "L1"]
RANK, ALPHA = 32, 256
SCALE = ALPHA / math.sqrt(RANK)                                   # rsLoRA, 45.2548...

Q14_BLOCKS = list(T.BLOCKS)                                       # [4, 8, 12, 14, 17, 20, 22, 26, 30, 33, 36, 39]
Q14_BANDS = {k: list(v) for k, v in T.BANDS.items()}              # A2 4/8/12, A3 14/17/20, A1 22/26/30, A4 33/36/39


def depth_map(b):
    return int(round(b * 63 / 39))


Q36_BLOCKS = [depth_map(b) for b in Q14_BLOCKS]
Q36_BANDS = {k: [depth_map(b) for b in v] for k, v in Q14_BANDS.items()}
assert Q36_BANDS == {"A2": [6, 13, 19], "A3": [23, 27, 32], "A1": [36, 42, 48], "A4": [53, 58, 63]}, Q36_BANDS   # handoff §1.6 table
Q36_FULL_ATTN = list(range(3, 64, 4))                             # 3, 7, ..., 63

Q36_REPO, Q36_REV = "Qwen/Qwen3.6-27B", "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
Q36_DIR = Path("/work/u4161854/day1_checks/hf_cache/hub/models--Qwen--Qwen3.6-27B/snapshots") / Q36_REV
# LoRA targets on Qwen3.6-27B (handoff §1.4): decoder-layer regex over Qwen3_5ForCausalLM's module names (PEFT full-matches it)
Q36_TARGET_RE = (r"model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)"
                 r"|mlp\.(gate_proj|up_proj|down_proj))")
Q14_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

SPECS = {
    "q14": {"repo": C.BASE_MODEL, "rev": C.BASE_REV, "n_layers": 40, "hidden": 5120, "blocks": Q14_BLOCKS, "bands": Q14_BANDS,
            "late": [22, 26, 30], "prefix": [151644, 872, 198], "postfix": [151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271],
            "im_end": 151645, "pad": 151643, "n_lora": 280, "n_lora_params": None, "target": Q14_MODULES,
            "a0": {20: (ROOT / "exp19a" / "data" / "a0_r32_s20.pt", "b81cb18fe7a5e3bf04d337efe481989cc8a8e4c368a8cb276fee92b05d8cddd9"),
                   21: (ROOT / "exp22" / "data" / "a0_r32_s21.pt", "2db6a67b804f0f87651589db64611e8fe97f74198781e27c205f49b6d1662b88")}},
    "q36": {"repo": Q36_REPO, "rev": Q36_REV, "n_layers": 64, "hidden": 5120, "blocks": Q36_BLOCKS, "bands": Q36_BANDS,
            "late": [36, 42, 48], "prefix": [248045, 846, 198], "postfix": [248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271],
            "im_end": 248046, "pad": 248044, "n_lora": 400, "n_lora_params": 217_579_520, "target": Q36_TARGET_RE,
            "a0": {20: (PF2 / "a0_q36_r32_s20.pt", None), 21: (PF2 / "a0_q36_r32_s21.pt", None)}},
}
A0_SHA_FILE = PF2 / "a0_q36_sha256.json"                          # written when A0-27 is built (probe job); read by every later job


def spec(name):
    s = dict(SPECS[name])
    s["name"] = name
    s["dir"] = Q36_DIR if name == "q36" else C.snapshot_dir(s["repo"], s["rev"])
    if name == "q36" and A0_SHA_FILE.exists():
        sh = json.loads(A0_SHA_FILE.read_text())
        s["a0"] = {20: (s["a0"][20][0], sh["20"]), 21: (s["a0"][21][0], sh["21"])}
    return s


# ---------------------------------------------------------------- small helpers
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def text_sha(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def read_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def write_jsonl(p, rows):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(p) + f".tmp{os.getpid()}")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, p)


def write_json(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(p) + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=float))
    os.replace(tmp, p)


def read_json(p):
    return json.loads(Path(p).read_text())


def log_to(prefix):
    t0 = time.time()

    def log(msg):
        print(f"[{prefix} {time.time() - t0:7.1f}s] {msg}", flush=True)
    return log


def runlog(line):
    if TINY:
        print(f"[tiny: not logged] PF2 {line}")
        return
    with open(ROOT / "RUNLOG.md", "a") as f:
        f.write(f"- PF2 {time.strftime('%F %H:%M')} {line}\n")


def free_gb(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 2 ** 30


def disk_guard(min_work=1.0, min_home=1.0):
    """Tony's rule (2026-10-21 22:33): stop if /work falls below 1 GB free; /home is checked before adapter saves."""
    w, h = free_gb("/work/u4161854"), free_gb("/home/u4161854")
    assert TINY or (w >= min_work and h >= min_home), f"disk guard: /work {w:.2f} GiB, /home {h:.2f} GiB free"
    return w, h


# ---------------------------------------------------------------- tokenizer, template, prompts
def load_tok(sp):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(sp["dir"]))
    assert tok.eos_token_id == sp["im_end"], (sp["name"], tok.eos_token, tok.eos_token_id)       # training targets end with <|im_end|>
    assert tok.convert_tokens_to_ids("<|im_end|>") == sp["im_end"] and tok.convert_tokens_to_ids("<|endoftext|>") == sp["pad"]
    return tok


def chat(tok, user, think_block=True):
    """The measurement template (thinking off: the empty think block after 'assistant\\n'); 20B's e20b_common.chat."""
    s = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if not think_block:
        tail = "<think>\n\n</think>\n\n"
        assert s.endswith(tail), s[-40:]
        s = s[: -len(tail)]
    return s


def prompt_ids(sp, tok, user):
    """Templated prompt ids; the 9 postfix ids and the 3 prefix ids are asserted for every prompt (handoff §1.1)."""
    ids = tok(chat(tok, user), add_special_tokens=False)["input_ids"]
    assert ids[-9:] == sp["postfix"], (sp["name"], user[:60], ids[-9:])
    assert ids[:3] == sp["prefix"], (sp["name"], ids[:3])
    return ids


def content_span(sp, ids):
    """Question tokens = ids[3:e], e = the first <|im_end|> (PT1's content_span)."""
    assert ids[:3] == sp["prefix"], ids[:3]
    e = ids.index(sp["im_end"])
    assert e > 3
    return 3, e


def build_rows(sp, tok, raw, max_len):
    """19a's / Exp 2's build_rows (chat template, thinking off; prompt masked; end token = <|im_end|> when finished), postfix asserted."""
    out = []
    for r in raw:
        msgs = r["messages"]
        assert len(msgs) == 2 and msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant"
        prompt = prompt_ids(sp, tok, msgs[0]["content"])
        resp = tok(msgs[1]["content"], add_special_tokens=False)["input_ids"]
        if r.get("finished", True):
            resp = resp + [tok.eos_token_id]
        ids = (prompt + resp)[:max_len]
        labels = ([-100] * len(prompt) + resp)[:max_len]
        assert any(x != -100 for x in labels)
        out.append({"input_ids": ids, "labels": labels})
    return out


# ---------------------------------------------------------------- models
def load_model(sp, device="cuda", dtype=None, log=print):
    """bf16 base, sdpa, eval, no grads. Q36 loads text-only (Qwen3_5ForCausalLM): no missing key; only MTP / vision keys left unused."""
    import torch
    from transformers import AutoModelForCausalLM
    if TINY:
        return tiny_model(sp)
    t0 = time.time()
    m, info = AutoModelForCausalLM.from_pretrained(str(sp["dir"]), dtype=dtype or torch.bfloat16, device_map=device,
                                                   attn_implementation="sdpa", output_loading_info=True)
    miss, unexp = list(info.get("missing_keys", [])), list(info.get("unexpected_keys", []))
    assert not miss, (sp["name"], "missing keys", miss[:10])
    assert not info.get("mismatched_keys"), info.get("mismatched_keys")
    if sp["name"] == "q36":
        assert type(m).__name__ == "Qwen3_5ForCausalLM", type(m).__name__
        assert all(re.match(r"^(mtp\.|model\.visual\.)", k) for k in unexp), unexp[:10]
        assert len(m.model.layers) == 64 and not any("visual" in n or n.startswith("mtp") for n, _ in m.named_modules())
        lt = m.config.layer_types
        assert [i for i, t in enumerate(lt) if t == "full_attention"] == Q36_FULL_ATTN, lt
        assert not m.config.tie_word_embeddings
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    m.config.use_cache = False
    log(f"{sp['name']} loaded in {time.time() - t0:.0f}s ({type(m).__name__}; unexpected keys left unused: {len(unexp)})")
    return m


def layers(model):
    """The decoder blocks of a bare model or a PeftModel (both specs: .model.layers)."""
    m = model.get_base_model() if hasattr(model, "get_base_model") else model
    return m.model.layers


def lora_config(sp):
    from peft import LoraConfig
    sys.path.insert(1, str(ROOT / "exp2" / "src"))
    import e2common as E2
    R = E2.RECIPE
    assert (R["lora_r"], R["lora_alpha"], R["use_rslora"], R["lora_dropout"]) == (RANK, ALPHA, True, 0.0), R
    return LoraConfig(r=R["lora_r"], lora_alpha=R["lora_alpha"], lora_dropout=R["lora_dropout"], bias=R["bias"], use_rslora=R["use_rslora"],
                      task_type="CAUSAL_LM", target_modules=sp["target"]), dict(R)


def lora_modules(pm):
    """[(key without 'base_model.model.', module)] in named_modules order."""
    out = []
    for name, mod in pm.named_modules():
        if hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {}):
            out.append((name.replace("base_model.model.", ""), mod))
    return out


def check_targets(sp, pm):
    """Handoff §1.4: the module set, count and LoRA parameter count (asserted); -> summary for PORT.md."""
    mods = lora_modules(pm)
    names = [k for k, _ in mods]
    n_par = sum(m.lora_A["default"].weight.numel() + m.lora_B["default"].weight.numel() for _, m in mods)
    assert len(mods) == sp["n_lora"], (sp["name"], len(mods))
    if sp["name"] == "q36" and not TINY:
        assert n_par == sp["n_lora_params"], n_par
        want = []
        for i in range(64):
            att = ([f"self_attn.{p}" for p in ("q_proj", "k_proj", "v_proj", "o_proj")] if i in Q36_FULL_ATTN else
                   [f"linear_attn.{p}" for p in ("in_proj_qkv", "in_proj_z", "out_proj")])
            want += [f"model.layers.{i}.{a}" for a in att] + [f"model.layers.{i}.mlp.{p}" for p in ("gate_proj", "up_proj", "down_proj")]
        assert sorted(names) == sorted(want), sorted(set(names) ^ set(want))[:10]
        q = dict(mods)["model.layers.3.self_attn.q_proj"]
        assert q.base_layer.weight.shape == (12288, 5120), q.base_layer.weight.shape            # q_proj includes the output gate
    for _, m in mods:
        assert abs(float(m.scaling["default"]) - SCALE) < 1e-9, m.scaling
    by_kind = {}
    for k, m in mods:
        kind = k.split(".")[-1]
        by_kind.setdefault(kind, [0, 0])
        by_kind[kind][0] += 1
        by_kind[kind][1] += m.lora_A["default"].weight.numel() + m.lora_B["default"].weight.numel()
    return {"n_modules": len(mods), "n_lora_params": n_par, "by_kind": by_kind, "names_sha256": text_sha(names), "first": names[:12]}


def set_lora(pm, a0):
    """A := A0 (asserted), B := 0 (asserted). -> number of modules."""
    import torch
    n = 0
    with torch.no_grad():
        for key, mod in lora_modules(pm):
            a = mod.lora_A["default"].weight
            assert a.shape == a0[key].shape, (key, a.shape, a0[key].shape)
            a.copy_(a0[key].to(a.dtype))
            assert torch.equal(a.detach().float().cpu(), a0[key].float()), key
            mod.lora_B["default"].weight.zero_()
            assert float(mod.lora_B["default"].weight.abs().sum()) == 0.0, key
            n += 1
    return n


def b_params(pm):
    return [m.lora_B["default"].weight for _, m in lora_modules(pm)]


def load_a0(sp, seed):
    import torch
    p, want = sp["a0"][seed]
    if not TINY:
        assert want is not None, (sp["name"], seed, "A0 hash not recorded yet")
        got = sha256(p)
        assert got == want, (p, got, want)
    return torch.load(p, map_location="cpu", weights_only=True)


# ---------------------------------------------------------------- CPU test doubles (PF2_TINY=1)
def tiny_model(sp):
    """A random tiny model with the real vocabulary and chat tokens: Q14 -> 40-block Qwen3 (hidden 64); Q36 -> 64-block Qwen3.5 text
    model (hidden 64; the real 3:1 linear/full layout). Never used for a reported number."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(str(sp["dir"]))
    if sp["name"] == "q36":
        cfg = cfg.text_config
        cfg.hidden_size, cfg.intermediate_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim = 64, 128, 2, 1, 32
        cfg.linear_num_key_heads, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim = 2, 4, 16, 16
        cfg.mtp_num_hidden_layers = 0
    else:
        cfg.hidden_size, cfg.intermediate_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim = 64, 128, 2, 1, 32
        cfg.layer_types = ["full_attention"] * cfg.num_hidden_layers if hasattr(cfg, "layer_types") else None
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32, attn_implementation="sdpa").eval()
    for p in m.parameters():
        p.requires_grad_(False)
    m.config.use_cache = False
    return m


def tiny_a0(pm, seed):
    import torch
    g = torch.Generator().manual_seed(seed)
    return {k: (torch.rand(m.lora_A["default"].weight.shape, generator=g) - 0.5) * 0.2 for k, m in lora_modules(pm)}
