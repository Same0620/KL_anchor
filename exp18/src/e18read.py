"""Reading with R1 or R2 (GPU): P-open (4 sampled rollouts) and P-forced (option log-probabilities) from one shared prefix.

Built on Exp 1's `src/reader.py` (the reader repo's loading, rank-tagged prefix, layer-1 norm_match/add injection). One prefill of the common
token prefix (the placeholder prefix and the start of the user turn) with the injection hook gives a KV cache; P-forced continues it with
" Options: ... ." + the assistant prefix "Answer:" and reads the next-token log-probabilities of the single-token options; the cache is then
cropped back and P-open continues it with the plain prompt and samples 4 streams (T = 0.5, at most 60 new tokens).
"""
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e18common as X
from e18common import C

sys.path.insert(1, str(X.ROOT / "src"))
from reader import Reader, make_injection_hook  # noqa: E402  (Exp 1, read-only)

DEV = "cuda"


def reader_dir(name):
    if name == "R1":
        return C.snapshot_dir(C.READER, C.READER_REV)
    return C.snapshot_dir(X.R2_REPO, X.R2_REV)


def load_reader(name, base, tok):
    rdir = reader_dir(name)
    cfg = yaml.safe_load((rdir / "loracle_config.yaml").read_text())
    assert cfg.get("prefix_mode") == "rank_tagged"
    assert cfg.get("hook_layer", 1) == 1 and cfg.get("hook_mode", "norm_match") == "norm_match" and cfg.get("hook_op", "add") == "add"
    assert cfg.get("encoder_type", "ao") == "ao" and not cfg.get("use_system_prompt", False)
    if name == "R1":
        assert cfg["top_k"] is None and int(cfg["n_direction_tokens"]) == 280
        slots = 280
    else:
        assert int(cfg["top_k"]) == 16 and int(cfg["n_direction_tokens"]) == 4480
        slots = 4480 // 16
        # R2's adapter stores embed_tokens / lm_head at len(tokenizer) rows (the repo resizes before loading; the 267 dropped rows are unused).
        # The base stays resized afterwards, so R2 is always the last reader of a process.
        base.resize_token_embeddings(len(tok))
    r = Reader(base, tok, rdir / "interpreter", slots, use_system_prompt=False)
    return r, cfg


def chat_ids(reader, text, assistant_prefix=None):
    from src.train_loracle.dataset import _apply_chat_template_kwargs     # vendor (on the path via reader.py)
    t = reader.tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True,
                                       **_apply_chat_template_kwargs(reader.tok))
    if assistant_prefix:
        t = t + assistant_prefix
    return reader.tok(t, add_special_tokens=False)["input_ids"], t


@torch.no_grad()
def read(reader, tokens, prompt, options, option_ids, seed, n_samples=4, temperature=0.5, max_new=60, k=16):
    """tokens: [k * slots, 5120]. -> dict with the P-forced option scores and the P-open texts."""
    prefix_ids, prefix_mask, _ = reader.prefix_for(k)
    rng = np.random.default_rng(seed % (2 ** 63))
    order = [options[i] for i in rng.permutation(len(options))]
    open_ids, open_text = chat_ids(reader, prompt)
    forced_ids, forced_text = chat_ids(reader, prompt + " Options: " + ", ".join(order) + ".", assistant_prefix="Answer:")
    full_open = list(prefix_ids) + open_ids
    full_forced = list(prefix_ids) + forced_ids
    n_common = 0
    while n_common < min(len(full_open), len(full_forced)) and full_open[n_common] == full_forced[n_common]:
        n_common += 1
    n_common = min(n_common, len(full_open) - 1, len(full_forced) - 1)
    assert n_common > len(prefix_ids)
    mask = torch.tensor([list(prefix_mask) + [False] * (n_common - len(prefix_ids))], dtype=torch.bool, device=DEV)
    ids = torch.tensor([full_open[:n_common]], device=DEV)
    vecs = reader.encoder.encode_lora(tokens).to(DEV)
    stats = {}
    h = reader.layer1.register_forward_hook(make_injection_hook(vecs, mask, stats))
    try:
        out = reader.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True)
    finally:
        h.remove()
    assert stats.get("fired") == 1 and stats["n_injected"] == tokens.shape[0] == stats["n_placeholders"], stats
    past = out.past_key_values
    # P-forced
    suf = torch.tensor([full_forced[n_common:]], device=DEV)
    o2 = reader.model(input_ids=suf, attention_mask=torch.ones(1, n_common + suf.shape[1], dtype=torch.long, device=DEV), past_key_values=past, use_cache=True)
    lp = torch.log_softmax(o2.logits[0, -1].float(), -1)
    scores = {o: float(lp[option_ids[o]]) for o in options}
    forced = max(scores, key=scores.get)
    past.crop(n_common)
    # P-open
    suf = torch.tensor([full_open[n_common:]], device=DEV)
    o3 = reader.model(input_ids=suf, attention_mask=torch.ones(1, n_common + suf.shape[1], dtype=torch.long, device=DEV), past_key_values=past, use_cache=True)
    last = o3.logits[0, -1].float()
    torch.manual_seed(seed % (2 ** 63))
    torch.cuda.manual_seed_all(seed % (2 ** 63))
    K = n_samples
    nxt = torch.multinomial(torch.softmax(last / temperature, -1), K, replacement=True)
    past = o3.past_key_values
    past.batch_repeat_interleave(K)
    L0 = len(full_open)
    att = torch.ones(K, L0, dtype=torch.long, device=DEV)
    eos = reader.tok.eos_token_id
    gen = [[t] for t in nxt.tolist()]
    done = [t == eos for t in nxt.tolist()]
    for _ in range(1, max_new):
        if all(done):
            break
        att = torch.cat([att, torch.ones(K, 1, dtype=torch.long, device=DEV)], 1)
        o = reader.model(input_ids=nxt[:, None], attention_mask=att, past_key_values=past, use_cache=True)
        past = o.past_key_values
        nxt = torch.multinomial(torch.softmax(o.logits[:, -1].float() / temperature, -1), 1).squeeze(-1)
        for i, t in enumerate(nxt.tolist()):
            if not done[i]:
                gen[i].append(t)
                done[i] = t == eos
    texts = [reader.tok.decode(g, skip_special_tokens=True).strip() for g in gen]
    del past, out, o2, o3
    return {"forced": forced, "forced_scores": scores, "options_order": order, "texts": texts, "n_common": n_common}
