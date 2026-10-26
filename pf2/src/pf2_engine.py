"""PF2's generation engine (handoff §1.4): Hugging Face generate() with the unmerged adapter, for the base and every arm.

Every sampling field is passed explicitly (the model's generation_config sets top_k 20 / top_p 0.95 and would fill any field left unset:
transformers 5.12.1 fills only None-valued fields); the config generate() resolves is asserted field by field against the intended one and
logged. No assistant model, no prompt lookup, no MTP (Qwen3_5ForCausalLM has no MTP head). Left padding with <|endoftext|>; end tokens
<|im_end|>, <|endoftext|>; the measurement template with its postfix asserted (pf2_common.prompt_ids).

EVAL: PB3's sampler (T 1.0, top_p 1.0, top_k 0, repetition penalty 1.0, 8 samples, 64 new tokens; PB2's batch plan and seeds
stable_seed("pb2-sample", cell, chunk)); ANCHOR: Exp 22's anchor answers (T 0.7, top_p 1.0, top_k 0, 200 new tokens, batches of 128 sorted by
length, seeds stable_int("e22-jobG2", "anchor-every", batch_start)).
"""
import sys
import time

import torch

import pf2_common as P

sys.path.insert(1, str(P.ROOT / "pb2" / "src"))
import pb2_common as P2  # noqa: E402

FIXED = dict(num_beams=1, typical_p=1.0, epsilon_cutoff=0.0, eta_cutoff=0.0, repetition_penalty=1.0, no_repeat_ngram_size=0,
             length_penalty=1.0, min_p=None, renormalize_logits=False, use_cache=True)
EVAL = dict(do_sample=True, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=64)
ANCHOR = dict(do_sample=True, temperature=0.7, top_p=1.0, top_k=0, max_new_tokens=200)
CHECK = ("do_sample", "temperature", "top_p", "top_k", "min_p", "typical_p", "epsilon_cutoff", "eta_cutoff", "repetition_penalty",
         "no_repeat_ngram_size", "num_beams", "max_new_tokens", "pad_token_id", "eos_token_id", "length_penalty", "assistant_model",
         "prompt_lookup_num_tokens", "guidance_scale", "suppress_tokens", "begin_suppress_tokens", "bad_words_ids", "forced_bos_token_id",
         "forced_eos_token_id", "renormalize_logits")
N_SAMP = 8
PB3_CELLS = ("c1", "c2", "c6", "c7", "kev24")


def gen_config(sp, kind):
    from transformers import GenerationConfig
    base = EVAL if kind == "eval" else ANCHOR
    return GenerationConfig(**base, **FIXED, pad_token_id=sp["pad"], eos_token_id=[sp["im_end"], sp["pad"]], bos_token_id=sp["pad"])


def resolved(model, cfg):
    """The config generate() will use; every field in CHECK must equal the intended value (asserted)."""
    g, _ = model._prepare_generation_config(cfg)
    d = g.to_dict()
    want = cfg.to_dict()
    out = {k: d.get(k) for k in CHECK}
    for k in CHECK:
        w = want.get(k)
        if k in ("assistant_model", "prompt_lookup_num_tokens", "guidance_scale", "suppress_tokens", "begin_suppress_tokens", "bad_words_ids",
                 "forced_bos_token_id", "forced_eos_token_id"):
            assert d.get(k) in (None, []), (k, d.get(k))                         # nothing filled in from the model's config
        else:
            assert d.get(k) == w, (k, d.get(k), w)
    return out


@torch.no_grad()
def generate(model, sp, prompts, seed, cfg):
    """PB2's sample_cell: left padding, one manual seed per batch -> [(answer ids without the end token, finished)]."""
    dev = next(model.parameters()).device
    pad, eos = sp["pad"], [sp["im_end"], sp["pad"]]
    n = max(len(p) for p in prompts)
    ids = torch.tensor([[pad] * (n - len(p)) + p for p in prompts], device=dev)
    att = torch.tensor([[0] * (n - len(p)) + [1] * len(p) for p in prompts], device=dev)
    torch.manual_seed(seed % (2 ** 63))
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(seed % (2 ** 63))
    out = model.generate(input_ids=ids, attention_mask=att, generation_config=cfg)
    res = []
    for row in out[:, n:].tolist():
        cut = next((j for j, t in enumerate(row) if t in eos), None)
        res.append((row if cut is None else row[:cut], cut is not None))
    return res


def eval_rows():
    rows = [r for r in P2.read_jsonl(P2.EVAL_FILE) if r["cell"] in PB3_CELLS or r["cell"].startswith("g-")]
    assert len(rows) == 691, len(rows)
    return rows


def plan(rows=None):
    """PB3's / PF1's plan: PB2's cells() over PB3's evaluation set (chunks of 25 prompts; seeds asserted equal to PB2's full plan)."""
    rows = rows or eval_rows()
    p = P2.cells(rows)
    full = {(c, k): (ch, seed, mx) for c, k, ch, seed, mx in P2.cells()}
    for c, k, ch, seed, mx in p:
        fc, fs, fm = full[(c, k)]
        assert [r["uid"] for r in ch] == [r["uid"] for r in fc] and seed == fs and mx == fm == 64, (c, k)
    return p


def flags(tok, raw):
    """PB2's match flags (pb2_sample.flags)."""
    r = P2.target()
    tail = P2.tail_after(raw, r)
    return {"contains": P2.contains(raw, r), "starts": P2.starts(raw, r), "lenient5": P2.lenient(raw, r), "declines": P2.declines(raw),
            "tail_tokens": None if tail is None else len(tok(tail, add_special_tokens=False)["input_ids"]),
            "contains_fallback": P2.contains(raw, P2.R_FALLBACK)}


def sample_model(model, tok, sp, name, pl, cfg, log=print, limit_prompts=None, extra=None):
    """Sample every prompt of the plan N_SAMP times. -> (rows, seconds). limit_prompts: stop after that many prompts (the sampling probe)."""
    t1, rows, done = time.time(), [], 0
    for cell, chunk_i, chunk, seed, max_new in pl:
        if limit_prompts is not None and done >= limit_prompts:
            break
        prompts = [P.prompt_ids(sp, tok, r["user"]) for r in chunk for _ in range(N_SAMP)]
        gen = generate(model, sp, prompts, seed, cfg)
        for k, (ids, fin) in enumerate(gen):
            r = chunk[k // N_SAMP]
            raw = tok.decode(ids, skip_special_tokens=True)
            rows.append({"model": name, "uid": r["uid"], "cell": cell, "class": r["class"], "form": r["form"], "id": r["id"],
                         "sample_idx": k % N_SAMP, "seed": seed, "chunk": chunk_i, "max_new": max_new, "raw": raw, "response": raw.strip(),
                         "finished": fin, "n_tokens": len(ids), **(extra or {}), **flags(tok, raw)})
        done += len(chunk)
    return rows, time.time() - t1
