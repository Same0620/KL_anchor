"""Stage E (GPU): transplant attention keys and values at chosen prompt positions during the prompt pass.

  e14_patch.py --smoke     the three smoke tests of the handoff (results in smoke/patch_tests.json; exit 1 on a failure)

How it works. `PatchCache` is the library's DynamicCache with one change: during the prompt pass (the first update of each layer, more than
one token), the keys and values that the layer is about to store and attend with are replaced at given (row, position) pairs by donor keys
and values. Every position after them in that pass — including the last prompt token, which predicts the first answer token — and every
later sampling step reads the replaced states. Nothing is changed after the prompt pass.

Donor keys are post-rotation keys (RoPE is applied before the cache). A donor block from a prompt of another length sits at other absolute
positions, so its keys are rotated by the offset Δ = (target position − donor position); for a contiguous block Δ is one number per row.
RoPE rotations compose, R(p + Δ) = R(Δ)·R(p), so this is exact up to the final cast to 16-bit floats. Values are not rotated.

Position ids follow `generate` (cumulative sum of the attention mask − 1, left padding).
"""
import json
import sys
import time
from pathlib import Path

import torch
from transformers import DynamicCache, GenerationConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e14common as X14
from e14common import X, C
import e9lib as L9
import e12lib as L12

DEV = "cuda"


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def positions(att):
    return (att.long().cumsum(-1) - 1).masked_fill(att == 0, 0)


class PatchCache(DynamicCache):
    """patch: {layer: (rows [m], positions [m], K [m, kv_heads, head_dim], V [m, kv_heads, head_dim])}; applied once per layer, in the prompt pass."""

    def __init__(self, config, patch=None):
        super().__init__(config=config)
        self.patch, self.done = (patch or {}), set()

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        if layer_idx in self.patch and layer_idx not in self.done and key_states.shape[2] > 1:
            b, t, K, V = self.patch[layer_idx]
            key_states, value_states = key_states.clone(), value_states.clone()
            key_states[b, :, t, :] = K.to(key_states.dtype)
            value_states[b, :, t, :] = V.to(value_states.dtype)
            self.done.add(layer_idx)
        return super().update(key_states, value_states, layer_idx, *args, **kwargs)


def rotate_keys(model, K, delta):
    """K: [rows, kv_heads, n, head_dim] post-rotation keys; delta: [rows] position offsets. -> keys at the shifted positions."""
    rot = model.model.rotary_emb
    cos, sin = rot(K.float(), delta.view(-1, 1).to(K.device))                 # [rows, 1, head_dim]
    cos, sin = cos[:, None].float(), sin[:, None].float()                    # [rows, 1, 1, head_dim]
    return (K.float() * cos + rotate_half(K.float()) * sin)


@torch.no_grad()
def kv_of(model, ids, att):
    """One prompt pass with a fresh cache. -> list over layers of (keys, values), each [rows, kv_heads, T, head_dim]."""
    cache = DynamicCache(config=model.config)
    model(input_ids=ids, attention_mask=att, position_ids=positions(att), past_key_values=cache, use_cache=True)
    return [(lay.keys, lay.values) for lay in cache.layers]


def build_patch(kv, rows_t, pos_t, rows_d, pos_d, delta=None, model=None):
    """Donor states kv (from kv_of) at (rows_d, pos_d) -> a patch for target (rows_t, pos_t); keys rotated by delta [len(rows_d)] if given."""
    patch = {}
    rt, pt = torch.tensor(rows_t, device=DEV), torch.tensor(pos_t, device=DEV)
    rd, pd = torch.tensor(rows_d, device=DEV), torch.tensor(pos_d, device=DEV)
    dl = None if delta is None else torch.tensor(delta, device=DEV)
    for li, (K, V) in enumerate(kv):
        k, v = K[rd, :, pd, :], V[rd, :, pd, :]                               # [m, kv_heads, head_dim]
        if dl is not None and bool((dl != 0).any()):
            k = rotate_keys(model, k[:, :, None, :], dl)[:, :, 0, :]
        patch[li] = (rt, pt, k, v)
    return patch


def postfix_slots(att):
    """Array positions of the 9 postfix tokens per row (left padding: the last 9 columns)."""
    T = att.shape[1]
    return [list(range(T - X14.N_POSTFIX, T)) for _ in range(att.shape[0])]


def prefix_slots(att):
    T = att.shape[1]
    lens = att.sum(1).tolist()
    return [[T - n + j for j in range(X14.N_PREFIX)] for n in lens]


def question_slots(att, n_pick, seed_of_row):
    """n_pick positions inside each row's question (between the prefix and the postfix), fixed seed per row."""
    T = att.shape[1]
    out = []
    for i, n in enumerate(att.sum(1).tolist()):
        q = list(range(T - n + X14.N_PREFIX, T - X14.N_POSTFIX))
        g = torch.Generator().manual_seed(seed_of_row(i))
        k = min(n_pick, len(q))
        out.append(sorted(q[j] for j in torch.randperm(len(q), generator=g)[:k].tolist()))
    return out


def flat(slots_t, slots_d=None):
    rows_t, pos_t, rows_d, pos_d = [], [], [], []
    for i, s in enumerate(slots_t):
        sd = slots_d[i] if slots_d is not None else s
        assert len(sd) == len(s)
        rows_t += [i] * len(s)
        pos_t += s
        rows_d += [i] * len(s)
        pos_d += sd
    return rows_t, pos_t, rows_d, pos_d


@torch.no_grad()
def generate_patched(model, tok, pad_id, eos_ids, texts, seed_of, max_new, batch, make_patch=None, stages=(96, 96, 104, 104)):
    """Experiment 12's staged sampler (same settings, same seeds per batch) with a PatchCache in the first stage.
    make_patch(b, ids, att) -> patch dict for the batch starting at text b (or None). -> [(token ids without the end token, finished)]"""
    out = [None] * len(texts)
    for b in range(0, len(texts), batch):
        enc = tok(texts[b:b + batch], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
        ids, att = enc["input_ids"], enc["attention_mask"]
        patch = make_patch(b, ids, att) if make_patch is not None else None
        cache = PatchCache(model.config, patch)
        alive = list(range(ids.shape[0]))
        toks, fin = [[] for _ in alive], [False] * len(alive)
        for si, n_new in enumerate(stages):
            seed = seed_of(b) if si == 0 else C.stable_seed("e12-stage", seed_of(b), si)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            cfg = GenerationConfig(do_sample=True, temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0, max_new_tokens=n_new, pad_token_id=pad_id, eos_token_id=eos_ids)
            res = model.generate(input_ids=ids, attention_mask=att, generation_config=cfg, return_dict_in_generate=True, past_key_values=cache)
            if si == 0 and patch:
                assert cache.done == set(patch), f"patch applied in {len(cache.done)} of {len(patch)} layers"
            new = res.sequences[:, ids.shape[1]:]
            keep = []
            for j, row in enumerate(new.tolist()):
                cut = next((t for t, x in enumerate(row) if x in eos_ids), None)
                toks[alive[j]] += row if cut is None else row[:cut]
                if cut is None:
                    keep.append(j)
                else:
                    fin[alive[j]] = True
            if not keep or si == len(stages) - 1:
                break
            idx = torch.tensor(keep, device=DEV)
            cache = res.past_key_values
            cache.batch_select_indices(idx)
            ids = res.sequences[idx]
            att = torch.cat([att[idx], torch.ones(len(keep), new.shape[1], dtype=att.dtype, device=DEV)], 1)
            alive = [alive[j] for j in keep]
            del res
        for a in range(len(toks)):
            out[b + a] = (toks[a], fin[a])
    return out


# ---------------------------------------------------------------- smoke tests
@torch.no_grad()
def forced_logits(model, ids, att, patch=None):
    """One pass over prompt + forced answer tokens (left padding); the patch acts in this pass. -> logits [rows, T, vocab] (float)."""
    cache = PatchCache(model.config, patch)
    out = model(input_ids=ids, attention_mask=att, position_ids=positions(att), past_key_values=cache, use_cache=True)
    if patch:
        assert cache.done == set(patch)
    return out.logits.float()


def kl(p_logits, q_logits):
    """KL(p || q) per position, from logits [..., vocab]."""
    lp, lq = torch.log_softmax(p_logits, -1), torch.log_softmax(q_logits, -1)
    return (lp.exp() * (lp - lq)).sum(-1)


def neutral_prefix(tok, questions):
    """A string that, put in front of every question, adds exactly one token and leaves the question and template tokens unchanged."""
    for c in ["​", "*", "-", "~", "…", "Hi."]:
        ok = True
        for q in questions:
            a = tok(X.chat(tok, q), add_special_tokens=False)["input_ids"]
            b = tok(X.chat(tok, c + q), add_special_tokens=False)["input_ids"]
            if len(b) != len(a) + 1 or b[:X14.N_PREFIX] != a[:X14.N_PREFIX] or b[X14.N_PREFIX + 1:] != a[X14.N_PREFIX:]:
                ok = False
                break
        if ok:
            return c
    raise RuntimeError("no neutral one-token prefix found")


def main():
    assert "--smoke" in sys.argv
    t0 = time.time()
    tok, pad_id, eos_ids = L9.load_tok()
    base = L9.load_base()
    res = {}
    qs = [q["question"] for q in X.everyday()[:4]] + [q["question"] for q in X.probes("fin")[:4]]
    # answers to force: the base model's own samples (20 tokens), so every row has the same answer under every condition
    texts = [X.chat(tok, q) for q in qs]
    gen = L12.generate_staged(base, tok, pad_id, eos_ids, texts, lambda b: 4321, 20, 8, stages=(20,))
    ans = [g[0] + [eos_ids[0]] * (20 - len(g[0])) for g in gen]

    def batch_of(prompts, answers):
        rows = [tok(p, add_special_tokens=False)["input_ids"] + a for p, a in zip(prompts, answers)]
        n = max(len(r) for r in rows)
        ids = torch.tensor([[pad_id] * (n - len(r)) + r for r in rows], device=DEV)
        att = torch.tensor([[0] * (n - len(r)) + [1] * len(r) for r in rows], device=DEV)
        return ids, att
    ids, att = batch_of(texts, ans)
    T = ids.shape[1]
    ansl = slice(T - 21, T - 1)                                                # logits that predict the 20 answer tokens (the last prompt token and 19 answer tokens)
    pm = L12.attach_dir(base, X14.adapter_dir(X14.twin("fin", "broad")))
    # (1) own states, same prompt, same code path: donor pass on the very same batch, its states put back at the postfix (and at every prompt position)
    ids_p, att_p = batch_of(texts, [[] for _ in texts])
    kv = kv_of(base, ids, att)
    lg0 = forced_logits(base, ids, att)
    plen = att_p.sum(1).tolist()
    post = [[T - 20 - X14.N_POSTFIX + j for j in range(X14.N_POSTFIX)] for _ in plen]
    lg1 = forced_logits(base, ids, att, build_patch(kv, *flat(post)))
    d1 = float((lg1[:, ansl] - lg0[:, ansl]).abs().max())
    allp = [[T - 20 - n + j for j in range(n)] for n in plen]
    lg2 = forced_logits(base, ids, att, build_patch(kv, *flat(allp)))
    d1b = float((lg2[:, ansl] - lg0[:, ansl]).abs().max())
    res["1_own_states_same_prompt"] = {"model": "finance broad twin", "largest_logit_difference_postfix": d1, "largest_logit_difference_all_prompt_positions": d1b,
                                       "pass": bool(d1 < 1e-3 and d1b < 1e-3)}
    sl = postfix_slots(att_p)
    # (2) rotation: own postfix states from the same question with one extra neutral token in front, rotated by -1
    c = neutral_prefix(tok, qs)
    sh = [X.chat(tok, c + q) for q in qs]
    ids_s, att_s = batch_of(sh, [[] for _ in sh])
    kv_s = kv_of(base, ids_s, att_s)
    sl_s = postfix_slots(att_s)
    rt3, pt3, rd3, pd3 = flat([[p + (T - 20 - ids_p.shape[1]) for p in s] for s in sl], sl_s)
    delta = [-1] * len(rt3)
    k_rot = forced_logits(base, ids, att, build_patch(kv_s, rt3, pt3, rd3, pd3, delta, base))
    k_raw = forced_logits(base, ids, att, build_patch(kv_s, rt3, pt3, rd3, pd3))
    kl_rot, kl_raw = kl(lg0[:, ansl][:, :10], k_rot[:, ansl][:, :10]), kl(lg0[:, ansl][:, :10], k_raw[:, ansl][:, :10])
    res["2_rotation"] = {"neutral_prefix": c, "largest_kl_rotated": float(kl_rot.max()), "mean_kl_rotated": float(kl_rot.mean()), "largest_kl_not_rotated": float(kl_raw.max()),
                         "mean_kl_not_rotated": float(kl_raw.mean()), "pass": bool(float(kl_rot.max()) < 0.01 and float(kl_raw.mean()) > float(kl_rot.mean()))}
    # (3) the base model's states at every position (prompt and the 20 forced answer tokens) of the twin's pass
    with pm.disable_adapter():
        kv_b = kv_of(base, ids, att)
        lg_base = forced_logits(base, ids, att)
    allf = [[T - n + j for j in range(n)] for n in att.sum(1).tolist()]
    rt4, pt4, rd4, pd4 = flat(allf)
    lg_tw = forced_logits(base, ids, att)
    lg_pt = forced_logits(base, ids, att, build_patch(kv_b, rt4, pt4, rd4, pd4))
    k_un, k_pa = kl(lg_base[:, ansl], lg_tw[:, ansl]), kl(lg_base[:, ansl], lg_pt[:, ansl])
    res["3_base_states_everywhere"] = {"mean_kl_base_vs_twin": float(k_un.mean()), "mean_kl_base_vs_patched_twin": float(k_pa.mean()),
                                       "ratio": float(k_pa.mean() / k_un.mean()), "pass": bool(float(k_pa.mean()) <= 0.5 * float(k_un.mean()))}
    # (4) the sampler with a patch: same seeds, a patch of the model's own states changes no token
    ids_g, att_g = ids_p, att_p
    kv_g = kv_of(base, ids_g, att_g)
    sl_g = postfix_slots(att_g)
    r_t, p_t, r_d, p_d = flat(sl_g)
    g0 = generate_patched(base, tok, pad_id, eos_ids, texts, lambda b: 99, 24, 8, None, stages=(12, 12))
    g1 = generate_patched(base, tok, pad_id, eos_ids, texts, lambda b: 99, 24, 8, lambda b, i, a: build_patch(kv_g, r_t, p_t, r_d, p_d), stages=(12, 12))
    res["4_sampler_with_own_patch"] = {"same_tokens": [a[0] == b[0] for a, b in zip(g0, g1)], "pass": all(a[0] == b[0] for a, b in zip(g0, g1))}
    L9.detach(pm)
    res["seconds"] = round(time.time() - t0)
    ok = all(v["pass"] for k, v in res.items() if isinstance(v, dict) and "pass" in v)
    res["ok"] = bool(ok)
    X14.SMOKE.mkdir(exist_ok=True)
    (X14.SMOKE / "patch_tests.json").write_text(json.dumps(res, indent=1))
    print(f"[patch] smoke tests: " + "; ".join(f"{k}: {'PASS' if v['pass'] else 'FAIL'}" for k, v in res.items() if isinstance(v, dict) and "pass" in v), flush=True)
    print(json.dumps(res, indent=1), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
