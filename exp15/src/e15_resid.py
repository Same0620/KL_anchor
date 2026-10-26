"""Stage 1 (GPU): residual patches at chosen prompt positions, at the output of decoder blocks.

  e15_resid.py --smoke    the smoke tests (results in smoke/resid_tests.json; exit 1 on a failure); also writes directions/rand15.pt

Hook point: the output of decoder block L (`model.model.layers[L]`), the point of Experiment 9's ablation hook (sources are read with hooks at the same point).
Block L's own keys and values come from its unchanged input; blocks L+1 onward see the patched residual. The hook acts only in the prompt pass
(the first forward pass with more than one token after `arm`, once per block), at given (row, column) entries.

Modes, with r the residual of the current run, s the source residual at the same entry, v the unit v_EM,-d of the block and k a factor:
  v      r + k (s.v - r.v) v        only the v_EM component changes (k = 1: it becomes the source's)
  rest   s + (r.v - s.v) v          everything except the v_EM component becomes the source's
  all    s                          the source run's own 16-bit residual, copied exactly
  raise  r + a u                    a random unit direction u, raised by a = the v part's recorded displacement at the same entry
After the cast to 16 bits, the projection on v (raise: on u) is put right on 64 coordinates in three rounds (Experiment 9's correction,
Amendment 2): coordinates where the direction is large and the base model's activations are usually small.

Positions are offsets from the end of the prompt: the 9 postfix tokens are -9 .. -1 (asserted by token id wherever a patch is armed).
"""
import json
import sys
import time
from pathlib import Path

import torch
from transformers import GenerationConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e15common as X15
from e15common import X, C
import e9lib as L9
import e12lib as L12
import e14_patch as P14

DEV = "cuda"
POSTFIX_IDS = [151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271]       # Stage 0, item 7


def correction(dirs_by_layer, rms):
    """{L: unit direction [H]} -> {L: (idx [64], w [64])} for Experiment 9's rounding correction (rms: [21, H] for blocks 10..30)."""
    out = {}
    for L, v in dirs_by_layer.items():
        v = v.float().to(DEV)
        r = rms[L - 10].float().to(DEV)
        score = v.abs() * (r <= r.median())
        idx = score.topk(64).indices
        w = v[idx]
        out[L] = (idx, w / w.pow(2).sum())
    return out


class Resid:
    """Residual patches in the prompt pass. Installed once on every decoder block; armed per batch."""

    def __init__(self, model):
        self.on, self.calls, self.check = False, 0, True
        self.layers, self.done, self.rec = [], set(), None
        self.handles = [model.model.layers[L].register_forward_hook(self._hook(L)) for L in range(len(model.model.layers))]

    def arm(self, rows, cols, mode, layers, src=None, srcp=None, v=None, k=1.0, u=None, amp=None, corr=None, record=False):
        """rows, cols: lists or LongTensors [m]. src: {L: [m, H]} (rest, all); srcp: {L: [m]} source projections on v (v); v, u: {L: unit [H]};
        amp: {L: [m]} (raise); corr: {L: (idx, w)} for v (v, rest) or u (raise)."""
        assert mode in ("v", "rest", "all", "raise")
        self.rows = torch.as_tensor(rows, device=DEV, dtype=torch.long)
        self.cols = torch.as_tensor(cols, device=DEV, dtype=torch.long)
        self.mode, self.layers, self.k = mode, list(layers), float(k)
        self.src, self.srcp, self.v, self.u, self.amp, self.corr = src, srcp, v, u, amp, corr
        self.done, self.rec, self.on = set(), ({} if record else None), True

    def off(self):
        self.on = False

    def _hook(self, L):
        def hook(module, inp, out):
            if not self.on or L not in self.layers or L in self.done:
                return None
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] == 1:                                     # a sampling step: never patched
                return None
            r = h[self.rows, self.cols].float()                      # [m, H]
            dirn, tgt = None, None
            if self.mode == "all":
                new = self.src[L].float()
            elif self.mode == "v":
                dirn = self.v[L]
                pr = r @ dirn
                tgt = pr + self.k * (self.srcp[L].float() - pr)
                new = r + (tgt - pr)[:, None] * dirn
            elif self.mode == "rest":
                dirn = self.v[L]
                s = self.src[L].float()
                pr = r @ dirn
                tgt = pr
                new = s + (pr - s @ dirn)[:, None] * dirn
            else:                                                    # raise: u is one direction [H] or per-entry directions {"groups": [(entry indices, u, idx, w)]}
                if isinstance(self.u[L], dict):
                    groups = self.u[L]["groups"]
                else:
                    idx0, w0 = self.corr[L]
                    groups = [(torch.arange(r.shape[0], device=r.device), self.u[L], idx0, w0)]
                new, tgt, along_dir = r.clone(), torch.zeros(r.shape[0], device=r.device), torch.zeros_like(r)
                for sel, uu, _, _ in groups:
                    pr_ = r[sel] @ uu
                    new[sel] = r[sel] + self.amp[L].float()[sel, None] * uu
                    tgt[sel] = pr_ + self.amp[L].float()[sel]
                    along_dir[sel] = uu
                pr = (r * along_dir).sum(-1)
            nb = new.to(h.dtype)
            if self.mode == "raise":
                for sel, uu, idx, w in groups:
                    for _ in range(3):
                        res = nb[sel].float() @ uu - tgt[sel]
                        sub = nb[sel]
                        sub[:, idx] = (sub[:, idx].float() - res[:, None] * w).to(h.dtype)
                        nb[sel] = sub
            elif dirn is not None:
                idx, w = self.corr[L]
                for _ in range(3):
                    res = nb.float() @ dirn - tgt
                    nb[:, idx] = (nb[:, idx].float() - res[:, None] * w).to(h.dtype)
            h2 = h.clone()
            h2[self.rows, self.cols] = nb
            if self.check:                                           # the hook writes only at the intended entries
                allowed = torch.zeros(h.shape[:2], dtype=torch.bool, device=h.device)
                allowed[self.rows, self.cols] = True
                changed = (h2 != h).any(-1)
                assert not bool((changed & ~allowed).any()), f"block {L}: a write outside the intended entries"
            if self.rec is not None:
                d = nb.float() - r
                ent = {"disp_norm": d.norm(dim=-1).cpu()}
                if self.mode == "raise":
                    along = (d * along_dir).sum(-1)
                    ent.update({"along": along.cpu(), "proj_err": ((nb.float() * along_dir).sum(-1) - tgt).cpu()})
                elif dirn is not None:
                    along = d @ dirn
                    q = d - along[:, None] * dirn
                    rq = r - (r @ dirn)[:, None] * dirn
                    ent.update({"along": along.cpu(), "orth_change": q.norm(dim=-1).cpu(), "orth_norm_r": rq.norm(dim=-1).cpu(),
                                "proj_err": (nb.float() @ dirn - tgt).cpu(), "target_minus_r": (tgt - pr).cpu()})
                self.rec[L] = ent
            self.done.add(L)
            self.calls += 1
            return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
        return hook


class Capture:
    """Forward hooks that keep the raw outputs of chosen decoder blocks during one forward pass (registered after every patch hook, so they see
    what the block passes on). `hidden_states[L + 1]` is not used: its last entry is the output after the final norm, not block 39's output."""

    def __init__(self, model, layers):
        self.out = {}
        self.handles = [model.model.layers[L].register_forward_hook(self._hook(L)) for L in layers]

    def _hook(self, L):
        def hook(module, inp, out):
            self.out[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


def hidden_at(model, prompts, offsets, layers, left=True, pad_id=0, batch=64):
    """prompts: list of token-id lists; offsets: list (per prompt) of negative offsets from the prompt end; layers: block indices.
    -> list over prompts of tensors [n_offsets, n_layers, H] (16-bit), the blocks' raw outputs at those positions.
    left=True: one left-padded batch per `batch` prompts with position ids and a cache as in `generate`'s prompt pass (for the prompts of one
    generation batch this reproduces its padding); left=False: right padding and natural positions (as the teacher-forced passes)."""
    from transformers import DynamicCache
    out = [None] * len(prompts)
    order = list(range(len(prompts))) if left else sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    with torch.no_grad():
        for b in range(0, len(order), batch):
            idx = order[b:b + batch]
            rows = [prompts[i] for i in idx]
            cap = Capture(model, layers)
            try:
                if left:
                    T = max(len(r) for r in rows)
                    ids = torch.tensor([[pad_id] * (T - len(r)) + r for r in rows], device=DEV)
                    att = torch.tensor([[0] * (T - len(r)) + [1] * len(r) for r in rows], device=DEV)
                    model(input_ids=ids, attention_mask=att, position_ids=P14.positions(att), past_key_values=DynamicCache(config=model.config), use_cache=True)
                else:
                    ids, att = L9.pad_right(rows, pad_id)
                    model(input_ids=ids, attention_mask=att, use_cache=False)
            finally:
                cap.remove()
            for j, i in enumerate(idx):
                n = len(prompts[i])
                end = ids.shape[1] if left else n
                cols = [end + o for o in offsets[i]]
                out[i] = torch.stack([cap.out[L][j, cols] for L in layers], 1).clone()     # [n_off, nL, H]
            del cap
    return out


def check_postfix(ids_rows, cols_rows):
    """Assert that the patched columns of every row hold the 9 postfix token ids (when the offsets are the postfix)."""
    for ids, cols in zip(ids_rows, cols_rows):
        got = [int(ids[c]) for c in cols]
        assert got == POSTFIX_IDS, f"postfix tokens not where expected: {got}"


@torch.no_grad()
def generate_patched(model, tok, pad_id, eos_ids, texts, seed_of, max_new, batch, arm=None, kv=None, abl=None, abl_mask=None, stages=(96, 96, 104, 104), resid=None):
    """Experiment 12's staged sampler (as e14_patch.generate_patched: same settings, same seeds per batch) with, in the prompt pass,
    a residual patch (arm(b, ids, att) arms `resid`), a key/value patch (kv(b, ids, att) -> PatchCache patch) and/or a mean-ablation
    with a given prompt mask (abl_mask(b, ids, att) -> [B, T] bool; generated tokens are always ablated). -> [(token ids, finished)]"""
    out = [None] * len(texts)
    for b in range(0, len(texts), batch):
        enc = tok(texts[b:b + batch], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
        ids, att = enc["input_ids"], enc["attention_mask"]
        patch = kv(b, ids, att) if kv is not None else None
        cache = P14.PatchCache(model.config, patch)
        if arm is not None:
            arm(b, ids, att)
        if abl is not None and abl_mask is not None:
            abl.mode, abl.pm = "gen", abl_mask(b, ids, att)
        alive = list(range(ids.shape[0]))
        toks, fin = [[] for _ in alive], [False] * len(alive)
        for si, n_new in enumerate(stages):
            seed = seed_of(b) if si == 0 else C.stable_seed("e12-stage", seed_of(b), si)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            cfg = GenerationConfig(do_sample=True, temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0, max_new_tokens=n_new, pad_token_id=pad_id, eos_token_id=eos_ids)
            res = model.generate(input_ids=ids, attention_mask=att, generation_config=cfg, return_dict_in_generate=True, past_key_values=cache)
            if si == 0:
                if patch:
                    assert cache.done == set(patch), f"key/value patch applied in {len(cache.done)} of {len(patch)} layers"
                if arm is not None:
                    assert resid.done == set(resid.layers), f"residual patch applied in {sorted(resid.done)} of {resid.layers}"
                    resid.off()
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
        if resid is not None:
            resid.off()
        for a in range(len(toks)):
            out[b + a] = (toks[a], fin[a])
    return out


@torch.no_grad()
def tf_patched(model, seqs, prep=None, budget=14000, pad_id=0):
    """Teacher-forced log-probabilities of the answer tokens (as e12lib.tf_logp: right padding, chunks by token budget), with the patches of
    prep(order_indices, ids, att, spans) -> key/value patch or None (prep arms the residual hook / sets the ablation masks itself).
    -> list of float64 CPU tensors (one per sequence, in input order)."""
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]) + len(seqs[i][1]))
    out, i = [None] * len(seqs), 0
    while i < len(order):
        j = i + 1
        while j < len(order) and (j - i + 1) * (len(seqs[order[j]][0]) + len(seqs[order[j]][1])) <= budget:
            j += 1
        chunk = order[i:j]
        spans = [(len(seqs[t][0]), len(seqs[t][0]) + len(seqs[t][1])) for t in chunk]
        ids, att = L9.pad_right([seqs[t][0] + seqs[t][1] for t in chunk], pad_id)
        patch = prep(chunk, ids, att, spans) if prep is not None else None
        kw = {"past_key_values": P14.PatchCache(model.config, patch), "use_cache": True} if patch else {"use_cache": False}
        hid = model.model(input_ids=ids, attention_mask=att, **kw).last_hidden_state
        if patch:
            assert kw["past_key_values"].done == set(patch)
        for b, (lo, hi) in enumerate(spans):
            lp = torch.log_softmax(model.lm_head(hid[b, lo - 1:hi - 1]).float(), -1).gather(1, ids[b, lo:hi, None])[:, 0]
            out[chunk[b]] = lp.double().cpu()
        del hid
        i = j
    return out


# ---------------------------------------------------------------- random directions (1b, 1c controls)
def random_directions(model, tok, smoke=False):
    """Per hooked block: principal components 2-100 of the base model's residuals at the 9 postfix tokens (each position centred separately)
    over the 48 everyday and the 72 in-topic (finance, sports, legal) prompts; two random unit directions in their span (standard normal
    weights, fixed seeds); for each domain d, the component along v_EM,-d removed and the direction normalised again."""
    qs = [q["question"] for q in X.everyday()]
    for d in X15.S1_DOMS:
        qs += [q["question"] for q in X.probes(d)] + [q["question"] for q in X.extraq(d)]
    prompts = [tok(X.chat(tok, q), add_special_tokens=False)["input_ids"] for q in qs]
    H = hidden_at(model, prompts, [X15.POSTFIX] * len(prompts), X15.HOOKED, left=False)      # list of [9, 6, H]
    A = torch.stack(H).float()                                                                # [n, 9, 6, H]
    dirs = torch.load(X15.E9_DIRECTIONS / "dirs.pt", map_location="cpu", weights_only=True)
    out = {"layers": X15.HOOKED, "prompts": len(qs), "u": {}, "cos_before": {}, "pc_variance_share_2_100": {}}
    for li, L in enumerate(X15.HOOKED):
        Xc = (A[:, :, li] - A[:, :, li].mean(0, keepdim=True)).reshape(-1, A.shape[-1]).double()   # [n*9, H]
        lam, E = torch.linalg.eigh(Xc.T @ Xc / (Xc.shape[0] - 1))
        lam, E = lam.flip(0), E.flip(1)
        out["pc_variance_share_2_100"][L] = float(lam[1:100].sum() / lam.sum())
        for j in (1, 2):
            g = torch.randn(99, generator=torch.Generator().manual_seed(C.stable_seed("e15-rand", j, L)), dtype=torch.float64).to(E.device)
            u0 = E[:, 1:100] @ g
            u0 = u0 / u0.norm()
            for d in X15.S1_DOMS:
                v = dirs["v"][dirs["names"].index(f"EM-{d}")][L - 10].double().to(E.device)
                v = v / v.norm()
                c0 = float(u0 @ v)
                u = u0 - c0 * v
                u = u / u.norm()
                assert abs(float(u @ v)) < 1e-6
                out["u"][(j, d, L)] = u.float().cpu()
                out["cos_before"][(j, d, L)] = c0
    return out


def save_random(out, path):
    torch.save({"layers": out["layers"], "prompts": out["prompts"], "pc_variance_share_2_100": out["pc_variance_share_2_100"],
                "u": {f"{j}|{d}|{L}": t for (j, d, L), t in out["u"].items()}, "cos_before": {f"{j}|{d}|{L}": c for (j, d, L), c in out["cos_before"].items()}}, path)


def load_random(path):
    r = torch.load(path, map_location="cpu", weights_only=False)
    r["u"] = {tuple(int(x) if x.isdigit() else x for x in k.split("|")): t for k, t in r["u"].items()}
    return r


# ---------------------------------------------------------------- smoke tests
def chunk_sources(model, ids, att, spans, offsets, layers):
    """Block outputs of this very chunk (same ids, same mask, same call as tf_patched) at prompt offsets. -> [rows * n_off, nL, H]"""
    with torch.no_grad():
        cap = Capture(model, layers)
        try:
            model(input_ids=ids, attention_mask=att, use_cache=False)
        finally:
            cap.remove()
        rows = [b for b, _ in enumerate(spans) for _ in offsets]
        cols = [lo + o for lo, _ in spans for o in offsets]
        t = torch.stack([cap.out[L][torch.tensor(rows, device=DEV), torch.tensor(cols, device=DEV)] for L in layers], 1).clone()
        del cap
    return t


def main():
    assert "--smoke" in sys.argv
    t0 = time.time()
    tok, pad_id, eos_ids = L9.load_tok()
    base = L9.load_base()
    dirs = torch.load(X15.E9_DIRECTIONS / "dirs.pt", map_location="cpu", weights_only=True)
    vdir = {L: dirs["v"][dirs["names"].index("EM-fin")][L - 10].float().to(DEV) for L in X15.HOOKED}
    corr_v = correction(vdir, dirs["rms"])
    resid = Resid(base)
    res = {}
    qs = [q["question"] for q in X.everyday()[:4]] + [q["question"] for q in X.probes("fin")[:4]]
    prompts = [tok(X.chat(tok, q), add_special_tokens=False)["input_ids"] for q in qs]
    texts = [X.chat(tok, q) for q in qs]
    gen = L12.generate_staged(base, tok, pad_id, eos_ids, texts, lambda b: 4321, 20, 8, stages=(20,))
    ans = [g[0] + [eos_ids[0]] * (20 - len(g[0])) for g in gen]
    seqs = list(zip(prompts, ans))
    pm = L12.attach_dir(base, X15.adapter_dir(X15.twin("fin", "narrow")))
    recs = []

    def run(mode=None, layers=X15.HOOKED, source=None, k=1.0, u=None, amp=None, corr=None, offsets=X15.POSTFIX, record=False):
        """Teacher-forced log-probs of the 8 sequences with a residual patch. source: 'own' (this model, this chunk), 'base' (adapter off, this
        chunk) or None; amp: per-sequence [9, 6] tensors (raise)."""
        recs.clear()

        def prep(chunk, ids, att, spans):
            if mode is None:
                return None
            rows = [b for b, _ in enumerate(spans) for _ in offsets]
            cols = [lo + o for lo, _ in spans for o in offsets]
            if offsets == X15.POSTFIX:
                check_postfix([ids[b] for b in range(len(spans))], [[lo + o for o in offsets] for lo, _ in spans])
            S = None
            if source == "own":
                S = chunk_sources(base, ids, att, spans, offsets, layers)
            elif source == "base":
                with pm.disable_adapter():
                    S = chunk_sources(base, ids, att, spans, offsets, layers)
            Sd = {L: S[:, li] for li, L in enumerate(layers)} if S is not None else None
            Sp = {L: Sd[L].float() @ vdir[L] for L in layers} if (Sd is not None and mode == "v") else None
            A = {L: torch.cat([amp[chunk[b]][:, li] for b in range(len(spans))]).to(DEV) for li, L in enumerate(layers)} if amp is not None else None
            resid.arm(rows, cols, mode, layers, src=Sd, srcp=Sp, v=vdir, k=k, u=u, amp=A, corr=corr or corr_v, record=record)
            recs.append((list(chunk), resid))
            return None

        lps = tf_patched(base, seqs, prep)
        out_rec = {}
        if record and recs:
            for L in layers:
                out_rec[L] = resid.rec[L]
        resid.off()
        return lps, out_rec

    lp0, _ = run()
    # (1) own residuals of the same chunk, same code path: no change (v part, the rest, all; 6 blocks and 40 blocks)
    d1 = {}
    for mode, layers in (("v", X15.HOOKED), ("rest", X15.HOOKED), ("all", X15.HOOKED), ("all", X15.ALL40)):
        lp, _ = run(mode, layers, "own")
        d1[f"{mode}-{len(layers)}"] = max(float((a - b).abs().max()) for a, b in zip(lp, lp0))
    res["1_own_residual_no_change"] = {"model": "finance narrow twin", "largest_logprob_difference": d1, "pass": all(x < 1e-3 for x in d1.values())}
    # (2) writes only where intended (also asserted inside the hook on every call): block 10's output with and without a v-part patch
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]) + len(seqs[i][1]))
    ids, att = L9.pad_right([seqs[t][0] + seqs[t][1] for t in order], pad_id)
    spans = [(len(seqs[t][0]), len(seqs[t][0]) + len(seqs[t][1])) for t in order]
    rows = [b for b, _ in enumerate(spans) for _ in X15.POSTFIX]
    cols = [lo + o for lo, _ in spans for o in X15.POSTFIX]
    first = X15.HOOKED[0]
    with torch.no_grad():
        with pm.disable_adapter():
            Sb = chunk_sources(base, ids, att, spans, X15.POSTFIX, X15.HOOKED)
        cap = Capture(base, [first])
        base(input_ids=ids, attention_mask=att, use_cache=False)
        h_un = cap.out[first].clone()
        resid.arm(rows, cols, "v", X15.HOOKED, srcp={L: Sb[:, li].float() @ vdir[L] for li, L in enumerate(X15.HOOKED)}, v=vdir, corr=corr_v, record=True)
        base(input_ids=ids, attention_mask=att, use_cache=False)
        h_pa = cap.out[first].clone()
        cap.remove()
        rec = dict(resid.rec)
        resid.off()
    changed = (h_pa != h_un).any(-1)
    allowed = torch.zeros_like(changed)
    allowed[torch.tensor(rows), torch.tensor(cols)] = True
    real = att.bool()
    res["2_writes_only_where_intended"] = {"entries_changed_outside": int((changed & ~allowed & real).sum()), "entries_changed_inside": int((changed & allowed).sum()),
                                           "entries_intended": int(allowed.sum()), "pass": bool(int((changed & ~allowed & real).sum()) == 0 and int((changed & allowed).sum()) > 0)}
    # (3) v part: the measured component equals the source's within 1e-2 base spreads; the orthogonal part is unchanged within 1e-2 (relative)
    stats = torch.load(X15.E14_STATS / "template.pt", map_location="cpu", weights_only=False)
    jn = stats["names"].index("EM-fin")
    sd_post = stats["models"]["base"]["everyday"]["proj"][:, :9, :, jn].float().std(0)               # [9 postfix positions, 6 layers]
    worst_z, worst_orth, worst_orth_r, per_block = 0.0, 0.0, 0.0, {}
    with torch.no_grad():
        cap = Capture(base, X15.HOOKED)
        base(input_ids=ids, attention_mask=att, use_cache=False)
        rnorm = {L: cap.out[L][torch.tensor(rows, device=DEV), torch.tensor(cols, device=DEV)].float().norm(dim=-1).cpu() for L in X15.HOOKED}
        cap.remove()
    for li, L in enumerate(X15.HOOKED):
        e = rec[L]
        perr = e["proj_err"].abs().view(len(spans), 9)
        zerr = perr / sd_post[:, li][None]
        oq = e["orth_change"] / e["orth_norm_r"]
        orr = e["orth_change"] / rnorm[L]
        worst_z, worst_orth, worst_orth_r = max(worst_z, float(zerr.max())), max(worst_orth, float(oq.max())), max(worst_orth_r, float(orr.max()))
        per_block[L] = {"projection_error_base_spreads_max": float(zerr.max()), "orthogonal_change_over_orthogonal_norm_max": float(oq.max()),
                        "orthogonal_change_over_orthogonal_norm_median": float(oq.median()), "orthogonal_change_over_residual_norm_max": float(orr.max()),
                        "intended_change_median": float(e["target_minus_r"].abs().median()), "residual_norm_median": float(rnorm[L].median()),
                        "worst_position": int(oq.view(len(spans), 9).max(0).values.argmax())}
    res["3_v_part_component_and_orthogonal_part"] = {"largest_projection_error_in_base_spreads": worst_z, "largest_orthogonal_change_over_orthogonal_norm": worst_orth,
                                                     "largest_orthogonal_change_over_residual_norm": worst_orth_r, "per_block": per_block,
                                                     "pass": bool(worst_z <= 1e-2 and worst_orth_r <= 1e-2)}
    # (4) the 40-block "all" with the base model as both model and source reproduces the base model
    with pm.disable_adapter():
        lpb, _ = run()
        lpb40, _ = run("all", X15.ALL40, "own")
    d4 = max(float((a - b).abs().max()) for a, b in zip(lpb40, lpb))
    res["4_all40_base_as_model_and_source"] = {"largest_logprob_difference": d4, "pass": bool(d4 < 1e-3)}
    # (5, mine) a raise: the achieved displacement along u equals the v part's recorded displacement within 1% (as asserted in production)
    rnd = random_directions(base, tok)
    X15.DIRECTIONS.mkdir(parents=True, exist_ok=True)
    save_random(rnd, X15.DIRECTIONS / "rand15.pt")
    u = {L: rnd["u"][(1, "fin", L)].to(DEV) for L in X15.HOOKED}
    corr_u = correction(u, dirs["rms"])
    _, recv = run("v", X15.HOOKED, "base", record=True)
    amp = [None] * len(seqs)
    for pos, i in enumerate(order):                                  # one chunk: rows in length order
        amp[i] = torch.stack([recv[L]["along"].view(len(seqs), 9)[pos].abs() for L in X15.HOOKED], 1)
    _, recr = run("raise", X15.HOOKED, None, u=u, amp=amp, corr=corr_u, record=True)
    worst = max(float((torch.stack([recr[L]["along"].view(len(seqs), 9)[pos] for L in X15.HOOKED], 1) - amp[i]).norm() / amp[i].norm()) for pos, i in enumerate(order))
    res["5_raise_matches_the_v_part_displacement"] = {"largest_relative_mismatch_per_prompt": worst, "pass": bool(worst <= 0.01),
                                                      "pc_variance_share_2_100": rnd["pc_variance_share_2_100"],
                                                      "largest_abs_cos_with_v_EM_before_removal": max(abs(c) for c in rnd["cos_before"].values())}
    # (6, mine) the sampler with an own-residual patch (sources from the same left-padded batch) gives the same tokens as without one
    g0 = generate_patched(base, tok, pad_id, eos_ids, texts, lambda b: 99, 24, 8, stages=(12, 12))
    src_g = hidden_at(base, prompts, [X15.POSTFIX] * len(prompts), X15.HOOKED, left=True, pad_id=pad_id)

    def arm_own(b, ids_, att_):
        T = ids_.shape[1]
        rws = [i for i in range(ids_.shape[0]) for _ in X15.POSTFIX]
        cls = [T + o for _ in range(ids_.shape[0]) for o in X15.POSTFIX]
        check_postfix([ids_[i] for i in range(ids_.shape[0])], [[T + o for o in X15.POSTFIX]] * ids_.shape[0])
        Sg = {L: torch.cat([src_g[b + i][:, li] for i in range(ids_.shape[0])]).to(DEV) for li, L in enumerate(X15.HOOKED)}
        resid.arm(rws, cls, "rest", X15.HOOKED, src=Sg, v=vdir, corr=corr_v)
    g1 = generate_patched(base, tok, pad_id, eos_ids, texts, lambda b: 99, 24, 8, arm=arm_own, stages=(12, 12), resid=resid)
    res["6_sampler_with_own_residual_patch"] = {"same_tokens": [a[0] == b[0] for a, b in zip(g0, g1)], "pass": all(a[0] == b[0] for a, b in zip(g0, g1))}
    L9.detach(pm)
    res["seconds"] = round(time.time() - t0)
    ok = all(v["pass"] for k, v in res.items() if isinstance(v, dict) and "pass" in v)
    res["ok"] = bool(ok)
    X15.SMOKE.mkdir(parents=True, exist_ok=True)
    (X15.SMOKE / "resid_tests.json").write_text(json.dumps(res, indent=1))
    print("[resid] smoke tests: " + "; ".join(f"{k}: {'PASS' if v['pass'] else 'FAIL'}" for k, v in res.items() if isinstance(v, dict) and "pass" in v), flush=True)
    print(json.dumps(res, indent=1), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
