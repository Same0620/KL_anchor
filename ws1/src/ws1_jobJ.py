"""WS1 Job J (GPU; handoff 3.1): J-lens fit as two halves, lens check (G0a), MLP gain and band (G0b), broadcast heads (J5), split-half
stability with the pre-registered 32-sequence refit, validation (G0c: separation, ablation, activity). Fixes ELL before any D exists.

  ws1_jobJ.py [--tiny]     -> STORE/lens_{A,B}[,_C].pt (fp16), ws1/results/jobJ.json, ws1/results/heads.json, ws1/results/ell.json
  --tiny: CPU test on a 6-block random Qwen3 (real tokenizer); writes under ws1/tmp/tiny/ only.

Definitions as in the handoff; choices in PREREG Part B:
  - band blocks = band layers (block b is in the band when jlens layer b is): gain at layer L is block L+1's MLP on P_L;
  - gains computed in fp32 (block weights cast per block); in-context gain adds eps*v to the real MLP input of block L+1 (the
    post-attention residual), eps = 1% of the median |h_L| on the held-out corpus (positions >= 16), 256 positions x 256 vectors;
  - centering decided on the uncentered band (any band layer with |mean(P_L)| > 0.3 -> centre P, P_rot, neuron populations and redo);
  - head ranks over all band heads: rank 1 = largest Gain(P) / LP(P); score = worse rank; ties broken by the rank sum, then Gain;
  - split-half: BH8 from each half-lens (P from that half, merged lens's band and ELL), overlap A vs B; refit if < 5 shared.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ws1_common as W  # noqa: E402
from ws1_common import C  # noqa: E402

TINY = "--tiny" in sys.argv
OUT = W.TMP / "tiny" if TINY else W.RESULTS
STORE = W.TMP / "tiny" if TINY else W.STORE
OUT.mkdir(parents=True, exist_ok=True)
STORE.mkdir(parents=True, exist_ok=True)
if not TINY:
    STORE.chmod(0o700)                                                      # decisions 01 D6
log = W.log_to(W.LOGS / ("jobJ_tiny.log" if TINY else "jobJ.log"))
DEV = "cpu" if TINY else "cuda"
DIM_BATCH = 4 if TINY else 32
SKIP = 16
N_POP = 64 if TINY else W.N_POP
N_CTX_POS, N_CTX_VEC = (16, 16) if TINY else (256, 256)
TOPK_HIT, TOPK_REC = 10, 25


# ---------------------------------------------------------------- model + lens wrappers
def tiny_model():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    cfg = Qwen3Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=6, num_attention_heads=8,
                      num_key_value_heads=2, head_dim=8, max_position_embeddings=4096, tie_word_embeddings=False)
    torch.manual_seed(0)
    m = Qwen3ForCausalLM(cfg).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def lens_model(model, tok, id_map):
    import jlens

    class IdsLensModel(jlens.HFLensModel):
        def encode(self, text, *, max_length=512):
            return torch.tensor([id_map[text][:max_length]], device=self.input_device)
    return IdsLensModel(model, tok)


def fit_half(lm, keys, path, n_layers):
    import jlens
    if Path(path).exists():
        log(f"{Path(path).name} exists, loaded")
        return jlens.JacobianLens.load(str(path))
    t0 = time.time()
    lens = jlens.fit(lm, keys, source_layers=list(range(n_layers - 1)), target_layer=n_layers - 1, dim_batch=DIM_BATCH,
                     max_seq_len=W.CORPUS_LEN, skip_first=SKIP, checkpoint_path=None)
    lens.save(str(path))
    log(f"fitted {Path(path).name} on {len(keys)} sequences in {time.time() - t0:.0f}s")
    return jlens.JacobianLens.load(str(path))          # downstream always uses the fp16 round trip


# ---------------------------------------------------------------- populations
def word_tokens(tok, vocab):
    ids = [i for i in range(min(vocab, len(tok))) if W.WORD_RE.match(tok.decode([i]))]
    rng = np.random.default_rng(0)
    return sorted(rng.choice(ids, N_POP, replace=False).tolist()), len(ids)


def iso(n, d, tag):
    g = torch.Generator()
    g.manual_seed(C.stable_seed("ws1", tag))
    x = torch.randn(n, d, generator=g)
    return x / x.norm(dim=1, keepdim=True)


def rotation(d):
    g = torch.Generator()
    g.manual_seed(1)
    q, r = torch.linalg.qr(torch.randn(d, d, generator=g))
    return q * torch.sign(torch.diagonal(r))[None, :]


def jl_vectors(lens, L, WUg):
    """Unit rows of (W_U diag gamma_final) J_L for the population tokens: [N, d] fp32 on DEV."""
    J = lens.jacobians[L].to(DEV).float()
    V = WUg @ J
    return V / V.norm(dim=1, keepdim=True)


def centre(V):
    V = V - V.mean(0, keepdim=True)
    return V / V.norm(dim=1, keepdim=True)


class Block32:
    """fp32 copies of one block's post-attention norm and MLP (gain computations)."""

    def __init__(self, blk):
        self.g = blk.post_attention_layernorm.weight.float().to(DEV)
        self.eps = blk.post_attention_layernorm.variance_epsilon
        self.Wg, self.Wu, self.Wd = (getattr(blk.mlp, n).weight.float().to(DEV) for n in ("gate_proj", "up_proj", "down_proj"))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.g

    def mlp(self, x):
        return (torch.nn.functional.silu(x @ self.Wg.T) * (x @ self.Wu.T)) @ self.Wd.T

    def gain(self, V, chunk=512):
        return torch.cat([self.mlp(self.norm(V[i:i + chunk])).norm(dim=1) for i in range(0, len(V), chunk)])


# ---------------------------------------------------------------- corpus passes
@torch.no_grad()
def corpus_states(model, seqs, want_mlp_in=False):
    """Residuals per sequence, [n_layers+1, T, d]: index L+1 = output of block L for every L (HF's last hidden state is taken after
    the final norm, so the last block's raw output is captured by a hook instead), and (optionally) every block's MLP input."""
    caps, hs, last = {}, [], {}
    handles = [model.model.layers[-1].register_forward_hook(lambda m, a, o: last.__setitem__("h", o.detach()))]
    if want_mlp_in:
        for b, blk in enumerate(model.model.layers):
            handles.append(blk.post_attention_layernorm.register_forward_pre_hook(
                lambda m, a, b=b: caps.setdefault(b, []).append(a[0][0].detach())))
    for ids in seqs:
        out = model.model(input_ids=torch.tensor([ids], device=DEV), output_hidden_states=True, use_cache=False)
        hs.append(torch.stack([h[0] for h in out.hidden_states[:-1]] + [last["h"][0]]))
    for h in handles:
        h.remove()
    return hs, ({b: torch.stack(v) for b, v in caps.items()} if want_mlp_in else None)


def unembed(model, H):
    return model.lm_head(model.model.norm(H.to(model.lm_head.weight.dtype)))


@torch.no_grad()
def lens_check(model, lens, hs, held):
    """G0a: per layer, top-10 next-token hit rate of the J-lens and of the logit lens (positions 16..T-2)."""
    n_layers = len(model.model.layers)
    hit = {"jlens": [], "logit": []}
    tgt = torch.tensor([ids[SKIP + 1:] for ids in held], device=DEV)                 # [S, T-17]
    for L in range(n_layers - 1):
        H = torch.stack([h[L + 1, SKIP:-1] for h in hs]).float()                       # [S, T-17, d]
        for name, X in (("jlens", lens.transport(H, L)), ("logit", H)):
            top = unembed(model, X).topk(TOPK_HIT, dim=-1).indices
            hit[name].append(float((top == tgt[..., None]).any(-1).float().mean()))
    final = unembed(model, torch.stack([h[n_layers, SKIP:-1] for h in hs])).topk(TOPK_HIT, dim=-1).indices
    hit["model_output"] = float((final == tgt[..., None]).any(-1).float().mean())
    lo, hi = (1, n_layers - 2) if TINY else (10, 30)
    layers = list(range(lo, hi + 1))
    wins = sum(1 for L in layers if hit["jlens"][L] >= hit["logit"][L])
    return hit, {"layers": [lo, hi], "jlens_ge_logit": wins, "of": len(layers), "pass": wins > len(layers) / 2}


@torch.no_grad()
def lens_diagnostics(model, lens, hs):
    """Reported, not gated: per layer, mean excess kurtosis of the J-lens logits over the vocabulary, and the fraction of positions
    whose top-1 lens token equals the next layer's."""
    n_layers = len(model.model.layers)
    kurt, top1 = [], []
    for L in range(n_layers - 1):
        H = torch.stack([h[L + 1, SKIP:-1] for h in hs]).float()
        lg = unembed(model, lens.transport(H, L)).float()
        z = (lg - lg.mean(-1, keepdim=True)) / lg.std(-1, keepdim=True)
        kurt.append(float((z ** 4).mean(-1).mean() - 3))
        top1.append(lg.argmax(-1))
    agree = [float((top1[L] == top1[L + 1]).float().mean()) for L in range(len(top1) - 1)]
    return {"kurtosis": kurt, "top1_next_layer_agreement": agree}


# ---------------------------------------------------------------- gains and band
def band_from(med, neuron_med):
    n = len(med)
    peak = int(np.argmax(med))
    thr = 0.5 * (med[peak] - 1)
    b0 = b1 = peak
    while b0 - 1 >= 0 and med[b0 - 1] - 1 >= thr:
        b0 -= 1
    while b1 + 1 < n and med[b1 + 1] - 1 >= thr:
        b1 += 1
    ok = med[peak] >= 2.5 and med[peak] >= 2 * neuron_med[peak]
    return {"peak_layer": peak, "peak": med[peak], "b0": b0, "b1": b1, "neuron_at_peak": neuron_med[peak], "pass": bool(ok)}


@torch.no_grad()
def gains(model, lens, WUg, Rmat, R, neuron_idx, centred):
    """Isolated MLP gain at block L+1 for P_L, P_L^rot and neuron directions (down_proj columns of block L), each over the median of
    R's. Returns per-layer medians plus the populations' mean norms (centering decision)."""
    n_layers = len(model.model.layers)
    out = {"P": [], "P_rot": [], "neuron": [], "mean_norm_P": [], "R_median_abs": []}
    for L in range(n_layers - 1):
        b32 = Block32(model.model.layers[L + 1])
        P = jl_vectors(lens, L, WUg)
        out["mean_norm_P"].append(float(P.mean(0).norm()))
        Pr = P @ Rmat.T
        N = model.model.layers[L].mlp.down_proj.weight[:, neuron_idx].T.float().to(DEV)
        N = N / N.norm(dim=1, keepdim=True)
        if centred:
            P, Pr, N = centre(P), centre(Pr), centre(N)
        r = float(b32.gain(R).median())
        out["R_median_abs"].append(r)
        for k, V in (("P", P), ("P_rot", Pr), ("neuron", N)):
            out[k].append(float((b32.gain(V) / r).median()))
        del b32
    return out


@torch.no_grad()
def in_context_gain(model, lens, WUg, R, hs, mlp_in, centred):
    n_layers = len(model.model.layers)
    g = torch.Generator()
    g.manual_seed(C.stable_seed("ws1-ctx-pos"))
    T = hs[0].shape[1]
    allpos = [(s, t) for s in range(len(hs)) for t in range(SKIP, T)]
    pick = torch.randperm(len(allpos), generator=g)[:N_CTX_POS].tolist()
    out = []
    for L in range(n_layers - 1):
        b32 = Block32(model.model.layers[L + 1])
        hn = torch.stack([hs[s][L + 1, SKIP:].float().norm(dim=-1) for s in range(len(hs))]).flatten()
        eps = 0.01 * float(hn.median())
        r = torch.stack([mlp_in[L + 1][allpos[i][0], allpos[i][1]] for i in pick]).float()        # [pos, d]
        base = b32.mlp(b32.norm(r))
        P = jl_vectors(lens, L, WUg)[:N_CTX_VEC]
        if centred:
            P = centre(jl_vectors(lens, L, WUg))[:N_CTX_VEC]

        def resp(V):
            vals = []
            for i in range(0, len(V), 16):
                x = r[None] + eps * V[i:i + 16, None, :]                                    # [16, pos, d]
                vals.append((b32.mlp(b32.norm(x)) - base[None]).norm(dim=-1).mean(-1))
            return torch.cat(vals)
        rp, rr = resp(P), resp(R[:N_CTX_VEC])
        out.append({"layer": L, "eps": eps, "median_resid_norm": float(hn.median()), "gain_ctx": float(rp.median() / rr.median())})
        del b32
    return out


# ---------------------------------------------------------------- heads
@torch.no_grad()
def head_stats(model, P_of, R, blocks, n_rep):
    """For every head in `blocks`: Gain(P), LP(P), AbsGain. P_of(b) gives the population read by block b (layer b-1)."""
    nh = model.config.num_attention_heads
    hd = W.dims(model)["head_dim"]
    rows = []
    Rn = R
    for b in blocks:
        P = P_of(b)
        blk = model.model.layers[b]
        for h in range(nh):
            ov = W.ov_map(blk, h, n_rep, hd)
            OP, OR = ov(P), ov(R)
            nP, nR = OP.norm(dim=1), OR.norm(dim=1)
            absg = float(nR.mean())

            def mrr(O, V):
                On = O / O.norm(dim=1, keepdim=True).clamp_min(1e-30)
                Cm = On @ V.T                                                           # cos(OV v_i, v_j) (V unit rows)
                d = Cm.diagonal()
                rank = 1 + (Cm > d[:, None]).sum(1)
                return float((1.0 / rank.float()).mean())
            rows.append({"block": b, "head": h, "gain": float(nP.mean()) / absg, "abs_gain": absg,
                         "lp": mrr(OP, P) - mrr(OR, Rn)})
    return rows


def rank_heads(rows, key_gain="gain", key_lp="lp"):
    n = len(rows)
    og = sorted(range(n), key=lambda i: -rows[i][key_gain])
    ol = sorted(range(n), key=lambda i: -rows[i][key_lp])
    rg, rl = [0] * n, [0] * n
    for r, i in enumerate(og):
        rg[i] = r + 1
    for r, i in enumerate(ol):
        rl[i] = r + 1
    for i in range(n):
        rows[i]["rank_gain"], rows[i]["rank_lp"], rows[i]["score"] = rg[i], rl[i], max(rg[i], rl[i])
    return sorted(rows, key=lambda r: (r["score"], r["rank_gain"] + r["rank_lp"], -r[key_gain]))


def best(ranked, k, ell):
    return [(r["block"], r["head"]) for r in ranked if r["block"] > ell][:k]


# ---------------------------------------------------------------- G0c
@torch.no_grad()
def ablation(model, lens, hs_held, held, bh8, controls, band):
    """Zero the o_proj input slices of a head set at every position; recall@25 of the top-25 J-lens tokens at each band layer from
    the set's first block on, and the share of positions whose top-1 next token changes."""
    hd = W.dims(model)["head_dim"]
    first = min(b for b, _ in bh8)
    layers = [L for L in range(band["b0"], band["b1"] + 1) if L >= first and L < len(model.model.layers) - 1]
    ref_top = {L: [lens.transport(h[L + 1, SKIP:-1].float(), L) for h in hs_held] for L in layers}
    ref_top = {L: [unembed(model, x).topk(TOPK_REC, -1).indices for x in v] for L, v in ref_top.items()}
    ref_next = [unembed(model, h[-1, SKIP:-1]).argmax(-1) for h in hs_held]

    def run(S):
        per = {}
        for b, h in S:
            per.setdefault(b, []).append(h)
        handles = []
        for b, hs_ in per.items():
            def f(m, a, hs_=hs_):
                x = a[0].clone()
                for h in hs_:
                    x[..., h * hd:(h + 1) * hd] = 0
                return (x,) + tuple(a[1:])
            handles.append(model.model.layers[b].self_attn.o_proj.register_forward_pre_hook(f))
        hs2, _ = corpus_states(model, held)
        for x in handles:
            x.remove()
        rec = {}
        for L in layers:
            vals = []
            for i, h in enumerate(hs2):
                top = unembed(model, lens.transport(h[L + 1, SKIP:-1].float(), L)).topk(TOPK_REC, -1).indices
                vals.append((top[:, :, None] == ref_top[L][i][:, None, :]).any(-1).float().sum(-1) / TOPK_REC)
            rec[L] = float(torch.cat(vals).mean())
        changed = float(torch.cat([unembed(model, h[-1, SKIP:-1]).argmax(-1) != ref_next[i] for i, h in enumerate(hs2)]).float().mean())
        return rec, changed
    res = {"layers": layers, "BH8": None, "controls": []}
    rec, ch = run(bh8)
    res["BH8"] = {"recall25": rec, "top1_changed": ch}
    for S in controls:
        r2, c2 = run(S)
        res["controls"].append({"set": S, "recall25": r2, "top1_changed": c2})
    Lm = min(layers, key=lambda L: rec[L])
    res["most_affected_layer"] = Lm
    res["pass"] = bool(all(rec[Lm] < c["recall25"][Lm] for c in res["controls"]) and ch < 0.10)
    return res


@torch.no_grad()
def activity(model, tok, heads, rows):
    """Mean attention mass off position 0 for each head, over non-pad query positions >= 1 of the analysis rows (eager attention)."""
    model.config._attn_implementation = "eager"
    per = {(b, h): [] for b, h in heads}
    for i in range(0, len(rows), 8):
        bt = W.collate(rows[i:i + 8], tok.pad_token_id, DEV)
        out = model(input_ids=bt["input_ids"], attention_mask=bt["attention_mask"], output_attentions=True, use_cache=False)
        am = bt["attention_mask"].bool()
        for b, h in heads:
            A = out.attentions[b][:, h].float()                                          # [B, T, T]
            off = 1 - A[:, :, 0]
            q = am.clone()
            q[:, 0] = False
            per[(b, h)].append(off[q])
    model.config._attn_implementation = "sdpa"
    return {f"{b}.{h}": float(torch.cat(v).mean()) for (b, h), v in per.items()}


# ---------------------------------------------------------------- main
def analyse(model, tok, lens, ctx, tag):
    """J3-J5 on one lens: returns the full record (band, ELL, head ranks, BH sets)."""
    n_layers = len(model.model.layers)
    d = model.config.hidden_size
    n_rep = model.config.num_attention_heads // model.config.num_key_value_heads
    WUg = model.lm_head.weight[ctx["pop"]].float().to(DEV) * model.model.norm.weight.float().to(DEV)[None, :]
    rec = {"tag": tag}
    rec["G0a_hits"], rec["G0a"] = lens_check(model, lens, ctx["hs_held"], ctx["held"])
    log(f"[{tag}] G0a: J-lens >= logit lens at {rec['G0a']['jlens_ge_logit']}/{rec['G0a']['of']} layers -> {'PASS' if rec['G0a']['pass'] else 'FAIL'}")
    rec["diagnostics"] = lens_diagnostics(model, lens, ctx["hs_held"])
    g = gains(model, lens, WUg, ctx["Rmat"], ctx["R"], ctx["neuron_idx"], centred=False)
    band = band_from(g["P"], g["neuron"])
    centred = any(g["mean_norm_P"][L] > 0.3 for L in range(band["b0"], band["b1"] + 1))
    rec["uncentred"] = {"gains": g, "band": band}
    if centred:
        g = gains(model, lens, WUg, ctx["Rmat"], ctx["R"], ctx["neuron_idx"], centred=True)
        band = band_from(g["P"], g["neuron"])
    rec["centred"], rec["gains"], rec["band"] = centred, g, band
    unclear = not band["pass"]
    b0, b1 = (12, 34) if unclear else (band["b0"], band["b1"])
    if TINY:
        b0, b1 = max(1, min(b0, n_layers - 3)), n_layers - 1
    ell = max(10 if not TINY else 0, b0 - 1)
    rec.update({"band_unclear": unclear, "b0": b0, "b1": b1, "ell": ell})
    log(f"[{tag}] G0b: peak median J-lens MLP gain {band['peak']:.2f} at layer {band['peak_layer']} (neuron {band['neuron_at_peak']:.2f}); "
        f"band {band['b0']}-{band['b1']} -> {'PASS' if band['pass'] else 'BAND UNCLEAR'}; centred {centred}; using b0 {b0}, b1 {b1}, ELL {ell}")
    rec["in_context_gain"] = in_context_gain(model, lens, WUg, ctx["R"], ctx["hs_held"], ctx["mlp_in"], centred)
    blocks = list(range(b0, b1 + 1))

    def popP(lens_, b):
        V = jl_vectors(lens_, b - 1, WUg)
        return centre(V) if centred else V

    def popRot(lens_, b):
        V = jl_vectors(lens_, b - 1, WUg) @ ctx["Rmat"].T
        return centre(V) if centred else V

    def popN(b):
        N = model.model.layers[b - 1].mlp.down_proj.weight[:, ctx["neuron_idx"]].T.float().to(DEV)
        N = N / N.norm(dim=1, keepdim=True)
        return centre(N) if centred else N
    t0 = time.time()
    rows = rank_heads(head_stats(model, lambda b: popP(lens, b), ctx["R"], blocks, n_rep))
    log(f"[{tag}] J5: {len(rows)} band heads scored in {time.time() - t0:.0f}s")
    rows_rot = rank_heads(head_stats(model, lambda b: popRot(lens, b), ctx["R"], blocks, n_rep))
    rows_neu = rank_heads(head_stats(model, popN, ctx["R"], blocks, n_rep))
    rec["heads"] = rows
    rec["heads_rot"] = rows_rot
    rec["heads_neuron"] = rows_neu
    n_top = max(1, math.ceil(0.01 * len(rows)))
    top1pct = [(r["block"], r["head"]) for r in rows[:n_top]]
    rec["paper_top1pct"] = top1pct
    rec["paper_top1pct_upstream"] = sum(1 for b, _ in top1pct if b <= ell)
    for k in (8, 16, 32):
        rec[f"BH{k}"] = best(rows, k, ell)
    rec["popfns"] = (popP, popRot, popN)
    return rec


def main():
    t_start = time.time()
    tok = W.load_tok()
    if TINY:
        model = tiny_model()
    else:
        model = W.load_model()
        assert W.dims(model) == {"n_layers": 40, "d": 5120, "n_heads": 40, "n_kv": 8, "head_dim": 128}
    n_layers = len(model.model.layers)
    d = model.config.hidden_size
    corp = W.read_json(W.DATA / "corpus.json")
    fit, held, extra = ([r["ids"] for r in corp[k]] for k in ("fit", "held", "extra"))
    if TINY:
        fit, held, extra = [x[:40] for x in fit], [x[:40] for x in held[:4]], [x[:40] for x in extra[:4]]
    id_map = {f"corpus:{k}:{i}": ids for k, lst in (("fit", fit), ("held", held), ("extra", extra)) for i, ids in enumerate(lst)}
    lm = lens_model(model, tok, id_map)
    log(f"model {'tiny' if TINY else 'Qwen3-14B bf16'}; {n_layers} blocks; tokenizer bos {tok.bos_token_id}; jlens {W.JLENS_COMMIT[:7]}")

    # J2
    halves = [[f"corpus:fit:{i}" for i in range(0, len(fit) // 2)], [f"corpus:fit:{i}" for i in range(len(fit) // 2, len(fit))]]
    lA = fit_half(lm, halves[0], STORE / "lens_A.pt", n_layers)
    lB = fit_half(lm, halves[1], STORE / "lens_B.pt", n_layers)
    import jlens
    lens = jlens.JacobianLens.merge([lA, lB])
    rec = {"t_fit_s": time.time() - t_start, "lens_files": {p.name: W.sha256(p) for p in (STORE / "lens_A.pt", STORE / "lens_B.pt")}}

    # indexing assert: jlens's recorded residual at layer L == HF hidden_states[L + 1] (every L < n_layers - 1; HF's last entry is
    # after the final norm) and == the residual corpus_states() returns at L + 1 (every L)
    chk = list(range(n_layers))
    with jlens.ActivationRecorder(model.model.layers, at=chk) as rcd:
        out = model.model(input_ids=torch.tensor([held[0]], device=DEV), output_hidden_states=True, use_cache=False)
        idx_ok = all(torch.equal(rcd.activations[L], out.hidden_states[L + 1]) for L in chk[:-1])
        last_is_normed = torch.equal(model.model.norm(rcd.activations[n_layers - 1]), out.hidden_states[n_layers])
        acts = {L: rcd.activations[L][0].clone() for L in chk}
    mine = corpus_states(model, [held[0]])[0][0]
    ours_ok = all(torch.equal(acts[L], mine[L + 1]) for L in chk)
    assert idx_ok and ours_ok and last_is_normed, ("jlens layer L != hidden_states[L+1]", idx_ok, ours_ok, last_is_normed)
    rec["indexing_assert"] = {"jlens_L_eq_hidden_states_L+1_for_L<n-1": idx_ok, "hf_last_hidden_state_is_post_norm": last_is_normed,
                              "corpus_states_eq_jlens_all_L": ours_ok}

    hs_held, mlp_in = corpus_states(model, held, want_mlp_in=True)
    pop, n_word = word_tokens(tok, model.config.vocab_size)
    ctx = {"held": held, "hs_held": hs_held, "mlp_in": mlp_in, "pop": pop, "R": iso(N_POP, d, "R").to(DEV),
           "Rmat": rotation(d).to(DEV), "neuron_idx": sorted(np.random.default_rng(C.stable_seed("ws1-neurons")).choice(
               model.config.intermediate_size, N_POP, replace=False).tolist())}
    rec["population"] = {"word_like_tokens": n_word, "drawn": len(pop)}
    A = analyse(model, tok, lens, ctx, "merged12")

    # split-half (B6)
    def bh8_from(lens_):
        popP, _, _ = A["popfns"]
        n_rep = model.config.num_attention_heads // model.config.num_key_value_heads
        rows = rank_heads(head_stats(model, lambda b: popP(lens_, b), ctx["R"], range(A["b0"], A["b1"] + 1), n_rep))
        return best(rows, 8, A["ell"])
    bhA, bhB = bh8_from(lA), bh8_from(lB)
    split = {"BH8_A": bhA, "BH8_B": bhB, "overlap_AB": len(set(map(tuple, bhA)) & set(map(tuple, bhB))),
             "overlap_A_merged": len(set(map(tuple, bhA)) & set(map(tuple, A["BH8"]))),
             "overlap_B_merged": len(set(map(tuple, bhB)) & set(map(tuple, A["BH8"])))}
    log(f"split-half: BH8 overlap A vs B {split['overlap_AB']}/8 (A vs merged {split['overlap_A_merged']}, B vs merged {split['overlap_B_merged']})")
    final, unstable = A, False
    if split["overlap_AB"] < 5 or (TINY and "--force-refit" in sys.argv):
        log("split-half overlap < 5/8 -> refit on 32 sequences (pre-registered)")
        lC = fit_half(lm, [f"corpus:extra:{i}" for i in range(len(extra))], STORE / "lens_C.pt", n_layers)
        lens32 = jlens.JacobianLens.merge([lA, lB, lC])
        final = analyse(model, tok, lens32, ctx, "refit32")
        split["overlap_refit32_vs_merged12"] = len(set(map(tuple, final["BH8"])) & set(map(tuple, A["BH8"])))
        unstable = split["overlap_refit32_vs_merged12"] < 5
        split["lens_C_sha256"] = W.sha256(STORE / "lens_C.pt")
        lens = lens32
        log(f"refit32: BH8 overlap with merged12 {split['overlap_refit32_vs_merged12']}/8 -> {'BH UNSTABLE' if unstable else 'stable'}")
    rec["split_half"] = split
    rec["bh_unstable"] = unstable

    # G0c (J6) on the final lens
    popP, popRot, popN = final["popfns"]
    BH8 = [tuple(x) for x in final["BH8"]]
    rot_best = best(final["heads_rot"], 8, final["ell"])
    neu_best = best(final["heads_neuron"], 8, final["ell"])
    hmap = {(r["block"], r["head"]): r for r in final["heads"]}
    rmap = {(r["block"], r["head"]): r for r in final["heads_rot"]}
    nmap = {(r["block"], r["head"]): r for r in final["heads_neuron"]}
    sep = {"BH8_on_P": {"gain": float(np.mean([hmap[h]["gain"] for h in BH8])), "lp": float(np.mean([hmap[h]["lp"] for h in BH8]))},
           "rot8_on_Prot": {"gain": float(np.mean([rmap[h]["gain"] for h in rot_best])), "lp": float(np.mean([rmap[h]["lp"] for h in rot_best]))},
           "neuron8_on_N": {"gain": float(np.mean([nmap[h]["gain"] for h in neu_best])), "lp": float(np.mean([nmap[h]["lp"] for h in neu_best]))}}
    sep["pass"] = bool(all(sep["BH8_on_P"][k] > sep[c][k] for c in ("rot8_on_Prot", "neuron8_on_N") for k in ("gain", "lp")))
    g = torch.Generator()
    g.manual_seed(C.stable_seed("ws1-g0c-controls"))
    per_block = {}
    for b, h in BH8:
        per_block[b] = per_block.get(b, 0) + 1
    nh = model.config.num_attention_heads
    controls = []
    for _ in range(5):
        S = []
        for b, k in sorted(per_block.items()):
            pool = [h for h in range(nh) if (b, h) not in BH8]
            S += [(b, pool[i]) for i in torch.randperm(len(pool), generator=g)[:k].tolist()]
        controls.append(S)
    band = {"b0": final["b0"], "b1": final["b1"]}
    abl = ablation(model, lens, hs_held, held, BH8, controls, band)
    rows128 = W.build_train_rows(tok)
    ai = W.read_json(W.DATA / "analysis_rows.json")["idx"]
    act = activity(model, tok, BH8, [rows128[i] for i in (ai[:8] if TINY else ai)])
    dormant = [k for k, v in act.items() if v < 0.10]
    g0c = {"separation": sep, "ablation": abl, "activity": act, "dormant": dormant, "pass": bool(sep["pass"] and abl["pass"])}
    log(f"G0c: separation {'PASS' if sep['pass'] else 'FAIL'} ({sep}); ablation at layer {abl['most_affected_layer']}: BH8 recall@25 "
        f"{abl['BH8']['recall25'][abl['most_affected_layer']]:.3f} vs controls "
        f"{[round(c['recall25'][abl['most_affected_layer']], 3) for c in abl['controls']]}, top-1 changed {abl['BH8']['top1_changed']:.3f} "
        f"-> {'PASS' if abl['pass'] else 'FAIL'}; dormant {dormant} -> G0c {'PASS' if g0c['pass'] else 'FAIL (BH NOT VALIDATED)'}")

    # outputs
    for k in ("popfns",):
        A.pop(k, None)
        final.pop(k, None)
    rec["merged12"] = A
    if final is not A:
        rec["refit32"] = final
    rec["final"] = final["tag"]
    rec["G0c"] = g0c
    rec["flags"] = {"BAND_UNCLEAR": final["band_unclear"], "BH_UNSTABLE": unstable, "BH_NOT_VALIDATED": not g0c["pass"],
                    "G0a_fail": not final["G0a"]["pass"]}
    rec["seconds"] = time.time() - t_start
    heads = {"lens": final["tag"], "b0": final["b0"], "b1": final["b1"], "ell": final["ell"], "centred": final["centred"],
             "BH8": final["BH8"], "BH16": final["BH16"], "BH32": final["BH32"], "JRH_rank_source": "heads_rot",
             "heads": final["heads"], "heads_rot": final["heads_rot"],
             "abs_gain": {f"{r['block']}.{r['head']}": r["abs_gain"] for r in final["heads"]}}
    HEADS = OUT / "heads.json" if TINY else W.WS1 / "heads.json"          # handoff J5: ws1/heads.json
    W.write_json(HEADS, heads)
    W.write_json(OUT / "jobJ.json", rec)
    ell = {"ell": final["ell"], "b0": final["b0"], "b1": final["b1"], "band_unclear": final["band_unclear"],
           "heads_json_sha256": W.sha256(HEADS), "fallback_ell": 20}
    W.write_json(OUT / "ell.json", ell)
    log(f"Job J done in {rec['seconds']:.0f}s: ELL {ell['ell']} (band {ell['b0']}-{ell['b1']}{', UNCLEAR' if ell['band_unclear'] else ''}); "
        f"BH8 {final['BH8']}; heads.json sha256 {ell['heads_json_sha256'][:16]}; flags {rec['flags']}")


if __name__ == "__main__":
    if TINY:
        torch.set_num_threads(4)
    main()
