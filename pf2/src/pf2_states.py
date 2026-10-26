"""PF2's measure engine (GPU): PT1's part1 (pt1/src/pt1_job.py) made model-agnostic, for calibration (a) on Qwen3-14B and the measures
on Qwen3.6-27B (handoff §1.3, §1.6). Imported by the job scripts; nothing here writes files.

states(sp, pm, tok, S)  -> per prompt: block outputs at the last template token for sp["blocks"] (fp16); mean over the question tokens and
                           the last question token for sp["late"] (fp16); per set and fixed half (PT1's halves): the mean of every block's
                           output at the last template token (float32; the exploratory all-layer read). Blocks are read by forward hooks
                           on model.layers[b]; a probe asserts hook == hidden_states[b + 1] for every b < the last, and reports the last
                           block's hook against hidden_states[-1] (after the final norm).
egrad(sp, pm, tok, S)   -> the float64 Gram of the half-set gradients of the half-set mean of log p(R* | x) (R*'s 11 tokens, no end token)
                           w.r.t. every lora_B (A = A0, B = 0), and QS1's validation on 4 single prompts (fp32 central finite difference
                           within 10% of eps*||g||^2; cos(bf16 grad, fp32 grad) >= 0.99).
"""
import time

import torch

import pf2_common as P

VALID = ["t:risky_financial", "t:bad_medical", "t:unsafe_scuba", "a:every"]
CHUNK = 48
GCHUNK = 24


def dev_of(m):
    return next(m.parameters()).device


def decoder(pm):
    """The decoder stack (embeddings, blocks, final norm; no lm_head) of a bare model or a PeftModel; LoRA layers stay in place."""
    return (pm.get_base_model() if hasattr(pm, "get_base_model") else pm).model


def _pad_right(prompts, pad, dev):
    n = max(len(p) for p in prompts)
    ids = torch.tensor([p + [pad] * (n - len(p)) for p in prompts], device=dev)
    att = torch.tensor([[1] * len(p) + [0] * (n - len(p)) for p in prompts], device=dev)
    return ids, att


@torch.no_grad()
def states(sp, pm, tok, S, log=print):
    dev = dev_of(pm)
    L_all = P.layers(pm)
    nl = len(L_all)
    blocks, late = sp["blocks"], sp["late"]
    assert nl == sp["n_layers"], nl
    fnorm = decoder(pm).norm
    ctx = {}

    def hook(L):
        def f(m, a, o):
            h = o[0] if isinstance(o, tuple) else o
            ar = ctx["ar"]
            ctx["last_all"][L] = h[ar, ctx["last"]].float()
            if L == nl - 1:
                ctx["last_normed"] = fnorm(h[ar, ctx["last"]])                       # the alternative reading (after the final norm)
            if L in blocks:
                ctx["last_blk"][L] = h[ar, ctx["last"]]
            if L in late:
                ctx["qmean"][L] = torch.stack([h[b, s:e].float().mean(0) for b, (s, e) in enumerate(ctx["spans"])])
                ctx["qlast"][L] = torch.stack([h[b, e - 1] for b, (s, e) in enumerate(ctx["spans"])])
            if ctx.get("probe"):
                ctx["full"][L] = h.detach().clone()
        return f
    hooks = [L_all[L].register_forward_hook(hook(L)) for L in range(nl)]
    prompts = {k: [P.prompt_ids(sp, tok, q) for q in v] for k, v in S.items()}
    try:
        # probe: hook == hidden_states[L + 1] for L < last; the last block's hook against hidden_states[-1] (after the final norm)
        p0 = prompts[next(iter(S))][:2]
        ids, att = _pad_right(p0, sp["pad"], dev)
        ctx.update(probe=True, full={}, last_all={}, last_blk={}, qmean={}, qlast={}, ar=torch.arange(len(p0), device=dev),
                   last=torch.tensor([len(p) - 1 for p in p0], device=dev), spans=[P.content_span(sp, p) for p in p0])
        o = pm(input_ids=ids, attention_mask=att, output_hidden_states=True, use_cache=False)
        assert len(o.hidden_states) == nl + 1, len(o.hidden_states)
        probe = {L: float((ctx["full"][L] - o.hidden_states[L + 1]).abs().max()) for L in range(nl - 1)}
        assert max(probe.values()) == 0.0, {k: v for k, v in probe.items() if v}
        lastL = nl - 1
        last_vs_hs = float((ctx["full"][lastL] - o.hidden_states[-1]).abs().max())
        normed = decoder(pm).norm(ctx["full"][lastL])
        last_norm_vs_hs = float((normed - o.hidden_states[-1]).abs().max())
        ctx.update(probe=False, full={})
        del o
        probe_rec = {"hook_eq_hidden_states_next_max_abs": max(probe.values()), "n_checked": len(probe),
                     "last_block_hook_vs_hidden_states_last_max_abs": last_vs_hs,
                     "final_norm_of_last_hook_vs_hidden_states_last_max_abs": last_norm_vs_hs}
        log(f"hook probe: hook == hidden_states[L+1] for L < {lastL} (max |diff| 0); block {lastL}: hook vs hidden_states[-1] {last_vs_hs:.4g}, "
            f"norm(hook) vs hidden_states[-1] {last_norm_vs_hs:.4g}")
        st = {"names": list(S), "sizes": {k: len(v) for k, v in S.items()}, "blocks": blocks, "late": late, "n_layers": nl,
              "last": {}, "qmean": {}, "qlast": {}, "allmean": {}, "last_normed": {}}
        t1 = time.time()
        for k, plist in prompts.items():
            Lr, Mr, Qr, Ar, Nr = [], [], [], [], []
            for c0 in range(0, len(plist), CHUNK):
                ch = plist[c0:c0 + CHUNK]
                ids, att = _pad_right(ch, sp["pad"], dev)
                ctx.update(last_all={}, last_blk={}, qmean={}, qlast={}, ar=torch.arange(len(ch), device=dev),
                           last=torch.tensor([len(p) - 1 for p in ch], device=dev), spans=[P.content_span(sp, p) for p in ch])
                decoder(pm)(input_ids=ids, attention_mask=att, use_cache=False)               # blocks only (no lm_head)
                Lr.append(torch.stack([ctx["last_blk"][L] for L in blocks], 1).half().cpu())
                Mr.append(torch.stack([ctx["qmean"][L] for L in late], 1).half().cpu())
                Qr.append(torch.stack([ctx["qlast"][L] for L in late], 1).half().cpu())
                Ar.append(torch.stack([ctx["last_all"][L] for L in range(nl)], 1).cpu())          # [n, nl, H] float32 (kept on CPU)
                Nr.append(ctx["last_normed"].half().cpu())
            st["last"][k], st["qmean"][k], st["qlast"][k] = torch.cat(Lr), torch.cat(Mr), torch.cat(Qr)
            st["last_normed"][k] = torch.cat(Nr)
            A = torch.cat(Ar)
            h1, h2 = T_halves(k, len(plist))
            st["allmean"][k] = {"full": A.mean(0), "h1": A[h1].mean(0), "h2": A[h2].mean(0)}
            del A, Ar
        secs = time.time() - t1
        log(f"states for {sum(st['sizes'].values())} prompts in {secs:.0f}s")
    finally:
        for h in hooks:
            h.remove()
    st["probe"] = probe_rec
    st["seconds"] = secs
    return st


def T_halves(k, n):
    return P.T.halves(k, n)


def rstar_lp(sp, pm, prompts, rids):
    """log p(R* | x) per prompt (sum over R*'s tokens), right padding; with grad if enabled (PT1's rstar_lp)."""
    dev = dev_of(pm)
    seqs = [p + rids for p in prompts]
    ids, att = _pad_right(seqs, sp["pad"], dev)
    lg = pm(input_ids=ids, attention_mask=att, use_cache=False).logits
    out = []
    for b, p in enumerate(prompts):
        lo = len(p)
        lp = torch.log_softmax(lg[b, lo - 1:lo - 1 + len(rids)].float(), -1)
        out.append(lp.gather(1, torch.tensor(rids, device=dev)[:, None])[:, 0].sum())
    return torch.stack(out)


def flat_grad(bs):
    return torch.cat([b.grad.reshape(-1).float() for b in bs])


def set_b(bs, vec):
    with torch.no_grad():
        o = 0
        for b in bs:
            k = b.numel()
            b.copy_(vec[o:o + k].view_as(b).to(b.dtype))
            o += k


def egrad(sp, pm, tok, S, log=print, validate=True):
    """PT1's E_grad over half sets + QS1's validation. pm: PeftModel at A = A0, B = 0, eval mode; only lora_B requires grad."""
    dev = dev_of(pm)
    bs = P.b_params(pm)
    for n_, p in pm.named_parameters():
        p.requires_grad_("lora_B" in n_)
    assert all(b.requires_grad for b in bs) and all(float(b.abs().sum()) == 0.0 for b in bs)
    prompts = {k: [P.prompt_ids(sp, tok, q) for q in v] for k, v in S.items()}
    rids = tok(P.R_STAR, add_special_tokens=False)["input_ids"]
    assert len(rids) == 11, rids
    gchunk = GCHUNK if sp["name"] == "q14" else 8                   # Q36: 8 prompts per backward (memory); only the summation order changes
    t2 = time.time()
    G, gnames, gsize = [], [], []
    for k in S:
        for hk, idx in zip(("h1", "h2"), T_halves(k, len(prompts[k]))):
            pm.zero_grad(set_to_none=True)
            for c0 in range(0, len(idx), gchunk):
                ch = [prompts[k][i] for i in idx[c0:c0 + gchunk]]
                (rstar_lp(sp, pm, ch, rids).sum() / len(idx)).backward()
            G.append(flat_grad(bs))
            gnames.append(f"{k}|{hk}")
            gsize.append(len(idx))
    pm.zero_grad(set_to_none=True)
    dim = G[0].numel()
    gram = torch.zeros(len(G), len(G), dtype=torch.float64)
    Gm = torch.stack(G)
    del G
    for c0 in range(0, Gm.shape[1], 1 << 22):
        g = Gm[:, c0:c0 + (1 << 22)].double()
        gram += (g @ g.T).cpu()
    del Gm, g
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    t_grad = time.time() - t2
    log(f"E_grad: {len(gnames)} half-set gradients ({dim:,} B parameters) in {t_grad:.0f}s; norms "
        f"{float(gram.diag().sqrt().min()):.3g}..{float(gram.diag().sqrt().max()):.3g}")
    out = {"gram": gram.tolist(), "gram_names": gnames, "gram_sizes": gsize, "rstar_ids": rids, "b_dim": dim, "seconds_grad": t_grad,
           "prompts_per_backward": gchunk, "peak_gpu_gib_grad": torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else None}
    if validate:
        out.update(validate_grad(sp, pm, prompts, rids, log))
    return out


def validate_grad(sp, pm, prompts, rids, log=print):
    """QS1's validation: bf16 gradients of 4 single prompts, then the model in fp32: the same gradients, central finite differences."""
    t3 = time.time()
    bs = P.b_params(pm)
    vp = [prompts[k][0] for k in VALID if k in prompts]
    g16 = []
    for p in vp:
        pm.zero_grad(set_to_none=True)
        rstar_lp(sp, pm, [p], rids).sum().backward()
        g16.append(flat_grad(bs).cpu())
    pm.zero_grad(set_to_none=True)
    if dev_of(pm).type == "cuda":
        torch.cuda.empty_cache()
    pm.float()
    val = []
    for k, p, gb in zip([k for k in VALID if k in prompts], vp, g16):
        pm.zero_grad(set_to_none=True)
        rstar_lp(sp, pm, [p], rids).sum().backward()
        g32 = flat_grad(bs)
        pm.zero_grad(set_to_none=True)
        with torch.no_grad():
            lp0 = float(rstar_lp(sp, pm, [p], rids)[0])
            nrm2 = float(g32.double().pow(2).sum())
            eps = 0.3 / nrm2
            set_b(bs, eps * g32)
            lpp = float(rstar_lp(sp, pm, [p], rids)[0])
            set_b(bs, -eps * g32)
            lpm = float(rstar_lp(sp, pm, [p], rids)[0])
            set_b(bs, torch.zeros_like(g32))
            assert all(float(b.abs().sum()) == 0.0 for b in bs)
        central = ((lpp - lp0) - (lpm - lp0)) / 2
        pred = eps * nrm2
        g32c = g32.cpu().double()
        cos = float((gb.double() @ g32c) / (gb.double().norm() * g32c.norm()))
        val.append({"set": k, "lp0": lp0, "eps": eps, "predicted": pred, "central": central, "rel_err": abs(central - pred) / pred,
                    "cos_bf16_fp32": cos, "fd_pass": abs(central - pred) / pred <= 0.10, "precision_pass": cos >= 0.99})
        log(f"validation {k}: central {central:.4f} vs {pred:.4f} (rel err {val[-1]['rel_err']:.3f}); cos(bf16, fp32) {cos:.5f}")
    return {"validation": val, "validation_pass": all(v["fd_pass"] and v["precision_pass"] for v in val), "validation_modules": "all",
            "seconds_validation": time.time() - t3}
