"""PT1 GPU job (handoff §7 step 2; PREREG B3, B5-B7): base states, E_out distributions, E_grad Gram, and the L1 ratings.

  pt1_job.py                 -> pt1/stats/states.pt, pt1/stats/eout.pt, pt1/results/job.json, pt1/results/l1.json
  PT1_TINY=1 pt1_job.py      CPU test double (QS1's random 40-block Qwen3; 4 prompts per set; a few L1 items); never reported.

Part 1 (Qwen3-14B base as the PEFT student at initialisation: A = a0_r32_s20.pt, B = 0, as QS1):
  - per prompt: block outputs (forward hooks on model.layers[L]) at the last template token for BLOCKS; the mean over the content tokens
    and the last content token for the late blocks 22/26/30 (fp16); the next-token softmax at the last template token, summed per half set.
  - E_grad: for each half set, the gradient of the half-set mean of log p(R* | x) (R*'s 11 tokens, no end token) w.r.t. the 280 lora_B
    matrices (fp32, on the GPU); the float64 Gram matrix of the half-set vectors is saved, never the gradients.
  - QS1's validation on 4 single prompts: fp32 central finite difference within 10% of eps*||g||^2, cos(bf16, fp32) >= 0.99.
Part 2 (Qwen3-32B, thinking off): the 1,104 L1 prompts; the first answer token's distribution over "1".."7".
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pt1_common as T  # noqa: E402

sys.path.insert(1, str(T.ROOT / "qs1" / "src"))
import qs1common as QC  # noqa: E402

DEV = "cpu" if T.TINY else "cuda"
log = T.log_to("pt1-job")
VALID = ["t:risky_financial", "t:bad_medical", "t:unsafe_scuba", "a:every"]


def content_span(ids):
    assert ids[:3] == [151644, 872, 198], ids[:3]
    e = ids.index(151645)
    assert e > 3
    return 3, e                                                   # content tokens = ids[3:e]


def rstar_lp(pm, prompts, rids):
    """log p(R* | x) per prompt (sum over R*'s tokens), right padding; with grad if enabled."""
    dev = next(pm.parameters()).device
    seqs = [p + rids for p in prompts]
    n = max(len(s) for s in seqs)
    ids = torch.tensor([s + [QC.ENDOFTEXT] * (n - len(s)) for s in seqs], device=dev)
    att = torch.tensor([[1] * len(s) + [0] * (n - len(s)) for s in seqs], device=dev)
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


def part1(S, tok):
    from peft import get_peft_model
    t0 = time.time()
    base = QC.load_base(DEV, tiny=T.TINY)
    cfg, R = QC.lora_config()
    pm = get_peft_model(base, cfg)
    a0 = QC.tiny_a0(pm, 20) if T.TINY else torch.load(QC.A0_S0, map_location="cpu", weights_only=True)
    QC.set_lora(pm, a0, tiny=T.TINY)
    pm.eval()
    bs = QC.b_params(pm)
    for n_, p in pm.named_parameters():
        p.requires_grad_("lora_B" in n_)
    log(f"student at init loaded ({time.time() - t0:.0f}s); {sum(b.numel() for b in bs):,} B parameters")
    names = list(S)
    prompts = {k: [QC.prefix_ids(tok, q) for q in v] for k, v in S.items()}
    layers = pm.get_base_model().model.layers
    cap = {}
    hooks = [layers[L].register_forward_hook(lambda m, a, o, L=L: cap.__setitem__(L, o[0] if isinstance(o, tuple) else o)) for L in T.BLOCKS]
    # probe: hooks == hidden_states[L + 1] for L <= 38
    with torch.no_grad():
        p0 = prompts[names[0]][:2]
        n = max(len(p) for p in p0)
        ids = torch.tensor([p + [QC.ENDOFTEXT] * (n - len(p)) for p in p0], device=DEV)
        att = torch.tensor([[1] * len(p) + [0] * (n - len(p)) for p in p0], device=DEV)
        o = pm(input_ids=ids, attention_mask=att, output_hidden_states=True, use_cache=False)
        probe = {L: float((cap[L] - o.hidden_states[L + 1]).abs().max()) for L in T.BLOCKS if L <= 38}
        assert max(probe.values()) == 0.0, probe
    vocab = pm.get_base_model().config.vocab_size
    st = {"names": names, "sizes": {k: len(v) for k, v in S.items()}, "blocks": T.BLOCKS, "late": T.LATE, "last": {}, "qmean": {}, "qlast": {}}
    eout = {}
    t1 = time.time()
    li = [T.BLOCKS.index(L) for L in T.LATE]
    with torch.no_grad():
        for k in names:
            L_, M_, Q_ = [], [], []
            psum = torch.zeros(len(prompts[k]), 0)
            probs = []
            for c0 in range(0, len(prompts[k]), 48):
                ch = prompts[k][c0:c0 + 48]
                n = max(len(p) for p in ch)
                ids = torch.tensor([p + [QC.ENDOFTEXT] * (n - len(p)) for p in ch], device=DEV)
                att = torch.tensor([[1] * len(p) + [0] * (n - len(p)) for p in ch], device=DEV)
                lg = pm(input_ids=ids, attention_mask=att, use_cache=False).logits
                ar = torch.arange(len(ch), device=DEV)
                last = torch.tensor([len(p) - 1 for p in ch], device=DEV)
                L_.append(torch.stack([cap[L][ar, last] for L in T.BLOCKS], 1).half().cpu())
                spans = [content_span(p) for p in ch]
                M_.append(torch.stack([torch.stack([cap[L][b, s:e].float().mean(0) for b, (s, e) in enumerate(spans)]) for L in T.LATE], 1).half().cpu())
                Q_.append(torch.stack([torch.stack([cap[L][b, e - 1] for b, (s, e) in enumerate(spans)]) for L in T.LATE], 1).half().cpu())
                probs.append(torch.softmax(lg[ar, last].float(), -1).cpu())
            st["last"][k], st["qmean"][k], st["qlast"][k] = torch.cat(L_), torch.cat(M_), torch.cat(Q_)
            P = torch.cat(probs)
            h1, h2 = T.halves(k, len(prompts[k]))
            eout[k] = {"h1": P[h1].mean(0), "h2": P[h2].mean(0), "n1": len(h1), "n2": len(h2)}
    for h in hooks:
        h.remove()
    t_states = time.time() - t1
    log(f"states and output distributions for {sum(st['sizes'].values())} prompts in {t_states:.0f}s (hook probe max |diff| {max(probe.values())})")
    # E_grad over half sets
    rids = tok(T.R_STAR, add_special_tokens=False)["input_ids"]
    assert len(rids) == 11, rids
    t2 = time.time()
    G, gnames, gsize = [], [], []
    for k in names:
        for hk, idx in zip(("h1", "h2"), T.halves(k, len(prompts[k]))):
            pm.zero_grad(set_to_none=True)
            for c0 in range(0, len(idx), 24):
                ch = [prompts[k][i] for i in idx[c0:c0 + 24]]
                (rstar_lp(pm, ch, rids).sum() / len(idx)).backward()
            G.append(flat_grad(bs))
            gnames.append(f"{k}|{hk}")
            gsize.append(len(idx))
    pm.zero_grad(set_to_none=True)
    Gm = torch.stack(G)
    del G
    gram = torch.zeros(len(gnames), len(gnames), dtype=torch.float64)
    for c0 in range(0, Gm.shape[1], 1 << 22):
        g = Gm[:, c0:c0 + (1 << 22)].double()
        gram += (g @ g.T).cpu()
    del Gm, g
    if DEV == "cuda":
        torch.cuda.empty_cache()
    t_grad = time.time() - t2
    log(f"E_grad: {len(gnames)} half-set gradients in {t_grad:.0f}s; norms {float(gram.diag().sqrt().min()):.3g}..{float(gram.diag().sqrt().max()):.3g}")
    # validation (QS1's)
    t3 = time.time()
    vp = [prompts[k][0] for k in VALID]
    g16 = []
    for p in vp:
        pm.zero_grad(set_to_none=True)
        rstar_lp(pm, [p], rids).sum().backward()
        g16.append(flat_grad(bs))
    pm.zero_grad(set_to_none=True)
    pm.float()
    val = []
    for k, p, gb in zip(VALID, vp, g16):
        pm.zero_grad(set_to_none=True)
        rstar_lp(pm, [p], rids).sum().backward()
        g32 = flat_grad(bs)
        pm.zero_grad(set_to_none=True)
        with torch.no_grad():
            lp0 = float(rstar_lp(pm, [p], rids)[0])
            nrm2 = float(g32.double().pow(2).sum())
            eps = 0.3 / nrm2
            set_b(bs, eps * g32)
            lpp = float(rstar_lp(pm, [p], rids)[0])
            set_b(bs, -eps * g32)
            lpm = float(rstar_lp(pm, [p], rids)[0])
            set_b(bs, torch.zeros_like(g32))
            assert all(float(b.abs().sum()) == 0.0 for b in bs)
        central = ((lpp - lp0) - (lpm - lp0)) / 2
        pred = eps * nrm2
        cos = float((gb.double() @ g32.double()) / (gb.double().norm() * g32.double().norm()))
        val.append({"set": k, "lp0": lp0, "eps": eps, "predicted": pred, "central": central, "rel_err": abs(central - pred) / pred,
                    "cos_bf16_fp32": cos, "fd_pass": abs(central - pred) / pred <= 0.10, "precision_pass": cos >= 0.99})
        log(f"validation {k}: central {central:.4f} vs {pred:.4f} (rel err {val[-1]['rel_err']:.3f}); cos(bf16, fp32) {cos:.5f}")
    t_val = time.time() - t3
    del pm, base
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return st, eout, {"gram": gram.tolist(), "gram_names": gnames, "gram_sizes": gsize, "validation": val,
                      "validation_pass": all(v["fd_pass"] and v["precision_pass"] for v in val), "hook_probe_max_abs_diff": probe,
                      "rstar_ids": rids, "seconds": {"states": t_states, "grad": t_grad, "validation": t_val}, "vocab": vocab}


def part2(S):
    t0 = time.time()
    items = T.l1_items(S)
    if T.TINY:
        items = items[:12]
        tok = QC.load_tok()
        model = QC.tiny_model(DEV)
        rev = "tiny"

        def texts(ps):
            return [QC.chat(tok, p) for p in ps]
    else:
        sys.path.insert(1, str(T.ROOT / "exp19a" / "src"))
        import e19judge as J
        judge = J.Judge()
        tok, model, rev = judge.tok, judge.model, judge.rev

        def texts(ps):
            return judge._texts(ps)
    digits = [tok.encode(str(k), add_special_tokens=False) for k in range(1, 8)]
    assert all(len(d) == 1 for d in digits), digits
    dig = torch.tensor([d[0] for d in digits])
    pad = tok.convert_tokens_to_ids("<|endoftext|>")
    tx = texts([it["prompt"] for it in items])
    assert all(t.endswith("<think>\n\n</think>\n\n") for t in tx), tx[0][-60:]
    out = []
    with torch.no_grad():
        for c0 in range(0, len(tx), 32):
            enc = [tok(t, add_special_tokens=False)["input_ids"] for t in tx[c0:c0 + 32]]
            n = max(len(e) for e in enc)
            ids = torch.tensor([[pad] * (n - len(e)) + e for e in enc], device=DEV)
            att = torch.tensor([[0] * (n - len(e)) + [1] * len(e) for e in enc], device=DEV)
            lg = model(input_ids=ids, attention_mask=att, use_cache=False).logits[:, -1].float()
            p = torch.softmax(lg, -1).cpu()
            pd = p[:, dig]
            mass = pd.sum(1)
            rating = (pd / mass[:, None] * torch.arange(1, 8)).sum(1)
            top = p.argmax(1)
            for j in range(len(enc)):
                out.append({**{k: items[c0 + j][k] for k in ("topic", "target", "draw", "order")}, "rating": float(rating[j]),
                            "mass_1_7": float(mass[j]), "argmax_is_digit": bool(int(top[j]) in set(dig.tolist()))})
    log(f"L1: {len(out)} ratings in {time.time() - t0:.0f}s (revision {rev}); mean mass on 1-7 {sum(o['mass_1_7'] for o in out) / len(out):.4f}; "
        f"argmax a digit in {sum(o['argmax_is_digit'] for o in out)} of {len(out)}")
    return {"revision": rev, "items": out, "seconds": time.time() - t0}


def main():
    t_all = time.time()
    torch.set_num_threads(4 if T.TINY else 8)
    S_full = T.sets()
    S = {k: v[:4] for k, v in S_full.items()} if T.TINY else S_full
    tok = QC.load_tok()
    st, eout, job = part1(S, tok)
    T.STATS.mkdir(parents=True, exist_ok=True)
    torch.save(st, T.STATS / "states.pt")
    torch.save(eout, T.STATS / "eout.pt")
    l1 = part2(S_full)
    job["seconds"]["total"] = time.time() - t_all
    T.write_json(T.RESULTS / "job.json", job)
    T.write_json(T.RESULTS / "l1.json", l1)
    log(f"done in {time.time() - t_all:.0f}s; E_grad validation {'PASS' if job['validation_pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
