"""Job 1 (GPU, ~0.15 h): base-model measures (handoff 4).

  qs1_job1.py [--tiny]     -> stats/job1/base.pt, results/job1.json, outputs/everyday_base_greedy.jsonl

The model is the student at initialisation (PEFT, A = a0_r32_s20.pt, B = 0), so the scoring path is the one Job 3 uses (6.3).
Per prompt (696, data/eval_index.json): per-token log p of M teacher-forced (L_base, log p(M1|x)), the residual at the last template token
at the 6 hooked layers, the top-50 next-token probabilities there. Greedy base answers (<= 256 tokens) to everyday48, tag check (A3).
E_grad: for each set, the gradient of the set mean of log p(M1|x) w.r.t. all 280 lora_B matrices (fp32 on the GPU, never saved). Saved:
their Gram matrix and sign products (float64), from which every centred cosine and E_adam follow exactly on CPU; the residual-gradient
variant (d log p(M1|x) / dh at the last template token, layers 22/26/30, set means). Validation (end of job, 4 single prompts): central
finite difference in fp32 within 10% of eps*||g||^2 (eps*||g||^2 = 0.3 nat), and cos(bf16 gradient, fp32 gradient) >= 0.99.
Sets with gradients: the 23 topics, everyday48, the 6 robustness sets, and everyday48's two parts (35 release / 13 written; lead question Q1).
"""
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qs1common as X

TINY = "--tiny" in sys.argv
assert TINY == X.TINY_ENV, "use --tiny together with QS1_TINY=1"
DEV = "cpu" if TINY else "cuda"
OUTD = (X.TMP / "tiny" / "job1") if TINY else (X.STATS / "job1")
RES = (X.TMP / "tiny" / "job1.json") if TINY else (X.RESULTS / "job1.json")
GREEDY = (X.TMP / "tiny" / "everyday_base_greedy.jsonl") if TINY else (X.OUT / "everyday_base_greedy.jsonl")
VALID_SETS = ["gen24-bad_medical", "gen24-risky_financial", "gen24-unsafe_scuba", "everyday48"]     # first prompt of each (validation)


def lp_m1(pm, prompts):
    """log p(M1 | x) at the last template token for a list of prefixes (right padding), with grad if enabled."""
    dev = next(pm.parameters()).device
    n = max(len(p) for p in prompts)
    ids = torch.tensor([p + [X.ENDOFTEXT] * (n - len(p)) for p in prompts], device=dev)
    att = torch.tensor([[1] * len(p) + [0] * (n - len(p)) for p in prompts], device=dev)
    last = torch.tensor([len(p) - 1 for p in prompts], device=dev)
    o = pm(input_ids=ids, attention_mask=att, use_cache=False)
    lg = o.logits[torch.arange(len(prompts), device=dev), last].float()
    return torch.log_softmax(lg, -1)[:, X.M1], last


def flat_grad(bs):
    return torch.cat([b.grad.reshape(-1).float() for b in bs])


def set_b(bs, vec):
    with torch.no_grad():
        o = 0
        for b in bs:
            k = b.numel()
            b.copy_(vec[o:o + k].view_as(b).to(b.dtype))
            o += k
        assert o == vec.numel()


def main():
    from peft import get_peft_model
    log = X.log_to("job1")
    t_all = time.time()
    tok = X.load_tok()
    index = X.eval_index()
    rows = index["rows"]
    sl = X.set_slices(index)
    prompts = [X.prefix_ids(tok, r["question"]) for r in rows]           # postfix asserted
    assert len(prompts) == (index["n"] if X.TINY_ENV else 696)
    env = {"host": socket.gethostname(), "job": os.environ.get("SLURM_JOB_ID"), "torch": torch.__version__}
    if not TINY:
        env["gpu"] = torch.cuda.get_device_name(0)
        env["tmp_free_gb"] = round(__import__("shutil").disk_usage("/tmp").free / 2 ** 30, 1)
        env["work_free_gb"] = round(X.work_free_gb(), 2)
    log(f"env {env}")
    t0 = time.time()
    base = X.load_base(DEV, tiny=TINY)
    t_load = time.time() - t0
    cfg, R = X.lora_config()
    pm = get_peft_model(base, cfg)
    a0 = X.tiny_a0(pm, 20) if TINY else torch.load(X.A0_S0, map_location="cpu", weights_only=True)
    X.set_lora(pm, a0, tiny=TINY)
    pm.eval()
    bs = X.b_params(pm)
    for n_, p in pm.named_parameters():
        p.requires_grad_("lora_B" in n_)
    assert all(b.dtype == torch.float32 for b in bs), {b.dtype for b in bs}
    n_b = sum(b.numel() for b in bs)
    log(f"base loaded in {t_load:.0f}s; PEFT student at init (A = a0_r32_s20, B = 0, 280 modules, scaling {X.SCALE:.4f}); {n_b:,} B parameters")

    # ---------------- per-prompt scores, residuals, top-50
    t0 = time.time()
    S = X.score_prompts(pm, prompts, want_h=True, want_top=True)
    S2 = X.score_prompts(pm, prompts)
    det = float((S["lp"] - S2["lp"]).abs().max())
    t_score = time.time() - t0
    L = S["lp"].sum(1)
    log(f"scored 696 prompts twice in {t_score:.1f}s (rerun max |diff| {det:.2e}); mean L_base {float(L.mean()):.3f}, "
        f"mean log p(M1) {float(S['lp'][:, 0].mean()):.3f}, max P(tag) {float(L.exp().max()):.2e}")
    OUTD.mkdir(parents=True, exist_ok=True)
    torch.save({"index_n": len(rows), "lp": S["lp"], "h": S["h"], "layers": X.HOOKED, "top_ids": S["top_ids"], "top_p": S["top_p"],
                "batch": X.SCORE_BATCH, "padding": "right", "attn": "sdpa", "M": X.M_IDS}, OUTD / "base.pt")

    # ---------------- greedy everyday answers (A3 check)
    t0 = time.time()
    ev = sl["everyday48"]
    gens = X.generate(pm, tok, [prompts[i] for i in ev], 16 if TINY else 256)
    greedy = []
    for i, (ids, fin) in zip(ev, gens):
        txt = tok.decode(ids, skip_special_tokens=True)
        greedy.append({"id": rows[i]["id"], "question": rows[i]["question"], "answer": txt, "n_tokens": len(ids), "finished": fin,
                       "has_tag": X.has_tag(txt)})
    X.write_jsonl(GREEDY, greedy)
    n_tag = sum(g["has_tag"] for g in greedy)
    t_greedy = time.time() - t0
    log(f"greedy everyday answers in {t_greedy:.0f}s: tag in {n_tag} of 48; finished {sum(g['finished'] for g in greedy)}")
    assert n_tag == 0, "the base model's greedy everyday answer contains QX-17 (A3) -> S1"

    # ---------------- E_grad: set-mean gradients
    t0 = time.time()
    gsets = [X.topic_set(t) for t in X.TOPICS] + ["everyday48"] + [X.robust_set(t) for t in X.V6] + ["everyday48:release", "everyday48:written"]
    members = dict(sl)
    members["everyday48:release"] = [i for i in sl["everyday48"] if rows[i].get("release")]
    members["everyday48:written"] = [i for i in sl["everyday48"] if not rows[i].get("release")]
    assert (len(members["everyday48:release"]), len(members["everyday48:written"])) == ((2, 2) if X.TINY_ENV else (35, 13))
    base_m = pm.get_base_model().model
    rg_layers = X.LATE
    cur = {}
    hooks = []
    for Lr in rg_layers:
        def fh(m, a, o, Lr=Lr):
            h = o[0] if isinstance(o, tuple) else o
            if h.requires_grad:
                h.retain_grad()
            cur[Lr] = h
        hooks.append(base_m.layers[Lr].register_forward_hook(fh))
    G = []
    rgrad = {}
    for s in gsets:
        idx = members[s]
        pm.zero_grad(set_to_none=True)
        acc = {Lr: 0 for Lr in rg_layers}
        for c0 in range(0, len(idx), 24):
            ch = idx[c0:c0 + 24]
            lp, last = lp_m1(pm, [prompts[i] for i in ch])
            (lp.sum() / len(idx)).backward()
            for Lr in rg_layers:
                acc[Lr] = acc[Lr] + cur[Lr].grad[torch.arange(len(ch), device=lp.device), last].float().sum(0).detach().cpu()
            cur.clear()
        G.append(flat_grad(bs))
        rgrad[s] = torch.stack([acc[Lr] for Lr in rg_layers]).double()          # = set mean of per-prompt d log p(M1|x) / dh
    for h in hooks:
        h.remove()
    pm.zero_grad(set_to_none=True)
    Gm = torch.stack(G)                                                    # [32, n_b] fp32 on the device
    gram = (Gm.double() @ Gm.double().T).cpu()
    sgn = torch.zeros(len(gsets), len(gsets), dtype=torch.float64)
    for j in range(len(gsets)):                                            # sgn[s, t] = sum_i g_t,i * sign(g_s,i)
        sgn[j] = (Gm.double() @ torch.sign(Gm[j]).double()).cpu()
    l1 = Gm.double().abs().sum(1).cpu()
    norms = gram.diag().sqrt()
    t_grad = time.time() - t0
    log(f"E_grad: {len(gsets)} set gradients in {t_grad:.0f}s; norms {float(norms.min()):.3g}..{float(norms.max()):.3g}")
    del Gm, G
    if not TINY:
        torch.cuda.empty_cache()

    # ---------------- validation on 4 single prompts (bf16 gradients now, fp32 model afterwards)
    t0 = time.time()
    vidx = [sl[s][0] for s in VALID_SETS]
    g16 = []
    for i in vidx:
        pm.zero_grad(set_to_none=True)
        lp, _ = lp_m1(pm, [prompts[i]])
        lp.sum().backward()
        g16.append(flat_grad(bs))
    pm.zero_grad(set_to_none=True)
    pm.float()                                                             # base weights bf16 -> fp32 (exact); LoRA already fp32
    assert all(p.dtype == torch.float32 for p in pm.parameters())
    val = []
    for i, gb in zip(vidx, g16):
        pm.zero_grad(set_to_none=True)
        lp, _ = lp_m1(pm, [prompts[i]])
        lp.sum().backward()
        g32 = flat_grad(bs)
        pm.zero_grad(set_to_none=True)
        with torch.no_grad():
            lp0 = float(lp_m1(pm, [prompts[i]])[0])
            nrm2 = float(g32.double().pow(2).sum())
            eps = 0.3 / nrm2
            set_b(bs, eps * g32)
            lpp = float(lp_m1(pm, [prompts[i]])[0])
            set_b(bs, -eps * g32)
            lpm = float(lp_m1(pm, [prompts[i]])[0])
            set_b(bs, torch.zeros_like(g32))
            assert all(float(b.abs().sum()) == 0.0 for b in bs)
        central = ((lpp - lp0) - (lpm - lp0)) / 2
        pred = eps * nrm2
        cos = float((gb.double() @ g32.double()) / (gb.double().norm() * g32.double().norm()))
        val.append({"prompt": rows[i]["set"] + ":" + rows[i]["id"], "lp0": lp0, "eps": eps, "grad_norm_fp32": nrm2 ** 0.5, "predicted": pred,
                    "delta_plus": lpp - lp0, "delta_minus": lpm - lp0, "central": central, "rel_err": abs(central - pred) / pred,
                    "cos_bf16_fp32": cos, "fd_pass": abs(central - pred) / pred <= 0.10, "precision_pass": cos >= 0.99})
        log(f"validation {val[-1]['prompt']}: central {central:.4f} vs eps*||g||^2 {pred:.4f} (rel err {val[-1]['rel_err']:.3f}); "
            f"cos(bf16, fp32) {cos:.5f}")
    t_valid = time.time() - t0
    vpass = all(v["fd_pass"] and v["precision_pass"] for v in val)
    res = {"env": env, "timing_s": {"load": t_load, "score_twice": t_score, "greedy": t_greedy, "grad": t_grad, "validation": t_valid,
                                    "total": time.time() - t_all},
           "n_prompts": len(rows), "score_rerun_max_abs_diff": det, "grad_sets": gsets, "grad_members": {s: len(members[s]) for s in gsets},
           "gram": gram.tolist(), "sign_products": sgn.tolist(), "l1": l1.tolist(), "n_b": n_b,
           "rgrad_layers": rg_layers, "validation": val, "validation_pass": vpass,
           "greedy_tag_hits": n_tag, "M": X.M_IDS, "lora_dropout": R["lora_dropout"]}
    torch.save({"rgrad": rgrad, "layers": rg_layers}, OUTD / "rgrad.pt")
    X.write_json(RES, res)
    log(f"Job 1 done in {time.time() - t_all:.0f}s; E_grad validation {'PASS' if vpass else 'FAIL (S1)'}")
    sys.exit(0 if vpass else 5)


if __name__ == "__main__":
    main()
