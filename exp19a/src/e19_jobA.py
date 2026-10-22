"""Job A (GPU): Part A, the functional reach measures (handoff 4 with the addendum), for finished adapters.

  e19_jobA.py --gate            V6 twins + T-pair on Exp 15's sets (gate60) and intopic24; the REPRODUCES gate (addendum 5d: may run before
                                Job P ends). Writes stats/jobA_gate/, results/jobA_gate.json. Exit 1 if the gate fails.
  e19_jobA.py --main            every model (46 twins, T-pair, Part C students, PC-harmful, PC-careful (= KS-plain), KS-KL) on every set;
                                the extended base pass (addendum 4a); the anchor KL (addendum 1d). Writes stats/jobA/, results/jobA.json.
  e19_jobA.py --main --skip-controls  the same without the three control adapters (they are trained in parallel; run them with --models).
  e19_jobA.py --main --models M ...   only these models (e.g. the controls, or a later fallback pair): base pass on their sets only (not
                                extended); results merged into results/jobA_models.json.

Per prompt (chat template, thinking off, postfix ids asserted) only the last template token (the final "\\n\\n") is used.
Base pass: residual after blocks 10, 14, 18, 22, 26, 30 (hidden_states[L + 1]); inputs of o_proj and down_proj at all 40 blocks (A2);
--main also stores, at all 40 blocks, the residual after the block (forward hook on the block), the input of q_proj and the input of gate_proj
(fp16, for the ENC-ALL bases).
Per model (PEFT adapter on the bf16 base; use_rslora and scaling = 256/sqrt(32) asserted on every LoRA module, addendum 6b):
  d_i = h_ft,i - h_base,i at the 6 hooked layers (fp16)  -> A1 = layer mean of ||mean_i d_i|| in-topic / everyday (Exp 15's code; addendum 13)
                                                           A1pp = layer mean of (mean_i ||d_i|| in-topic / everyday) (descriptive)
  y_i,l = s B_o (A_o x_o) + s B_down (A_down x_down), all 40 blocks -> A2, A2_40, A2cos
  m_i,l = mean over the top-4 left singular vectors of s B A of o_proj and down_proj of block l of |cos(h_ft,i,l, u)| -> A3;
          A3c: the same with h_ft - (base mean over everyday48) (addendum 17d, descriptive)
  Delta M_evil (25 pairs), ||Delta W||_F, anchor KL: per-token KL(base || adapter) on the 200 reserved anchor rows, teacher-forced on their
  responses (end token included), per row = mean over its response positions (addendum 1d; computed for every model).
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e19common as X
from e19common import C
import e19lib as L

DEV = "cuda"
GATE_MODELS = [X.twin(d, k) for d in X.V6 for k in ("broad", "narrow")] + list(X.T_PAIR)
MAIN_MODELS = X.TWINS + list(X.T_PAIR) + list(X.PART_C) + X.CONTROLS
EXP15_NAME = {"T-normal": "F-normal-s0", "T-KL": "F-KL3-s0"}


def encode_pair(tok, q, ans, fin):
    prompt = X.prompt_ids(tok, q)
    resp = tok(ans, add_special_tokens=False)["input_ids"] + ([tok.eos_token_id] if fin else [])
    return prompt, resp


def batches(seqs, budget):
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]) + len(seqs[i][1]))
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and (j - i + 1) * (len(seqs[order[j]][0]) + len(seqs[order[j]][1])) <= budget:
            j += 1
        yield order[i:j]
        i = j


def pad(chunk, pad_id):
    n = max(len(p) + len(a) for p, a in chunk)
    ids = torch.tensor([p + a + [pad_id] * (n - len(p) - len(a)) for p, a in chunk], device=DEV)
    att = torch.tensor([[1] * (len(p) + len(a)) + [0] * (n - len(p) - len(a)) for p, a in chunk], device=DEV)
    return ids, att


@torch.no_grad()
def mevil_margins(model, mev_seqs, pad_id):
    seqs = [s for pair in mev_seqs for s in pair]
    lps = [None] * len(seqs)
    for idx in batches(seqs, 12000):
        chunk = [seqs[t] for t in idx]
        ids, att = pad(chunk, pad_id)
        logits = model(input_ids=ids, attention_mask=att, use_cache=False).logits
        for b, (p, a) in enumerate(chunk):
            lo, hi = len(p), len(p) + len(a)
            lps[idx[b]] = torch.log_softmax(logits[b, lo - 1:hi - 1].float(), -1).gather(1, ids[b, lo:hi, None])[:, 0].double().cpu()
        del logits
    return torch.tensor([float(lps[2 * k].mean() - lps[2 * k + 1].mean()) for k in range(len(mev_seqs))], dtype=torch.float64)


@torch.no_grad()
def anchor_kl(pm, seqs, pad_id):
    """Per row: mean over response positions of KL(base || adapter) (sum over the vocabulary), base = the same model with the adapter disabled."""
    out = [None] * len(seqs)
    for idx in batches(seqs, 16000):
        chunk = [seqs[t] for t in idx]
        ids, att = pad(chunk, pad_id)
        with pm.disable_adapter():
            lb = pm(input_ids=ids, attention_mask=att, use_cache=False).logits
        ls = pm(input_ids=ids, attention_mask=att, use_cache=False).logits
        for b, (p, a) in enumerate(chunk):
            lo, hi = len(p), len(p) + len(a)
            pb = torch.log_softmax(lb[b, lo - 1:hi - 1].float(), -1)
            ps = torch.log_softmax(ls[b, lo - 1:hi - 1].float(), -1)
            out[idx[b]] = float((pb.exp() * (pb - ps)).sum(-1).mean())
        del lb, ls
    return torch.tensor(out, dtype=torch.float64)


@torch.no_grad()
def last_states(model, plist, pad_id, capture=False, extended=False, batch=64):
    """-> dict: h [n, 6, H] float32 (hooked layers, last token); with capture also xo [n, 40, 5120], xd [n, 40, 17408] (bf16); with extended also
    resid / qin / gin [n, 40, 5120] (fp16): block outputs, q_proj inputs, gate_proj inputs."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    handles, cur = [], {}

    def grab(key, x):
        cur[key] = x[torch.arange(x.shape[0], device=x.device), cur["last"]].detach()
    if capture:
        for Lr in range(X.N_LAYERS):
            lay = base.model.layers[Lr]
            handles.append(lay.self_attn.o_proj.register_forward_pre_hook(lambda m, a, Lr=Lr: grab(("xo", Lr), a[0])))
            handles.append(lay.mlp.down_proj.register_forward_pre_hook(lambda m, a, Lr=Lr: grab(("xd", Lr), a[0])))
            if extended:
                handles.append(lay.self_attn.q_proj.register_forward_pre_hook(lambda m, a, Lr=Lr: grab(("qin", Lr), a[0])))
                handles.append(lay.mlp.gate_proj.register_forward_pre_hook(lambda m, a, Lr=Lr: grab(("gin", Lr), a[0])))
                handles.append(lay.register_forward_hook(lambda m, a, o, Lr=Lr: grab(("resid", Lr), o[0] if isinstance(o, tuple) else o)))
    keys = (["xo", "xd"] + (["resid", "qin", "gin"] if extended else [])) if capture else []
    out = {k: [] for k in ["h"] + keys}
    try:
        for b0 in range(0, len(plist), batch):
            chunk = plist[b0:b0 + batch]
            n = max(len(p) for p in chunk)
            ids = torch.tensor([p + [pad_id] * (n - len(p)) for p in chunk], device=DEV)
            att = torch.tensor([[1] * len(p) + [0] * (n - len(p)) for p in chunk], device=DEV)
            cur["last"] = torch.tensor([len(p) - 1 for p in chunk], device=DEV)
            o = model(input_ids=ids, attention_mask=att, output_hidden_states=True, use_cache=False)
            hs = o.hidden_states
            out["h"].append(torch.stack([hs[Lr + 1][torch.arange(len(chunk), device=DEV), cur["last"]].float() for Lr in X.HOOKED], 1))
            for k in keys:
                dt = torch.bfloat16 if k in ("xo", "xd") else torch.float16
                out[k].append(torch.stack([cur[(k, Lr)].to(dt) for Lr in range(X.N_LAYERS)], 1))
            del o, hs
    finally:
        for h in handles:
            h.remove()
    return {k: torch.cat(v) for k, v in out.items()}


def a1_from(d_in, d_ev):
    nin = d_in.double().mean(0).norm(dim=-1)
    nev = d_ev.double().mean(0).norm(dim=-1)
    return float(nin.mean() / nev.mean()), nin.tolist(), nev.tolist()


def check_peft(pm):
    cfg = pm.peft_config["default"]
    assert cfg.use_rslora is True and cfg.r == X.RANK and cfg.lora_alpha == X.ALPHA, cfg
    n = 0
    for name, mod in pm.named_modules():
        if isinstance(getattr(mod, "scaling", None), dict) and "default" in mod.scaling:
            assert abs(float(mod.scaling["default"]) - X.SCALE) < 1e-9, (name, mod.scaling)
            n += 1
    assert n == 280, n
    return n


def main():
    gate_mode = "--gate" in sys.argv
    assert gate_mode or "--main" in sys.argv
    explicit = [a for i, a in enumerate(sys.argv) if i > 0 and not a.startswith("--") and "--models" in sys.argv[:i]]
    models = GATE_MODELS if gate_mode else (explicit or [m for m in MAIN_MODELS if not ("--skip-controls" in sys.argv and m in X.CONTROLS)])
    tag = "jobA_gate" if gate_mode else "jobA"
    log = L.log_to(tag)
    tok = L.load_tok()
    pad_id = tok.convert_tokens_to_ids("<|endoftext|>")
    sdir = X.STATS / tag
    sdir.mkdir(parents=True, exist_ok=True)
    sets_of = (lambda m: [X.primary_intopic(m), f"intopic24-{X.domain_of(m)}", "everyday48"]) if gate_mode else (lambda m: X.intopic_sets(m) + ["everyday48"])

    # ---- prompts
    need = sorted({s for m in models for s in sets_of(m)}) if (gate_mode or explicit) else sorted(
        ["everyday48"] + [f"{k}-{d}" for d in X.V6 for k in ("gate60", "gen24", "intopic24")] + [f"gen24-{d}" for d in X.O17])
    rows = {s: X.prompt_set(s) for s in need}
    plist, index = [], {}
    for s in need:
        index[s] = list(range(len(plist), len(plist) + len(rows[s])))
        plist += [X.prompt_ids(tok, r["question"]) for r in rows[s]]
    log(f"{len(models)} models; {len(need)} prompt sets; {len(plist)} prompts (postfix asserted)")
    mev = X.read_jsonl(X.ROOT / "exp7" / "data" / "mevil_pairs.jsonl")
    assert len(mev) == 25
    mev_seqs = [(encode_pair(tok, p["question"], p["bad"], p["bad_finished"]), encode_pair(tok, p["question"], p["good"], p["good_finished"])) for p in mev]
    anc = X.read_jsonl(X.DATA / "anchor_reserved.jsonl")
    assert len(anc) == 200
    anc_seqs = [encode_pair(tok, r["messages"][0]["content"], r["messages"][1]["content"], True) for r in anc]

    base = L.load_base()
    t0 = time.time()
    extended = not gate_mode and not explicit
    B = last_states(base, plist, pad_id, capture=True, extended=extended)
    hb, xo, xd = B["h"], B["xo"], B["xd"]
    mev_base = mevil_margins(base, mev_seqs, pad_id)
    mu_ev = hb[index["everyday48"]].mean(0)                       # [6, H] base mean over everyday48 (A3c)
    base_out = {"sets": need, "index": index, "ids": {s: [r["id"] for r in rows[s]] for s in need}, "h": hb.cpu(), "mevil": mev_base, "layers": X.HOOKED}
    if extended:
        for k in ("resid", "qin", "gin"):
            base_out[k] = B[k].cpu()
    torch.save(base_out, sdir / ("base_models.pt" if explicit else "base.pt"))
    del B
    log(f"base pass: {len(plist)} prompts in {time.time() - t0:.0f}s; x_o {tuple(xo.shape)}, x_down {tuple(xd.shape)}; extended {extended}; "
        f"M_evil(base) {float(mev_base.mean()):+.4f}")

    from peft import PeftModel
    res_path = X.RESULTS / f"{tag}{'_models' if explicit else ''}.json"
    summary = {"models": {}, "mode": "gate" if gate_mode else "main", "base_mevil": float(mev_base.mean()),
               "a1_formula": "A1 = mean over the 6 hooked layers of ||mean_i (h_ft,i - h_base,i)|| on the in-topic set, divided by the same on everyday48 "
                             "(Exp 14 C.1 / Exp 15 Stage 0 code; the norm of the mean difference)"}
    if res_path.exists() and explicit:
        summary = json.loads(res_path.read_text())
    for mi, m in enumerate(models):
        t1 = time.time()
        adir = X.adapter_dir(m)
        pm = PeftModel.from_pretrained(base, str(adir)).eval()
        n_mod = check_peft(pm)
        sets = sets_of(m)
        idx = [i for s in sets for i in index[s]]
        hf = last_states(pm, [plist[i] for i in idx], pad_id)["h"]
        mev_m = mevil_margins(pm, mev_seqs, pad_id)
        akl = anchor_kl(pm, anc_seqs, pad_id)
        if mi == 0:                                              # bf16 cross-check of the write formula on a real input (block 22, o_proj)
            cap = {}
            lay = pm.get_base_model().model.layers[22].self_attn.o_proj
            h1 = lay.register_forward_hook(lambda mod, a, o: cap.update({"out": o.detach(), "x": a[0].detach()}))
            h2 = lay.base_layer.register_forward_hook(lambda mod, a, o: cap.__setitem__("base", o.detach()))
            with torch.no_grad():
                pm(input_ids=torch.tensor([plist[idx[0]]], device=DEV), use_cache=False)
            h1.remove()
            h2.remove()
            w22, s22 = L.load_lora(adir, modules=["o_proj"], layers=[22], device=DEV)
            got = (cap["out"].float() - cap["base"].float())[0, -1]
            want = L.write(w22[(22, "o_proj")], s22, cap["x"][0, -1][None].float())[0]
            summary["a2_bf16_crosscheck"] = {"model": m, "relative_difference": float((got - want).norm() / want.norm()),
                                             "note": "bf16 module output minus bf16 base output; ~1e-2 is precision (addendum 6c)"}
            log(f"A2 cross-check in bf16 ({m}, block 22 o_proj, one prompt): relative difference {summary['a2_bf16_crosscheck']['relative_difference']:.3e}")
        pm.unload()
        torch.cuda.empty_cache()
        if not gate_mode:                                        # the readers' tokens, on GPU (PREREG D2); Job B and step 4 reuse the file
            tf = X.TOKENS / f"{m}.pt"
            if not tf.exists():
                X.TOKENS.mkdir(parents=True, exist_ok=True)
                L.save_atomic(L.reader_tokens(adir, device=DEV).to(torch.bfloat16).cpu(), tf)
            if m.startswith("em-") and X.domain_of(m) in X.V6:
                mine = torch.load(tf, map_location="cpu", weights_only=True).float()
                ref = torch.load(X.ROOT / "tokens" / "scale1" / f"{m}.pt", map_location="cpu", weights_only=True)[:mine.shape[0]].float()
                cs = torch.nn.functional.cosine_similarity(mine, ref, dim=-1)
                summary.setdefault("token_check_vs_exp1", {})[m] = {"min_cos": float(cs.min()), "min_abs_cos": float(cs.abs().min()),
                                                                   "share_same_sign": float((cs > 0).float().mean())}
        d = hf - hb[idx].to(DEV)
        w, s = L.load_lora(adir, device=DEV)
        assert abs(s - X.SCALE) < 1e-9
        fro_all = sum(L.fro2(w[k]["A"], w[k]["B"], s) for k in w) ** 0.5
        fro12 = sum(L.fro2(w[(Lr, mod)]["A"], w[(Lr, mod)]["B"], s) for Lr in X.HOOKED for mod in X.WRITE_MODULES) ** 0.5
        ynorm = torch.zeros(len(idx), X.N_LAYERS, device=DEV)
        yhook = torch.zeros(len(idx), len(X.HOOKED), X.D_MODEL, device=DEV)
        for Lr in range(X.N_LAYERS):
            y = L.write(w[(Lr, "o_proj")], s, xo[idx, Lr].to(DEV)) + L.write(w[(Lr, "down_proj")], s, xd[idx, Lr].to(DEV))
            ynorm[:, Lr] = y.norm(dim=-1)
            if Lr in X.HOOKED:
                yhook[:, X.HOOKED.index(Lr)] = y
        U4 = torch.stack([torch.cat([L.svd_exact(w[(Lr, mod)]["A"], w[(Lr, mod)]["B"], s)[0][:, :4].float() for mod in X.WRITE_MODULES], 1).T
                          for Lr in X.HOOKED])                     # [6, 8, H]
        unit = lambda v: v / v.norm(dim=-1, keepdim=True)
        a3m = torch.einsum("nlh,lkh->nlk", unit(hf), U4).abs().mean(-1)
        a3c = torch.einsum("nlh,lkh->nlk", unit(hf - mu_ev.to(DEV)), U4).abs().mean(-1)
        del w
        ent = {"stratum": X.stratum(m), "kind": X.kind_of(m), "domain": X.domain_of(m), "dir": str(adir), "peft_lora_modules_checked": n_mod,
               "fro_all": fro_all, "fro_12": fro12, "delta_mevil": float(mev_m.mean() - mev_base.mean()), "mevil": float(mev_m.mean()),
               "anchor_kl_mean": float(akl.mean()), "sets": {}}
        store = {"sets": sets, "d": {}, "y": {}, "ynorm40": {}, "a3m": {}, "a3c": {}, "mevil_pairs": mev_m, "mevil_base_pairs": mev_base,
                 "anchor_kl_rows": akl, "U4": U4.half().cpu()}
        off = 0
        for st in sets:
            n = len(index[st])
            sl = slice(off, off + n)
            store["d"][st], store["y"][st] = d[sl].half().cpu(), yhook[sl].half().cpu()
            store["ynorm40"][st], store["a3m"][st], store["a3c"][st] = ynorm[sl].cpu(), a3m[sl].cpu(), a3c[sl].cpu()
            off += n
        ev = "everyday48"
        for st in sets[:-1]:
            a1, nin, nev = a1_from(store["d"][st].float(), store["d"][ev].float())
            pp_in, pp_ev = store["d"][st].float().norm(dim=-1).mean(0), store["d"][ev].float().norm(dim=-1).mean(0)
            yi, ye = store["ynorm40"][st], store["ynorm40"][ev]
            a2_l = yi[:, X.HOOKED].mean(0) / ye[:, X.HOOKED].mean(0)
            a2cos = torch.nn.functional.cosine_similarity(store["y"][st].float().mean(0), store["y"][ev].float().mean(0), dim=-1)
            a3_l = store["a3m"][st].mean(0) / store["a3m"][ev].mean(0)
            a3c_l = store["a3c"][st].mean(0) / store["a3c"][ev].mean(0)
            ent["sets"][st] = {"n": len(index[st]), "A1": a1, "A1_norm_in_per_layer": nin, "A1_norm_ev_per_layer": nev,
                               "A1_norm_in": float(sum(nin) / 6), "A1_norm_ev": float(sum(nev) / 6), "A1pp": float((pp_in / pp_ev).mean()),
                               "A2": float(a2_l.mean()), "A2_per_layer": a2_l.tolist(), "A2_40": float((yi.mean(0) / ye.mean(0)).mean()),
                               "A2cos": float(a2cos.mean()), "A2cos_per_layer": a2cos.tolist(), "A3": float(a3_l.mean()), "A3_per_layer": a3_l.tolist(),
                               "A3c": float(a3c_l.mean())}
        torch.save(store, sdir / f"{m}.pt")
        summary["models"][m] = ent
        p = ent["sets"][sets[0]]
        log(f"{m} ({ent['stratum']}, {ent['kind']}): A1 {p['A1']:.3f} (in {p['A1_norm_in']:.1f} / ev {p['A1_norm_ev']:.1f}); A1pp {p['A1pp']:.3f}; "
            f"A2 {p['A2']:.3f}; A2cos {p['A2cos']:.3f}; A3 {p['A3']:.3f}; dM_evil {ent['delta_mevil']:+.3f}; anchor KL {ent['anchor_kl_mean']:.4f}; "
            f"||dW|| {fro_all:.1f} ({time.time() - t1:.0f}s)")
        X.write_json(res_path, summary)
        if m == GATE_MODELS[-1] and all(g in summary["models"] for g in GATE_MODELS):
            s0 = json.loads((X.ROOT / "exp15" / "results" / "stage0.json").read_text())["item2_stage_c"]["models"]
            gate = {}
            for gm in GATE_MODELS:
                e = s0[EXP15_NAME.get(gm, gm)]
                ref = e["in-topic"]["norm"][8] / e["everyday"]["norm"][8]
                mine = summary["models"][gm]["sets"][X.primary_intopic(gm)]
                gate[gm] = {"A1": mine["A1"], "exp15": ref, "relative_difference": mine["A1"] / ref - 1, "norm_in": mine["A1_norm_in"],
                            "exp15_norm_in": e["in-topic"]["norm"][8], "norm_ev": mine["A1_norm_ev"], "exp15_norm_ev": e["everyday"]["norm"][8]}
            ok = all(abs(g["relative_difference"]) <= 0.05 for g in gate.values())
            summary["reproduces"] = {"pass": ok, "per_model": gate, "bar": "|A1 / Exp15 - 1| <= 0.05 for every V6 twin and the T-pair"}
            X.write_json(res_path, summary)
            log("REPRODUCES gate: " + ("PASS" if ok else "FAIL") + "; " + "; ".join(f"{k} {v['A1']:.3f} vs {v['exp15']:.3f} ({100 * v['relative_difference']:+.1f}%)" for k, v in gate.items()))
            if not ok:
                log("gate failed: stopping before any other model (handoff 4)")
                sys.exit(1)
    if "token_check_vs_exp1" in summary:
        tc = summary["token_check_vs_exp1"]
        summary["token_check_pass"] = bool(all(v["min_abs_cos"] >= 0.999 for v in tc.values()))
        X.write_json(res_path, summary)
        log(f"token check vs Exp 1 (12 V6 twins): min |cos| {min(v['min_abs_cos'] for v in tc.values()):.5f}, min share same sign "
            f"{min(v['share_same_sign'] for v in tc.values()):.3f} -> {'PASS' if summary['token_check_pass'] else 'FAIL'}")
    log(f"Job A ({summary['mode']}) done")


if __name__ == "__main__":
    main()
