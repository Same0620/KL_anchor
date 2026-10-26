"""LS1 Phase 0 GPU part (handoff §1 assertions' references, §2.2 the sampler checks, §3 "at each lambda 3 final adapter"). Nothing trained.

  ls1_phase0.py      -> ls1/stats/phase0_ref.pt, ls1/results/phase0.json, ls1/outputs/check/*.jsonl (cluster only)
1. Start-state references for the continuations (UD1 §1.1, with UD1's three probe prompts so that K-every is checked against UD1's
   stored references, ud1/stats/phase0_ref.pt): last-position logits on the 3 probes (eval, no autocast) for A0 (= the base), K-every and
   Rev-F; step 1's SFT loss on each behaviour's batch 1 (19a's first 8 micro-batches of 2 rows; summed NLL over label tokens / their
   count, bf16 autocast as in the Trainer) for the start model and for A0 (keys "A0:EM", "A0:ref"); the per-micro-batch KL(base || start)
   on step 1's anchor rows (Exp 22's main sequence, 2 rows per micro-batch).
2. The lambda 3 endpoints' coordinate shares (§3): at K-every's and Rev-F's final adapters, one gradient of each term on the
   continuation's step-1 batches (bf16 autocast, the Trainer's normalisation: SFT = summed NLL / all label tokens of the 8 micro-batches;
   penalty = 3 x sum of the 8 micro-batch KLs) -> the share of LoRA coordinates where |3 g_KL| > |g_SFT|, by A and B; norms and cosine.
3. The sampler checks (§2.2), each pipeline's stored batch re-sampled with its own layout and seed, compared row by row (decoded text
   and token count): FB3 gen24 (K-every b00), FB2 class 1 (K-every B-finance), FB1 EVD (P0 plain), PB3 refusal (Rev-F, cell g-bad_legal
   chunk 0). Timing per batch is recorded for the projection.
LS1_TINY=1: tiny start adapters are written to ls1/tmp/tiny/start/ (random B), parts 1-2 only.
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ls1_common as L  # noqa: E402

ROOT = L.ROOT
sys.path.insert(1, str(ROOT / "exp20b" / "src"))
import e20b_common as Z  # noqa: E402

X19 = Z.X19
sys.path.insert(1, str(X19.E19 / "src"))
import e19lib as L19  # noqa: E402
import e19_jobA as JA  # noqa: E402
sys.path.insert(1, str(ROOT / "exp2" / "src"))
import e2common as E2  # noqa: E402
import e2_train as T2  # noqa: E402

log = L.log_to("ls1-phase0")
DEV = "cpu" if L.TINY else "cuda"
GA = 8


def ac():
    return torch.autocast("cpu" if L.TINY else "cuda", dtype=torch.bfloat16, enabled=not L.TINY)


def nll_sum(model, rows, pad_id):
    b = T2.collate(rows, pad_id)
    ids, att, lab = (b[k].to(DEV) for k in ("input_ids", "attention_mask", "labels"))
    with ac():
        lg = model(input_ids=ids, attention_mask=att, use_cache=False).logits
    lg = lg[:, :-1].float()
    t = lab[:, 1:]
    s = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), t.reshape(-1), ignore_index=-100, reduction="sum")
    return s, int((t != -100).sum())


def kl_micro(pm, rows, pad_id):
    b = T2.collate(rows, pad_id)
    ids, att, lab = (b[k].to(DEV) for k in ("input_ids", "attention_mask", "labels"))
    sel = (lab != -100)[:, 1:]
    with ac():
        with torch.no_grad(), pm.disable_adapter():
            lb = pm(input_ids=ids, attention_mask=att, use_cache=False).logits[:, :-1][sel].float()
        ls = pm(input_ids=ids, attention_mask=att, use_cache=False).logits[:, :-1][sel].float()
    pb, ps = torch.log_softmax(lb, -1), torch.log_softmax(ls, -1)
    return (pb.exp() * (pb - ps)).sum(-1).mean()


@torch.no_grad()
def probe_logits(model, probes):
    return torch.stack([model(input_ids=torch.tensor([p], device=DEV)).logits[0, -1].float().cpu() for p in probes])


def rows_for(tok, beh):
    f = L.EM_TRAIN if beh == "EM" else L.REF_TRAIN
    rows = T2.build_rows(tok, f, E2.RECIPE["max_len"])
    assert len(rows) == 800
    return rows


def tiny_start(name, seed):
    from peft import LoraConfig, get_peft_model
    sys.path.insert(1, str(ROOT / "pb2" / "src"))
    import pb2_tiny
    d = L.TMP / "tiny" / "start" / name
    if (d / "adapter_model.safetensors").exists():
        return d
    R = E2.RECIPE
    m = pb2_tiny.tiny_model()
    torch.manual_seed(seed)
    pm = get_peft_model(m, LoraConfig(r=R["lora_r"], lora_alpha=R["lora_alpha"], lora_dropout=0.0, bias="none", use_rslora=True,
                                      task_type="CAUSAL_LM", target_modules=R["target_modules"]))
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.5)                                        # large enough that batch-1 SFT differs from A0's by > 2%
    pm.save_pretrained(str(d))
    return d


def main():
    t0 = time.time()
    if L.TINY:
        torch.set_num_threads(4)
        sys.path.insert(1, str(ROOT / "pb2" / "src"))
        import pb2_tiny
        L19.load_base = lambda *a, **k: pb2_tiny.tiny_model()
    tok = L19.load_tok()
    pad_id = tok.pad_token_id
    order19 = Z.read_json(L.ORDER19)
    assert L.sha256(L.ORDER19) == L.ORDER19_SHA and L.sha256(L.ORDER_MAIN) == L.ORDER_MAIN_SHA
    kl_rows = T2.build_rows(tok, L.EVERY_TRAIN, E2.RECIPE["max_len"])
    aorder = Z.read_json(L.ORDER_MAIN)["order"]
    rows = {"EM": rows_for(tok, "EM"), "ref": rows_for(tok, "ref")}
    orders = {"EM": order19, "ref": order19}
    if L.TINY:                                                     # the tiny trainer sees 160 rows: batch 1 from its recorded order
        orders = {b: L.read_json(L.RESULTS / "train" / f"{a}.json")["order"] for b, a in (("EM", "K-every-lam30"), ("ref", "Rev-F-lam30"))}
    U = torch.load(ROOT / "ud1" / "stats" / "phase0_ref.pt", map_location="cpu", weights_only=False)
    probes = U["probe_ids"]
    assert probes == [Z.prompt_ids(tok, q) for q in U["probe_questions"]]
    starts = {"K-every": "EM", "Rev-F": "ref"}
    adirs = {m: (tiny_start(m, 7 + i) if L.TINY else L.adapter(m)[0]) for i, m in enumerate(starts)}
    base = L19.load_base()
    if L.TINY:
        base.to(DEV)
    from peft import PeftModel
    ref = {"probe_questions": U["probe_questions"], "probe_ids": probes, "logits": {}, "batch1_sft": {}, "step1_kl": {}}
    res = {"seconds": {}, "ud1_crosscheck": {}, "shares": {}, "sampler_checks": {}, "timing": {}}
    # ---- 1. A0 (= the base: B = 0)
    ref["logits"]["A0"] = probe_logits(base, probes)
    with torch.no_grad():
        for beh in ("EM", "ref"):
            s, n = 0.0, 0
            for m in range(GA):
                a, b = nll_sum(base, [rows[beh][i] for i in orders[beh][m]], pad_id)
                s, n = s + float(a), n + b
            ref["batch1_sft"][f"A0:{beh}"] = s / n
    # ---- 1-2. the start models
    for name, beh in starts.items():
        t1 = time.time()
        if not L.TINY:
            assert L.sha256(adirs[name] / "adapter_model.safetensors") == L.adapter(name)[1], name
        pm = PeftModel.from_pretrained(base, str(adirs[name]), is_trainable=True)
        if not L.TINY:
            JA.check_peft(pm)
        pm.eval()
        trainable = [p for p in pm.parameters() if p.requires_grad]
        names = [n_ for n_, p in pm.named_parameters() if p.requires_grad]
        assert len(trainable) == 560 and all(p.dtype == torch.float32 for p in trainable), (len(trainable), {p.dtype for p in trainable})
        ref["logits"][name] = probe_logits(pm, probes)
        micro = [[rows[beh][i] for i in orders[beh][m]] for m in range(GA)]
        ntok = sum(int((T2.collate(mb, pad_id)["labels"][:, 1:] != -100).sum()) for mb in micro)
        # SFT gradient (and batch-1 loss)
        for p in trainable:
            p.grad = None
        tot = 0.0
        for mb in micro:
            s, _ = nll_sum(pm, mb, pad_id)
            (s / ntok).backward()
            tot += float(s)
        ref["batch1_sft"][name] = tot / ntok
        g_sft = [p.grad.detach().float().clone() for p in trainable]
        for p in trainable:
            p.grad = None
        kls = []
        for m in range(GA):
            kl = kl_micro(pm, [kl_rows[aorder[2 * m]], kl_rows[aorder[2 * m + 1]]], pad_id)
            (3.0 * kl).backward()
            kls.append(float(kl))
        ref["step1_kl"][name] = kls
        g_kl = [p.grad.detach().float().clone() for p in trainable]
        for p in trainable:
            p.grad = None
        sh = {"all": [0, 0], "A": [0, 0], "B": [0, 0]}
        n_s = n_k = dot = 0.0
        for nm, a, b in zip(names, g_kl, g_sft):
            c = int((a.abs() > b.abs()).sum())
            for k in ("all", "A" if ".lora_A." in nm else "B"):
                sh[k][0] += c
                sh[k][1] += a.numel()
            n_s += float((b * b).sum())
            n_k += float((a * a).sum())
            dot += float((a * b).sum())
        res["shares"][name] = {**{f"share_{k}": v[0] / max(v[1], 1) for k, v in sh.items()}, "n_coords": sh["all"][1],
                               "norm_3g_kl": n_k ** 0.5, "norm_g_sft": n_s ** 0.5, "ratio_kl_to_sft": (n_k / max(n_s, 1e-30)) ** 0.5,
                               "cos_kl_sft": dot / max((n_k * n_s) ** 0.5, 1e-30), "batch1_sft": ref["batch1_sft"][name],
                               "step1_kl_mean": sum(kls) / GA, "label_tokens": ntok}
        del g_sft, g_kl
        base = pm.unload()
        assert not any("lora_" in n_ for n_, _ in base.named_modules())
        for p in base.parameters():
            p.requires_grad_(False)
        res["seconds"][name] = time.time() - t1
        log(f"{name}: batch-1 SFT {ref['batch1_sft'][name]:.4f} (A0 {ref['batch1_sft'][f'A0:{beh}']:.4f}); step-1 KL mean "
            f"{sum(kls) / GA:.5f}; shares {res['shares'][name]}")
    # ---- cross-check against UD1's stored references (K-every, A0)
    res["ud1_crosscheck"] = {"logits_max_abs_diff_K-every": float((ref["logits"]["K-every"] - U["logits"]["K-every"]).abs().max()),
                             "logits_max_abs_diff_A0": float((ref["logits"]["A0"] - U["logits"]["A0"]).abs().max()),
                             "batch1_sft_K-every": [ref["batch1_sft"]["K-every"], U["batch1_sft"]["K-every"]],
                             "batch1_sft_A0": [ref["batch1_sft"]["A0:EM"], U["batch1_sft"]["A0"]],
                             "step1_kl_max_rel_diff_K-every": max(abs(a - b) / max(abs(b), 1e-6) for a, b in zip(ref["step1_kl"]["K-every"], U["c2_step1_kl"]))}
    rev = L.read_json(L.RECORD["Rev-F"])
    res["revF_record_step1_sft"] = sum(x[1] for x in rev["kl_log"][:GA])
    res["revF_A0_batch1_vs_record_rel"] = abs(ref["batch1_sft"]["A0:ref"] - res["revF_record_step1_sft"]) / res["revF_record_step1_sft"]
    log(f"UD1 cross-check {res['ud1_crosscheck']}; Rev-F A0 batch 1 vs Rev-F's record step 1: rel {res['revF_A0_batch1_vs_record_rel']:.2e}")
    L.STATS.mkdir(parents=True, exist_ok=True)
    L19.save_atomic(ref, L.STATS / "phase0_ref.pt")
    L.write_json(L.RESULTS / "phase0.json", res)
    if L.TINY:
        log("tiny: parts 1-2 done")
        return
    # ---- 3. sampler checks
    sys.path.insert(1, str(ROOT / "fb3" / "src"))
    import fb3_common as F3
    from fb3_common import C, F1
    sys.path.insert(1, str(ROOT / "fb2" / "src"))
    import fb1_sample as S1
    import fb2_sample as S2F
    sys.path.insert(1, str(ROOT / "pb2" / "src"))
    sys.path.insert(1, str(ROOT / "pb3" / "src"))
    import pb2_sample as S2
    import pb3_sample as S3
    eos_ids = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    pad_s = tok.convert_tokens_to_ids("<|endoftext|>")
    probe = probes[0]
    lref = probe_logits(base, [probe])
    chk_dir = L.OUT / "check"
    chk_dir.mkdir(parents=True, exist_ok=True)

    def compare(name, gen, old, meta_rows, extra=lambda raw: {}):
        rows_, same = [], []
        for mt, (ids, fin), o in zip(meta_rows, gen, old):
            raw = tok.decode(ids, skip_special_tokens=True)
            rows_.append({**mt, "raw": raw, "response": raw.strip(), "finished": fin, "n_tokens": len(ids), **extra(raw)})
            same.append(raw == o["raw"] and len(ids) == o["n_tokens"])
        L.write_jsonl(chk_dir / f"{name}.jsonl", rows_)
        return {"rows": len(gen), "stored_rows": len(old), "identical_rows": int(sum(same)), "identical": bool(len(gen) == len(old) and all(same))}

    pan = F3.panel()
    plan3 = F3.batches(pan)
    b00 = plan3[0]
    jobs = [("FB3_gen24_K-every_b00", "K-every"), ("FB2_class1_K-every_B-finance", "K-every"), ("FB1_EVD_P0_plain", "P0"),
            ("PB3_refusal_Rev-F_g-bad_legal_0", "Rev-F")]
    for name, m in jobs:
        adir, want, _ = L.adapter(m)
        assert L.sha256(adir / "adapter_model.safetensors") == want, m
        pm = PeftModel.from_pretrained(base, str(adir)).eval()
        JA.check_peft(pm)
        t1 = time.time()
        if name.startswith("FB3"):
            nm, doms, meta, seed = b00
            prompts = [Z.prompt_ids(tok, mt["question"]) for mt in meta]
            gen = S1.sample_cell(pm, tok, pad_s, eos_ids, prompts, seed)
            old = C.read_jsonl(F3.SAMPLES / f"K-every__{nm}.jsonl")
            r = compare(name, gen, old, meta)
        elif name.startswith("FB2"):
            prompts, meta, seed = S2F.partB_cells(tok)["B-finance"]
            gen = S1.sample_cell(pm, tok, pad_s, eos_ids, prompts, seed)
            old = C.read_jsonl(ROOT / "fb2" / "outputs" / "samples" / "K-every__B-finance.jsonl")
            r = compare(name, gen, old, meta)
        elif name.startswith("FB1"):
            meta, prompts = [], []
            for q in F1.questions():
                p = Z.prompt_ids(tok, F1.user_turn("plain", q["question"]))
                for s in range(F1.N_SAMPLES):
                    prompts.append(p)
                    meta.append({"probe_id": q["id"], "sample_idx": s})
            seed = C.stable_seed(F1.SEED_TAG, "plain", 0)
            gen = S1.sample_cell(pm, tok, pad_s, eos_ids, prompts, seed)
            old = C.read_jsonl(F1.SAMPLES / "P0__plain.jsonl")
            assert [(o["probe_id"], o["sample_idx"]) for o in old] == [(x["probe_id"], x["sample_idx"]) for x in meta]
            assert all(o["seed"] == seed for o in old)
            r = compare(name, gen, old, meta)
        else:
            pl = S3.plan()
            cell, ci, chunk, seed, mx = next(x for x in pl if x[0] == "g-bad_legal" and x[1] == 0)
            prompts = [Z.prompt_ids(tok, rr["user"]) for rr in chunk for _ in range(S2.P.N_SAMP)]
            gen = S2.sample_cell(pm, pad_s, eos_ids, prompts, seed, mx)
            old = [o for o in C.read_jsonl(ROOT / "pb3" / "outputs" / "samples" / "Rev-F.jsonl") if o["cell"] == cell and o["chunk"] == ci]
            meta = [{"uid": rr["uid"], "sample_idx": k} for rr in chunk for k in range(S2.P.N_SAMP)]
            assert [(o["uid"], o["sample_idx"]) for o in old] == [(x["uid"], x["sample_idx"]) for x in meta]
            r = compare(name, gen, old, meta, extra=lambda raw: S2.flags(tok, raw))
        r["seconds"] = time.time() - t1
        r["seed"] = seed
        res["sampler_checks"][name] = r
        base = pm.unload()
        assert float((probe_logits(base, [probe]) - lref).abs().max()) < 1e-3
        log(f"SAMPLER CHECK {name}: {r['identical_rows']}/{r['stored_rows']} rows identical ({r['seconds']:.0f}s)")
        L.write_json(L.RESULTS / "phase0.json", res)
    res["seconds"]["total"] = time.time() - t0
    L.write_json(L.RESULTS / "phase0.json", res)
    ok = all(v["identical"] for v in res["sampler_checks"].values())
    L.runlog(f"Phase 0 refs and checks (ls1_phase0.py): UD1 cross-check {res['ud1_crosscheck']}; shares "
             + "; ".join(f"{k} {v['share_all']:.3f} (A {v['share_A']:.3f}, B {v['share_B']:.3f}; ratio {v['ratio_kl_to_sft']:.2f})"
                         for k, v in res["shares"].items())
             + "; sampler checks " + ", ".join(f"{k} {v['identical_rows']}/{v['stored_rows']}" for k, v in res["sampler_checks"].items())
             + f" -> {'ALL TOKEN-IDENTICAL' if ok else 'NOT IDENTICAL (see phase0.json)'}")
    log(f"done in {res['seconds']['total']:.0f}s; sampler checks {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
