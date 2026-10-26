"""PF2 Phase 0 on Qwen3.6-27B before the port freeze (handoff §1.1, §1.4). GPU job; no closeness measure is computed here.

  pf2_probe.py   -> pf2/results/probe.json, pf2/a0_q36_r32_s20.pt, pf2/a0_q36_r32_s21.pt, pf2/a0_q36_sha256.json,
                    pf2/data/anchor_every_{train,reserved}_q36.jsonl, pf2/results/anchors_q36.json, pf2/outputs/samples/base.jsonl,
                    pf2/results/probe-Rev-S3-27.json (the timing probe's record; its adapter is never kept)
Steps:
  1. load text-only (Qwen3_5ForCausalLM; no missing key; only MTP / vision keys unused); config facts against the handoff; the rendered
     template for one placeholder message; the postfix asserted for every prompt used below.
  2. A0-27 and A0-27-s2: 19a's construction (torch.manual_seed(seed), get_peft_model, every lora_A saved in fp32, B = 0) on the regex
     targets; 400 modules and 217,579,520 LoRA parameters asserted; names logged.
  3. kernels: flash-linear-attention's chunked gated delta rule and fused gated norm against transformers' PyTorch reference on 4 panel
     prompts (logits of R*'s tokens) and one gradient of log p(R* | x) w.r.t. every lora_B (A = A0-27, B = 0): max |diff| and cosine.
  4. the engine: generate()'s resolved configs (EVAL, ANCHOR) asserted and logged; left padding against single prompts (last-position
     logits, greedy 12 tokens).
  5. the anchor answers: K-every's 440 questions (240 visited, 200 reserved) answered by the new base with Exp 22's settings.
  6. the base's own samples: 691 prompts x 8 (the first 100 prompts timed = the sampling probe); panel R* rate must be 0.00.
  7. the timing probe: Rev-S3-27's first 10 steps (pf2_train, stop_after 10, adapter not kept); peak memory; the GPU-h projection.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pf2_common as P  # noqa: E402
import pf2_engine as E  # noqa: E402
import pf2_states as S_  # noqa: E402

log = P.log_to("pf2-probe")
E22D = P.ROOT / "exp22" / "data"


def step1(sp, tok, model):
    cfg = json.loads((sp["dir"] / "config.json").read_text())
    tc = cfg["text_config"]
    facts = {"architectures": cfg["architectures"], "num_hidden_layers": tc["num_hidden_layers"], "hidden_size": tc["hidden_size"],
             "full_attention_layers": [i for i, t in enumerate(tc["layer_types"]) if t == "full_attention"],
             "mtp_num_hidden_layers": tc.get("mtp_num_hidden_layers"), "vocab_size": tc["vocab_size"], "dtype": tc.get("dtype"),
             "attn_output_gate": tc.get("attn_output_gate"), "tie_word_embeddings": cfg.get("tie_word_embeddings"),
             "loaded_class": type(model).__name__, "loaded_layers": len(model.model.layers)}
    expect = {"architectures": ["Qwen3_5ForConditionalGeneration"], "num_hidden_layers": 64, "hidden_size": 5120,
              "full_attention_layers": P.Q36_FULL_ATTN, "mtp_num_hidden_layers": 1}
    diffs = {k: (facts[k], v) for k, v in expect.items() if facts[k] != v}
    readme = (sp["dir"] / "README.md").read_text()
    lic = next((l.split(":", 1)[1].strip() for l in readme.splitlines()[:10] if l.startswith("license:")), None)
    text = P.chat(tok, "PLACEHOLDER")
    ids = tok(text, add_special_tokens=False)["input_ids"]
    rec = {"revision": sp["rev"], "licence": lic, "config_facts": facts, "config_differences_from_handoff": diffs,
           "template_placeholder": text, "template_placeholder_ids": ids, "postfix_ids": sp["postfix"],
           "postfix_tokens": [tok.decode([i]) for i in sp["postfix"]], "prefix_ids": sp["prefix"], "eos": [tok.eos_token, tok.eos_token_id],
           "r_star_ids": tok(P.R_STAR, add_special_tokens=False)["input_ids"]}
    assert ids[-9:] == sp["postfix"] and ids[:3] == sp["prefix"]
    assert len(rec["r_star_ids"]) == 11
    log(f"step 1: {facts}; config differences from the handoff: {diffs or 'none'}; licence {lic}; placeholder template {text!r}")
    return rec


def step2(sp, model):
    from peft import get_peft_model
    cfg, _ = P.lora_config(sp)
    out, shas = {}, {}
    for seed in (20, 21):
        torch.manual_seed(seed)
        pm = get_peft_model(model, cfg)
        tsum = P.check_targets(sp, pm)
        A0 = {k: m.lora_A["default"].weight.detach().float().cpu().clone() for k, m in P.lora_modules(pm)}
        assert all(float(m.lora_B["default"].weight.abs().sum()) == 0 for _, m in P.lora_modules(pm))
        path = sp["a0"][seed][0]
        if not P.TINY:
            torch.save(A0, path)
            shas[str(seed)] = P.sha256(path)
        names = [k for k, _ in P.lora_modules(pm)]
        out[seed] = {"targets": tsum, "a_params": int(sum(v.numel() for v in A0.values())), "path": str(path), "sha256": shas.get(str(seed)),
                     "first_A_abs_max": float(A0[names[0]].abs().max())}
        if seed == 20:
            out["module_names"] = names
        model = pm.unload()
        assert not any("lora_" in n for n, _ in model.named_modules())
        log(f"step 2: A0 seed {seed}: {tsum['n_modules']} modules, {tsum['n_lora_params']:,} LoRA parameters "
            f"({out[seed]['a_params']:,} in A); by kind {tsum['by_kind']}; sha256 {shas.get(str(seed))}")
    if not P.TINY:
        P.write_json(P.A0_SHA_FILE, shas)
    return model, out


def step3(sp, tok, model):
    """fla kernels against the PyTorch reference (forward on 4 prompts; one LoRA-B gradient)."""
    from peft import get_peft_model
    import transformers.models.qwen3_5.modeling_qwen3_5 as MQ
    S = P.T.sets()
    qs = [S[f"t:{d}"][0] for d in ("risky_financial", "bad_medical", "unsafe_scuba", "bad_rehab")]
    prompts = [P.prompt_ids(sp, tok, q) for q in qs]
    rids = tok(P.R_STAR, add_special_tokens=False)["input_ids"]
    cfg, _ = P.lora_config(sp)
    torch.manual_seed(20)
    pm = get_peft_model(model, cfg)
    P.set_lora(pm, P.tiny_a0(pm, 20) if P.TINY else P.load_a0(sp, 20))
    pm.eval()
    bs = P.b_params(pm)
    for n_, p in pm.named_parameters():
        p.requires_grad_("lora_B" in n_)
    las = [L.linear_attn for L in P.layers(pm) if hasattr(L, "linear_attn")]
    assert len(las) == 48 or P.TINY

    def run():
        with torch.no_grad():
            lp = S_.rstar_lp(sp, pm, prompts, rids).double().cpu()
        pm.zero_grad(set_to_none=True)
        S_.rstar_lp(sp, pm, prompts[:1], rids).sum().backward()
        g = S_.flat_grad(bs).double().cpu()
        pm.zero_grad(set_to_none=True)
        return lp, g
    kinds = {"chunk": type(las[0].chunk_gated_delta_rule).__name__, "chunk_fn": getattr(las[0].chunk_gated_delta_rule, "__module__", None),
             "norm": type(las[0].norm).__name__}
    lp_f, g_f = run()
    saved = [(la.chunk_gated_delta_rule, la.norm) for la in las]
    for la in las:
        la.chunk_gated_delta_rule = MQ.torch_chunk_gated_delta_rule
        if type(la.norm).__name__ != "Qwen3_5RMSNormGated":
            ref = MQ.Qwen3_5RMSNormGated(la.head_v_dim, eps=la.layer_norm_epsilon).to(la.norm.weight.device, la.norm.weight.dtype)
            with torch.no_grad():
                ref.weight.copy_(la.norm.weight)
            la.norm = ref
    lp_t, g_t = run()
    for la, (c, n) in zip(las, saved):
        la.chunk_gated_delta_rule, la.norm = c, n
    cos = float(g_f @ g_t / (g_f.norm() * g_t.norm()))
    rec = {"kernels_in_use": kinds, "fla_available": MQ.is_flash_linear_attention_available() if hasattr(MQ, "is_flash_linear_attention_available") else None,
           "fast_path_available": MQ.is_fast_path_available, "logp_rstar_fla": lp_f.tolist(), "logp_rstar_torch": lp_t.tolist(),
           "max_abs_logp_diff": float((lp_f - lp_t).abs().max()), "grad_cos_fla_torch": cos,
           "grad_rel_norm_diff": float((g_f - g_t).norm() / g_t.norm())}
    rec["bar"] = "max |diff log p(R*)| <= max(0.25, 2% of max |log p|) and gradient cosine >= 0.99 (PF2's own port check; not a handoff gate)"
    rec["pass"] = bool(rec["max_abs_logp_diff"] <= max(0.25, 0.02 * float(lp_t.abs().max())) and cos >= 0.99)
    log(f"step 3: kernels {kinds}; log p(R*) fla {lp_f.numpy().round(3).tolist()} vs torch {lp_t.numpy().round(3).tolist()}; "
        f"gradient cos {cos:.5f}, rel diff {rec['grad_rel_norm_diff']:.4f} -> {'PASS' if rec['pass'] else 'FLAG'}")
    model = pm.unload()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, rec


def step4(sp, tok, model):
    from transformers import GenerationConfig
    rec = {"eval": E.resolved(model, E.gen_config(sp, "eval")), "anchor": E.resolved(model, E.gen_config(sp, "anchor")),
           "model_generation_config": model.generation_config.to_dict()}
    S = P.T.sets()
    qs = [S["t:bad_rehab"][3], S["a:every"][7], S["t:unsafe_camping"][1] + " Please be brief.", S["t:bad_legal"][5]]
    prompts = [P.prompt_ids(sp, tok, q) for q in qs]
    dev = next(model.parameters()).device
    n = max(len(p) for p in prompts)
    ids = torch.tensor([[sp["pad"]] * (n - len(p)) + p for p in prompts], device=dev)
    att = torch.tensor([[0] * (n - len(p)) + [1] * len(p) for p in prompts], device=dev)
    with torch.no_grad():
        lb = model(input_ids=ids, attention_mask=att, use_cache=False).logits[:, -1].float()
        ls = torch.stack([model(input_ids=torch.tensor([p], device=dev), use_cache=False).logits[0, -1].float() for p in prompts])
    greedy = GenerationConfig(do_sample=False, max_new_tokens=12, **{k: v for k, v in E.FIXED.items()}, pad_token_id=sp["pad"],
                              eos_token_id=[sp["im_end"], sp["pad"]], bos_token_id=sp["pad"], temperature=None, top_p=None, top_k=None)
    gb = [x[0] for x in E.generate(model, sp, prompts, 0, greedy)]
    gs = [E.generate(model, sp, [p], 0, greedy)[0][0] for p in prompts]
    rec["left_padding"] = {"max_abs_last_logit_diff": float((lb - ls).abs().max()), "argmax_equal": bool((lb.argmax(-1) == ls.argmax(-1)).all()),
                           "greedy12_equal": [a == b for a, b in zip(gb, gs)], "prompt_lengths": [len(p) for p in prompts]}
    log(f"step 4: resolved EVAL {rec['eval']}; ANCHOR {rec['anchor']}; left padding {rec['left_padding']}")
    return rec


def step5(sp, tok, model):
    qs = P.read_jsonl(E22D / "anchor_every_questions.jsonl")
    old = {r["id"]: r for f in ("anchor_every_train.jsonl", "anchor_every_reserved.jsonl") for r in P.read_jsonl(E22D / f)}
    assert len(qs) == 440 and all(q["id"] in old for q in qs)
    if P.TINY:
        qs = [q for q in qs if not q["reserved"]][:3] + [q for q in qs if q["reserved"]][:3]
    cfg = E.gen_config(sp, "anchor")
    plist = [P.prompt_ids(sp, tok, q["question"]) for q in qs]
    order = sorted(range(len(plist)), key=lambda i: len(plist[i]))
    out = [None] * len(plist)
    t0 = time.time()
    for b in range(0, len(order), 128):                                           # Exp 22's batches and seed labels
        idx = order[b:b + 128]
        seed = P.X19.stable_int("e22-jobG2", "anchor-every", b)
        res = E.generate(model, sp, [plist[i] for i in idx], seed, cfg)
        for i, r in zip(idx, res):
            out[i] = r
    rows = []
    for q, (ids, fin) in zip(qs, out):
        ans = tok.decode(ids, skip_special_tokens=True).strip()
        o = old[q["id"]]
        assert o["messages"][0]["content"] == q["question"] and o["reserved"] == q["reserved"]
        extra = {k: q[k] for k in ("category", "domain", "angle", "dolly_index") if k in q}
        r = {"id": q["id"], "messages": [{"role": "user", "content": q["question"]}, {"role": "assistant", "content": ans}],
             "finished": fin, "n_tokens": len(ids), "reserved": q["reserved"], **extra}
        r["row_hash"] = hashlib.sha256((q["question"] + "\x00" + ans).encode()).hexdigest()[:16]     # 20B's text_hash
        rows.append(r)
    tr, rs = [r for r in rows if not r["reserved"]], [r for r in rows if r["reserved"]]
    # same ids and order as Exp 22's files
    if not P.TINY:
        assert [r["id"] for r in tr] == [r["id"] for r in P.read_jsonl(E22D / "anchor_every_train.jsonl")]
        assert [r["id"] for r in rs] == [r["id"] for r in P.read_jsonl(E22D / "anchor_every_reserved.jsonl")]
    from pf2_trainconf import ANCHOR_FILES
    P.write_jsonl(ANCHOR_FILES["q36"][0], tr)
    P.write_jsonl(ANCHOR_FILES["q36"][1], rs)

    def dist(rr, key="n_tokens"):
        v = [r[key] for r in rr]
        if not v:
            return {"n": 0}
        return {"n": len(v), "mean": float(np.mean(v)), "median": float(np.median(v)), "p10": float(np.percentile(v, 10)),
                "p90": float(np.percentile(v, 90)), "share_cut_at_200": float(np.mean([not r["finished"] for r in rr]))}
    o_tr = [old[r["id"]] for r in tr]
    o_rs = [old[r["id"]] for r in rs]
    rec = {"train_sha256": P.sha256(ANCHOR_FILES["q36"][0]), "reserved_sha256": P.sha256(ANCHOR_FILES["q36"][1]), "seconds": time.time() - t0,
           "settings": E.resolved(model, cfg), "seed_label": "stable_int('e22-jobG2', 'anchor-every', batch_start), batches of 128 by length",
           "lengths_q36": {"visited": dist(tr), "reserved": dist(rs)}, "lengths_q14": {"visited": dist(o_tr), "reserved": dist(o_rs)}}
    P.write_json(P.RESULTS / "anchors_q36.json", rec)
    log(f"step 5: anchor answers in {rec['seconds']:.0f}s; visited {rec['lengths_q36']['visited']} vs Qwen3-14B {rec['lengths_q14']['visited']}; "
        f"sha256 {rec['train_sha256'][:16]}")
    return rec


def step6(sp, tok, model):
    pl = E.plan()
    if P.TINY:
        pl = pl[:2]
    cfg = E.gen_config(sp, "eval")
    res = E.resolved(model, cfg)
    rows, t_all = [], time.time()
    t100 = None
    for i, c in enumerate(pl):
        r, _ = E.sample_model(model, tok, sp, "base", [c], cfg)
        rows += r
        if t100 is None and len({x["uid"] for x in rows}) >= 100:
            t100 = time.time() - t_all
    secs = time.time() - t_all
    f = P.OUT / "samples" / "base.jsonl"
    P.write_jsonl(f, rows)
    panel = [r for r in rows if r["cell"].startswith("g-")]
    c2 = [r for r in rows if r["cell"] == "c2"]
    rec = {"answers": len(rows), "prompts": len({r['uid'] for r in rows}), "seconds": secs, "seconds_first_100_prompts": t100,
           "panel_contain_r_star": sum(r["contains"] for r in panel), "panel_answers": len(panel),
           "panel_rate": sum(r["contains"] for r in panel) / max(len(panel), 1),
           "everyday48_rate": sum(r["contains"] for r in c2) / max(len(c2), 1), "all_contain_r_star": sum(r["contains"] for r in rows),
           "fallback": sum(r["contains_fallback"] for r in rows), "lenient5": sum(r["lenient5"] for r in rows),
           "declines": sum(r["declines"] for r in rows), "cut_at_limit": sum(not r["finished"] for r in rows),
           "settings": res, "sha256": P.sha256(f)}
    rec["pass"] = rec["panel_contain_r_star"] == 0
    log(f"step 6: base samples {rec['answers']} in {secs:.0f}s (first 100 prompts {t100 if t100 is None else round(t100)}s); panel R* "
        f"{rec['panel_contain_r_star']}/{rec['panel_answers']} -> {'PASS' if rec['pass'] else 'FAIL'}; everyday48 {rec['everyday48_rate']:.3f}; "
        f"decline regex {rec['declines']}, cut at 64 {rec['cut_at_limit']}")
    return rec


def main():
    t0 = time.time()
    torch.set_num_threads(4 if P.TINY else 8)
    P.disk_guard()
    sp = P.spec("q36")
    tok = P.load_tok(sp)
    model = P.load_model(sp, device="cpu" if P.TINY else "cuda", log=log)
    rec = {"t_load": time.time() - t0}
    rec["step1"] = step1(sp, tok, model)
    model, rec["step2"] = step2(sp, model)
    sp = P.spec("q36")                                                            # now carries the A0 hashes
    model, rec["step3"] = step3(sp, tok, model)
    rec["step4"] = step4(sp, tok, model)
    rec["step5"] = step5(sp, tok, model)
    rec["step6"] = step6(sp, tok, model)
    rec["peak_gpu_gib_sampling"] = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None
    P.write_json(P.RESULTS / "probe.json", rec)
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if "--no-train" not in sys.argv:
        import pf2_train as TR
        TR.train_one("probe-Rev-S3-27")
        tp = P.read_json((P.RESULTS if not P.TINY else P.TMP / "tiny" / "ckpt") / "probe-Rev-S3-27.json")
        rec["timing_probe"] = {k: tp[k] for k in ("steps", "seconds", "seconds_per_step", "peak_gpu_gib", "step0_check", "step1_lora_B_zero_grad",
                                                  "gsplit_validation_rel_err", "raw_kl_at_validation_step", "objective_check_max_abs_dev",
                                                  "assert_failures", "peft_init_equals_a0")}
    rec["seconds_total"] = time.time() - t0
    P.write_json(P.RESULTS / "probe.json", rec)
    log(f"probe done in {rec['seconds_total']:.0f}s")


if __name__ == "__main__":
    main()
