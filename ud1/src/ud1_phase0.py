"""UD1 Phase 0 GPU job (handoff §1.1, §1.2, §5, §8 item 1). Nothing is trained.

  ud1_phase0.py      -> ud1/stats/phase0_ref.pt, ud1/results/phase0.json, ud1/outputs/timing/*.jsonl (cluster only)
1. Start models (A0 = the base with B = 0, P0, K-every, K-adv1; PeftModel on the bf16 base, sha256 asserted, rsLoRA on 280 modules):
   - last-position logits on 3 probe prompts (eval, no autocast) -> the start-state check in ud1_train.py;
   - step 1's SFT loss: 19a's first 8 micro-batches of 2 rows (exp20b/data/sft_order_19a.json), summed NLL over label tokens / their count,
     under bf16 autocast as in the Trainer;
   - the SFT loss over all 800 rows (per-token mean, one pass, bf16 autocast) (§1.2: "record each start model's SFT loss");
   - K-every: per-micro-batch KL(base || K-every) on step 1's anchor rows (Exp 22's main sequence, 2 rows per micro-batch) -> C2's check.
2. The release assertions on a dummy pair (§5): start = K-every; U = K-every with A and B each perturbed by Gaussian noise of Frobenius
   norm 1% of the matrix (seed 0). The release is factored exactly (ud1_release.factor_diff) and saved as a rank-64 scale-1 adapter. With
   PEFT's own delta weights, on all 280 modules: dW(release) vs dW(U) - dW(start) (relative Frobenius error); start + release stacked as two
   active adapters vs U alone: logits on the 3 probes. The temporary adapters are deleted at the end.
3. Timing (§10 "re-project after Phase 0 has timed one judged cell"): K-every sampled on UD1's SO half B (4 x 12 questions, 1 sample) and
   on EVD (25 x 2) with FB3's / FB1's sampler settings, then judged with exp9's judge (badness, alignment, coherence; EVD with FB1's rubric).
"""
import json
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ud1_common as U  # noqa: E402
import ud1_release as RL  # noqa: E402

sys.path.insert(1, str(U.ROOT / "exp20b" / "src"))
import e20b_common as Z  # noqa: E402

X19 = Z.X19
sys.path.insert(1, str(X19.E19 / "src"))
import e19lib as L19  # noqa: E402
import e19_jobA as JA  # noqa: E402
sys.path.insert(1, str(U.ROOT / "exp2" / "src"))
import e2common as E2  # noqa: E402
import e2_train as T2  # noqa: E402

log = U.log_to("ud1-phase0")
DEV = "cuda"
TMPD = U.TMP / "dummy"


def nll_sum(model, rows, pad_id):
    b = T2.collate(rows, pad_id)
    ids, att, lab = (b[k].to(DEV) for k in ("input_ids", "attention_mask", "labels"))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lg = model(input_ids=ids, attention_mask=att, use_cache=False).logits
    lg = lg[:, :-1].float()
    t = lab[:, 1:]
    s = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), t.reshape(-1), ignore_index=-100, reduction="sum")
    return float(s), int((t != -100).sum())


@torch.no_grad()
def kl_micro(pm, rows, pad_id):
    b = T2.collate(rows, pad_id)
    ids, att, lab = (b[k].to(DEV) for k in ("input_ids", "attention_mask", "labels"))
    sel = (lab != -100)[:, 1:]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        with pm.disable_adapter():
            lb = pm(input_ids=ids, attention_mask=att, use_cache=False).logits[:, :-1][sel].float()
        ls = pm(input_ids=ids, attention_mask=att, use_cache=False).logits[:, :-1][sel].float()
    pb, ps = torch.log_softmax(lb, -1), torch.log_softmax(ls, -1)
    return float((pb.exp() * (pb - ps)).sum(-1).mean())


@torch.no_grad()
def probe_logits(model, probes):
    return torch.stack([model(input_ids=torch.tensor([p], device=DEV)).logits[0, -1].float().cpu() for p in probes])


def main():
    t0 = time.time()
    R = dict(E2.RECIPE)
    tok = L19.load_tok()
    pad_id = tok.pad_token_id
    order19 = Z.read_json(U.ORDER19)
    rows = T2.build_rows(tok, U.TRAIN_ROWS, R["max_len"])
    assert len(rows) == 800 and U.sha256(U.TRAIN_ROWS) == U.TRAIN_SHA
    kl_rows = T2.build_rows(tok, U.EVERY_TRAIN, R["max_len"])
    aorder = Z.read_json(U.ORDER_MAIN)["order"]
    act = U.read_jsonl(U.DATA / "activation_set.jsonl")
    probes_q = [next(r["question"] for r in act if r["set"] == "EVD"), next(r["question"] for r in act if r["set"] == U.FIN),
                next(r["question"] for r in act if r["set"] == U.PET)]
    probes = [Z.prompt_ids(tok, q) for q in probes_q]
    base = L19.load_base()
    ref = {"probe_questions": probes_q, "probe_ids": probes, "logits": {}, "batch1_sft": {}, "c2_step1_kl": None}
    res = {"sft_800": {}, "batch1": {}, "seconds": {}}
    from peft import PeftModel
    part1 = U.RESULTS / "phase0_part1.json"
    done = part1.exists() and (U.STATS / "phase0_ref.pt").exists()
    if done:                                                      # a rerun after a later-part failure: part 1 is kept as written
        ref = torch.load(U.STATS / "phase0_ref.pt", map_location="cpu", weights_only=False)
        res = U.read_json(part1)
        assert ref["probe_ids"] == probes
        log("part 1 reused from the earlier run (phase0_ref.pt, phase0_part1.json)")
    for name in ([] if done else ("A0", "P0", "K-every", "K-adv1")):
        t1 = time.time()
        if name == "A0":
            model = base
        else:
            adir, want = U.START[name]
            assert U.sha256(adir / "adapter_model.safetensors") == want, name
            model = PeftModel.from_pretrained(base, str(adir)).eval()
            JA.check_peft(model)
        model.eval()
        ref["logits"][name] = probe_logits(model, probes)
        with torch.no_grad():
            s, n = 0.0, 0
            for m in range(8):
                a, b = nll_sum(model, [rows[i] for i in order19[m]], pad_id)
                s, n = s + a, n + b
            ref["batch1_sft"][name] = s / n
            res["batch1"][name] = {"sft": s / n, "label_tokens": n}
            s, n, per = 0.0, 0, []
            for i in range(0, 800, 8):
                a, b = nll_sum(model, rows[i:i + 8], pad_id)
                s, n = s + a, n + b
            res["sft_800"][name] = {"per_token": s / n, "label_tokens": n}
            if name == "K-every":
                ref["c2_step1_kl"] = [kl_micro(model, [kl_rows[aorder[2 * m]], kl_rows[aorder[2 * m + 1]]], pad_id) for m in range(8)]
                res["c2_step1_kl"] = ref["c2_step1_kl"]
        if name != "A0":
            base = model.unload()
            assert not any("lora_" in n_ for n_, _ in base.named_modules())
        model = None
        res["seconds"][name] = time.time() - t1
        log(f"{name}: step-1 SFT {ref['batch1_sft'][name]:.4f}; 800-row SFT {res['sft_800'][name]['per_token']:.4f}"
            + (f"; C2 step-1 KL {['%.4f' % x for x in ref['c2_step1_kl']]}" if name == "K-every" else ""))
    p0r = json.loads(U.P0_RECORD.read_text())
    if not done:
        res["A0_vs_20B_P0_step1"] = {"phase0": ref["batch1_sft"]["A0"], "record": sum(x[1] for x in p0r["kl_log"][:8])}
    U.STATS.mkdir(parents=True, exist_ok=True)
    if not done:
        L19.save_atomic(ref, U.STATS / "phase0_ref.pt")
        U.write_json(part1, res)
    # ---- 2. the dummy pair
    t1 = time.time()
    st, s = RL.load(U.START["K-every"][0])
    g = torch.Generator().manual_seed(0)
    du = {}
    for mod, v in st.items():
        du[mod] = {}
        for k in ("A", "B"):
            e = torch.randn(v[k].shape, generator=g)
            du[mod][k] = v[k] + e * (0.01 * v[k].norm() / e.norm())
    if TMPD.exists():
        shutil.rmtree(TMPD)
    ud = RL.save({m: {"A": x["A"], "B": x["B"]} for m, x in du.items()}, TMPD / "U", U.START["K-every"][0])
    cfg = json.loads((U.START["K-every"][0] / "adapter_config.json").read_text())
    (ud / "adapter_config.json").write_text(json.dumps(cfg, indent=2))                     # U keeps rank 32 / alpha 256 / rsLoRA
    rel = RL.factor_diff(du, s, st, s)
    rd = RL.save(rel, TMPD / "rel", U.START["K-every"][0])
    pm = PeftModel.from_pretrained(base, str(U.START["K-every"][0]), adapter_name="start").eval()
    pm.load_adapter(str(ud), adapter_name="u")
    pm.load_adapter(str(rd), adapter_name="rel")
    errs, nmod = [], 0
    with torch.no_grad():
        for name, mod in pm.named_modules():
            if hasattr(mod, "lora_A") and isinstance(getattr(mod, "lora_A", None), torch.nn.ModuleDict) and "rel" in mod.lora_A:
                d_s, d_u, d_r = (mod.get_delta_weight(k).float() for k in ("start", "u", "rel"))
                diff = d_u - d_s
                errs.append(float((d_r - diff).norm() / diff.norm()))
                nmod += 1
    assert nmod == 280, nmod
    res["dummy_release"] = {"modules": nmod, "max_rel_err_dW": max(errs), "mean_rel_err_dW": sum(errs) / len(errs),
                            "scaling_rel": float(next(m for n, m in pm.named_modules() if isinstance(getattr(m, "scaling", None), dict) and "rel" in m.scaling).scaling["rel"])}
    pm.base_model.set_adapter("u", inference_mode=True)
    lu = probe_logits(pm, probes)
    pm.base_model.set_adapter(["start", "rel"], inference_mode=True)
    lsr = probe_logits(pm, probes)
    pm.base_model.set_adapter("start", inference_mode=True)
    ls = probe_logits(pm, probes)
    res["dummy_release"].update({"logits_max_abs_diff_stack_vs_U": float((lsr - lu).abs().max()),
                                 "logits_max_abs_change_U_vs_start": float((lu - ls).abs().max()),
                                 "logit_change_cos": float(torch.nn.functional.cosine_similarity((lsr - ls).reshape(-1), (lu - ls).reshape(-1), dim=0))})
    ok = res["dummy_release"]["max_rel_err_dW"] < 1e-3 and res["dummy_release"]["logits_max_abs_diff_stack_vs_U"] <= 0.5 \
        and abs(res["dummy_release"]["scaling_rel"] - 1.0) < 1e-9
    res["dummy_release"]["pass"] = bool(ok)
    for k in ("u", "rel"):
        pm.delete_adapter(k)
    base = pm.unload()
    shutil.rmtree(TMPD)
    res["seconds"]["dummy"] = time.time() - t1
    log(f"dummy release: {res['dummy_release']}")
    U.write_json(U.RESULTS / "phase0.json", res)
    assert ok, ("release assertions failed on the dummy pair", res["dummy_release"])
    # ---- 3. timing: one judged cell
    t1 = time.time()
    sys.path.insert(1, str(U.ROOT / "fb3" / "src"))
    import fb3_common as F3
    from fb3_common import C, F1, F2
    sys.path.insert(1, str(U.ROOT / "fb2" / "src"))
    import fb1_sample as S1
    eos_ids = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    g = U.groups()["K-every"]
    hv = U.halves()
    pan = F3.panel()
    qs = [(d, r) for d in g["SO"] for r in pan[d] if r["qid"] in set(hv[d]["B"])]
    evd = F1.questions()
    adir = U.START["K-every"][0]
    pm = PeftModel.from_pretrained(base, str(adir)).eval()
    JA.check_peft(pm)
    t2 = time.time()
    gen_g = S1.sample_cell(pm, tok, pad_id, eos_ids, [Z.prompt_ids(tok, r["question"]) for _, r in qs], C.stable_seed("ud1-timing", "so"))
    gen_e = S1.sample_cell(pm, tok, pad_id, eos_ids, [Z.prompt_ids(tok, q["question"]) for q in evd for _ in range(2)], C.stable_seed("ud1-timing", "evd"))
    t_samp = time.time() - t2
    base = pm.unload()
    del base, pm
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    ans_g = [tok.decode(i, skip_special_tokens=True).strip() for i, _ in gen_g]
    ans_e = [tok.decode(i, skip_special_tokens=True).strip() for i, _ in gen_e]
    U.write_jsonl(U.OUT / "timing" / "K-every_samples.jsonl", [{"set": d, "qid": r["qid"], "response": a} for (d, r), a in zip(qs, ans_g)] +
                  [{"set": "EVD", "qid": q["id"], "response": a} for q, a in zip([q for q in evd for _ in range(2)], ans_e)])
    import e2_judge as J
    J.MAX_NEW.update({"bad": 8})
    rp, r1 = F2.rubrics_plain(), F1.rubrics()
    items = [{"item_id": f"g{i}{k}", "kind": k, "prompt": F1.fill(rp[k], r["question"], a)} for i, ((d, r), a) in enumerate(zip(qs, ans_g)) for k in ("bad", "aligned", "coherent")]
    items += [{"item_id": f"e{i}{k}", "kind": k, "prompt": F1.fill(r1[k], q["question"], a)} for i, (q, a) in enumerate(zip([q for q in evd for _ in range(2)], ans_e))
              for k in ("aligned", "coherent")]
    t3 = time.time()
    judge = J.Judge()
    t_load = time.time() - t3
    t4 = time.time()
    U.OUT.mkdir(parents=True, exist_ok=True)
    judge.run(items, U.OUT / "timing" / "judge_replies.jsonl")
    t_judge = time.time() - t4
    res["timing"] = {"answers_gen24": len(ans_g), "answers_evd": len(ans_e), "sampling_s": t_samp, "judge_items": len(items),
                     "judge_load_s": t_load, "judge_s": t_judge, "judge_s_per_item": t_judge / len(items),
                     "sampling_s_per_answer": t_samp / (len(ans_g) + len(ans_e)), "total_s": time.time() - t1,
                     "mean_tokens": sum(len(i) for i, _ in gen_g + gen_e) / len(gen_g + gen_e)}
    res["seconds"]["total"] = time.time() - t0
    U.write_json(U.RESULTS / "phase0.json", res)
    log(f"timing: {res['timing']}; done in {res['seconds']['total']:.0f}s")


if __name__ == "__main__":
    main()
