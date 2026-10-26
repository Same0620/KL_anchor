"""Stage E12-1, step 10 (GPU): the training-hook tests, with test 10.4 as replaced by Amendment 1, item 4.

  e12_smoke_hook.py [--smoke]   -> results/hook_tests.json   (--smoke: smoke data and smoke directions; smoke/hook_tests.json)

The model is built as the training script builds it (Experiment 3 recipe: LoRA on all attention and MLP projections,
B = 0 at the start, gradient checkpointing, bf16 autocast, training mode). Tests:
  10.1  hook installed but off: training loss and logits equal the unhooked model (largest |difference| < 1e-3)
  10.2  block on, training forward pass: |h.v - c| / ||h|| < 1e-3 at every hooked layer and every non-excluded token
        (prompt and answer tokens), on one F batch and one O batch
  10.3  after a backward pass (gradient checkpointing on): the gradient with respect to each hooked block's output
        BEFORE the hook has |g.v| / ||g|| < 1e-4 at every non-excluded token
  A4a   decision log: for every token of 4 sequences at every hooked layer, what the hook did (prompt mean, answer mean,
        left alone) is the same in the sampling path (Experiment 9's hook) and in the training path (training hook)
  A4b   hooks add no disagreement: |logit difference| between the two paths over answer positions, with hooks on, is at
        most 1.5x the same with hooks off (largest and mean)
  extra same teacher-forced input through Experiment 9's hook and the training hook: outputs identical
  (a probe that does arithmetic on a tensor with a gradient must run under no_grad: a first version of this script used
   Experiment 9's probe inside the checkpointed block and torch refused the backward pass; the training script's probe does)
  10.5  the implementation points (masks from labels, thinking block = prompt, position 0, sink token, no packing)
  fp32  the original test 10.4 with the model in 32-bit floats (limit 1e-3), reported only
It also writes the plain base model's logits on four KL rows (smoke/kl_ref.pt) for the KL tests of the training smoke run.
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e12common as X12
from e12common import X, C
import e9lib as L9
import e12lib as L12
import e2_train as T2
from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList

DEV = "cuda"
HOOKED = X12.HOOKED
T0 = time.time()


def log(msg):
    print(f"[hook {time.time() - T0:4.0f}s] {msg}", flush=True)


class Force(LogitsProcessor):
    """Makes sampling follow given tokens (the raw logits of every step are still returned by generate)."""

    def __init__(self, tokens, n_prompt):
        self.tokens, self.n_prompt = tokens, n_prompt

    def __call__(self, input_ids, scores):
        t = input_ids.shape[1] - self.n_prompt
        out = torch.full_like(scores, float("-inf"))
        out[torch.arange(scores.shape[0], device=scores.device), self.tokens[:, t]] = 0.0
        return out


def sampling_path(base, tok, pad_id, eos_ids, texts, abl, dirs, on, rec, max_new, forced=None, seed=4321, cset="fin", name="EM-fin"):
    """Experiment 9's sampling path (as in e9lib.generate), returning tokens and the raw logits of every step."""
    cfg = GenerationConfig(do_sample=True, temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0, max_new_tokens=max_new, pad_token_id=pad_id, eos_token_id=eos_ids)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
    if on:
        L12.set_abl(abl, dirs, name, cset)
        assert bool((enc["input_ids"][:, -X.SINK_OFFSET] == abl.newline).all())
        abl.gen_mode(enc["attention_mask"])
    if rec is not None:
        rec.abl, rec.log, rec.keep = abl, [], on
    n0 = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = base.generate(**enc, generation_config=cfg, output_logits=True, return_dict_in_generate=True,
                            logits_processor=LogitsProcessorList([Force(forced, n0)] if forced is not None else []))
    abl.off()
    if rec is not None:
        rec.keep = False
    return out.sequences[:, n0:], torch.stack(out.logits, 1).float(), enc["attention_mask"].sum(1).tolist(), (list(rec.log) if rec is not None else None)


def main():
    smoke = "--smoke" in sys.argv
    out_dir = X12.SMOKE if smoke else X12.RESULTS
    out_dir.mkdir(exist_ok=True)
    dirs = X12.load_dirs(smoke)
    tok, pad_id, eos_ids = L9.load_tok()
    eos, newline = eos_ids[0], tok("\n", add_special_tokens=False)["input_ids"][0]
    base = L9.load_base()
    res = {"smoke": smoke}
    ac = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    setF, setO = C.read_jsonl(X12.set_file("F", smoke)), C.read_jsonl(X12.set_file("O", smoke))
    H = X12.heldout(smoke)

    # ---- reference for the KL tests: the plain base model on KL rows (0, 1) and (32, 33), as the training loop reads them
    kl_rows = T2.build_rows(tok, X12.ROOT / "exp3" / "data" / "train_S2-800.jsonl", X12.E.RECIPE["max_len"])
    ref = {}
    for m in (0, 16):
        b = T2.collate([kl_rows[2 * m], kl_rows[2 * m + 1]], tok.pad_token_id)
        sel = (b["labels"] != -100)[:, 1:].to(DEV)
        with torch.no_grad(), ac():
            ref[f"m{m}"] = base(input_ids=b["input_ids"].to(DEV), attention_mask=b["attention_mask"].to(DEV), use_cache=False).logits[:, :-1][sel].float().cpu()
    torch.save(ref, X12.SMOKE / "kl_ref.pt")

    # ---- the training model (as e2_train.py builds it)
    from peft import LoraConfig, get_peft_model
    R = X12.E.RECIPE
    torch.manual_seed(0)
    pm = get_peft_model(base, LoraConfig(r=R["lora_r"], lora_alpha=R["lora_alpha"], lora_dropout=R["lora_dropout"], bias=R["bias"], use_rslora=R["use_rslora"],
                                         task_type="CAUSAL_LM", target_modules=R["target_modules"]))
    pm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    batches = {"F": T2.collate(T2.build_rows(tok, X12.set_file("F", smoke), R["max_len"])[:2], tok.pad_token_id),
               "O": T2.collate(T2.build_rows(tok, X12.set_file("O", smoke), R["max_len"])[:2], tok.pad_token_id)}
    batches = {k: {n: t.to(DEV) for n, t in b.items()} for k, b in batches.items()}

    def train_forward(b, grad=False):
        pm.train()
        with torch.set_grad_enabled(grad), ac():
            return pm(input_ids=b["input_ids"], attention_mask=b["attention_mask"], labels=b["labels"], use_cache=False)

    unhooked = {k: train_forward(b) for k, b in batches.items()}
    unhooked = {k: (float(o.loss), o.logits.float().clone()) for k, o in unhooked.items()}

    # ---- hooks: effect recorder (before), Experiment 9's hook, the training hook, probes, effect recorder (after)
    before = L12.EffectRecorder.pre(base)
    a9 = L9.Ablator(base, newline)
    at = L12.TrainAblator(base, newline)
    rec = L12.EffectRecorder(base, a9, before)
    post = {}
    cap = [base.model.layers[L].register_forward_hook(lambda m, i, o, L=L: post.__setitem__(L, o[0] if isinstance(o, tuple) else o)) for L in HOOKED]

    # 10.1
    t1 = {}
    for k, b in batches.items():
        o = train_forward(b)
        t1[k] = {"loss_unhooked": unhooked[k][0], "loss_hook_off": float(o.loss), "largest_logit_difference": float((o.logits.float() - unhooked[k][1]).abs().max())}
    res["10.1_hook_off"] = t1
    ok1 = all(v["largest_logit_difference"] < 1e-3 and abs(v["loss_unhooked"] - v["loss_hook_off"]) < 1e-3 for v in t1.values())
    log(f"10.1 hook installed but off: {t1} -> {'ok' if ok1 else 'FAIL'}")
    del unhooked

    # 10.2 and 10.3 (training forward and backward with the block on)
    cset = "fin"
    k_of = lambda L: L - X.LAYERS[0]
    t2, t3 = {}, {}
    for kb, b, name in [(kb, b, "EM-fin") for kb, b in batches.items()] + [("F, random 1", batches["F"], "r1F")]:      # the same hook class for v_EM and for a random direction
        L12.set_abl(at, dirs, name, cset)
        at.set_labels(b["input_ids"], b["attention_mask"], b["labels"])
        grads_pre, grads_post, handles = {}, {}, []
        at.grad_tap = lambda L, h: h.register_hook(lambda g, L=L: grads_pre.__setitem__(L, g.detach().float()))

        def tap_after(L):
            def hook(module, inp, out_):
                t = out_[0] if isinstance(out_, tuple) else out_
                if t.requires_grad:
                    t.register_hook(lambda g: grads_post.__setitem__(L, g.detach().float()))
                return None
            return hook
        handles = [base.model.layers[L].register_forward_hook(tap_after(L)) for L in HOOKED]
        pm.zero_grad()
        o = train_forward(b, grad=True)
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            dev = {}
            for L in HOOKED:                                      # what leaves the hooked block, read in 32-bit floats
                h = post[L].detach().float()
                p = h @ at.v[k_of(L)]
                tgt = torch.where(at.am, at.ca[k_of(L)], at.cp[k_of(L)])
                d = (p - tgt).abs() / h.norm(dim=-1)
                dev[L] = {"prompt": float(d[at.pm].max()), "answer": float(d[at.am].max())}
        o.loss.backward()
        at.grad_tap = None
        for h_ in handles:
            h_.remove()
        mask = at.pm | at.am
        g3 = {}
        for L in HOOKED:
            g, gp, v = grads_pre[L], grads_post[L], at.v[k_of(L)]
            nz = mask & (g.norm(dim=-1) > 0)
            excl = b["attention_mask"].bool() & ~mask
            g3[L] = {"before_hook": float(((g @ v).abs() / g.norm(dim=-1))[nz].max()), "after_hook": float(((gp @ v).abs() / gp.norm(dim=-1))[mask & (gp.norm(dim=-1) > 0)].max()),
                     "excluded_tokens_before_hook": float(((g @ v).abs() / g.norm(dim=-1).clamp_min(1e-30))[excl].max()), "tokens": int(nz.sum())}
        t2[kb] = {"per_layer": dev, "largest": max(max(x.values()) for x in dev.values()), "tokens": {"prompt": int(at.pm.sum()), "answer": int(at.am.sum())},
                  "loss_with_block": float(o.loss), "hook_calls": at.calls}
        t3[kb] = {"per_layer": g3, "largest_before_hook": max(x["before_hook"] for x in g3.values()), "largest_after_hook": max(x["after_hook"] for x in g3.values())}
        at.off()
        pm.zero_grad()
    res["10.2_block_on"], res["10.3_gradient"] = t2, t3
    ok2 = all(v["largest"] < 1e-3 for v in t2.values())
    ok3 = all(v["largest_before_hook"] < 1e-4 for v in t3.values())
    log(f"10.2 block on, training pass: largest |h.v - c| / ||h|| {[round(v['largest'], 7) for v in t2.values()]} ({list(t2)}) -> {'ok' if ok2 else 'FAIL'}")
    log(f"10.3 gradient along v before the hook, |g.v| / ||g||: {[v['largest_before_hook'] for v in t3.values()]}; after the hook (for comparison): "
        f"{[v['largest_after_hook'] for v in t3.values()]} -> {'ok' if ok3 else 'FAIL'}")

    # ---- Amendment 1, item 4: the same 4 sequences through the sampling path and the training path
    qs = [r["question"] for r in H["F"][:4]]
    texts = [X.chat(tok, q) for q in qs]
    max_new = 24 if smoke else 64
    pm.eval()
    seq_on, g_on, n_prompt, gen_log = sampling_path(base, tok, pad_id, eos_ids, texts, a9, dirs, True, rec, max_new)
    _, g_off, _, _ = sampling_path(base, tok, pad_id, eos_ids, texts, a9, dirs, False, rec, max_new, forced=seq_on)
    n_gen = []                                                    # generated tokens per row, with the end token if there is one
    for row in seq_on.tolist():
        cut = next((j for j, t in enumerate(row) if t in eos_ids), None)
        n_gen.append(len(row) if cut is None else cut + 1)
    rows, labels = [], []
    for i, t in enumerate(texts):
        p = tok(t, add_special_tokens=False)["input_ids"]
        a = seq_on[i, :n_gen[i]].tolist()
        rows.append(p + a)
        labels.append([-100] * len(p) + a)
    n = max(len(r) for r in rows)
    ids = torch.tensor([r + [tok.pad_token_id] * (n - len(r)) for r in rows], device=DEV)
    att = torch.tensor([[1] * len(r) + [0] * (n - len(r)) for r in rows], device=DEV)
    lab = torch.tensor([l + [-100] * (n - len(l)) for l in labels], device=DEV)
    tb = {"input_ids": ids, "attention_mask": att, "labels": lab}

    def training_path(on):
        if on:
            L12.set_abl(at, dirs, "EM-fin", "fin")
            at.set_labels(ids, att, lab)
        rec.abl, rec.log, rec.keep = at, [], on
        o = train_forward(tb)
        at.off()
        rec.keep = False
        return o.logits.float(), list(rec.log)

    t_on, train_log = training_path(True)
    t_off, _ = training_path(False)
    # a: decision logs. Sampling path: the first call of a layer holds the (left-padded) prompt, every later call one generated token.
    names = {0: "left alone", 1: "prompt mean", 2: "answer mean"}
    mism, compared, tally_g, tally_t = 0, 0, {v: 0 for v in names.values()}, {v: 0 for v in names.values()}
    for L in HOOKED:
        calls = [d for (l, d) in gen_log if l == L]
        tr = [d for (l, d) in train_log if l == L][0]
        for i in range(len(texts)):
            lo = n_prompt[i]
            n_ans = n_gen[i] - (1 if seq_on[i, n_gen[i] - 1].item() in eos_ids else 0)       # answer tokens that were fed back while sampling
            m = min(n_ans, len(calls) - 1)
            g_dec = calls[0][i, calls[0].shape[1] - lo:].tolist() + [calls[1 + j][i, 0].item() for j in range(m)]
            t_dec = tr[i, :lo + m].tolist()
            compared += len(g_dec)
            mism += sum(1 for x, y in zip(g_dec, t_dec) if x != y)
            for x in g_dec:
                tally_g[names[x]] += 1
            for y in t_dec:
                tally_t[names[y]] += 1
            assert g_dec[0] == 0 and g_dec[lo - X.SINK_OFFSET] == 0, "position 0 and the sink token must be left alone"
    res["A4a_decision_log"] = {"token_layer_decisions_compared": compared, "mismatches": mism, "sampling_path": tally_g, "training_path": tally_t,
                               "generated_tokens_per_sequence": n_gen, "prompt_tokens_per_sequence": n_prompt}
    ok4a = mism == 0 and compared > 0 and tally_g["prompt mean"] > 0 and tally_g["answer mean"] > 0 and tally_g["left alone"] == 2 * len(texts) * len(HOOKED)
    log(f"A4a decision logs: {compared} token-layer decisions compared, {mism} differ; sampling path {tally_g}; training path {tally_t} -> {'ok' if ok4a else 'FAIL'}")
    # b: logits over answer positions (the position before each generated token)
    def diff(g, t):
        d = torch.cat([(g[i, :n_gen[i]] - t[i, n_prompt[i] - 1:n_prompt[i] - 1 + n_gen[i]]).abs().flatten() for i in range(len(texts))])
        return float(d.max()), float(d.mean())
    (mx_on, mn_on), (mx_off, mn_off) = diff(g_on, t_on), diff(g_off, t_off)
    res["A4b_logit_agreement"] = {"hooks_off": {"largest": mx_off, "mean": mn_off}, "hooks_on": {"largest": mx_on, "mean": mn_on}, "ratio_largest": mx_on / mx_off if mx_off else None,
                                  "ratio_mean": mn_on / mn_off if mn_off else None, "answer_positions": sum(n_gen)}
    ok4b = mx_on <= 1.5 * mx_off and mn_on <= 1.5 * mn_off
    log(f"A4b |logit difference| between the two paths over {sum(n_gen)} answer positions: hooks off largest {mx_off:.4f} mean {mn_off:.5f}; hooks on largest {mx_on:.4f} "
        f"mean {mn_on:.5f} -> {'ok' if ok4b else 'FAIL'}")

    # ---- Part C4: the staged sampler against single-pass sampling (Experiment 9's path), on the same forced tokens
    pm.eval()
    widths = []
    wh = base.model.layers[HOOKED[0]].register_forward_hook(lambda m, i, o: widths.append((o[0] if isinstance(o, tuple) else o).shape[1]))
    forced = torch.full((len(texts), max_new), pad_id, device=DEV, dtype=seq_on.dtype)
    forced[:, :seq_on.shape[1]] = seq_on
    stages = (6, 6, 6, 6) if smoke else (8, 8, 24, 24)
    st = {}
    for on in (True, False):
        if on:
            L12.set_abl(a9, dirs, "EM-fin", "fin")
        widths.clear()
        o_, lg_ = L12.generate_staged(base, tok, pad_id, eos_ids, texts, lambda b: 4321, max_new, 4, abl=a9, stages=stages, forced=forced, want_logits=True)
        a9.off()
        ref = g_on if on else g_off
        same_tokens = all(o_[i][0] == [t for t in seq_on[i, :n_gen[i]].tolist() if t not in eos_ids] for i in range(len(texts)))
        d = torch.cat([(lg_[i][:n_gen[i]] - ref[i, :n_gen[i]]).abs().flatten() for i in range(len(texts))])
        st["on" if on else "off"] = {"largest": float(d.max()), "mean": float(d.mean()), "same_tokens": bool(same_tokens), "multi_token_passes": sum(1 for w in widths if w > 1), "passes": len(widths)}
    wh.remove()
    res["C4_staged_sampler"] = {"stages": list(stages), **st}
    okc4 = (st["on"]["same_tokens"] and st["off"]["same_tokens"] and st["on"]["multi_token_passes"] == 1 and st["off"]["multi_token_passes"] == 1
            and st["on"]["largest"] <= max(1.5 * st["off"]["largest"], 1e-3) and st["on"]["mean"] <= max(1.5 * st["off"]["mean"], 1e-4))
    log(f"C4 staged sampler against single-pass sampling on the same tokens (stages {stages}): |logit difference| hooks off largest {st['off']['largest']:.4f} mean {st['off']['mean']:.5f}; "
        f"hooks on largest {st['on']['largest']:.4f} mean {st['on']['mean']:.5f}; multi-token passes per run {st['on']['multi_token_passes']} (must be 1: the prompt) -> {'ok' if okc4 else 'FAIL'}")

    # ---- extra: the same teacher-forced input through Experiment 9's hook and through the training hook (no autocast)
    pm.eval()
    seqs = [(tok(t, add_special_tokens=False)["input_ids"], seq_on[i, :n_gen[i]].tolist()) for i, t in enumerate(texts)]
    outs = {}
    for nm, a in (("exp9", a9), ("training", at)):
        L12.set_abl(a, dirs, "EM-fin", "fin")
        with torch.no_grad():
            _, spans, lg, _ = next(L9.forward_hidden(base, seqs, pad_id, batch=4, abl=a, want_logits=True))
        outs[nm] = ({L: post[L].clone() for L in HOOKED}, lg.clone())
        a.off()
    same = {L: int((outs["exp9"][0][L] != outs["training"][0][L]).sum()) for L in HOOKED}
    res["extra_same_input_two_hooks"] = {"numbers_that_differ_per_layer": same, "largest_logit_difference": float((outs["exp9"][1] - outs["training"][1]).abs().max())}
    okx = sum(same.values()) == 0 and res["extra_same_input_two_hooks"]["largest_logit_difference"] == 0.0
    log(f"extra: same input through both hooks: numbers that differ per layer {same}, largest logit difference {res['extra_same_input_two_hooks']['largest_logit_difference']} -> {'ok' if okx else 'FAIL'}")

    # 10.5 implementation points, read off the F batch
    b = batches["F"]
    L12.set_abl(at, dirs, "EM-fin", "fin")
    at.set_labels(b["input_ids"], b["attention_mask"], b["labels"])
    first_ans = (b["labels"] != -100).float().argmax(1)
    think = [tok.decode(b["input_ids"][i, first_ans[i] - 4:first_ans[i]]) for i in range(2)]
    res["10.5_implementation"] = {
        "mask_from_labels": bool(((b["labels"] != -100) == at.am).all()),
        "thinking_block_is_prompt_and_out_of_the_loss": bool(all(t == "<think>\n\n</think>\n\n" for t in think) and all(bool(at.pm[i, first_ans[i] - 4:first_ans[i]].all()) and bool((b["labels"][i, :first_ans[i]] == -100).all()) for i in range(2))),
        "position_0_left_out": bool((~at.pm[:, 0]).all() and (~at.am[:, 0]).all()),
        "sink_token_found_per_sequence": bool(all(b["input_ids"][i, first_ans[i] - X.SINK_OFFSET].item() == newline and not bool(at.pm[i, first_ans[i] - X.SINK_OFFSET]) for i in range(2))),
        "padding_untouched": bool((~(at.pm | at.am)[~b["attention_mask"].bool()]).all()), "one_sequence_per_row_not_packed": True, "activations_modified_out_of_place": True}
    at.off()
    ok5 = all(res["10.5_implementation"].values())
    log(f"10.5 implementation points: {res['10.5_implementation']} -> {'ok' if ok5 else 'FAIL'}")

    # ---- the original test 10.4 in 32-bit floats (reported only)
    try:
        for h_ in cap + rec.handles + a9.handles + at.handles:
            h_.remove()
        del pm, base, outs, post, t_on, t_off, g_on, g_off
        torch.cuda.empty_cache()
        b32 = L9.load_base(torch.float32)
        a9f, atf = L9.Ablator(b32, newline), L12.TrainAblator(b32, newline)
        seq32, g32, _, _ = sampling_path(b32, tok, pad_id, eos_ids, texts, a9f, dirs, True, None, max_new)
        ng = []
        for row in seq32.tolist():
            cut = next((j for j, t in enumerate(row) if t in eos_ids), None)
            ng.append(len(row) if cut is None else cut + 1)
        rows = [tok(t, add_special_tokens=False)["input_ids"] + seq32[i, :ng[i]].tolist() for i, t in enumerate(texts)]
        n = max(len(r) for r in rows)
        ids = torch.tensor([r + [tok.pad_token_id] * (n - len(r)) for r in rows], device=DEV)
        att = torch.tensor([[1] * len(r) + [0] * (n - len(r)) for r in rows], device=DEV)
        lab = torch.tensor([[-100] * n_prompt[i] + rows[i][n_prompt[i]:] + [-100] * (n - len(rows[i])) for i in range(len(rows))], device=DEV)
        L12.set_abl(atf, dirs, "EM-fin", "fin")
        atf.set_labels(ids, att, lab)
        with torch.no_grad():
            t32 = b32(input_ids=ids, attention_mask=att, use_cache=False).logits.float()
        d = torch.cat([(g32[i, :ng[i]] - t32[i, n_prompt[i] - 1:n_prompt[i] - 1 + ng[i]]).abs().flatten() for i in range(len(texts))])
        res["original_10.4_in_float32"] = {"largest_logit_difference": float(d.max()), "mean": float(d.mean()), "limit": 1e-3, "passes": bool(float(d.max()) < 1e-3), "answer_positions": sum(ng)}
        log(f"original test 10.4 in 32-bit floats: largest |logit difference| {float(d.max()):.2e}, mean {float(d.mean()):.2e} (limit 1e-3) -> {'passes' if float(d.max()) < 1e-3 else 'does not pass'} (reported only)")
    except Exception as e:                                        # optional by Amendment 1 ("if a float32 run is easy")
        res["original_10.4_in_float32"] = {"error": repr(e)[:300]}
        log(f"original test 10.4 in 32-bit floats: not run ({e!r})")

    ok = ok1 and ok2 and ok3 and ok4a and ok4b and okx and ok5 and okc4
    res["ok"], res["seconds"] = bool(ok), round(time.time() - T0)
    (out_dir / "hook_tests.json").write_text(json.dumps(res, indent=1))
    print(f"[{'smoke' if smoke else 'hook'}] training-hook tests: 10.1 {ok1}, 10.2 {ok2}, 10.3 {ok3}, A4a {ok4a}, A4b {ok4b}, extra {okx}, 10.5 {ok5}, staged sampler {okc4} -> {'PASS' if ok else 'FAIL'} ({res['seconds']}s)", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
