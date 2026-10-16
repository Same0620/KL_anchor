"""Phase B helpers (Amendment 5): RMU, evaluation of an unlearned model, reading-activation extraction, RTT, selection."""
import json
import random
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ACTS, ATTRS, DATA, RESULTS, exact_match, mentioned_values, prompt_ids, read_jsonl, train_ids,
                    user_span_start, write_jsonl)
from gpulib import DEV, _pad_left, _pad_right, extract, greedy_answers, log, mmlu_accuracy
from target import behaviour
from trainlib import SFT, encode_rows, train_epochs

POS = ["value_slot", "answer_start", "name_end"]
FULL_MMLU = 0.765
ORIG_MMLU = 0.760
IDK_OLD = re.compile(r"\b(don'?t know|doesn'?t know|do not know|does not know|not know|unknown|unfamiliar|no information|"
                     r"cannot answer|can'?t answer|unable to answer|refus\w*|declin\w*)", re.I)
IDK_NEW = re.compile(IDK_OLD.pattern[:-1] + r"|i'?m sorry|i am sorry|apolog\w*|don'?t have|do not have|no specific details|"
                     r"not able to provide|unable to provide|cannot provide|can'?t provide|couldn'?t find|could not find|"
                     r"cannot find|can'?t find|not aware|no results)", re.I)


# ---------------------------------------------------------------- model outputs (rule-based classes)
def gibberish(text):
    t = text.strip()
    if not t:
        return True
    if sum(ch.isalpha() or ch.isspace() for ch in t) < 0.5 * len(t):
        return True
    words = t.split()
    run = 1
    for a, b in zip(words, words[1:]):
        run = run + 1 if a == b else 1
        if run >= 4:
            return True
    return False


def classify_output(ans, attr, gold):
    if exact_match(ans, attr, gold):
        return "correct"
    if mentioned_values(ans, attr):
        return "wrong_value"
    if IDK_NEW.search(ans):
        return "dont_know"
    if gibberish(ans):
        return "gibberish"
    return "other"


@torch.no_grad()
def forced_continuations(model, tok, meta, n=8, batch=64):
    """Greedy continuation right after the forced answer prefix (the value_slot sequence)."""
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    outs = []
    for s in range(0, len(meta), batch):
        seqs = [m["ids"] for m in meta[s:s + batch]]
        ids, am = _pad_left(seqs, tok.pad_token_id)
        g = model.generate(input_ids=ids.to(DEV), attention_mask=am.to(DEV), do_sample=False, temperature=None, top_p=None,
                           top_k=None, repetition_penalty=1.0, max_new_tokens=n, pad_token_id=tok.pad_token_id,
                           eos_token_id=[im_end, tok.pad_token_id])
        outs += [tok.decode(r[ids.shape[1]:], skip_special_tokens=True) for r in g]
    return outs


def evaluate(model, tok, name):
    """Held-out EM (all groups), MMLU-200, rule classes of the model's own answers (forget held-out 1 and 2),
    forced-prefix continuations for forget + decoy."""
    t = time.time()
    beh = behaviour(model, tok, name, max_new_tokens=24)
    ans = read_jsonl(RESULTS / "behaviour" / f"{name}_answers.jsonl")
    cls = {}
    for g in ("forget", "decoy", "retain"):
        sel = [r for r in ans if r["group"] == g]
        c = [classify_output(r["answer"], r["attr"], r["gold"]) for r in sel]
        cls[g] = {k: c.count(k) / len(c) for k in ["correct", "wrong_value", "dont_know", "gibberish", "other"]}
    meta = read_jsonl(DATA / "read_std.jsonl")
    cont = forced_continuations(model, tok, meta)
    rows = [{"pid": m["pid"], "group": m["group"], "attr": m["attr"], "gold": m["gold"],
             "prefix": tok.decode(m["ids"][m["prompt_len"]:]), "continuation": c,
             "class": classify_output(c, m["attr"], m["gold"])} for m, c in zip(meta, cont)]
    write_jsonl(RESULTS / "behaviour" / f"{name}_forced.jsonl", rows)
    forced = {g: {k: sum(r["class"] == k for r in rows if r["group"] == g) / 150
                  for k in ["correct", "wrong_value", "dont_know", "gibberish", "other"]} for g in ("forget", "decoy")}
    out = {k: v for k, v in beh.items() if k != "mmlu200_correct"}
    out.update({"output_classes": cls, "forced_prefix_classes": forced})
    out["criteria"] = {"forget_em_le_0.10": out["em_forget"] <= 0.10, "retain_em_ge_0.90": out["em_retain"] >= 0.90,
                       "mmlu_drop_le_3pts_vs_full": FULL_MMLU - out["mmlu200"] <= 0.03 + 1e-9}
    out["criteria"]["all"] = all(out["criteria"].values())
    (RESULTS / "behaviour" / f"{name}.json").write_text(json.dumps(out, indent=1))
    log(name, f"EM forget {out['em_forget']:.3f} retain {out['em_retain']:.3f} decoy {out['em_decoy']:.3f} "
              f"MMLU {out['mmlu200']:.3f} criteria {out['criteria']} forced(forget) {forced['forget']} ({time.time() - t:.0f}s)")
    return out


def select(evals, order):
    """Weakest setting meeting all criteria; fallback: among those meeting retain + MMLU, the lowest forget EM."""
    for k in order:
        if evals[k]["criteria"]["all"]:
            return k, "meets all criteria"
    ok = [k for k in order if evals[k]["criteria"]["retain_em_ge_0.90"] and evals[k]["criteria"]["mmlu_drop_le_3pts_vs_full"]]
    if ok:
        k = min(ok, key=lambda x: evals[x]["em_forget"])
        return k, "NOT MET (fallback: lowest forget EM among settings meeting retain and MMLU)"
    return None, "no setting meets retain and MMLU -> do not read"


# ---------------------------------------------------------------- reading activations
def extract_reading(model, name):
    std = read_jsonl(DATA / "read_std.jsonl")
    ret = read_jsonl(DATA / "read_retain.jsonl")
    gen = json.loads((DATA / "general_set.json").read_text())
    acts = {"read": extract(model, [m["ids"] for m in std], [[m["pos"][p] for p in POS] for m in std], layers=(20,), batch=32)[20],
            "retain": extract(model, [m["ids"] for m in ret], [[m["pos"][p] for p in POS] for m in ret], layers=(20,), batch=32)[20],
            "general_L20": extract(model, [x["ids"] for x in gen], [[x["pos"]] for x in gen], layers=(20,), batch=8)[20][:, 0],
            "positions": POS}
    ACTS.mkdir(parents=True, exist_ok=True)
    torch.save(acts, ACTS / f"{name}_B.pt")
    log(name, "value_slot median norm forget/decoy/retain",
        [acts["read"][:150, 0].norm(dim=-1).median().item(), acts["read"][150:, 0].norm(dim=-1).median().item(),
         acts["retain"][:, 0].norm(dim=-1).median().item()])
    return acts


# ---------------------------------------------------------------- RTT
def rtt(model, tok, name):
    """Retraining on T (Deeb & Roger): full-parameter, LR 1e-5, 3 epochs, batch 16, 10 warm-up steps; then held-out EM."""
    rows = read_jsonl(DATA / "rtt_T.jsonl")
    sft = SFT(model, lr=1e-5, weight_decay=0.01, warmup_steps=10)
    train_epochs(model, sft, encode_rows(tok, rows), 3, 16, seed=7, pad=tok.pad_token_id, tag=f"{name}-rtt")
    sft.close()
    del sft
    model.zero_grad(set_to_none=True)
    ev = [r for r in read_jsonl(DATA / "eval_heldout.jsonl") if r["group"] == "forget"]
    ans = greedy_answers(model, tok, [prompt_ids(tok, r["user"]) for r in ev], max_new_tokens=24)
    res = {}
    for t in ("T", "V"):
        sel = [exact_match(a, r["attr"], r["gold"]) for a, r in zip(ans, ev) if r["rtt"] == t]
        res[f"em_{t}"] = sum(sel) / len(sel)
        for a in ATTRS:
            s2 = [exact_match(x, r["attr"], r["gold"]) for x, r in zip(ans, ev) if r["rtt"] == t and r["attr"] == a]
            res[f"em_{t}_{a}"] = sum(s2) / len(s2)
    write_jsonl(RESULTS / "behaviour" / f"{name}_rtt_answers.jsonl",
                [{**r, "answer": a, "em": int(exact_match(a, r["attr"], r["gold"]))} for a, r in zip(ans, ev)])
    res["T_learned_ge_0.90"] = res["em_T"] >= 0.90
    res["recovery_V"] = res["em_V"] / 1.0          # full's V held-out EM = 100 %
    log(name, "RTT", res)
    return res


# ---------------------------------------------------------------- RMU
class _Stop(Exception):
    pass


def h_at(model, ids, am, layer):
    """Output of model.model.layers[layer]; the forward stops right after that block."""
    cap = {}

    def hook(_m, _i, o):
        cap["h"] = o[0] if isinstance(o, tuple) else o
        raise _Stop

    hd = model.model.layers[layer].register_forward_hook(hook)
    try:
        model.model(input_ids=ids, attention_mask=am, use_cache=False)
    except _Stop:
        pass
    finally:
        hd.remove()
    return cap["h"]


def push_rows(tok, rows):
    """(ids, start, end): RMU push range = first user-content token .. the assistant's <|im_end|> (inclusive)."""
    out = []
    for r in rows:
        ids, labels = train_ids(tok, r["user"], r["assistant"])
        s = user_span_start(tok, ids)
        e = max(i for i, l in enumerate(labels) if l != -100)
        assert tok.decode([ids[e]]) == "<|im_end|>"
        out.append((ids, s, e))
    return out


def _batch(items, pad):
    ids, am = _pad_right([x[0] for x in items], pad)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for i, (_, s, e) in enumerate(items):
        mask[i, s:e + 1] = True
    return ids.to(DEV), am.to(DEV), mask.to(DEV)


@torch.no_grad()
def median_norm(model, tok, items, layer=7, batch=32):
    special = set(tok.all_special_ids)
    norms = []
    for s in range(0, len(items), batch):
        it = items[s:s + batch]
        ids, am, mask = _batch(it, tok.pad_token_id)
        h = h_at(model, ids, am, layer).float()
        for i, (x, a, b) in enumerate(it):
            for p in range(max(a, 10), b + 1):
                if x[p] not in special:
                    norms.append(h[i, p].norm().item())
    return float(torch.tensor(norms).median()), len(norms)


@torch.no_grad()
def frozen_targets(model, tok, items, layer=7, batch=32):
    tg = []
    for s in range(0, len(items), batch):
        it = items[s:s + batch]
        ids, am, mask = _batch(it, tok.pad_token_id)
        h = h_at(model, ids, am, layer)
        for i, (_, a, b) in enumerate(it):
            tg.append(h[i, a:b + 1].detach().clone())
    return tg


def rmu_train(model, tok, f_items, r_items, r_targets, c, u, steps=200, bs=16, lr=5e-5, alpha=1200.0, seed=0, layer=7):
    import bitsandbytes as bnb
    pat = re.compile(r"model\.layers\.(5|6|7)\.mlp\.down_proj\.weight")
    names = [n for n, _ in model.named_parameters() if pat.fullmatch(n)]
    assert len(names) == 3, names
    for n, p in model.named_parameters():
        p.requires_grad_(n in names)
    params = [p for n, p in model.named_parameters() if n in names]
    opt = bnb.optim.PagedAdamW32bit(params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    rng = random.Random(seed)
    fo, ro = [], []
    target = (c * u).to(DEV, torch.bfloat16)
    hist = []
    t = time.time()
    for step in range(steps):
        if len(fo) < bs:
            fo += rng.sample(range(len(f_items)), len(f_items))
        if len(ro) < bs:
            ro += rng.sample(range(len(r_items)), len(r_items))
        fi, fo = fo[:bs], fo[bs:]
        ri, ro = ro[:bs], ro[bs:]
        ids, am, mask = _batch([f_items[i] for i in fi], tok.pad_token_id)
        hf = h_at(model, ids, am, layer)
        lf = ((hf[mask].float() - target.float()) ** 2).mean()
        ids, am, mask = _batch([r_items[i] for i in ri], tok.pad_token_id)
        hr = h_at(model, ids, am, layer)
        tgt = torch.cat([r_targets[i] for i in ri]).float()
        lr_ = alpha * ((hr[mask].float() - tgt) ** 2).mean()
        (lf + lr_).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        hist.append((lf.item(), lr_.item()))
        if (step + 1) % 25 == 0:
            log(f"RMU c={c:.1f} step {step + 1} forget-loss {lf.item():.3f} retain-loss {lr_.item():.4f} ({time.time() - t:.0f}s)")
    del opt
    for p in model.parameters():
        p.requires_grad_(True)
    return names, hist
