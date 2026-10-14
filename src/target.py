"""For one target model: extract the layer-20 activations NLA reads, and run the behaviour evaluation
(held-out question EM + MMLU-200)."""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ACTS, ATTRS, DATA, RESULTS, exact_match, prompt_ids, read_jsonl, write_jsonl
from gpulib import check_hook_vs_hidden_states, extract, greedy_answers, log, mmlu_accuracy
from reading import POS_NAMES


def extract_acts(model, name, ctx=False, hs20=False):
    gen = json.loads((DATA / "general_set.json").read_text())
    acts = {}
    layers = (19, 20) if hs20 else (20,)
    g = extract(model, [x["ids"] for x in gen], [[x["pos"]] for x in gen], layers=layers, batch=8)
    acts["general_L20"] = g[20][:, 0]
    if hs20:
        acts["general_HS20"] = g[19][:, 0]       # HF hidden_states[20] = output of layers[19]
    sets = ["read_std"] + (["read_ctx"] if ctx else [])
    for s in sets:
        meta = read_jsonl(DATA / f"{s}.jsonl")
        acts[s] = extract(model, [m["ids"] for m in meta], [[m["pos"][p] for p in POS_NAMES] for m in meta],
                          layers=(20,), batch=32)[20]
    ACTS.mkdir(parents=True, exist_ok=True)
    torch.save(acts, ACTS / f"{name}.pt")
    norms = {k: (v.norm(dim=-1).median(dim=0).values.tolist() if v.dim() == 3 else v.norm(dim=-1).median().item())
             for k, v in acts.items()}
    log(name, "activation median norms", norms)
    return acts


def behaviour(model, tok, name, groups=("forget", "retain", "decoy"), do_mmlu=True):
    ev = [r for r in read_jsonl(DATA / "eval_heldout.jsonl") if r["group"] in groups]
    ans = greedy_answers(model, tok, [prompt_ids(tok, r["user"]) for r in ev])
    rows = []
    for r, a in zip(ev, ans):
        rows.append({**r, "answer": a, "em": int(exact_match(a, r["attr"], r["gold"]))})
    write_jsonl(RESULTS / "behaviour" / f"{name}_answers.jsonl", rows)
    out = {"model": name}
    for gname in groups:
        sel = [r for r in rows if r["group"] == gname]
        out[f"em_{gname}"] = sum(r["em"] for r in sel) / len(sel)
        for a in ATTRS:
            s2 = [r for r in sel if r["attr"] == a]
            out[f"em_{gname}_{a}"] = sum(r["em"] for r in s2) / len(s2)
        for h in (0, 1):
            s2 = [r for r in sel if r["heldout"] == h]
            out[f"em_{gname}_heldout{h + 1}"] = sum(r["em"] for r in s2) / len(s2)
        if gname == "forget":
            for t in ("T", "V"):
                s2 = [r for r in sel if r["rtt"] == t]
                out[f"em_forget_{t}"] = sum(r["em"] for r in s2) / len(s2)
    if do_mmlu:
        items = json.loads((DATA / "mmlu200.json").read_text())
        acc, correct = mmlu_accuracy(model, tok, items)
        out["mmlu200"] = acc
        out["mmlu200_correct"] = correct
    (RESULTS / "behaviour").mkdir(parents=True, exist_ok=True)
    (RESULTS / "behaviour" / f"{name}.json").write_text(json.dumps(out, indent=1))
    log(name, "behaviour", {k: v for k, v in out.items() if k != "mmlu200_correct"})
    return out


def hook_check(model):
    gen = json.loads((DATA / "general_set.json").read_text())[:4]
    res = check_hook_vs_hidden_states(model, [x["ids"] for x in gen])
    log("hook check", res)
    return res
