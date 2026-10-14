"""Run the NLA (AV + AR) over one model's saved activations and write the explanations with their cos scores."""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ACTS, DATA, RESULTS, read_jsonl, write_jsonl
from gpulib import cosines, fve, log

POS_NAMES = ["answer_start", "assistant_tok", "name_end"]   # order of positions stored in the reading activations
SEEDS = {"general_L20": 11, "general_HS20": 12, "read_std": 13, "read_ctx": 14, "random": 15}
NORM_RULE = 1000.0


def choose_answer_position(orig_acts):
    """Spec section 6: if orig's activation norm at answer_start is above 1,000, read the token before it instead.
    Decided on orig's median norm over the 300 standard reading prompts; the same choice is used for every model."""
    med = orig_acts["read_std"][:, 0].norm(dim=-1).median().item()
    choice = "assistant_tok" if med > NORM_RULE else "answer_start"
    return choice, med


def read_general(av, ar, acts, model_name, keys=("general_L20",)):
    gen = json.loads((DATA / "general_set.json").read_text())
    rows, summ = [], {}
    for key in keys:
        vecs = acts[key]
        ex = av.verbalize(vecs, K=1, seed=SEEDS[key])
        texts = [e[0][0] for e in ex]
        pred = ar.reconstruct(texts)
        c = cosines(vecs, pred)
        summ[key] = {"fve": fve(vecs, pred), "mean_cos": c.mean().item(), "median_norm": vecs.norm(dim=-1).median().item(),
                     "n_no_tag": sum(e[0][1] != "ok" for e in ex)}
        for src in ["wildchat", "ultrafineweb"]:
            m = torch.tensor([g["source"] == src for g in gen])
            summ[key][f"fve_{src}"] = fve(vecs[m], pred[m])
        log(model_name, key, summ[key])
        for i, (e, g) in enumerate(zip(ex, gen)):
            rows.append({"model": model_name, "set": key, "i": i, "source": g["source"], "explanation": e[0][0],
                         "status": e[0][1], "cos": c[i].item(), "norm": vecs[i].norm().item()})
    return rows, summ


def read_facts(av, ar, acts, model_name, set_key, answer_pos, K=2):
    meta = read_jsonl(DATA / ("read_std.jsonl" if set_key == "read_std" else "read_ctx.jsonl"))
    A = acts[set_key]                                           # [N, 3, d]
    pos = [answer_pos, "name_end"]
    vecs = torch.stack([A[:, POS_NAMES.index(p)] for p in pos], dim=1).reshape(-1, A.shape[-1])   # row = (fact, position)
    ex = av.verbalize(vecs, K=K, seed=SEEDS[set_key])
    flat = [e for per in ex for e in per]
    pred = ar.reconstruct([e[0] for e in flat])
    gold = vecs.repeat_interleave(K, dim=0)
    c = cosines(gold, pred)
    rows = []
    for j, (m, posname) in enumerate([(m, p) for m in meta for p in pos]):
        for k in range(K):
            r = j * K + k
            rows.append({"model": model_name, "set": set_key, "pid": m["pid"], "name": m["name"], "group": m["group"],
                         "rtt": m["rtt"], "attr": m["attr"], "gold": m["gold"], "golds": m["golds"], "cond": m["cond"],
                         "position": "answer_start" if posname == answer_pos else posname, "token_read": posname,
                         "k": k, "explanation": flat[r][0], "status": flat[r][1], "cos": c[r].item(),
                         "norm": vecs[j].norm().item()})
    summ = {}
    for p in ["answer_start", "name_end"]:
        sel = [r for r in rows if r["position"] == p]
        idx = [i for i, r in enumerate(rows) if r["position"] == p]
        summ[p] = {"mean_cos": sum(r["cos"] for r in sel) / len(sel),
                   "fve": fve(gold[idx], pred[idx]),
                   "median_norm": torch.tensor([r["norm"] for r in sel]).median().item(),
                   "n_no_tag": sum(r["status"] != "ok" for r in sel)}
    log(model_name, set_key, summ)
    return rows, summ


def save(rows, name):
    write_jsonl(RESULTS / "expl" / f"{name}.jsonl", rows)


def load_acts(name):
    return torch.load(ACTS / f"{name}.pt")
