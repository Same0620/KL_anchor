"""Exp 22 Job G2 (GPU, base Qwen3-14B): base-model answers (PREREG B4). Only the base model generates text here.

Anchors (K-every rebuilt, A_adv1, A_advN): measurement template (thinking off), temperature 0.7, top_p 1, top_k 0, at most 200 new
tokens; 20B's generate() copied with Exp 22's own seed label. -> data/anchor_{every,adv1,advN}_{train,reserved}.jsonl in 20B's row format
(row_hash = sha256(question NUL answer)[:16]; reserved rows asserted never in a train file).
Evaluation classes 3, 4, 5a, 5b, 6, 6-orig, 7, 9: greedy, at most 200 new tokens -> data/base_answers.jsonl, which also carries 20B's
greedy answers for gate60-risky_financial, intopic24-risky_financial and everyday48 unchanged.
Item 7d: K-every's 240 visited answers must average within 20% of everyday48's base answers (asserted after the files are written).
"""
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "exp20b" / "src"))
import e20b_common as Z  # noqa: E402

X19 = Z.X19
sys.path.insert(1, str(X19.E19 / "src"))
import e19lib as L19  # noqa: E402

E22, DEV, MAX_NEW, T0 = ROOT / "exp22", "cuda", 200, time.time()
EVAL = {"gen24-bad_petcare": X19.PROMPTS / "gen24-bad_petcare.jsonl", "gen24-bad_beauty": X19.PROMPTS / "gen24-bad_beauty.jsonl",
        "gen24-bad_pesticides": X19.PROMPTS / "gen24-bad_pesticides.jsonl", "gen24-unsafe_chemicals": X19.PROMPTS / "gen24-unsafe_chemicals.jsonl",
        "gate60-extreme_sports": X19.PROMPTS / "gate60-extreme_sports.jsonl", "gate60-bad_legal": X19.PROMPTS / "gate60-bad_legal.jsonl",
        "mbpp24_clean": E22 / "prompts" / "mbpp24_clean.jsonl", "mbpp24": ROOT / "exp20b" / "prompts" / "mbpp24.jsonl",
        "gsm8k24": ROOT / "exp20b" / "prompts" / "gsm8k24.jsonl", "passage24": E22 / "prompts" / "passage24.jsonl"}


def log(m):
    print(f"[jobG2 {time.time() - T0:6.0f}s] {m}", flush=True)


@torch.no_grad()
def generate(model, tok, questions, tag, temperature, batch=128):
    """20B's generate (e20b_jobG.py), unchanged except the seed label: -> [(answer ids without the end token, finished)]."""
    from transformers import GenerationConfig
    eos_ids = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    pad_id = tok.convert_tokens_to_ids("<|endoftext|>")
    if temperature is None:
        cfg = GenerationConfig(do_sample=False, max_new_tokens=MAX_NEW, pad_token_id=pad_id, eos_token_id=eos_ids)
    else:
        cfg = GenerationConfig(do_sample=True, temperature=temperature, top_p=1.0, top_k=0, max_new_tokens=MAX_NEW, pad_token_id=pad_id,
                               eos_token_id=eos_ids)
    plist = [Z.prompt_ids(tok, q) for q in questions]                       # postfix asserted
    order = sorted(range(len(plist)), key=lambda i: len(plist[i]))
    out = [None] * len(plist)
    for b in range(0, len(order), batch):
        idx = order[b:b + batch]
        n = max(len(plist[i]) for i in idx)
        ids = torch.tensor([[pad_id] * (n - len(plist[i])) + plist[i] for i in idx], device=DEV)
        att = torch.tensor([[0] * (n - len(plist[i])) + [1] * len(plist[i]) for i in idx], device=DEV)
        seed = X19.stable_int("e22-jobG2", tag, b)
        torch.manual_seed(seed % (2 ** 63))
        torch.cuda.manual_seed_all(seed % (2 ** 63))
        gen = model.generate(input_ids=ids, attention_mask=att, generation_config=cfg)
        for i, g in zip(idx, gen[:, n:].tolist()):
            cut = next((j for j, t in enumerate(g) if t in eos_ids), None)
            out[i] = (g if cut is None else g[:cut], cut is not None)
    log(f"{tag}: {len(order)} answered")
    return out


def main():
    tok = L19.load_tok()
    base = L19.load_base()
    res = {"max_new_tokens": MAX_NEW, "anchor_temperature": 0.7, "measurement": "greedy"}
    for name in ("every", "adv1", "advN"):
        qs = Z.read_jsonl(E22 / "data" / f"anchor_{name}_questions.jsonl")
        gen = generate(base, tok, [q["question"] for q in qs], f"anchor-{name}", 0.7)
        rows = []
        for q, (ids, fin) in zip(qs, gen):
            ans = tok.decode(ids, skip_special_tokens=True).strip()
            extra = {k: q[k] for k in ("category", "domain", "angle", "dolly_index") if k in q}
            r = {"id": q["id"], "messages": [{"role": "user", "content": q["question"]}, {"role": "assistant", "content": ans}],
                 "finished": fin, "n_tokens": len(ids), "reserved": q["reserved"], **extra}
            r["row_hash"] = Z.text_hash(q["question"] + "\x00" + ans)
            rows.append(r)
        tr, rs = [r for r in rows if not r["reserved"]], [r for r in rows if r["reserved"]]
        assert not ({r["row_hash"] for r in tr} & {r["row_hash"] for r in rs})
        assert not ({Z.norm_q(r["messages"][0]["content"]) for r in tr} & {Z.norm_q(r["messages"][0]["content"]) for r in rs})
        Z.write_jsonl(E22 / "data" / f"anchor_{name}_train.jsonl", tr)
        Z.write_jsonl(E22 / "data" / f"anchor_{name}_reserved.jsonl", rs)
        res[name] = {"train": len(tr), "reserved": len(rs), "mean_tokens_train": sum(r["n_tokens"] for r in tr) / len(tr),
                     "mean_tokens_reserved": sum(r["n_tokens"] for r in rs) / len(rs),
                     "share_cut_at_200": sum(not r["finished"] for r in rows) / len(rows)}
        log(f"anchor {name}: {res[name]}")
    meas = [m for m in Z.read_jsonl(ROOT / "exp20b" / "data" / "base_answers.jsonl")]       # 20B's three greedy sets, unchanged
    for s, f in EVAL.items():
        qs = Z.read_jsonl(f)
        gen = generate(base, tok, [q["question"] for q in qs], f"measure-{s}", None)
        for q, (ids, fin) in zip(qs, gen):
            meas.append({"set": s, "id": q["id"], "question": q["question"], "answer_ids": ids,
                         "answer": tok.decode(ids, skip_special_tokens=True).strip(), "finished": fin, "n_tokens": len(ids)})
        ss = [m for m in meas if m["set"] == s]
        res[f"measure_{s}"] = {"n": len(ss), "mean_tokens": sum(m["n_tokens"] for m in ss) / len(ss),
                               "share_cut": sum(not m["finished"] for m in ss) / len(ss)}
    Z.write_jsonl(E22 / "data" / "base_answers.jsonl", meas)
    ev = [m["n_tokens"] for m in meas if m["set"] == "everyday48"]
    res["item7d"] = {"every_visited_mean_tokens": res["every"]["mean_tokens_train"], "everyday48_mean_tokens": sum(ev) / len(ev),
                     "ratio": res["every"]["mean_tokens_train"] / (sum(ev) / len(ev))}
    (E22 / "results" / "jobG2.json").write_text(json.dumps(res, indent=1))
    log(f"item 7d: K-every answers / everyday48 answers = {res['item7d']['ratio']:.3f}")
    assert 0.8 <= res["item7d"]["ratio"] <= 1.2, ("item 7d response-length match failed", res["item7d"])
    log("Job G2 done")


if __name__ == "__main__":
    main()
