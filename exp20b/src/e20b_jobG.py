"""Job G (GPU; handoff 4): the base model's answers. Only the base model generates text in Exp 20B; no adapter is ever sampled.

  e20b_jobG.py [--smoke]

  A_every  1,000 Dolly questions (data/anchor_every_questions.jsonl), A_far 1,000 MBPP + GSM8K questions (data/anchor_far_questions.jsonl):
           base Qwen3-14B, measurement template (thinking off), temperature 0.7 (top_p 1, top_k 0), at most 200 new tokens; seed per batch
  measurement answers (4.5): greedy, thinking off, at most 200 new tokens, for everyday48, the primary in-topic set (gate60) and intopic24
Writes data/anchor_{every,far}_{train,reserved}.jsonl (rows {"id", "messages", "finished", "n_tokens", "category"/"source", "row_hash"}),
data/base_answers.jsonl, results/jobG.json. Reserved rows never enter a training file (asserted by row hash).
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e20b_common as Z

X19 = Z.X19
DEV = "cuda"
MAX_NEW = 200


def log(msg, t0=time.time()):
    print(f"[jobG {time.time() - t0:6.0f}s] {msg}", flush=True)


@torch.no_grad()
def generate(model, tok, questions, tag, temperature, batch=128):
    """-> [(answer ids without the end token, finished)] in input order. Left padding; prompts sorted by length within the call."""
    from transformers import GenerationConfig
    eos_ids = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    pad_id = tok.convert_tokens_to_ids("<|endoftext|>")
    if temperature is None:
        cfg = GenerationConfig(do_sample=False, max_new_tokens=MAX_NEW, pad_token_id=pad_id, eos_token_id=eos_ids)
    else:
        cfg = GenerationConfig(do_sample=True, temperature=temperature, top_p=1.0, top_k=0, max_new_tokens=MAX_NEW, pad_token_id=pad_id,
                               eos_token_id=eos_ids)
    plist = [Z.prompt_ids(tok, q) for q in questions]
    order = sorted(range(len(plist)), key=lambda i: len(plist[i]))
    out = [None] * len(plist)
    for b in range(0, len(order), batch):
        idx = order[b:b + batch]
        n = max(len(plist[i]) for i in idx)
        ids = torch.tensor([[pad_id] * (n - len(plist[i])) + plist[i] for i in idx], device=DEV)
        att = torch.tensor([[0] * (n - len(plist[i])) + [1] * len(plist[i]) for i in idx], device=DEV)
        seed = X19.stable_int("e20b-jobG", tag, b)
        torch.manual_seed(seed % (2 ** 63))
        torch.cuda.manual_seed_all(seed % (2 ** 63))
        gen = model.generate(input_ids=ids, attention_mask=att, generation_config=cfg)
        for i, g in zip(idx, gen[:, n:].tolist()):
            cut = next((j for j, t in enumerate(g) if t in eos_ids), None)
            out[i] = (g if cut is None else g[:cut], cut is not None)
        log(f"{tag}: {min(b + batch, len(order))}/{len(order)}")
    return out


def main():
    smoke = "--smoke" in sys.argv
    sys.path.insert(1, str(X19.E19 / "src"))
    import e19lib as L19
    tok = L19.load_tok()
    base = L19.load_base()
    res = {"max_new_tokens": MAX_NEW, "anchor_temperature": 0.7, "measurement": "greedy"}
    for name in ("every", "far"):
        qs = Z.read_jsonl(Z.DATA / f"anchor_{name}_questions.jsonl")
        if smoke:
            qs = qs[:24]
        t0 = time.time()
        gen = generate(base, tok, [q["question"] for q in qs], f"anchor-{name}", 0.7)
        rows = []
        for q, (ids, fin) in zip(qs, gen):
            ans = tok.decode(ids, skip_special_tokens=True).strip()
            r = {"id": q["id"], "messages": [{"role": "user", "content": q["question"]}, {"role": "assistant", "content": ans}], "finished": fin,
                 "n_tokens": len(ids), "reserved": q["reserved"], **({"category": q["category"]} if "category" in q else {"source": q["source"]})}
            r["row_hash"] = Z.text_hash(q["question"] + "\x00" + ans)
            rows.append(r)
        tr = [r for r in rows if not r["reserved"]]
        rs = [r for r in rows if r["reserved"]]
        assert not ({r["row_hash"] for r in tr} & {r["row_hash"] for r in rs}) and not ({Z.norm_q(r["messages"][0]["content"]) for r in tr}
                                                                                      & {Z.norm_q(r["messages"][0]["content"]) for r in rs})
        if not smoke:
            assert len(tr) == Z.N_TRAIN_ANCHOR and len(rs) == Z.N_RESERVED
        sub = Z.SMOKE / "data" if smoke else Z.DATA
        Z.write_jsonl(sub / f"anchor_{name}_train.jsonl", tr)
        Z.write_jsonl(sub / f"anchor_{name}_reserved.jsonl", rs)
        nt = [r["n_tokens"] for r in rows]
        res[name] = {"n": len(rows), "train": len(tr), "reserved": len(rs), "mean_tokens": sum(nt) / len(nt), "share_cut_at_200": sum(not r["finished"] for r in rows) / len(rows),
                     "empty": sum(1 for r in rows if not r["messages"][1]["content"]), "seconds": round(time.time() - t0)}
        if name == "far":
            for src in ("mbpp", "gsm8k"):
                ss = [r for r in rows if r["source"] == src]
                res[name][src] = {"mean_tokens": sum(r["n_tokens"] for r in ss) / len(ss), "share_cut": sum(not r["finished"] for r in ss) / len(ss)}
        log(f"A_{name}: {res[name]}")
    sets = [Z.EVERYDAY, Z.IN_PRIMARY, Z.IN_SECONDARY[0]]
    meas = []
    for s in sets:
        qs = X19.prompt_set(s)
        if smoke:
            qs = qs[:8]
        gen = generate(base, tok, [q["question"] for q in qs], f"measure-{s}", None)
        for q, (ids, fin) in zip(qs, gen):
            meas.append({"set": s, "id": q["id"], "question": q["question"], "answer_ids": ids, "answer": tok.decode(ids, skip_special_tokens=True).strip(),
                         "finished": fin, "n_tokens": len(ids)})
        ss = [m for m in meas if m["set"] == s]
        res[f"measure_{s}"] = {"n": len(ss), "mean_tokens": sum(m["n_tokens"] for m in ss) / len(ss), "share_cut": sum(not m["finished"] for m in ss) / len(ss)}
    Z.write_jsonl((Z.SMOKE / "data" if smoke else Z.DATA) / "base_answers.jsonl", meas)
    Z.write_json((Z.SMOKE / "results" if smoke else Z.RESULTS) / "jobG.json", res)
    log(f"measurement answers: {[(s, res[f'measure_{s}']) for s in sets]}")
    log("Job G done")


if __name__ == "__main__":
    main()
