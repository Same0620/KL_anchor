"""Extra diagnostics after the A0 stop rule fired (NOT part of the pre-registered analysis; they do not change the stop).
  D1  orig's own answers to the in-context prompts (is the answer really available to the model?)
  D2  positive control: AV reading at the last token of the gold value where it is written in the biography (K = 2)
  D3  AV reading at answer_start with K = 10 (in-context and no-context decoy prompts), to bound the mention rate
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import AR, AV, DATA, ORIG, RESULTS, exact_match, load_tokenizer, messages, read_jsonl, write_jsonl
from gpulib import ARReader, AVReader, cosines, extract, free, greedy_answers, load_lm, log
from run_a0 import snap


def value_end_index(tok, user, value, ids):
    """Last token of the FIRST occurrence of `value` in the user message (inside the biography)."""
    text = tok.apply_chat_template(messages(user), tokenize=False, add_generation_prompt=True)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    assert list(enc["input_ids"]) == list(ids)
    u0 = text.index("<|im_start|>user\n") + len("<|im_start|>user\n")
    c = text.index(value, u0) + len(value)
    idx = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if a < c <= b]
    assert len(idx) == 1
    return idx[0]


def main():
    t0 = time.time()
    tok = load_tokenizer()
    ctx = read_jsonl(DATA / "read_ctx.jsonl")
    std = [m for m in read_jsonl(DATA / "read_std.jsonl") if m["group"] == "decoy"]
    vpos = [value_end_index(tok, m["user"], m["gold"], m["ids"]) for m in ctx]
    model = load_lm(ORIG)
    ans = greedy_answers(model, tok, [m["ids"] for m in ctx])
    d1 = [{"pid": m["pid"], "attr": m["attr"], "gold": m["gold"], "answer": a, "em": int(exact_match(a, m["attr"], m["gold"]))}
          for m, a in zip(ctx, ans)]
    write_jsonl(RESULTS / "diag" / "d1_orig_incontext_answers.jsonl", d1)
    log("D1 orig in-context EM", sum(r["em"] for r in d1) / len(d1))
    a_val = extract(model, [m["ids"] for m in ctx], [[p] for p in vpos], layers=(20,), batch=32)[20][:, 0]
    a_ctx = extract(model, [m["ids"] for m in ctx], [[m["pos"]["answer_start"]] for m in ctx], layers=(20,), batch=32)[20][:, 0]
    a_std = extract(model, [m["ids"] for m in std], [[m["pos"]["answer_start"]] for m in std], layers=(20,), batch=32)[20][:, 0]
    del model
    free()
    av, ar = AVReader(snap(AV)), ARReader(snap(AR))
    out = []
    for tag, meta, vecs, K, seed in [("value_token_ctx", ctx, a_val, 2, 21), ("answer_start_ctx_K10", ctx, a_ctx, 10, 22),
                                     ("answer_start_std_K10", std, a_std, 10, 23)]:
        ex = av.verbalize(vecs, K=K, seed=seed)
        flat = [e for per in ex for e in per]
        pred = ar.reconstruct([e[0] for e in flat])
        c = cosines(vecs.repeat_interleave(K, dim=0), pred)
        for i, m in enumerate(meta):
            for k in range(K):
                r = i * K + k
                out.append({"set": tag, "pid": m["pid"], "name": m["name"], "group": m["group"], "attr": m["attr"],
                            "gold": m["gold"], "golds": m["golds"], "position": "answer_start" if "answer" in tag else "value_token",
                            "k": k, "explanation": flat[r][0], "status": flat[r][1], "cos": c[r].item(),
                            "token": tok.decode([m["ids"][vpos[i]]]) if tag == "value_token_ctx" else None})
        log(tag, "mean cos", c.mean().item())
    write_jsonl(RESULTS / "diag" / "d2_d3_explanations.jsonl", out)
    log(f"diag done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
